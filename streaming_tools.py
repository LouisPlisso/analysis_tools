#!/usr/bin/env python
"""New scripts to understand the traces:
repartition of nb cns per AS
top AS in the trace
bar plot
plot all cdfs
"""

from __future__ import division, print_function

import INDEX_VALUES
import aggregate
import cdfplot
from load_dipcp_file import filter_dipcp_array
from flow2session import process_stream_session, GAP #, process_cnx_sessions

from collections import defaultdict
from itertools import imap, izip, chain
import functional
from operator import itemgetter, setitem, concat, mul
from os import sep, listdir
import sys
import numpy as np
import re
# in case of non-interactive usage
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import matplotlib.patches as mpatches
import pylab

THRESHOLD = 10**6 #500*10**3
MIN_FLOWS_NB = 1
MIN_PERCENT = 2*1e-2

DEBUG = False

def connections_per_as(datas, field='dscp', as_field='asBGP',
                       agg_field='l3Bytes', key_ext='_GVB',
                       already_filtered=False, dir_check=True):
    """Return a dict of dict of number of connections and volume per AS for
    each trace. It's a wrapper to aggregate functions.
    Use as:
datas = tools.load_hdf5_data.load_h5_file('flows/hdf5/traces_lzf.h5')
cns_streaming_per_as = tools.streaming_tools.connections_per_as(datas)
    NEW: with as_field not-restricted to AS separation
    """
    traces = set(key.strip('_DIPCP').strip('_GVB') for key in datas)
    cns_per_as = {}
    for trace in traces:
        data = datas[trace + key_ext]
        if dir_check:
            data = data.compress(data['direction'] == INDEX_VALUES.DOWN)
        if not already_filtered:
            (dscp_http_stream, _, _) = un_mismatch_dscp(data)
            data = data.compress(data[field] == dscp_http_stream)
        cns_per_as[trace] = aggregate.aggregate_sum_nb(data,
                as_field, agg_field)
    return cns_per_as

def data_stream_aggregator(datas_stream):
    "Return a dict of (nb videos, vol) per AS for each trace"
    datas_stream_agg = {}
    for t in datas_stream:
        flows = datas_stream[t].compress(datas_stream[t]['Session-Bytes'] < 1e9)
        datas_stream_agg[t.split('_GVB_STR_AS_GVB')[0]] = \
                aggregate.aggregate_sum_nb(flows,
                                           'asBGP', 'Session-Bytes')
    return datas_stream_agg

def split_as_name(as_name):
    """Return a tuple of AS nb and AS name out of a string
    >>> split_as_name('AS8068 Microsoft European Data Center')
    (8068, 'Microsoft European Data Center')
    >>> split_as_name('AS15169 Google')
    (15169, 'AS15169')
    >>> split_as_name('AS8075')
    (8075, 'AS8075')
    """
    assert as_name.startswith('AS'), "incorrect AS number " + as_name
    value = as_name.split('AS', 1)[1].split()
    if len(value) == 1 or value[1].startswith('-'):
        return (int(value[0]), 'AS' + value[0])
    else:
        return (int(value[0]), '\n'.join((value[1:3])))

try:
    AS_DB
except NameError:
    #WARNING: hard coded and global variables for efficiency
    AS_FILE = open('/home/louis/streaming/flows/AS/GeoIPASNum2_old.csv')
    AS_DB = dict([split_as_name(line.split(', ')[2].strip('" \n'))
                  for line in AS_FILE.readlines()])
    #fucking maxmind doesn't know google
    AS_DB[15169] = "Google"
    AS_DB[22822] = "Limelight"
    AS_DB[8075] = "Microsoft (MSN)"
    AS_DB[2914] = "NTT-Comm US"
    AS_DB[29748] = "Carpathia Hosting"
    AS_DB[24963] = "Yacast FR"

def change_as_nb_in_agg(top_as):
    "Return a list with by AS name instead of AS nb"
    return dict([(t, [(AS_DB.get(k, k), v) for (k, v) in top_as[t]])
                 for t in top_as])
#    new_dict = {}
#    for t in top_as:
#        new_dict[t] = [(AS_DB.get(k, k), v)
#                       for (k, v) in top_as[t]]
#    return new_dict

def adjust_title(as_name, split_size=3):
    """Formats the name to be printed on pie chart
    >>> adjust_title('Microsoft European Data Center')
    'Microsoft European Data\nCenter'
    >>> adjust_title('Google')
    'Google'
    """
    split_name = as_name.split()
    return '\n'.join(' '.join(split_name[i:i+split_size])
                     for i in xrange(0, len(split_name), split_size))

def get_top_n(datas_agg, nb_top=7, rank='sum', exclude_list=None):
    """Return a list of nb_top tuples with each AS and the value and other
    values aggregated for each trace
    INPUT: a dict of dict indexed by AS and values as tuple (nb, sum)
    OUTPUT: a dict of list of (AS, value) including an other aggregator
        (max len=nb_top-1)
    """
    top_n = {}
    if rank == 'sum':
        field = 1
    elif  rank == 'nb':
        field = 0
    else:
        assert False, "Rank function not correct"
    for t in datas_agg:
        data = [(k, itemgetter(field)(v))
                for (k, v) in datas_agg[t].iteritems()]
        values = sorted(data, key=itemgetter(1), reverse=True)
        for i, v in enumerate(map(itemgetter(0), values[:nb_top])):
            # usecase: removing unknown user id
            if v in exclude_list:
                excluded_value = values.pop(i)
                print('excluded: ' + v)
                #values.append(excluded_value)
        top_n[t] = values[:nb_top]
        #top_n[t].append(('Other', sum(map(itemgetter(1), values[nb_top:]))))
    return top_n

def get_top_n_nok(datas, nb_top=7, rank='sum', include_goo=True):
    """Return a dict of top n of AS ranked by nb or sum:
    INPUT data: a dict indexed by AS and values as tuple (nb, sum)
    OUTPUT data: a dict indexed by AS and values as nb (or sum)
    Use as:
    datas = tools.load_hdf5_data.load_h5_file('flows/hdf5/traces_lzf.h5')
    cns_streaming_per_as = tools.streaming_tools.connections_per_as(datas)
    top_as = tools.streaming_tools.get_top_n(cns_streaming_per_as)
    """
    top_as = {}
    if rank == 'sum':
        field = 1
    elif  rank == 'nb':
        field = 0
    else:
        print("Rank function not correct")
        return -1
    # top_as_global aggregates all volumes on all traces
    top_as_global = defaultdict(int)
    [setitem(top_as_global, key, top_as_global[key] + value[field])
            for trace in datas.values() for key, value in trace.items()]
    # list_as_global lists the top n AS in all traces
    list_as_global = imap(itemgetter(0), sorted(top_as_global.items(),
        key=itemgetter(1), reverse=True)[:nb_top])
    if include_goo:
        list_as_global = chain(list_as_global, INDEX_VALUES.ALL_AS_GOOGLE)
    list_as_global = set(list_as_global)
    for trace in datas:
        data = datas[trace]
#        ranked_data = sorted(data.items(), key=itemgetter(field), reverse=True)
        top_as[trace] = dict((key, data.get(key, (0, 0)))
                for key in list_as_global)
        top_as[trace]['other'] = tuple(sum(values) for values in
                izip(*[data[key] for key in data
                    if key not in list_as_global]))
    return top_as

def plot_vol_nb_video_per_as(top_as, out_dir='rapport/http_stats',
                             out_file='pie_volume',
                     title='Most important AS for HTTP Streaming by Volume'):
    """Draw a pie chart for the volume per as and nb of videos
    USE AS:
datas_stream = tools.load_hdf5_data.load_h5_file('flows/hdf5/hdf5_streaming.h5')
datas_stream_agg = tools.streaming_tools.data_stream_aggregator(datas_stream)
top_as = tools.streaming_tools.get_top_n(datas_stream_agg)
top_as_name = tools.streaming_tools.change_as_nb_in_agg(top_as)
tools.streaming_tools.plot_vol_nb_video_per_as(top_as_name)
top_as_nb = tools.streaming_tools.get_top_n(datas_stream_agg, rank='nb')
top_as_nb_name = tools.streaming_tools.change_as_nb_in_agg(top_as_nb)
tools.streaming_tools.plot_vol_nb_video_per_as(top_as_nb_name,
            title='Most important AS for HTTP Streaming by Nb of Cnxs',
            out_file='pie_nb')
    IDEM FOR FLOWS >1MB:
datas_stream_1mb = dict([(t, v.compress(v['Session-Bytes']>1e6)) for
    (t, v) in datas_stream.iteritems()])
datas_stream_agg_1mb = tools.streaming_tools.data_stream_aggregator(
                                                datas_stream_1mb)
top_as_1mb = tools.streaming_tools.get_top_n(datas_stream_agg_1mb)
top_as_name_1mb = tools.streaming_tools.change_as_nb_in_agg(top_as_1mb)
tools.streaming_tools.plot_vol_nb_video_per_as(top_as_name_1mb,
    title='Most important AS for HTTP Streaming by Volume (flows >1MB)',
    out_file='pie_volume_1mb')
top_as_nb_1mb = tools.streaming_tools.get_top_n(datas_stream_agg_1mb,
                                                rank='nb')
top_as_nb_name_1mb = tools.streaming_tools.change_as_nb_in_agg(top_as_nb_1mb)
tools.streaming_tools.plot_vol_nb_video_per_as(top_as_nb_name_1mb,
    title='Most important AS for HTTP Streaming by Nb of Cnxs (flows >1MB)',
            out_file='pie_nb')
    """
    colors = list('bgrcmy') + ['0.5'] + ['w']
    for t in top_as:
        # make a square figure and axes
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111)
        # pie returns (patches, texts, autotexts)
        (_, texts, _) = ax.pie(map(itemgetter(1), top_as[t]),
                  labels=map(functional.compose(adjust_title, itemgetter(0)),
                             top_as[t]),
                  explode=[0.15]*len(top_as[t]), autopct='%1.0f%%',
                              colors=colors)
        map(lambda s: s.set_fontsize(12), texts)
        ax.set_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_title('\n'.join((title, 'for ' + t.replace('_', ' '))))
        fig.savefig(sep.join((out_dir, '_'.join((out_file, t)))) + '.pdf',
                               format='pdf')

#
#def bar_plot_key_tuple(datas, value_index, percent=False,
#        output_path='rapport/test'):
#    """Plot a bar graph of data.
#    INPUT data: a dict with a key and a tuple of values for each element.
#    Use as:
#    datas = tools.load_hdf5_data.load_h5_file('flows/hdf5/traces_lzf.h5')
#    cns_streaming_per_as = tools.streaming_tools.connections_per_as(datas)
#    top_as = tools.streaming_tools.get_top_n(cns_streaming_per_as)
#    tools.streaming_tools.bar_plot_key_tuple(top_as, 0)
#    """
#    to_plot = {}
#    for trace in sorted(datas.keys()):
#        data = datas[trace]
#        total = 0.0 + sum(([value[value_index] for value
#            in data.values()]))
#        plot_data = dict(total=total, values=[])
#        if percent:
#            divisor = total if total != 0 else 1
#        else:
#            divisor = 1
#        bottom = 0 #pylab.zeros(nb_graphs)
#        for (key, values) in sorted(data.items(), key=itemgetter(0)):
#            value = values[value_index] / divisor
#            plot_data['values'].append((key, value, bottom))
#            bottom += value
#        to_plot[trace] = plot_data
#    ylabel = ' '.join(('Volume' if value_index else 'Number',
#            "of flows for top AS in", 'percent' if percent else 'Bytes'))
#    bar_plot(to_plot, ylabel=ylabel, output_path=output_path)
##    pylab.show()
#
#def bar_plot(to_plot, ylabel='', output_path=None):
#    "Wrapper for the bar plot: draw ticks, legend and save the plot"
#    # plotting
#    pylab.clf()
#    pylab.subplot(111)
#    pylab.subplots_adjust(top=0.8)
#    pylab.subplots_adjust(right=0.75)
#    legend = plot_bars(to_plot)
#    pylab.grid(True)
#    pylab.ylabel(ylabel)
#    axes = pylab.gca()
#    pylab.text(0.2, -0.1, "HTTP Streaming Down", size=12,
#            transform = axes.transAxes)
#    pylab.xticks(1 + pylab.arange(2*len(to_plot), step=2),
#            map(format_title, sorted(to_plot.keys())))
#    legend_labels = set(key for plot_data in to_plot.values() for (key, _, _)
#        in plot_data['values'])
#    pylab.legend(legend[-1], sorted(legend_labels), loc=(1.1, 0.1))
#    if output_path:
#        pylab.savefig('%s/%s.pdf' % (output_path,
#            ylabel.replace(' ', '_').lower()), format='pdf')
#
#def plot_bars(to_plot):
#    "Return the legend and do the inner part of the plot"
#    legend = {}
#    hatches = ['/', '\\', 'x', '.', '*', '+', '|', '-', 'o', 'O']
#    hatches_len = len(hatches)
#    # first find out scale: add last bottom value and its value
#    max_scale = max((last + bottom) for plot_data in to_plot.values()
#            for (_, last, bottom) in (plot_data['values'][-1],))
#    # width of bars: can also be len(x) sequence
#    width = 0.35
#    x_values = 1 - width / 2 #pylab.arange(2*nb_graphs, step=2)
#    for data in sorted(to_plot.keys()):
#        plot_data = to_plot[data]
#        legend[data] = []
#        for i, (key, value, bottom) in enumerate(plot_data['values']):
#            legend[data].append(pylab.bar(x_values, value, width,
#                    hatch=hatches[i % hatches_len], color='w', bottom=bottom,
#                    lw=2, antialiased=True, label=key))
#            last_bottom, last_value = bottom, value
#        # using last loop variables: bottom and value
#        pylab.text(x_values + width / 2., max_scale / 40 + last_bottom +
#                last_value, "total: %.4g" % float(plot_data['total']),
#                rotation=45)
#        x_values += 2
#    return legend
#

def format_title(trace):
    r"""Return a formatted string for the trace name
    >>> format_title('2010_02_07_FTTH.npy')
    '2010/02\nFTTH'
    >>>
    """
    if trace.startswith('20'):
        trace_name = trace.split('.')[0]
        trace_date = '/'.join(trace_name.split('_')[:2])
        trace_type = ' '.join(trace_name.split('_')[3:])
        if not trace_type.endswith('R'):
            trace_type = ' '.join((trace_type, 'M'))
        return '\n'.join((trace_date, trace_type))
    else:
        return '\n' + trace

def un_mismatch_dscp(flow, flow_type='GVB'):
    """Return a tuple of the correct values of DSCP according to highest
    numbers of flows classified Web.
    OUTPUT: (dscp_http_stream, dscp_other_stream, dscp_web)
    >>> un_mismatch_dscp(pylab.array( \
            [pylab.ushort(INDEX_VALUES.DSCP_WEB)]*5, \
            dtype=pylab.dtype([('dscp', pylab.ushort)])))
    (11, 10, 1)
    >>> un_mismatch_dscp(pylab.array( \
            [pylab.ushort(INDEX_VALUES.DSCP_MARCIN_WEB)]*5, \
            dtype=pylab.dtype([('dscp', pylab.ushort)])))
    (8, 9, 11)
    >>> un_mismatch_dscp(pylab.array( \
            [pylab.ushort(INDEX_VALUES.DSCP_WEB)]*3+ \
            [pylab.ushort(INDEX_VALUES.DSCP_MARCIN_WEB)]*4, \
            dtype=pylab.dtype([('dscp', pylab.ushort)])))
    (8, 9, 11)
    >>>
    """
    if flow_type != 'GVB':
        print("not implemented")
        return None
    if (len(flow.compress(flow['dscp'] == INDEX_VALUES.DSCP_WEB)) >
        len(flow.compress(flow['dscp'] ==
            INDEX_VALUES.DSCP_MARCIN_WEB))):
        # normal case
        return (INDEX_VALUES.DSCP_HTTP_STREAM,
                INDEX_VALUES.DSCP_OTHER_STREAM,
                INDEX_VALUES.DSCP_WEB)
    else:
        # mismatched case
        return (INDEX_VALUES.DSCP_MARCIN_HTTP_STREAM,
                INDEX_VALUES.DSCP_MARCIN_OTHER_STREAM,
                INDEX_VALUES.DSCP_MARCIN_WEB)

def return_cnx_id_str_GVB(GVB_line):
    "Return the connection ID of GVB flow."
    return (GVB_line['srcAddr'], GVB_line['srcPort'],
            GVB_line['dstAddr'], GVB_line['protocol'],
            GVB_line['dscp'])

def match_flow_stream(flow_id, GVB_flows):
    "Returns the first match of flow_id in a GVB flow"
    for flow in GVB_flows:
        if return_cnx_id_str_GVB(flow) == flow_id:
            return flow
    print("No match for flow: ", flow_id)

def filter_stream_array(stream_flows, GVB_flows):
    """Return an array with streaming flows corresponding to the value of
    fields in GVB flow: it takes only GVB flows on port 80, proto TCP, and
    dscp streaming
    """
    dscp_http_stream, _, _ = un_mismatch_dscp(GVB_flows)
    new_stream_flows = []
    print("%d flows to treat" % len(stream_flows))
    treated_flow = 0
    # constuct dict of GVB flow
    # what happen in case of multiple matches?
    GVB_dict = dict(izip(map(return_cnx_id_str_GVB, GVB_flows),
                         GVB_flows['asBGP']))
#    print("GVB_flows dict constucted")
    for flow in stream_flows.compress(stream_flows['valid'] == 'OK'):
        # always TCP on port 80 and dscp streaming
        flow_id = (flow['srcAddr'], 80, flow['dstAddr'], 6, dscp_http_stream)
        if flow_id in GVB_dict:
            new_stream_flows.append(tuple(list(flow) + [GVB_dict[flow_id]]))
        treated_flow += 1
#        if treated_flow % 100 == 0:
#            print("%d flows treated" % treated_flow)
    print("%d flows treated" % treated_flow)
    return pylab.array(new_stream_flows, copy=True,
                       dtype=INDEX_VALUES.dtype_GVB_streaming_AS)

def filter_stream_dict(datas):
    "Return a new dict of datas filtered"
    filtered_dict = dict()
    traces = set(key.strip('_DIPCP').strip('_GVB').split('_GVB_STR')[0]
                 for key in datas)
    for trace in traces:
        print('filtering trace ' + trace)
        filtered_dict[trace + '_GVB_STR'] = filter_stream_array(
            datas[trace + '_GVB_STR'], datas[trace + '_GVB'])
    return filtered_dict


def append_data_dict_with_stream(datas, stream_flows_dir):
    "Include in the datas the streaming arrays loaded from stream_flows_dir"
    traces = set(key.strip('_DIPCP').strip('_GVB').split('_GVB_STR')[0]
                 for key in datas)
#    filtered_data = {}
    for trace in traces:
        stream_file = sep.join((stream_flows_dir.rstrip(sep),
                               trace + '_GVB_STR.npy'))
        try:
            datas[trace + '_GVB_STR'] = \
                    pylab.np.load(stream_file)
        except IOError:
            print("Error while loading file: " + stream_file)

# from web
def bar_chart(data, title='HTTP Streaming Flows Normalized Volume',
              ylabel='Volume in Bytes per Second', out_dir='.'):
    ''' Input: a dict that maps bar labels to values and a title for the chart.
    Output: relative path to a PDF bar chart.  ylabel controls the y axis
    label. It labels the bars with the values, by default a percent.
    '''
#    data = clean_dict(data)
    labels = sorted(data.keys())
    traces = [format_title(k).split(' AS ')[0] for k in labels]
    ref = traces[0]
    x_titles = [ref]
    x_loc = [0]
    for t in traces[1:]:
        if t == ref:
            x_titles.append('')
            x_loc.append(x_loc[-1] + 2)
        else:
            ref = t
            x_titles.append(t)
            x_loc.append(x_loc[-1] + 6)
    x_loc = np.array(x_loc)
    values = [data[k] for k in labels]
#    ind = pylab.arange(0, 2 * len(values), 2)  # the x locations for the groups
    width = 2       # the width of the bars
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    p1 = plt.bar(x_loc, values, width, color='r'),
    ax.yaxis.grid(b=True)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ymax=ax.get_ylim()[1] + 0.1)
    ax.set_title(title)
    ax.set_xticks(x_loc+(width/1.0))
    ax.set_xticklabels(x_titles, size='small', rotation=30)
#    pylab.xlim(-width,2*len(x_loc))
    #pylab.yticks(pylab.arange(0,41,10))
    # Labels!
#    for x,y in zip(xrange(len(values)), values):
    for x, y, t in zip(x_loc, values,
                       [format_title(k).split(' AS ')[1].capitalize()
                                     for k in labels]):
        ax.text(x+width/2., y, t, rotation=90, ha='center', size='small')
    short_fname = sep.join((out_dir, title.replace(' ', '_') + '.pdf'))
    fig.savefig(short_fname)
    # http://matplotlib.sourceforge.net/screenshots/barchart_demo.py
    # shows how to smarten the legend
    return short_fname

def separate_flows_stream(datas_streaming, as_list):
    """Quick hack to treat a dict of streaming flows
    Return a dict of (all and > 1MB) of downstream streaming flows
    Use as:
    flows_stream, flows_stream_1mb = separate_flows_stream(datas_streaming,
        ('AS_YOUTUBE', 'AS_YOUTUBE_EU', 'AS_GOOGLE'))
    """
    flows_stream = {}
    flows_stream_1mb = {}
    traces = set(key.strip('_DIPCP').strip('_GVB').split('_GVB_STR')[0] for key
                 in datas_streaming)
    print(traces)
    for trace in sorted(traces):
        for as_name in as_list:
            for name_postfix in ('_GVB_STR', '_GVB_STR_AS_GVB'):
                if trace + name_postfix in datas_streaming:
                    name = trace + name_postfix
            if not name:
                print("no streaming flow for trace: %s" % trace,
                      file=sys.stderr)
                continue
            data = datas_streaming[name]
            flows_stream['%s_%s' % (trace, as_name)] = data.compress(
                    map(lambda x: x in
                        INDEX_VALUES.__getattribute__(as_name),
                        data['asBGP']))
            flows_stream_1mb['%s_%s' % (trace, as_name)] = flows_stream[
                '%s_%s' % (trace, as_name)].compress(flows_stream['%s_%s' %
                        (trace, as_name)]['Session-Bytes'] > 10**6)
            del name
    # remove non signicative flows
    for flow_type in (flows_stream, flows_stream_1mb):
        for flow in flow_type.keys():
            if len(flow_type[flow]) < MIN_FLOWS_NB:
                del flow_type[flow]
    return flows_stream, flows_stream_1mb

def patch_dipcp_1mb(flows_dipcp, threshold=THRESHOLD):
    """Return a new dict filtered on flows larger than 1MB
    This patch is needed because of separation process of dipcp flows (based on
    Gvb id) and of flows dump policy in dipcp resulting in flows being split
    """
    # haskell perversion:
    return dict([(t, flows_dipcp[t].compress(
        [x['DIP-Volume-Sum-Bytes-Down']+x['DIP-Volume-Sum-Bytes-Up']>1e6
         for x in flows_dipcp[t]])) for t in flows_dipcp])
#    new_flows_dipcp = {}
#    for t in flows_dipcp:
#        new_flows_dipcp[t] = flows_dipcp[t].compress(
#            [x['DIP-Volume-Sum-Bytes-Down']+x['DIP-Volume-Sum-Bytes-Up']>1e6
#             for x in flows_dipcp[t]])
#    return new_flows_dipcp

def separate_flows_dscp(datas, threshold=THRESHOLD):
    """Returns two dict (all and > 1MB) of web + streaming flows
    """
    flows_gvb = {}
    flows_1mb_gvb = {}
    traces = set(key.split('_GVB')[0] for key in datas
                 if key.endswith('_GVB'))
    for trace in sorted(traces):
        print("processing trace: " + trace)
        data_gvb = datas['_'.join((trace, 'GVB'))]
        dscp_http_streaming, dscp_other_stream, _ = \
                un_mismatch_dscp(data_gvb)
        # separate by dscp
        flows_gvb[trace] = data_gvb.compress([x['dscp'] #in
             == dscp_http_streaming #, dscp_other_stream)
             for x in data_gvb])
        flows_1mb_gvb[trace] = flows_gvb[trace].compress(
            flows_gvb[trace]['l3Bytes'] > 1e6)
    return flows_gvb, flows_1mb_gvb

def separate_flows_all_gvb(datas, threshold=THRESHOLD):
    """Returns two dict (all and > 1MB) of all flows
    Use as:
    flows_gvb, flows_1mb_gvb = \
        tools.streaming_tools.separate_flows_all_gvb(datas)
    """
    flows_gvb = {}
    flows_1mb_gvb = {}
    traces = set(key.split('_GVB')[0] for key in datas
                 if key.endswith('_GVB'))
    for trace in sorted(traces):
        print("processing trace: " + trace)
        data_gvb = datas['_'.join((trace, 'GVB'))]
        dscp_http_streaming, dscp_other_stream, dscp_web = \
                un_mismatch_dscp(data_gvb)
        for direction in ('UP', 'DOWN'):
            key = '_'.join((trace, direction))
            # separate by dscp
            flows_gvb[key] = data_gvb.compress([x['dscp'] in
                 (dscp_http_streaming, dscp_other_stream, dscp_web)
                 for x in data_gvb])
            # separate by direction
            flows_gvb[key] = flows_gvb[key].compress(
                flows_gvb[key]['direction'] ==
                INDEX_VALUES.__getattribute__(direction))
            flows_1mb_gvb[key] = flows_gvb[key].compress(
                flows_gvb[key]['l3Bytes'] > 1e6)
    return flows_gvb, flows_1mb_gvb

def separate_flows(datas, as_list, threshold=THRESHOLD):
    """Returns two dict (all and > 1MB) of downstream streaming flows
    separated by AS list, and prints some information of number of DIPCP and
    GVB flows processed.
    Use as:
    flows_gvb, flows_1mb_gvb, flows_dipcp, flows_1mb_dipcp = separate_flows(
            datas, ('AS_YOUTUBE', 'AS_YOUTUBE_EU', 'AS_GOOGLE'))
    (flows_gvb_streams, flows_1mb_gvb_streams,
     flows_dipcp_streams, flows_1mb_dipcp_streams) = separate_flows(
            datas, ('AS_YOUTUBE', 'AS_YOUTUBE_EU', 'AS_DAILYMOTION',
            'AS_DEEZER', 'AS_GOOGLE'))
    """
    flows_gvb = {}
    flows_1mb_gvb = {}
    flows_dipcp = {}
    flows_1mb_dipcp = {}
    traces = set(key.strip('_DIPCP').strip('_GVB') for key in datas)
    for trace in sorted(traces):
        data = dict((('GVB', datas['_'.join((trace, 'GVB'))]),
            ('DIPCP', datas['_'.join((trace, 'DIPCP'))])))
        print("Nb of flows for: ", trace)
        print("GVB / 2: ", len(data['GVB']) / 2)
        print("DIPCP: ", len(data['DIPCP']))
        data['FILTERED_DIPCP'] = filter_dipcp_array(data['DIPCP'],
                                                    data['GVB'])
        print("FILTERED_DIPCP: ", len(data['FILTERED_DIPCP']))
        del data['DIPCP']
        # flows_tmp stores downstream flows
        flows_tmp = data['GVB'].compress(data['GVB']['direction'] ==
                INDEX_VALUES.DOWN)
        dscp_http_streaming, _, _ = un_mismatch_dscp(data['GVB'])
        # flows_tmp stores downstream streaming flows
        flows_tmp = flows_tmp.compress(flows_tmp['dscp'] == dscp_http_streaming)
        print("GVB DOWN STREAMING:", len(flows_tmp))
        for as_name in as_list:
            print("Construct data for AS: ", as_name)
            flows_gvb['%s_%s' % (trace, as_name)] = flows_tmp.compress(
                    map(lambda x: x in INDEX_VALUES.__getattribute__(as_name),
                        flows_tmp['asBGP']))
            print("Generated gvb flows: ", len(flows_gvb['%s_%s' % (trace,
                                                                    as_name)]))
            flows_dipcp['%s_%s' % (trace, as_name)] = filter_dipcp_array(
                    data['FILTERED_DIPCP'],
                    flows_gvb['%s_%s' % (trace, as_name)])
            print("Generated dipcp flows: ", len(flows_dipcp['%s_%s' % (trace,
                                                                    as_name)]))
            flows_1mb_gvb['%s_%s' % (trace, as_name)] = flows_gvb['%s_%s'
                    % (trace, as_name)].compress(flows_gvb['%s_%s' %
                        (trace, as_name)]['l3Bytes'] > threshold)
            print("Generated 1mb gvb flows: ", len(flows_1mb_gvb['%s_%s' % (
                trace, as_name)]))
            flows_1mb_dipcp['%s_%s' % (trace, as_name)] = filter_dipcp_array(
                    flows_dipcp['%s_%s' % (trace, as_name)],
                    flows_1mb_gvb['%s_%s' % (trace, as_name)])
            print("Generated 1mb dipcp flows: ", len(flows_1mb_dipcp['%s_%s' % (
                trace, as_name)]))
        # remove non signicative flows
        for flow_type in (flows_gvb, flows_1mb_gvb, flows_dipcp,
            flows_1mb_dipcp):
            for flow in flow_type.keys():
                if len(flow_type[flow]) < MIN_FLOWS_NB:
                    print("removing flow: ", flow)
                    del flow_type[flow]
    return flows_gvb, flows_1mb_gvb, flows_dipcp, flows_1mb_dipcp

def compute_interrupt(flows_stream, flows_stream_1mb, buffer_nb=29,
                      separate=('YOUTUBE', 'GOOGLE', 'DAILYMOTION', 'DEEZER')):
    "Returns a dict of percentage of interruptions"
    for is_1mb in (True, False):
        if is_1mb:
            flow_type = flows_stream_1mb
            flow_size = "Flows Larger than 1MB"
        else:
            flow_type = flows_stream
            flow_size = "All Flows"
        result = {}
        for trace_type in separate:
            for name in sorted(filter(lambda x: x.find(trace_type) >=0,
                    flow_type.keys())):
#        indicators_gvb=[(lambda data: map(lambda f: f[29], data['nb_hangs']),
#        "Nb hangs with buffer of 3s", "Nb")], indicators_dipcp=[])
                data = [x[buffer_nb] for x in flow_type[name]['nb_hangs']]
                len_data = len(data) + 0.
                if len_data > 0:
                    result[name + ("_1mb" if is_1mb else "") +
                           "_%d" % len_data] = len(filter(lambda x: x>0, data)) \
                                               / len_data
    return result

def plot_str_info(datas_stream, out_dir='rapport/http_stats',
                  field='Content-Avg-Bitrate-kbps',
                  out_file='cdf_avg_br.pdf',
                  title='Computed Content Average Bitrate',
                  xlabel='Content Average Bitrate in kbps'):
    """Plots the average bit rates for all flows in trace
    Use as:
    for bitrate
    tools.streaming_tools.plot_str_info(datas_stream)
    for content duration
    tools.streaming_tools.plot_str_info(datas_stream, field='Content-Duration',
        out_file='cdf_cont_dur.pdf', title='Content Duration',
        xlabel='Duration in s')
    """
    traces = set(key.strip('_DIPCP').strip('_GVB').split('_GVB_STR')[0]
                 for key in datas_stream)
    as_list = ('DAILYMOTION', 'ALL_YOUTUBE', 'GOOGLE')
    for as_name in as_list:
        args = []
        for t in sorted(traces):
            flows_with_bit_rate = filter_avg_br_ok(
                                    datas_stream[t + '_GVB_STR_AS_GVB'])
            filtered_flows = flows_with_bit_rate.compress(
                [x['asBGP'] in INDEX_VALUES.__getattribute__('AS_' + as_name)
                 for x in flows_with_bit_rate])
            if len(filtered_flows) > 10:
                args.append((format_title(t), filtered_flows[field]))
        pylab.clf()
        if field == 'Content-Duration':
            pylab.plot([16, 16], [0, 1], label='16s', linewidth=2, color='red')
            pylab.plot([208, 208], [0, 1], label='208s', linewidth=2,
                       color='blue')
            pylab.plot([700, 700], [0, 1], label='700s', linewidth=2,
                       color='green')
        cdfplot.cdfplotdataN(args, _title=' '.join((title, 'for', as_name))
                            , _xlabel=xlabel)
        if field == 'Content-Avg-Bitrate':
            pylab.xlim((1e1, 1e4))
        elif field == 'Content-Duration':
            pylab.xlim((3,1e3))
        pylab.savefig(out_dir + sep + as_name + '_' + out_file, format='pdf')
    else:
        args = []
        as_excluded = reduce(concat,
                             [INDEX_VALUES.__getattribute__('AS_' + as_name)
                                      for as_name in as_list])
        for t in sorted(traces):
            flows_with_bit_rate = filter_avg_br_ok(
                                    datas_stream[t + '_GVB_STR_AS_GVB'])
            filtered_flows = flows_with_bit_rate.compress(
                [x['asBGP'] not in as_excluded for x in flows_with_bit_rate])
            if len(filtered_flows) > 10:
                args.append((format_title(t), filtered_flows[field]))
        pylab.clf()
        if field == 'Content-Duration':
            pylab.plot([16, 16], [0, 1], label='16s', linewidth=2, color='red')
            pylab.plot([208, 208], [0, 1], label='208s', linewidth=2,
                       color='blue')
            pylab.plot([700, 700], [0, 1], label='700s', linewidth=2,
                       color='green')
        cdfplot.cdfplotdataN(args, _title=' '.join((title, 'for', 'Other ASes'))
                            , _xlabel=xlabel)
        if field == 'Content-Avg-Bitrate':
            pylab.xlim((1e1, 1e4))
        elif field == 'Content-Duration':
            pylab.xlim((3,1e3))
        pylab.savefig(out_dir + sep + 'OTHER_' + out_file, format='pdf')

def filter_datas_nb_skip(datas_stream):
    """Return two dict of arrays filtered according to
    the skip nb
    """
    datas_ok = {}
    datas_nok = {}
    for (t, data) in datas_stream.iteritems():
        data_noerr = data.compress(data['valid'] == 'OK')
        datas_ok[t] = data_noerr.compress(data_noerr['nb_skips'] == 0)
        datas_nok[t] = data_noerr.compress(data_noerr['nb_skips'] != 0)
    return datas_ok, datas_nok

def filter_datas_rates(datas_stream):
    """Return two dict of arrays filtered according to
    the skip nb
    """
    datas_ok = {}
    datas_nok = {}
    for (t, data) in datas_stream.iteritems():
        data_noerr = data.compress(data['valid'] == 'OK')
        datas_ok[t] = data_noerr.compress(
            [(1e3 * x['Content-Avg-Bitrate-kbps'])
             > (8 * x['Session-Bytes'] / x['Session-Duration'])
             for x in data_noerr])
        datas_nok[t] = data_noerr.compress(
            [(1e3 * x['Content-Avg-Bitrate-kbps'])
             <= (8 * x['Session-Bytes'] / x['Session-Duration'])
             for x in data_noerr])
    return datas_ok, datas_nok

def filter_avg_br_ok(flows):
    "Return a numpy array filtered with only correct Bitrate"
    flows_duration_ok = flows.compress(flows['Content-Duration']>0)
    flows_with_bit_rate = flows_duration_ok.compress(
        flows_duration_ok['Content-Avg-Bitrate-kbps'] != pylab.np.inf)
    flows_with_bit_rate = flows_with_bit_rate.compress(
        [repr(x['Content-Avg-Bitrate-kbps'])!='nan'
         for x in flows_with_bit_rate])
    flows_with_bit_rate = flows_with_bit_rate.compress(
        flows_with_bit_rate['Content-Avg-Bitrate-kbps'] != 0)
    return flows_with_bit_rate

def separate_flows_interrupt(flows):
    """Return 2 arrays: one with flows with interruptions than content rate
    and for other flows
    """
    flows_ok = [x['Session-Duration'] for x in flows
         if x['nb_skips'] == 0]
    flows_nok = [x['Session-Duration'] for x in flows
         if x['nb_skips'] != 0]
    return (flows_ok, flows_nok)

def separate_flows_content_bitrate(flows):
    """Return 2 arrays: one with flows with mean rate larger than content rate
    and for other flows
    """
    flows_ok = [x['Session-Duration'] for x in flows
         if 8e-3 * x['Session-Bytes'] / float(x['Session-Duration'])
        >= x['Content-Avg-Bitrate-kbps']]
    flows_nok = [x['Session-Duration'] for x in flows
         if 8e-3 * x['Session-Bytes'] / float(x['Session-Duration'])
        < x['Content-Avg-Bitrate-kbps']]
    return (flows_ok, flows_nok)

def trace_size_hack(datas_stream, out_dir='rapport/http_stats',
                    out_file='cdf_sizes_depending_function.pdf',
                    title='\nseparated according to average throughput',
                    separate_function=separate_flows_content_bitrate):
    traces = set(key.strip('_DIPCP').strip('_GVB').split('_GVB_STR')[0]
                 for key in datas_stream)
    as_list = ('DAILYMOTION', 'ALL_YOUTUBE', 'GOOGLE')
#    short_list = ('DM', 'YT', 'GOO')
#    short_name = dict(zip(as_list, short_list))
    for as_name in as_list:
        args = []
        for t in sorted(traces):
            flows_with_bit_rate = filter_avg_br_ok(
                                    datas_stream[t + '_GVB_STR_AS_GVB'])
            filtered_flows = flows_with_bit_rate.compress(
                [x['asBGP'] in INDEX_VALUES.__getattribute__('AS_' + as_name)
                 for x in flows_with_bit_rate])
            if len(filtered_flows) > 10:
                flows_ok, flows_nok = separate_function(filtered_flows)
                args.extend([(format_title('_'.join((t, 'ok'))), flows_ok),
                            (format_title('_'.join((t, 'nok'))), flows_nok)])
        pylab.clf()
        cdfplot.cdfplotdataN(args, _xlabel='Content Average Bitrate in kbps',
                             _title='Computed Content Average Bitrate for %s'
                             % as_name + title, _fs_legend='small')
#        pylab.xlim((1e1, 1e4))
        pylab.savefig(out_dir + sep + as_name + '_' + out_file, format='pdf')
    else:
        args = []
        as_excluded = reduce(concat,
                             [INDEX_VALUES.__getattribute__('AS_' + as_name)
                                      for as_name in as_list])
        for t in sorted(traces):
            flows_with_bit_rate = filter_avg_br_ok(
                                    datas_stream[t + '_GVB_STR_AS_GVB'])
            filtered_flows = flows_with_bit_rate.compress(
                [x['asBGP'] not in as_excluded for x in flows_with_bit_rate])
            if len(filtered_flows) > 10:
                flows_ok, flows_nok = separate_function(filtered_flows)
                args.extend([(format_title('_'.join((t, 'ok'))), flows_ok),
                            (format_title('_'.join((t, 'nok'))), flows_nok)])
        pylab.clf()
        cdfplot.cdfplotdataN(args, _xlabel='Content Average Bitrate in kbps',
                             _title='Computed Content Average Bitrate for %s'
                             % 'OTHER' + title, _fs_legend='small')
#        pylab.xlim((1e1, 1e4))
        pylab.savefig(out_dir + sep + 'OTHER_' + out_file, format='pdf')

def trace_size_according_fct(datas_stream, out_dir='rapport/http_stats',
                             out_file='cdf_sizes_depending_function.pdf',
                             separate_function=separate_flows_content_bitrate):
    """Plots the cdf of sizes for flows with mean rate larger than content rate
    and for other flows
    """
    traces = set(key.strip('_DIPCP').strip('_GVB').split('_GVB_STR')[0]
                 for key in datas_stream)
    args = []
    as_list = (('DailyMotion', INDEX_VALUES.AS_DAILYMOTION[0]),
               ('YouTube', INDEX_VALUES.AS_YOUTUBE[0]),
               ('YouTube-EU', INDEX_VALUES.AS_YOUTUBE_EU[0]),
               ('Google', INDEX_VALUES.AS_GOOGLE[0]))
    for t in sorted(traces):
        flows_1mb = datas_stream[t + '_GVB_STR_AS_GVB'].compress(
            datas_stream[t + '_GVB_STR_AS_GVB']['Session-Bytes']>1e6)
        flows_with_bit_rate = filter_avg_br_ok(flows_1mb)
        for (as_name, as_nb) in as_list:
            filtered_flows = flows_with_bit_rate.compress(
                flows_1mb['asBGP'] == as_nb)
            if len(filtered_flows) > 10:
                flows_ok, flows_nok = separate_function(
                    filtered_flows)
                print("trace %s, AS %s, (nok, ok, ratio): " % (t, as_name),
                        len(flows_nok), len(flows_ok),
                        len(flows_nok) / float(len(flows_nok) + len(flows_ok)))
                args.extend([('%s, %s: ok rate' % (t, as_name), flows_ok),
                             ('%s, %s: nok rate' % (t, as_name), flows_nok)])
        else:
            filtered_flows = flows_with_bit_rate.compress(
                    [x['asBGP'] not in map(itemgetter(1), as_list)
                                                for x in flows_1mb])
            if len(filtered_flows) > 10:
                flows_ok, flows_nok = separate_function(
                    filtered_flows)
                print("trace %s, AS %s, (nok, ok, ratio): " % (t, 'other'),
                        len(flows_nok), len(flows_ok),
                        len(flows_nok) / float(len(flows_nok) + len(flows_ok)))
                args.extend([('%s, %s: ok rate' % (t, 'other'), flows_ok),
                             ('%s, %s: nok rate' % (t, 'other'), flows_nok)])
    pylab.clf()
    cdfplot.cdfplotdataN(args, _xlabel="Session Sizes in Bytes",
                         _title="""Sessions Sizes depending on network speed
                         for large flows (>1MB)""",
                         _loc=0, _fs_legend='xx-small', logy=False)
    pylab.savefig(out_dir + sep + out_file, format='pdf')


def process_flows_yt_goo(flows_gvb, flows_1mb_gvb, flows_dipcp,
        flows_1mb_dipcp, indicators_gvb=None, indicators_dipcp=None):
    #TODO: make a function with only 2 input arrays
    """Take filtered flows (GVB and DIPCP) to generate a lot of graphs related
    to Youtube and Google.
    This function is a wrapper for plot_indic function and only defines the
    indicators and to which flows to apply it.
    USE AS:
    datas = tools.load_hdf5_data.load_h5_file('flows/hdf5/traces_lzf.h5')
    flows_gvb, flows_1mb_gvb, flows_dipcp, flows_1mb_dipcp = \
        tools.streaming_tools.separate_flows(datas,
            ('AS_YOUTUBE', 'AS_YOUTUBE_EU', 'AS_GOOGLE'))
    tools.streaming_tools.process_flows_yt_goo(flows_gvb, flows_1mb_gvb,
        flows_dipcp, flows_1mb_dipcp)
    For dailymotion vs yt comparison:
    (flows_gvb_yt_dai, flows_1mb_gvb_yt_dai,
     flows_dipcp_yt_dai, flows_1mb_dipcp_yt_dai) = separate_flows(
            datas, ('AS_YOUTUBE', 'AS_YOUTUBE_EU', 'AS_DAILYMOTION'))
    tools.streaming_tools.process_flows_yt_goo(flows_gvb_yt_dai,
        flows_1mb_gvb_yt_dai, flows_dipcp_yt_dai, flows_1mb_dipcp_yt_dai)
    For dailymotion and deezer:
    flows_gvb, flows_1mb_gvb, flows_dipcp, flows_1mb_dipcp = separate_flows(
            datas, ('AS_YOUTUBE', 'AS_YOUTUBE_EU', 'AS_GOOGLE'))
    (flows_gvb_streams, flows_1mb_gvb_streams,
     flows_dipcp_streams, flows_1mb_dipcp_streams) = separate_flows(
            datas, ('AS_YOUTUBE', 'AS_YOUTUBE_EU', 'AS_DAILYMOTION',
            'AS_DEEZER', 'AS_GOOGLE'))
    tools.streaming_tools.process_flows_yt_goo(flows_gvb_streams,
        flows_1mb_gvb_streams, flows_dipcp_streams, flows_1mb_dipcp_streams)
    FOR LOSS, USE AS:
    datas_mix = dict(((key, datas[key]) for key in datas
        if key.endswith('GVB')))
    del datas
    datas_loss = \
        tools.load_hdf5_data.load_h5_file('flows/hdf5/dipcp_loss_lzf.h5')
    for key in datas_loss:
        datas_mix[key] = datas_loss[key]
    del datas_loss
    flows_gvb, flows_1mb_gvb, flows_dipcp, flows_1mb_dipcp = \
        tools.streaming_tools.separate_flows(datas_mix, ('AS_YOUTUBE',
            'AS_YOUTUBE_EU', 'AS_DAILYMOTION', 'AS_DEEZER', 'AS_GOOGLE'))
    tools.streaming_tools.process_flows_yt_goo(flows_gvb, flows_1mb_gvb,
        flows_dipcp, flows_1mb_dipcp, indicators_gvb=[], indicators_dipcp=[
            (lambda data: [(x['DIP-DSQ-NbMes-sec-TCP-' + "Up"] + 1e-99)
                /x['DIP-Volume-Number-Packets-' + "Up"] for x in data
                if x['DIP-Volume-Number-Packets-' + "Up"] != 0],
                    "Loss Rate " + "Up", "Percent"),
            (lambda data: [(x['DIP-DSQ-NbMes-sec-TCP-' + "Down"] + 1e-99)
                /x['DIP-Volume-Number-Packets-' + "Down"] for x in data
                if x['DIP-Volume-Number-Packets-' + "Down"] != 0],
                    "Loss Rate " + "Down", "Percent")])
    FOR RTT DATA:
    datas_mix = dict(((key, datas[key]) for key in datas
        if key.endswith('GVB')))
    datas_rtt = cPickle.load(open('flows_dipcp_rtt_data.pickle'))
    for key in datas_rtt:
        datas_mix[key + '_DIPCP'] = datas_rtt[key]
    del datas_rtt
    _, _, flows_dipcp, flows_1mb_dipcp = \
        tools.streaming_tools.separate_flows(datas_mix, ('AS_YOUTUBE',
            'AS_YOUTUBE_EU', 'AS_DAILYMOTION', 'AS_DEEZER', 'AS_GOOGLE'))
    ALSO PICKLED as flows_dipcp_rtt_streams.pickle and
        flows_1mb_dipcp_rtt_streams.pickle
    tools.streaming_tools.process_flows_yt_goo(None, None,
        flows_dipcp, flows_1mb_dipcp, indicators_gvb=[],
            indicators_dipcp=[((lambda data: data['DIP-RTT-Min-ms-TCP-Down']),
                    'Min RTT DATA Down', 'ms')])
    tools.streaming_tools.process_flows_yt_goo(None, None,
        flows_dipcp, flows_1mb_dipcp, indicators_gvb=[],
            indicators_dipcp=[((lambda data: data['DIP-RTT-Mean-ms-TCP-Down']),
                    'Mean RTT DATA Down', 'ms')])
    tools.streaming_tools.process_flows_yt_goo(None, None,
        flows_dipcp, flows_1mb_dipcp, indicators_gvb=[],
            indicators_dipcp=[((lambda data: data['DIP-RTT-Max-ms-TCP-Down']),
                    'Max RTT DATA Down', 'ms')])
    tools.streaming_tools.process_flows_yt_goo(None, None,
        flows_dipcp, flows_1mb_dipcp, indicators_gvb=[],
            indicators_dipcp=[((lambda data: data['DIP-RTT-Min-ms-TCP-Up']),
                    'Min RTT DATA Up', 'ms')])
    tools.streaming_tools.process_flows_yt_goo(None, None,
        flows_dipcp, flows_1mb_dipcp, indicators_gvb=[],
            indicators_dipcp=[((lambda data: data['DIP-RTT-Mean-ms-TCP-Up']),
                    'Mean RTT DATA Up', 'ms')])
    tools.streaming_tools.process_flows_yt_goo(None, None,
        flows_dipcp, flows_1mb_dipcp, indicators_gvb=[],
            indicators_dipcp=[((lambda data: data['DIP-RTT-Max-ms-TCP-Up']),
                    'Max RTT DATA Up', 'ms')])

    DO NOT WORK WITH TOO MUCH LIST COMPREHENSION!!!
            [x for rtt_type in ('Min', 'Mean', 'Max') for x in
            ((lambda data: data['DIP-RTT-%s-ms-TCP-%s' % (rtt_type, "Up")],
                    ' '.join((rtt_type, "RTT DATA", "Up")), "ms"),
            (lambda data: data['DIP-RTT-%s-ms-TCP-%s' % (rtt_type, "Down")],
                    ' '.join((rtt_type, "RTT DATA", "Down")), "ms"))])

    FOR STREAMING STATS,
    tools.streaming_tools.append_data_dict_with_stream(datas,
        'flows/stream_quality/links')
    datas_streaming = tools.streaming_tools.filter_stream_dict(datas)
    flows_stream, flows_stream_1mb =
        tools.streaming_tools.separate_flows_stream(
            datas_streaming, ('AS_YOUTUBE', 'AS_YOUTUBE_EU', 'AS_GOOGLE'))
    tools.streaming_tools.process_flows_yt_goo(flows_stream, flows_stream_1mb,
        {}, {},
        indicators_gvb=[(lambda data: map(lambda f: f[29], data['nb_hangs']),
        "Nb hangs with buffer of 3s", "Nb")], indicators_dipcp=[])
    tools.streaming_tools.process_flows_yt_goo(flows_stream, flows_stream_1mb,
        {}, {},
        indicators_gvb=[(lambda data: data['Content-Duration'],
        'Content-Duration', "Seconds")], indicators_dipcp=[])
    """
    if indicators_gvb is None:
        indicators_gvb = (
            # 8*bytes/(1000s) => kbit/s
            (lambda data: [8*x['l3Bytes']/(1000.0*x['duration']) for x in data
                if x['duration'] > 0], "Mean Rate ", "kbit/s"),
            #80*bytes/100ms => bit/s
            (lambda data: 80*data['peakRate'], "Peak Rate", "bit/s"),
            (lambda data : data['duration'], "Duration", "Seconds"),
            (lambda data : data['l3Bytes'], "Size", "Bytes")
            )
    for indicator_gvb in indicators_gvb:
        plot_indic((flows_gvb, flows_1mb_gvb), indicator_gvb)
    if indicators_dipcp is None:
        indicators_dipcp = []
        for rtt_type in ("Mean", "Min", "Max"):
            indicators_dipcp.append(
                    (lambda data: data['DIP-RTT-%s-ms-TCP-%s' % (
                        rtt_type, "Up")],
                        ' '.join((rtt_type, "RTT SYN-ACK", "Up")), "ms"))
            indicators_dipcp.append(
                    (lambda data: data['DIP-RTT-%s-ms-TCP-%s' % (
                        rtt_type, "Down")],
                        ' '.join((rtt_type, "RTT SYN-ACK", "Down")), "ms")
                    )
#        for direction in ("Up", "Down"):
#            dipcp_rtt_field = 'DIP-RTT-%s-ms-TCP-%s' % (rtt_type, direction)
#            indicators_dipcp.append((lambda data: data[dipcp_rtt_field],
#                "RTT " + direction, "ms"))
    for indicator_dipcp in indicators_dipcp:
        plot_indic((flows_dipcp, flows_1mb_dipcp), indicator_dipcp)

def filter_dict(flows, match):
    "Return a dict with only matching string in keys"
    return dict([(k, v) for (k, v) in flows.iteritems()
                 if match in k])

def plot_indic((flows, flows_1mb), (func, indic_title, unit),
        output_path = 'rapport/http_stats', prefix='yt_goo_n',
        title = 'HTTP Streaming Downstream ', separate=('YOUTUBE', 'GOOGLE',
                                                    'DAILYMOTION', 'DEEZER')):
    """Plot the cdf and ccdf of an specified indicator:
    separates ADSL vs. FTTH and all vs. large flows
    Use with wrapper
    Or for all peak rates:
    flows_gvb, flows_1mb_gvb = \
        tools.streaming_tools.separate_flows_all_gvb(datas)
    tools.streaming_tools.plot_indic((flows_gvb, flows_1mb_gvb),
        (lambda data: 80*data['peakRate'], "Peak Rate", "bit/s"),
        prefix='all', title='Web + Streaming', separate=('UP', 'DOWN'))
    """
    for trace_type in separate:
        for is_1mb in (True, False):
            if is_1mb:
                flow_type = flows_1mb
                flow_size = "Flows Larger than 1MB"
            else:
                flow_type = flows
                flow_size = "All Flows"
            args = []
            for name in sorted([k for k in flow_type if trace_type in k]):
                data = func(flow_type[name])
                if len(data) > 0:
                    args.append((format_title(name.split(trace_type)[0]), data))
            pylab.clf()
            plot_title = (' '.join((title, flow_size)) + ':\n'
                      + indic_title + ' for ' + trace_type).replace('_', ' ')
            cdfplot.cdfplotdataN(args, _title=plot_title, _loc=0,
                                 _fs_legend='small', logy=False,
                                 _xlabel='%s in %s' % (indic_title, unit))
            # for loss rate
            if 'Loss' in indic_title:
                pylab.xlim(xmin=1e-5)
            if indic_title == "Mean Rate ":
                pylab.plot([380, 380], [0, 1], linewidth=2, color='red')
            if indic_title == "Duration":
                pylab.plot([20, 20], [0, 1], linewidth=2, color='red')
                pylab.xlim([1,1e4])
            pylab.savefig(output_path + sep + '_'.join((prefix,
                indic_title.strip(), flow_size, trace_type, 'cdf', 'logx',
                'new')).lower().replace(' ', '_') + '.pdf')
            pylab.clf()
            cdfplot.cdfplotdataN(args, _title=plot_title, _loc=0,
                                 _fs_legend='x-small', logy=True,
                                 _xlabel='%s in %s' % (indic_title, unit))
#            pylab.xlim(xmin=1e-5)
            if indic_title == "Mean Rate ":
                pylab.plot([380, 380], [0, 1], linewidth=2, color='red')
            pylab.savefig(output_path + sep + '_'.join((prefix,
                indic_title.strip(), flow_size, trace_type, 'cdf', 'loglog',
                'new')).lower().replace(' ', '_') + '.pdf')
            pylab.clf()
            cdfplot.ccdfplotdataN(args, _title=plot_title, _loc=0,
                                  _fs_legend='x-small',
                                  _xlabel='%s in %s' % (indic_title, unit))
            pylab.savefig(output_path + sep + '_'.join((prefix,
                indic_title.strip(), flow_size, trace_type, 'ccdf', 'new')).lower().
                replace(' ', '_') + '.pdf')

def extract_data(data, percent=True):
    "Return the data formated to get the remaining time"
    if percent:
        flow = [max(MIN_PERCENT, (1 - (0.0 + x['Session-Bytes'])
                                  / x['Content-Length']))
                for x in data
                if x['Content-Length']>0]
    else:
        flow = [max(MIN_PERCENT, x['Content-Duration']
                    * (1 - (0.0 + x['Session-Bytes'])
                       / x['Content-Length']))
                for x in data
                if x['Content-Duration']>0
                and x['Content-Length']>0]
    return flow

def plot_flows_per_client(datas, with_errors=False,
                          separate=(('AS_YOUTUBE', 'AS_YOUTUBE_EU'),
                                    ('AS_GOOGLE',), ('AS_DAILYMOTION',),
                                    ('AS_DEEZER',), None)):
    """Plot cdf of nb of flows per client per AS
    """
    for as_name in separate:
        data_len = 0
        args = []
        for name in sorted(datas.keys()):
            if as_name:
                data = datas[name].compress(
                    map(lambda x:
                        any(x in INDEX_VALUES.__getattribute__(as_n)
                            for as_n in as_name),
                            datas[name]['asBGP']))
                print_as_name = as_name[0]
            else:
                # find out other ASes
                separate_concat = reduce(concat, [x for x in separate if x])
                all_as = reduce(concat, [INDEX_VALUES.__getattribute__(x)
                                for x in separate_concat if x])
                data = datas[name].compress([x not in all_as for x in
                                             datas[name]['asBGP']])
                print_as_name = 'OTHER'
            if not with_errors:
                data = data.compress(data['Content-Type']!='error')
            data = data.compress(data['Content-Length'] > 0)
            data = data.compress(data['Content-Avg-Bitrate-kbps'] > 1)
            cur_len = len(data)
            args.append((format_title(name.split('_GVB_STR_AS_GVB')[0])
                         + ", nb: %d" % cur_len,
                        aggregate.aggregate_nb(data, 'dstAddr').values()))
            data_len += cur_len
        plot_args(args, (print_as_name, "", data_len),
                  output_path='flows_per_client',
                  xlabel="Nb of flows per client",
                  prefix="nb_flows_per_client",
                  with_errors=with_errors, percent=False)

def filter_flows(datas, AS_include=('AS_YOUTUBE', 'AS_YOUTUBE_EU',
                                    'AS_GOOGLE', 'AS_DAILYMOTION',
                                    'AS_DEEZER')):
    """Return a dict with only flows on these ASes
    datas = tools.load_hdf5_data.load_h5_file('flows/hdf5/hdf5_streaming.h5')
    sessions_filtered = {}
    for t in sorted(datas_filtered):
        print("processing " + t)
        sessions_filtered[t] = tools.flow2session.process_stream_session(
            datas_filtered[t], message="filtered ASes")
    tools.streaming_tools.plot_sessions_per_clients(sessions_filtered)
 """
    out_dict = {}
    for name in sorted(datas.keys()):
        data = datas[name]
        cur_len = len(data)
        all_as = reduce(concat, [INDEX_VALUES.__getattribute__(x)
                                 for x in AS_include if x])
        out_dict[name] = datas[name].compress([x in all_as for x in
                                     datas[name]['asBGP']])
    return out_dict


def plot_sessions_per_clients(datas, message="all ASes"):
    """Plot cdf of nb of flows per sessions
    datas = tools.load_hdf5_data.load_h5_file('flows/hdf5/hdf5_streaming.h5')
    sessions = {}
    for t in sorted(datas):
        print("processing " + t)
        sessions[t] = tools.flow2session.process_stream_session(datas[t])
    tools.streaming_tools.plot_sessions_per_clients(sessions)
    """
    args = []
    data_len = 0
    for name in sorted(datas.keys()):
        data = datas[name]
        cur_len = len(data)
        args.append((format_title(name.split('_GVB_STR_AS_GVB')[0])
                     + ", nb: %d" % cur_len,
                    aggregate.aggregate_nb(data, 'dstAddr').values()))
        data_len += cur_len
    plot_args(args, ("gap_%d" % 30, message, data_len),
          output_path='flows_per_session',
          xlabel="Nb of flows per session",
          prefix="nb_flows_per_session",
          with_errors=False, percent=False)


#for f in os.listdir('flows/stream_quality/links/AS') :
#    print(f)
#    datas[f.split('.')[0]]=np.load('/'.join(('flows/stream_quality/links/AS',
#    f)))
def plot_remaining_streaming_times(datas, with_errors=False, percent=True,
                                   separate=('AS_YOUTUBE', 'AS_YOUTUBE_EU',
                                             'AS_GOOGLE', 'AS_DAILYMOTION',
                                             'AS_DEEZER', None)):
    """Plot cdf of remaining times according to content duration
    Use as:
    datas = cPickle.load(open('streaming_quality_dict.pickle'))
    tools.streaming_tools.plot_remaining_streaming_times(datas,
        with_errors=False)
    """
    for name in sorted(datas.keys()):
        for as_name in separate:
            if as_name:
                data = datas[name].compress(
                    map(lambda x: x in INDEX_VALUES.__getattribute__(as_name),
                            datas[name]['asBGP']))
            else:
                # find out other ASes
                all_as = reduce(concat, [INDEX_VALUES.__getattribute__(x)
                                for x in separate if x])
                data = datas[name].compress([x not in all_as for x in
                                             datas[name]['asBGP']])
                as_name = 'OTHER'
            if not with_errors:
                data = data.compress(data['Content-Type']!='error')
            data = data.compress(data['Content-Length'] > 0)
            data = data.compress(data['Content-Avg-Bitrate-kbps'] > 1)
            args = []
            for minutes in map(lambda x : 2**x, xrange(8)):
                # for having 0 included
                min_dur = 60 * (minutes / 2)
                max_dur = 60 * minutes
                filtered_data = data.compress(
                    [x > min_dur and x <= max_dur
                     for x in data['Content-Duration']])
                if len(filtered_data) > 0:
                    args.append(("%d content < %d min"
                                 % (len(filtered_data),minutes),
                                 extract_data(filtered_data, percent)))
            # for else
            else:
                # for 0 durations
                filtered_data = data.compress(data['Content-Duration']==0)
                if len(filtered_data) > 0:
                    args.append(("%d content = 0 min" % len(filtered_data),
                             extract_data(filtered_data, percent)))

            plot_args(args, (name, as_name, len(data)),
                      output_path='remaining_time_percent',
                      with_errors=with_errors, percent=percent,
                      _xlabel='HTTP Streaming Down:' + 'Remaining time in '
                                 + 'percent of total size' if percent
                                     else 's')


def plot_args(args, (name, as_name, data_len), with_errors=False, percent=True,
              output_path='remaining_time_percent', xlabel="X",
              prefix='separate_remaining'):
    "Plot arguments"
#    title = 'HTTP Streaming Down: '
    if args:
        # linear plot
        pylab.clf()
        cdfplot.cdfplotdataN(args,
                             _title=name.replace('_', ' ')
                                 + (' with errors' if with_errors
                                     else ' no errors')
                                 + '\n' + as_name + ' '
                                 + "total nb of streams: %d" % data_len,
                             _xlabel=xlabel,
                             _loc=0, _fs_legend='small', logx=False)
        pylab.savefig(output_path + sep
                      + '_'.join((prefix, name, as_name, 'linear', 'cdf',
                                  '' if with_errors else 'no_errors',
                                  'percent' if percent else ''))
                      .lower().replace(' ', '_') + '.pdf')

        # log plot
        pylab.clf()
        cdfplot.cdfplotdataN(args,
                             _title=name.replace('_', ' ')
                                 + (' with errors' if with_errors
                                     else ' no errors')
                                 + '\n' + as_name + ' '
                                 + "total nb of streams: %d" % data_len,
                             _xlabel=xlabel,
                             _loc=0, _fs_legend='small', logx=True)
        pylab.savefig(output_path + sep
                      + '_'.join((prefix, name, as_name, 'xlog', 'cdf',
                                  '' if with_errors else 'no_errors',
                                  'percent' if percent else ''))
                      .lower().replace(' ', '_') + '.pdf')


def check_streaming_size(datas):
    "Prints the mean flow size of streaming flows"
    all_values = []
    traces = set(key.strip('_DIPCP').strip('_GVB') for key in datas)
    for trace in sorted(traces):
        print("Mean stream flow size for large flows: ", trace, end='')
        all_values.append(pylab.mean([x['l3Bytes']
                                      for x in datas[trace + '_GVB']
                                       if x['l3Bytes'] > 10**6]))
        print(all_values[-1])
    print("Overall mean: ", pylab.mean(all_values))

def load_cnx_stream(data_dir='flows/links/cnx_stream'):
    "Return a dict of cnx_stream data"
    pattern = re.compile('^cnx_stream_.*\.npy')
    return dict([(f.split('.')[0].split('cnx_stream_')[1],
                  np.load(sep.join((data_dir, f))))
                 for f in filter(pattern.match, listdir(data_dir))])

def retrieve_bitrate(d):
    "Return the extracted bitrate"
    if d.startswith(('mp4', 'MP4', 'FLV', 'M4', 'FACE', 'qt', 'isom', '3gp')):
        bitrate = d.split('_')[-1]
        if re.search("\dk$", bitrate):
            bitrate = int(bitrate.strip('k'))
            if bitrate != 0:
                return bitrate
#        else:
#            print("unkown format in " + d)
#    else:
#        print("unexpected format " + d)

def retrieve_resolution(d):
    "Return the extracted resolution"
    if len(d.split('_')) == 4:
        return  mul(*map(int, d.split('_')[1:3]))

def extract_bitrate(cnx_stream):
    "Wrapper on generic function"
    return extract_function(cnx_stream, retrieve_bitrate)

def extract_resolution(cnx_stream):
    "Wrapper on generic function"
    return extract_function(cnx_stream, retrieve_resolution)

def extract_function(cnx_stream, retrieve_function):
    """Return two argument list of tuple (name, data) with name of trace and
    data is a list of either resolution or bitrate
    """
    args = []
    for t in sorted(cnx_stream.keys()):
        data = cnx_stream[t]['Application']
        values = []
        nb_tot = nb_ok = 0
        for d in data:
            value = retrieve_function(d)
            nb_tot += 1
            if value:
                values.append(value)
                nb_ok += 1
        if len(values) > 0:
            args.append((format_title(t), values))
        print("Trace %s found %d records and %d valid ones"
              % (t, nb_tot, nb_ok))
    return args


def trace_resol_hack(cnx_stream, out_dir='rapport/http_stats',
                    out_file='cdf_resolution.pdf',
                    title='Video Resolution for ',
                    extract=extract_resolution,
                    filtered_large=False):
    as_list = ('DAILYMOTION', 'ALL_GOOGLE')
    for as_name in as_list:
        filtered_cnx_stream = {}
        for t in sorted(cnx_stream):
            filtered_flows = cnx_stream[t].compress(
                [x['RemAS'] in map(lambda s: 'AS' + str(s),
                   INDEX_VALUES.__getattribute__('AS_' + as_name))
                 for x in cnx_stream[t]])
            if filtered_large:
                filtered_flows = filtered_flows.compress(
                                                filtered_flows['ByteDn']>1e6)
            if len(filtered_flows) > 10:
                filtered_cnx_stream[t] = filtered_flows
        args = extract(filtered_cnx_stream)
        print(as_name)
        for (k, v) in args:
            print(k)
            print("Mean: ", np.mean(v))
            print("Median: ", np.median(v))
        pylab.clf()
        pylab.plot([921600, 921600], [0, 1], linewidth=2, color='red',
                   label='HD resolution')
        cdfplot.cdfplotdataN(args, _xlabel='Video Resolution in square pixels',
                             _title=title + as_name) #, _fs_legend='small')
        pylab.xlim((1e3, 1e6))
        pylab.savefig(out_dir + sep + as_name + '_' + out_file, format='pdf')
    else:
        filtered_cnx_stream = {}
        as_excluded = reduce(concat,
                             [INDEX_VALUES.__getattribute__('AS_' + as_name)
                                      for as_name in as_list])
        for t in sorted(cnx_stream.keys()):
            filtered_flows = cnx_stream[t].compress(
                [x['RemAS'] not in map(lambda s: 'AS' + str(s), as_excluded)
                 for x in cnx_stream[t]])
            if len(filtered_flows) > 10:
                filtered_cnx_stream[t] = filtered_flows
        args = extract(filtered_cnx_stream)
        print(as_name)
        for (k, v) in args:
            print(k)
            print("Mean: ", np.mean(v))
            print("Median: ", np.median(v))
        pylab.clf()
        pylab.plot([921600, 921600], [0, 1], linewidth=2, color='red',
                   label='HD resolution')
        cdfplot.cdfplotdataN(args, _xlabel='Video Resolution in square pixels',
                             _title=title + 'OTHER') #, _fs_legend='small')
        pylab.xlim((1e3, 1e6))
        pylab.savefig(out_dir + sep + 'OTHER_' + out_file, format='pdf')

def extract_volumes_per_as(datas_stream, normalized=False,
       as_list=('DAILYMOTION', 'YOUTUBE', 'YOUTUBE_EU', 'GOOGLE', 'LIMELIGHT')):
    """Return a dict of volumes per AS over the trace duration
    USE AS:
datas_stream = tools.load_hdf5_data.load_h5_file('flows/hdf5/hdf5_streaming.h5')
datas_stream_1mb = dict([(k, v.compress(v['Session-Bytes']>1e6))
                                for (k,v) in datas_stream.iteritems()])
datas_stream_as = tools.streaming_tools.extract_volumes_per_as(datas_stream,
                                                        normalized=True)
tools.streaming_tools.bar_chart(datas_stream_as,
                title='HTTP Streaming Flows Normalized Volume per User',
                ylabel='Volume per User in Bytes per Second')
datas_stream_1mb_as = tools.streaming_tools.extract_volumes_per_as(
                                        datas_stream_1mb, normalized=True)
tools.streaming_tools.bar_chart(datas_stream_1mb_as,
    title='HTTP Streaming Flows Normalized Volume per User (Large flows only)',
    ylabel='Volume per User in Bytes per Second per User')
Or use with CnxStream files:
datas_cnx = dict([(t.split('.npy')[0], np.load('flows/links/cnx_stream/' + t))
    for t in os.listdir('flows/links/cnx_stream') if t.endswith('.npy')])
datas_cnx_as = tools.streaming_tools.extract_volumes_per_as(datas_cnx,
                                                        normalized=True)
tools.streaming_tools.bar_chart(datas_cnx_as,
    title='Cnx Streaming Flows Normalized Volume per User',
    ylabel='Volume per User in Bytes per Second per User')
datas_cnx_1mb = dict([(k, v.compress(v['nByte']>1e6))
                                for (k,v) in datas_cnx.iteritems()])
datas_cnx_1mb_as = tools.streaming_tools.extract_volumes_per_as(
                                        datas_cnx_1mb, normalized=True)
tools.streaming_tools.bar_chart(datas_cnx_1mb_as,
    title='Cnx Streaming Flows Normalized Volume per User (Large flows only)',
    ylabel='Volume per User in Bytes per Second per User')
    """
    datas = {}
    for (t, data) in sorted(datas_stream.iteritems()):
        # clean up
        if data.dtype == INDEX_VALUES.dtype_GVB_streaming_AS:
            bytes_field = 'Session-Bytes'
            clients_field = 'srcAddr'
            as_index = 'asBGP'
            adjust_AS = functional.id
            flows = data.compress(data[bytes_field] < 1e9)
            start_time = min(flows['initTime'])
            end_time = max([x['initTime'] + x['Session-Duration']
                            for x in flows])
            duration = end_time - start_time
        elif (data.dtype == INDEX_VALUES.dtype_cnx_stream
              or data.dtype == INDEX_VALUES.dtype_cnx_stream_loss):
            clients_field = 'Name'
            bytes_field = 'nByte'
            as_index = 'RemAS'
            adjust_AS = lambda s: int(s.lstrip('AS') if s.startswith('AS')
                                      else 0)
            # We assume correctness of data!
            flows = data
            duration = 1
        elif (data.dtype == INDEX_VALUES.dtype_all_stream_indics_final_tstat):
            clients_field = 'client_id'
            bytes_field = 'Session-Bytes'
            as_index = 'asBGP'
            adjust_AS = functional.id
            # We assume correctness of data!
            flows = data
            start_time = min(flows['initTime'])
            end_time = max([x['initTime'] + x['Session-Duration']
                            for x in flows])
            duration = end_time - start_time
        else:
            assert False, 'Unknown dtype'
        nb_clients = len(set(flows[clients_field]))
        print("duration, nb record, nb clients for %s: (%d, %d, %d)"
                % (t, duration/60, len(flows), nb_clients))
        for as_name in as_list:
            filtered_flows = flows.compress(
                [adjust_AS(x[as_index]) in
                 INDEX_VALUES.__getattribute__('AS_' + as_name) for x in flows])
            if len(filtered_flows) > 5:
                datas['_AS_'.join((t.split('_GVB_STR_AS_GVB')[0], as_name))] = (
                    sum(filtered_flows[bytes_field])
                    / duration
                    / float(1 if not normalized
                            else
                            len(set(filtered_flows[clients_field]))))
    # should normalize on total nb of clients or only on AS??
    return datas

def calculate_sessions(datas_stream, gap=GAP):
    """Wrapper for process_stream_session on a data streaming dict
    """
    return dict([(t, process_stream_session(v, gap=gap, skip_nok=False))
                 for (t, v) in datas_stream.items()])

def plot_all_sessions(datas_sessions, nb_hh=20, out_dir='sessions_plots',
                      duration=60):
    """Wrapper for plot_sessions_per_client on a session dict
    USE AS:
datas_stream = tools.load_hdf5_data.load_h5_file(
    'flows/hdf5/hdf5_streaming.h5')
datas_sessions = dict([(f.split('_GVB_STR_AS.npy')[0],
                    np.load('sessions_no_nok/' + f))
                    for f in filter(lambda s: s.endswith('.npy'),
                    os.listdir('sessions_no_nok'))])
    DON'T USE THE PICKLE!
tools.streaming_tools.plot_all_sessions(datas_sessions)
    FOR CNX_STREAMS:
cnx_stream = dict([(f.split('sessions_cnx_stream_')[1].split('.txt')[0],
    np.loadtxt('flows/links/cnx_stream/' + f, delimiter=';',
        dtype=tools.INDEX_VALUES.dtype_streaming_session_other))
    for f in os.listdir('flows/links/cnx_stream')
        if f.startswith('sessions_cnx_stream_')])
tools.streaming_tools.plot_all_sessions(cnx_stream, duration=24*60,
    out_dir='sessions_cnx_stream')
    FOR CNX_STREAMS 10 MIN:
cnx_10min_stream = dict([
    (f.split('sessions_10min_cnx_stream_')[1].split('.txt')[0],
    np.loadtxt('flows/links/cnx_stream/' + f, delimiter=';',
        dtype=tools.INDEX_VALUES.dtype_streaming_session_other))
    for f in os.listdir('flows/links/cnx_stream')
        if f.startswith('sessions_10min_cnx_stream_')])
tools.streaming_tools.plot_all_sessions(cnx_10min_stream, duration=24*60,
    out_dir='sessions_10min_cnx_stream')
    """
#    nb_data = len(datas_sessions)
#    assert nb_data % 2 == 0, "Missing sessions"
    for n, (trace, data) in enumerate(datas_sessions.iteritems()):
        print("Processing trace:", trace)
        heavy_hitters = map(itemgetter(0), sorted(aggregate.aggregate_sum(data,
                                                    'dstAddr', 'session_bytes'),
                                      key=itemgetter(1), reverse=True)[0:nb_hh])
        for field in ('session_bytes', ): #None, 'nb_streams'):
#            fig = plt.figure()
#            ax = fig.add_subplot(111) #2, nb_data / 2, n + 1)
#            plot_sessions_per_client(data, ax, trace_name=trace, field=field)
#            fig.savefig(sep.join((out_dir,
#                                  'sessions_%s_%s.pdf' % (field, trace))))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plot_sessions_per_client(
                data.compress([x['dstAddr'] in heavy_hitters for x in data]),
                ax , duration=duration, trace_name=trace, field=field,
                title='Sessions for Heavy Hitters')
            fig.savefig(sep.join((out_dir,
                                  'sessions_HH_%s_%s.pdf' % (field, trace))))

def plot_sessions_per_client(sessions_stats, ax, trace_name=None,
                             field='session_bytes',
                             duration=60,
                             title='Sessions per client'):
    """Plot a graph representing the session ranges out of sessions stats
    To add or not ont to add nok stats: this is the question
    """
    assert sessions_stats.dtype == INDEX_VALUES.dtype_streaming_session \
        or sessions_stats.dtype == INDEX_VALUES.dtype_streaming_session_other, \
            "Incorrect dtype"
    colors = dict(enumerate("bgrcmy"))
    # give clients by increasing number of sessions
    y_offset = 0
    heights = []
    for nb, (client, vol) in enumerate(sorted([(d, sum(
        sessions_stats.compress(sessions_stats['dstAddr']==d)['session_bytes']))
            for d in set(sessions_stats['dstAddr'])],
        key=itemgetter(1))):
        max_height = 0
        for (beg, end) , height in [((x['beg'], x['end']),
                                     x[field] if field else 1)
                    for x in sessions_stats.compress(
                        sessions_stats['dstAddr']==client)]:
            width = end - beg
            if width == 0:
                print("removed 0 duration flow", sys.stderr)
                continue
            height_normalized = 8 * height / width
            # errors in cnx_stream files
            if height_normalized > 5e9:
                print("removed too high throughput flow:", sys.stderr)
                continue
            rect = mpatches.Rectangle((beg/60, y_offset), width/60,
                                      height_normalized,
                                      linewidth=0, label='Vol: %d' % vol)
            rect.set_facecolor(colors[nb % len(colors)])
            rect.set_edgecolor('k')
            ax.add_patch(rect)
            max_height = max(max_height, height_normalized)
        ax.text(duration, y_offset, '%.1e' % vol, size=8)
        y_offset += max_height
        heights.append(max_height)
    print('height: median %f, mean %f, std dev: %f' % (np.median(heights)/1e3,
                                   np.mean(heights)/1e3, np.std(heights)/1e3))
    ax.set_xlim([0, duration])
    ax.set_ylim([0, y_offset])
    if field == 'session_bytes':
        ylabel = 'with height indicating the mean throughput in b/s'
#        step = 480
#        ydata = [x * step for x in np.arange(y_offset // step + 1)]
#        ax.yaxis.set_ticks(ydata)
#        ax.grid(ydata=ydata)
    elif field == 'nb_streams':
        ylabel = 'with height indicating the mean number of flows per second'
    else:
        ylabel = ''
    ax.set_ylabel('\n'.join(('Client ordered by increasing volume',
                 ylabel)))
    ax.set_xlabel('Time in Minutes')
#    ax.yaxis.set_major_locator(NullLocator())
    if trace_name:
        ax.set_title(': '.join((title, format_title(
                                trace_name.split('sessions_GVB_')[1]
        if 'sessions_GVB_' in trace_name else trace_name))))

def main():
    "Run a function in non-interactive mode"
    import load_hdf5_data
    datas = load_hdf5_data.load_h5_file('flows/hdf5/traces_lzf.h5')
    flows_gvb, flows_1mb_gvb, flows_dipcp, flows_1mb_dipcp = separate_flows(
            datas, ('AS_YOUTUBE', 'AS_YOUTUBE_EU', 'AS_GOOGLE'))
    process_flows_yt_goo(flows_gvb, flows_1mb_gvb, flows_dipcp,
        flows_1mb_dipcp)

if __name__ == '__main__':
    print("run doctest")
    from doctest import testmod
    testmod()
#    main()

