#!/usr/bin/env python
"""Module to provide missing stats for streaming analysis
"""

from __future__ import division, print_function
from operator import concat, itemgetter
from collections import defaultdict
from itertools import islice, cycle
from random import random
from tempfile import NamedTemporaryFile
import os
import numpy as np
# in case of non-interactive usage
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#import matplotlib.ticker as ticker

import INDEX_VALUES
import streaming_tools
#import aggregate
import flow2session
#from filter_streaming import generate_remaining_download_cnx_stream

# for 3D plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
#from matplotlib.ticker import NullLocator

# from sage colors
ORANGE = (1.0, 0.6470588235294118, 0.0)

from INDEX_VALUES import UNKNOWN_ID, EPSILON

WIDTH_IDX = 1
NBFLOW_IDX = 0
VOL_IDX = 0

def construct_bar_data(data_raw, vol_th, percent_functions,
                       as_list=('DAILYMOTION', 'ALL_YOUTUBE', 'GOOGLE')):
    "Return a list of tuple (AS, dict of values) for plotting bars"
    # warning hardcoded match of percent and as values
    return dict([(trace,
                  dict([(as_name, dict([(percent_type,
                                         len(filter(comp_func, data_as)))
                                        for percent_type, comp_func
                                        in percent_functions.items()]))
                        for as_name in as_list
                        for data_as in ([x[1] for x in data if x[2]
                        in INDEX_VALUES.__getattribute__('AS_' + as_name)],)]))
                 for trace in data_raw
                 for data in ([y for y in data_raw[trace] if y[0] > vol_th],)])

def load_stream_qual(data_dir='flows/stream_quality/links/AS'):
    "Wrapper to load all streaming quality stats in a dict"
    return dict([(f.split('GVB_', 1)[1].split('_GVB_STR_AS.npy')[0],
                  np.load(os.sep.join((data_dir, f))))
                 for f in os.listdir(data_dir)])

def generate_remaining_download(cnx_stream):
    "Filter flows and generate the data of remainig download volume in percent"
    return dict([(k, zip(v['Content-Length'],
                         100 * v['Session-Bytes'] / v['Content-Length'],
                         v['asBGP'], v['valid']))
                 for k, data in cnx_stream.iteritems()
                 for v in (data.compress(data['Content-Length'] != 0),)])
#                 for tmp in (data,) #.compress(data['valid'] == 'OK'),)
#                 for v in (tmp.compress(tmp['Session-Bytes'] <=
#                                        tmp['Content-Length']),)])

def load_files_to_validate():
    "Return a dict of remaining download stats for validation"
    cnx_stream = {}
    active_stats_dir = 'traces/active_captures/captures_streaming_full'
    stats_file = 'streaming_stats_txt_AS_txt.npy'
    for entry in os.listdir(active_stats_dir):
        dir_entry = os.sep.join((active_stats_dir, entry))
        if (os.path.isdir(dir_entry) and 'deezer' not in entry and
            stats_file in os.listdir(dir_entry)):
            cnx_stream[entry] = np.load(os.sep.join((dir_entry, stats_file)))
    return cnx_stream

def plot_data(data, ax, color, as_name, val_max=120, err=False, hatch='+'):
    "Plot data and its linear interpolation"
    if len(data) > 0:
        x, y = zip(*[(a, b) for a, b in data if b < val_max])
        ax.plot(x, y , color + hatch,
                label=': '.join((short(as_name)
                                 + (' err' if err else ''), str(len(data)))))
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        xis = sorted(x)
        ax.plot(xis, map(p, xis), color + (':' if err else ''), lw=2)


def plot_filtered(filtered, ax, as_name, color, hatch='+'):
    "Plot the filtered data on axes"
    # instead of itemgetter: lambda (vol, per, bgp, valid): (vol, per)
#    data_ok, data_err = [[(vol, per) for vol, per, bgp, valid in filtered
#                          if valid == valids]
#                         for valids in ('OK', 'ERROR')]
    # need to hardcode in order to cope with multiple format: bad!
    data_ok, data_err = [[(x[0], x[1]) for x in filtered
                          if x[3] == valids]
                         for valids in ('OK', 'ERROR')]
    plot_data(data_err, ax, color, as_name, err=True, hatch=hatch)
    plot_data(data_ok, ax, color, as_name, hatch=hatch)

def short(as_name):
    """Formats the as name in short form
    """
    as_list = ('DAILYMOTION', 'GOOGLE', 'ALL_YOUTUBE', 'OTHER', 'ALL_GOOGLE')
    short_list = ('DM', 'GOO', 'YT', 'OTH', 'GOO+YT')
    try:
        return dict(zip(as_list, short_list))[as_name]
    except KeyError:
        return as_name

def plot_remaining_download(data_remaining,
                            as_list=('ALL_YOUTUBE', 'GOOGLE', 'DAILYMOTION'),
                            #as_list=('DAILYMOTION', 'ALL_GOOGLE'),
                            plot_excluded=True, prefix='remaining_time',
                            out_dir='rapport/complements',
                            use_host_referer=False, good_indic=None,
                            loglog=True, logx=True, th=None, rescale=True):
    """Plot cdf for each value in dict
    USE WITH
tools.filter_streaming.generate_remaining_cnx_stream
    OR
DEPRECATED generate_remaining_download
    """
#    formatter = dict(zip(as_list + ('OTHER',), ('bx', 'r*', 'g+')))
    colors = dict(zip(('OTHER',) + as_list, ('g', 'b', 'r', 'c')))
    hatches = cycle('xo+')
    as_excluded = reduce(concat,
                         [INDEX_VALUES.__getattribute__('AS_' + as_name)
                          for as_name in as_list])
    if good_indic:
        all_streams = []
    for k, v in data_remaining.iteritems():
        if good_indic:
            all_streams.extend(v)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes([0.105, 0.2, 0.8, 0.7])
        if th:
            v = [x for x in v if x[0] > th]
        # to have other plotted behind
#        filtered = filter(lambda (vol, per, bgp, valid):
#                          bgp not in as_excluded, v)
        if plot_excluded:
            filtered = filter(lambda x: itemgetter(2)(x) not in as_excluded, v)
            if len(filtered) != 0:
                plot_filtered(filtered, ax, 'OTHER', colors['OTHER'],
                              hatch=hatches.next())
        for as_name in as_list:
#            filtered = filter(lambda (vol, per, bgp, valid): bgp in
#                          INDEX_VALUES.__getattribute__('AS_' + as_name), v)
            #assert False
            if not use_host_referer:
                filtered = filter(lambda x: itemgetter(2)(x) in
                              INDEX_VALUES.__getattribute__('AS_' + as_name), v)
            else:
                if 'YOUTUBE' in as_name:
                    host_referer = 'youtube'
                elif 'DAILYMOTION' in as_name:
                    host_referer = 'dailymotion'
                else:
                    print('Warning: Assign youtube host to AS:', as_name)
                    host_referer = 'youtube'
                filtered = filter(lambda x: host_referer in x[4], v)
            corrected = filter(lambda x: x[1] >= 0 and x[1] <= 110, filtered)
            if len(corrected) != 0:
                plot_filtered(corrected, ax, as_name, colors[as_name],
                              hatch=hatches.next())
        ax.set_title('Remaining Volume for ' + streaming_tools.format_title(k))
        ax.set_ylabel('Percentage of Dowloaded Volume')
        ax.set_xlabel('Content Length in Bytes' +
                      ((' filtered on flows > %g' % th) if th else ''))
        ax.grid(True)
        #        ax.legend(loc=(1.03,0.2), prop={'size': 10})
        ax.legend(bbox_to_anchor=(0., -.22, 1., .102), loc=4,
                  ncol=len(as_list) + 1, mode="expand", borderaxespad=0.)
        save_file = os.sep.join((out_dir, '_'.join((prefix, k))))
        if th:
            save_file += '_%g' % th
        if logx:
            ax.semilogx()
            if rescale:
                ax.set_ylim(0, 110)
#            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
            ax.grid(True)
            fig.savefig(save_file + '_logx.pdf', format='pdf')
        if loglog:
            ax.loglog()
            ax.grid(True)
            fig.savefig(save_file + '_loglog.pdf', format='pdf')
        del(fig)
    if good_indic:
        good_streams = [(x[0], x[1]) for x in all_streams if x[5]]
        bad_streams = [(x[0], x[1]) for x in all_streams if not x[5]]
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes([0.105, 0.2, 0.8, 0.7])
        plot_data(good_streams, ax, 'r', 'Good', hatch='+')
        plot_data(bad_streams, ax, 'b', 'Bad', hatch='*')
        ax.set_title('Remaining Volume for Streams on All Traces '
                     + str(good_indic))
        ax.set_ylabel('Percentage of Dowloaded Volume')
        ax.set_xlabel('Content Length in Bytes' +
                      ((' filtered on flows > %g' % th) if th else ''))
        ax.grid(True)
        #        ax.legend(loc=(1.03,0.2), prop={'size': 10})
        ax.legend(bbox_to_anchor=(0., -.22, 1., .102), loc=4,
                  ncol=len(as_list) + 1, mode="expand", borderaxespad=0.)
        save_file = os.sep.join((out_dir,
                                 '_'.join((prefix, 'all',
                                           str(good_indic).replace(' ', '_')))))
        if th:
            save_file += '_%g' % th
        if logx:
            ax.semilogx()
            if rescale:
                ax.set_ylim(0, 110)
#            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
            ax.grid(True)
            fig.savefig(save_file + '_logx.pdf', format='pdf')
        if loglog:
            ax.loglog()
            ax.grid(True)
            fig.savefig(save_file + '_loglog.pdf', format='pdf')
        del(fig)

def get_hhs(cnx_stream, as_field, agg_field,  nb_hh=100):
    "Return a dict of all heavy hitters for all ASes"
    clients_per_vol = streaming_tools.connections_per_as(cnx_stream,
                             as_field=as_field, key_ext='', agg_field=agg_field,
                             already_filtered=True, dir_check=False)
    return streaming_tools.get_top_n(clients_per_vol, nb_top=nb_hh,
                                      exclude_list=UNKNOWN_ID)

def stats_for_hh(flows, flows_1mb, nb_hh=20,
                 as_list=('DAILYMOTION', 'GOOGLE', 'ALL_YOUTUBE')):
    """Print stats for heavy hitters
    Use with data from h5 then separate_flows_dscp
    datas = tools.load_hdf5_data.load_h5_file('flows/hdf5/traces_lzf.h5')
    flows, flows_1mb = tools.streaming_tools.separate_flows_dscp(datas)
    stats, stats_hh = tools.complements.stats_for_hh(flows, flows_1mb)
    print(stats, open('rapport/table_stats_compl.tex', 'w'))
    print(stats_hh,  open('rapport/table_stats_compl_hh.tex', 'w'))
    """
#    clients_per_vol = streaming_tools.connections_per_as(flows, key_ext='',
#                                                         as_field='client_id',
#                                                         already_filtered=True)
#    hh = streaming_tools.get_top_n(clients_per_vol, nb_top=(nb_hh+1),
#                                   exclude_list=UNKNOWN_ID)
    # test if ok
    hh = get_hhs(flows, 'client_id', 'l3Bytes', nb_hh=(nb_hh + 1))
    nb_flows_hh = {}
    nb_clients_hh = {}
    vol_hh = {}
    nb_flows = {}
    nb_clients = {}
    vol = {}
    #    for k in filter(lambda k: any([k.endswith(t) for t in as_list]),
    # easier to manage
    assert flows.keys() == flows_1mb.keys()
    for k in sorted(flows):
        for is_1mb in (False, True):
            if is_1mb:
                flow_type = flows_1mb
                flow_size = '1MB'
            else:
                flow_type = flows
                flow_size = ''
            # only downstream matters
            data = flow_type[k].compress(flow_type[k]['direction'] ==
                                         INDEX_VALUES.DOWN)
            data_hh = data.compress([x['client_id']
                                     in map(itemgetter(0), hh[k])
                                     for x in data])
            for as_name in as_list:
                data_as = data.compress([x['asBGP']
                         in INDEX_VALUES.__getattribute__('AS_' + as_name)
                                         for x in data])
                nb_flows[k + as_name + flow_size] = len(data_as)
                nb_clients[k + as_name + flow_size] = len(np.unique(
                    data_as['client_id']))
                vol[k + as_name + flow_size] = sum(data_as['l3Bytes'])
                data_hh_as = data_hh.compress([x['asBGP']
                           in INDEX_VALUES.__getattribute__('AS_' + as_name)
                                               for x in data_hh])
                nb_flows_hh[k + as_name + flow_size] = len(data_hh_as)
                nb_clients_hh[k + as_name + flow_size] = len(np.unique(
                    data_hh_as['client_id']))
                vol_hh[k + as_name + flow_size] = sum(data_hh_as['l3Bytes'])
    stats = []
    stats_hh = []
    for adsl, ftth in [t for t in zip(islice(sorted(flows), 0, None, 2),
                                      islice(sorted(flows), 1, None, 2))]:
        stats.append('\n\hline\n'.join((
            ' & '.join([' ']
                       + concat(*[[k.split('_', 3)[-1].replace('_', ' ')
                                   + ' ' + short(a) for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['Date'] +
                       concat(*[['/'.join((k.split('_', 3)[0:3]))
                                 for a in as_list]
                                for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['nb flows down in Nb']
                       + concat(*[[str(nb_flows[k + a]) for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['nb flows 1mb down in Nb']
                      + concat(*[[str(nb_flows[k + a + '1MB']) for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['ratio nb flows 1MB/all']
                       + concat(*[['NA' if nb_flows[k + a] == 0 else
                                   '%.3g' %
                                   (nb_flows[k + a + '1MB'] / nb_flows[k + a])
                                   for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['nb clients down in Nb']
                       + concat(*[[str(nb_clients[k + a]) for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['nb clients 1mb down in Nb']
                       + concat(*[[str(nb_clients[k + a + '1MB'])
                                   for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['vol down in Bytes']
                       + concat(*[['%.3g' % vol[k + a] for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['vol down 1mb in Bytes']
                       + concat(*[['%.3g' % vol[k + a + '1MB'] for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['avg nb flows down in Nb']
                       + concat(*[['NA' if nb_clients[k + a] == 0 else
                                  '%.3g' % (nb_flows[k + a] / nb_clients[k + a])
                                   for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['avg nb flows down 1mb in Nb']
                       + concat(*[['NA' if nb_clients[k + a + '1MB'] == 0 else
                                   '%.3g' % (nb_flows[k + a + '1MB']
                                             / nb_clients[k + a + '1MB'])
                                   for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['avg vol down in Bytes']
                       + concat(*[['NA' if nb_clients[k + a] == 0 else
                                   '%.3g' % (vol[k + a] / nb_clients[k + a])
                                   for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['avg vol down 1mb in Bytes']
                       + concat(*[['NA' if nb_clients[k + a + '1MB'] == 0 else
                                   '%.3g' %
                                   (vol[k + a + '1MB']
                                    / nb_clients[k + a + '1MB'])
                                   for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\'))
             + '\n\hline\n')
        # duplicate code :(
        stats_hh.append('\n\hline\n'.join((
            ' & '.join(['%d Heavy Hitters' % nb_hh]
                       + concat(*[[k.split('_', 3)[-1].replace('_', ' ')
                                   + ' ' + short(a) for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['Date'] +
                       concat(*[['/'.join((k.split('_', 3)[0:3]))
                                 for a in as_list]
                                for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['nb flows down in Nb']
                          + concat(*[[str(nb_flows_hh[k + a]) for a in as_list]
                                     for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['nb flows 1mb down in Nb']
                       + concat(*[[str(nb_flows_hh[k + a + '1MB'])
                                   for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['ratio nb flows 1MB/all']
                       + concat(*[['NA' if nb_flows_hh[k + a] == 0 else
                                   '%.3g' %
                                   (nb_flows_hh[k + a + '1MB']
                                    / nb_flows_hh[k + a])
                                   for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['nb clients down in Nb']
                       + concat(*[[str(nb_clients_hh[k + a]) for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['nb clients 1mb down in Nb']
                       + concat(*[[str(nb_clients_hh[k + a + '1MB'])
                                   for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['vol down in Bytes']
                       + concat(*[['%.3g' % vol_hh[k + a] for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['vol down 1mb in Bytes']
                       + concat(*[['%.3g' % vol_hh[k + a + '1MB']
                                   for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['avg nb flows down in Nb']
                       + concat(*[['NA' if nb_clients_hh[k + a] == 0 else
                                   '%.3g' % (nb_flows_hh[k + a]
                                             / nb_clients_hh[k + a])
                                   for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['avg nb flows down 1mb in Nb']
                       + concat(*[['NA' if nb_clients_hh[k + a + '1MB'] == 0
                                   else '%.3g' % (nb_flows_hh[k + a + '1MB']
                                              / nb_clients_hh[k + a + '1MB'])
                                   for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
            ' & '.join(['avg vol down in Bytes']
                       + concat(*[['NA' if nb_clients_hh[k + a] == 0 else
                                   '%.3g' % (vol_hh[k + a]
                                             / nb_clients_hh[k + a])
                                   for a in as_list]
                                  for k in (adsl, ftth)])) + r' \\',
        ' & '.join(['avg vol down 1mb in Bytes']
                   + concat(*[['NA' if nb_clients_hh[k + a + '1MB'] == 0
                               else '%.3g' % (vol_hh[k + a + '1MB'] /
                                              nb_clients_hh[k + a + '1MB'])
                               for a in as_list]
                              for k in (adsl, ftth)])) + r' \\'))
        + '\n\hline\n')
    return '\hline\n'.join(stats), '\hline\n'.join(stats_hh)

def extract_sessions_hh_as(data_hh_as, tmp_dir, gap=600,
                           client_field='client_id'):
    "Return a np array of streaming sessions for data_hh_as"
    with NamedTemporaryFile(prefix='tmp_session_file', suffix='.txt',
                       dir=tmp_dir) as tmp_session_file:
        flow2session.process_cnx_sessions(data_hh_as, tmp_session_file.name,
                                          gap=gap, reset_errors=True,
                                          client_field=client_field)
        return np.loadtxt(tmp_session_file.name,
                                    delimiter=';',
                                dtype=INDEX_VALUES.dtype_cnx_streaming_session)

def aggregate_sessions_hh_as(hh, sessions_hh_as):
    "Return a list of tuple of stats resume on heavy hitters"
    return [(sum(data_ok['duration']), sum(data_ok['tot_bytes']),
             np.mean(8e-3 * data_ok['tot_bytes'] / data_ok['duration']))
            for client in hh[:-1]
            for data in (sessions_hh_as.compress(
                sessions_hh_as['Name'] == str(client)),)
            for data_ok in (data.compress(data['duration'] > 0),)
            if len(data_ok) > 0]

def full_sessions_hh_as(data_hh_as):
    "Return a list of tuple of bytes and average bit-rate for all flows"
    data_ok = data_hh_as.compress(data_hh_as['DurationDn'] > 0)
    return zip(data_ok['DurationDn'], data_ok['ByteDn'],
               8e-3 * data_ok['ByteDn'] / data_ok['DurationDn'])

def extract_sessions_single_hh_as(sessions_hh_as):
    "Return a list of tuple of bytes and average bit-rate for all sessions"
    data_ok = sessions_hh_as.compress(sessions_hh_as['duration'] > 0)
    return zip(data_ok['duration'], data_ok['tot_bytes'],
               8e-3 * data_ok['tot_bytes'] / data_ok['duration'])

def compute_active_time_per_as(nb_hh=100, flows_dir='flows/links/cnx_stream',
                               cnx_stream=None,
#                           as_list=('ALL_YOUTUBE', 'GOOGLE', 'DAILYMOTION'),
                               url_list=('.youtube.', '.dailymotion.'),
                               file_name_start='cnx_stream_', extra_data=False,
                               gap=600):
    """Graph for heavy hitters separating by AS
    USE with plot_active_time_as
    """
    if not cnx_stream:
        cnx_stream = dict([(f.split(file_name_start)[1].split('.npy')[0],
                            np.load(os.sep.join((flows_dir, f))))
                           for f in os.listdir(flows_dir)
                           if f.startswith(file_name_start)])
    hhs = get_hhs(cnx_stream, 'client_id', 'ByteDn', nb_hh=nb_hh)
    out_dict = {}
#    as_excluded = map(lambda x: 'AS' + str(x),
#                             [v for as_name in as_list
#                      for v in INDEX_VALUES.__getattribute__('AS_' + as_name)])
    tmp_dir = os.sep.join((os.getcwd(), 'active_time'))
    for trace, data in cnx_stream.iteritems():
        data = cnx_stream[trace]
        print('data: ', len(data))
        hh = map(itemgetter(0), hhs[trace])
#        for client_id in UNKNOWN_ID:
#            # removing unknown user id
#            try:
#                hh.remove(client_id)
#                print('trace: %s; removed: %s' % (trace, client_id))
#            except ValueError:
#                pass
        data_hh = data.compress([x['client_id'] in hh[:-1] for x in data])
        print('data_hh: ', len(data_hh))
        for as_url in url_list:
            as_name = as_url.strip('.').upper()
            data_hh_as = data_hh.compress(
                [as_url in x['Host_Referer'] for x in data_hh])
            print('data_hh_as: ', len(data_hh_as))
            sessions_hh_as = extract_sessions_hh_as(data_hh_as, tmp_dir,
                                                    gap=gap)
            print('sessions_hh_as: ', len(sessions_hh_as))
            out_dict['_'.join((trace, as_name))] = aggregate_sessions_hh_as(hh,
                                                                sessions_hh_as)
            if extra_data:
                out_dict['_'.join((trace, as_name, 'full'))] = \
                        extract_sessions_single_hh_as(sessions_hh_as)
#                        full_sessions_hh_as(data_hh_as)
        else:
            data_hh_as = data_hh.compress(
                [not(any([url in x['Host_Referer'] for url in url_list]))
                 for x in data_hh])
            sessions_hh_as = extract_sessions_hh_as(data_hh_as, tmp_dir,
                                                    gap=gap)
            out_dict['_'.join((trace, 'OTHERS'))] = aggregate_sessions_hh_as(hh,
                                                                sessions_hh_as)
            if extra_data:
                out_dict['_'.join((trace, 'OTHERS', 'full'))] = \
                        extract_sessions_single_hh_as(sessions_hh_as)
#                        full_sessions_hh_as(data_hh_as)
    return out_dict

def plot_full_active_time_as(act, out_dir='active_time', nb_hh=100, prefix=None,
                             as_list=('DAILYMOTION', 'YOUTUBE'),
                             semilog=False, gap=600, large_th=None):
    """Plots the active time graphs
    Use as
    top100_600 = tools.complements.compute_active_time_per_as(extra_data=True)
    tools.complements.plot_full_active_time_as(top100_600,
        out_dir='active_time_new')
    tools.complements.plot_full_active_time_as(top100_600,
        large_th=5e6, out_dir='active_time_new')
    top100_60 = tools.complements.compute_active_time_per_as(extra_data=True)
    tools.complements.plot_full_active_time_as(top100_60,
        out_dir='active_time_new', gap=60)
    tools.complements.plot_full_active_time_as(top100_60,
        large_th=1e6, out_dir='active_time_new', gap=60)
    """
    if not prefix:
        prefix = 'top%d_per_as' % nb_hh
    for trace in [k.split('_OTHERS_full')[0] for k in act
                  if k.endswith('_OTHERS_full')]:
        for (indic, field, unit) in (('Volume', 1, 'Bytes'),
                                     ('Average bit-rate', 2, 'kb/s')):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_axes([0.12, 0.18, 0.8, 0.75])
            markers = cycle("x+*o")
            colors = cycle("bgrcmyk")
            for as_name in ('OTHERS',) + as_list:
                data = act['_'.join((trace, as_name, 'full'))]
                if large_th:
                    # filter on large sessions (not on flows!)
                    # hard coded index
                    data = [x for x in data if x[1] > large_th]
                if len(data) == 0:
                    print('PROBLEM with %s on %s' % (trace, as_name))
                    continue
                x, y = zip(*(map(itemgetter(0, field), data)))
                cur_color = colors.next()
                ax.plot(x, y, color=cur_color, linestyle='',
                        marker=markers.next(),
                        label='%s: %d' % (as_name.replace('_', ' '), len(data)))
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                if p(max(x)) < 0:
                    x_max = (1 - z[1]) / z[0]
                    print(trace, as_name, 'switch lin x', x_max, p(x_max))
                else:
                    x_max = max(x)
                xis = sorted([xi for xi in x if xi <= x_max])
                ax.plot(xis, map(p, xis),
                        color=cur_color, lw=2)
            ax.legend(bbox_to_anchor=(0., -.22, 1., .102), loc=4,
                      ncol=2, mode="expand", borderaxespad=0.)
            ax.set_title(indic + ' vs. Session Duration for %d Heavy Hitters\n'
                         % nb_hh + trace.replace('_', ' '))
            ax.set_ylabel('Dowloaded %s in %s' % (indic, unit))
            ax.set_xlabel(
                '\n'.join(('Session Duration in Seconds (session: gap %d sec'
                           % gap + ((', threshold: %g B)' % large_th)
                              if large_th else ')'),
                           '1 point per session '
                           + '(constructed on per provider flows)')))
            ax.grid(True)
            save_file = '_'.join((prefix, indic.replace(' ', '_').lower(),
                                  'act_time_full', trace, str(gap),
                                  ('th_%g' % large_th) if large_th else 'all'))
#            fig.savefig(os.sep.join((out_dir, save_file + '_lin.pdf')))
            ax.loglog()
            fig.savefig(os.sep.join((out_dir, save_file + '_loglog.pdf')))
            if semilog:
                ax.semilogx()
                fig.savefig(os.sep.join((out_dir, save_file + '_logx.pdf')),
                            format='pdf')
                ax.semilogy()
                fig.savefig(os.sep.join((out_dir, save_file + '_logy.pdf')),
                            format='pdf')

def plot_active_time_as(act, out_dir='active_time', nb_hh=100, prefix=None,
                        as_list=('ALL_YOUTUBE', 'DAILYMOTION'), semilog=False):
    """Plots the active time graphs
    Use as
    top100 = tools.complements.compute_active_time_per_as()
    tools.complements.plot_active_time_as(top100)
    """
    if not prefix:
        prefix = 'top%d_per_as' % nb_hh
    for trace in [k.split('_OTHERS')[0] for k in act if k.endswith('_OTHERS')]:
        markers = cycle("x+*o")
        colors = cycle("bgrcmyk")
        for (indic, field, unit) in (('Volume', 1, 'Bytes'),
                                     ('Average bit-rate', 2, 'kb/s')):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_axes([0.12, 0.18, 0.8, 0.75])
            for as_name in as_list + ('OTHERS',):
                data = act['_'.join((trace, as_name))]
                x, y = zip(*(map(itemgetter(0, field), data)))
                color = colors.next()
                ax.plot(x, y, color=color, linestyle='',
                        marker=markers.next(),
                        label='%s: %d' % (as_name.replace('_', ' '), len(data)))
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                if p(max(x)) < 0:
                    x_max = (1 - z[1]) / z[0]
                    print(trace, as_name, 'switch lin x', x_max, p(x_max))
                else:
                    x_max = max(x)
                xis = sorted([xi for xi in x if xi <= x_max])
                ax.plot(xis, map(p, xis),
                        color=color)
            ax.legend(bbox_to_anchor=(0., -.22, 1., .102), loc=4,
                      ncol=2, mode="expand", borderaxespad=0.)
            ax.set_title(indic + ' vs. Active Time for %d Heavy Hitters '
                         % nb_hh + streaming_tools.format_title(trace))
            ax.set_ylabel('Dowloaded %s in %s' % (indic, unit))
            ax.set_xlabel('Active Time (sum of all sessions) in Seconds')
            ax.grid(True)
            save_file = '_'.join((prefix, indic.replace(' ', '_').lower(),
                                  'act_time', 'lin', trace))
            fig.savefig(os.sep.join((out_dir, save_file + '.pdf')))
            ax.loglog()
            save_file = '_'.join((prefix, indic.replace(' ', '_').lower(),
                                  'act_time', 'loglog', trace))
            fig.savefig(os.sep.join((out_dir, save_file + '.pdf')))
            if semilog:
                ax.semilogx()
                save_file = '_'.join((prefix, indic.replace(' ', '_').lower(),
                                      'act_time', 'logx', trace))
                fig.savefig(os.sep.join((out_dir, save_file + '.pdf')),
                            format='pdf')
                ax.semilogy()
                save_file = '_'.join((prefix, indic.replace(' ', '_').lower(),
                                      'act_time', 'logy', trace))
                fig.savefig(os.sep.join((out_dir, save_file + '.pdf')),
                            format='pdf')
            del(fig)

def compute_active_time(nb_hh=20, sessions_dir='flows/links/cnx_stream',
                        file_name_start='sessions_cnx_stream_'):
    """Compute some nice graphs for Heavy Hitters
    use file_name_start='sessions_10min_cnx_stream_' for other sessions
    see plot_active_time
    """
    cnx_stream = dict([(f.split(file_name_start)[1].split('.txt')[0],
                        np.loadtxt('flows/links/cnx_stream/' + f, delimiter=';',
                               dtype=INDEX_VALUES.dtype_cnx_streaming_session))
                       for f in os.listdir(sessions_dir)
                       if f.startswith(file_name_start)])
    hh = get_hhs(cnx_stream, 'client_id', 'tot_bytes', nb_hh=nb_hh)
    return dict([(k, [(sum(data_ok['duration']), sum(data_ok['tot_bytes']),
                       np.mean(8e-3 * data_ok['tot_bytes']/data_ok['duration']))
                      for client in map(itemgetter(0), hh[k][:-1])
                      for data in (v.compress(v['client_id'] == client),)
                      for data_ok in (data.compress(data['duration']>0),)]
                  + [(sum(data_ok['duration']), sum(data_ok['tot_bytes']),
                      np.mean(8e-3 * data_ok['tot_bytes']/data_ok['duration']))
                     for data in (v.compress([
                         x['client_id'] not in map(itemgetter(0), hh[k][:-1])
                         for x in v]),)
                     for data_ok in (data.compress(data['duration']>0),)])
                 for k, v in cnx_stream.iteritems()])

def plot_active_time(act, out_dir='active_time', prefix='top20', semilog=False):
    """Plots the active time graphs
    Use as
    top20 = tools.complements.compute_active_time()
    tools.complements.plot_active_time(top20)
    top100 = tools.complements.compute_active_time(nb_hh=100)
    tools.complements.plot_active_time(top100, prefix='top100')
    """
    for k, v in act.iteritems():
        for (indic, field, unit) in (('Volume', 1, 'Bytes'),
                                     ('Average bit-rate', 2, 'kb/s')):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_axes([0.12, 0.18, 0.8, 0.75])
            x, y = zip(*(map(itemgetter(0, field), v)))
            ax.plot(x[:-1], y[:-1], '+', label='Heavy Hitters')
            ax.plot(x[-1], y[-1], '*', label='All Others')
            z = np.polyfit(x[:-1], y[:-1], 1)
            p = np.poly1d(z)
            if p(max(x[:-1])) < 0:
                x_max = (1 - z[1]) / z[0]
                print(k, 'switch lin x', x_max, p(x_max))
            else:
                x_max = max(x[:-1])
            xis = sorted([xi for xi in x[:-1] if xi <= x_max])
            ax.plot(xis, map(p, xis))
            ax.legend(bbox_to_anchor=(0., -.22, 1., .102), loc=4,
                      ncol=3, mode="expand", borderaxespad=0.)
            ax.set_title(indic + ' vs. Active Time for %d Heavy Hitters '
                         % (len(v) - 1) + streaming_tools.format_title(k))
            ax.set_ylabel('Dowloaded %s in %s' % (indic, unit))
            ax.set_xlabel('Active Time (sum of all sessions) in Seconds')
            ax.grid(True)
            ax.loglog()
#            plt.setp(ax.get_xticklabels() + ax.get_yticklabels())
            save_file = '_'.join((prefix, indic.replace(' ', '_').lower(),
                                  'act_time', 'loglog', k))
            fig.savefig(os.sep.join((out_dir, save_file + '.pdf')))
            if semilog:
                ax.semilogx()
                save_file = '_'.join((prefix, indic.replace(' ', '_').lower(),
                                      'act_time', 'logx', k))
                fig.savefig(os.sep.join((out_dir, save_file + '.pdf')))
                ax.semilogy()
                save_file = '_'.join((prefix, indic.replace(' ', '_').lower(),
                                      'act_time', 'logy', k))
                fig.savefig(os.sep.join((out_dir, save_file + '.pdf')))
            del(fig)

def nb_flows_per_client_color(flows_stats, heavy_hitters, bin_size=30,
                              duration_indic='Session-Duration',
                              start_indic='StartDn'):
                              #duration_indic='DurationDn'):
    """Return the data representing the number of flows per bin for the heavy
    hitters
    """
    assert (flows_stats.dtype == INDEX_VALUES.dtype_cnx_stream
            or flows_stats.dtype == INDEX_VALUES.dtype_cnx_stream_loss
            or flows_stats.dtype ==
            INDEX_VALUES.dtype_all_stream_indics_final_tstat), \
            "Incorrect dtype"
    nb_flows = {}
    nb_vols = {}
    for client in heavy_hitters:
        nb_flows[client] = defaultdict(int)
        nb_vols[client] = defaultdict(int)
        for flow in flows_stats.compress(flows_stats['client_id']==client):
            flow_bins = range(int(flow[start_indic]) // bin_size,
                           1 + (int(flow[start_indic] + flow[duration_indic])
                                // bin_size))
            # if vol is zero, the accumulation process will remove it and
            # mismatch volume bins with nb_flows bins, so I trick it
            # random because of bad luck of 2 flows having exactly the same
            # total volume than the next flow in the bin (incredible but true)
            vol_per_bin = (random() + max(1, flow['ByteDn'])) / len(flow_bins)
            for i in flow_bins:
                nb_flows[client][i] += 1
                nb_vols[client][i] += vol_per_bin
    return nb_flows, nb_vols

def plot_all_nb_flows(cnx_stream, nb_hh=20, out_dir='nb_flows_plots',
                      postfix='', duration=86400, bin_size=30, hour_graph=False,
                      color=False, thresholds=(1e6, 10e6),
                      start_indic='StartDn'):
    """Wrapper for plot_nb_flows_per_client_acc on a flows dict
    Use as:
cnx_stream = tools.streaming_tools.load_cnx_stream()
for b in (60, 30, 10, 1):
    th = (1e6 * b / 30, 1e7 * b / 30)
    tools.complements.plot_all_nb_flows(cnx_stream, bin_size=b, color=True,
        hour_graph=True, thresholds=th, out_dir='nb_flows_plots/all_flows_color')
    Filter on large flows with:
cnx_stream_1mb = dict([(k, v.compress(v['nByte'] > 1e6))
    for k, v in cnx_stream.iteritems()])
for b in (60, 30, 10, 1):
    th = (1e6 * b / 30, 1e7 * b / 30)
    tools.complements.plot_all_nb_flows(cnx_stream_1mb, bin_size=b,
        color=True, thresholds=th,
        postfix='_1mb', out_dir='nb_flows_plots/large_flows_color')
    """
    # client_id instead of Name
    clients_per_vol = streaming_tools.connections_per_as(cnx_stream,
                             as_field='client_id', key_ext='',
                             agg_field='ByteDn', already_filtered=True,
                             dir_check=False)
    hh = streaming_tools.get_top_n(clients_per_vol, nb_top=nb_hh, rank='nb',
                                   exclude_list=UNKNOWN_ID)
    for trace, data in cnx_stream.iteritems():
        print(' '.join(("Processing trace:", trace)))
        heavy_hitters = map(itemgetter(0), hh[trace])
#        heavy_hitters = map(itemgetter(0), sorted(
#                                aggregate.aggregate_sum(data, 'Name', 'nByte'),
#                                   key=itemgetter(1), reverse=True)[0:nb_hh])
        if color:
            all_nb_flows, all_nb_vol = nb_flows_per_client_color(
                data.compress([x['client_id'] in heavy_hitters for x in data]),
                heavy_hitters, bin_size=bin_size, start_indic=start_indic)
            accumulated_vol = accumulate_flows_nb(all_nb_vol,
                                                  duration=duration,
                                                  bin_size=bin_size)
            tmp_accumulated_nb = accumulate_flows_nb(all_nb_flows,
                                                     duration=duration,
                                                     bin_size=bin_size)
            accumulated_nb = split_nb_flows_according_vols(tmp_accumulated_nb,
                                                           accumulated_vol)
            del(tmp_accumulated_nb)
            del(all_nb_flows)
            del(all_nb_vol)
        else:
            all_nb_flows = nb_flows_per_client(
                data.compress([x['client_id'] in heavy_hitters for x in data]),
                heavy_hitters, bin_size=bin_size, start_indic=start_indic)
            accumulated_nb = accumulate_flows_nb(all_nb_flows,
                                                 duration=duration,
                                                 bin_size=bin_size)
            del(all_nb_flows)
        if hour_graph:
            #x_min = INDEX_VALUES.TIME_START[trace]
            x_min = min(map(itemgetter(1), map(itemgetter(0),
                                               accumulated_nb.values())))
            x_max = max(sum(map(itemgetter(1), xs)) for xs
                        in [ys[:-1] for ys in accumulated_nb.values()])
        else:
            x_min = 0
        fig = plt.figure()
        if color:
            ax = fig.add_axes([0.125, 0.16, 0.81, 0.7])
        else:
            ax = fig.add_subplot(111)
        plot_nb_flows_per_client_acc((accumulated_nb, heavy_hitters), ax,
                                     bin_size=bin_size,
                                     duration=((x_max - x_min) if hour_graph
                                               else duration),
                                     trace_name=trace, color=color,
                                     x_min=x_min, thresholds=thresholds,
                                 all_nb_vols=accumulated_vol if color else None,
                                 title='Number of %s Flows for %d Heavy Hitters'
                                 % (postfix.replace('_', ' ').upper(), nb_hh))
        fig.savefig(os.sep.join((out_dir, 'nb_flows_%d_HH_%s_%s%s.pdf'
                                 % (nb_hh, trace, bin_size, postfix))))
        fig.clf()
        del(ax)
        del(fig)
        fig = plt.figure(8, 8)
        #ax = fig.add_subplot(111)
        plot_nb_flows_per_client_line((accumulated_nb, heavy_hitters), fig,
                                      bin_size=bin_size,
                                      duration=((x_max - x_min) if hour_graph
                                                else duration),
                                      x_min=x_min, trace_name=trace,
                                 title='Number of %s Flows for %d Heavy Hitters'
                                  % (postfix.replace('_', ' ').upper(), nb_hh))
        fig.savefig(os.sep.join((out_dir, 'nb_flows_line_3d_%d_HH_%s_%s_%s.pdf'
                                 % (nb_hh, trace, postfix, bin_size))))
        fig.clf()
        del(fig)

def split_nb_flows_according_vols(clients_nb_flows, clients_nb_vols):
    """Return corrected version of nb_flows, because vol accumulation is more
    restrictive than nb_flows one
    """
    new_clients_nb_flows = {}
    for client in clients_nb_vols.keys():
        nb_vols = clients_nb_vols[client]
        nb_flows = clients_nb_flows[client]
        assert len(nb_vols) >= len(nb_flows), \
                "nb_flows more restrictive than vols"
        new_nb_flows = []
        vol_index = 0
        for (nb_flow, n_width) in nb_flows:
            v_width = nb_vols[vol_index][WIDTH_IDX]
            if v_width == n_width:
                new_nb_flows.append((nb_flow, n_width))
                vol_index += 1
            else:
                assert n_width > v_width, \
                        "Problem in width for client %s" % client
                added_width = 0
                # hack due to float representation
                while (n_width - added_width > EPSILON):
                    v_width = nb_vols[vol_index][WIDTH_IDX]
                    new_nb_flows.append((nb_flow, v_width))
                    added_width += v_width
                    vol_index += 1
        assert vol_index == len(nb_vols), "Not all volumes processed"
        new_clients_nb_flows[client] = new_nb_flows
    return new_clients_nb_flows

def nb_flows_per_client(flows_stats, heavy_hitters, bin_size=30,
                        start_indic='StartDn', duration_indic='DurationDn'):
    """Return the data representing the number of flows per bin for the heavy
    hitters
    """
    assert flows_stats.dtype == INDEX_VALUES.dtype_cnx_stream \
            or flows_stats.dtype == INDEX_VALUES.dtype_cnx_stream_loss, \
            "Incorrect dtype"
    nb_flows = {}
    for client in heavy_hitters:
        nb_flows[client] = defaultdict(int)
        for flow in flows_stats.compress(flows_stats['client_id']==client):
            for i in range(flow[start_indic] // bin_size,
                           1 + ((flow[start_indic] + flow[duration_indic])
                                // bin_size)):
                nb_flows[client][i] += 1
    return nb_flows

def accumulate_flows_nb(client_nb_flows, duration=86400, bin_size=30):
    "Return a new dict of nb_flows to aggregate the bins with same values"
    new_nb_flows = {}
    for client, nb_flows in client_nb_flows.iteritems():
        nb_flows_widths = [(0, 0)]
        for i in xrange(duration // bin_size):
            if nb_flows[i] == nb_flows_widths[-1][0]:
                nb, width = nb_flows_widths[-1]
                nb_flows_widths[-1] = (nb, width + bin_size / 60)
            else:
                nb_flows_widths.append((nb_flows[i], bin_size / 60))
        new_nb_flows[client] = nb_flows_widths
    return new_nb_flows

def plot_nb_flows_per_client_acc((clients_nb_flows_widths, heavy_hitters), ax,
                                 color=False, all_nb_vols=None,
                                 x_min=0, thresholds=(1e6, 10e6),
                                 duration=86400, bin_size=30, trace_name=None,
                                 title='Number of flows per client'):
    """Plot a graph representing the number of sessions out of flows stats
    Compress data with accumulate_flows_nb before
    """
    y_offset = 0
    height = 1
    if color:
        assert len(thresholds) == 2, "Incorrect thresholds list"
        (min_vol, max_vol) = thresholds
    for client in heavy_hitters:
        nb_flows_width = clients_nb_flows_widths[client]
        if color:
            vols_width = all_nb_vols[client]
            assert len(nb_flows_width) == len(vols_width), \
                    "Mismatch between volumes and flows"
        max_nb_flows = max(map(itemgetter(0), nb_flows_width))
        if max_nb_flows == 0:
            print('Problem with client: ', client)
            continue
        cur_height = height * max_nb_flows
        bin_start = 0
        for index, (nb, width) in enumerate(nb_flows_width):
            if color:
                vol, width_v = vols_width[index]
                assert width == width_v, "Mismatch between vol and nb widths"
                if vol < min_vol:
                    flow_color = 'g'
                elif vol < max_vol:
                    flow_color = ORANGE
                else:
                    flow_color = 'r'
            else:
                flow_color = 'k'
            rect = mpatches.Rectangle((bin_start, y_offset),
                                      width, cur_height,
                                      linewidth=0, color=flow_color,
                                      alpha=nb / max_nb_flows)
            ax.add_patch(rect)
            bin_start += width
#            ax.text(duration/60, y_offset, '$%s$ ' % client + str(max_nb_flows)
#                    size=8)
        del rect
        y_offset += cur_height + height
    ax.set_xlim([x_min / 60, (x_min + duration) / 60])
    #ax.set_xlim([0, duration/60])
    ax.set_ylim([0, y_offset])
    ax.set_ylabel('''Client (largest volume at bottom)
height: maximum number of parallel flows in a bin''')
    ax.set_xlabel('\n'.join((
        'Time in Minutes (bin size of %d seconds)' % bin_size,
        'Bin color: green ($x < %g$), orange ($%g < x < %g$), red ($x > %g$)'
        % (min_vol, min_vol, max_vol, max_vol) if color else '')))
#    ax.yaxis.set_major_locator(NullLocator())
    if trace_name:
        ax.set_title(': '.join((title, streaming_tools.format_title(
                                trace_name.split('sessions_GVB_')[1]
        if 'sessions_GVB_' in trace_name else trace_name))))

def plot_nb_flows_per_client_line((clients_nb_flows_width, heavy_hitters), fig,
                             duration=86400, bin_size=30, trace_name=None,
                             x_min=0, title='Number of flows per client'):
    """Plot a graph representing the number of sessions out of flows stats
    here we plot a line for each heavy hitter with a diffrent color
    Compress data with accumulate_flows_nb before
    fou = '1579_00:23:48:12:c2:6a'
    """
    ax = Axes3D(fig, azim=-55, elev=35)
    colors = cycle("bgrcmy")
    cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)
    markers = cycle("x+*o")
    y_offset = 0
    zs = []
    verts = []
    face_colors = []
    x_lim_min, x_lim_max = x_min, x_min + duration
    for i, client in enumerate(heavy_hitters):
        zs.append(2 * i)
        x = []
        y = []
        nb_flows_width = clients_nb_flows_width[client]
        cur_time = 0
        for (nb, width) in nb_flows_width:
            if cur_time != 0:
                x.append(cur_time - x_min)
                y.append(nb)
            cur_time += width
            x.append(cur_time - x_min)
            y.append(nb)
        #ax.plot(x, y, label='HH nb: %d' % i,
        #y[0], y[-1] = 0, 0
        if y[-1] == 0:
            y.pop()
            x.pop()
            #x_lim_max -= width
        verts.append(zip(x, y))
        face_colors.append(cc(colors.next()))
        #ax.plot(x, y, label='HH nb: %s' % client,
                #color=colors.next(), marker=markers.next())
        y_offset = max(max(map(itemgetter(0), nb_flows_width)), y_offset)
    poly = PolyCollection(verts, facecolors=face_colors, linewidth=1)
    poly.set_alpha(0.6)
    ax.add_collection3d(poly, zs=zs, zdir='y')
    #ax.set_xlim3d(x_lim_min, x_lim_max)
    ax.set_xlim3d(0, duration)
    ax.set_zlim3d(0, y_offset + 1)
    ax.set_ylim3d(-1, 2 * len(heavy_hitters) + 1)
    #ax.grid(True)
    #ax.legend()
    ax.set_zlabel('Number of parallel flows in a bin')
    ax.set_xlabel('Time in Minutes\n(bin size of %d seconds)' % bin_size)
    ax.set_ylabel('Clients')
    # make ticklabels and ticklines invisible
    for a in ax.w_yaxis.get_ticklabels(): # ax.w_yaxis.get_ticklines() +
        a.set_visible(False)
        #ax.yaxis.set_major_locator(NullLocator())
    if trace_name:
        #ax.set_title
        fig.suptitle(': '.join((title, streaming_tools.format_title(
                                trace_name.split('sessions_GVB_')[1]
        if 'sessions_GVB_' in trace_name else trace_name))))
    del(ax)


