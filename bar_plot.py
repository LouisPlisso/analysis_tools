"Module to create bar graphs"

from __future__ import division, print_function
import numpy as np
import pylab
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from itertools import izip
from collections import defaultdict

import INDEX_VALUES
# import aggregate
from streaming_tools import format_title
import complements

#def plot_bar(data, output_path='rapport/bar_graphs/'):
#    """Plots bar plot of loss rates from datasets.
#    """
#    pass



def construct_lambda_comp(type_index, types, thresholds):
    "Returns a function to tell if a value is in the range of thresholds"
    i = types.index(type_index)
    if i == 0:
        lower_bound = np.finfo(np.float).min
    else:
        lower_bound = thresholds[i - 1]
    upper_bound = thresholds[i]
    assert lower_bound < upper_bound, "Problem in loss table definition"
    return lambda x, y: (y > 0) and (lower_bound < float(x) / y <= upper_bound)


def construct_loss_data(flows, dipcp_loss_field, fields, loss_functions,
        percent=False, min_size=0):
    """Returns 2 dict:
        one containing loss values interval for the flows
        one containing the corresponding names
        """
    traces = set(key.strip('_DIPCP').strip('_GVB') for key in flows)
    loss_data = defaultdict(list)
    names = defaultdict(list)
    # data collection
    for name, as_number in fields:
        if as_number == None:
            as_number = INDEX_VALUES.Everything()
        for trace in sorted(traces):
            names[name].append(trace)
            for loss_interval, loss_func in loss_functions.items():
                nb_flows_in_interval = len([x for x in flows[trace]
                    if loss_func(x[dipcp_loss_field],
                        x['DIP-Volume-Number-Packets-Down'])
                    and x['DIP-Volume-Sum-Bytes-Down'] >= min_size
                    and x['asBGP'] in as_number])
                if percent:
                    tot_nb_flows = len([x for x in flows[trace]
                        if x['DIP-Volume-Number-Packets-Down'] > 0
                            and x['DIP-Volume-Sum-Bytes-Down'] >= min_size
                            and x['asBGP'] in as_number]) + 0.
                    result = (nb_flows_in_interval / tot_nb_flows
                              if (tot_nb_flows!=0) else 0)
                else:
                    result = nb_flows_in_interval
                loss_data[(name, loss_interval)].append(result)
#                print trace, field, loss_interval, result
#                print (field, loss_interval), loss_data[(field, loss_interval)]
    return loss_data , names


def plot_loss_bar(flows, fields, percent=False, loss_type='DSQ',
        output_path='rapport/test_bar_graph', min_size=0):
    """Plots bar plot of loss rates from datasets.
    Use as:
datas = tools.load_hdf5_data.load_h5_file('flows/hdf5/traces_lzf.h5')
# hack for getting loss data from streaming_tools
dipcp_streaming = tools.load_dipcp_file.filter_dipcp_dict(datas,
    field='dscp', values=(tools.INDEX_VALUES.DSCP_HTTP_STREAM,))
# or with pickle:
dipcp_streaming = cPickle.load(open('dipcp_streaming_loss_AS.pickle'))
tools.bar_plot.plot_loss_bar(dipcp_streaming, (('HTTP_STREAM_DOWN', None), ))
tools.bar_plot.plot_loss_bar(dipcp_streaming, (('HTTP_STREAM_DOWN', None),
                        ('YT_EU_STREAM_DOWN', (43515, )),
                        ('YT_STREAM_DOWN', (36561, 1901))))
tools.bar_plot.plot_loss_bar(dipcp_streaming, (
                        ('DAILY_M_STREAM_DOWN', (41690,)),
                        ('DEEZER', (39605,)),
                        ('YT_EU_STREAM_DOWN', (43515, )),
                        ('YT_STREAM_DOWN', (36561, 1901))))
#    tools.bar_plot.plot_loss_bar(dipcp_streaming, ('HTTP_STREAM_DOWN',
#                            'YT_EU_STREAM_DOWN', 'YT_STREAM_DOWN'),
#                            percent=True)
# tools.bar_plot.plot_loss_bar(dipcp_streaming,
# ('YT_EU_STREAM_DOWN', 'YT_STREAM_DOWN'))
# tools.bar_plot.plot_loss_bar(dipcp_streaming, ('YT_EU_STREAM_DOWN',
#                            'YT_STREAM_DOWN'), percent=True)
"""
    traces = sorted(set(key.strip('_DIPCP').strip('_GVB') for key in flows))

    dipcp_loss_field = 'DIP-%s-NbMes-sec-TCP-Down' % loss_type

    loss_types = ['no', 'low', 'medium', 'high', 'pb']
    loss_threshold = [0, .01, .05, .1, 1]
#    dict_loss_th = dict(izip(loss_types, loss_threshold))
    loss_functions =  dict(izip(loss_types, (construct_lambda_comp(index,
        loss_types, loss_threshold) for index in loss_types)))

    loss_data, names = construct_loss_data(flows, dipcp_loss_field, fields,
            loss_functions, percent=percent, min_size=min_size)

    pylab.clf()
    shift = 0
    colors = dict(izip(loss_types, ['w']*4+['k']))
    hatch_type = ['**', '//', 'xx', 'OO', '*'] #'/', '+', '|', '-', 'o', 'O']
    assert len(loss_types) == len(hatch_type)
    hatches = dict(izip(loss_types, hatch_type))

    # plotting
    nb_graphs = len(traces)
    x_values = np.arange(2*nb_graphs, step=2)
    # width of bars: can also be len(x) sequence
    width = 0.35

    # first find out scale
    max_scale = max([sum([loss_data[(fields[-1][0], loss_interval)][n]
        for loss_interval in loss_types]) for n in range(nb_graphs)])

    # enumerate to handle duplicate fields
    for field_number, (name, as_number) in enumerate(fields):
        plot = {}
        bottom = pylab.zeros(nb_graphs)
        for i, loss_interval in enumerate(loss_types):
#            hatch = hatches[i % hatches_len]
            hatch = hatches[loss_interval]
            # duplicate fields ok with enumerate
            plot[(field_number, loss_interval)] = \
                pylab.bar(x_values+shift, loss_data[(name, loss_interval)],
                          width, color=colors[loss_interval], hatch=hatch,
                          bottom=bottom, label=name, lw=1, antialiased=True)
            bottom = [sum(t) for t in
                      izip(bottom, loss_data[(name, loss_interval)])]
        for graph_absciss, graph_nb, trace in \
                izip(x_values+shift+width/2., range(nb_graphs), traces):
#            if 'dipcp_%s_%s' % (trace, field) in flows.keys():
            if trace in flows.keys():
#                tmp = flows['dipcp_%s_%s' % (trace, field)].compress(
#                        flows['dipcp_%s_%s' % (trace, field)][
                tmp = flows[trace].compress(flows[trace][
                        'DIP-Volume-Number-Packets-Down'] > 0)
                if as_number == None:
                    as_number = INDEX_VALUES.Everything()
                tmp = tmp.compress([data['asBGP'] in as_number for data in tmp])
                total = len(tmp.compress(tmp['DIP-Volume-Sum-Bytes-Down']
                    >= min_size))
            else:
                total = 0
#            print trace, name, total
            # hard coded name split as '_STREAM'
            pylab.text(graph_absciss, max_scale / 40 +
                       sum([loss_data[(name, loss_interval)][graph_nb]
                            for loss_interval in loss_types]),
                       name.split('_STREAM')[0] + ": %d" % total,
                       rotation=90, fontsize=8)
        shift += .5
    save_title = ''
    if percent:
        retitle = ' in percent'
        save_title = '_' + retitle.replace(' ', '_')
        loc = 3
    else:
        retitle = ''
        loc = 2
    pylab.grid(True)
    pylab.ylabel('Nb of Flows %s' % retitle)
    my_title = "Nb of flows%s by loss rate per traffic capture \
with min flow size: %d Bytes" % (retitle, min_size)
    axes = pylab.gca()
    pylab.text(0.0, -0.1, my_title, size=12,
            transform = axes.transAxes)
    pylab.xticks(x_values+shift/2., map(format_title, traces), size=8)
    legend_data = []
    legend_name = []
    for loss_interval in loss_types:
        # take last plot as ref
        legend_data.append(plot[(len(fields) - 1, loss_interval)][0])
        legend_name.append(loss_interval)
    font = FontProperties(size = 'x-small')
    pylab.legend( legend_data, legend_name, loc=loc, prop=font)
    pylab.savefig('%s/loss%s_%s_%s_min_%s.pdf' % (output_path, save_title,
        '_'.join(name for name, _ in fields), loss_type, min_size),
                  format='pdf')
#    pylab.show()



def pie_graph(flows, fields):
    """Plots pie charts of loss rates from datasets.
    data = tools.load_hdf5_data.load_h5_file('hdf5/lzf_streaming_yt.h5',
    fullpath=False)
    Use as pie_graph(data, ('HTTP_STREAM',))
    Trace list and indicators are hardcoded.
"""
    traces = set(key.strip('_DIPCP').strip('_GVB') for key in flows)
    loss_type = 'DSQ',
    dipcp_loss_field = 'DIP-%s-NbMes-sec-TCP-Down' % loss_type
    loss_types = ['no', 'low', 'medium', 'high', 'pb']
    loss_threshold = [0, .01, .05, .1, 1]
#    dict_loss_th = dict(izip(loss_types, loss_threshold))
    loss =  dict(izip(loss_types, (construct_lambda_comp(index, loss_types,
        loss_threshold) for index in loss_types)))
    loss_data, names = construct_loss_data(flows, dipcp_loss_field, fields,
            loss)
    fig = pylab.figure()
    fig.clf()
#    shift = 0
#    colors = dict(izip(loss_types, ['w']*4+['k']))
#    hatches = ['/', '\\', 'x', '.', '*', '+', '|', '-', 'o', 'O']
#    hatches_len = len(hatches)
    # data collection
    for field in fields:
        for loss_type in loss.keys():
            loss_data[(field, loss_type)] = []
        names[field] = []
        for trace in traces:
            names[field].append(trace)
            for loss_type, loss_func in izip(loss.keys(), loss.values()):
                if 'dipcp_%s_%s' % (trace, field) not in flows:
                    loss_data[(field, loss_type)].append(0)
                    continue
                loss_data[(field, loss_type)].append( \
                    len([x for x in flows['dipcp_%s_%s' % (trace, field)]
                         if loss_func(x[dipcp_loss_field],
                                      x['DIP-Volume-Number-Packets-Down'])]))
    # plotting
    # enumerate to handle duplicate fields
    for field_nb, field in enumerate(fields):
        for trace_nb in range(len(traces)):
            axes = fig.add_subplot(len(fields), len(traces),
                    field_nb*len(traces) + trace_nb + 1)
            axes.pie([loss_data[(field, loss_type)][trace_nb]
                for loss_type in loss_types],
                labels=loss_types, autopct='%1.f%%')
            pylab.title('%s\n%s' % (names[field][trace_nb], field),
                    size='small')
    pylab.show()

def construct_lambda_percent(type_index, types, thresholds):
    "Returns a function to tell if a value is in the range of thresholds"
    i = types.index(type_index)
    if i == 0:
        lower_bound = 0
    else:
        lower_bound = thresholds[i - 1]
    upper_bound = thresholds[i]
    assert lower_bound < upper_bound, "Problem in loss table definition"
    return lambda x: lower_bound < x <= upper_bound


def new_bar_plot(data_remaining, percent=False,
                 as_list=('DAILYMOTION', 'ALL_YOUTUBE', 'GOOGLE'),
                 output_path='rapport/complements/mix_bar_graph', min_size=0):
    """Plots bar plot of percentage of viewing time separated by as
    Use as:
cnx_stream = tools.streaming_tools.load_cnx_stream()
stream_qual = tools.complements.load_stream_qual()
data_remaining = tools.complements.generate_remaining_download_cnx_stream(
    cnx_stream, stream_qual, strict=True)
tools.bar_plot.new_bar_plot(data_remaining)
tools.bar_plot.new_bar_plot(data_remaining, min_size=1e6)
    """
    percent_types = ['low', 'medium', 'high', 'full', 'pb']
    percent_threshold = [10, 50, 95, 110, 1e6]
    percent_functions =  dict(izip(percent_types,
                               (construct_lambda_percent(index, percent_types,
                                                         percent_threshold)
                                for index in percent_types)))
#    loss_data, names = construct_loss_data(flows, dipcp_loss_field, fields,
#            loss_functions, percent=percent, min_size=min_size)
#    shift = 0
    colors = dict(izip(percent_types, 'rmgwk'))
    hatch_type = ['**', '//', 'xx', 'OO', '*'] #, '/', '+', '|', '-', 'o', 'O']
    assert len(percent_types) == len(hatch_type) == len(colors)
    hatches = dict(izip(percent_types, hatch_type))
#    traces = sorted(data_remaining)
#    nb_graphs = len(traces)
    data = complements.construct_bar_data(data_remaining, min_size,
                                          percent_functions, as_list=as_list)
    # plotting
    fig = plt.figure(figsize=(8,8))
#    ax = fig.add_subplot(111)
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.75])
    # width of bars: can also be len(x) sequence
    width = 1
#    x_values = np.arange(2*nb_graphs, step=2)
    x_loc_titles = []
    x_locations = [1]
    labels = None
    for (trace, data_trace) in sorted(data.items()):
        plot = {}
        for as_name in as_list:
            bottom = 0 #np.zeros(nb_graphs)
            tot_nb_flows_as = sum(data_trace[as_name].values())
            for percent_type in reversed(percent_types):
                if not percent_type in data_trace[as_name]:
                    continue
                value = (data_trace[as_name][percent_type] if (not percent)
                     else data_trace[as_name][percent_type] / tot_nb_flows_as)
                hatch = hatches[percent_type]
                if not labels:
                    plot[percent_type] = ax.bar(x_locations[-1], value, width,
                                                color=colors[percent_type],
                                                hatch=hatches[percent_type],
                                                bottom=bottom,
                                                label=percent_type,
                                                lw=1, antialiased=True)
                else:
                    plot[percent_type] = ax.bar(x_locations[-1], value, width,
                                                color=colors[percent_type],
                                                hatch=hatches[percent_type],
                                                bottom=bottom,
                                                lw=1, antialiased=True)
                bottom += value
            # add as_name on bar
            ax.text(x_locations[-1], bottom + (5 if not percent else .001),
                    '%s: %d' % (complements.short(as_name), tot_nb_flows_as),
                    rotation=40, fontsize='small')
#                [sum(t) for t in
#                          izip(bottom, loss_data[(name, loss_interval)])]
            labels = True
            x_locations.append(x_locations[-1] + width)
        x_loc_titles.append((x_locations[-1], trace))
        x_locations.append(x_locations[-1] + 2 * width)
    save_title = ''
    if percent:
        retitle = ' in percent'
        save_title = retitle.replace(' ', '_')
        loc = 3
    else:
        retitle = ''
        loc = 2
    ax.set_ylabel('Nb of Flows %s' % retitle)
#    new_x_loc = np.array(x_loc[:-1]) - 2
    ax.set_xticks([cur_loc - len(as_list) / 2
                   for (cur_loc, _ ) in x_loc_titles])
    #values+shift/2., map(format_title, traces), size=8)
    ax.set_xticklabels([format_title(cur_title)
                        for (_, cur_title) in x_loc_titles],
                       size='small') #, rotation=30)
    ax.yaxis.grid(b=True)
    legend_data = []
    legend_name = []
#    for percent_type in percent_types:
#        # take last plot as ref
#        legend_data.append(plot[(len(fields) - 1, loss_interval)][0])
#        legend_name.append(loss_interval)
#    font = FontProperties(size = 'x-small')
    ax.legend(loc=(0 if not percent else 3))
#legend_data, legend_name, loc=loc, prop=font)
    title = """Nb of flows%s by remaining download per traffic capture
with min flow size: %g Bytes""" % (retitle, min_size)
    if not percent:
        ax.set_title(title)
    else:
        fig.suptitle(title)
    fig.savefig('%s/remaining_bars%s_min_%d.pdf'
                % (output_path, save_title, min_size),
                format='pdf')

