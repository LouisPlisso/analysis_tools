#!/usr/bin/env python
"""Module to filter the flows indicator to retrieve in one numpy array all
relevant indicatiors
"""

from __future__ import division, print_function, absolute_import
import cPickle
import sys
# in case of non-interactive usage
import matplotlib
matplotlib.use('PDF')
IMAGE_FORMAT = 'pdf'
#import matplotlib.pyplot as plt
from math import log as logarithm
from math import ceil
from os import sep
import logging
LOG_LEVEL = logging.DEBUG
if 'LOG' not in locals():
    LOG = None
import numpy as np
from collections import defaultdict
from operator import itemgetter, concat

if __name__ == "__main__" and __package__ is None:
    __package__ = "tools"
    import tools

from . import INDEX_VALUES
from . import streaming_tools
from . import filter_streaming
from . import cdfplot_2

def configure_log():
    "Configure logger"
    if LOG:
        return LOG
    log = logging.getLogger('filter_streaming')
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        log_formatter = logging.Formatter(
            "%(asctime)s - %(filename)s:%(lineno)d - "
            "%(levelname)s - %(message)s")
        handler.setFormatter(log_formatter)
        log.addHandler(handler)
    log.setLevel(LOG_LEVEL)
    return log

LOG = configure_log()

def get_convex_metric(data, with_log=True):
    "Return the convex index of data"
    if with_log:
        func = logarithm
    else:
        func = (lambda x: x)
    sorted_data = sorted(data)
    len_data = len(data)
    median = sorted_data[int(len_data * .5)]
    percentile = sorted_data[int(len_data * .9)]
    print(median, percentile)
    return func(median) / func(percentile)

def get_args(flows, indic, need_format_title=True):
    "Return the formated arg list to pass to plot_indic"
    if need_format_title:
        format_title = streaming_tools.format_title
    else:
        format_title = (lambda x: x)
    return sorted([(format_title(k), [indic(x) for x in v])
                   for k, v in sorted(flows.items())])

AS_NB_NAME = {'43515': 'YT EU',
              '15169': 'GOO',
              '36561': 'YT',
              '1273': 'C\&W',
              '3549': 'GBLX',
              '41690': 'DM',
              '22822': 'LL',
             }

MIN_VOL_AS = 1e7

def print_vol_rtt_per_as(youtube_flows, dailymotion_flows, out_dir):
    "Print a formatted table of vol and RTT per AS for latex inclusion"
    # YouTube
    yt_tot_vol_per_as = defaultdict(float)
    yt_tot_vol_per_trace = defaultdict(float)
    vols_yt = defaultdict(dict)
    rtt_yt = defaultdict(dict)
    for trace, data in sorted(youtube_flows.items()):
        if 'asBGP' in data.dtype.names:
            indic_BGP = 'asBGP'
            splitter = lambda x: x
        elif 'RemAS' in data.dtype.names:
            indic_BGP = 'RemAS'
            splitter = lambda x: int(x.split('AS')[1])
        else:
            LOG.error('no AS field found for this data')
        for cur_as in map(splitter, set(data[indic_BGP])):
            if ((cur_as == INDEX_VALUES.AS_YOUTUBE[0])
                and (not trace.startswith('2008'))):
                continue
            if ((cur_as == INDEX_VALUES.AS_YOUTUBE_EU[0])
                and trace.startswith('2008')):
                continue
            cur_data = data.compress([splitter(x[indic_BGP]) == cur_as
                                      for x in data])
            vols_yt[trace][cur_as] = sum(cur_data['ByteDn'])
            yt_tot_vol_per_trace[trace] += vols_yt[trace][cur_as]
            yt_tot_vol_per_as[cur_as] += vols_yt[trace][cur_as]
            rtt_yt[trace][cur_as] = np.median(cur_data['DIP-RTT-Min-ms-TCP-Up'])
    yt_all_ases_vols = sorted(yt_tot_vol_per_as.items(), key=itemgetter(1),
                           reverse=True)
    # DailyMotion
    # HARD CODED AS list
    dm_as_list = set((41690, 22822))
    dm_tot_vol_per_as = defaultdict(float)
    dm_tot_vol_per_trace = defaultdict(float)
    vols_dm = defaultdict(dict)
    rtt_dm = defaultdict(dict)
    for trace, data in sorted(dailymotion_flows.items()):
        if 'asBGP' in data.dtype.names:
            indic_BGP = 'asBGP'
            splitter = lambda x: x
        elif 'RemAS' in data.dtype.names:
            indic_BGP = 'RemAS'
            splitter = lambda x: int(x.split('AS')[1])
        else:
            LOG.error('no AS field found for this data')
        for cur_as in set(map(splitter,
                              data[indic_BGP])).intersection(dm_as_list):
            cur_data = data.compress([splitter(x[indic_BGP]) == cur_as
                                      for x in data])
            vols_dm[trace][cur_as] = sum(cur_data['ByteDn'])
            dm_tot_vol_per_trace[trace] += vols_dm[trace][cur_as]
            dm_tot_vol_per_as[cur_as] += vols_dm[trace][cur_as]
            rtt_dm[trace][cur_as] = np.median(cur_data['DIP-RTT-Min-ms-TCP-Up'])
    dm_all_ases_vols = sorted(dm_tot_vol_per_as.items(), key=itemgetter(1),
                           reverse=True)
    LOG.debug(dm_all_ases_vols)
    if dm_all_ases_vols and yt_all_ases_vols:
        header_line = [' & '.join([''] +
                            [r'\multicolumn{%d}{c}{YouTube}'
                             % (2 * len(yt_all_ases_vols))] +
                            ([r'\multicolumn{%d}{c}{DailyMotion}'
                             % (2 * len(dm_all_ases_vols))]
                             if dm_all_ases_vols else []))]
    else:
         header_line = []
    table = header_line
    table += [' & '.join(['']
                         + ['\multicolumn{2}{c}{%s}'
                                % AS_NB_NAME[str(as_nb)]
                                for as_nb in map(itemgetter(0),
                                                 yt_all_ases_vols)]
                         + ['\multicolumn{2}{%sc}{%s}'
                                % ('|' if as_nb == 41690 else '',
                                   AS_NB_NAME[str(as_nb)])
                                for as_nb in map(itemgetter(0),
                                                 dm_all_ases_vols)])
             + '\\\\\n' + ' & '.join([''] + ['\multicolumn{2}{c}{AS %s}' % as_nb
                         for as_nb in map(itemgetter(0), yt_all_ases_vols)]
                                     + ['\multicolumn{2}{%sc}{AS %s}'
                                        % ('|' if as_nb == 41690 else '', as_nb)
                         for as_nb in map(itemgetter(0), dm_all_ases_vols)])]
    table += [' & '.join(['']
                         + ['Vol.', 'RTT'] * (len(yt_all_ases_vols) +
                                                len(dm_all_ases_vols)))]
    # youtube_flows and dailymotion_flows have same traces
    for trace in sorted(youtube_flows):
        # at least 10 MB of data to consider it
        table += ['&'.join(
            [streaming_tools.format_title(trace).replace('\n', ' ')]
            +reduce(concat, [
                [(' $%d\\%%$ ' % round(100 * (vols_yt[trace][cur_as] /
                                              yt_tot_vol_per_trace[trace]))
                  if cur_as in vols_yt[trace] else '--'),
                 (' $%d$ ' % ceil(rtt_yt[trace][cur_as])
                  if cur_as in rtt_yt[trace] else '--')]
                for cur_as, tot_vol in yt_all_ases_vols
                if tot_vol > MIN_VOL_AS], [])
            + reduce(concat, [
                [(' $%d\\%%$ ' % int(round(100 * (vols_dm[trace][cur_as] /
                                     dm_tot_vol_per_trace[trace])))
                  if cur_as in vols_dm[trace] else '--'),
                 (' $%d$ ' % ceil(rtt_dm[trace][cur_as])
                  if cur_as in rtt_dm[trace] else '--')]
                for cur_as, tot_vol in dm_all_ases_vols
                if tot_vol > MIN_VOL_AS], []))]
    with open(sep.join((out_dir, 'table_vol_rtt_as.tex')), 'w') as out_file:
        print(r'\begin{tabular}{l%s}'
              % ('c' * 2 * len(yt_all_ases_vols)
                 + ('|' + 'c' * 2 * len(dm_all_ases_vols)
                    if dm_all_ases_vols else '')),
              file=out_file)
        print(r'\toprule{}', file=out_file)
        print('\\\\\n\\midrule{}\n'.join(table).replace('_', '\\_'),
              file=out_file)
        print(r'\\ \bottomrule{}', file=out_file)
        print(r'\end{tabular}', file=out_file)
    return (map(itemgetter(0), yt_all_ases_vols),
            map(itemgetter(0), dm_all_ases_vols))

MIN_DOWNLOADED_DURATION = .9

def get_downloaded_duration(x):
    "Return the downloaded duration according to bitrate and downloaded vol"
    return (x['DIP-Volume-Sum-Bytes-Down'] * 8e-3
            / x['Content-Avg-Bitrate-kbps'])

def get_wasted_bytes(x):
    "Return the number of bytes downloaded but not watched"
    # (downloaded_bytes - video_rate_bytes/s * duration)
    return (x['ByteDn'] -
            ((x['Content-Length'] / x['Content-Duration']) *
             x['DurationDn']))
#    return (x['Session-Bytes'] -
#            ((x['Content-Length'] / x['Content-Duration']) *
#             x['Session-Duration']))
    # is_good defined as:
    #return ((8e-3 * flow['ByteDn'] / flow['DurationDn'])
            #> (8e-3 * flow['Content-Length'] / flow['Content-Duration'])

def get_fraction_wasted_bytes(x):
    "Return the fractio of bytes downloaded but not watched"
    return 100 * (get_wasted_bytes(x) / x['ByteDn'])

def get_downloaded_vol(x):
    "Return the fraction of downloaded volume"
    return min(100, 100 * (get_downloaded_duration(x)
                           / x['Content-Duration']))
MIN_DURATION = 3

def is_short_duration(x):
    "Return if the video is short"
    return get_downloaded_duration(x) < 60 * MIN_DURATION

def is_complete(x):
    "Return if the video is almost completely downloaded"
    return ((get_downloaded_duration(x) / x['Content-Duration'])
            > MIN_DOWNLOADED_DURATION)

def plot_relation_qual_dow(flows, out_dir, bin_size=10,
                           max_time=700, provider='YouTube'):
    """Plot a graph of percentage of complete downloaded videos in function of
    content duration with bin seconds, and related to quality
    """
    LOG.info('process vol download: %s' % provider)
    data = {}
    data['Good'] = sorted([(x['Content-Duration'], get_downloaded_duration(x))
                           for flow in flows.values() for x in flow
                           if x['good']])
    data['Bad'] = sorted([(x['Content-Duration'], get_downloaded_duration(x))
                          for flow in flows.values() for x in flow
                          if not x['good']])
    for quality in ('Good', 'Bad'):
        to_plot = []
        for i in xrange(len(data[quality]) // bin_size):
            current_data = data[quality][bin_size * i:bin_size * (i + 1)]
            xs = map(itemgetter(0), current_data)
            ys = map(itemgetter(1), current_data)
            left = min(xs)
            width = max(xs) - left
            height = (len([x for x in current_data
                           if x[1] > MIN_DOWNLOADED_DURATION * x[0]]) / bin_size)
            yerr = np.std(ys) / bin_size**2
            to_plot.append((left, width, height, yerr))
        figure = cdfplot_2.bin_plot(to_plot,
                              ylabel='Fraction of Complete Videos per bin',
            title='Fraction of Complete Videos for %s: %s' % (provider, quality),
            xlabel='Content Duration in sec. (bins of %d videos)' % bin_size)
        figure.set_xlim((0, max_time))
        figure.set_ylim((0, 1))
        figure.savefig(sep.join((out_dir,
                                 '.'.join(('vol_quality_download_%s_%s'
                                           % (provider.lower(), quality.lower()),
                                          IMAGE_FORMAT)))))
    LOG.info('%s vol download done' % provider)

def plot_global_downloaded_size_cdf(flows, out_dir):
    "Plot a CDF of global fraction of downloaded duration"
    LOG.info('process global downloaded quality')
    data = {}
    for i, (service, abbr) in enumerate([('YouTube', 'YT'),
                                         ('DailyMotion', 'DM')]):
        data['\phantom{%d}Good %s' % (i, abbr)] = [x for flow in flows.values()
                                                   for x in flow if x['good']
                                                   and x['Service'] == service]
        data['\phantom{%d}Bad %s' % (i, abbr)] = [x for flow in flows.values()
                                                  for x in flow if not x['good']
                                                  and x['Service'] == service]
#    else:
#        data['\phantom{%d}Good Others' % (i+1)] = [x for flow in flows.values()
#                                                   for x in flow if x['good']
#                                                  and x['Service'] != 'YouTube'
#                                           and x['Service'] != 'DailyMotion']
#        data['\phantom{%d}Bad Others' % (i+1)] = [x for flow in flows.values()
#                                                 for x in flow if not x['good']
#                                                  and x['Service'] != 'YouTube'
#                                             and x['Service'] != 'DailyMotion']
    scale_downloaded = (lambda (xmin, xmax): (xmin, min(xmax, 101)))
    filter_streaming.plot_indic(get_args(data, get_downloaded_vol,
                                         need_format_title=False),
                            'Downloaded Duration in fct. of Reception Quality',
                                'in percent',
                                scale_downloaded, out_dir, '', legend_ncol=3,
                                plot_line=[(20, '20\%'), (95, '95\%')],
                                loc='upper left', plot_all_x=True,
                                plot_ccdf=False, dashes=True)
    LOG.info('done global downloaded quality')

def plot_relation_vol_quality(flows, out_dir, provider='YouTube', max_time=700):
    """Plot a scatterplot of downloaded duration in function of content length
    and separate by quality
    """
    LOG.info('process vol quality: %s' % provider)
    data = {}
    get_durations = lambda x: (x['Content-Duration'],
                               get_downloaded_duration(x))
        #x['Content-Length'] * 8e-3 / x['Content-Avg-Bitrate-kbps'],
    data['Good'] = [x for flow in flows.values() for x in flow if x['good']]
    data['Bad'] = [x for flow in flows.values() for x in flow if not x['good']]
    for quality in ('Good', 'Bad'):
        to_plot = {}
        to_plot['\phantom{0}$> %d$ min.\nand completly downloaded'
                % MIN_DURATION] = [ get_durations(x) for x in data[quality]
                                   if is_complete(x) and not is_short_duration(x)]
        to_plot['$\phantom{1}\leq %d$ min.' % MIN_DURATION] = [
            get_durations(x) for x in data[quality]
            if is_short_duration(x)]
        to_plot['\phantom{2}other ($> %d$ min and incomplete)'
                % MIN_DURATION] = [get_durations(x) for x in data[quality]
                                   if not is_complete(x)
                                   and not is_short_duration(x)]
        figure = cdfplot_2.scatter_plot_multi(to_plot,
                                  title='%s Videos with %s Reception Quality'
                                          % (provider, quality),
                                          xlabel='Content Duration in sec.',
                                          ylabel='Downloaded Duration in sec.')
        figure.set_xlim((0, max_time))
        figure.set_ylim((0, max_time))
        figure.savefig(sep.join((out_dir,
                                 '.'.join(('vol_quality_%s_%s'
                                           % (provider.lower(), quality.lower()),
                                           IMAGE_FORMAT)))))
    LOG.info('%s vol quality done' % provider)

def get_nb_as_per_client(flows, min_nb_videos=1):
    """"Return a dict of nb of videos and nb of distinct ASes per client for
    each trace in flows
    only consider clients with at least min_nb_videos
    """
    out_dict = dict()
    for trace, data in flows.items():
        if 'asBGP' in data.dtype.names:
            indic_BGP = 'asBGP'
            #splitter = lambda x: x
        elif 'RemAS' in data.dtype.names:
            indic_BGP = 'RemAS'
            #splitter = lambda x: int(x.split('AS')[1])
        else:
            LOG.error('no AS field found for this data')
        if 'client_id' in data.dtype.names:
            indic_client = 'client_id'
        elif 'Name' in data.dtype.names:
            indic_client = 'Name'
        else:
            LOG.error('no client field found for this data')
        out_dict[trace] = [(client,
                   sorted(zip([host_referer.split('.')
                           for host_referer in clients_videos['Host_Referer']],
                              clients_videos[indic_BGP]),
                          key=lambda (x, y): (x[1] if len(x) > 1 else x)),
                   len(clients_videos), len(set(clients_videos[indic_BGP])))
                  for client in set(data[indic_client])
                  for clients_videos in
                  (data.compress(data[indic_client] == client),)
                  if len(clients_videos) >= min_nb_videos]
    return out_dict
#    dict((trace, [(client, sorted(zip([host_referer.split('.')
#                           for host_referer in clients_videos['Host_Referer']],
#                              clients_videos[indic_BGP]),
#                          key=lambda (x, y): (x[1] if len(x) > 1 else x)),
#                   len(clients_videos), len(set(clients_videos[indic_BGP])))
#                  for client in set(data['client_id'])
#                  for clients_videos in
#                  (data.compress(data['client_id'] == client),)
#                  if len(clients_videos) >= min_nb_videos])
#                for trace, data in flows.items())

def table_nb_as_per_client(youtube_flows, out_dir='graph_filtered_imc'):
    """Return a table of nb distinct ASes in function of nb of videos per client
    dedicated to YouTube flows
    """
    LOG.info('start table nb as per client')
    # hard coded
    max_nb_ases = 4
    total_nb = {}
    nb_as = {}
    table = [' & '.join(['', 'Total',
                         r'\multicolumn{%d}{c}{\# distinct ASes per client}\\'
                         % max_nb_ases]) + '\n' +
             ' & '.join(['Trace', '\\# ASes']
                         + map(str, range(1, max_nb_ases + 1)))]
    for trace, data in sorted(get_nb_as_per_client(youtube_flows,
                                                   min_nb_videos=4).items()):
        nb_as[trace] = defaultdict(int)
        total_nb[trace] = len(data)
        for (client, host_as_list, nb_clients_videos, nb_ases) in data:
            nb_as[trace][nb_ases] += 1
        # print table
        table += [' & '.join(
            [streaming_tools.format_title(trace).replace('\n', ' '),
             str(max(nb_as[trace]))] +
            ['%d\\%%' % (round(100 * nb_as[trace][nb_ases] /
                                  total_nb[trace]))
             for nb_ases in sorted(nb_as[trace])]
            + ['--'] * (max_nb_ases - max(nb_as[trace])))]
    with open(sep.join((out_dir, 'table_nb_as.tex')), 'w') as out_file:
        print(r'\begin{tabular}{lccccc}', file=out_file)
        print(r'\toprule{}', file=out_file)
        print('\\\\\n\\midrule{}\n'.join(table).replace('_', '\\_'),
              file=out_file)
        print(r'\\ \bottomrule{}', file=out_file)
        print(r'\end{tabular}', file=out_file)
    LOG.info('table nb as per client done')

def table_user_exp(youtube_flows, dailymotion_flows, yt_as_list, dm_as_list,
                   out_dir):
    """Return a formatted table for latex of fraction of flows with lower
    average bit-rate than video encoding rate
    """
    LOG.info('start table user exp')
    yt_as_list = list(yt_as_list)
    yt_qual = defaultdict(dict)
    total_yt = defaultdict(int)
    total_yt_as = defaultdict(dict)
    for trace, data in youtube_flows.items():
        for cur_as in yt_as_list:
            if ((cur_as == INDEX_VALUES.AS_YOUTUBE[0])
                and (not trace.startswith('2008'))):
                continue
            if ((cur_as == INDEX_VALUES.AS_YOUTUBE_EU[0])
                and trace.startswith('2008')):
                continue
            data_as = data.compress(data['asBGP'] == cur_as)
            if sum(data_as['ByteDn']) < MIN_VOL_AS:
                continue
            yt_qual[trace][cur_as] = len([x for x in data_as
                                          if x['good'] == False])
            total_yt_as[trace][cur_as] = len(data_as)
            total_yt[trace] += len(data_as)
    LOG.debug(sorted(yt_qual.items()))
    LOG.debug(sorted(total_yt.items()))
    dm_as_list = list(dm_as_list)
    dm_qual = defaultdict(dict)
    total_dm = defaultdict(int)
    total_dm_as = defaultdict(dict)
    for trace, data in dailymotion_flows.items():
        for cur_as in dm_as_list:
            if (cur_as == INDEX_VALUES.AS_TINET[0]):
                continue
            data_as = data.compress(data['asBGP'] == cur_as)
            if sum(data_as['ByteDn']) < MIN_VOL_AS:
                continue
            dm_qual[trace][cur_as] = len([x for x in data_as
                                          if x['good'] == False])
            total_dm_as[trace][cur_as] = len(data_as)
            total_dm[trace] += len(data_as)
    LOG.debug(sorted(dm_qual.items()))
    LOG.debug(sorted(total_dm.items()))
    table = [' & '.join([''] + ['\multicolumn{%d}{c}{%s}'
                                % (len(yt_as_list), 'YouTube')]
                        + ['\multicolumn{%d}{c}{%s}'
                           % (len(dm_as_list), 'DailyMotion')])]
    table += [' & '.join([''] + [AS_NB_NAME[str(as_nb)] for as_nb in yt_as_list]
                         + [AS_NB_NAME[str(as_nb)] for as_nb in dm_as_list])
              + r'\\' + '\n' +
              ' & '.join(['Trace'] + ['AS %d' % x
                                      for x in (yt_as_list + dm_as_list)])]
    for trace in sorted(youtube_flows):
        table += ['&'.join(
            [streaming_tools.format_title(trace).replace('\n', ' ')]
            + [(' $%d\\%%$ ' % int(round(100 * (yt_qual[trace][cur_as] /
                                               total_yt_as[trace][cur_as])))
                if cur_as in yt_qual[trace] else '--')
               for cur_as in yt_as_list]
            + [(' $%d\\%%$ ' % int(round(100 * (dm_qual[trace][cur_as] /
                                               total_dm_as[trace][cur_as])))
                if cur_as in dm_qual[trace] else '--')
               for cur_as in dm_as_list]
                            )]
    with open(sep.join((out_dir, 'table_mean_enc_rate.tex')), 'w') as out_file:
        print(r'\begin{tabular}{l%s}'
              % ('c' * len(yt_as_list) + '|' + 'c' * len(dm_as_list)),
              file=out_file)
        print(r'\toprule{}', file=out_file)
        print('\\\\\n\\midrule{}\n'.join(table).replace('_', '\\_'),
              file=out_file)
        print(r'\\ \bottomrule{}', file=out_file)
        print(r'\end{tabular}', file=out_file)
    LOG.info('table user exp done')

MIN_NB_SAMPLES = 4

def diff_unsigned(a, b):
    'Function to compare unsigned values'
    return abs(float(a) - float(b)) / float(a) if a else 0

def merge_dipcp_cnx_stream_tstat(dipcp_cnx_array, tstat_array):
    '''Return an array with corresponding files merged
    '''
    assert dipcp_cnx_array.dtype == np.dtype(INDEX_VALUES.dtype_dipcp_cnx_stream), \
            'incorrect dtype of input dipcp_cnx_array'
    assert tstat_array.dtype == np.dtype(INDEX_VALUES.dtype_tstat2), \
            'incorrect dtype of input tstat array'
    output = []
    nb_not_found = 0
    nb_multi = 0
    nb_tot = 0
    flow_dict = filter_streaming.construct_hash_tstat(tstat_array)
    for data in dipcp_cnx_array:
        nb_tot += 1
        cnx_id = None
        if (data['FlowIPSource'], data['FlowPortSource'],
            data['FlowIPDest'], data['FlowPortDest']) in flow_dict:
            cnx_id = (data['FlowIPSource'], data['FlowPortSource'],
                      data['FlowIPDest'], data['FlowPortDest'])
        elif (data['FlowIPDest'], data['FlowPortDest'],
              data['FlowIPSource'], data['FlowPortSource']) in flow_dict:
            cnx_id = (data['FlowIPDest'], data['FlowPortDest'],
                      data['FlowIPSource'], data['FlowPortSource'])
        else:
            nb_not_found += 1
            LOG.debug('Stream not found in cnx_stream: %s'
                      % ' '.join(map(str, (data['FlowIPSource'],
                                           data['FlowPortSource'],
                                           data['FlowIPDest'],
                                           data['FlowPortDest']))))
        if cnx_id:
            if len(flow_dict[cnx_id]) > 1:
                LOG.info("Multiple flows for this stream: %s" % str(cnx_id))
                nb_multi += 1
            output.append(tuple(list(data) + list(flow_dict[cnx_id][0])))
    print('nb_tot, nb_not_found, nb_multi', nb_tot, nb_not_found, nb_multi)
    return np.array(output, dtype=INDEX_VALUES.dtype_dipcp_cnx_stream_tstat2)

def merge_dipcp_cnx_stream(dipcp_array, cnx_stream_array, strict=True):
    '''Return an array with corresponding files merged
    unfound connections are removed?
    assumption: the dipcp_array is much smaller than the cnx_stream_array one
    filter dipcp_array with:
dipcp_stream = dipcp_test.compress(dipcp_test['TOS']
                            == 4 * tools.INDEX_VALUES.DSCP_MARCIN_HTTP_STREAM)
    '''
    assert cnx_stream_array.dtype == np.dtype(INDEX_VALUES.dtype_cnx_stream_loss), \
            'incorrect dtype of input dipcp_array'
    assert dipcp_array.dtype == np.dtype(INDEX_VALUES.dtype_dipcp_compat), \
            'incorrect dtype of input dipcp_array'
    assert len(set(dipcp_array['TOS'])) == 1, 'unfiltered dipcp array'
    output = []
    nb_not_found = 0
    nb_multi = 0
    nb_small = 0
    nb_tot = 0
    flow_dict = filter_streaming.construct_hash_cnx_stream(cnx_stream_array)
    for data in dipcp_array:
        nb_tot += 1
        if data['DIP-Volume-Sum-Bytes-Down'] < 1e5:
            # skip too small stream flows
            nb_small += 1
            continue
        cnx_id = None
        # src port is unreliable in cnx_stream!
        if (data['FlowIPSource'],
            #data['FlowPortSource'],
            data['FlowIPDest'], data['FlowPortDest']) in flow_dict:
            cnx_id = (data['FlowIPSource'],
                      #data['FlowPortSource'],
                      data['FlowIPDest'], data['FlowPortDest'])
        elif (data['FlowIPDest'],
              #data['FlowPortDest'],
              data['FlowIPSource'], data['FlowPortSource']) in flow_dict:
            cnx_id = (data['FlowIPDest'],
                      #data['FlowPortDest'],
                      data['FlowIPSource'], data['FlowPortSource'])
        else:
            nb_not_found += 1
            LOG.debug('Stream not found in cnx_stream: %s'
                      % ' '.join(map(str, (data['FlowIPSource'],
                                           data['FlowPortSource'],
                                           data['FlowIPDest'],
                                           data['FlowPortDest']))))
        if cnx_id:
            if (len(flow_dict[cnx_id]) > 1
                and max(map(itemgetter('nByte'), flow_dict[cnx_id])) > 1e5):
                # log multi only if there are cnx larger than 100ko
                LOG.info("Multiple flows for this stream: %s" % str(cnx_id))
                nb_multi += 1
                if strict:
                    continue
            closest_flow = flow_dict[cnx_id][0]
            bytes_difference = diff_unsigned(flow_dict[cnx_id][0]['nByte'],
                                             data['DIP-Volume-Sum-Bytes-Down'])
            for flow in flow_dict[cnx_id]:
                if (diff_unsigned(flow['nByte'],
                                  data['DIP-Volume-Sum-Bytes-Down'])
                    < bytes_difference):
                    closest_flow = flow
                    bytes_difference = diff_unsigned(flow['nByte'],
                                             data['DIP-Volume-Sum-Bytes-Down'])
            output.append(tuple(list(data) + list(closest_flow)))
    print('nb_tot, nb_not_found, nb_multi, nb_small',
          nb_tot, nb_not_found, nb_multi, nb_small)
    return np.array(output, dtype=INDEX_VALUES.dtype_dipcp_cnx_stream)

def main(
    flows_file='/home/louis/streaming/filtered_flows_complete_all_tstat.pickle',
    out_dir='graph_filtered_imc', fast=True, filtered_flows=None, compat=False):
    "Plot main graphs in one step"
    LOG.info('Start plotting with fast mode: %s' % fast)
    if not filtered_flows:
        filtered_flows = cPickle.load(open(flows_file))
    if not compat:
        plot_global_downloaded_size_cdf(filtered_flows, out_dir)
    LOG.info('Focus on FTTH')
    flows_ftth = dict([(key, value) for key, value in filtered_flows.items()
                       if 'FTTH' in key])
    dailymotion_flows = filter_streaming.filter_service(filtered_flows,
                                                        'DailyMotion')
    youtube_flows = filter_streaming.filter_service(filtered_flows,
                                                    'YouTube')
    del(filtered_flows)
    table_nb_as_per_client(youtube_flows, out_dir=out_dir)
    get_peak = (lambda x: 80e-3 * x['peakRate'])
    get_mean = (lambda x: 8e-3 * x['ByteDn'] / x['DurationDn'])
    get_rtt = (lambda x: x['DIP-RTT-DATA-Min-ms-TCP-Up'])
    get_rtt_compat = (lambda x: x['DIP-RTT-Min-ms-TCP-Up'])
    get_loss = (lambda x: (100 * x['S_out_seq_packets'] / x['S_data_packets'])
                if x['S_data_packets'] != 0 else None)
    get_burst = (lambda x: (1 - (x['DIP-DSQ-NbMes-sec-TCP-Down']
                           / (x['S_rexmit_packets'] + x['S_out_seq_packets'])))
                 if (x['S_rexmit_packets'] + x['S_out_seq_packets'])
                 > 0 else None)
    get_video_rate = (lambda x: (8e-3 * x['Content-Length']
                                 / x['Content-Duration']))
    scale_burst = (lambda (xmin, xmax): (0, 1))
    scale_other = (lambda (xmin, xmax): (5e2, 1e5))
    scale_yt_peak_as = (lambda (xmin, xmax): (1e3, 4e4))
    scale_yt_peak = (lambda (xmin, xmax): (4e2, 4e4))
    scale_yt_mean_as = (lambda (xmin, xmax): (1e2, 1e4))
    scale_yt_mean = (lambda (xmin, xmax): (1e2, 1e5))
    scale_rtt = (lambda (xmin, xmax): (10, 200))
    scale_downloaded = (lambda (xmin, xmax): (xmin, min(xmax, 101)))
    scale_dm_peak = (lambda (xmin, xmax): (5e3, 1e5))
    scale_loss = (lambda (xmin, xmax): (1e-1, 1e2))
    scale_fract_wasted = (lambda (xmin, xmax): (1, 1e2))
    scale_dm_mean = (lambda (xmin, xmax): (1e2, 1e5))
    scale_video_rate = (lambda (xmin, xmax): (max(xmin, 1e2), min(3e3, xmax)))
    LOG.info('start final_plot')
    if not compat:
        LOG.info('process window size')
        get_window_size = (lambda x: (2**x['S_window_scale']) * x['S_win_max'])
        args = sorted([(streaming_tools.format_title(key),
                        get_window_size(value))
                       for (key, value) in flows_ftth.items()])
        fig = cdfplot_2.cdfplotdata(args, loc='lower right',
                                      title='Max Window Size for FTTH flows',
                                      xlabel='Window Size in Bytes')
        del(args)
        fig.set_xlim((5e3, 1e7))
        fig.savefig(sep.join((out_dir, '.'.join(('cdf_window_sizes_ftth',
                                                 IMAGE_FORMAT)))))
        del(fig)
        LOG.info('window sizes done')
    LOG.info('process DailyMotion losses')
    filter_streaming.plot_indic(get_args(dailymotion_flows, get_loss),
                                'Backbone Loss Rate per flow', 'in percent',
                                scale_loss, out_dir, 'DailyMotion',
                                loc='lower right',
                                plot_line=[(2, '2\%')],
                                fs_legend='small',
                                plot_all_x=False, plot_ccdf=False)
    LOG.info('DailyMotion losses done')
    dailymotion_ftth = filter_streaming.filter_service(flows_ftth,
                                                       'DailyMotion')
    if not fast:
        plot_relation_qual_dow(dailymotion_flows, out_dir,
                               provider='DailyMotion')
    if not compat:
        plot_relation_vol_quality(dailymotion_flows, out_dir,
                                  provider='DailyMotion')
        LOG.info('process DailyMotion video rates')
        filter_streaming.plot_indic(get_args(dailymotion_flows, get_video_rate),
                                    'Encoding Rate of Video', 'in kb/s',
                                    scale_video_rate, out_dir, 'DailyMotion',
                                    loc='lower right', legend_ncol=1,
                                    fs_legend='medium',
                                    plot_line=[(474, 'Median\nRate')],
                                    plot_all_x=True, plot_ccdf=False)
        LOG.info('DailyMotion video rates done')
    if not fast:
        LOG.info('process DailyMotion bursts')
        filter_streaming.plot_indic(get_args(dailymotion_flows, get_burst),
                                    'Loss Burstiness per flow', '',
                                    scale_burst, out_dir, 'DailyMotion',
                                    loc='upper left', legend_ncol=2,
                                    plot_all_x=True, plot_ccdf=False)
        LOG.info('DailyMotion bursts done')
    if not compat:
        LOG.info('process DailyMotion downloaded vol')
        filter_streaming.plot_indic(get_args(dailymotion_flows, get_downloaded_vol),
                                    'Downloaded Duration per flow',
                                    'in percent',
                                    scale_downloaded, out_dir, 'DailyMotion',
                                    loc='upper left', fs_legend='medium',
                                    plot_all_x=True, plot_ccdf=False, legend_ncol=2)
        LOG.info('DailyMotion downloaded vol done')
    # we do not want dailymotion in rtt table, so put {} instead of
    # dailymotion_flows and HARDCODE AS list
    (yt_as_list, dm_as_list) = print_vol_rtt_per_as(youtube_flows,
                                                    dailymotion_flows, out_dir)
    dm_as_list = INDEX_VALUES.AS_DAILYMOTION + INDEX_VALUES.AS_LIMELIGHT
    if not compat:
        table_user_exp(youtube_flows, dailymotion_flows, yt_as_list, dm_as_list,
                       out_dir)
    if not compat:
        LOG.info('process DailyMotion wasted bytes')
        dailymotion_flows_good = dict([(k, v.compress(v['good'] == True))
                                   for (k, v) in dailymotion_flows.items()])
        filter_streaming.plot_indic(get_args(dailymotion_flows_good,
                                             get_wasted_bytes),
                                    'Wasted Bytes per Flow (Good)',
                                    'in bytes',
                                    (lambda x: x), out_dir, 'DailyMotion',
                                    fs_legend='medium',
                                    plot_all_x=True, plot_ccdf=False,
                                    legend_ncol=2)
        LOG.info('DailyMotion wasted bytes done')
        LOG.info('process DailyMotion wasted fraction')
        filter_streaming.plot_indic(get_args(dailymotion_flows_good,
                                             get_fraction_wasted_bytes),
                                    'Fraction of Wasted Bytes per Flow',
                                    'in percent',
                                    scale_fract_wasted, out_dir, 'DailyMotion',
                                    fs_legend='small', loc='lower center',
                                    plot_all_x=True, plot_ccdf=False,
                                    legend_ncol=1)
        del(dailymotion_flows_good)
        LOG.info('DailyMotion wasted fraction done')
        del(dailymotion_flows)
        LOG.info('process DailyMotion FTTH peaks')
        filter_streaming.plot_indic(get_args(dailymotion_ftth, get_peak),
                                    'Peak Rate per flow', 'in kb/s',
                                    scale_dm_peak, out_dir, 'DailyMotion',
                                    loc='upper left',
                                    plot_all_x=False, plot_ccdf=False)
        LOG.info('DailyMotion peak done')
    LOG.info('process DailyMotion FTTH mean rates')
    filter_streaming.plot_indic(get_args(dailymotion_ftth, get_mean),
                                'Mean Rate per flow', 'in kb/s',
                                scale_dm_mean, out_dir, 'DailyMotion',
                                loc='lower right',
                                plot_line=[(474, 'Median\nEncoding Rate')],
                                plot_all_x=False, plot_ccdf=False)
    LOG.info('DailyMotion mean rates done')
    if not fast:
        LOG.info('start DailyMotion client')
        dm_peak_rate_per_client = (
            filter_streaming.retrieve_peak_rate_per_client(dailymotion_ftth))
        filter_streaming.plot_indic(sorted(dm_peak_rate_per_client),
                                    'Peak Rate per Client', 'in kb/s',
                                    scale_loss, out_dir, 'DailyMotion',
                                    loc='upper left',
                                    plot_all_x=False, plot_ccdf=False)
        LOG.info('DailyMotion client done')
        LOG.info('process DailyMotion FTTH bursts')
        filter_streaming.plot_indic(get_args(dailymotion_ftth, get_burst),
                                    'Loss Burstiness per flow', '',
                                    scale_burst, out_dir, 'DailyMotion FTTH',
                                    loc='lower right',
                                    plot_all_x=True, plot_ccdf=False)
        LOG.info('DailyMotion FTTH bursts done')
    del(dailymotion_ftth)
    LOG.info('DailyMotion done')
    if not compat:
        plot_relation_qual_dow(youtube_flows, out_dir,
                               provider='YouTube')
        plot_relation_vol_quality(youtube_flows, out_dir,
                                  provider='YouTube')
    LOG.info('process YouTube ADSL mean rates per prefix')
    filter_streaming.plot_per_prefix(youtube_flows, '2010_02_07_ADSL_R',
                                     get_mean,
                                     'Mean Rate per flow', 'in kb/s',
                                     scale_yt_mean_as, loc='lower right',
                                     plot_line=[(323, 'Median\nEncoding rate')],
                                     out_dir=out_dir, service='YouTube',
                                     fs_legend='small', plot_all_x=False,
                                     plot_ccdf=False)
    if compat:
        filter_streaming.plot_per_prefix(youtube_flows, '2011_05_03_ADSL_R',
                                         get_mean,
                                         'Mean Rate per flow', 'in kb/s',
                                         scale_yt_mean_as, loc='upper left',
                                         ip_field='FlowIPDest',
                                     plot_line=[(323, 'Median\nEncoding rate')],
                                         out_dir=out_dir, service='YouTube',
                                         fs_legend='small', plot_all_x=False,
                                         plot_ccdf=False)
    LOG.info('YouTube ADSL mean rates per prefix done')
    LOG.info('process YouTube losses')
    filter_streaming.plot_indic(get_args(youtube_flows, get_loss),
                                'Backbone Loss Rate per Flow',
                                'in percent',
                                scale_loss, out_dir, 'YouTube',
                                loc='lower right', fs_legend='small',
                                plot_line=[(2, '2\%')],
                                plot_all_x=False, plot_ccdf=False)
    LOG.info('YouTube losses done')
    if not compat:
        LOG.info('process YouTube wasted bytes')
        youtube_flows_good = dict([(k, v.compress(v['good'] == True))
                                   for (k, v) in youtube_flows.items()])
        filter_streaming.plot_indic(get_args(youtube_flows_good,
                                             get_wasted_bytes),
                                    'wasted bytes per flow (good)',
                                    'in bytes',
                                    (lambda x: x), out_dir, 'YouTube',
                                    fs_legend='medium',
                                    plot_all_x=True, plot_ccdf=False,
                                    legend_ncol=2)
        LOG.info('YouTube wasted bytes done')
        LOG.info('process YouTube wasted fraction')
        test_args = get_args(youtube_flows_good, get_fraction_wasted_bytes)
        #import pdb; pdb.set_trace()
        filter_streaming.plot_indic(test_args,
                                    'Fraction of Wasted Bytes per Flow',
                                    'in percent',
                                    scale_fract_wasted, out_dir, 'YouTube',
                                    fs_legend='medium',
                                    loc=0,
                                    plot_all_x=True, plot_ccdf=False,
                                    legend_ncol=2)
        del(youtube_flows_good)
        LOG.info('YouTube wasted fraction done')
        LOG.info('process YouTube downloaded vol')
        filter_streaming.plot_indic(get_args(youtube_flows, get_downloaded_vol),
                                    'Downloaded Duration per flow',
                                    'in percent',
                                    scale_downloaded, out_dir, 'YouTube',
                                    loc='upper left', fs_legend='medium',
                                    plot_all_x=True, plot_ccdf=False,
                                    legend_ncol=2)
        LOG.info('YouTube downloaded vol done')
        LOG.debug('process YouTube downloaded per AS and quality for 2010/02 FTTH')
        trace = '2010_02_07_FTTH'
        data = youtube_flows[trace]
        # HARDCODED
        yt_as_list = (43515, 15169, 1273)
        first_both = (set(data.compress(data['asBGP'] == 1273)['client_id'])
                        .intersection(
                        set(data.compress(data['asBGP'] == 43515)['client_id'])))
        second_both = (set(data.compress(data['asBGP'] == 15169)['client_id'])
                        .intersection(
                        set(data.compress(data['asBGP'] == 43515)['client_id'])))
        clients_both = first_both.union(second_both)
        flows_both = data.compress([x['client_id'] in clients_both for x in data])
        good_both = flows_both.compress(flows_both['good']==True)
        bad_both = flows_both.compress(flows_both['good']==False)
        args = ([('Good: AS %d' % as_nb,
                  map(get_downloaded_vol,
                      good_both.compress(good_both['asBGP']==as_nb)))
                 for as_nb in yt_as_list]
                + [('Bad: AS %d' % as_nb,
                    map(get_downloaded_vol,
                        bad_both.compress(bad_both['asBGP']==as_nb)))
                   for as_nb in yt_as_list])
        # filter not enough samples
        args = [(name, data) for name, data in args if len(data) > MIN_NB_SAMPLES]
        filter_streaming.plot_indic(args,
                                    '''Downloaded Duration per AS
and Reception Quality for %s''' % (trace + '_M').replace('_', ' '),
                                    'in percent', scale_downloaded, out_dir,
                                    'YouTube',
                                    legend_ncol=2, dashes=True, plot_all_x=True,
                                    plot_ccdf=False, subplot_top=0.85)
        del(args)
        LOG.debug('YouTube downloaded per AS and quality done')
    LOG.info('process YouTube downloaded per AS')
    filter_streaming.plot_per_as(youtube_flows, '2010_02_07_FTTH',
                                 get_downloaded_vol,
                                 (INDEX_VALUES.AS_GOOGLE
                                  + INDEX_VALUES.AS_YOUTUBE_EU
                                  + INDEX_VALUES.AS_CABLE_WIRELESS),
                                 'Downloaded Duration per flow',
                                 'in percent',
                                 scale_downloaded, loc='upper left',
                                 out_dir=out_dir, service='YouTube',
                                 plot_all_x=True, plot_ccdf=False)
    filter_streaming.plot_per_as(youtube_flows, '2010_02_07_ADSL_R',
                                 get_downloaded_vol,
                                 (INDEX_VALUES.AS_GOOGLE
                                  + INDEX_VALUES.AS_YOUTUBE_EU
                                  + INDEX_VALUES.AS_CABLE_WIRELESS),
                                 'Downloaded Duration per flow',
                                 'in percent',
                                 scale_downloaded, loc='upper left',
                                 out_dir=out_dir, service='YouTube',
                                 plot_all_x=True, plot_ccdf=False)
    LOG.info('YouTube downloaded per AS done')
    if compat:
        LOG.info('process YouTube loss per AS')
        filter_streaming.plot_per_as(youtube_flows, '2011_05_03_ADSL_R',
                                     get_loss,
                                     (INDEX_VALUES.AS_GOOGLE +
                                      INDEX_VALUES.AS_YOUTUBE_EU +
                                      INDEX_VALUES.AS_CABLE_WIRELESS),
                                     'Backbone Loss Rate per flow', 'in percent',
                                     scale_loss, plot_line=[(2, '2\%')],
                                     out_dir=out_dir, service='YouTube',
                                     plot_all_x=False, plot_ccdf=False)
        LOG.info('YouTube loss per AS done')
    LOG.info('process YouTube mean rate per AS')
    filter_streaming.plot_per_as(youtube_flows, '2010_02_07_ADSL_R',
                                 get_mean,
                                 (INDEX_VALUES.AS_GOOGLE
                                  + INDEX_VALUES.AS_YOUTUBE_EU
                                  + INDEX_VALUES.AS_CABLE_WIRELESS),
                                 'Mean Rate per flow', 'in kb/s',
                                 scale_yt_mean_as, loc='lower right',
                                 plot_line=[(323, 'Median\nEncoding rate')],
                                 out_dir=out_dir, service='YouTube',
                                 plot_all_x=False, plot_ccdf=False)
    if compat:
        filter_streaming.plot_per_as(youtube_flows, '2011_05_03_ADSL_R',
                                     get_mean,
                                     (INDEX_VALUES.AS_GOOGLE
                                      + INDEX_VALUES.AS_YOUTUBE_EU
                                      + INDEX_VALUES.AS_CABLE_WIRELESS),
                                     'Mean Rate per flow', 'in kb/s',
                                     scale_yt_mean_as, loc='lower right',
                                     plot_line=[(323, 'Median\nEncoding rate')],
                                     out_dir=out_dir, service='YouTube',
                                     plot_all_x=False, plot_ccdf=False)
    LOG.info('YouTube mean rate per AS done')
    if not compat:
        LOG.info('process YouTube video rates')
        filter_streaming.plot_indic(get_args(youtube_flows, get_video_rate),
                                    'Encoding Rate of Video', 'in kb/s',
                                    scale_video_rate, out_dir, 'YouTube',
                                    loc='lower right', legend_ncol=1,
                                    plot_line=[(323, 'Median\nRate')],
                                    plot_all_x=True, plot_ccdf=False)
        LOG.info('YouTube video rates done')
    del(youtube_flows)
    youtube_ftth = filter_streaming.filter_service(flows_ftth, 'YouTube')
    if not fast:
        LOG.info('process YouTube FTTH bursts')
        filter_streaming.plot_indic(get_args(youtube_ftth, get_burst),
                                    'Loss Burstiness per flow', '',
                                    scale_burst, out_dir, 'YouTube FTTH',
                                    loc='best',
                                    plot_all_x=True, plot_ccdf=False)
        LOG.info('YouTube FTTH bursts done')
        LOG.info('process YouTube FTTH bursts per AS')
        filter_streaming.plot_per_as(youtube_ftth,  '2010_02_07_FTTH',
                                     get_burst,
                                     (INDEX_VALUES.AS_GOOGLE
                                      + INDEX_VALUES.AS_YOUTUBE_EU
                                      + INDEX_VALUES.AS_CABLE_WIRELESS),
                                    'Loss Burstiness per flow', '',
                                     scale_burst, loc='best',
                                     out_dir=out_dir, service='YouTube',
                                     plot_all_x=True, plot_ccdf=False)
        LOG.info('YouTube FTTH burst per AS done')
    if compat:
        LOG.info('process YouTube FTTH RTT per prefix')
        filter_streaming.plot_per_prefix(youtube_ftth, '2011_05_03_FTTH',
                                         get_rtt_compat,
                                         'Min RTT per flow', 'in mili-seconds',
                                         (lambda x: x), loc='lower right',
                                         mask_length=24, ip_field='FlowIPDest',
                                         fs_legend='small',
                                         out_dir=out_dir, service='YouTube',
                                         plot_all_x=False, plot_ccdf=False)
        LOG.info('YouTube FTTH RTT per prefix done')
        LOG.info('process YouTube FTTH mean rates per prefix')
        filter_streaming.plot_per_prefix(youtube_ftth, '2011_05_03_FTTH',
                                         get_mean,
                                         'Mean Rate per flow', 'in kb/s',
                                         scale_yt_mean_as, loc='upper left',
                                         mask_length=24,
                                         ip_field='FlowIPDest',
                                     plot_line=[(323, 'Median\nEncoding rate')],
                                         out_dir=out_dir, service='YouTube',
                                         fs_legend='small', plot_all_x=False,
                                         plot_ccdf=False)
        LOG.info('YouTube FTTH mean rates per prefix done')
    LOG.info('process YouTube FTTH rtt per AS')
    if compat:
        filter_streaming.plot_per_as(youtube_ftth, '2011_05_03_FTTH',
                                     get_rtt_compat,
                                     (INDEX_VALUES.AS_GOOGLE
                                      + INDEX_VALUES.AS_YOUTUBE_EU),
                                     'Min RTT per flow', 'in mili-seconds',
                                     (lambda x: x), loc='center',
                                     legend_ncol=2,
                                     out_dir=out_dir, service='YouTube',
                                     plot_all_x=False, plot_ccdf=False)
    filter_streaming.plot_per_as(youtube_ftth, '2009_12_14_FTTH',
                                 get_rtt,
                                 (INDEX_VALUES.AS_GOOGLE
                                  + INDEX_VALUES.AS_YOUTUBE_EU),
                                 'Min RTT per flow', 'in mili-seconds',
                                 scale_rtt, loc='lower center',
                                 out_dir=out_dir, service='YouTube',
                                 plot_all_x=False, plot_ccdf=False)
    LOG.info('YouTube FTTH rtt per AS done')
    LOG.info('process YouTube loss per AS')
    if compat:
        filter_streaming.plot_per_as(youtube_ftth, '2011_05_03_FTTH',
                                     get_loss,
                                     (INDEX_VALUES.AS_GOOGLE +
                                      INDEX_VALUES.AS_YOUTUBE_EU +
                                      INDEX_VALUES.AS_CABLE_WIRELESS),
                                     'Backbone Loss Rate per flow',
                                     'in percent',
                                     scale_loss, plot_line=[(2, '2\%')],
                                     out_dir=out_dir, service='YouTube',
                                     plot_all_x=False, plot_ccdf=False)
    filter_streaming.plot_per_as(youtube_ftth, '2010_02_07_FTTH',
                                 get_loss,
                                 (INDEX_VALUES.AS_GOOGLE +
                                  INDEX_VALUES.AS_YOUTUBE_EU +
                                  INDEX_VALUES.AS_CABLE_WIRELESS),
                                 'Backbone Loss Rate per flow', 'in percent',
                                 scale_loss, plot_line=[(2, '2\%')],
                                 out_dir=out_dir, service='YouTube',
                                 plot_all_x=False, plot_ccdf=False)
    LOG.info('YouTube loss per AS done')
    LOG.info('process YouTube FTTH mean rates')
    filter_streaming.plot_indic(get_args(youtube_ftth, get_mean),
                                'Mean Rate per flow', 'in kb/s',
                                scale_yt_mean, out_dir, 'YouTube',
                                loc='lower right',
                                plot_line=[(323, 'Median\nEncoding rate')],
                                plot_all_x=False, plot_ccdf=False)
    LOG.info('YouTube mean rates done')
    LOG.info('process YouTube FTTH mean rates per AS')
    if compat:
        filter_streaming.plot_per_as(youtube_ftth, '2011_05_03_FTTH',
                                     get_mean,
                                     (INDEX_VALUES.AS_GOOGLE
                                      + INDEX_VALUES.AS_YOUTUBE_EU
                                      + INDEX_VALUES.AS_CABLE_WIRELESS),
                                     'Mean Rate per flow', 'in kb/s',
                                     scale_yt_mean_as, loc='lower right',
                                     plot_line=[(323, 'Median\nEncoding rate')],
                                     out_dir=out_dir, service='YouTube',
                                     plot_all_x=False, plot_ccdf=False)
    filter_streaming.plot_per_as(youtube_ftth, '2010_02_07_FTTH',
                                 get_mean,
                                 (INDEX_VALUES.AS_GOOGLE
                                  + INDEX_VALUES.AS_YOUTUBE_EU
                                  + INDEX_VALUES.AS_CABLE_WIRELESS),
                                 'Mean Rate per flow', 'in kb/s',
                                 scale_yt_mean_as, loc='lower right',
                                 plot_line=[(323, 'Median\nEncoding rate')],
                                 out_dir=out_dir, service='YouTube',
                                 plot_all_x=False, plot_ccdf=False)
    LOG.info('YouTube FTTH mean rates per AS done')
    if not compat:
        LOG.info('process YouTube peak')
        filter_streaming.plot_indic(get_args(youtube_ftth, get_peak),
                                    'Peak Rate per flow', 'in kb/s',
                                    scale_yt_peak, out_dir, 'YouTube',
                                    loc='upper left',
                                    plot_all_x=False, plot_ccdf=False)
        LOG.info('YouTube peak done')
        LOG.info('process YouTube peak AS')
    #    filter_streaming.plot_per_as(youtube_ftth, '2010_02_07_FTTH',
    #                                 get_peak, (INDEX_VALUES.AS_ALL_GOOGLE),
    #                                 'Peak Rate', 'in kb/s', scale_yt_peak_as,
    #                                 out_dir=out_dir, service='YouTube')
        filter_streaming.plot_per_as(youtube_ftth, '2009_12_14_FTTH',
                                     get_peak, (INDEX_VALUES.AS_ALL_GOOGLE),
                                     'Peak Rate', 'in kb/s', scale_yt_peak_as,
                                     out_dir=out_dir, service='YouTube',
                                     plot_all_x=False, plot_ccdf=False)
    if not fast:
        yt_peak_rate_per_client = (
            filter_streaming.retrieve_peak_rate_per_client(youtube_ftth))
        filter_streaming.plot_indic(sorted(yt_peak_rate_per_client),
                                    'Peak Rate per Client', 'in kb/s',
                                    (lambda (xmin, xmax): (4e2, 4e4)),
                                    out_dir, 'YouTube',
                                    loc='upper left',
                                    plot_all_x=False, plot_ccdf=False)
        LOG.info('YouTube client done')
    del(youtube_ftth)
    LOG.info('YouTube done')
    LOG.info('start other providers')
    non_youtube_ftth = filter_streaming.filter_service(flows_ftth, 'YouTube',
                                                        excluded=True)
    non_yt_dm_ftth = filter_streaming.filter_service(non_youtube_ftth,
                                                     'DailyMotion',
                                                     excluded=True)
    del(non_youtube_ftth)
    if not compat:
        filter_streaming.plot_indic(get_args(non_yt_dm_ftth, get_peak),
                                    'Peak Rate per flow', 'in kb/s',
                                    scale_other, out_dir, 'Other Providers',
                                    loc='upper left',
                                    plot_all_x=False, plot_ccdf=False)
        LOG.info('other provider peak rate per flow done')
        LOG.info('process other provider per AS')
        as_list_other_nov = [3356, 16265, 16276, 22822]
        filter_streaming.plot_per_as(non_yt_dm_ftth, '2009_11_26_FTTH',
                                     get_peak, as_list_other_nov,
                                     'Peak Rate', 'in kb/s', scale_other,
                                     out_dir=out_dir, service='Other Providers',
                                     plot_all_x=False, plot_ccdf=False)
        as_list_other_dec = [5511, 39572, 16276, 16265, 22822]
        filter_streaming.plot_per_as(non_yt_dm_ftth, '2009_12_14_FTTH',
                                     get_peak, as_list_other_dec,
                                     'Peak Rate', 'in kb/s', scale_other,
                                     out_dir=out_dir, service='Other Providers',
                                     plot_all_x=False, plot_ccdf=False)
        LOG.info('filtered_stats on FTTH done')
#    LOG.info('process remaining download')
#    remaining_data = dict([(k, extract_remaining_list(v))
#                           for k, v in filtered_flows.items()])
#    complements.plot_remaining_download(remaining_data,
#                                        plot_excluded=False,
#                                        prefix='remaining_time_mix_dm_goo',
#                                        out_dir=out_dir,
#                                        good_indic='on all services',
#                                        loglog=True, logx=True, th=None)
#    del(remaining_data)
#    LOG.info('remaining download done')
    return 0

if __name__ == "__main__":
    sys.exit(main(fast=False))

