#!/usr/bin/env python
"""Module to filter the flows indicator to retrieve in one numpy array all
relevant indicatiors
"""

from __future__ import division, print_function, absolute_import
from collections import defaultdict
from os import sep, getcwd
from operator import itemgetter, concat, eq, ne
from functional import compose
from itertools import cycle, groupby
import cPickle
import sys
import numpy as np
# in case of non-interactive usage
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

import logging
LOG_LEVEL = logging.DEBUG

if __name__ == "__main__" and __package__ is None:
    __package__ = "tools"
    import tools

from .load_dipcp_file import (filter_dipcp_array, return_cnx_id_GVB,
                             return_cnx_id_dipcp)
from .flow2session import ip2int
from . import INDEX_VALUES
from . import cdfplot
from . import cdfplot_2
from . import streaming_tools
from . import complements
from .INDEX_VALUES import (TIME_START, TIME_STOP, ACCESS_DEFAULT_RATES,
                           ACCESS_RATES_NB_RENNES_2009_12,
                           #ACCESS_RATES_NB_RENNES_2010_02,
                           #ACCESS_RATES_NB_MONT_2009_11,
                           ACCESS_RATES_NB_MONT_2008_07)

def cnx_id_str_qual(str_qual_flow):
    "Return the connection ID of stream_quality flow."
    return (str_qual_flow['srcAddr'], str_qual_flow['dstAddr'])

def cnx_id_cnx_stream(cnx_stream_flow):
    "Return the connection ID of cnx_stream flow."
    return (cnx_stream_flow['RemAddr'], cnx_stream_flow['LocAddr'])

def create_hash_map_cnx(cur_cnx):
    "Upload cnx id of str qual or cnx str into a dict"
    if ((cur_cnx.dtype == INDEX_VALUES.dtype_cnx_stream)
        or (cur_cnx.dtype == INDEX_VALUES.dtype_cnx_stream_loss)):
        cnx_id_func = cnx_id_cnx_stream
    elif (cur_cnx.dtype == INDEX_VALUES.dtype_GVB_streaming_AS):
        cnx_id_func = cnx_id_str_qual
    else:
        print("Undefined dtype for creating cnx id")
        return None
    hash_cnx = {}
    for cnx_flow in cur_cnx:
        # no dscp match (index 5)
        hash_cnx[cnx_id_func(cnx_flow)] = True
    return hash_cnx

def date(trace):
    """Return the date of capture out of the name of trace
    >>> date('2010_02_07_FTTH')
    '07/02/2010'
    """
    return '/'.join(reversed(trace.split('_')[:3]))

def filter_time_cnx_str(cur_cnx, trace):
    "Return an array of cnx_stream filtered on time in the stream_qual flows"
# not (x['EndTime'] < TIME_START[trace] or x['StartTime'] > TIME_STOP[trace])
    if trace == '2008_07_01_FTTH':
        # hack because of reconstructed cnx_stream file
        return cur_cnx
    return cur_cnx.compress([(x['EndTime'] >= TIME_START[trace]
                              and x['StartTime'] <= TIME_STOP[trace]
                              and x['Date'] == date(trace))
                             for x in cur_cnx])

def assign_service(cur_cnx):
    """Assign the value of service out of url (host referer)
    Useful for 2008_07_01_FTTH trace because recontructed cnx_stream file
    Work with side effect assignment
    """
    for flow in cur_cnx:
        if 'youtube' in flow['Host_Referer']:
            flow['Service'] = 'YouTube'
        elif 'dailymotion' in flow['Host_Referer']:
            flow['Service'] = 'DailyMotion'
        elif ('megaupload' in flow['Host_Referer']
              or 'megavideo' in flow['Host_Referer']):
            flow['Service'] = 'Megaupload'
        elif ('porn' in flow['Host_Referer'] or 'adult' in flow['Host_Referer']
              or 'empflix' in flow['Host_Referer']
              or 'sex' in flow['Host_Referer']
              or 'tube8' in flow['Host_Referer']
              or 'xvideo' in flow['Host_Referer']):
            flow['Service'] = 'DiversX'
        else:
            flow['Service'] = 'Unknown'

def assign_name(cur_cnx):
    """Assign the Name field as Name1_Name2 (because of reconstructed
    cnx_stream file)
    Work with side effect assignment
    """
    for flow in cur_cnx:
        if flow['Name'] == '0':
            flow['Name'] = flow['client_id']

def all_equal(values):
    "Tells if all passed values are equal"
    return all([x == values[0] for x in values])

def flow_id_cnx_stream(cnx_stream_flow):
    "Return the connection ID of cnx_stream flow."
    return (cnx_stream_flow['RemAddr'], cnx_stream_flow['RemPort'],
            cnx_stream_flow['LocAddr'], cnx_stream_flow['LocPort'])

def get_cnx_str(best_match, filter_cnx_str):
    "Return an array of cnx_str corresponding to the best match cnx"
    return filter_cnx_str.compress([flow_id_cnx_stream(x) ==
                                    flow_id_cnx_stream(best_match)
                                    for x in filter_cnx_str])

def compress_possible(best_match):
    """Aggregate the list of chunks into one
    """
    tmp = best_match.compress(best_match['ByteDn']
                              == max(best_match['ByteDn']))
    return tmp.compress(tmp['DurationDn'] == max(tmp['DurationDn']))[0]

def is_same_cnx(filter_cnx_str):
    "Tells if all filtered flows belong to the same connection"
    return all_equal(zip(filter_cnx_str['LocAddr'], filter_cnx_str['LocPort'],
                         filter_cnx_str['RemAddr'], filter_cnx_str['RemPort']))

def construct_hash_tstat(flow, output=None):
    "Return a dict indexed on cnx_id of tstats indics"
    if not output:
        output = defaultdict(list)
    for indics in flow:
        output[(indics['Client_IP_address'],
                indics['Client_TCP_port'],
                indics['Server_IP_address'],
                indics['Server_TCP_port'])].append(indics)
    return output

def construct_hash_cnx_stream(flow, output=None):
    "Return a dict indexed on cnx_id of tstats indics"
    if not output:
        output = defaultdict(list)
    for indics in flow:
        # loc port is unreliable in cnx_stream!
        output[(indics['LocAddr'],
                #indics['LocPort'],
                indics['RemAddr'],
                indics['RemPort'])].append(indics)
    return output

def find_tstat_info(cnx_str, flow, debug=False):
    "Return an array of cnx_str augmented by tstats informations"
    flow_dict = construct_hash_tstat(flow)
    output = []
    nb_not_found = 0
    nb_tot = 0
    for data in cnx_str:
        nb_tot += 1
        cnx_id = None
        if (data['srcAddr'], data['srcPort'],
            data['dstAddr'], data['dstPort']) in flow_dict:
            cnx_id = (data['srcAddr'], data['srcPort'],
                      data['dstAddr'], data['dstPort'])
        elif (data['dstAddr'], data['dstPort'],
              data['srcAddr'], data['srcPort']) in flow_dict:
            cnx_id = (data['dstAddr'], data['dstPort'],
                      data['srcAddr'], data['srcPort'])
        else:
            nb_not_found += 1
            if debug:
                print("Stream not found in tstat",
                      (data['dstAddr'], data['dstPort'],
                       data['srcAddr'], data['srcPort']))
        if cnx_id:
            if len(flow_dict[cnx_id]) > 1:
                print("Multiple flows for this stream", cnx_id)
            output.append(tuple(list(data) + list(flow_dict[cnx_id][0])))
    print('nb_tot, nb_not_found', nb_tot, nb_not_found)
    return np.array(output,
                    dtype=INDEX_VALUES.dtype_all_stream_indics_final_tstat)

def load_all_tstat(input_dir='flows/tstat'):
    "Return a dict with all tstats files loaded"
    output = dict()
    for trace in INDEX_VALUES.TRACE_LIST:
        log_tcp_complete = np.load(sep.join((input_dir, trace,
                                             'log_tcp_complete.npy')))
        log_tcp_nocomplete = np.load(sep.join((input_dir, trace,
                                               'log_tcp_nocomplete.npy')))
        output[trace] = np.array(list(log_tcp_complete)
                                 + list(log_tcp_nocomplete),
                       dtype=INDEX_VALUES.dtype_all_stream_indics_final_tstat)
    return output

def add_tstat_cnx_str(cnx_str, input_dir='flows/tstat'):
    "Return a new dict with tstats indics added to streams"
    output = dict()
    for trace in INDEX_VALUES.TRACE_LIST:
        # load tstats files separately to spare memory
        print('Processing trace: ', trace)
        log_tcp_complete = np.load(sep.join((input_dir, trace,
                                             'log_tcp_complete.npy')))
        log_tcp_nocomplete = np.load(sep.join((input_dir, trace,
                                               'log_tcp_nocomplete.npy')))
        output[trace] = find_tstat_info(cnx_str[trace],
                                        np.array(list(log_tcp_complete)
                                                 + list(log_tcp_nocomplete),
                       dtype=INDEX_VALUES.dtype_tstat))
    return output

def find_cnx_str(filtered_cnx_str, str_qual_flow, start_time, log,
                 th_diff_time=20, strict=True):
    "Return corresponding cnx stream out of heuristics on stream qual stats"
    # return the max download in case of same cnx
    if is_same_cnx(filtered_cnx_str):
        log['disambiguated through single connection match'] += 1
        #tmp = filtered_cnx_str.compress(filtered_cnx_str['ByteDn']
                                         #== max(filtered_cnx_str['ByteDn']))
        #return tmp.compress(tmp['DurationDn'] == max(tmp['DurationDn']))[0]
        return compress_possible(filtered_cnx_str)
    best_match = filtered_cnx_str[0]
    init_time = int(str_qual_flow['initTime']) + start_time
    min_time_diff = abs(init_time - int(best_match['StartTime']))
    for cnx_str_flow in filtered_cnx_str:
        if min_time_diff > abs(init_time - int(cnx_str_flow['StartTime'])):
            best_match = cnx_str_flow
            min_time_diff = abs(init_time - int(cnx_str_flow['StartTime']))
    if min_time_diff < th_diff_time:
        possible_match = get_cnx_str(best_match, filtered_cnx_str)
        filtered_possible = possible_match.compress(possible_match['DurationDn']
                                        == max(possible_match['DurationDn']))
        if len(filtered_possible) == 1:
            log['disambiguated through time and max ByteDn'] += 1
            best_match = filtered_possible[0]
        else:
            # silverlight chunks
            log['silverlight flow'] += 1
            best_match = (compress_possible(filtered_possible) if (not strict)
                          else None)
    else:
        log['unpossible to match'] += 1
        best_match = None
    return best_match

def filter_ad_late(cur_cnx, url_file_name='ads_url.txt',
                   urn_file_name='ads_urn.txt'):
    """Return a new array of cnx_str with Advertisement filtered out
    to be run afterwards to filter out based on host name
    """
    with open(url_file_name) as url_file:
        urls = [x.translate(None, '/\\') for x in
                np.loadtxt(url_file, dtype=str)]
    with open(urn_file_name) as urn_file:
        urns = [x.translate(None, '/\\') for x in
                np.loadtxt(urn_file, dtype=str)]
    filtered_cnx = []
    for flow in cur_cnx:
        is_ad = False
        for url in urls:
            if url in flow['Host_Referer']:
                is_ad = True
        for urn in urns:
            if urn in flow['RemName']:
                is_ad = True
        if not is_ad:
            # it was not caught by url or urn
            filtered_cnx.append(flow)
    return np.array(filtered_cnx, dtype=cur_cnx.dtype)

def filter_ad_cnx_str(cur_cnx):
    "Return a new array of cnx_str with Advertisement filtered out"
    return cur_cnx.compress(cur_cnx['Type'] != 'Advertisement')

def filter_id_cnx_str(cur_cnx):
    "Return a new array of cnx_str with unknown clients filtered out"
    return cur_cnx.compress([x['Name'] not in INDEX_VALUES.UNKNOWN_ID
                             for x in cur_cnx])

def filter_player_cnx_str(cur_cnx):
    "Return a new array of cnx_str with player flows filtered out"
    return cur_cnx.compress(['player' not in x['RemName']
                             for x in cur_cnx])

def filter_valid_str_qual(cur_str_qual, strict=True):
    """Return a new array of str_qual with only valid flows having positive
    duration
    """
    return cur_str_qual.compress([x['Content-Length'] > 0 and
                                  ((x['valid'] == 'OK') if strict else True)
                                  for x in cur_str_qual])

def match_str_cnx(cur_cnx, hash_cnx):
    "Return a new array of str_qual for those matched in cnx_str"
    if ((cur_cnx.dtype == INDEX_VALUES.dtype_cnx_stream)
        or (cur_cnx.dtype == INDEX_VALUES.dtype_cnx_stream_loss)):
        cnx_id_func = cnx_id_cnx_stream
    elif (cur_cnx.dtype == INDEX_VALUES.dtype_GVB_streaming_AS):
        cnx_id_func = cnx_id_str_qual
    else:
        print("Undefined dtype for creating cnx id")
        return cur_cnx
    return cur_cnx.compress([cnx_id_func(x) in hash_cnx
                             for x in cur_cnx])

def mix_flows_stats(str_qual_flow, cnx_str_flow, loss=True):
    """Return a tuple of interesting values from a str qual flow and a cnx str
    flow
    """
    return (
        # from cnx_stream
        cnx_str_flow['Name'],
        cnx_str_flow['LocPort'],
        cnx_str_flow['RemPort'],
        cnx_str_flow['Date'],
        cnx_str_flow['StartTime'],
        cnx_str_flow['Host_Referer'],
        cnx_str_flow['RemName'],
        cnx_str_flow['StartDn'],
        cnx_str_flow['DurationDn'],
        cnx_str_flow['ByteDn'],
        cnx_str_flow['PacketDn'],
        cnx_str_flow['Type'],
        cnx_str_flow['LostDn'] if loss else 0,
        cnx_str_flow['DesyncDn'] if loss else 0,
        cnx_str_flow['Application'],
        cnx_str_flow['Service'],
        loss,
        # from str_stats
        str_qual_flow['srcAddr'],
        str_qual_flow['dstAddr'],
        str_qual_flow['initTime'],
        str_qual_flow['Content-Type'],
        str_qual_flow['Content-Length'],
        str_qual_flow['Content-Duration'],
        str_qual_flow['Content-Avg-Bitrate-kbps'],
        str_qual_flow['Session-Bytes'],
        str_qual_flow['Session-Pkts'],
        str_qual_flow['Session-Duration'],
        str_qual_flow['nb_skips'],
        str_qual_flow['valid'],
        str_qual_flow['asBGP'])

def retrieve_str_qual_cnx_str(matching_str_qual, matching_cnx_str, start_time,
                              strict=True):
    """Return a new array mixing interesting features of str qual and cnx str
    flows
    """
    # loss data available
    loss = (matching_cnx_str.dtype == INDEX_VALUES.dtype_cnx_stream_loss)
    log = defaultdict(int)
    found_cnx_str = [] #np.array([], dtype=INDEX_VALUES.dtype_all_stream_indics)
    for str_qual_flow in matching_str_qual:
        filtered_cnx_flows = matching_cnx_str.compress(
            [(x['RemAddr'] == str_qual_flow['srcAddr'] and
              x['LocAddr'] == str_qual_flow['dstAddr'])
             for x in matching_cnx_str])
        if len(filtered_cnx_flows) == 1:
            log['find only 1'] += 1
            found_cnx_str.append(mix_flows_stats(str_qual_flow,
                                                 filtered_cnx_flows[0],
                                                 loss=loss))
        else:
            log['mult cnx str'] += 1
            cnx_str_flow = find_cnx_str(filtered_cnx_flows, str_qual_flow,
                                        start_time, log, strict=strict)
            if not cnx_str_flow:
                continue
            found_cnx_str.append(mix_flows_stats(str_qual_flow, cnx_str_flow,
                                                 loss=loss))
    print('\n'.join(['\t\t%s: %d' % (k, v) for k, v in sorted(log.items())]))
    return np.array(found_cnx_str, dtype=INDEX_VALUES.dtype_stream_indics_tmp)

def extract_remaining_list(final_str_qual):
    "Return a list of remaining stats info"
    return [(x['Content-Length'], (100 * x['ByteDn'] / x['Content-Length']),
             x['asBGP'], x['valid'], x['Host_Referer'], x['good'])
            for x in final_str_qual]

def filter_cnx_stream_qual(cur_cnx_str, cur_str_qual, trace, strict=True,
                           check_id=False):
    "Return a list of data for remaining time combining cnx_stream and str_qual"
    if __debug__:
        print('processing trace', trace)
        print('\tcnx str flows initial nb', len(cur_cnx_str))
    # 2010 traces have identification problem...
    if check_id:
        filtered_id_cnx_str = filter_id_cnx_str(cur_cnx_str)
    else:
        filtered_id_cnx_str = cur_cnx_str
    if __debug__:
        print('\tcnx str flows with correct id', len(filtered_id_cnx_str))
    filtered_ad_cnx_str = filter_ad_cnx_str(filtered_id_cnx_str)
    if __debug__:
        print('\tcnx str flows without ads', len(filtered_ad_cnx_str))
    filtered_player_cnx_str = filter_player_cnx_str(filtered_ad_cnx_str)
    if __debug__:
        print('\tcnx str flows without player', len(filtered_player_cnx_str))
    filtered_cnx_str = filter_time_cnx_str(filtered_player_cnx_str, trace)
    if __debug__:
        print('\tcnx str flows at corresponding time', len(filtered_cnx_str))
    if __debug__:
        print('\tstr qual flows initial nb', len(cur_str_qual))
    filtered_str_qual = filter_valid_str_qual(cur_str_qual, strict=strict)
    if __debug__:
        print('\tstr qual valid flows', len(filtered_str_qual))
    hash_cnx_str = create_hash_map_cnx(filtered_cnx_str)
    matching_str_qual = match_str_cnx(filtered_str_qual, hash_cnx_str)
    if __debug__:
        print('\tmatched str qual', len(matching_str_qual))
    hash_str_qual = create_hash_map_cnx(matching_str_qual)
    matching_cnx_str = match_str_cnx(filtered_cnx_str, hash_str_qual)
    if __debug__:
        print('\tre-matched cnx str', len(matching_cnx_str))
    return retrieve_str_qual_cnx_str(matching_str_qual, matching_cnx_str,
                                     TIME_START[trace], strict=strict)

def generate_filtered_flows(cnx_stream, stream_qual, strict=True,
                            check_id=False):
    """Filter flows from cnx_stream
    USE AS:
cnx_stream = tools.streaming_tools.load_cnx_stream()
stream_qual = tools.complements.load_stream_qual()
filtered_flows = tools.filter_streaming.generate_filtered_flows(cnx_stream,
    stream_qual)
    PICKLE:
filtered_flows = cPickle.load(open('filtered_correct_streaming.pickle'))
    CHECK DATA:
[(k, len(v), [cdn + ': ' + str(len([x for x in v if x['Service']==cdn]))
    for cdn in np.unique(v['Service'])])
    for k,v in sorted(filtered_flows.items())]
    """
    return dict([(k, filter_cnx_stream_qual(cnx_stream[k], stream_qual[k], k,
                                            strict=strict, check_id=check_id))
                 for k in cnx_stream])

def rename_dtype(filtered_flows,
                 reorder_dtype=INDEX_VALUES.dtype_all_stream_indics_reorder,
                 new_dtype=INDEX_VALUES.dtype_all_stream_indics_rename):
    "Rename and reorder some fields"
    extract_fields = itemgetter(*map(itemgetter(0), reorder_dtype))
    return dict([(k, np.array(map(extract_fields, v), dtype=new_dtype))
                 for k, v in filtered_flows.items()])

def is_good_stream(flow):
    "Return if the mean rate is higher than encoding rate"
    return ((8e-3 * flow['ByteDn'] / flow['DurationDn'])
            > (8e-3 * flow['Content-Length'] / flow['Content-Duration'])
            if flow['DurationDn'] > 0 else False)

def extend_stream_with_qual(flows):
    """Adds information on stream quality out of encoding rate and mean rate
    """
    assert flows.dtype == INDEX_VALUES.dtype_all_stream_indics_final
    return np.array([tuple(v) + (is_good_stream(v),) for v in flows],
                    dtype=INDEX_VALUES.dtype_all_stream_indics_final_good)

def generate_flows_with_qual(cnx_stream):
    """Generate stream dict with information on stream quality out of encoding
    rate and mean rate
    """
    return dict([(k, extend_stream_with_qual(v))
                 for k, v in cnx_stream.items()])

def is_almost_same_time(ts_1, ts_2, diff=10):
    "Return True if the time stamps are distant from less than diff"
    return abs(ts_2 - ts_1) < diff

def generate_dipcp_filtered_flows(flows_gvb, datas, new_dtype=None,
                                  augment_list=('DIP-RTT-NbMes-ms-TCP-Up',
                                                'DIP-RTT-NbMes-ms-TCP-Down',
                                                'DIP-RTT-Mean-ms-TCP-Up',
                                                'DIP-RTT-Mean-ms-TCP-Down',
                                                'DIP-RTT-Min-ms-TCP-Up',
                                                'DIP-RTT-Min-ms-TCP-Down',
                                                'DIP-RTT-Max-ms-TCP-Up',
                                                'DIP-RTT-Max-ms-TCP-Down')):
    """Return a dict of arrays completed with dipcp stats
    USE AS:
filtered_gvb = cPickle.load(open('filtered_flows_complete_gvb.pickle'))
datas = tools.load_hdf5_data.load_h5_file('flows/hdf5/traces_lzf.h5')
filtered_flows_dipcp_rtt = tools.filter_streaming.generate_dipcp_filtered_flows(
    filtered_gvb, datas, new_dtype=tools.INDEX_VALUES.dtype_rtt_stream_indics)
datas_mix = dict(((key, datas[key]) for key in datas
    if key.endswith('GVB')))
del(datas)
datas_loss = tools.load_hdf5_data.load_h5_file('flows/hdf5/dipcp_loss_lzf.h5')
for key in datas_loss:
    datas_mix[key] = datas_loss[key]

del(datas_loss)
filtered_flows_final = tools.filter_streaming.generate_dipcp_filtered_flows(
    filtered_flows_dipcp_rtt, datas_mix,
    tools.INDEX_VALUES.dtype_all_stream_indics,
    augment_list=('DIP-Volume-Number-Packets-Down',
                  'DIP-Volume-Number-Packets-Up',
                  'DIP-Volume-Sum-Bytes-Down',
                  'DIP-Volume-Sum-Bytes-Up',
                  'DIP-DSQ-NbMes-sec-TCP-Up',
                  'DIP-DSQ-NbMes-sec-TCP-Down',
                  'DIP-RTM-NbMes-sec-TCP-Up',
                  'DIP-RTM-NbMes-sec-TCP-Down',
                  'DIP-DST-Number-Milliseconds-Up',
                  'DIP-DST-Number-Milliseconds-Down',
                  'DIP-CLT-Number-Milliseconds-Up',
                  'DIP-CLT-Number-Milliseconds-Down'))
    FOR OTHER RTT
datas_mix = dict(((key, datas[key]) for key in datas
    if key.endswith('GVB')))
del(datas)
for f in open('flows/list_files_rtt.txt').readlines():
    datas_mix[f.split('/')[1].split('dipcp_output_')[1] + '_DIPCP'] = \
            np.load('flows/' + f.strip())

filtered_flows = tools.filter_streaming.generate_dipcp_filtered_flows(
                                                    filtered_flows, datas_mix)
    """
    if not new_dtype:
        new_dtype = INDEX_VALUES.dtype_all_stream_indics_final
    output = {}
    errors = {}
    extract_dipcp_stats = itemgetter(*augment_list)
    for trace, flow_gvb in flows_gvb.items():
        print('Processing trace: ', trace)
        errors[trace] = defaultdict(int)
        (dscp_http_stream, _, _) = streaming_tools.un_mismatch_dscp(
                                                        datas[trace + '_GVB'])
        #if 'TOS' in datas[trace + '_DIPCP'].dtype.names:
        data_dipcp = datas[trace + '_DIPCP'].compress(
            datas[trace + '_DIPCP']['TOS'] == 4 * dscp_http_stream)
        #else:
            #data_dipcp = datas[trace + '_DIPCP']
        streaming_dipcp = filter_dipcp_array(data_dipcp, flow_gvb)
        dipcp_2_rtt = {}
        for flow in streaming_dipcp:
            dipcp_2_rtt[return_cnx_id_dipcp(flow)] = extract_dipcp_stats(flow)
        output[trace] = np.array([tuple(f) + (dipcp_2_rtt[return_cnx_id_GVB(f)]
                                      if return_cnx_id_GVB(f) in dipcp_2_rtt
                                              else ((0,) * len(augment_list)))
                                  for f in flow_gvb],
                                 dtype=new_dtype)
    print(errors)
    return output

def generate_gvb_filtered_flows(flows, datas):
    """Return a dict of arrays completed with dipcp stats
    USE AS:
filtered_flows = cPickle.load(open('filtered_flows_complete.pickle'))
datas = tools.load_hdf5_data.load_h5_file('flows/hdf5/traces_lzf.h5')
filtered_flows_gvb = tools.filter_streaming.generate_gvb_filtered_flows(
    filtered_flows, datas)
    """
    output = {}
    errors = {}
    extract_cnx_id = itemgetter('dstAddr', 'srcAddr')
    extract_gvb_stats = itemgetter('protocol', 'srcPort', 'dstPort',
                                   'initTime', 'direction', 'client_id',
                                   'dscp', 'peakRate')
    for trace in flows:
        print('Processing trace: ', trace)
        errors[trace] = defaultdict(int)
        (dscp_http_stream, _, _) = streaming_tools.un_mismatch_dscp(
                                                        datas[trace + '_GVB'])
        data_gvb = datas[trace + '_GVB'].compress(datas[trace + '_GVB']['dscp']
                                              == dscp_http_stream)
        #data_dipcp = datas[trace + '_DIPCP'].compress(
                        #datas[trace + '_DIPCP']['tos'] == 4 * dscp_http_stream)
        new_flows = []
        # pass all filtered flows and check datas
        # less efficient but better to avoid multiple matches
        for flow in flows[trace]:
            cnx_id = extract_cnx_id(flow)
            init_time = flow['initTime']
            flow_gvb = data_gvb.compress([extract_cnx_id(x) == cnx_id and
                                  is_almost_same_time(x['initTime'], init_time)
                                          for x in data_gvb])
            if len(flow_gvb) == 0:
                errors[trace]['no match flow'] += 1
                continue
            elif len(flow_gvb) > 1:
                errors[trace]['multiple match flow'] += 1
                continue
            new_flows.append(tuple(flow) + extract_gvb_stats(flow_gvb[0]))
        output[trace] = np.array(new_flows,
                             dtype=INDEX_VALUES.dtype_gvb_stream_indics)
    print(errors)
    return output

def remaining_cnx_stream_qual_id(cur_cnx_str, cur_str_qual, trace,
                                 strict=True):
    "Return a list of data for remaining time combining cnx_stream and str_qual"
    final_str_qual = filter_cnx_stream_qual(cur_cnx_str, cur_str_qual, trace,
                                            strict=strict)
    return extract_remaining_list(final_str_qual)

def generate_remaining_cnx_stream(cnx_stream, stream_qual, strict=True):
    """Filter flows from cnx_stream and generate the data of remainig download
    volume in percent
    USE AS:
cnx_stream = tools.streaming_tools.load_cnx_stream()
stream_qual = tools.complements.load_stream_qual()
data_remaining = tools.filter_streaming.generate_remaining_download_cnx_stream(
    cnx_stream, stream_qual, strict=True)
tools.complements.plot_remaining_download(data_remaining,
                        as_list=('DAILYMOTION', 'ALL_GOOGLE'),
                        prefix='remaining_time_mix_dm_goo',
                        out_dir='rapport/complements/mix',
                        loglog=True, logx=False, th=None)
tools.complements.plot_remaining_download(data_remaining,
                        as_list=('DAILYMOTION', 'ALL_YOUTUBE', 'GOOGLE'),
                        prefix='remaining_time_mix_dm_yt_goo',
                        out_dir='rapport/complements/mix',
                        loglog=True, logx=False, th=None)
tools.complements.plot_remaining_download(data_remaining,
                        as_list=('DAILYMOTION', 'ALL_GOOGLE'),
                        prefix='remaining_time_mix_dm_goo',
                        out_dir='rapport/complements/mix_strict',
                        loglog=True, logx=True, th=1e6, rescale=False)
tools.complements.plot_remaining_download(data_remaining,
                        as_list=('DAILYMOTION', 'ALL_YOUTUBE', 'GOOGLE'),
                        prefix='remaining_time_mix_dm_yt_goo',
                        out_dir='rapport/complements/mix_strict',
                        loglog=True, logx=True, th=1e6, rescale=False)
    """
    return dict([(k, remaining_cnx_stream_qual_id(cnx_stream[k],
                                                  stream_qual[k],
                                                  k, strict=strict))
                 for k in cnx_stream])

def plot_indic(args, title, xlabel, rescale, out_dir, service, loc=0,
               plot_line=None, initial_clf=True, dashes=True,
               fs_legend='large', plot_ccdf=True, plot_all_x=True,
               legend_ncol=1, subplot_top=None):
    """Plot cdf of data on indic and save it in lin logx and loglog scales
    """
    save_title = sep.join((out_dir,
                           '_'.join((title.
                                 translate(None, '/()[]').strip(), service)).
                           lower().replace(' ', '_').replace('\n', '_').
                           translate(None, '.`\'"')))
    plot_title = ' '.join((service, title)).replace('_', ' ')
    if ' per' in title:
        xlabel = ' '.join((title.split(' (')[0].split(' per')[0].rstrip(),
                           xlabel))
    elif ' in fct' in title:
        xlabel = ' '.join((title.split(' (')[0].split(' in fct')[0].rstrip(),
                           xlabel))
    if not initial_clf:
        # TODO
        pass
    figure = cdfplot_2.CdfFigure(subplot_top=subplot_top)
    if plot_line:
        for x_value, label in plot_line:
            figure.plot((x_value, x_value), [0, 1],
                             linewidth=2, color='red', label=label)
    figure.cdfplotdata(args, title=plot_title, xlabel=xlabel, loc=loc,
                       dashes=dashes, fs_legend=fs_legend,
                       legend_ncol=legend_ncol)
    if rescale:
        figure.set_xlim(rescale(figure.get_xlim()))
    figure.savefig(save_title + '_logx.pdf')
    del(figure)
    if plot_all_x:
        figure = cdfplot_2.CdfFigure(subplot_top=subplot_top)
        if plot_line:
            for x_value, label in plot_line:
                figure.plot((x_value, x_value), [0, 1],
                                 linewidth=2, color='red', label=label)
        figure.cdfplotdata(args, title=plot_title, xlabel=xlabel, loc=loc,
                           logy=True, dashes=dashes, fs_legend=fs_legend,
                           legend_ncol=legend_ncol)
        if rescale:
            figure.set_xlim(rescale(figure.get_xlim()))
        figure.savefig(save_title + '_loglog.pdf')
        del(figure)
        figure = cdfplot_2.CdfFigure(subplot_top=subplot_top)
        if plot_line:
            for x_value, label in plot_line:
                figure.plot((x_value, x_value), [0, 1],
                                 linewidth=2, color='red', label=label)
        figure.cdfplotdata(args, title=plot_title, xlabel=xlabel, loc=loc,
                           logx=False, dashes=dashes, fs_legend=fs_legend,
                           legend_ncol=legend_ncol)
        if rescale:
            figure.set_xlim(rescale(figure.get_xlim()))
        figure.savefig(save_title + '_lin.pdf')
        del(figure)
    if plot_ccdf:
        figure = cdfplot_2.CdfFigure(subplot_top=subplot_top)
        if plot_line:
            for x_value, label in plot_line:
                figure.plot((x_value, x_value), [0, 1],
                                 linewidth=2, color='red', label=label)
        figure.ccdfplotdata(args, title=plot_title, xlabel=xlabel, loc=3,
                            logy=True, fs_legend=fs_legend,
                            legend_ncol=legend_ncol)
        if rescale:
            figure.set_xlim(rescale(figure.get_xlim()))
        figure.savefig(save_title + '_ccdf_loglog.pdf')
        del(figure)
    return save_title

def plot_per_as(data, trace, indic, as_list, title, xlabel, rescale,
                out_dir='graph_filtered_final', loc=0, legend_ncol=1,
                service='YouTube', plot_line=None,
                dashes=True, plot_ccdf=True, plot_all_x=True):
    """Plot cdf of data[indic] separated by as_list
    USE AS
tools.filter_streaming.plot_per_as(filtered_flows, '2009_11_26_FTTH',
    itemgetter('DIP-RTT-DATA-Mean-ms-TCP-Up'), (43515, 15169),
    'Mean RTT DATA Up', 'in ms', (lambda (xmin, xmax): (xmin, 300)))
tools.filter_streaming.plot_per_as(filtered_flows, '2009_12_14_FTTH',
    itemgetter('DIP-RTT-DATA-Mean-ms-TCP-Up'), (43515, 15169),
    'Mean RTT DATA Up', 'in ms', (lambda (xmin, xmax): (xmin, 300)))
tools.filter_streaming.plot_per_as(filtered_flows, '2010_02_07_FTTH',
    itemgetter('DIP-RTT-DATA-Mean-ms-TCP-Up'), (43515, 15169),
    'Mean RTT DATA Up', 'in ms', (lambda (xmin, xmax): (xmin, 300)))
tools.filter_streaming.plot_per_as(filtered_flows, '2009_12_14_FTTH',
    (lambda x: 80 * x['peakRate']), (43515, 15169), 'Peak Rate', 'in b/s',
    (lambda (xmin, xmax): (max(xmin, 1e5), xmax)))
tools.filter_streaming.plot_per_as(filtered_flows, '2009_12_14_ADSL_R',
    (lambda x: 80 * x['peakRate']), (43515, 15169), 'Peak Rate', 'in b/s',
    (lambda (xmin, xmax): (max(xmin, 1e4), xmax)))
tools.filter_streaming.plot_per_as(filtered_flows, '2009_11_26_FTTH',
    (lambda x: 80 * x['peakRate']), (43515, 15169), 'Peak Rate', 'in b/s',
    (lambda (xmin, xmax): (max(xmin, 1e4), xmax)))
tools.filter_streaming.plot_per_as(filtered_flows, '2010_02_07_FTTH',
    (lambda x: 80 * x['peakRate']), (43515, 15169), 'Peak Rate', 'in b/s',
    None)
tools.filter_streaming.plot_per_as(filtered_flows, '2008_07_01_ADSL',
    (lambda x: 80 * x['peakRate']), (36561, 15169), 'Peak Rate', 'in b/s',
    None)
tools.filter_streaming.plot_per_as(filtered_flows, '2008_07_01_ADSL',
    itemgetter('DIP-RTT-DATA-Mean-ms-TCP-Up'), (36561, 15169),
    'Mean RTT DATA Up', 'in ms', None)
    """
    #if plot_line:
        #pylab.plot((plot_line, plot_line), [0, 1],
                   #linewidth=2, color='red',
                   #label='median video\nrate: %dkb/s' % plot_line)
    if 'asBGP' in data[trace].dtype.names:
        as_field = 'asBGP'
        format_as = lambda x: x
    elif 'RemAS' in data[trace].dtype.names:
        as_field = 'RemAS'
        format_as = lambda x: 'AS' + str(x)
    else:
        assert False, 'no AS field found in trace: %s' % trace
    args = [('AS ' + str(cur_as),
             [indic(x) for x in data[trace]
              if x[as_field] == format_as(cur_as)])
            for cur_as in as_list]
    # skip if only zero data
    if all(map(lambda (x, ys): all([y == 0 or y == None
                                    for y in ys]), args)):
        print('no data to plot')
        return None
    plot_title = ' '.join([title, 'per AS', '\nfor'] +
                          streaming_tools.format_title(trace).replace('/', '_')
                          .split())
    save_title = plot_indic(args, plot_title,
        xlabel, rescale, out_dir, service, plot_line=plot_line, dashes=dashes,
        loc=loc, plot_all_x=plot_all_x, plot_ccdf=plot_ccdf,
                            legend_ncol=legend_ncol)
    print(save_title)

def plot_filtered_stats(flows, out_dir='graph_filtered_final', tstat=True,
                        indic_list=None, service_title=None):
    """Plot interesting graphs on the filtered data
    Use as:
filtered_flows = cPickle.load(open('filtered_flows_final.pickle'))
tools.filter_streaming.plot_filtered_stats(filtered_flows)
    """
    median_video_rates = {}
    if not indic_list:
        indic_list = [
#            (itemgetter('DIP-RTT-Mean-ms-TCP-Down'),
#                    'Mean RTT Down', 'in ms', None),
#            (itemgetter('DIP-RTT-Mean-ms-TCP-Up'),
#                    'Mean RTT Up', 'in ms', None),
#            (itemgetter('DIP-RTT-Max-ms-TCP-Down'),
#                    'Max RTT Down', 'in ms', None),
#            (itemgetter('DIP-RTT-Max-ms-TCP-Up'),
#                    'Max RTT Up', 'in ms', None),
#            (itemgetter('DIP-RTT-Min-ms-TCP-Down'),
#                    'Min RTT Down', 'in ms', None),
#            (itemgetter('DIP-RTT-Min-ms-TCP-Up'),
#                    'Min RTT Up', 'in ms', None),
        (itemgetter('DIP-RTT-DATA-Mean-ms-TCP-Down'),
                'Mean RTT DATA Down', 'in ms', None),
        (itemgetter('DIP-RTT-DATA-Mean-ms-TCP-Up'),
                'Mean RTT DATA Up', 'in ms', None),
        (itemgetter('DIP-RTT-DATA-Max-ms-TCP-Down'),
                'Max RTT DATA Down', 'in ms', None),
        (itemgetter('DIP-RTT-DATA-Max-ms-TCP-Up'),
                'Max RTT DATA Up', 'in ms', None),
        (itemgetter('DIP-RTT-DATA-Min-ms-TCP-Down'),
                'Min RTT DATA Down', 'in ms', None),
        (itemgetter('DIP-RTT-DATA-Min-ms-TCP-Up'),
                'Min RTT DATA Up', 'in ms', None),
        #((lambda x: 100 * x['DIP-DSQ-NbMes-sec-TCP-Up']
        # / x['DIP-Volume-Number-Packets-Up']),
         #'Loss Rate Up (loss events)', 'in Percent',
         #(lambda (xmin, xmax): (xmin, min(xmax, 100)))),
        #((lambda x: 100 * x['DIP-RTM-NbMes-sec-TCP-Up']
        # / x['DIP-Volume-Number-Packets-Down']),
         #'Loss Rate Up (retransmitted packets)', 'in Percent',
         #(lambda (xmin, xmax): (xmin, min(xmax, 100)))),
        ((lambda x: (100 * x['DIP-DSQ-NbMes-sec-TCP-Down']
          / x['DIP-Volume-Number-Packets-Down']) if
          x['DIP-Volume-Number-Packets-Down'] != 0 else None),
         'Percentage of Loss Events (Down)', '',
         (lambda (xmin, xmax): (.1, 10))),
         #(lambda (xmin, xmax): (xmin, min(xmax, 100)))),
        ((lambda x: (100 * x['DIP-RTM-NbMes-sec-TCP-Down']
          / x['DIP-Volume-Number-Packets-Down']) if
          x['DIP-Volume-Number-Packets-Down'] != 0 else None),
         'Percentage of Retransmitted Packets (Down)', '',
         (lambda (xmin, xmax): (.1, 10))),
         #(lambda (xmin, xmax): (xmin, min(xmax, 100)))),
        (itemgetter('Content-Length'), 'Content Length (from header)',
         'in Bytes', (lambda x: (1e5, 1e9))),
        #(itemgetter('Session-Bytes'), 'Session Length (from flow)',
             #'in Bytes', (lambda x: (1e2, 1e7))),
        (itemgetter('Session-Duration'), 'Session Duration (from flow)',
             'in Seconds', (lambda x: (1e-1, 1e4))),
        (lambda x: 80e-3 * x['peakRate'],
         'Peak Rate (from flow)', 'in kb/s',
         (lambda x: (1e2, 1e5))),
        (itemgetter('DurationDn'), 'Download Duration (from flow)',
             'in Seconds', None),
        (itemgetter('ByteDn'), 'Download Size (from flow)', 'in Bytes',
         None),
        #(itemgetter('nb_skips'),
         #'Skips in video (from flow playback simulation)', 'in Nb',
         #(lambda x: (-1, 10))),
        #((lambda x: 8e-3 * x['Session-Bytes'] / x['Session-Duration']),
         #'Average bit-rate ses (from flow)', 'in kb/s'),
        (compose(streaming_tools.retrieve_bitrate,
                 itemgetter('Application')), 'Video Rate (from header)',
         'in kb/s', (lambda (xmin, xmax): (max(xmin, 10), xmax))),
        (compose(streaming_tools. retrieve_resolution,
                 itemgetter('Application')),
         'Video Resolution (from header)', 'in sq. pixel',
         (lambda (xmin, xmax): (max(xmin, 10), xmax))),
        (itemgetter('Content-Duration'), 'Content Duration (from header)',
             'in Seconds', (lambda x: (1, 1e4))),
        (itemgetter('Content-Duration'),
         'Video Length (from header)', 'in Seconds', (lambda x: (10, 1e3))),
        ((lambda x: 8e-3 * x['Content-Length'] / x['Content-Duration']),
         'Recomputed Video Rate (from header)', 'in kb/s',
         (lambda (xmin, xmax): (max(xmin, 1e2), min(1e4, xmax)))),
        (itemgetter('ByteDn'), 'Flow Size', 'in Bytes', None),
        ((lambda x: ((8e-3 * x['ByteDn'] / x['DurationDn'])
                     if x['DurationDn'] > 0 else None)),
         'Average bit-rate (from flow)', 'in kb/s',
         (lambda (xmin, xmax): (max(xmin, 10), xmax))),
        ((lambda x: 100 * x['ByteDn'] / x['Content-Length']),
         'Downloaded size (from flow and header)', 'in %',
         (lambda (xmin, xmax): (xmin, min(xmax, 110))))]
    if tstat:
        indic_list.append((
            (lambda x: (100 * (x['C_data_bytes'] - x['C_unique_bytes'])
                    / x['C_unique_bytes']) if x['C_data_bytes'] != 0 else None),
         '(Total Bytes - Unique Bytes / Unique Bytes) (Up)', 'in Percent',
                          (lambda (xmin, xmax): (1e-2, 10))))
        indic_list.append((
            (lambda x: (100 * (x['S_data_bytes'] - x['S_unique_bytes'])
                    / x['S_unique_bytes']) if x['S_data_bytes'] != 0 else None),
         '(Total Bytes - Unique Bytes / Unique Bytes) (Down)', 'in Percent',
                          (lambda (xmin, xmax): (1e-2, 10))))
        indic_list.append(((lambda x: (100 * x['S_out_seq_packets']
                                       / x['S_data_packets']) if
                            x['S_data_packets'] != 0 else None),
                   'Percentage of Out of Order Packets (Down)', '',
                   (lambda (xmin, xmax): (xmin, min(xmax, 100)))))
        indic_list.append(((lambda x: (100 * x['C_out_seq_packets']
                                       / x['C_data_packets']) if
                            x['C_data_packets'] != 0 else None),
                   'Percentage of Out of Order Packets (Up)', '',
                   (lambda (xmin, xmax): (xmin, min(xmax, 100)))))
        indic_list.append(((lambda x: (100 * x['S_rexmit_packets']
                                       / x['S_data_packets']) if
                            x['S_data_packets'] != 0 else None),
                   'Percentage of Retransmitted Packets (Down)', '',
                   (lambda (xmin, xmax): (xmin, min(xmax, 100)))))
        indic_list.append(((lambda x: (100 * x['C_rexmit_packets']
                                       / x['C_data_packets']) if
                            x['C_data_packets'] != 0 else None),
                   'Percentage of Retransmitted Packets (Up)', '',
                   (lambda (xmin, xmax): (xmin, min(xmax, 100)))))
        # burst cdf
        indic_list.append(((lambda x: (1 - (x['DIP-DSQ-NbMes-sec-TCP-Down']
                           / (x['S_rexmit_packets'] + x['S_out_seq_packets'])))
                            if (x['S_rexmit_packets'] + x['S_out_seq_packets'])
                            > 0 else None),
                          'Burstiness of Losses (Down)', '', None))
    for service in [cdn for v in (np.concatenate(flows.values()),)
                    for cdn in np.unique(v['Service'])
                    if len([x for x in v if x['Service'] == cdn]) > 100]:
        for indic, title, xlabel, rescale in indic_list:
            plot_line = []
            args = [(streaming_tools.format_title(k),
                     [indic(x) for x in v.compress(v['Service'] == service)])
                    for k, v in sorted(flows.items())]
            # skip if only zero data
            if all(map(lambda (x, ys): all([y == 0 or y == None
                                            for y in ys]), args)):
                continue
            if 'Video Rate' in title:
                # line for median video rate
                video_rates = [x for k, v in args for x in v if x]
                if len(video_rates) > 0:
                    median_video_rates[service] = np.median(video_rates)
                    plot_line.append((median_video_rates[service],
                                      'median video\nrate: %dkb/s' %
                                      median_video_rates[service]))
                    #pylab.plot([median_video_rates[service],
                                #median_video_rates[service]], [0, 1],
                               #linewidth=2, color='red',
                           #label='median video\nrate: %dkb/s' %
                               #median_video_rates[service])
            if title.startswith('Average bit-rate'):
                if service in median_video_rates:
                    plot_line.append((median_video_rates[service],
                                      'median video\nrate: %dkb/s' %
                                      median_video_rates[service]))
                    #pylab.plot([median_video_rates[service],
                                #median_video_rates[service]], [0, 1],
                               #linewidth=2, color='red',
                           #label='median video\nrate: %dkb/s' %
                               #median_video_rates[service])
            if title.startswith('Video Length'):
                for x_value in (16, 208, 600):
                    plot_line.append((x_value, '%d s' % x_value))
                #pylab.plot([16, 16], [0, 1], linewidth=2, color='red',
                           #label='16 s')
                #pylab.plot([208, 208], [0, 1], linewidth=2, color='red',
                           #label='208 s')
                #pylab.plot([600, 600], [0, 1], linewidth=2, color='red',
                           #label='600 s')
            if 'Out of Order' in title:
                plot_line.append((2, '2% threshold'))
            plot_indic(args, title, xlabel, rescale, out_dir, service,
                       plot_line=plot_line)

def plot_flows_stats(datas, out_dir='graph_stats'):
    """Plot GVB (or DIPCP) stats
    first objective: peak rate per user for each trace
    """
    args = []
    for k in sorted(datas):
        if not k.endswith('_GVB'):
            continue
        data = datas[k]
        max_peak = defaultdict(float)
        for x in data.compress([x['direction'] == INDEX_VALUES.DOWN
                                and x['nbPkt'] > 5
                                for x in data]):
            if 80e-3 * x['peakRate'] > max_peak[x['client_id']]:
                max_peak[x['client_id']] = 80e-3 * x['peakRate']
        args.append((streaming_tools.format_title(k.rstrip('_GVB')),
                     max_peak.values()))
    #args = [(streaming_tools.format_title(k.rstrip('_GVB')),
             #[80 * max(flows['peakRate'])
              #for client_id in set(v['client_id'])
              #for flows in (v.compress(v['client_id'] == client_id),)])
            #for k, v in datas.items() if k.endswith('_GVB')]
    save_title = sep.join((out_dir, 'cdf_peak_rate_down_per_client'))
    title = 'Max. Peak Rate (for each client)'
    xlabel = ' '.join((title.split(' (')[0], 'in kb/s'))
    figure = cdfplot_2.CdfFigure()
    figure.cdfplotdata(args, title=title, xlabel=xlabel, loc=0,
                       fs_legend='small')
    figure.savefig(save_title + '_logx.pdf')
    del(figure)

def load_filtered_sessions(flows, gap=300):
    """Return a dict of sessions computed on filtered flows
    USE AS:
filtered_flows = cPickle.load(open('filtered_flows_gvb.pickle'))
filtered_sessions = tools.filter_streaming.load_filtered_sessions(
    filtered_flows)
    """
    return dict([(k, complements.extract_sessions_hh_as(v, getcwd(), gap=gap,
                                                    client_field='client_id'))
                 for k, v in flows.items() if len(v) > 0])

def plot_session_stats(flows, out_dir='graph_sessions', gap=300):
    """Plot sessions stats: call on normal flows to extract the sessions per
    service
filtered_flows = cPickle.load(open('filtered_flows_gvb.pickle'))
tools.filter_streaming.plot_session_stats(filtered_flows)
    """
    for indic, title, xlabel, rescale, select in (
        (itemgetter('nb_flows'),
         'Number of Videos in User Session (gap %d)' % gap, '', None,
         (lambda x: True)),
        (itemgetter('duration'),
         'User Session Duration (gap %d) \n (for sessions of at least 2 flows)'
         % gap, 'in Seconds', (lambda x: (10, 5e3)),
         (lambda x: x['nb_flows'] > 1)),
        ):
        for service in [cdn for v in (np.concatenate(flows.values()),)
                        for cdn in np.unique(v['Service'])
                        if len([x for x in v if x['Service'] == cdn]) > 100]:
            sessions_stats = load_filtered_sessions(
                dict([(k, v.compress(v['Service'] == service))
                      for k, v in flows.items()]), gap=gap)
            try:
                args = [(streaming_tools.format_title(k),
                         [indic(x) for x in v if select(x)])
                        for k, v in sorted(sessions_stats.items())]
                plot_indic(args, title, xlabel, rescale, out_dir, service)
            except TypeError:
                print('Indic: %s. Skipped service: %s' % (title, service))
        else: # for else
            # compute sessions for all services
            sessions_stats = load_filtered_sessions(flows)
            args = [(streaming_tools.format_title(k),
                     [indic(x) for x in v if select(x)])
                    for k, v in sorted(sessions_stats.items())]
            plot_indic(args, title, xlabel, rescale, out_dir, '')

def plot_scatter_stats(flows, out_dir='graph_filtered_final', logy=False):
    """Plot scatterplot of interesting stats (designed for downloaded volume)
    """
    for service in [cdn for v in (np.concatenate(flows.values()),)
                    for cdn in np.unique(v['Service'])
                    if len([x for x in v if x['Service'] == cdn]) > 100]:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes([0.105, 0.2, 0.8, 0.75])
        for indic, title, xlabel, rescale in (
            ((lambda x: (x['Content-Length'],
                         100 * x['ByteDn'] / x['Content-Length'])),
             'Downloaded size (from flow and header)',
             'in % vs. Content Length (from header)',
             (lambda x: (0, 110))),
             #(lambda (xmin, xmax): (xmin, min(xmax, 110))))
            ):
            args = [(streaming_tools.format_title(k),
                     [indic(x) for x in v.compress(v['Service'] == service)])
                    for k, v in sorted(flows.items())]
            colors = cycle('kcbmrgy')
            markers = cycle("x+*o")
            for name, data in args:
                if not data:
                    continue
                x, y = zip(*data)
                ax.plot(x, y , colors.next() + markers.next(),
                        label=': '.join((name, str(len(data)))))
            save_title = sep.join((out_dir,
                               '_'.join(('scatterplot', title.split(' (')[0],
                                         service)).lower().replace(' ', '_')))
            # for latex output (no type 3)
            plot_title = ' '.join((service, title)).replace('_', ' ')
            xlabel = ' '.join((title.split(' (')[0], xlabel))
            ax.set_title(plot_title)
            #ax.set_ylabel('Percentage of Dowloaded Volume')
            ax.set_xlabel(xlabel)
            ax.legend(bbox_to_anchor=(0., -.26, 1., .102), loc=4,
                      ncol=len(args) // 2 + 1, mode="expand", borderaxespad=0.)
            if logy:
                ax.loglog()
            else:
                ax.logx()
            if rescale:
                ax.set_ylim(rescale(ax.get_ylim))
            ax.grid(True)
            fig.savefig(save_title + ('_loglog' if logy else '_logx') + '.pdf')
        del(fig)

def plot_access_rate(trace, rates, color, access_default_rates):
    "Plot the percentage of customers for each access type"
    # normalisation step
    total = sum(rates)
    x = [0]
    y = [0]
    for access, rate in zip(access_default_rates, rates):
        x.append(access)
        y.append(y[-1])
        x.append(access)
        y.append(y[-1] + rate / total)
    figure = cdfplot_2.CdfFigure()
    figure.plot(x, y, color, lw=4, label=trace)
    return figure

def retrieve_peak_rate_per_client(data, key_filter='', as_bgp=None,
                                  insert_key=''):
    "Return a formatted dict with the peak rate per client on each trace"
    return [(insert_key + streaming_tools.format_title(k),
             [max([80e-3 * x['peakRate'] for x in data
                   if x['client_id'] == client_id])
              for data in (v.compress(
                  [((x['asBGP'] in as_bgp) if as_bgp else True)
                   #and ((x['Service'] == service) if service else True)
                   for x in v]),)
              for client_id in set(data['client_id'])])
            for k, v in data.items() if key_filter in k]

def filter_service(data, service, excluded=False):
    "Return a dict with the flows for the service on each trace"
    if excluded:
        comp_func = ne
    else:
        comp_func = eq
    return dict([(k, v.compress([comp_func(x['Service'], service) for x in v]))
                 for k, v in data.items()])

def plot_access_peak(peak_rate_data, title, access=True,
                     out_dir='graph_filtered_final'):
    "Plot CDF of access rates and peak rates (focus on ADSL)"
    if access:
        for trace, rates, color in (
                            ('Access rate\nADSL M',
                             ACCESS_RATES_NB_MONT_2008_07, 'k--'),
                            #(streaming_tools.format_title('2009_11_26_ADSL'),
                             #ACCESS_RATES_NB_MONT_2009_11, 'c:'),
                            ('Access rate\nADSL R',
                             ACCESS_RATES_NB_RENNES_2009_12, 'y--')):
                            #(streaming_tools.format_title('2010_02_07_ADSL_R'),
                             #ACCESS_RATES_NB_RENNES_2010_02, 'y:')):
            plot_access_rate(trace, rates, color, ACCESS_DEFAULT_RATES)
    args = sorted(peak_rate_data +
                  [(streaming_tools.format_title('2008_07_01_FTTH'), []),
                   (streaming_tools.format_title('2009_11_26_FTTH'), []),
                   (streaming_tools.format_title('2009_12_14_FTTH'), []),
                   (streaming_tools.format_title('2010_02_07_FTTH'), [])])
    plot_indic(args, 'Peak Rate per client',
               'in kb/s', (lambda (xmin, xmax): (1e2, xmax)),
               out_dir, title, initial_clf=False)

def plot_peak_rate_all_except(data, service, out_dir='graph_filtered_final'):
    "Plot cdf of data['indic'] by separating service"
    other_service = filter_service(data, service, excluded=True)
    only_service = filter_service(data, service)
    args = (retrieve_peak_rate_per_client(other_service, key_filter='FTTH')
            #+ [(streaming_tools.format_title('2008_07_01_ADSL'), []),
               #(streaming_tools.format_title('2009_11_26_ADSL'), []),
               #(streaming_tools.format_title('2009_12_14_ADSL_R'), []),
               #(streaming_tools.format_title('2010_02_07_ADSL_R'), [])]
            +  retrieve_peak_rate_per_client(only_service, key_filter='FTTH',
                                             insert_key='DM ')
            #+ [('DM ' + streaming_tools.format_title('2008_07_01_ADSL'), []),
               #('DM ' + streaming_tools.format_title('2009_11_26_ADSL'), []),
               #('DM ' + streaming_tools.format_title('2009_12_14_ADSL_R'), []),
               #('DM ' + streaming_tools.format_title('2010_02_07_ADSL_R'), [])]
           )
    plot_indic(sorted(args), 'Peak Rate per Client',
               'in kb/s', (lambda (xmin, xmax): (1e3, 1e5)),
               out_dir, 'DailyMotion vs. all CDNs FTTH', loc=2,
               plot_line=[(2e3, '2Mb/s')])

def table_matching_peak_rates(datas, max_diff=0.25):
    """Return a dict with for each trace the percentage of users with
    non-matching peak rates between YouTube and others (mismatch)"""
    output = {}
    datas_yt = filter_service(datas, 'YouTube')
    for trace, data in sorted(datas.items()):
        data_yt = datas_yt[trace]
        nb_mis_match = 0
        nb_clients_both = 0
#        as_list = (INDEX_VALUES.AS_ALL_YOUTUBE
#                   if not trace.startswith('2008_07_01_')
#                   else INDEX_VALUES.AS_ALL_GOOGLE)
#        as_list = INDEX_VALUES.AS_ALL_GOOGLE
#        data_yt = data.compress([x['asBGP'] in as_list for x in data])
        clients_yt = set(data_yt['client_id'])
        for client in clients_yt:
            yt_rates = (data_yt.compress(data_yt['client_id'] == client)
                              ['peakRate'])
            all_rates = (data.compress(data['client_id'] == client)
                           ['peakRate'])
            if len(all_rates) > len(yt_rates):
                nb_clients_both += 1
            if abs((max(yt_rates) / max(all_rates)) - 1) > max_diff:
                if max(yt_rates) > max(all_rates):
                    print('Higher YouTube rate than others!')
                nb_mis_match += 1
        output[trace] = (nb_mis_match, nb_clients_both, len(clients_yt),
                         len(set(data['client_id'])))
        # for latex integration
        print(streaming_tools.format_title(trace).replace('\n', ': '),
              r"$%d\,\%%$" % int(100 * len(clients_yt) /
                                 len(set(data['client_id']))),
              r"$%d\,\%%$" % int(100 * nb_clients_both / len(clients_yt)),
              r"$%d\,\%%$" % int(100 * nb_mis_match / nb_clients_both),
              sep=' & ', end=' \\\\\n\midrule\n')
    return output

def burst_loss(datas, thr=.8):
    "Return a dict with information on burst losses per trace"
    output = {}
    for trace, data in datas.items():
        output[trace] = 100 * (len(data.compress([
                        1 - (x['DIP-DSQ-NbMes-sec-TCP-Down']
                             / (x['S_rexmit_packets'] + x['S_out_seq_packets']))
                        > thr for x in data
                    if (x['S_rexmit_packets'] + x['S_out_seq_packets'] > 0)]))
            / len(data))
    return output

def table_burst_loss(datas, thrs=(.8, .5)):
    "Print a formatted table of burst losses"
    bursts = zip(sorted(datas.keys()) , zip(*[map(itemgetter(1),
                                     sorted(burst_loss(datas, thr).items()))
                                 for thr in thrs]))
    for trace, (fst, snd) in  bursts:
        print(' & '.join((streaming_tools.format_title(trace).
                          replace('\n', ': '),
                          str(int(fst)) + '\\%', str(int(snd)) + '\\% '))
              + r'\\')

def democratisation(datas, min_nb_flows=1, out_dir='graph_filtered_final'):
    """Return a dict with information on number of clients with multiple flows
    Plots the repartition of nb of flows
    """
    output = {}
    args_flow = []
    args_vol = []
    for trace, datas in sorted(datas.items()):
        nb_client_few_flow = 0
        client_2_flow = defaultdict(int)
        client_2_vol = defaultdict(int)
        for data in datas:
            client_2_flow[data['client_id']] += 1
            client_2_vol[data['client_id']] += data['Session-Bytes']
        args_flow.append((streaming_tools.format_title(trace),
                          client_2_flow.values()))
        args_vol.append((streaming_tools.format_title(trace),
                         client_2_vol.values()))
        nb_client_few_flow = len([nb_flows
                                  for nb_flows in client_2_flow.values()
                                  if nb_flows <= min_nb_flows])
        print(trace, int(100 * nb_client_few_flow / len(client_2_flow)))
        output[trace] = (nb_client_few_flow, len(client_2_flow))
    #args.append((streaming_tools.format_title('2008_07_01_FTTH'), []))
    title = 'Repartiton of Number of flows per Client\nfor HTTP Streaming flows'
    cdfplot.repartplotdataN(sorted(args_flow), _title=title,
                            _ylabel='Cumulative Percentage of Number of flows',
                    savefile=sep.join((out_dir, 'repart_nb_per_client.pdf')))
    title = 'Repartiton of Volume per Client\nfor HTTP Streaming flows'
    cdfplot.repartplotdataN(sorted(args_vol), _title=title,
                            _ylabel='Cumulative Percentage of Streaming Volume',
                    savefile=sep.join((out_dir, 'repart_vol_per_client.pdf')))
    return output

def indic_per_client(datas, indic='Session-Bytes', client_id='client_id',
                    out_dir='graph_filtered_final'):
    "Plot the repartition of indic per client"
    args = []
    for trace, data in sorted(datas.items()):
        client_2_vol = defaultdict(int)
        for flow in data:
            client_2_vol[flow[client_id]] += flow[indic]
        args.append((streaming_tools.format_title(trace),
                client_2_vol.values()))
    #args.append((streaming_tools.format_title('2008_07_01_FTTH'), []))
    title = 'Repartiton of Volume per Client\nfor HTTP Streaming flows'
    cdfplot.repartplotdataN(sorted(args), _title=title,
                            _ylabel='Cumulative Percentage of Volume',
                    savefile=sep.join((out_dir, 'repart_vol_per_client.pdf')))

def bin2ip(bits_ip):
    "Return a string corresponding to ip address in bits form"
    assert len(bits_ip) == 32
    ip_address = []
    for i in range(4):
        ip_address.append(int('0b' + bits_ip[(i * 8):((i + 1) * 8)], 2))
    return '.'.join(map(str, ip_address))

def get_prefix(ip, mask_length=16):
    """Return the prefix of the ip address
    >>> get_prefix('255.255.255.255')
    '255.255.0.0/16'
    >>> get_prefix('255.255.255.255', mask_length=32)
    '255.255.255.255/32'
    >>> get_prefix('255.255.255.255', mask_length=15)
    '255.254.0.0/15'
    """
    bits_ip = bin(ip2int(ip)).split('0b')[1]
    bits_ip = '0' * (32 - len(bits_ip)) + bits_ip
    bits_ip = bits_ip[:mask_length] + '0' * (32 - mask_length)
    return '/'.join((bin2ip(bits_ip), str(mask_length)))

def plot_per_prefix(data, trace, indic, title, xlabel, rescale, mask_length=16,
                    out_dir='graph_filtered_final', loc=2, plot_line=None,
                    plot_all_x=False, plot_ccdf=False, service='YouTube',
                    fs_legend='large', ip_field='srcAddr', figure=None):
    "Plot indic according to IP prefix"
    cur_data = data[trace]
    args = [('prefix ' + str(cur_prefix),
             [indic(x) for x in cur_data
              if
              (get_prefix(x[ip_field], mask_length=mask_length) == cur_prefix)])
            for cur_prefix in sorted(set([get_prefix(x, mask_length=mask_length)
                                          for x in cur_data[ip_field]]))]
    print([(k, len(v)) for (k, v) in args])
    plot_indic(args, ' '.join(
                [title, 'per Prefix', '\nfor'] +
                streaming_tools.format_title(trace).replace('/', '_').split()),
        xlabel, rescale, out_dir, service, loc=loc, plot_line=plot_line,
        fs_legend=fs_legend, plot_all_x=plot_all_x, plot_ccdf=plot_ccdf)

def find_loss_location(flows):
    """Return a list of tuple indicating loss location and also flows with
    losses"""
    return [(k,
             # pure access loss
             len(data.compress([x for x in data
                                if x['S_out_seq_packets'] == 0
                               and x['S_rexmit_packets'] > 0])),
             #len(data.compress([x for x in data if
                                #x['DIP-DSQ-NbMes-sec-TCP-Down'] ==
                                #x['DIP-RTM-NbMes-sec-TCP-Down']])),
             # pure backbone loss
             #len(data.compress(data['DIP-RTM-NbMes-sec-TCP-Down'] == 0)),
             len(data.compress([x for x in data
                                if x['S_rexmit_packets'] == 0
                               and x['S_out_seq_packets'] > 0])),
             # any loss
             len(data.compress([x for x in data if x['S_out_seq_packets'] > 0
                                or x['S_rexmit_packets'] > 0])),
             len(data), len(v),
             #len(data.compress(data['DIP-RTM-NbMes-sec-TCP-Down'] > 0)),
             #len(data.compress(data['DIP-DSQ-NbMes-sec-TCP-Down'] > 0))
            )
            for  k,v in sorted(flows.items())
            for data in (v.compress([x['S_data_packets'] > 0
                                     #x['DIP-DSQ-NbMes-sec-TCP-Down'] > 0
                                     #or x['DIP-RTM-NbMes-sec-TCP-Down'] > 0
                                     for x in v]),)]

def table_link_location(flows, separate_both=False):
    """Return a formatted table for latex of losses location information
    USE AS:
tools.filter_streaming.table_link_location(flows)
tools.filter_streaming.table_link_location(
    tools.filter_streaming.filter_service(flows, 'DiversX', excluded=True))
tools.filter_streaming.table_link_location(flows, separate_porn=True)
    """
    idx_trace = 0
    idx_pure_access = 1
    idx_pure_backbone = 2
    idx_any_loss = 3
    idx_any_pkt = 4
    idx_tot_flows = 5
    if not separate_both:
        output = ' \\\\\n\midrule\n'.join([' & '.join((
            streaming_tools.format_title(v[idx_trace]).replace('\n', ': '),
            '$%d\%%$' % (100 * v[idx_pure_backbone] / v[idx_any_loss]),
            '$%d\%%$' % (100 * v[idx_any_loss] / v[idx_any_pkt])))
            for v in find_loss_location(flows)])
    else:
        youtube_flows = filter_service(flows, 'YouTube')
        dailymotion_flows = filter_service(flows, 'DailyMotion')
        non_youtube_flows = filter_service(flows, 'YouTube', excluded=True)
        non_porn_non_youtube_flows = filter_service(non_youtube_flows,
                                                    'DiversX', excluded=True)
        output = ' \\\\\n\midrule\n'.join([' & '.join((
            streaming_tools.format_title(v[idx_trace]).replace('\n', ': '),
            '$%d\%%$' % round(100 * v[idx_any_loss] / v[idx_any_pkt]),
            '$%d\%%$' % round(100 * v[idx_pure_backbone] / v[idx_any_loss]),
            '$%d\%%$' % round(100 * v[idx_pure_access] / v[idx_any_loss]))
            + ('$%d\%%$' % round(100 * n[idx_any_loss] / n[idx_any_pkt]),
               '$%d\%%$' % round(100 * n[idx_pure_backbone] / n[idx_any_loss]),
               '$%d\%%$' % round(100 * n[idx_pure_access] / n[idx_any_loss]))
            + ('$%d\%%$' % round(100 * y[idx_any_loss] / y[idx_any_pkt]),
               '$%d\%%$' % round(100 * y[idx_pure_backbone] / y[idx_any_loss]),
               '$%d\%%$' % round(100 * y[idx_pure_access] / y[idx_any_loss]))
            + ('$%d\%%$' % round(100 * d[idx_any_loss] / d[idx_any_pkt]),
               '$%d\%%$' % round(100 * d[idx_pure_backbone] / d[idx_any_loss]),
               '$%d\%%$' % round(100 * d[idx_pure_access] / d[idx_any_loss]))
        )
            for v, n, y, d in zip(find_loss_location(flows),
                               find_loss_location(non_porn_non_youtube_flows),
                                  find_loss_location(youtube_flows),
                                  find_loss_location(dailymotion_flows))])
    print('\\midrule\n' + output + '\\\\\n\\bottomrule')

def extract_good_bad_flows(flows):
    """Return a dict of dict with for each trace a good and bad video array"""
    output = {}
    for trace in sorted(flows):
        data = flows[trace].compress(flows[trace]['DurationDn'] > 0)
        bad_flows = data.compress(
            [(8e-3 * x['Content-Length'] / x['Content-Duration'])
             > (8e-3 * x['ByteDn'] / x['DurationDn']) for x in data])
        good_flows = data.compress(
            [(8e-3 * x['Content-Length'] / x['Content-Duration'])
             <= (8e-3 * x['ByteDn'] / x['DurationDn']) for x in data])
        output[trace] = {'good': good_flows, 'bad': bad_flows}
    return output

def compute_bad_videos(flows):
    """Return the fraction of flows with lower average bit-rate than video
    encoding rate
    """
    output = []
    for trace in sorted(flows):
        data = flows[trace].compress(flows[trace]['DurationDn'] > 0)
        bad_flows = [x['nb_skips'] for x in data
                     if (8e-3 * x['Content-Length'] / x['Content-Duration'])
                         > (8e-3 * x['ByteDn'] / x['DurationDn'])]
        output.append(int(100 * len(bad_flows) / len(data)))
    return output

def table_user_exp(flows):
    """Return a formatted table for latex of fraction of flows with lower
    average bit-rate than video encoding rate
    """
    youtube_flows = filter_service(flows, 'YouTube')
    dailymotion_flows = filter_service(flows, 'DailyMotion')
    output = ' \\\\\n\midrule\n'.join([' & '.join((
        streaming_tools.format_title(trace).replace('\n', ': '),
        '$%d\%%$' % yt, '$%d\%%$' % dm))
        for trace, yt, dm in zip(sorted(flows.keys()),
                                 compute_bad_videos(youtube_flows),
                                 compute_bad_videos(dailymotion_flows))])
    print('\\midrule\n' + output + '\\\\\n\\bottomrule')

def plot_indic_yt_others(filtered_flows, out_dir, tstat=True):
    "Plot indicators by separating YouTube traffic from other"
    non_youtube_service = filter_service(filtered_flows, 'YouTube',
                                         excluded=True)
    #plot_peak_rate_all_except(non_youtube_service, 'DailyMotion')
    indic_list = [
        ((lambda x: 8e-3 * x['Content-Length'] / x['Content-Duration']),
         'Recomputed Video Rate (from header)', 'in kb/s',
         (lambda (xmin, xmax): (max(xmin, 1e2), min(1e4, xmax)))),
        ((lambda x: ((8e-3 * x['ByteDn'] / x['DurationDn'])
                     if x['DurationDn'] > 0 else None)),
         'Average bit-rate (from flow)', 'in kb/s',
         (lambda (xmin, xmax): (max(xmin, 10), xmax))),
        (itemgetter('ByteDn'), 'Flow Size', 'in Bytes', None),
        #            (itemgetter('DIP-RTT-Max-ms-TCP-Down'),
        #                    'Max RTT Down', 'in ms', None),
        #            (itemgetter('DIP-RTT-Max-ms-TCP-Up'),
        #                    'Max RTT Up', 'in ms', None),
        #            (itemgetter('DIP-RTT-Min-ms-TCP-Down'),
        #                    'Min RTT Down', 'in ms', None),
        #            (itemgetter('DIP-RTT-Min-ms-TCP-Up'),
        #                    'Min RTT Up', 'in ms', None),
        #            (itemgetter('DIP-RTT-Mean-ms-TCP-Down'),
        #                    'Mean RTT Down', 'in ms', None),
        #            (itemgetter('DIP-RTT-Mean-ms-TCP-Up'),
        #                    'Mean RTT Up', 'in ms', None),
        (itemgetter('DIP-RTT-DATA-Max-ms-TCP-Down'),
         'Max RTT DATA Down', 'in ms', None),
        (itemgetter('DIP-RTT-DATA-Max-ms-TCP-Up'),
         'Max RTT DATA Up', 'in ms', None),
        (itemgetter('DIP-RTT-DATA-Min-ms-TCP-Down'),
         'Min RTT DATA Down', 'in ms', None),
        (itemgetter('DIP-RTT-DATA-Min-ms-TCP-Up'),
         'Min RTT DATA Up', 'in ms', None),
        (itemgetter('DIP-RTT-DATA-Mean-ms-TCP-Down'),
         'Mean RTT DATA Down', 'in ms', None),
        (itemgetter('DIP-RTT-DATA-Mean-ms-TCP-Up'),
         'Mean RTT DATA Up', 'in ms', None),
        (itemgetter('DIP-RTT-DATA-NbMes-ms-TCP-Down'),
         'NbMes RTT DATA Down', 'in ms', None),
        (itemgetter('DIP-RTT-DATA-NbMes-ms-TCP-Up'),
         'NbMes RTT DATA Up', 'in ms', None),
        ((lambda x: (100 * x['DIP-DSQ-NbMes-sec-TCP-Down']
                     / x['DIP-Volume-Number-Packets-Down']) if
          x['DIP-Volume-Number-Packets-Down'] != 0 else 0),
         'Percentage of Loss Events (Down)', '',
         (lambda (xmin, xmax): (.1, 10))),
        #(lambda (xmin, xmax): (xmin, min(xmax, 100)))),
        ((lambda x: (100 * x['DIP-RTM-NbMes-sec-TCP-Down']
                     / x['DIP-Volume-Number-Packets-Down']) if
          x['DIP-Volume-Number-Packets-Down'] != 0 else 0),
         'Percentage of Retransmitted Packets (Down)', '',
         (lambda (xmin, xmax): (.1, 10)))]
        #(lambda (xmin, xmax): (xmin, min(xmax, 100)))))
    if tstat:
        indic_list.append((
            (lambda x: (100 * (x['C_data_bytes'] - x['C_unique_bytes'])
                    / x['C_unique_bytes']) if x['C_data_bytes'] != 0 else None),
         '(Total Bytes - Unique Bytes / Unique Bytes) (Up)', 'in Percent',
                           (lambda (xmin, xmax): (1e-2, 10))))
        indic_list.append((
            (lambda x: (100 * (x['S_data_bytes'] - x['S_unique_bytes'])
                    / x['S_unique_bytes']) if x['S_data_bytes'] != 0 else None),
         '(Total - Unique Bytes / Unique Bytes) (Down)', 'in Percent',
                           (lambda (xmin, xmax): (1e-2, 10))))
        indic_list.append(((lambda x: (100 * x['S_out_seq_packets']
                                       / x['S_data_packets']) if
                            x['S_data_packets'] != 0 else None),
                   'Percentage of Out of Order Packets (Down)', '',
                   (lambda (xmin, xmax): (xmin, min(xmax, 100)))))
        indic_list.append(((lambda x: (100 * x['C_out_seq_packets']
                                       / x['C_data_packets']) if
                            x['C_data_packets'] != 0 else None),
                   'Percentage of Out of Order Packets (Up)', '',
                   (lambda (xmin, xmax): (xmin, min(xmax, 100)))))
        indic_list.append(((lambda x: (100 * x['S_rexmit_packets']
                                       / x['S_data_packets']) if
                            x['S_data_packets'] != 0 else None),
                   'Percentage of Retransmitted Packets (Down)', '',
                   (lambda (xmin, xmax): (xmin, min(xmax, 100)))))
        indic_list.append(((lambda x: (100 * x['C_rexmit_packets']
                                       / x['C_data_packets']) if
                            x['C_data_packets'] != 0 else None),
                   'Percentage of Retransmitted Packets (Up)', '',
                   (lambda (xmin, xmax): (xmin, min(xmax, 100)))))
    for data, service in ((filtered_flows, ''),
                          (non_youtube_service, 'Other')):
        for indic, title, xlabel, rescale in indic_list:
            args = [(streaming_tools.format_title(k), [indic(x) for x in v])
                    for k, v in sorted(data.items())]
            plot_indic(args, title, xlabel, rescale, out_dir, service)

def plot_per_as_and_prefix(youtube_service, dailymotion_service, tstat=True):
    "Wrapper to plot interesting graphs per AS and prefix"
    indic_list = [((lambda x: (100 * x['DIP-DSQ-NbMes-sec-TCP-Down']
                               / x['DIP-Volume-Number-Packets-Down']) if
                    x['DIP-Volume-Number-Packets-Down'] != 0 else None),
                   'Percentage of Loss Events (Down)', '',
                   #(lambda (xmin, xmax): (.1, 10)), None),
                   (lambda (xmin, xmax): (xmin, min(xmax, 100))), None),
                  ((lambda x: (100 * x['DIP-RTM-NbMes-sec-TCP-Down']
                               / x['DIP-Volume-Number-Packets-Down']) if
                    x['DIP-Volume-Number-Packets-Down'] != 0 else None),
                   'Percentage of Retransmitted Packets (Down)', '',
                   #(lambda (xmin, xmax): (.1, 10)), None),
                   (lambda (xmin, xmax): (xmin, min(xmax, 100))), None),
                  #(itemgetter('DIP-RTT-Min-ms-TCP-Down'), 'Min RTT Down',
                  #'in ms', (lambda (xmin, xmax): (xmin, 300)), None),
                  #(itemgetter('DIP-RTT-Min-ms-TCP-Up'), 'Min RTT Up',
                  #'in ms', (lambda (xmin, xmax): (xmin, 300)), None),
                  #(itemgetter('DIP-RTT-Mean-ms-TCP-Up'), 'Mean RTT Up',
                  #'in ms', (lambda (xmin, xmax): (xmin, 300)), None),
                  (itemgetter('DIP-RTT-DATA-Min-ms-TCP-Down'),
                   'Min RTT DATA Down', 'in ms',
                   (lambda (xmin, xmax): (xmin, 300)), None),
                  (itemgetter('DIP-RTT-DATA-Min-ms-TCP-Up'), 'Min RTT DATA Up',
                   'in ms', (lambda (xmin, xmax): (xmin, 300)), None),
                  (itemgetter('DIP-RTT-DATA-Mean-ms-TCP-Down'),
                   'Mean RTT DATA Up', 'in ms',
                   (lambda (xmin, xmax): (xmin, 300)), None),
                  (itemgetter('DIP-RTT-DATA-Mean-ms-TCP-Up'),
                   'Mean RTT DATA Up', 'in ms',
                   (lambda (xmin, xmax): (xmin, 300)), None),
                  (itemgetter('DIP-RTT-DATA-NbMes-ms-TCP-Down'),
                   'NbMes RTT DATA Up', 'in ms',
                   (lambda (xmin, xmax): (xmin, 300)), None),
                  (itemgetter('DIP-RTT-DATA-NbMes-ms-TCP-Up'),
                   'NbMes RTT DATA Up', 'in ms',
                   (lambda (xmin, xmax): (xmin, 300)), None),
                  (itemgetter('ByteDn'), 'Flow Size', 'in Bytes', None, None),
                  ((lambda x: 80e-3 * x['peakRate']), 'Peak Rate', 'in kb/s',
                   (lambda (xmin, xmax): (max(xmin, 1e2), xmax)), None),
                  ((lambda x: ((8e-3 * x['ByteDn'] / x['DurationDn'])
                               if x['DurationDn'] > 0 else None)),
                   'Average bit-rate', 'in kb/s',
                   (lambda (xmin, xmax): (max(xmin, 10), xmax)),
                   [(324, 'median video\nrate: 324kb/s')]),
                  ((lambda x: 100 * x['ByteDn'] / x['Content-Length']),
                   'Downloaded size (from flow and header)', 'in %',
                   (lambda (xmin, xmax): (xmin, min(xmax, 110))), None)]
    if tstat:
        indic_list.append((
            (lambda x: (100 * (x['C_data_bytes'] - x['C_unique_bytes'])
                    / x['C_unique_bytes']) if x['C_data_bytes'] != 0 else None),
         '(Total Bytes - Unique Bytes / Unique Bytes) (Up)', 'in Percent',
                           (lambda (xmin, xmax): (1e-2, 10)), None))
        indic_list.append((
            (lambda x: (100 * (x['S_data_bytes'] - x['S_unique_bytes'])
                    / x['S_unique_bytes']) if x['S_data_bytes'] != 0 else None),
         '(Total Bytes - Unique Bytes / Unique Bytes) (Down)', 'in Percent',
                           (lambda (xmin, xmax): (1e-2, 10)), None))
        indic_list.append(((lambda x: (100 * x['S_out_seq_packets']
                                       / x['S_data_packets']) if
                            x['S_data_packets'] != 0 else None),
                   'Percentage of Out of Order Packets (Down)', '',
                   (lambda (xmin, xmax): (xmin, min(xmax, 100))),
                           ((2, '2% threshold'),)))
        indic_list.append(((lambda x: (100 * x['C_out_seq_packets']
                                       / x['C_data_packets']) if
                            x['C_data_packets'] != 0 else None),
                   'Percentage of Out of Order Packets (Up)', '',
                   (lambda (xmin, xmax): (xmin, min(xmax, 100))),
                           ((2, '2% threshold'),)))
        indic_list.append(((lambda x: (100 * x['S_rexmit_packets']
                                       / x['S_data_packets']) if
                            x['S_data_packets'] != 0 else None),
                   'Percentage of Retransmitted Packets (Down)', '',
                   (lambda (xmin, xmax): (xmin, min(xmax, 100))), None))
        indic_list.append(((lambda x: (100 * x['C_rexmit_packets']
                                       / x['C_data_packets']) if
                            x['C_data_packets'] != 0 else None),
                   'Percentage of Retransmitted Packets (Up)', '',
                   (lambda (xmin, xmax): (xmin, min(xmax, 100))), None))
    as_list_yt = sorted(set(reduce(concat, map(list, map(itemgetter('asBGP'),
                                                  youtube_service.values())))),
                        reverse=True)
    as_list_dm = sorted(set(reduce(concat, map(list, map(itemgetter('asBGP'),
                                              dailymotion_service.values())))),
                        reverse=True)
    for trace in INDEX_VALUES.TRACE_LIST:
        for indic, title, xlabel, rescale, plot_line in indic_list:
            plot_per_as(youtube_service, trace, indic, as_list_yt, title,
                        xlabel, rescale, plot_line=plot_line)
            # plot per prefix only for RTT data
            if 'RTT' in title or 'Loss' in title:
                plot_per_as(dailymotion_service, trace, indic,
                                     as_list_dm, title, xlabel, rescale,
                                     plot_line=plot_line,
                                     service='DailyMotion')
                plot_per_prefix(youtube_service, trace, indic, title, xlabel,
                                rescale)

def main(
    flows_file='/home/louis/streaming/filtered_flows_complete_all_tstat.pickle',
    out_dir='graph_filtered_final', tstat=True, fast=True):
    "Plot main graphs in one step"
    log = logging.getLogger('filter_streaming')
    handler = logging.StreamHandler(sys.stdout)
    log_formatter = logging.Formatter("%(asctime)s - %(filename)s:%(lineno)d - "
                                      "%(levelname)s - %(message)s")
    handler.setFormatter(log_formatter)
    log.addHandler(handler)
    log.setLevel(LOG_LEVEL)
    filtered_flows = cPickle.load(open(flows_file))
    log.info('Start plotting with fast mode: %s' % fast)
    log.debug('start Dailymotion peak')
    plot_peak_rate_all_except(filtered_flows, 'DailyMotion')
    log.debug('Dailymotion peak done')
    log.debug('start filtered_stats')
    plot_filtered_stats(filtered_flows, out_dir=out_dir, tstat=tstat)
    log.debug('filtered_stats done')
    log.debug('start peak rates yt dm')
    for service, as_bgp in (('YouTube AS', INDEX_VALUES.AS_ALL_YOUTUBE),
                            ('DailyMotion', INDEX_VALUES.AS_DAILYMOTION)):
        args = sorted(retrieve_peak_rate_per_client(filtered_flows,
                                     as_bgp=as_bgp))
        plot_indic(args, 'Peak Rate per client', 'in kb/s',
                   (lambda (xmin, xmax): (1e2, xmax)), out_dir,
                   service)
        del(args)
    log.debug('peak rates yt dm done')
    log.debug('start plot yt others')
    plot_indic_yt_others(filtered_flows, out_dir, tstat=tstat)
    log.debug('plot yt others done')
    if not fast:
        log.debug('start access_peak')
        plot_access_peak(retrieve_peak_rate_per_client(filtered_flows,
                                                       key_filter='FTTH'),
                         'FTTH All CDNs')
        log.debug('access_peak done')
        plot_access_peak(retrieve_peak_rate_per_client(filtered_flows,
                                                       key_filter='FTTH',
                                           as_bgp=INDEX_VALUES.AS_ALL_YOUTUBE),
                         'FTTH AS YouTube', out_dir=out_dir)
        log.debug('start compute active time graphs')
        nb_hh = 100
        top_hh = complements.compute_active_time_per_as(
                                        cnx_stream=filtered_flows, nb_hh=nb_hh)
        complements.plot_active_time_as(top_hh, out_dir=out_dir, nb_hh=nb_hh,
                                        as_list=('YOUTUBE', 'DAILYMOTION'))
        del(top_hh)
        log.debug('active_time done')
        log.debug('start democratisation graph')
        democratisation(filtered_flows, out_dir=out_dir)
        log.debug('democratisation done')
        log.debug('start vol per user per second')
        datas_stream_as = streaming_tools.extract_volumes_per_as(filtered_flows,
                                                                normalized=True)
        streaming_tools.bar_chart(datas_stream_as,
                      title='HTTP Streaming Flows Normalized Volume per User',
                                  ylabel='Volume per User in Bytes per Second',
                                  out_dir=out_dir)
        del(datas_stream_as)
        log.debug('vol per user per sec done')
        plot_access_peak(retrieve_peak_rate_per_client(filtered_flows,
                                                       key_filter='ADSL'),
                         'ADSL All CDNs')
        log.debug('access_peak done')
        plot_access_peak(retrieve_peak_rate_per_client(filtered_flows,
                                                       key_filter='ADSL',
                                           as_bgp=INDEX_VALUES.AS_ALL_YOUTUBE),
                         'ADSL AS YouTube', out_dir=out_dir)
        log.debug('retrieve_access_peak done')
        log.debug('start session_stats')
        plot_session_stats(filtered_flows, out_dir=out_dir)
        log.debug('session_stats done')
        print('\n'.join([str((k,
                              [(service, len(set([x['client_id'] for x in v
                                                  if x['Service'] == service])))
                               for service in ('YouTube', 'DailyMotion',
                                               'Megaupload')]))
                         for k, v in sorted(filtered_flows.items())]))
        print('\n'.join([str((trace, [(k, len(list(g))) for k, g in groupby(
            sorted(filtered_flows[trace].compress(
                filtered_flows[trace]['Service']=='YouTube')['asBGP']))]))
            for trace in sorted(filtered_flows)]))
        remaining_data = dict([(k, extract_remaining_list(v))
                             for k, v in filtered_flows.items()])
        complements.plot_remaining_download(remaining_data,
                                            plot_excluded=False,
                                            prefix='remaining_time_mix_dm_goo',
                                            out_dir=out_dir,
                                            good_indic='on all services',
                                            loglog=True, logx=True, th=None)
        del(remaining_data)
        log.debug('all remaining done')
        log.debug('start all flows in paralell')
        if not fast:
            for b in (60, 30, 10, 1):
                th = (1e6 * b / 30, 1e7 * b / 30)
                complements.plot_all_nb_flows(filtered_flows, bin_size=b,
                                              color=True, thresholds=th,
                                              out_dir=out_dir, hour_graph=True,
                                              start_indic='initTime')
            log.debug('all flows in paralell done')
    log.debug('start large flows in parallel')
    large_flows = dict([(k, v.compress(v['Session-Bytes'] > 10e3))
                        for k, v in filtered_flows.items()])
    for b in (60, 30, 10, 1):
        th = (1e6 * b / 30, 1e7 * b / 30)
        complements.plot_all_nb_flows(large_flows, bin_size=b, color=True,
                                      thresholds=th, out_dir=out_dir,
                                      hour_graph=True, postfix='10kBytes',
                                      start_indic='initTime')
    del(large_flows)
    log.debug('large flows done')
    youtube_service = filter_service(filtered_flows, 'YouTube')
    log.debug('start plot peak yt flow')
    service = 'YouTube AS'
    indic, title, xlabel, rescale = (lambda x: 80e-3 * x['peakRate'],
                                     'Peak Rate (from flow)',
                                     'in kb/s', (lambda x: (1e2, 1e5)))
    args = [(streaming_tools.format_title(k),
             [indic(x) for x in v.compress(
                 [x['asBGP'] in INDEX_VALUES.AS_ALL_YOUTUBE for x in v])])
            for k, v in sorted(youtube_service.items())]
    plot_indic(args, title, xlabel, rescale, out_dir,
                   service)
    del(args)
    log.debug('plot peak yt flow done')
    log.debug('start yt remaining vol')
    remaining_yt = dict([(k, extract_remaining_list(v))
                         for k, v in youtube_service.items()])
    complements.plot_remaining_download(remaining_yt,
                                        plot_excluded=False,
                                        prefix='remaining_time_mix_dm_goo',
                                        out_dir=out_dir,
                                        good_indic='on YouTube',
                                        loglog=True, logx=True, th=None)
    del(remaining_yt)
    log.debug('yt remaining done')
    log.debug('start yt paralell flows')
    for b in (60, 30, 10, 1):
        th = (1e6 * b / 30, 1e7 * b / 30)
        complements.plot_all_nb_flows(youtube_service, bin_size=b, color=True,
                                      thresholds=th, out_dir=out_dir,
                                      hour_graph=True, postfix='youtube',
                                      start_indic='initTime')
    log.debug('yt paralell flows done')
    log.debug('start dm remaining vol')
    dailymotion_service = filter_service(filtered_flows, 'DailyMotion')
    del(filtered_flows)
    log.debug('start plot as prefix')
    plot_per_as_and_prefix(youtube_service, dailymotion_service, tstat=tstat)
    del(youtube_service)
    log.debug('plot as prefix done')
    log.debug('start remaining dm')
    remaining_dm = dict([(k, extract_remaining_list(v))
                         for k, v in dailymotion_service.items()])
    complements.plot_remaining_download(remaining_dm,
                                        plot_excluded=False,
                                        prefix='remaining_time_mix_dm_goo',
                                        out_dir=out_dir,
                                        good_indic='on DailyMotion',
                                        loglog=True, logx=True, th=None)
    del(remaining_dm)
    log.debug('dm remaining done')
    log.info('all processing done')
    return 0

if __name__ == "__main__":
    sys.exit(main())

