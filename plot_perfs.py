#!/usr/bin/env python
"Module to plot many relevant cdf for comparing traces."

#from optparse import OptionParser
#import numpy as np
import pylab
#import os

import cdfplot
import INDEX_VALUES

from load_dipcp_file import filter_dipcp_array

#global_trace_list = ('ADSL_2008', 'FTTH_2008', 'ADSL_nov_2009',
#'FTTH_nov_2009', 'ADSL_dec_2009', 'FTTH_dec_2009',
#'FTTH_Montsouris_2010_02_07', 'FTTH_mar_2010')

def validate_dipcp_yt_traces(data_flow, dipcp_flow, name,
                        output_path = 'rapport/BGP_figs/valid_YT_NT',
                        prefix = 'YT_N_',
                        title = 'Youtube HTTP Streaming Downstream Flows:\n'):
    "Take N tuples of numpy record arrays (dipcp and GVB) \
    to generate a lot of graphs related to Youtube"

    flows = {}
    index = name.replace(' ', '_')
    flows['flows_%s_stream' % index] = data_flow.compress(
        data_flow['dscp'] == INDEX_VALUES.DSCP_HTTP_STREAM)
    flows['flows_%s_stream_down' % index] = flows['flows_%s_stream'
            % index].compress(flows['flows_%s_stream' % index]['direction']
                    == INDEX_VALUES.DOWN)
    flows['flows_%s_down_yt' % index] = flows['flows_%s_stream_down'
            % index].compress([x['asBGP'] in INDEX_VALUES.AS_YOUTUBE
                for x in flows['flows_%s_stream_down' % index]])
    flows['flows_%s_down_yt_1MB' % index] = flows['flows_%s_down_yt'
                % index].compress(flows['flows_%s_down_yt' % index]
                        ['l3Bytes'] > 10**6)
    flows['flows_%s_down_yt_eu' % index] = flows['flows_%s_stream_down'
            % index].compress([x['asBGP'] in INDEX_VALUES.AS_YOUTUBE_EU
                for x in flows['flows_%s_stream_down' % index]])
    flows['flows_%s_down_yt_eu_1MB' % index] = flows['flows_%s_down_yt_eu'
            % index].compress(flows['flows_%s_down_yt_eu' % index]
                    ['l3Bytes'] > 10**6)
    #dipcp flows
    flows['dipcp_%s_down_yt' % index] = filter_dipcp_array(dipcp_flow,
            flows['flows_%s_down_yt' % index])
    flows['dipcp_%s_down_yt_1MB' % index] = flows['dipcp_%s_down_yt'
            % index].compress(flows['dipcp_%s_down_yt' % index]
                    ['DIP-Volume-Sum-Bytes-Down'] > 10**6)
    flows['dipcp_%s_down_yt_eu' % index] = filter_dipcp_array(dipcp_flow,
            flows['flows_%s_down_yt_eu' % index])
    flows['dipcp_%s_down_yt_eu_1MB' % index] = flows[
            'dipcp_%s_down_yt_eu' % index].compress(
                    flows['dipcp_%s_down_yt' % index]
                    ['DIP-Volume-Sum-Bytes-Down'] > 10**6)

    # TODO: make a wrapper function, in order to call directly on filtered
    # arrays


    #plot cdf of mean rate
    args = []
    args.append(([8*x['l3Bytes']/(1000.0*x['duration'])
        for x in flows['flows_%s_down_yt' % index]
                  if x['duration']>0],
                 'GVB YouTube %s' % name))
    args.append((flows['dipcp_%s_down_yt' % index]['DIP-Thp-Number-Kbps-Down'],
        'dipcp YouTube %s' % name))
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Mean Rate' % title,
                         _xlabel='Downstream Mean Rate in kbit/s', _loc=2)
    pylab.savefig(output_path + '/%sflows_mean_rate.pdf' % prefix, format='pdf')

    #plot cdf of mean rate over flows_A larger than 1MB
    args = []
    args.append(([8*x['l3Bytes']/(1000.0*x['duration'])
                  for x in flows['flows_%s_down_yt_1MB' % index]
                  if x['duration']>0],
                 'YouTube %s' % name))
    args.append((flows['dipcp_%s_down_yt_1MB' % index]
        ['DIP-Thp-Number-Kbps-Down'], 'dipcp YouTube %s' % name))
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Mean Rate for flows \
larger than 1MBytes' % title,
            _xlabel='Downstream Mean Rate in kbit/s', _loc=2)
    pylab.savefig(output_path + '/%sflows_mean_rate_sup1MB.pdf' % prefix)

    #plot cdf of duration
    args = []
    args.append((flows['flows_%s_down_yt' % index]['duration'],
                 'YouTube %s' % name))
    args.append(((flows['dipcp_%s_down_yt' % index]['FirstPacketDate']
                  - flows['dipcp_%s_down_yt' % index]['LastPacketDate']) / 1000,
                 'dipcp effective YouTube %s' % name))
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Flows Duration' % title,
                         _xlabel='Downstream Flows Duration in Seconds', _loc=2)
    pylab.savefig(output_path + '/%sflows_duration.pdf' %  prefix, format='pdf')

    #plot cdf of duration over flows larger than 1MB
    args = []
    args.append((flows['flows_%s_down_yt_1MB' % index]['duration'],
                 'YouTube %s' % name))
    args.append(((flows['dipcp_%s_down_yt' % index]['FirstPacketDate']
                  - flows['dipcp_%s_down_yt' % index]['LastPacketDate']),
                 'dipcp effective YouTube %s' % name))
    pylab.clf()
    cdfplot.cdfplotdataN(args,
            _title='%s Downstream Flows Duration for Flows \
                    larger than 1MBytes' % title,
                    _xlabel='Downstream Flows Duration in Bytes')
    pylab.savefig(output_path + '/%sflows_duration_sup1MB.pdf' %  prefix)

    #plot cdf of size
    args = []
    args.append((flows['flows_%s_down_yt' % index]['l3Bytes'],
        'YouTube %s' % name))
    args.append((flows['dipcp_%s_down_yt' % index]['DIP-Volume-Sum-Bytes-Down'],
        'dipcp YouTube %s' % name))
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Flows Size' % title,
            _xlabel='Downstream Flows Size in Bytes', _loc=2)
    pylab.savefig(output_path + '/%sflows_size.pdf' %  prefix, format='pdf')

    #plot cdf of size over flows larger than 1MB
    args = []
    args.append((flows['flows_%s_down_yt_1MB' % index]['l3Bytes'],
        'YouTube %s' % name))
    args.append((flows['dipcp_%s_down_yt_1MB' % index]
        ['DIP-Volume-Sum-Bytes-Down'], 'dipcp YouTube %s' % name))
    pylab.clf()
    cdfplot.cdfplotdataN(args,
            _title='%s Downstream Flows Size for Flows \
larger than 1MBytes' % title,
                    _xlabel='Downstream Flows Size in Bytes')
    pylab.savefig(output_path + '/%sflows_size_sup1MB.pdf' %  prefix)


#def filter_outsider_dipcp(data, path):
#    "Wrap filter_dipcp_yt_n_traces in case of pickle files"
#    trace_list = set(dataset.rpartition('_')[0] for dataset in data.keys())
#    new_data = {}
#    for trace in trace_list:
#        in_file = [f for f in os.listdir('%s/dipcp_output_%s' % (path, trace))
#                if f.find('_Flows_TOS_STR.npy') >= 0][0]
#        dipcp = np.load('%s/dipcp_output_%s/%s' % (path, trace, in_file))
#        new_data["%s_GVB" % trace] = data["%s_GVB" % trace]
#        new_data["%s_dipcp" % trace] = dipcp
#    return filter_dipcp_yt_n_traces(new_data)


def filter_dipcp_yt_n_traces(data, min_nb_flows=10):
    """Filter a data collection (imported from hdf5) to retrieve dipcp flows
    with min number of flows in the filtered data.
    Use as:
    data = tools.load_hdf5_data.load_h5_file('hdf5/lzf_data.h5')
    dipcp_flows = tools.plot_perfs.filter_dipcp_yt_n_traces(data)
"""
    #trace_list = set(dataset.rpartition('_')[0] for dataset in data.keys())
    flows_dipcp = {}
    flows_gvb = {}
    for trace in set(key.strip('_DIPCP').strip('_GVB') for key in data):
        gvb_flow = data["%s_GVB" % trace]
        gvb_flow_down = gvb_flow.compress(
            gvb_flow['direction'] == INDEX_VALUES.DOWN)
        dipcp_flow = data["%s_DIPCP" % trace]
        gvb_flow_streaming_down = gvb_flow_down.compress(
            gvb_flow_down['dscp'] == INDEX_VALUES.DSCP_HTTP_STREAM)
        del gvb_flow_down, gvb_flow
        flows_gvb['flows_%s_down_yt' % trace] = \
            gvb_flow_streaming_down.compress(
                    [x['asBGP'] in INDEX_VALUES.AS_YOUTUBE
                        for x in gvb_flow_streaming_down])
        flows_gvb['flows_%s_down_yt_1MB' % trace] = \
            flows_gvb['flows_%s_down_yt' % trace].compress(
            flows_gvb['flows_%s_down_yt' % trace]['l3Bytes'] > 10**6)
        flows_gvb['flows_%s_down_yt_eu' % trace] = \
            gvb_flow_streaming_down.compress(
                    [x['asBGP'] in INDEX_VALUES.AS_YOUTUBE_EU
                        for x in gvb_flow_streaming_down])
        flows_gvb['flows_%s_down_yt_eu_1MB' % trace] = \
            flows_gvb['flows_%s_down_yt_eu' % trace].compress(
            flows_gvb['flows_%s_down_yt_eu' % trace]['l3Bytes'] > 10**6)
        #dipcp flows
        # temp array storing dipcp flows http streaming (corresponding to GVB)
        #TMP_DIPCP = filter_dipcp_array(dipcp_flow, gvb_flow_streaming_down)
        flows_dipcp['dipcp_%s_down_yt' % trace] = filter_dipcp_array(
            dipcp_flow, flows_gvb['flows_%s_down_yt' % trace])
        flows_dipcp['dipcp_%s_down_yt_1MB' % trace] = \
            flows_dipcp['dipcp_%s_down_yt' % trace].compress(
                flows_dipcp['dipcp_%s_down_yt' % trace]
                ['DIP-Volume-Sum-Bytes-Down'] > 10**6)
        flows_dipcp['dipcp_%s_down_yt_eu' % trace] = \
            filter_dipcp_array(dipcp_flow, flows_gvb['flows_%s_down_yt_eu'
                % trace])
        flows_dipcp['dipcp_%s_down_yt_eu_1MB' % trace] = \
            flows_dipcp['dipcp_%s_down_yt_eu' % trace].compress(
                flows_dipcp['dipcp_%s_down_yt_eu' % trace]
                ['DIP-Volume-Sum-Bytes-Down'] > 10**6)
    # use of keys because modification of dict inside the loop
    for flows in flows_dipcp.keys():
        if len(flows_dipcp[flows]) < min_nb_flows:
            del flows_dipcp[flows]
    return flows_dipcp

def process_rtt_dipcp(flows_dipcp, rtt_type='Mean', direction='Up',
        prefix = 'YT_PERF_RTT_',
        output_path = 'rapport/BGP_figs/YT_YT_EU_dipcp',
        title = 'Youtube HTTP Streaming Flows:\n'):
    """Plots CDF of RTTs.
    Launch as:
    data = tools.load_hdf5_data.load_h5_file('hdf5/lzf_data.h5')
    dipcp_flows = tools.plot_perfs.filter_dipcp_yt_n_traces(data)
    tools.plot_perfs.process_rtt_dipcp(dipcp_flows)
    tools.plot_perfs.process_rtt_dipcp(dipcp_flows, rtt_type='Min')
"""
    dipcp_field = 'DIP-RTT-%s-ms-TCP-%s' % (rtt_type, direction)
    args = []
    args_1mb = []

    keys = flows_dipcp.keys()
    keys.sort()
    for name in keys:
        formatted_name = name.replace('dipcp_', '').replace('_down_', '_').\
                replace('_', ' ').upper()
        if name.find("1MB") == -1:
            args.append((flows_dipcp[name][dipcp_field], formatted_name))
        elif name.find("1MB") >= 0:
            args_1mb.append((flows_dipcp[name][dipcp_field], formatted_name))
        else:
            raise Exception, "index error"
    #plot cdf of RTT
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Flows RTT' % title,
            _xlabel='%s %s RTT in miliseconds' % (rtt_type, direction), _loc=2)
    pylab.savefig(output_path + '/%sflows_%s_rtt_%s.pdf' % (prefix, rtt_type,
        direction))
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Flows RTT' % title,
            _xlabel='%s %s RTT in miliseconds' % (rtt_type, direction))
    pylab.savefig(output_path + '/%sflows_%s_rtt_ccdf_%s.pdf' % (prefix,
        rtt_type, direction))

    #plot cdf of RTT over flows larger than 1MB
    pylab.clf()
    cdfplot.cdfplotdataN(args,
            _title='%s Downstream Flows RTT for Flows larger than 1MBytes'
            % title, _xlabel='%s %s RTT in miliseconds' % (rtt_type, direction))
    pylab.savefig(output_path + '/%sflows_%s_rtt_sup1MB_%s.pdf' % (prefix,
        rtt_type, direction))
    pylab.clf()
    cdfplot.ccdfplotdataN(args,
            _title='%s Downstream Flows RTT for Flows larger than 1MBytes'
            % title, _xlabel='%s %s RTT in miliseconds'
            % (rtt_type, direction), _loc=1)
    pylab.savefig(output_path + '/%sflows_%s_rtt_sup1MB_ccdf_%s.pdf' % (prefix,
        rtt_type, direction))
    pylab.clf()


def process_flows_dipcp(flows_dipcp, loss_type='DSQ', direction='Down',
        prefix = 'YT_PERF_', output_path = 'rapport/BGP_figs/YT_YT_EU_dipcp',
        title = 'Youtube HTTP Streaming Flows:\n'):
    """
    Plot cdf of field (TODO: adapt for non-loss field).
    Use as:
    data = tools.load_hdf5_data.load_h5_file('hdf5/lzf_data.h5')
    dipcp_flows = tools.plot_perfs.filter_dipcp_yt_n_traces(data)
    tools.plot_perfs.process_flows_dipcp(dipcp_flows)
"""
#    flows_name_list = flows_dipcp.keys()
#    flows_name_len = len(flows_name_list)

    dipcp_field = 'DIP-%s-NbMes-sec-TCP-%s' % (loss_type, direction)
    #plot cdf of down loss rate
    args = []
    args_1mb = []
#    for i in xrange(flows_name_len):
#	name = flows_name_list[i][-1] #name is last field
#	index = name.replace(' ', '_')
    for name in sorted(flows_dipcp.keys()):
        tmp = ([(0.0 + x[dipcp_field])
            / x['DIP-Volume-Number-Packets-Down'] for x in flows_dipcp[name]
            if x['DIP-Volume-Number-Packets-Down'] > 0], name.replace('_', ' '))
        if name.find("1MB") == -1:
            args.append(tmp)
        elif name.find("1MB") >= 0:
            args_1mb.append(tmp)
        else:
            raise Exception, "index error"
#        args.append(([(0.0 + x['DIP-RTM-NbMes-sec-TCP-Down'])
#                      / x['DIP-Volume-Number-Packets-Down']
#                      for x in flows['dipcp_%s_down_yt_eu' % index]
#                      if x['DIP-Volume-Number-Packets-Down']>0],
#                     'YouTube-EU %s' % name))
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s %s Loss Rate (%s)' % (title,
        direction, loss_type), _xlabel='Downstream Loss Rate in Percent',
        _loc=4)
    pylab.savefig(output_path + '/%sdipcp_down_loss_rate_%s.pdf'
            % (prefix, loss_type))
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s %s Loss Rate (%s)' % (title,
        direction, loss_type), _xlabel='Downstream Loss Rate in Percent',
        _loc=1)
    pylab.savefig(output_path + '/%sdipcp_down_loss_rate_%s_ccdf.pdf'
            % (prefix, loss_type))

    #plot cdf of down loss rate for >1MB flows
    pylab.clf()
    cdfplot.cdfplotdataN(args_1mb, _title='%s %s Loss Rate  (%s) for flows \
larger than 1MBytes' % (title, direction, loss_type),
            _xlabel='Downstream Loss Rate in Percent', _loc=4)
    pylab.savefig(output_path + '/%sdipcp_down_loss_rate_%s_sup1MB.pdf'
            % (prefix, loss_type))
    pylab.clf()
    cdfplot.ccdfplotdataN(args_1mb, _title='%s %s Loss Rate  (%s) for flows \
larger than 1MBytes' % (title, direction, loss_type),
            _xlabel='Downstream Loss Rate in Percent', _loc=1)
    pylab.savefig(output_path + '/%sdipcp_down_loss_rate_%s_ccdf_sup1MB.pdf'
            % (prefix, loss_type))
#
#    #plot cdf of up loss rate
#    args = []
#    for i in xrange(flows_name_len):
#	name = flows_name_list[i][-1] #name is last field
#	index = name.replace(' ', '_')
#        args.append(([(10**(-9) + x['DIP-RTM-NbMes-sec-TCP-Up'])
#                      / x['DIP-Volume-Number-Packets-Up']
#                      for x in flows['dipcp_%s_down_yt' % index]
#                      if x['DIP-Volume-Number-Packets-Up']>0],
#                     'YouTube %s' % name))
#        args.append(([(10**(-9) + x['DIP-RTM-NbMes-sec-TCP-Up'])
#                      / x['DIP-Volume-Number-Packets-Up']
#                      for x in flows['dipcp_%s_down_yt_eu' % index]
#                      if x['DIP-Volume-Number-Packets-Up']>0],
#                     'YouTube-EU %s' % name))
#    pylab.clf()
#    cdfplot.cdfplotdataN(args, _title='%s Upstream Loss Rate' % title,
#                         _xlabel='Upstream Loss Rate in Percent', _loc=2)
#    pylab.savefig(output_path + '/%sdipcp_up_loss_rate.pdf' % prefix)
#    pylab.clf()
#    cdfplot.ccdfplotdataN(args, _title='%s Upstream Loss Rate' % title,
#                          _xlabel='Upstream Loss Rate in Percent', _loc=3)
#    pylab.savefig(output_path + '/%sdipcp_up_loss_rate_ccdf.pdf' % prefix)
#
#    #plot cdf of up loss rate
#    args = []
#    for i in xrange(flows_name_len):
#	name = flows_name_list[i][-1] #name is last field
#	index = name.replace(' ', '_')
#        args.append(([(10**(-9) + x['DIP-RTM-NbMes-sec-TCP-Up'])
#            / x['DIP-Volume-Number-Packets-Up']
#                      for x in flows['dipcp_%s_down_yt_1MB' % index]
#                      if x['DIP-Volume-Number-Packets-Up']>0],
#                     'YouTube %s' % name))
#        args.append(([(10**(-9) + x['DIP-RTM-NbMes-sec-TCP-Up'])
#                      / x['DIP-Volume-Number-Packets-Up']
#                      for x in flows['dipcp_%s_down_yt_eu_1MB' % index]
#                      if x['DIP-Volume-Number-Packets-Up']>0],
#                     'YouTube-EU %s' % name))
#    pylab.clf()
#    cdfplot.cdfplotdataN(args, _title='%s Upstream Loss Rate for flows \
#            larger than 1MBytes' % title,
#            _xlabel='Upstream Loss Rate in Percent', _loc=2)
#    pylab.savefig(output_path + '/%sdipcp_up_loss_rate_sup1MB.pdf' % prefix)
#    pylab.clf()
#    cdfplot.ccdfplotdataN(args, _title='%s Upstream Loss Rate for flows \
#            larger than 1MBytes' % title,
#            _xlabel='Upstream Loss Rate in Percent', _loc=3)
#    pylab.savefig(output_path + '/%sdipcp_up_loss_rate_ccdf_sup_1MB.pdf' %
#            prefix)
#
