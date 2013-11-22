#!/usr/bin/env python
"Module to plot many relevant cdf for comparing traces."

from optparse import OptionParser
import numpy as np
import pylab

import INDEX_VALUES
import cdfplot

# 500kB
THRESHOLD = 500*10**3

def process_flows_YT_OT_2_traces(flows_A, flows_B, prefix = 'YT_OT',
        name_A = 'Data A', name_B = 'Data B',
        output_path = 'rapport/new_figs/YT_OT_2T',
        title = 'HTTP Streaming Downstream Flows:\n'):
    "Take 2 numpy record arrays to generate a lot of graphs related to streaming"
    flows_A_stream = flows_A.compress(flows_A.dscp == INDEX_VALUES.DSCP_HTTP_STREAM)
    flows_A_stream_down = flows_A_stream.compress(flows_A_stream.direction == INDEX_VALUES.DOWN)
    flows_A_down_yt = flows_A_stream_down.compress([x['asBGP'] in INDEX_VALUES.AS_YOUTUBE
                                               for x in flows_A_stream_down])
    flows_A_down_other = flows_A_stream_down.compress([x['asBGP'] not in INDEX_VALUES.AS_YOUTUBE
                                               for x in flows_A_stream_down])
    flows_A_down_yt_1MB = flows_A_down_yt.compress(flows_A_down_yt.l3Bytes > 10**6)
    flows_A_down_other_1MB = flows_A_down_other.compress(flows_A_down_other.l3Bytes > 10**6)

    flows_B_stream = flows_B.compress(flows_B.dscp == INDEX_VALUES.DSCP_HTTP_STREAM)
    flows_B_stream_down = flows_B_stream.compress(flows_B_stream.direction == INDEX_VALUES.DOWN)
    flows_B_down_yt = flows_B_stream_down.compress([x['asBGP'] in INDEX_VALUES.AS_YOUTUBE
                                               for x in flows_B_stream_down])
    flows_B_down_other = flows_B_stream_down.compress([x['asBGP'] not in INDEX_VALUES.AS_YOUTUBE
                                               for x in flows_B_stream_down])
    flows_B_down_yt_1MB = flows_B_down_yt.compress(flows_B_down_yt.l3Bytes > 10**6)
    flows_B_down_other_1MB = flows_B_down_other.compress(flows_B_down_other.l3Bytes > 10**6)


    #plot cdf of mean rate
    mean_rate_A_yt=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_A_down_yt if x['duration']>0]
    mean_rate_A_other=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_A_down_other if x['duration']>0]
    mean_rate_B_yt=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_B_down_yt if x['duration']>0]
    mean_rate_B_other=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_B_down_other if x['duration']>0]
    args = [(mean_rate_A_yt, 'YouTube %s' % name_A),
            (mean_rate_B_yt, 'YouTube %s' % name_B),
            (mean_rate_A_other, 'Other %s' % name_A),
            (mean_rate_B_other, 'Other %s' % name_B)]
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Mean Rate' % title,
                         _xlabel='Downstream Mean Rate in kbit/s', _loc=2)
    pylab.savefig(output_path + '/%sflows_mean_rate.pdf' % prefix, format='pdf')
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Mean Rate' % title, _xlabel='Downstream Mean Rate in kbit/s')
    pylab.savefig(output_path + '/%sflows_mean_rate_ccdf.pdf' % prefix, format='pdf')

    #plot cdf of mean rate over flows_A larger than 1MB
    mean_rate_A_sup1MB_yt=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_A_down_yt_1MB if x['duration']>0]
    mean_rate_A_sup1MB_other=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_A_down_other_1MB if x['duration']>0]
    mean_rate_B_sup1MB_yt=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_B_down_yt_1MB if x['duration']>0]
    mean_rate_B_sup1MB_other=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_B_down_other_1MB if x['duration']>0]
    args = [(mean_rate_A_sup1MB_yt, 'YouTube %s' % name_A),
            (mean_rate_B_sup1MB_yt, 'YouTube %s' % name_B),
            (mean_rate_A_sup1MB_other, 'Other %s' % name_A),
            (mean_rate_B_sup1MB_other, 'Other %s' % name_B)]
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Mean Rate for flows larger than 1MBytes' % title,
                 _xlabel='Downstream Mean Rate in kbit/s', _loc=2)
    pylab.savefig(output_path + '/%sflows_mean_rate_sup1MB.pdf' % prefix, format='pdf')
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Mean Rate for flows larger than 1MBytes' % title,
                 _xlabel='Downstream Mean Rate in kbit/s')
    pylab.savefig(output_path + '/%sflows_mean_rate_sup1MB_ccdf.pdf' % prefix, format='pdf')


    #plot cdf of peak rate
    #80*bytes/100ms => bit/s
    args = [(80*flows_A_down_yt.peakRate, 'YouTube %s' % name_A),
            (80*flows_B_down_yt.peakRate, 'YouTube %s' % name_B),
            (80*flows_A_down_other.peakRate, 'Other %s' % name_A),
            (80*flows_B_down_other.peakRate, 'Other %s' % name_B)]
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Peak Rate' % title,
                         _xlabel='Downstream Peak Rate in bit/s over 100 ms')
    pylab.savefig(output_path + '/%sflows_peak_rate.pdf' % prefix, format='pdf')
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Peak Rate' % title,
                         _xlabel='Downstream Peak Rate in bit/s over 100 ms')
    pylab.savefig(output_path + '/%sflows_peak_rate_ccdf.pdf' % prefix, format='pdf')

    #plot cdf of peak rate over flows larger than 1MB
    #80*bytes/100ms => bit/s
    args = [(80*flows_A_down_yt_1MB.peakRate, 'YouTube %s' % name_A),
            (80*flows_B_down_yt_1MB.peakRate, 'YouTube %s' % name_B),
            (80*flows_A_down_other_1MB.peakRate, 'Other %s' % name_A),
            (80*flows_B_down_other_1MB.peakRate, 'Other %s' % name_B)]
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Peak Rate for flows larger than 1MBytes' % title,
                 _xlabel='Downstream Peak Rate in bit/s over 100 ms', _loc=2)
    pylab.savefig(output_path + '/%sflows_peak_rate_sup1MB.pdf' % prefix, format='pdf')
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Peak Rate for flows larger than 1MBytes' % title,
                 _xlabel='Downstream Peak Rate in bit/s over 100 ms', _loc=1)
    pylab.savefig(output_path + '/%sflows_peak_rate_sup1MB_ccdf.pdf' % prefix, format='pdf')

    #plot cdf of duration
    args = [(flows_A_down_yt.duration, 'YouTube %s' % name_A),
            (flows_B_down_yt.duration, 'YouTube %s' % name_B),
            (flows_A_down_other.duration, 'Other %s' % name_A),
            (flows_B_down_other.duration, 'Other %s' % name_B)]
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Flows Duration' % title,
                         _xlabel='Downstream Flows Duration in Seconds', _loc=2)
    pylab.savefig(output_path + '/%sflows_duration.pdf' %  prefix, format='pdf')
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Flows Duration' % title,
                         _xlabel='Downstream Flows Duration in Seconds')
    pylab.savefig(output_path + '/%sflows_duration_ccdf.pdf' %  prefix, format='pdf')

    #plot cdf of size
    args = [(flows_A_down_yt.l3Bytes, 'YouTube %s' % name_A),
            (flows_B_down_yt.l3Bytes, 'YouTube %s' % name_B),
            (flows_A_down_other.l3Bytes, 'Other %s' % name_A),
            (flows_B_down_other.l3Bytes, 'Other %s' % name_B)]
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Flows Size' % title,
                         _xlabel='Downstream Flows Size in Bytes')
    pylab.savefig(output_path + '/%sflows_size.pdf' %  prefix, format='pdf')
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Flows Size' % title,
                          _xlabel='Downstream Flows Size in Bytes', _loc=1)
    pylab.savefig(output_path + '/%sflows_size_ccdf.pdf' %  prefix, format='pdf')

    #plot cdf of size over flows larger than 1MB
    args = [(flows_A_down_yt_1MB.l3Bytes, 'YouTube %s' %name_A),
            (flows_B_down_yt_1MB.l3Bytes, 'YouTube %s' %name_B),
            (flows_A_down_other_1MB.l3Bytes, 'Other %s' %name_A),
            (flows_B_down_other_1MB.l3Bytes, 'Other %s' %name_B)]
    pylab.clf()
    cdfplot.cdfplotdataN(args,
                         _title='%s Downstream Flows Size for Flows larger than 1MBytes' % title,
                         _xlabel='Downstream Flows Size in Bytes')
    pylab.savefig(output_path + '/%sflows_size_sup1MB.pdf' %  prefix, format='pdf')
    pylab.clf()
    cdfplot.ccdfplotdataN(args,
                          _title='%s Downstream Flows Size for Flows larger than 1MBytes' % title,
                          _xlabel='Downstream Flows Size in Bytes', _loc=1)
    pylab.savefig(output_path + '/%sflows_size_sup1MB_ccdf.pdf' %  prefix, format='pdf')


def process_flows_YT_4_traces(flows_A,
                              flows_B,
                              flows_C,
                              flows_D,
                              name_A = 'Data A',
                              name_B = 'Data B',
                              name_C = 'Data C',
                              name_D = 'Data D',
                              prefix = 'YT_4_',
                              output_path = 'rapport/BGP_figs/YT_4T',
                              title = 'Youtube HTTP Streaming Downstream Flows:\n'):
    "Take 4 numpy record arrays to generate a lot of graphs related to Youtube"

    flows_A_stream = flows_A.compress(flows_A.dscp == INDEX_VALUES.DSCP_HTTP_STREAM)
    flows_A_stream_down = flows_A_stream.compress(flows_A_stream.direction == INDEX_VALUES.DOWN)
    flows_A_down_yt = flows_A_stream_down.compress([x['asBGP'] in INDEX_VALUES.AS_YOUTUBE
                                               for x in flows_A_stream_down])
    flows_A_down_yt_1MB = flows_A_down_yt.compress(flows_A_down_yt.l3Bytes > 10**6)

    flows_B_stream = flows_B.compress(flows_B.dscp == INDEX_VALUES.DSCP_HTTP_STREAM)
    flows_B_stream_down = flows_B_stream.compress(flows_B_stream.direction == INDEX_VALUES.DOWN)
    flows_B_down_yt = flows_B_stream_down.compress([x['asBGP'] in INDEX_VALUES.AS_YOUTUBE
                                               for x in flows_B_stream_down])
    flows_B_down_yt_1MB = flows_B_down_yt.compress(flows_B_down_yt.l3Bytes > 10**6)

    flows_C_stream = flows_C.compress(flows_C.dscp == INDEX_VALUES.DSCP_HTTP_STREAM)
    flows_C_stream_down = flows_C_stream.compress(flows_C_stream.direction == INDEX_VALUES.DOWN)
    flows_C_down_yt = flows_C_stream_down.compress([x['asBGP'] in INDEX_VALUES.AS_YOUTUBE
                                               for x in flows_C_stream_down])
    flows_C_down_yt_1MB = flows_C_down_yt.compress(flows_C_down_yt.l3Bytes > 10**6)

    flows_D_stream = flows_D.compress(flows_D.dscp == INDEX_VALUES.DSCP_HTTP_STREAM)
    flows_D_stream_down = flows_D_stream.compress(flows_D_stream.direction == INDEX_VALUES.DOWN)
    flows_D_down_yt = flows_D_stream_down.compress([x['asBGP'] in INDEX_VALUES.AS_YOUTUBE
                                               for x in flows_D_stream_down])
    flows_D_down_yt_1MB = flows_D_down_yt.compress(flows_D_down_yt.l3Bytes > 10**6)


    #plot cdf of mean rate
    mean_rate_A_yt=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_A_down_yt if x['duration']>0]
    mean_rate_B_yt=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_B_down_yt if x['duration']>0]
    mean_rate_C_yt=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_C_down_yt if x['duration']>0]
    mean_rate_D_yt=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_D_down_yt if x['duration']>0]

    args = [(mean_rate_A_yt, 'YouTube %s' % name_A),
            (mean_rate_B_yt, 'YouTube %s' % name_B),
            (mean_rate_C_yt, 'YouTube %s' % name_C),
            (mean_rate_D_yt, 'YouTube %s' % name_D)]
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Mean Rate' % title,
                         _xlabel='Downstream Mean Rate in kbit/s', _loc=2)
    pylab.savefig(output_path + '/%sflows_mean_rate.pdf' % prefix, format='pdf')
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Mean Rate' % title, _xlabel='Downstream Mean Rate in kbit/s')
    pylab.savefig(output_path + '/%sflows_mean_rate_ccdf.pdf' % prefix, format='pdf')

    #plot cdf of mean rate over flows_A larger than 1MB
    mean_rate_A_sup1MB_yt=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_A_down_yt_1MB if x['duration']>0]
    mean_rate_B_sup1MB_yt=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_B_down_yt_1MB if x['duration']>0]
    mean_rate_C_sup1MB_yt=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_C_down_yt_1MB if x['duration']>0]
    mean_rate_D_sup1MB_yt=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_D_down_yt_1MB if x['duration']>0]

    args = [(mean_rate_A_sup1MB_yt, 'YouTube %s' % name_A),
            (mean_rate_B_sup1MB_yt, 'YouTube %s' % name_B),
            (mean_rate_C_sup1MB_yt, 'YouTube %s' % name_C),
            (mean_rate_D_sup1MB_yt, 'YouTube %s' % name_D)]
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Mean Rate for flows larger than 1MBytes' % title,
                 _xlabel='Downstream Mean Rate in kbit/s', _loc=2)
    pylab.savefig(output_path + '/%sflows_mean_rate_sup1MB.pdf' % prefix, format='pdf')
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Mean Rate for flows larger than 1MBytes' % title,
                 _xlabel='Downstream Mean Rate in kbit/s', _loc=3)
    pylab.savefig(output_path + '/%sflows_mean_rate_sup1MB_ccdf.pdf' % prefix, format='pdf')


    #plot cdf of peak rate
    #80*bytes/100ms => bit/s
    args = [(80*flows_A_down_yt.peakRate, 'YouTube %s' % name_A),
            (80*flows_B_down_yt.peakRate, 'YouTube %s' % name_B),
            (80*flows_C_down_yt.peakRate, 'YouTube %s' % name_C),
            (80*flows_D_down_yt.peakRate, 'YouTube %s' % name_D)]
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Peak Rate' % title,
                         _xlabel='Downstream Peak Rate in bit/s over 100 ms')
    pylab.savefig(output_path + '/%sflows_peak_rate.pdf' % prefix, format='pdf')
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Peak Rate' % title,
                         _xlabel='Downstream Peak Rate in bit/s over 100 ms')
    pylab.savefig(output_path + '/%sflows_peak_rate_ccdf.pdf' % prefix, format='pdf')

    #plot cdf of peak rate over flows larger than 1MB
    #80*bytes/100ms => bit/s
    args = [(80*flows_A_down_yt_1MB.peakRate, 'YouTube %s' % name_A),
            (80*flows_B_down_yt_1MB.peakRate, 'YouTube %s' % name_B),
            (80*flows_C_down_yt_1MB.peakRate, 'YouTube %s' % name_C),
            (80*flows_D_down_yt_1MB.peakRate, 'YouTube %s' % name_D)]
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Peak Rate for flows larger than 1MBytes' % title,
                 _xlabel='Downstream Peak Rate in bit/s over 100 ms', _loc=2)
    pylab.savefig(output_path + '/%sflows_peak_rate_sup1MB.pdf' % prefix, format='pdf')
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Peak Rate for flows larger than 1MBytes' % title,
                 _xlabel='Downstream Peak Rate in bit/s over 100 ms', _loc=1)
    pylab.savefig(output_path + '/%sflows_peak_rate_sup1MB_ccdf.pdf' % prefix, format='pdf')

    #plot cdf of duration
    args = [(flows_A_down_yt.duration, 'YouTube %s' % name_A),
            (flows_B_down_yt.duration, 'YouTube %s' % name_B),
            (flows_C_down_yt.duration, 'YouTube %s' % name_C),
            (flows_D_down_yt.duration, 'YouTube %s' % name_D)]
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Flows Duration' % title,
                         _xlabel='Downstream Flows Duration in Seconds', _loc=2)
    pylab.savefig(output_path + '/%sflows_duration.pdf' %  prefix, format='pdf')
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Flows Duration' % title,
                         _xlabel='Downstream Flows Duration in Seconds')
    pylab.savefig(output_path + '/%sflows_duration_ccdf.pdf' %  prefix, format='pdf')

    #plot cdf of size
    args = [(flows_A_down_yt.l3Bytes, 'YouTube %s' % name_A),
            (flows_B_down_yt.l3Bytes, 'YouTube %s' % name_B),
            (flows_C_down_yt.l3Bytes, 'YouTube %s' % name_C),
            (flows_D_down_yt.l3Bytes, 'YouTube %s' % name_D)]
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Flows Size' % title,
                         _xlabel='Downstream Flows Size in Bytes')
    pylab.savefig(output_path + '/%sflows_size.pdf' %  prefix, format='pdf')
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Flows Size' % title,
                          _xlabel='Downstream Flows Size in Bytes', _loc=1)
    pylab.savefig(output_path + '/%sflows_size_ccdf.pdf' %  prefix, format='pdf')

    #plot cdf of size over flows larger than 1MB
    args = [(flows_A_down_yt_1MB.l3Bytes, 'YouTube %s' %name_A),
            (flows_B_down_yt_1MB.l3Bytes, 'YouTube %s' %name_B),
            (flows_C_down_yt_1MB.l3Bytes, 'YouTube %s' %name_C),
            (flows_D_down_yt_1MB.l3Bytes, 'YouTube %s' %name_D)]
    pylab.clf()
    cdfplot.cdfplotdataN(args,
            _title='%s Downstream Flows Size for Flows larger than 1MBytes' % title,
            _xlabel='Downstream Flows Size in Bytes')
    pylab.savefig(output_path + '/%sflows_size_sup1MB.pdf' %  prefix, format='pdf')
    pylab.clf()
    cdfplot.ccdfplotdataN(args,
            _title='%s Downstream Flows Size for Flows larger than 1MBytes' % title,
            _xlabel='Downstream Flows Size in Bytes', _loc=1)
    pylab.savefig(output_path + '/%sflows_size_sup1MB_ccdf.pdf' %  prefix, format='pdf')


def process_flows_YT_YT_EU_N_traces(data,
        output_path = 'rapport/http_stats',
        prefix = 'YT_YT_EU_N_',
        title = 'Youtube Google HTTP Streaming Downstream Flows (only if > 10 flows):\n'):
    """Take N numpy record arrays to generate a lot of graphs related to Youtube
    Use as:
    data = tools.load_hdf5_data.load_h5_file('hdf5/lzf_data.h5')
    names = ((data['ADSL_2008_GVB'], 'ADSL_2008'),
        (data['FTTH_2008_GVB'], 'FTTH_2008'),
        (data['ADSL_nov_2009_GVB'], 'ADSL_nov_2009'),
        (data['FTTH_nov_2009_GVB'], 'FTTH_nov_2009'),
        (data['ADSL_dec_2009_GVB'], 'ADSL_dec_2009'),
        (data['FTTH_dec_2009_GVB'], 'FTTH_dec_2009'))
    tools.plot_all_figs_YT_vs_other.process_flows_YT_YT_EU_N_traces(names)
    """

    flows_tmp = {}
    flows = {}
    flows_1MB = {}
    for name, data_flow in data.items():
        flows_tmp['flows_%s_stream' % name] = data_flow.compress(
            data_flow['dscp'] == INDEX_VALUES.DSCP_HTTP_STREAM)
        flows_tmp['flows_%s_stream_down' % name] = flows_tmp['flows_%s_stream'
                % name].compress(flows_tmp['flows_%s_stream' % name]\
                        ['direction'] == INDEX_VALUES.DOWN)
        del flows_tmp['flows_%s_stream' % name]
        flows['%s_YT' % name] = flows_tmp['flows_%s_stream_down' % name].compress(
            [x['asBGP'] in INDEX_VALUES.AS_YOUTUBE
             for x in flows_tmp['flows_%s_stream_down' % name]])
        flows_1MB['%s_YT_1MB' % name] = flows['%s_YT' % name].compress(
            flows['%s_YT' % name]['l3Bytes'] > THRESHOLD)
        flows['%s_YT_EU' % name] = flows_tmp['flows_%s_stream_down' % name].compress(
            [x['asBGP'] in INDEX_VALUES.AS_YOUTUBE_EU
             for x in flows_tmp['flows_%s_stream_down' % name]])
        del flows_tmp['flows_%s_stream_down' % name]
        flows_1MB['%s_YT_EU_1MB' % name] = flows['%s_YT_EU' % name].compress(
            flows['%s_YT_EU' % name]['l3Bytes'] > THRESHOLD)
        # remove non signicative flows
        if len(flows['%s_YT' % name]) < 10:
            del flows['%s_YT' % name]
        if len(flows['%s_YT_EU' % name]) < 10:
            del flows['%s_YT_EU' % name]
        if len(flows_1MB['%s_YT_1MB' % name]) < 10:
            del flows_1MB['%s_YT_1MB' % name]
        if len(flows_1MB['%s_YT_EU_1MB' % name]) < 10:
            del flows_1MB['%s_YT_EU_1MB' % name]
    del flows_tmp
#    flows_name_len = len(flows_name_list)

    #plot cdf of mean rate
    args = []
    for name in sorted(flows.keys()):
        args.append((name, [8*x['l3Bytes']/(1000.0*x['duration'])
                      for x in flows[name]
                      if x['duration']>0]))
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Mean Rate' % title,
                         _xlabel='Downstream Mean Rate in kbit/s')
    pylab.savefig(output_path + '/%sflows_mean_rate.pdf' % prefix)
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Mean Rate' % title,
                          _xlabel='Downstream Mean Rate in kbit/s')
    pylab.savefig(output_path + '/%sflows_mean_rate_ccdf.pdf' % prefix)


    #plot cdf of mean rate over flows_A larger than 1MB
    args = []
    for name in sorted(flows_1MB.keys()):
        args.append((name, [8*x['l3Bytes']/(1000.0*x['duration'])
                      for x in flows_1MB[name]
                      if x['duration']>0]))
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Mean Rate for flows \
larger than 1MBytes' % title, _xlabel='Downstream Mean Rate in kbit/s')
    pylab.savefig(output_path + '/%sflows_mean_rate_sup1MB.pdf' % prefix)
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Mean Rate for flows \
larger than 1MBytes' % title, _xlabel='Downstream Mean Rate in kbit/s')
    pylab.savefig(output_path + '/%sflows_mean_rate_sup1MB_ccdf.pdf' % prefix)


    #plot cdf of peak rate
    #80*bytes/100ms => bit/s
    args = []
    for name in sorted(flows.keys()):
        args.append((name, 80*flows[name]['peakRate']))
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Peak Rate' % title,
                         _xlabel='Downstream Peak Rate in bit/s over 100 ms')
    pylab.savefig(output_path + '/%sflows_peak_rate.pdf' % prefix)
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Peak Rate' % title,
                          _xlabel='Downstream Peak Rate in bit/s over 100 ms')
    pylab.savefig(output_path + '/%sflows_peak_rate_ccdf.pdf' % prefix)

    #plot cdf of peak rate over flows larger than 1MB
    #80*bytes/100ms => bit/s
    args = []
    for name in sorted(flows_1MB.keys()):
        args.append((name, 80*flows_1MB[name]['peakRate']))
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Peak Rate for flows \
larger than 1MBytes' % title, _xlabel='Downstream Peak Rate in bit/s over 100 ms')
    pylab.savefig(output_path + '/%sflows_peak_rate_sup1MB.pdf' % prefix)
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Peak Rate for flows \
larger than 1MBytes' % title, _xlabel='Downstream Peak Rate in bit/s over 100 ms')
    pylab.savefig(output_path + '/%sflows_peak_rate_sup1MB_ccdf.pdf' % prefix)

    #plot cdf of duration
    args = []
    for name in sorted(flows.keys()):
        args.append((name, flows[name]['duration']))
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Flows Duration' % title,
                         _xlabel='Downstream Flows Duration in Seconds')
    pylab.savefig(output_path + '/%sflows_duration.pdf' %  prefix)
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Flows Duration' % title,
                          _xlabel='Downstream Flows Duration in Seconds')
    pylab.savefig(output_path + '/%sflows_duration_ccdf.pdf' %  prefix)

    #plot cdf of duration over flows larger than 1MB
    args = []
    for name in sorted(flows_1MB.keys()):
        args.append((name, flows_1MB[name]['duration']))
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Flows Duration for \
Flows larger than 1MBytes' % title,
                         _xlabel='Downstream Flows Duration in Seconds')
    pylab.savefig(output_path + '/%sflows_duration_sup1MB.pdf' %  prefix)
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Flows Duration for \
Flows larger than 1MBytes' % title,
                          _xlabel='Downstream Flows Duration in Seconds')
    pylab.savefig(output_path + '/%sflows_duration_sup1MB_ccdf.pdf' %  prefix)

    #plot cdf of size
    args = []
    for name in sorted(flows.keys()):
        args.append((name, flows[name]['l3Bytes']))
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Flows Size' % title,
                         _xlabel='Downstream Flows Size in Bytes')
    pylab.savefig(output_path + '/%sflows_size.pdf' %  prefix)
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Flows Size' % title,
                          _xlabel='Downstream Flows Size in Bytes')
    pylab.savefig(output_path + '/%sflows_size_ccdf.pdf' %  prefix)

    #plot cdf of size over flows larger than 1MB
    args = []
    for name in sorted(flows_1MB.keys()):
        args.append((name, flows_1MB[name]['l3Bytes']))
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Downstream Flows Size for Flows \
 larger than 1MBytes' % title, _xlabel='Downstream Flows Size in Bytes')
    pylab.savefig(output_path + '/%sflows_size_sup1MB.pdf' %  prefix)
    pylab.clf()
    cdfplot.ccdfplotdataN(args, _title='%s Downstream Flows Size for Flows \
larger than 1MBytes' % title, _xlabel='Downstream Flows Size in Bytes')
    pylab.savefig(output_path + '/%sflows_size_sup1MB_ccdf.pdf' %  prefix)


def process_flows_YT_OT(flows, prefix='YT_OT',
                   output_path='rapport/new_figs/YT_OT', title='HTTP Streaming Downstream Flows\n'):

    flows_down = flows.compress(flows.direction == INDEX_VALUES.DOWN)
    #inefficient!
    flows_down_yt = np.array([x for x in flows_down if x['asBGP'] in INDEX_VALUES.AS_YOUTUBE],
                                dtype = INDEX_VALUES.dtype_GVB_AS).view(np.recarray)
    flows_down_other = np.array([x for x in flows_down if x['asBGP'] not in INDEX_VALUES.AS_YOUTUBE],
                                dtype = INDEX_VALUES.dtype_GVB_AS).view(np.recarray)
#should be like this
#    flows_down_yt = flows_down.compress(flows_down.asBGP in INDEX_VALUES.AS_YOUTUBE)
#    flows_down_yt = flows_down.compress(flows_down.asBGP not in INDEX_VALUES.AS_YOUTUBE)

    flows_down_yt_1MB = flows_down_yt.compress(flows_down_yt.l3Bytes > 10**6)
    flows_down_other_1MB = flows_down_other.compress(flows_down_other.l3Bytes > 10**6)

    #plot cdf of mean rate
    pylab.clf()
    mean_rate_yt=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_down_yt if x['duration']>0]
    mean_rate_other=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_down_other if x['duration']>0]
    args = [(mean_rate_yt, 'YouTube'), (mean_rate_other, 'Other')]
    cdfplot.cdfplotdataN(args, _title='%s Downstream Mean Rate' % title, _xlabel='Downstream Mean Rate in kbit/s')
    pylab.savefig(output_path + '/%sflows_mean_rate.pdf' % prefix, format='pdf')

    #plot cdf of mean rate over flows larger than 1MB
    pylab.clf()
    mean_rate_sup1MB_yt=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_down_yt_1MB if x['duration']>0]
    mean_rate_sup1MB_other=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_down_other_1MB if x['duration']>0]
    args = [(mean_rate_sup1MB_yt, 'YouTube'), (mean_rate_sup1MB_other, 'Other')]
    cdfplot.cdfplotdataN(args, _title='%s Downstream Mean Rate for flows larger than 1MBytes' % title,
                 _xlabel='Downstream Mean Rate in kbit/s')
    pylab.savefig(output_path + '/%sflows_mean_rate_sup1MB.pdf' % prefix, format='pdf')


    #plot cdf of peak rate
    pylab.clf()
    #80*bytes/100ms => bit/s
    args = [(80*flows_down_yt.peakRate, 'YouTube'), (80*flows_down_other.peakRate, 'Other')]
    cdfplot.cdfplotdataN(args, _title='%s Downstream Peak Rate' % title, _xlabel='Downstream Mean Rate in bit/s over 100 ms')
    pylab.savefig(output_path + '/%sflows_peak_rate.pdf' % prefix, format='pdf')

    #plot cdf of peak rate over flows larger than 1MB
    pylab.clf()
    #80*bytes/100ms => bit/s
    args = [(80*flows_down_yt_1MB.peakRate, 'YouTube'), (80*flows_down_other_1MB.peakRate, 'Other')]
    cdfplot.cdfplotdataN(args, _title='%s Downstream Peak Rate for flows larger than 1MBytes' % title,
                 _xlabel='Downstream Mean Rate in bit/s over 100 ms')
    pylab.savefig(output_path + '/%sflows_peak_rate_sup1MB.pdf' % prefix, format='pdf')

    #plot cdf of duration
    pylab.clf()
    args = [(flows_down_yt.duration, 'YouTube'), (flows_down_other.duration, 'Other')]
    cdfplot.cdfplotdataN(args, _title='%s Downstream Flows Duration' % title, _xlabel='Downstream Flows Duration in Seconds')
    pylab.savefig(output_path + '/%sflows_duration.pdf' %  prefix, format='pdf')

    #plot cdf of size
    pylab.clf()
    args = [(flows_down_yt.l3Bytes, 'YouTube'), (flows_down_other.l3Bytes, 'Other')]
    cdfplot.cdfplotdataN(args, _title='%s Downstream Flows Size' % title, _xlabel='Downstream Flows Size in Bytes')
    pylab.savefig(output_path + '/%sflows_size.pdf' %  prefix, format='pdf')

    #plot cdf of size over flows larger than 1MB
    pylab.clf()
    args = [(flows_down_yt_1MB.l3Bytes, 'YouTube'), (flows_down_other_1MB.l3Bytes, 'Other')]
    cdfplot.cdfplotdataN(args, _title='%s Downstream Flows Size for Flows larger than 1MBytes' % title,
                 _xlabel='Downstream Flows Size in Bytes')
    pylab.savefig(output_path + '/%sflows_size_sup1MB.pdf' %  prefix, format='pdf')

def plot_from_file(file):
    process_flows_YT_OT(np.load(file))

def main():
    usage = "%prog -r data_file"

    parser = OptionParser(usage = usage)
    parser.add_option("-r", dest = "file_list", type = "string",
                      help = "input data file")
    (options, args) = parser.parse_args()

    if not options.file:
        parser.print_help()
        exit()

    plot_from_file(options.file)

if __name__ == '__main__':
    main()
