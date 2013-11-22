#!/usr/bin/env python
from INDEX_VALUES import *

from cdfplot import *
#from compute_AT import *

def process_flows_streaming(flows_ftth, prefix='FTTH_Streaming_',
			    output_path='rapport/figs/FTTH_streaming',
			    title='July 2008 FTTH Streaming Flows:\n'):

    flows_down_ftth=[x for x in flows_ftth if x['direction']==DOWN]

    flows_down_ac_ftth=pylab.array([x for x in flows_down_ftth
				    if (x['AS'] in AS_ACRONOC)],
				   dtype=dtype_GVB)
    flows_down_dm_ftth=pylab.array([x for x in flows_down_ftth
				    if (x['AS'] in AS_DAILYMOTION)],
				   dtype=dtype_GVB)
    flows_down_go_ftth=pylab.array([x for x in flows_down_ftth
				    if (x['AS'] in AS_GOOGLE)],
				   dtype=dtype_GVB)
    flows_down_ll_ftth=pylab.array([x for x in flows_down_ftth
				    if (x['AS'] in AS_LIMELIGHT)],
				   dtype=dtype_GVB)
    flows_down_yt_ftth=pylab.array([x for x in flows_down_ftth
				    if (x['AS'] in AS_YOUTUBE)],
				   dtype=dtype_GVB)



    #plot cdf of mean rate
#    pylab.clf()
#    mean_rate_ac_ftth=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_down_ac_ftth if x['duration']>0]
#    mean_rate_dm_ftth=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_down_dm_ftth if x['duration']>0]
#    mean_rate_go_ftth=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_down_go_ftth if x['duration']>0]
#    mean_rate_ll_ftth=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_down_ll_ftth if x['duration']>0]
#    mean_rate_yt_ftth=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_down_yt_ftth if x['duration']>0]
#    args = [(mean_rate_ac_ftth, 'Acronoc'), (mean_rate_dm_ftth, 'DailyMotion'), \
#                (mean_rate_go_ftth, 'Google'), (mean_rate_ll_ftth, 'Limelight'), (mean_rate_yt_ftth, 'YouTube')]
#    cdfplotdataN(args, _title='%s Downstream Mean Rate' % title, _xlabel='Downstream Mean Rate in kbit/s')
#    pylab.savefig(output_path + '/%sflows_mean_rate.pdf' % prefix, format='pdf')
#
#    #plot cdf of mean rate over flows larger than 1MB
#    pylab.clf()
#    mean_rate_sup1MB_ac_ftth=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_down_ac_ftth if x['duration']>0 and x['l3Bytes']>10**6]
#    mean_rate_sup1MB_dm_ftth=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_down_dm_ftth if x['duration']>0 and x['l3Bytes']>10**6]
#    mean_rate_sup1MB_go_ftth=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_down_go_ftth if x['duration']>0 and x['l3Bytes']>10**6]
#    mean_rate_sup1MB_ll_ftth=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_down_ll_ftth if x['duration']>0 and x['l3Bytes']>10**6]
#    mean_rate_sup1MB_yt_ftth=[8*x['l3Bytes']/(1000.0*x['duration']) for x in flows_down_yt_ftth if x['duration']>0 and x['l3Bytes']>10**6]
#    args = [(mean_rate_sup1MB_ac_ftth, 'Acronoc'), (mean_rate_sup1MB_dm_ftth, 'DailyMotion'), \
#                (mean_rate_sup1MB_go_ftth, 'Google'), (mean_rate_sup1MB_ll_ftth, 'Limelight'), \
#                (mean_rate_sup1MB_yt_ftth, 'YouTube')]
#    cdfplotdataN(args, _title='%s Downstream Mean Rate for Flows larger than 1MBytes' % title, \
#                     _xlabel='Downstream Mean Rate in kbit/s', _loc=2)
#    pylab.savefig(output_path + '/%sflows_mean_rate_sup1MB.pdf' % prefix)

    #plot cdf of peak rate
    pylab.clf()
    #80*bytes/100ms => bit/s
    peak_ac_ftth=80*flows_down_ac_ftth['peakRate']
    peak_dm_ftth=80*flows_down_dm_ftth['peakRate']
    peak_go_ftth=80*flows_down_go_ftth['peakRate']
    peak_ll_ftth=80*flows_down_ll_ftth['peakRate']
    peak_yt_ftth=80*flows_down_yt_ftth['peakRate']
    args = [(peak_ac_ftth, 'Acronoc'), (peak_dm_ftth, 'DailyMotion'), \
                (peak_go_ftth, 'Google'), (peak_ll_ftth, 'Limelight'), (peak_yt_ftth, 'YouTube')]
    cdfplotdataN(args, _title='%s Downstream Peak Rate' % title, _xlabel='Downstream Peak Rate in kbit/s')
    pylab.savefig(output_path + '/%sflows_peak_rate.pdf' % prefix)

    #plot cdf of peak rate over flows larger than 1MB
    pylab.clf()
    #80*bytes/100ms => bit/s
    peak_ac_sup1MB_ftth=[80*x['peakRate'] for x in flows_down_ac_ftth if x['l3Bytes']>10**6]
    peak_dm_sup1MB_ftth=[80*x['peakRate'] for x in flows_down_dm_ftth if x['l3Bytes']>10**6]
    peak_go_sup1MB_ftth=[80*x['peakRate'] for x in flows_down_go_ftth if x['l3Bytes']>10**6]
    peak_ll_sup1MB_ftth=[80*x['peakRate'] for x in flows_down_ll_ftth if x['l3Bytes']>10**6]
    peak_yt_sup1MB_ftth=[80*x['peakRate'] for x in flows_down_yt_ftth if x['l3Bytes']>10**6]
    args = [(peak_ac_sup1MB_ftth, 'Acronoc'), (peak_dm_sup1MB_ftth, 'DailyMotion'), \
                (peak_go_sup1MB_ftth, 'Google'), (peak_ll_sup1MB_ftth, 'Limelight'), \
                (peak_yt_sup1MB_ftth, 'YouTube')]
    cdfplotdataN(args, _title='%s Downstream Peak Rate for Flows larger than 1MBytes' % title, \
                     _xlabel='Downstream Peak Rate in kbit/s', _loc=2)
    pylab.savefig(output_path + '/%sflows_peak_rate_sup1MB.pdf' % prefix)

    #plot cdf of duration
    pylab.clf()
    args = [(flows_down_ac_ftth['duration'], 'Acronoc'), (flows_down_dm_ftth['duration'], 'DailyMotion'), \
                (flows_down_go_ftth['duration'], 'Google'), (flows_down_ll_ftth['duration'], 'Limelight'), \
                (flows_down_yt_ftth['duration'], 'YouTube')]
    cdfplotdataN(args, _title='%s Downstream Duration' % title, _xlabel='Downstream Duration in Seconds', _loc=2)
    pylab.savefig(output_path + '/%sflows_duration.pdf' %  prefix)

    #plot cdf of size
    pylab.clf()
    args = [(flows_down_ac_ftth['l3Bytes'], 'Acronoc'), (flows_down_dm_ftth['l3Bytes'], 'DailyMotion'), \
                (flows_down_go_ftth['l3Bytes'], 'Google'), (flows_down_ll_ftth['l3Bytes'], 'Limelight'), \
                (flows_down_yt_ftth['l3Bytes'], 'YouTube')]
    cdfplotdataN(args, _title='%s Downstream Size' % title, _xlabel='Downstream Size in Bytes')
    pylab.savefig(output_path + '/%sflows_size.pdf' %  prefix, format='pdf')

    #plot cdf of size over flows larger than 1MB
    pylab.clf()
    size_ac_sup1MB_ftth=[b for b in flows_down_ac_ftth['l3Bytes'] if b>10**6]
    size_dm_sup1MB_ftth=[b for b in flows_down_dm_ftth['l3Bytes'] if b>10**6]
    size_go_sup1MB_ftth=[b for b in flows_down_go_ftth['l3Bytes'] if b>10**6]
    size_ll_sup1MB_ftth=[b for b in flows_down_ll_ftth['l3Bytes'] if b>10**6]
    size_yt_sup1MB_ftth=[b for b in flows_down_yt_ftth['l3Bytes'] if b>10**6]
    args = [(size_ac_sup1MB_ftth, 'Acronoc'), (size_dm_sup1MB_ftth, 'DailyMotion'), \
                (size_go_sup1MB_ftth, 'Google'), (size_ll_sup1MB_ftth, 'Limelight'), \
                (size_yt_sup1MB_ftth, 'YouTube')]
    cdfplotdataN(args, _title='%s Downstream Size for Flows larger than 1MBytes' % title, \
                     _xlabel='Downstream Size in Bytes')
    pylab.savefig(output_path + '/%sflows_size_sup1MB.pdf' %  prefix, format='pdf')

    #plot cdf of size over flows larger than 10kB
    pylab.clf()
    size_ac_sup10kB_ftth=[b for b in flows_down_ac_ftth['l3Bytes'] if b>10**4]
    size_dm_sup10kB_ftth=[b for b in flows_down_dm_ftth['l3Bytes'] if b>10**4]
    size_go_sup10kB_ftth=[b for b in flows_down_go_ftth['l3Bytes'] if b>10**4]
    size_ll_sup10kB_ftth=[b for b in flows_down_ll_ftth['l3Bytes'] if b>10**4]
    size_yt_sup10kB_ftth=[b for b in flows_down_yt_ftth['l3Bytes'] if b>10**4]
    args = [(size_ac_sup10kB_ftth, 'Acronoc'), (size_dm_sup10kB_ftth, 'DailyMotion'), \
                (size_go_sup10kB_ftth, 'Google'), (size_ll_sup10kB_ftth, 'Limelight'), \
                (size_yt_sup10kB_ftth, 'YouTube')]
    cdfplotdataN(args, _title='%s Downstream Size for Flows larger than 10kBytes' % title, \
                     _xlabel='Downstream Size in Bytes')
    pylab.savefig(output_path + '/%sflows_size_sup10kB.pdf' %  prefix)
