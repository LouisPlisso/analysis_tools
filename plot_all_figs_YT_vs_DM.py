from INDEX_VALUES import *


from cdfplot import *
from compute_AT import *

def process_flows2(flows_adsl, flows_ftth, prefix='AI1_', output_path='rapport/figs', title=''):

    flows_down_adsl=[x for x in flows_adsl if x[DIR]==1]
    flows_down_yt_adsl=pylab.array([x for x in flows_down_adsl if ((x[AS]==36561)or(x[AS]==43515))])
    flows_down_dm_adsl=pylab.array([x for x in flows_down_adsl if (x[AS]==41690)])

    flows_down_ftth=[x for x in flows_ftth if x[DIR]==1]
    flows_down_yt_ftth=pylab.array([x for x in flows_down_ftth if ((x[AS]==36561)or(x[AS]==43515))])
    flows_down_dm_ftth=pylab.array([x for x in flows_down_ftth if (x[AS]==41690)])


    #plot cdf of mean rate
    pylab.clf()
    mean_rate_yt_adsl=[8*x[BYTES]/(1000.0*x[DUR]) for x in flows_down_yt_adsl if x[DUR]>0] 
    mean_rate_dm_adsl=[8*x[BYTES]/(1000.0*x[DUR]) for x in flows_down_dm_adsl if x[DUR]>0] 
    mean_rate_yt_ftth=[8*x[BYTES]/(1000.0*x[DUR]) for x in flows_down_yt_ftth if x[DUR]>0] 
    mean_rate_dm_ftth=[8*x[BYTES]/(1000.0*x[DUR]) for x in flows_down_dm_ftth if x[DUR]>0] 
    args = [(mean_rate_yt_adsl, 'YouTube ADSL'), (mean_rate_yt_ftth, 'YouTube FTTH'), \
                      (mean_rate_dm_adsl, 'DailyMotion ADSL'), (mean_rate_dm_ftth, 'DailyMotion FTTH')]
    cdfplotdataN(args, _title='%s Downstream Mean Rate' % title, _xlabel='Downstream Mean Rate in kbit/s')
#    cdfplotdata(mean_rate_yt_adsl, _name='YouTube ADSL')
#    cdfplotdata(mean_rate_yt_ftth, _name='YouTube FTTH', _ls='-.')
#    cdfplotdata(mean_rate_dm_adsl, _name='DailyMotion ADSL', _ls=':', _lw=3)
#    cdfplotdata(mean_rate_dm_ftth, _name='DailyMotion FTTH', _title='%s Downstream Mean Rate' % title, \
#                    _xlabel='Downstream Mean Rate in kbit/s', _ls='--', _lw=3)
#    setgraph_log(_loc=4)
    pylab.savefig(output_path + '/%sflows_mean_rate.pdf' % prefix, format='pdf')
    return

    #plot cdf of mean rate over flows larger than 1MB
    pylab.clf()
    mean_rate_sup1MB_yt_adsl=[8*x[BYTES]/(1000.0*x[DUR]) for x in flows_down_yt_adsl if x[DUR]>0 and x[BYTES]>10**6] 
    mean_rate_sup1MB_dm_adsl=[8*x[BYTES]/(1000.0*x[DUR]) for x in flows_down_dm_adsl if x[DUR]>0 and x[BYTES]>10**6] 
    mean_rate_sup1MB_yt_ftth=[8*x[BYTES]/(1000.0*x[DUR]) for x in flows_down_yt_ftth if x[DUR]>0 and x[BYTES]>10**6] 
    mean_rate_sup1MB_dm_ftth=[8*x[BYTES]/(1000.0*x[DUR]) for x in flows_down_dm_ftth if x[DUR]>0 and x[BYTES]>10**6] 
    cdfplotdata(mean_rate_sup1MB_yt_adsl, _name='YouTube ADSL')
    cdfplotdata(mean_rate_sup1MB_yt_ftth, _name='YouTube FTTH', _ls='-.')
    cdfplotdata(mean_rate_sup1MB_dm_adsl, _name='DailyMotion ADSL', _ls=':', _lw=3)
    cdfplotdata(mean_rate_sup1MB_dm_ftth, _name='DailyMotion FTTH', \
                    _title='%s Downstream Mean Rate for flows larger than 1MBytes' % title, \
                    _xlabel='Downstream Mean Rate in kbit/s', _ls='--', _lw=3)
    setgraph_log(_loc=4)
    pylab.savefig(output_path + '/%sflows_mean_rate_sup1MB.pdf' % prefix, format='pdf')


    #plot cdf of peak rate
    pylab.clf()
    #80*bytes/100ms => bit/s
    peak_yt_adsl=80*flows_down_yt_adsl[:,PEAK]
    peak_dm_adsl=80*flows_down_dm_adsl[:,PEAK]
    peak_yt_ftth=80*flows_down_yt_ftth[:,PEAK]
    peak_dm_ftth=80*flows_down_dm_ftth[:,PEAK]
    cdfplotdata(peak_yt_adsl, _name='YouTube ADSL')
    cdfplotdata(peak_yt_ftth, _name='YouTube FTTH', _ls='-.')
    cdfplotdata(peak_dm_adsl, _name='DailyMotion ADSL', _ls=':', _lw=3)
    cdfplotdata(peak_dm_ftth, _name='DailyMotion FTTH', _title='%s Downstream Peak Rate' % title, \
                    _xlabel='Downstream Peak Rate in bit/s over 100ms', _ls='--', _lw=3)
    setgraph_log(_loc=4)
    pylab.savefig(output_path + '/%sflows_peak_rate.pdf' % prefix, format='pdf')

    #plot cdf of peak rate over flows larger than 1MB
    pylab.clf()
    #80*bytes/100ms => bit/s
    peak_yt_sup10kB_adsl=[80*x[PEAK] for x in flows_down_yt_adsl if x[BYTES]>10**4]
    peak_dm_sup10kB_adsl=[80*x[PEAK] for x in flows_down_dm_adsl if x[BYTES]>10**4]
    peak_yt_sup10kB_ftth=[80*x[PEAK] for x in flows_down_yt_ftth if x[BYTES]>10**4]
    peak_dm_sup10kB_ftth=[80*x[PEAK] for x in flows_down_dm_ftth if x[BYTES]>10**4]
    cdfplotdata(peak_yt_sup10kB_adsl, _name='YouTube ADSL')
    cdfplotdata(peak_yt_sup10kB_ftth, _name='YouTube FTTH', _ls='-.')
    cdfplotdata(peak_dm_sup10kB_adsl, _name='DailyMotion ADSL', _ls=':', _lw=3)
    cdfplotdata(peak_dm_sup10kB_ftth, _name='DailyMotion FTTH', _title='%s Downstream Peak Rate' % title, \
                    _xlabel='Downstream Peak Rate in bit/s over 100ms', _ls='--', _lw=3)
    setgraph_log(_loc=4)
    pylab.savefig(output_path + '/%sflows_peak_rate_sup1MB.pdf' % prefix, format='pdf')

    #plot cdf of duration
    pylab.clf()
    cdfplotdata(flows_down_yt_adsl[:,DUR],_name='YouTube ADSL')
    cdfplotdata(flows_down_yt_ftth[:,DUR],_name='YouTube FTTH', _ls='-.')
    cdfplotdata(flows_down_dm_adsl[:,DUR],_name='DailyMotion ADSL', _ls=':', _lw=3)
    cdfplotdata(flows_down_dm_ftth[:,DUR],_name='DailyMotion FTTH', _xlabel='Downstream Flows Duration in Seconds', \
                    _title='%s Downstream Flows Duration' % title, _ls='--', _lw=3)
    setgraph_log(_loc=4)
    pylab.savefig(output_path + '/%sflows_duration.pdf' %  prefix, format='pdf')

    #plot cdf of size
    pylab.clf()
    cdfplotdata(flows_down_yt_adsl[:,BYTES],_name='YouTube ADSL')
    cdfplotdata(flows_down_yt_ftth[:,BYTES],_name='YouTube FTTH', _ls='-.')
    cdfplotdata(flows_down_dm_adsl[:,BYTES],_name='DailyMotion ADSL', _ls=':', _lw=3)
    cdfplotdata(flows_down_dm_ftth[:,BYTES],_name='DailyMotion FTTH',_ls='--',_title='%s Downstream Flow Size' % title, \
                    _xlabel='Downstream Flow Size in Bytes', _lw=3)
    setgraph_log(_loc=4)
    pylab.savefig(output_path + '/%sflows_size.pdf' %  prefix, format='pdf')

    #plot cdf of size over flows larger than 1MB
    pylab.clf()
    cdfplotdata([b for b in flows_down_yt_adsl[:,BYTES] if b>10**6], _name='YouTube ADSL')
    cdfplotdata([b for b in flows_down_yt_ftth[:,BYTES] if b>10**6], _name='YouTube FTTH', _ls='-.')
    cdfplotdata([b for b in flows_down_dm_adsl[:,BYTES] if b>10**6], _name='DailyMotion ADSL', _ls=':', _lw=3)
    cdfplotdata([b for b in flows_down_dm_ftth[:,BYTES] if b>10**6], _name='DailyMotion FTTH', \
                    _ls='--', _title='FTTH Downstream Flow Size for Flows larger than 1MBytes', \
                    _xlabel='Downstream Flow Size in Bytes', _lw=3)
    setgraph_log(_loc=4)
    pylab.savefig(output_path + '/%sflows_size_sup1MB.pdf' %  prefix, format='pdf')

    #plot cdf of size over flows larger than 10kB
    pylab.clf()
    cdfplotdata([b for b in flows_down_yt_adsl[:,BYTES] if b>10**4], _name='YouTube ADSL')
    cdfplotdata([b for b in flows_down_yt_ftth[:,BYTES] if b>10**4], _name='YouTube FTTH', _ls='-.')
    cdfplotdata([b for b in flows_down_dm_adsl[:,BYTES] if b>10**4], _name='DailyMotion ADSL', _ls=':', _lw=3)
    cdfplotdata([b for b in flows_down_dm_ftth[:,BYTES] if b>10**4], _name='DailyMotion FTTH', \
                    _ls='--', _title='FTTH Downstream Flow Size for Flows larger than 10kBytes', \
                    _xlabel='Downstream Flow Size in Bytes', _lw=3)
    setgraph_log(_loc=4)
    pylab.savefig(output_path + '/%sflows_size_sup10kB.pdf' %  prefix, format='pdf')
