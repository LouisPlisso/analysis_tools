#!/usr/bin/python



#from INDEX_VALUES import *

#from cdfplot import *
#from compute_AT import *


def process_flows(flows, prefix='_', output_path='rapport/figs', title=''):

    flows_down=[x for x in flows if x[DIR]==1]
    flows_down_yt=pylab.array([x for x in flows_down 
			       if ((x[AS]==36561)or(x[AS]==43515))])
    flows_down_dm=pylab.array([x for x in flows_down if (x[AS]==41690)])

    #plot cdf of mean rate
    pylab.clf()
    mean_rate_yt=[8*x[BYTES]/(1000.0*x[DUR]) 
		  for x in flows_down_yt if x[DUR]>0] 
    mean_rate_dm=[8*x[BYTES]/(1000.0*x[DUR]) 
		  for x in flows_down_dm if x[DUR]>0] 
    cdfplotdata(mean_rate_yt, _name='YouTube')
    cdfplotdata(mean_rate_dm, _name='DailyMotion', 
		_title='%s Downstream Mean Rate' % title, 
		_xlabel='Downstream Mean Rate in kbit/s', _ls='--')
    setgraph_log()
    return
    pylab.savefig(output_path + '/%sflows_mean_rate.pdf' % prefix, 
		  format='pdf')

    #plot cdf of mean rate over flows larger than 1MB
    pylab.clf()
    mean_rate_sup1MB_yt=[8*x[BYTES]/(1000.0*x[DUR]) 
			 for x in flows_down_yt if x[DUR]>0 and x[BYTES]>10**6] 
    mean_rate_sup1MB_dm=[8*x[BYTES]/(1000.0*x[DUR]) 
			 for x in flows_down_dm if x[DUR]>0 and x[BYTES]>10**6] 
    cdfplotdata(mean_rate_sup1MB_yt, _name='YouTube')
    cdfplotdata(mean_rate_sup1MB_dm, _name='DailyMotion', 
		_title='%s Downstream Mean Rate for flows larger than 1MBytes' 
		% title, 
		_xlabel='Downstream Mean Rate in kbit/s', _ls='--')
    setgraph_log()
    pylab.savefig(output_path + '/%sflows_mean_rate_sup1MB.pdf' % prefix, 
		  format='pdf')

    #plot cdf of peak rate
    pylab.clf()
    #80*bytes/100ms => bit/s
    peak_yt=80*flows_down_yt[:,PEAK]
    peak_dm=80*flows_down_dm[:,PEAK]
    cdfplotdata(peak_yt, _name='YouTube')
    cdfplotdata(peak_dm, _name='DailyMotion', 
		_title='%s Downstream Peak Rate' % title, 
		_xlabel='Downstream Peak Rate in bit/s over 100ms', 
		_ls='--')
    setgraph_log()
    pylab.savefig(output_path + '/%sflows_peak_rate.pdf' % prefix, 
		  format='pdf')

    #plot cdf of peak rate over flows larger than 1MB
    pylab.clf()
    #80*bytes/100ms => bit/s
    peak_yt_sup10kB=[80*x[PEAK] for x in flows_down_yt if x[BYTES]>10**4]
    peak_dm_sup10kB=[80*x[PEAK] for x in flows_down_dm if x[BYTES]>10**4]
    cdfplotdata(peak_yt_sup10kB, _name='YouTube')
    cdfplotdata(peak_dm_sup10kB, _name='DailyMotion', 
		_title='%s Downstream Peak Rate' % title, 
		_xlabel='Downstream Peak Rate in bit/s over 100ms', 
		_ls='--')
    setgraph_log()
    pylab.savefig(output_path + '/%sflows_peak_rate_sup1MB.pdf' % prefix, 
		  format='pdf')

    #plot cdf of duration
    pylab.clf()
    cdfplotdata(flows_down_yt[:, DUR], _name='YouTube')
    cdfplotdata(flows_down_dm[:, DUR], _name='DailyMotion', 
		_xlabel='Downstream Flows Duration in Seconds', 
		_title='%s Downstream Flows Duration' % title, _ls='--')
    setgraph_log(_loc=4)
    pylab.savefig(output_path + '/%sflows_duration.pdf' %  prefix, 
		  format='pdf')

    #plot cdf of size
    pylab.clf()
    cdfplotdata(flows_down_yt[:, BYTES], _name='YouTube')
    cdfplotdata(flows_down_dm[:, BYTES], _name='DailyMotion',
		_ls='--',_title='%s Downstream Flow Size' % title, 
		_xlabel='Downstream Flow Size in Bytes')
    setgraph_log(_loc=4)
    pylab.savefig(output_path + '/%sflows_size.pdf' %  prefix, format='pdf')

    #plot cdf of size over flows larger than 1MB
    pylab.clf()
    cdfplotdata([b for b in flows_down_yt[:,BYTES] if b>10**6], 
		_name='YouTube')
    cdfplotdata([b for b in flows_down_dm[:,BYTES] if b>10**6], 
		_name='DailyMotion', _ls='--', 
		_title='FTTH Downstream Flow Size for Flows larger than 1MBytes', 
		_xlabel='Downstream Flow Size in Bytes')
    setgraph_log(_loc=4)
    pylab.savefig(output_path + '/%sflows_size_sup1MB.pdf' %  prefix, 
		  format='pdf')

    #plot cdf of size over flows larger than 10kB
    pylab.clf()
    cdfplotdata([b for b in flows_down_yt[:,BYTES] if b>10**4], 
		_name='YouTube')
    cdfplotdata([b for b in flows_down_dm[:,BYTES] if b>10**4], 
		_name='DailyMotion',_ls='--', 
		_title='FTTH Downstream Flow Size for Flows larger than 10kBytes', 
		_xlabel='Downstream Flow Size in Bytes')
    setgraph_log(_loc=4)
    pylab.savefig(output_path + '/%sflows_size_sup10kB.pdf' %  prefix, 
		  format='pdf')

    #plot flow arrival rate in sec YT
    pylab.clf()
    compute_AT(flows_down_yt[:,INIT], _title='YouTube (%s)' % title)
    pylab.savefig(output_path + '/%sflows_arrival_rate_second_YT.pdf' % prefix, 
		  format='pdf')

    #plot flow arrival rate in sec DM
    pylab.clf()
    compute_AT(flows_down_dm[:,INIT], _title='DailyMotion (%s)' % title)
    pylab.savefig(output_path + '/%sflows_arrival_rate_second_DM.pdf' %  prefix,
		  format='pdf')


    #plot flow arrival rate in min YT
    pylab.clf()
    compute_AT(flows_down_yt[:,INIT], _title='YouTube (%s)' % title, 
                   _ylabel='Nb of New Flows per Minute', _divide=60.0)
    pylab.savefig(output_path + '/%sflows_arrival_rate_minute_YT.pdf' % prefix,
		  format='pdf')

    #plot flow arrival rate in min DM
    pylab.clf()
    compute_AT(flows_down_dm[:,INIT], _title='DailyMotion (%s)' % title, 
                   _ylabel='Nb of New Flows per Minute', _divide=60.0)
    pylab.savefig(output_path + '/%sflows_arrival_rate_minute_DM.pdf' %  prefix,
		  format='pdf')


    #show()

def load_GVB_file(file, FTTH=False):
    #input file index
    IDX_PROTOCOL = 0
    IDX_CLIENT = 1
    IDX_DIR = 2
    IDX_INIT = 7
    IDX_DUR = 10
    IDX_BYTES = 11
    IDX_PEAK = 14
    IDX_DSCP = 15
    #WARNING FTTH AS INDEX IS DIFFERENT FROM ADSL
    if FTTH:
        IDX_AS = 17
    else:
        IDX_AS = 16 
    
    return pylab.loadtxt(file, 
         usecols=[IDX_PROTOCOL, IDX_CLIENT, IDX_DIR, 
         IDX_INIT, IDX_DUR, IDX_BYTES, IDX_PEAK, IDX_AS, IDX_DSCP])


if __name__ == '__main__':
    from optparse import OptionParser
    usage ="%prog -r data_file [-o output_dir -F -p PREFIX -t TITLE]"
    parser = OptionParser(usage = usage)
    parser.add_option("-r", dest = "file",
                      help = "input data file")
    parser.add_option("-o", dest = "output_path", default = "./rapport/figs", 
                      help = "directory for new figures [default value = ./rapport/figs] (created before)")
    parser.add_option("-F",  action="store_true", dest="FTTH",
                      help = "sets input data format to FTTH (default is ADSL)")
    parser.add_option("-p", dest="prefix", default="_",
                      help = "prefix pdf figures files with PREFIX [default = \'_\']")
    parser.add_option("-t", dest="title", default="",
                      help = "prefix the title of pdf figures with TITLE [default = \'\']")

    (options, args) = parser.parse_args()

    if not options.file:
        parser.print_help()
        exit()

    flows = load_file(options.file, options.FTTH)
    process_flows(flows, prefix=options.prefix, 
		  output_path=options.output_path, title=options.title)
