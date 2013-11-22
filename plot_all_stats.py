#!/usr/bin/env python
"Module to plot many relevant cdf for comparing traces."

from optparse import OptionParser
import pylab

import INDEX_VALUES
import cdfplot
import load_hdf5_data

# 500kB
THRESHOLD = 500*10**3

AS_NAMES = ('YT', 'YT_EU', 'GOO')
AS_NUMBERS = (INDEX_VALUES.AS_YOUTUBE, INDEX_VALUES.AS_YOUTUBE_EU,
              INDEX_VALUES.AS_GOOGLE)

assert len(AS_NAMES) == len(AS_NUMBERS)

def process_stats_traces(data,
        output_path = 'rapport/http_stats',
#        prefix = 'YT_YT_EU_N_',
        title = 'Youtube Google HTTP Streaming Downstream Flows (only if > 10 flows):\n'):
    """Take N numpy record arrays to generate a lot of graphs related to Youtube
    Use as:
    data = tools.load_hdf5_data.load_h5_file('hdf5/lzf_data.h5')
    tools.plot_all_stats.process_stats_traces(datas)
    """
    prefix = '_'.join(AS_NAMES).lower()

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
        for as_name, as_value in zip(AS_NAMES, AS_NUMBERS):
            flows['%s_%s' % (name, as_name)] = \
                    flows_tmp['flows_%s_stream_down' % name].compress(
                        [x['asBGP'] in as_value
                         for x in flows_tmp['flows_%s_stream_down' % name]])
            flows_1MB['%s_%s_1MB' % (name, as_name)] = \
                    flows['%s_%s' % (name, as_name)].compress(
                        flows['%s_%s' % (name, as_name)]['l3Bytes'] > THRESHOLD)
        del flows_tmp['flows_%s_stream_down' % name]
        # remove non signicative flows
        for as_name in AS_NAMES:
            if len(flows['%s_%s' % (name, as_name)]) < 10:
                del flows['%s_%s' % (name, as_name)]
            if len(flows['%s_%s_1MB' % (name, as_name)]) < 10:
                del flows['%s_%s_1MB' % (name, as_name)]
    del flows_tmp

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


def plot_from_file(in_file):
#    datas = load_hdf5_data.load_h5_file('flows/hdf5/traces_lzf.h5')
    datas = load_hdf5_data.load_h5_file(in_file)
#    process_flows_YT_YT_EU_N_traces(datas)
    process_stats_traces(datas)

def main():
    usage = "%prog -r data_file"

    parser = OptionParser(usage = usage)
    parser.add_option("-r", dest = "in_file", type = "string",
                      help = "input data file")
    (options, args) = parser.parse_args()

    if not options.in_file:
        parser.print_help()
        exit()

    plot_from_file(options.in_file)

if __name__ == '__main__':
    main()
