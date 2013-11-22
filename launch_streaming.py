#!/usr/bin/env python

from optparse import OptionParser
import numpy as np
import os
import re
import cdfplot
import pylab
#from matplotlib.font_manager import FontProperties
import flow2session
import INDEX_VALUES
from streaming_tools import format_title

AS_LIST = ('AS_YOUTUBE', 'AS_YOUTUBE_EU', 'AS_GOOGLE')
MIN_NB_FLOWS = 20

def sessions_stats(datas, gap, min_dur, out_pattern):
    "Plot streaming sessions stats"
    stream_datas = {}
    for f in datas:
        stream_datas[f] = flow2session.process_stream_session(datas[f], gap)

    pylab.clf()
    cdfplot.cdfplotdataN([(format_title(key),
                           [x['nb_streams'] for x in stream_datas[key]
                            if x['duration'] > min_dur])
                          for key in sorted(stream_datas.keys())], logx=False,
                         _title="""Nb of streaming sessions with gap interval (%ds)
Sessions longer than %d sec""" % (gap, min_dur), _loc=4)
    cdfplot.setgraph_logy(_loc=4)
    pylab.savefig('%s_duration_gap_%ds_min_%d.pdf' % (out_pattern,
                                                      gap, min_dur))
    pylab.clf()
    cdfplot.cdfplotdataN([(format_title(key), stream_datas[key]['duration'])
                          for key in sorted(stream_datas.keys())]
                         , _title="Duration of streaming sessions with gap \
                         interval (%ds)" % gap
                        )
    pylab.savefig('%s_cdf_duration_gap_%ds_min_%d.pdf' % (out_pattern,
                                                          gap, min_dur))

def format_title_streaming(trace):
    r"""Return a formatted string for the trace name
    >>> format_title_streaming('2009_11_26_ADSL_GVB_STR_AS_GOOGLE')
    '2010/02/07\nADSL GOOGLE'
    >>>
    """
    trace_name = trace.split('.')[0]
    trace_date = '/'.join(trace_name.split('_')[:3])
    trace_type = ' '.join(trace_name.split('_')[3:])
    return '\n'.join((trace_date, trace_type.split(' ')[0] + ' ' +
                      trace_type.split(' ')[-1]))


def stream_stats(datas, out_pattern, filter_datas=None):
    "Plots streaming stats"
    pylab.clf()
    cdfplot.cdfplotdataN([(format_title(key), datas[key]['Content-Duration'])
                          for key in sorted(datas.keys())]
                         , _title="""Duration of Streaming Contents
annonced in header"""
                         , _xlabel="Duration in Seconds")

    axes = pylab.gca()
    cdfplot.setgraph_logx(_loc=4)
    axes.set_xlim((1, 10**4))
#    font = FontProperties(size = 'xx-small')
#    leg = pylab.legend(loc = _loc, prop = font)
#    leg = pylab.legend(loc='best', mode="expand", fancybox=True)
#    leg.get_frame().set_alpha(0.3)
    pylab.savefig('%s_cdf_duration_stream_content.pdf' % out_pattern)

    if filter_datas:
        for name in ('YOUTUBE', 'GOOGLE', 'OTHER'):
            pylab.clf()
            cdfplot.cdfplotdataN([(format_title_streaming(key),
                                   filter_datas[key]['Content-Duration'])
                                  for key in sorted(filter_datas.keys())
                                 if key.find(name) >= 0]
                                 , _title="""Duration of Streaming Contents
annonced in header"""
                                 , _xlabel="Duration in Seconds")
            axes = pylab.gca()
            cdfplot.setgraph_logx(_loc=4)
            axes.set_xlim((1, 10**4))
            pylab.savefig('%s_cdf_duration_stream_content_%s.pdf'
                          % (out_pattern, name))

    pylab.clf()
    cdfplot.cdfplotdataN([(format_title(key), datas[key]['Session-Duration'])
                          for key in sorted(datas.keys())]
                         , _title="""Duration of Streaming Views
computed from average rate and transmitted bytes"""
                         , _xlabel="Duration in Seconds")
    axes = pylab.gca()
    cdfplot.setgraph_logx(_loc=4)
    axes.set_xlim((10**-1, 10**4))
    pylab.savefig('%s_cdf_duration_stream_session.pdf' % out_pattern)

    if filter_datas:
        for name in ('YOUTUBE', 'GOOGLE', 'OTHER'):
            pylab.clf()
            cdfplot.cdfplotdataN([(format_title_streaming(key),
                                   filter_datas[key]['Session-Duration'])
                                  for key in sorted(filter_datas.keys())
                                 if key.find(name) >= 0]
                                 , _title="""Duration of Streaming Views
computed from average rate and transmitted bytes"""
                                 , _xlabel="Duration in Seconds")
            axes = pylab.gca()
            cdfplot.setgraph_logx(_loc=4)
            axes.set_xlim((1, 10**4))
            pylab.savefig('%s_cdf_duration_stream_session_%s.pdf'
                          % (out_pattern, name))



    traces = set(re.split('_(FTTH|ADSL)', key)[0] for key in datas)
    for key in sorted(traces):
        if '_'.join((key, 'ADSL', 'R', 'GVB_STR')) in datas:
            adsl_key = '_'.join((key, 'ADSL', 'R', 'GVB_STR'))
        elif '_'.join((key, 'ADSL', 'GVB_STR')) in datas:
            adsl_key = '_'.join((key, 'ADSL', 'GVB_STR'))
        else:
            print "Key not found in trace list: " + key
            continue
        ftth_key = '_'.join((key, 'FTTH', 'GVB_STR'))
        pylab.clf()
        cdfplot.cdfplotdataN([(' '.join((access, indic, "Duration")),
                               datas[adsl_key if access == 'ADSL' else ftth_key]
                               [indic + '-Duration'])
                              for indic in ('Session', 'Content')
                              for access in ('ADSL', 'FTTH')]
                             , _title="Duration of Streaming Views vs. Contents: "
                             + format_title(key)
                             , _xlabel="Duration in Seconds")
        axes = pylab.gca()
        cdfplot.setgraph_logx(_loc=4)
        axes.set_xlim((1, 10**4))
        pylab.savefig('%s_cdf_duration_stream_session_vs_content_%s.pdf'
                      % (out_pattern, key))
        if filter_datas:
            args = []
            for k in sorted(filter(lambda x: x.find(key) >= 0, filter_datas.keys())):
                for indic in ('Session', 'Content'):
                    args.append((' '.join((indic, k.split('_')[-1],
                                          'ADSL' if k.find('ADSL')>=0 else
                                           'FTTH')),
                                 filter_datas[k][indic + '-Duration']))
#            print args
            pylab.clf()
            cdfplot.cdfplotdataN(args,
                                 _title="Duration of Streaming Views vs. Contents: "
                                 + format_title(key)
                                 , _xlabel="Duration in Seconds")
            axes = pylab.gca()
            cdfplot.setgraph_logx(_loc=4)
            axes.set_xlim((1, 10**4))
            pylab.savefig('%s_cdf_duration_stream_session_vs_content_%s_AS.pdf'
                          % (out_pattern, key))



def main():
    "Program wrapper"
    parser = OptionParser(usage = "%prog \
-d flows_dir [-f] [-s [-w out_pattern -g gap -m min_dur]]|[-c]")
    parser.add_option("-d", dest = "flows_dir", type = "string",
        action = "store", help = "input dir where are stream stats file (.npy)")
    parser.add_option("-f", dest = "filter_AS",
        action = "store_true", default = False,
        help = "separate streams according to AS")
    parser.add_option("-c", dest = "stream_stats",
        action = "store_true", default = False,
        help = "plot streaming stats")
    parser.add_option("-s", dest = "sessions_stats",
        action = "store_true", default = False,
        help = "plot streaming sessions stats")
    parser.add_option("-w", dest = "out_pattern", type = "string",
        action = "store", default = "stream_session",
        help = "output session plot file [DEFAULT=stream_session_XXX.pdf]")
    parser.add_option("-m", dest = "min_dur", type = "int", default = 3,
        action = "store", help = "minimum duration of session to be plot (DEFAULT=3)")
    parser.add_option("-g", dest = "gap", type = "int", default = 60,
        action = "store", help = "interval between sessions (DEFAULT=60)")
    (options, _) = parser.parse_args()

    if not options.flows_dir:
        parser.print_help()
        exit()

    stream_quality_dir = options.flows_dir
    datas = {}

    for f in [fs for fs in os.listdir(stream_quality_dir)
              if fs.endswith('.npy') and fs.find('_AS') == -1]:
        print "Loading file: " + f
        datas[f.split('.npy')[0]] = np.load(
            os.sep.join((stream_quality_dir, f)))

    if options.sessions_stats:
        sessions_stats(datas, options.gap, options.min_dur,
                       options.out_pattern)

    if options.filter_AS:
        datas_to_filter = {}
        for f in [fs for fs in os.listdir(stream_quality_dir)
                  if fs.endswith('.npy') and fs.find('_AS') >= 0]:
            print "Loading file: " + f
            datas_to_filter[f.split('.npy')[0]] = np.load(
                os.sep.join((stream_quality_dir, f)))

        filter_datas = {}
        for k in datas_to_filter:
            data = datas_to_filter[k]
            for as_name in AS_LIST:
                filter_datas['%s_%s' %
                             (k, as_name.lstrip('AS_').replace('_', '-'))] = \
                        data.compress(
                            map(lambda x: x in INDEX_VALUES.__getattribute__(as_name),
                                data['asBGP']))
#            else:
#                filter_datas['%s_%s' % (k, 'OTHER')] = data.compress(
#                    map(lambda x: x not in
#                        set(INDEX_VALUES.__getattribute__(as_name)
#                            for as_name in AS_LIST),
#                        data['asBGP']))
        del datas_to_filter

        # remove non signicative flows
        for k in filter_datas.keys():
            if len(filter_datas[k]) < MIN_NB_FLOWS:
                del filter_datas[k]

    if options.stream_stats:
        stream_stats(datas, options.out_pattern, filter_datas)

if __name__ == '__main__':
    main()
