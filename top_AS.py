#!/usr/bin/env python
"Module to sum up top AS in a trace."


from optparse import OptionParser
import numpy as np

import INDEX_VALUES
import aggregate

def extract_aggregated_field(data, key, ref):
    return [f['aggregation'] for f in data if f[key] == ref][0]

def top_as(file='flows_FTTH_2009.npy'):
    "Extract top AS of a preloaded GVB flow file and format it into latex tabular."
    flows = np.load(file).view(np.recarray )
    format_latex_tab_AS(top_as_data(flows))
    

def top_as_data(flows):
    "Return top AS of GVB flow recarray in the form of a dictonnary."
    resume = {}
    flows_down = flows.compress(flows.direction == INDEX_VALUES.DOWN)
    flows_down_as = aggregate.aggregate(flows_down, 'orgSrc', 'l3Bytes', sum)
    flows_down_as.sort(order='aggregation')
    for i in range(11):
        (resume['name_as_down_%d' % i], resume['vol_as_down_%d' % i]) = flows_down_as[-(i+1)]
    resume['total_other_down_as'] = np.sum(flows_down_as[:-10][:].aggregation)

    flows_down_web = flows_down.compress(flows_down.dscp == INDEX_VALUES.DSCP_WEB)
    flows_down_as_web = aggregate.aggregate(flows_down_web, 'orgSrc', 'l3Bytes', sum)
    flows_down_as_web.sort(order='aggregation') 
    for i in range(11):
        (resume['name_as_down_web_%d' % i], resume['vol_as_down_web_%d' % i]) = flows_down_as_web[-(i+1)]
    resume['total_other_as_down_web'] = np.sum(flows_down_as_web[:-10][:].aggregation)

    flows_down_other_stream = flows_down.compress(flows_down.dscp == INDEX_VALUES.DSCP_OTHER_STREAM)
    flows_down_as_other_stream = aggregate.aggregate(flows_down_other_stream, 'orgSrc', 'l3Bytes', sum)
    flows_down_as_other_stream.sort(order='aggregation')
    for i in range(11):
        (resume['name_as_down_other_stream_%d' % i], resume['vol_as_down_other_stream_%d' % i]) \
            = flows_down_as_other_stream[-(i+1)]
    resume['total_other_as_down_other_stream'] = np.sum(flows_down_as_other_stream[:-10][:].aggregation)

    flows_down_http_stream = flows_down.compress(flows_down.dscp == INDEX_VALUES.DSCP_HTTP_STREAM)
    flows_down_as_http_stream = aggregate.aggregate(flows_down_http_stream, 'orgSrc', 'l3Bytes', sum)
    flows_down_as_http_stream.sort(order='aggregation')
    for i in range(11):
        (resume['name_as_down_http_stream_%d' % i], resume['vol_as_down_http_stream_%d' % i]) \
            = flows_down_as_http_stream[-(i+1)]
    resume['total_other_as_down_http_stream'] = np.sum(flows_down_as_http_stream[:-10][:].aggregation)

    return resume 

def top_bgp_data(flows):
    "Return top AS of GVB flow recarray in the form of a dictonnary."
    resume = {}
    flows_down = flows.compress(flows.direction == INDEX_VALUES.DOWN)
    flows_down_as = aggregate.aggregate(flows_down, 'asBGP', 'l3Bytes', sum)
    flows_down_as.sort(order='aggregation')
    for i in range(11):
        (resume['name_as_down_%d' % i], resume['vol_as_down_%d' % i]) = flows_down_as[-(i+1)]
    resume['total_other_down_as'] = np.sum(flows_down_as[:-10][:].aggregation)

    flows_down_web = flows_down.compress(flows_down.dscp == INDEX_VALUES.DSCP_WEB)
    flows_down_as_web = aggregate.aggregate(flows_down_web, 'asBGP', 'l3Bytes', sum)
    flows_down_as_web.sort(order='aggregation') 
    for i in range(11):
        (resume['name_as_down_web_%d' % i], resume['vol_as_down_web_%d' % i]) = flows_down_as_web[-(i+1)]
    resume['total_other_as_down_web'] = np.sum(flows_down_as_web[:-10][:].aggregation)

    flows_down_other_stream = flows_down.compress(flows_down.dscp == INDEX_VALUES.DSCP_OTHER_STREAM)
    flows_down_as_other_stream = aggregate.aggregate(flows_down_other_stream, 'asBGP', 'l3Bytes', sum)
    flows_down_as_other_stream.sort(order='aggregation')
    for i in range(11):
        (resume['name_as_down_other_stream_%d' % i], resume['vol_as_down_other_stream_%d' % i]) \
            = flows_down_as_other_stream[-(i+1)]
    resume['total_other_as_down_other_stream'] = np.sum(flows_down_as_other_stream[:-10][:].aggregation)

    flows_down_http_stream = flows_down.compress(flows_down.dscp == INDEX_VALUES.DSCP_HTTP_STREAM)
    flows_down_as_http_stream = aggregate.aggregate(flows_down_http_stream, 'asBGP', 'l3Bytes', sum)
    flows_down_as_http_stream.sort(order='aggregation')
    for i in range(11):
        (resume['name_as_down_http_stream_%d' % i], resume['vol_as_down_http_stream_%d' % i]) \
            = flows_down_as_http_stream[-(i+1)]
    resume['total_other_as_down_http_stream'] = np.sum(flows_down_as_http_stream[:-10][:].aggregation)

    return resume 

def format_latex_tab_AS(resume):
    "Formats the AS top 10 dictionnary into a latex tabular."
    out_string = r"""\begin{tabular}{|l|c|c|}
\hline
AS Number from BGP tables & Nb of Bytes & Direct Peering?\\
\hline"""
    for name in ['http_stream', 'other_stream', 'web']:
        out_string = ''.join((out_string, r"""\multicolumn{3}{c}{%s}\\
\hline""" % name.replace('_',' ').upper()))
        for i in range(11):
            out_string = ''.join((out_string, r"%(top_as)s & %(vol_as)4.2e\,Bytes & \\" 
                                         % {'top_as': resume['name_as_down_%s_%d' % (name, i)],
                                            'vol_as': resume['vol_as_down_%s_%d' % (name, i)]}))
        out_string = ''.join((out_string, r"""All Other ASes & %4.2e\,Bytes & \\
\hline""" %  resume['total_other_as_down_%s' % name]))
    out_string = ''.join((out_string, "\end{tabular}"))
    return out_string

def main():
    usage = "%prog -r data_file"

    parser = OptionParser(usage = usage)
    parser.add_option("-r", dest = "file", type = "string",
                      help = "input data file") 
    (options, args) = parser.parse_args()

    if not options.file:
        parser.print_help()
        exit()
	
    top_as(options.file)

if __name__ == '__main__':
    main()
