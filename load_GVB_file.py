#!/usr/bin/env python
"""Module to load text file, add AS information and store them in python binary
format.
"""

import sys
from optparse import OptionParser
import numpy as np

import INDEX_VALUES
import add_ASN_geoip

from load_dipcp_file import return_cnx_id_GVB

def return_cnx_id_dipcp(dipcp_line):
    "Return the connection ID of dipcp flow."
    return (dipcp_line['FlowIPSource'], dipcp_line['FlowPortSource'],
            dipcp_line['FlowIPDest'], dipcp_line['FlowPortDest'])

def filter_GVB_array(GVB_flows, dipcp_flows):
    cnxs_list = set([return_cnx_id_dipcp(x)
                     for x in dipcp_flows])
    return np.array([x for x in GVB_flows if return_cnx_id_GVB(x) in cnxs_list],
                    dtype=INDEX_VALUES.dtype_GVB_BGP_AS)


def load_and_process_GVB(in_file, bgp=False, no_as=False):
    "Load a file in GVB format, add AS info and store it in binary format."
    if no_as:
        flows = np.loadtxt(in_file, dtype=INDEX_VALUES.dtype_GVB)
    else:
        if not bgp:
            flows = np.loadtxt(in_file, dtype=INDEX_VALUES.dtype_GVB,
                               skiprows=1)
            flows = add_ASN_geoip.extend_array_AS(flows)
        else:
            flows = np.loadtxt(in_file, dtype=INDEX_VALUES.dtype_GVB_BGP,
                               skiprows=1)
            flows = add_ASN_geoip.extend_array_BGP_AS(flows)
    np.save('%s_AS' % '_'.join(in_file.split('.')[:-1]), flows)
    del flows


def main():
    usage = "%prog [-b] [-n] -r data_file_1 [-r data_file_2 ... -r data_file_n]"

    parser = OptionParser(usage = usage)
    parser.add_option("-r", dest = "file", type = "string",
                      action = "append", help = "list of input data files")
    parser.add_option("-n", dest = "no_as", action="store_true", default=False,
                      help = "indicate if no AS info has to be added")
    parser.add_option("-b", dest = "bgp", action="store_true", default=False,
                      help = "indicate if file has AS info from BGP")
    (options, _) = parser.parse_args()

    if not options.file:
        parser.print_help()
        return 1

    for f in options.file:
        load_and_process_GVB(f, bgp=options.bgp, no_as=options.no_as)

if __name__ == '__main__':
    sys.exit(main())

#    flows_FTTH_2008 = np.loadtxt('marked_GVB_juill_2008_FTTH/flows_stats.txt',
#dtype=INDEX_VALUES.dtype_GVB, skiprows=1).view(np.recarray)
#    flows_FTTH_2008 = add_ASN_geoip.extend_array_AS(flows_FTTH_2008)
#    np.save('flows_FTTH_2008', flows_FTTH_2008)
#    del flows_FTTH_2008
#
#    flows_FTTH_2009 = np.loadtxt('marked_GVB_nov_2009_FTTH/flows_stats.txt',
#dtype=INDEX_VALUES.dtype_GVB, skiprows=1).view(np.recarray)
#    flows_FTTH_2009 = add_ASN_geoip.extend_array_AS(flows_FTTH_2009)
#    np.save('flows_FTTH_2009', flows_FTTH_2009)
#    del flows_FTTH_2009
#
#    flows_ADSL_2009 = np.loadtxt('marked_GVB_nov_2009_ADSL/flows_stats.txt',
#dtype=INDEX_VALUES.dtype_GVB, skiprows=1).view(np.recarray)
#    flows_ADSL_2009 = add_ASN_geoip.extend_array_AS(flows_ADSL_2009)
#    np.save('flows_ADSL_2009', flows_ADSL_2009)
#    del flows_ADSL_2009
#
#
#    flows_ADSL_2008 = np.loadtxt('marked_GVB_juill_2008_ADSL/flows_stats.txt',
#dtype=INDEX_VALUES.dtype_GVB, skiprows=1).view(np.recarray)
#    flows_ADSL_2008 = add_ASN_geoip.extend_array_AS(flows_ADSL_2008)
#    np.save('flows_ADSL_2008', flows_ADSL_2008)
#    del flows_ADSL_2008
