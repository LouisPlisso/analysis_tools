#!/usr/bin/env python
"Module to load text file, add AS information and store them in python binary format."

import sys
from optparse import OptionParser
import numpy as np

import INDEX_VALUES
#import add_ASN_geoip

MIN_FIELDS_NB = 16

def load_stream_GVB(file_name, bgp=False, no_as=False):
    "Load a streaming stats file and return a numpy array"
    in_file = open(file_name)
    header = in_file.readline()
    flows = []
    for nb, line in enumerate(in_file.readlines()):
        fields = line.strip().split(INDEX_VALUES.sep_GVB)
        assert len(fields) >= MIN_FIELDS_NB, "incorrect number of fields in line %d: %d" \
                % (nb, len(fields))
        for index in [2, 5, 6, 9]:
            try:
                fields[index] = float(fields[index])
            except ValueError:
                fields[index] = 0.0
        for index in [4, 7, 8, 12]:
            try:
                fields[index] = int(fields[index])
            except ValueError:
                fields[index] = 0
        for index in (10, 11):
            #special parsing for ; separated fields
            if fields[index] == ';':
                fields[index] = [0] * 100
            else:
                fields[index] = fields[index].strip(';').split(';')
        if bgp:
            index = 16
            try:
                fields[index] = int(fields[index])
            except ValueError:
                fields[index] = 0
        flows.append(tuple(fields))
    return flows

def save_stream_GVB(file_name, flows, bgp=False):
    "Save the flows array into reformatted filename"
    if not bgp:
        np.save('_'.join(file_name.split('.')),
                np.array(flows, dtype=INDEX_VALUES.dtype_GVB_streaming))
    else:
        np.save('_'.join(file_name.split('.')),
                np.array(flows, dtype=INDEX_VALUES.dtype_GVB_streaming_AS))


def main():
    usage = "%prog [-b] [-n] -r data_file_1 [-r data_file_2 ... -r data_file_n]"

    parser = OptionParser(usage = usage)
    parser.add_option("-n", dest = "no_as", action="store_true", default=False,
                      help = "TODO: indicate if no AS info has to be added")
    parser.add_option("-b", dest = "bgp", action="store_true", default=False,
                      help = "indicate if file has AS info from BGP")
    parser.add_option("-r", dest = "in_file", type = "string",
                      action = "append", help = "list of input data files")
    (options, args) = parser.parse_args()

    if not options.in_file:
        parser.print_help()
        return 1

    for f in options.in_file:
        flows = load_stream_GVB(f, bgp=options.bgp, no_as=options.no_as)
        save_stream_GVB(f, flows, bgp=options.bgp)

if __name__ == '__main__':
    sys.exit(main())

#    flows_FTTH_2008 = np.loadtxt('marked_GVB_juill_2008_FTTH/flows_stats.txt', dtype=INDEX_VALUES.dtype_GVB, skiprows=1).view(np.recarray)
#    flows_FTTH_2008 = add_ASN_geoip.extend_array_AS(flows_FTTH_2008)
#    np.save('flows_FTTH_2008', flows_FTTH_2008)
#    del flows_FTTH_2008
#
#    flows_FTTH_2009 = np.loadtxt('marked_GVB_nov_2009_FTTH/flows_stats.txt', dtype=INDEX_VALUES.dtype_GVB, skiprows=1).view(np.recarray)
#    flows_FTTH_2009 = add_ASN_geoip.extend_array_AS(flows_FTTH_2009)
#    np.save('flows_FTTH_2009', flows_FTTH_2009)
#    del flows_FTTH_2009
#
#    flows_ADSL_2009 = np.loadtxt('marked_GVB_nov_2009_ADSL/flows_stats.txt', dtype=INDEX_VALUES.dtype_GVB, skiprows=1).view(np.recarray)
#    flows_ADSL_2009 = add_ASN_geoip.extend_array_AS(flows_ADSL_2009)
#    np.save('flows_ADSL_2009', flows_ADSL_2009)
#    del flows_ADSL_2009
#
#
#    flows_ADSL_2008 = np.loadtxt('marked_GVB_juill_2008_ADSL/flows_stats.txt', dtype=INDEX_VALUES.dtype_GVB, skiprows=1).view(np.recarray)
#    flows_ADSL_2008 = add_ASN_geoip.extend_array_AS(flows_ADSL_2008)
#    np.save('flows_ADSL_2008', flows_ADSL_2008)
#    del flows_ADSL_2008
