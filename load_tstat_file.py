#!/usr/bin/env python
"""Module to load tstat file and store them in python binary
format.
"""

import sys
from optparse import OptionParser
import numpy as np

#import cPickle

import INDEX_VALUES

def load_and_process_tstat(in_file, new_tstat=False):
    "Load a file in cnx_str format and store it in binary format."
    flows = np.loadtxt(in_file,
                       dtype=(INDEX_VALUES.dtype_tstat2 if new_tstat
                              else INDEX_VALUES.dtype_tstat),
                       converters=(INDEX_VALUES.converters_tstat2 if new_tstat
                                   else INDEX_VALUES.converters_tstat),
                       delimiter=' ')
    np.save('.'.join((in_file, 'npy')), flows)
#cPickle.dump(flows, open('_'.join(in_file.split('.')[:-1]) + '.pickle', 'w'))

def main():
    usage = "%prog [-n] data_file_1 [data_file_2 ... data_file_n]"
    parser = OptionParser(usage = usage)
    parser.add_option("-n", dest = "new_tstat", action="store_true",
                      default=False,
                      help = "indicate if the file is from tstat-2.2")
    (options, args) = parser.parse_args()
    if not args:
        parser.print_help()
        return 1
    for f in args:
        load_and_process_tstat(f, new_tstat=options.new_tstat)

if __name__ == '__main__':
    sys.exit(main())
