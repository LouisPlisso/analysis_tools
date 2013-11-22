#!/usr/bin/env python
"""Module to load cnx stream file and store them in python binary
format.
"""

import sys
from optparse import OptionParser
import numpy as np
from tempfile import TemporaryFile

#import cPickle

import INDEX_VALUES

# because urls can contain # sign and np.loadtxt automatically assigns it
# (and does not want to remove it)
NO_COMMENT = 'Z' * 100


def filter_lines(in_file, out_file, nb_fields, delimiter='\t'):
    """Writes a file with the lines in in_file that match the specified number
    of fields. Seeks the file at the end.
    """
    nb = tot = 0
    with open(in_file) as f:
        for line in f.readlines():
            tot += 1
            if len(line.split(delimiter)) == nb_fields:
                nb += 1
                print >> out_file, line
    print "%s: filtered %d lines out of %d" % (in_file, tot - nb, tot)
    out_file.seek(0)


def load_and_process_cnx_str(in_file, loss=None):
    "Load a file in cnx_str format and store it in binary format."
    if loss:
        with TemporaryFile() as f:
            filter_lines(in_file, f, len(INDEX_VALUES.dtype_cnx_stream_loss))
            flows = np.loadtxt(f, dtype=INDEX_VALUES.dtype_cnx_stream_loss,
                           converters=INDEX_VALUES.converters_cnx_stream_loss,
                               comments=NO_COMMENT,
                               skiprows=2, delimiter='\t')
    elif loss == False:
        with TemporaryFile() as f:
            filter_lines(in_file, f, len(INDEX_VALUES.dtype_cnx_stream))
            flows = np.loadtxt(f, dtype=INDEX_VALUES.dtype_cnx_stream,
                               converters=INDEX_VALUES.converters_cnx_stream,
                               comments=NO_COMMENT,
                               skiprows=2, delimiter='\t')
    else:
        # discover the number of fields in header
        with open(in_file) as opened_file:
            assert "OK\n" == opened_file.readline()
            header = opened_file.readline()
            nb_fields = len(header.split('\t'))
        for header_type in ('', '_loss'):
            if nb_fields == len(
                INDEX_VALUES.__getattribute__("dtype_cnx_stream%s"
                                              % header_type)):
                with TemporaryFile() as f:
#                with open('toto.test', 'w+r') as f:
                    filter_lines(in_file, f, len(INDEX_VALUES.__getattribute__(
                            "dtype_cnx_stream%s" % header_type)))
                    flows = np.loadtxt(f, skiprows=1, delimiter='\t',
                               comments=NO_COMMENT,
                               converters=INDEX_VALUES.__getattribute__(
                                   "converters_cnx_stream%s" % header_type),
                               dtype=INDEX_VALUES.__getattribute__(
                                  "dtype_cnx_stream%s" % header_type))
                break
        else:
            # for else
            print >> sys.stderr, "File %s has no corresponding header type" \
                                    % in_file
            print >> sys.stderr, "header length: %d" % nb_fields
            return
    np.save('%s' % '_'.join(in_file.split('.')[:-1]), flows)
#cPickle.dump(flows, open('_'.join(in_file.split('.')[:-1]) + '.pickle', 'w'))
    del flows


def main():
    usage = "%prog [-l|n] data_file_1 [data_file_2 ... data_file_n]"
    parser = OptionParser(usage = usage)
    parser.add_option("-n", dest = "loss", action="store_false", default=None,
                      help = "indicate if the file is not including loss stats")
    parser.add_option("-l", dest = "loss", action="store_true", default=None,
                      help = "indicate if the file is including loss stats")
    (options, args) = parser.parse_args()
    if not args:
        parser.print_help()
        return 1
    for f in args:
        load_and_process_cnx_str(f, loss=options.loss)

if __name__ == '__main__':
    sys.exit(main())

