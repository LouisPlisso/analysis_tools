#!/usr/bin/env python
"Write a new hdf5 file containing all data from directory"

from optparse import OptionParser
import sys
import os

import numpy as np
import h5py

def do_job(in_dir, in_files, out_file, verbose=False):
    "Do the work"
    capture_list = set(f.lstrip('GVB_').lstrip('dipcp_').rstrip('.npy')
            for f in in_files if f.startswith('GVB_') or f.startswith('dipcp_'))
    for capture in capture_list:
        if verbose:
            print "Start working with capture: %s" % capture
        group = out_file.create_group(capture)
        try:
            file_gvb = ''.join((in_dir, os.sep, 'GVB_', capture, '.npy'))
            data_gvb = np.load(file_gvb)
            group.create_dataset('GVB', data=data_gvb, compression='lzf',
                    chunks=True)
            del data_gvb
        except IOError:
            print >> sys.stderr, "Cannot open file %s" % file_gvb
        try:
            file_dipcp = ''.join((in_dir, os.sep, 'dipcp_', capture, '.npy'))
            data_dipcp = np.load(file_dipcp)
            group.create_dataset('DIPCP', data=data_dipcp, compression='lzf',
                    chunks=True)
            del data_dipcp
        except IOError:
            print >> sys.stderr, "Cannot open file %s" % file_dipcp
    out_file.close()


def main(argv=None):
    "Program wrapper."
    if argv is None:
        argv = sys.argv[1:]
    usage = "%prog [-r in_dir] -w out_file"
    parser = OptionParser(usage = usage)
    parser.add_option("-r", dest = "in_dir", default='.',
            help = "input dir (current dir by default)")
    parser.add_option("-w", dest = "out_file",
            help = "output hdf5 file")
    parser.add_option("-V", "--verbose", dest = "verbose",
            action="store_true", default=False,
            help = "run as verbose mode")

    #I don't like args;)
    (options, _) = parser.parse_args(argv)
    if not options.out_file:
        print "Must provide an output filename."
        parser.print_help()
        return(1)
    try:
        in_files = os.listdir(options.in_dir)
    except OSError:
        print "Directory, %s, cannot be listed." % options.in_dir
        parser.print_help()
        return(1)
    try:
        out_file = h5py.File(options.out_file, 'w')
    except IOError:
        print "Problem opening file: %s" % options.out_file
        parser.print_help()
        return(1)

    if options.verbose:
        print "Will create %s on dir %s" % (options.out_file, options.in_dir)
    do_job(options.in_dir, in_files, out_file, options.verbose)
    return 0

if __name__ == '__main__':
    sys.exit(main())
