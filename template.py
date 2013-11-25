#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :
'''Module docstring
Template version: 1.3
'''

# for python2
from __future__ import division, print_function

VERSION = '1.0'

import argparse
import sys
import os
import functools
import logging
#import heapq
#from operator import itemgetter
#from collections import defaultdict
#from collections import deque
#from array import array
#from bisect import bisect
#from math import sqrt

# for interactive call: do not add multiple times the handler
if 'LOG' not in locals():
    LOG = None
LOG_LEVEL = logging.ERROR
FORMATER_STRING = ('%(asctime)s - %(filename)s:%(lineno)d - '
                   '%(levelname)s - %(message)s')

def configure_log(level=LOG_LEVEL, log_file=None):
    'Configure logger'
    if LOG:
        LOG.setLevel(level)
        return LOG
    log = logging.getLogger('%s log' % os.path.basename(__file__))
    if log_file:
        handler = logging.FileHandler(filename=log_file)
    else:
        handler = logging.StreamHandler(sys.stderr)
    log_formatter = logging.Formatter(FORMATER_STRING)
    handler.setFormatter(log_formatter)
    log.addHandler(handler)
    log.setLevel(level)
    return log

LOG = configure_log()

class memoized(object):
    '''Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        try:
            return self.cache[args]
        except KeyError:
            value = self.func(*args)
            self.cache[args] = value
            return value
        except TypeError:
            # uncachable -- for instance, passing a list as an argument.
            # Better to not cache than to blow up entirely.
            return self.func(*args)
    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__
    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)

class CommentedFile:
    'Implements comments skip for file'
    def __init__(self, in_file, commentstring='#'):
        self.in_file = in_file
        self.commentstring = commentstring
    def next(self):
        'The next line but skips comments'
        line = self.in_file.next()
        while line.startswith(self.commentstring):
            line = self.in_file.next()
        return line
    def __iter__(self):
        return self


def do_job(in_file, out_file=sys.stdout):
    'Do the work'
    LOG.debug('Start working with files: %s and %s',
              in_file.name, out_file.name)
    # first line is number of test cases
    T = int(in_file.readline())
    for testcase in range(T):
        N = int(in_file.readline())
        # for integer input
        values = map(int, in_file.readline().split())
        # for other inputs
#        values = in_file.readline().rstrip('\n')
        assert len(values) == N
        result = 0
        print_output(out_file, testcase, result)

def print_output(out_file, testcase, result):
    'Formats and print result'
    print('Case #%d:' % (testcase + 1), end=' ', file=out_file)
    print(result, file=out_file)
    #print('%.6g' % result, file=out_file)


def main(argv=None):
    'Program wrapper'
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version', version=VERSION)
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help=argparse.SUPPRESS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-q', '--quiet', dest='quiet',
                        action='store_true', default=False,
                        help='run as quiet mode')
    group.add_argument('-v', '--verbose', dest='verbose',
                        action='store_true', default=False,
                        help='run as verbose mode')
    parser.add_argument('-t', dest='template', action='store_true',
                        default=False, help=('template name for construct'
                            'out file name as in_file.out (default False)'))
    parser.add_argument('-w', dest='out_file',
            help=('output file or stdout if FILE is - (default case)'
                  'or TEMPLATE.out (default if template is given)'))
    parser.add_argument('in_file', help='input file')
    args = parser.parse_args(argv)
#    if args.quiet and args.verbose:
#        parser.error('Options quiet and verbose are mutually exclusive')
    if args.verbose:
        LOG.setLevel(logging.INFO)
    if args.quiet:
        LOG.setLevel(logging.CRITICAL)
    if args.debug:
        LOG.setLevel(logging.DEBUG)
#    # unset verbose for easy option check
#    args.verbose = False
#    if not any(args.__dict__.values()):
#        parser.error('Must provide at least one option')
    if args.in_file == '-':
        in_file = sys.stdin
    else:
        try:
            in_file = open(args.in_file, 'r')
        except IOError:
            parser.error('File, %s, does not exist.' % args.in_file)
    if args.template and not args.out_file:
        args.out_file = ''.join((args.in_file, '.out'))
    if not args.out_file or args.out_file == '-':
        out_file = sys.stdout
    else:
        try:
            out_file = open(args.out_file, 'w')
        except IOError:
            parser.error('Problem opening file: %s' % args.out_file)
    sys.setrecursionlimit(2**30-1)
    do_job(in_file, out_file)
    return 0

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    sys.exit(main())

