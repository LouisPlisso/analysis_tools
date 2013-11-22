#!/usr/bin/env python
"Module to plot cdf from data or file. Can be called directly."

from __future__ import division, print_function

from optparse import OptionParser
import sys
import pylab
from matplotlib.font_manager import FontProperties

_VERSION = '1.2'

#TODO: possibility to place legend outside graph:
#pylab.subfigure(111)
#pylab.subplots_adjust(right=0.8) or (top=0.8)
#pylab.legend(loc=(1.1, 0.5)


#CCDF
def ccdfplotdataN(list_data_name, _xlabel = 'x',
        _ylabel = r'1 - P(X$\leq$x)',
        _title = 'Empirical Distribution',
        _fs_legend='medium',
        _fs = 'x-large', _loc=0):
    "Plot the ccdf of a list of data arrays and names"
    #corresponding line width with larger width for '-.' and ':'
    if not list_data_name:
        print("no data to plot", sys.stderr)
        return
    _ls = ['-', '-.', '--'] #, ':']
    _lw = [1, 2, 3] #, 4]
    _ls_len = len(_ls)
        #plot all cdfs except last one
    for i in range(len(list_data_name) - 1):
        name, data = list_data_name[i]
        #plot with round robin line style (ls)
        #and increasing line width
        (div, mod) = divmod(i, _ls_len)
        ccdfplotdata(data, _name=name, _lw=_lw[mod]+3*div,
                _ls=_ls[mod], _fs=_fs, _fs_legend=_fs_legend)
        #for last cdf, we put the legend and names
    (name, data) = list_data_name[-1]
    (div, mod) = divmod(len(list_data_name), _ls_len)
    ccdfplotdata(data, _name=name, _title=_title, _xlabel=_xlabel,
            _ylabel=_ylabel, _lw=_lw[mod]+2*div, _ls=_ls[mod], _fs=_fs)
    setgraph_logx(_loc=_loc)

def ccdfplotdata(data_in, _xlabel = 'x', _ylabel = r'1 - P(X$\leq$x)',
        _title = 'Empirical Distribution',
        _name = 'Data', _lw = 2, _fs = 'x-large', _fs_legend='medium',
        _ls = '-', _loc=0):
    "Plot the ccdf of a data array"
    data = pylab.array(data_in, copy=True)
    data.sort()
    data_len = len(data)
    ccdf = 1 - pylab.arange(data_len)/(data_len - 1.0)
    pylab.plot(data, ccdf, 'k', lw = _lw, drawstyle = 'steps',
            label = _name, ls = _ls)
    pylab.xlabel(_xlabel, size = _fs)
    pylab.ylabel(_ylabel, size = _fs)
    pylab.title(_title, size = _fs)
    font = FontProperties(size = _fs_legend)
    pylab.legend(loc = _loc, prop = font)

def ccdfplot(_file, col = 0, xlabel = 'X', ylabel = r'1 - P(X$\leq$x)',
        title = 'Empirical Distribution', name = 'Data',
        _lw = 2, _fs = 'x-large', _ls = '-', _loc=0):
    "Plot the ccdf of a column in file"
    data = pylab.loadtxt(_file, usecols = [col])
    ccdfplotdata(data, _xlabel = xlabel, _ylabel = ylabel,
            _title = title, _name = name,
            _lw = _lw, _fs = _fs, _ls = _ls, _loc = _loc)

    #CDF
def cdfplotdataN(list_data_name, _xlabel = 'x', _ylabel = r'P(X$\leq$x)',
        _title = 'Empirical Distribution', _fs = 'x-large',
        _fs_legend='medium', _loc = 0, do_color=True, logx=True, logy=False):
    "Plot the cdf of a list of names and data arrays"
    #corresponding line width with larger width for '-.' and ':'
    if not list_data_name:
        print("no data to plot", sys.stderr)
        return
    _ls = ['-', '-.', '-', '--'] * 2 #, ':']
#    _lw = [1, 1] + [2, 4, 2, 4, 2, 4]#, 4]
    _lw = [2, 4] + [2, 4, 2, 4, 2, 4]#, 4]
    assert len(_ls) == len(_lw)
#    _colors = ['k', 'k', 'g', 'c', 'm', 'r', 'y', 'pink']
    # consequent plots are same color
    _colors = ['k', 'k', 'c', 'c', 'm', 'm', 'y', 'y']
    for i in range(len(list_data_name)):# - 1):
        name, data = list_data_name[i]
        #plot with round robin line style (ls)
        #and increasing line width
        (div, mod) = divmod(i, len(_ls))
        if not do_color:
            color = 'k'
#            line_width = _lw[mod]+2*div
        else:
            color = _colors[i % len(_colors)]
#            line_width = 2 + div
        line_width = _lw[mod]+2*div
        cdfplotdata(data, _name=name, _title=_title, _xlabel=_xlabel,
                _ylabel=_ylabel, _lw=line_width, _ls=_ls[mod], _fs=_fs,
                _color=color)
    if logx and logy:
        setgraph_loglog(_loc=_loc, _fs_legend=_fs_legend)
    elif logy:
        setgraph_logy(_loc=_loc, _fs_legend=_fs_legend)
    elif logx:
        setgraph_logx(_loc=_loc, _fs_legend=_fs_legend)
    else:
        setgraph_lin(_loc=_loc, _fs_legend=_fs_legend)
#        cdfplotdata(data, _name=name, _lw=line_width, _ls=_ls[mod],
#                _fs=_fs, _color=color)
#    for last cdf, we put the legend and names
#    (data, name) = list_data_name[-1]
#    (div, mod) = divmod(len(list_data_name), len(_ls))
#    if not do_color:
#        color = 'k'
#        line_width = _lw[mod]+2*div
#    else:
#        color = _colors[i % len(_colors)]
#        line_width = 1 + div
#    cdfplotdata(data, _name=name, _title=_title, _xlabel=_xlabel,
#            _ylabel=_ylabel, _lw=line_width, _ls=_ls[mod], _fs=_fs,
#            _fs_legend=_fs_legend, _color=color)

def cdfplotdata(data_in, _color='k', _xlabel='x', _ylabel=r'P(X$\leq$x)',
        _title='Empirical Distribution', _name='Data', _lw=2, _fs='x-large',
        _fs_legend='medium', _ls = '-', _loc=0):
    "Plot the cdf of a data array"
#    data = pylab.array(data_in, copy=True)
    data = sorted(data_in)
    data_len = len(data)
    if data_len == 0:
        print("no data to plot", sys.stderr)
        return
    cdf = pylab.arange(data_len+1)/(data_len - 0.0)
    data.append(data[-1])
    pylab.plot(data, cdf, _color, lw = _lw, drawstyle = 'steps',
               label = _name + ': %d' % data_len, ls = _ls)
    pylab.xlabel(_xlabel, size = _fs)
    pylab.ylabel(_ylabel, size = _fs)
    pylab.title(_title, size = _fs)
    font = FontProperties(size = _fs_legend)
    pylab.legend(loc = _loc, prop = font)

def cdfplot(_file, col = 0, xlabel = 'X',
        ylabel = r'P(X$\leq$x)',
        title = 'Empirical Distribution', name = 'Data',
        _lw = 2, _fs = 'x-large', _ls = '-', _loc=0):
    "Plot the cdf of a column in file"
    data = pylab.loadtxt(_file, usecols = [col])
    cdfplotdata(data, _xlabel = xlabel, _ylabel = ylabel,
                _title = title, _name = name,
                _lw = _lw, _fs = _fs, _ls = _ls, _loc = _loc)

def setgraph_lin(_fs = 'x-large', _loc = 2, _fs_legend = 'medium'):
    "Set graph in xlogscale and adjusts x&y markers"
    pylab.grid(True)
    _ax = pylab.gca()
    for tick in _ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(_fs)
    for tick in _ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(_fs)
    font = FontProperties(size = _fs_legend)
    pylab.legend(loc = _loc, prop = font)


def setgraph_logx(_fs = 'x-large', _loc = 2, _fs_legend = 'medium'):
    "Set graph in xlogscale and adjusts x&y markers"
    pylab.grid(True)
    pylab.semilogx(nonposy='clip', nonposx='clip')
    _ax = pylab.gca()
    for tick in _ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(_fs)
    for tick in _ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(_fs)
    font = FontProperties(size = _fs_legend)
    pylab.legend(loc = _loc, prop = font)


def setgraph_loglog(_fs = 'x-large', _loc = 2, _fs_legend = 'medium'):
    "Set graph in xlogscale and adjusts x&y markers"
    pylab.grid(True)
    pylab.loglog(nonposy='clip', nonposx='clip')
    _ax = pylab.gca()
    for tick in _ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(_fs)
    for tick in _ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(_fs)
    font = FontProperties(size = _fs_legend)
    pylab.legend(loc = _loc, prop = font)

def setgraph_logy(_fs = 'x-large', _loc = 2, _fs_legend = 'medium'):
    "Set graph in xlogscale and adjusts x&y markers"
    pylab.grid(True)
    pylab.semilogy(nonposy='clip', nonposx='clip')
    _ax = pylab.gca()
    for tick in _ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(_fs)
    for tick in _ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(_fs)
    font = FontProperties(size = _fs_legend)
    pylab.legend(loc = _loc, prop = font)


#repartition plots
def repartplotdataN(list_data_name, _xlabel = 'Rank',
        _ylabel = 'Cumulative Percentage of Data',
        _title = 'Repartition of values',
        _fs = 'x-large', do_color=True, _loc=0, loglog=True):
    "Plot the repartition of a list of data arrays and names"
    #corresponding line width with larger width for '-.' and ':'
    if not list_data_name:
        print("no data to plot", sys.stderr)
        return
    _ls = ['-', '-.', '-', '--'] * 2 #, ':']
#    _ls = ['-', '-.', '--', ':']
    _lw = [2, 4] + [2, 4, 2, 4, 2, 4]#, 4]
#    _lw = [1, 2, 3, 4]
    assert len(_ls) == len(_lw)
    _len_ls = len(_ls)
    # consequent plots are same color
    _colors = ['k', 'k', 'c', 'c', 'm', 'm', 'y', 'y']
    for i in range(len(list_data_name)):# - 1):
        name, data = list_data_name[i]
        #plot with round robin line style (ls)
        #and increasing line width
        (div, mod) = divmod(i, _len_ls)
        if not do_color:
            color = 'k'
#            line_width = _lw[mod]+2*div
        else:
            color = _colors[i % len(_colors)]
#            line_width = 2 + div
        line_width = _lw[mod]+2*div
        repartplotdata(data, _name=name, _title=_title, _xlabel=_xlabel,
                       _ylabel=_ylabel, _lw=line_width, _ls=_ls[mod], _fs=_fs,
                       _color=color)
    if loglog:
        setgraph_loglog(_loc=_loc)
    else:
        setgraph_lin(_loc=_loc)

#    #for last cdf, we put the legend and names
#    (name, data) = list_data_name[-1]
#    (div, mod) = divmod(len(list_data_name), _len_ls)
#    repartplotdata(data, _name=name, _title=_title, _xlabel=_xlabel,
#            _ylabel=_ylabel, _lw=_lw[mod]+2*div, _ls=_ls[mod], _fs=_fs)
#    setgraph_loglog(_loc=_loc)

def repartplotdata(data_in, _color='k', _xlabel = 'Rank',
        _ylabel = 'Cumulative Percentage of Data',
        _title = 'Repartition of values', _name = 'Data', _lw = 2,
        _fs = 'x-large', _fs_legend='medium', _ls = '-', _loc=0):
    "Plot the repartition of a data array"
    data = pylab.array(data_in, copy=True)
    data.sort()
    rank = pylab.arange(1, len(data) + 1)
    values = pylab.cumsum(data[::-1])
    pylab.plot(rank, 100 * values / values[-1], _color, lw = _lw,
               drawstyle = 'steps', label = _name + ': %d' % len(data),
               ls = _ls)
    pylab.xlabel(_xlabel, size = _fs)
    pylab.ylabel(_ylabel, size = _fs)
    pylab.title(_title, size = _fs)
    font = FontProperties(size = _fs_legend)
    pylab.legend(loc = _loc, prop = font)

def repartplot(_file, col = 0, xlabel = 'Rank',
        ylabel = 'Cumulative Percentage of Data',
        title = 'Repartition of values', name = 'Data',
        _lw = 2, _fs = 'x-large', _ls = '-', _loc=0):
    "Plot the cdf of a column in file"
    data = pylab.loadtxt(_file, usecols = [col])
    repartplotdata(data, _xlabel = xlabel, _ylabel = ylabel,
            _title = title, _name = name,
            _lw = _lw, _fs = _fs, _ls = _ls, _loc = _loc)

def main():
    "Program wrapper."
    usage = "%prog -r data_file [-c col -x x_label -y y_label -t title \
            -n data_name -lw line_width -fs fontsize [-g|-p]]"

    parser = OptionParser(usage = usage, version="%prog " + _VERSION)
    parser.add_option("-r", dest = "file",
            help = "input data file or stdin if FILE is -")
    parser.add_option("-c", dest = "col", type = "int", default = 0,
            help = "column in the file [default value = 0]")
    parser.add_option("-x", dest = "xlabel", default = 'X',
            help = "x label")
    parser.add_option("-y", dest = "ylabel",
            default = r'P(X$\leq$x)', help = "y label")
    parser.add_option("-t", dest = "title",
            default = 'Empirical Distribution',
            help = "graph title")
    parser.add_option("-n", dest = "name", default = 'Data',
            help = "data name")
    parser.add_option("-l", "--lw", dest = "lw", type = "int",
            default = 2, help = "line width")
    parser.add_option("-f", "--fs", dest = "fs", type = "int",
            default = 18, help = "font size")
    parser.add_option("-g", "--ccdf", dest = "g",
            action="store_true", default=False,
            help = "plot ccdf instead of cdf")
    parser.add_option("-p", "--repartition", dest = "p",
            action="store_true", default=False,
            help = "plot repartition instead of cdf")
    (options, _) = parser.parse_args()

    if not options.file:
        print("Must provide filename.")
        parser.print_help()
        exit(1)

    if options.file == '-':
        out_file = sys.stdin
    else:
        try:
            out_file = open(options.file, 'r')
        except IOError:
            print("File, %s, does not exist." % options.file)
            parser.print_help()
            exit(1)


    if options.g and options.p:
        print("g and p options are exclusive.")
        parser.print_help()
        exit(1)

    pylab.clf()
    if options.g:
        ccdfplot(out_file, col=options.col, _lw=options.lw, _fs=options.fs)
    elif options.p:
        repartplot(out_file, col=options.col, _lw=options.lw, _fs=options.fs)
    else:
        cdfplot(out_file, col=options.col, xlabel=options.xlabel,
                ylabel=options.ylabel, title=options.title,
                name=options.name, _lw=options.lw, _fs=options.fs)
        setgraph_logx(_fs = options.fs)
    pylab.show()

if __name__ == '__main__':
    sys.exit(main())
