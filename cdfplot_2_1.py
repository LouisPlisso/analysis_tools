#!/usr/bin/env python
"Module to plot cdf from data or file. Can be called directly."

from __future__ import division, print_function

from optparse import OptionParser
import logging
import sys
import os
from itertools import cycle
import numpy as np

# for removing matplotlib warnings
import warnings
warnings.showwarning = lambda *args: None

# in case of non-interactive usage
#import matplotlib
#matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import colorConverter

_VERSION = '2.1'

MARKERS_PER_LINE = 15

# possibility to place legend outside graph:
#pylab.subfigure(111)
#pylab.subplots_adjust(right=0.8) or (top=0.8)
#pylab.legend(loc=(1.1, 0.5)

# for interactive call: do not add multiple times the handler
if 'LOG' not in locals():
    LOG = None
LOG_LEVEL = logging.INFO
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

class CdfFigure(object):
    "Hold the figure and its default properties"
    def __init__(self, xlabel='x', ylabel=r'P(X$\leq$x)',
                 title='Empirical Distribution', fontsize='xx-large',
                 legend_fontsize='large', legend_ncol=1, markerscale=None,
                 subplot_top=None):
        self._figure = plt.figure()
        if subplot_top:
            self._figure.subplotpars.top = subplot_top
        self._axis = self._figure.add_subplot(111)
        self._lines = {}
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.fontsize = fontsize
        self.legend_fontsize = legend_fontsize
        self.markerscale = markerscale
        self.legend_ncol = legend_ncol

    def savefig(self, *args, **kwargs):
        "Saves the figure: interface to plt.Figure.savefig"
        # new in matplotlib 1.1
        if 'tight_layout' in kwargs and kwargs['tight_layout']:
            plt.tight_layout()
        self._figure.savefig(*args, **kwargs)

    def bar(self, *args, **kwargs):
        "Plot in the axis: interface to plt.Axes.bar"
        self._axis.bar(*args, **kwargs)

    def plot(self, *args, **kwargs):
        "Plot in the axis: interface to plt.Axes.plot"
        self._axis.plot(*args, **kwargs)

    def get_xlim(self, *args, **kwargs):
        "Plot in the axis: interface to plt.Axes.get_xlim()"
        return self._axis.get_xlim(*args, **kwargs)

    def set_xlim(self, *args, **kwargs):
        "Plot in the axis: interface to plt.Axes.set_xlim()"
        self._axis.set_xlim(*args, **kwargs)

    def set_ylim(self, *args, **kwargs):
        "Plot in the axis: interface to plt.Axes.set_ylim()"
        self._axis.set_ylim(*args, **kwargs)

    @staticmethod
    def simplify_cdf(data):
        '''Return the cdf and data to plot
        Remove unnecessary points in the CDF in case of repeated data
        '''
        data_len = len(data)
        assert data_len != 0
        cdf = np.arange(data_len) / data_len
        simple_cdf = [0]
        simple_data = [data[0]]
        if data_len > 1:
            simple_cdf.append(1 / data_len)
            simple_data.append(data[1])
            for cdf_value, data_value in zip(cdf, data):
                if data_value == simple_data[-1]:
                    simple_cdf[-1] = cdf_value
                else:
                    simple_cdf.append(cdf_value)
                    simple_data.append(data_value)
        assert len(simple_cdf) == len(simple_data)
        # to have cdf up to 1
        simple_cdf.append(1)
        simple_data.append(data[-1])
        return simple_cdf, simple_data

    def cdfplot(self, data_in, name='Data', finalize=False, count_samples=True):
        """Plot the cdf of a data array
        Wrapper to call the plot method of axes
        """
        # cannot shortcut lambda, otherwise it will drop values at 0
        data = sorted(filter(lambda x: (x is not None and ~np.isnan(x)
                                        and ~np.isinf(x)),
                             data_in))
        data_len = len(data)
        if data_len == 0:
            LOG.info("no data to plot")
            return
        simple_cdf, simple_data = self.simplify_cdf(data)
        if count_samples:
            label = name + ': %d' % len(data)
        else:
            label = name
        line = self._axis.plot(simple_data, simple_cdf,
                               drawstyle='steps', label=label)
        self._lines[name] = line[0]
        if finalize:
            self.adjust_plot()

    def ccdfplot(self, data_in, name='Data', finalize=False,
                 count_samples=True):
        """Plot the cdf of a data array
        Wrapper to call the plot method of axes
        """
        data = sorted(filter(lambda x: x is not None, data_in))
        data_len = len(data)
        if data_len == 0:
            LOG.info("no data to plot")
            return
        ccdf = 1 - np.arange(data_len + 1) / data_len
        # to have cdf up to 1
        data.append(data[-1])
        if count_samples:
            label = name + ': %d' % len(data)
        else:
            label = name
        line = self._axis.plot(data, ccdf, drawstyle='steps', label=label)
        self._lines[name] = line[0]
        if finalize:
            self.adjust_plot()

    def show(self):
        "Show the figure, and hold to do interactive drawing"
        self._figure.show()
        self._figure.hold(True)

    @staticmethod
    def generate_line_properties():
        "Cycle through the lines properties"
        colors = cycle('mgcb')
        line_width = 2.5
        dashes = cycle([(1, 0), (8, 5)]) #self.dash_generator()
        linestyles = cycle(['-'])
        #alphas = cycle([.3, 1.])
        markers = cycle(' oxv*d')
        while True:
            dash = dashes.next()
            yield (colors.next(), line_width, dash, linestyles.next(),
                   markers.next())
            yield (colors.next(), line_width, dash, linestyles.next(),
                   markers.next())
            dash = dashes.next()
            yield (colors.next(), line_width, dash, linestyles.next(),
                   markers.next())
            yield (colors.next(), line_width, dash, linestyles.next(),
                   markers.next())

    def adjust_lines(self, dashes=True, leg_loc='best'):
        """Put correct styles in the axes lines
        Should be launch when all lines are plotted
        Optimised for up to 8 lines in the plot
        """
        generator = self.generate_line_properties()
        for key in sorted(self._lines):
            (color, line_width, dash, linestyle, marker) = generator.next()
            line = self._lines[key]
            line.set_color(color)
            line.set_lw(line_width)
            line.set_linestyle(linestyle)
            if dashes:
                line.set_dashes(dash)
            line.set_marker(marker)
            line.set_markersize(12)
            line.set_markeredgewidth(1.5)
            line.set_markerfacecolor('1.')
            line.set_markeredgecolor(color)
            # we want at most MARKERS_PER_LINE markers per line
            x_len = len(line.get_xdata())
            if x_len < MARKERS_PER_LINE:
                markevery = 1
            else:
                markevery = 1 + x_len // MARKERS_PER_LINE
            line.set_markevery(markevery)
        self.adjust_plot(leg_loc=leg_loc)

    def adjust_plot(self, leg_loc='best'):
        "Adjust main plot properties (grid, ticks, legend)"
        self.put_labels()
        self.adjust_ticks()
        self._axis.grid(True)
        #self._axis.legend(loc=leg_loc, ncol=self.legend_ncol)
        self.legend(loc=leg_loc)

    def put_labels(self):
        "Put labels for axes and title"
        self._axis.set_xlabel(self.xlabel, size=self.fontsize)
        self._axis.set_ylabel(self.ylabel, size=self.fontsize)
        self._axis.set_title(self.title, size=self.fontsize)

    def legend(self, loc='best'):
        "Plot legend with correct font size"
        font = FontProperties(size=self.legend_fontsize)
        self._axis.legend(loc=loc, ncol=self.legend_ncol, prop=font,
                          markerscale=self.markerscale)

    def adjust_ticks(self):
        """Adjusts ticks sizes
        To call after a rescale (log...)
        """
        self._axis.minorticks_on()
        for tick in self._axis.xaxis.get_major_ticks():
            tick.label1.set_fontsize(self.fontsize)
        for tick in self._axis.yaxis.get_major_ticks():
            tick.label1.set_fontsize(self.fontsize)

    def setgraph_logx(self):
        "Set graph in xlogscale and adjusts plot (grid, ticks, legend)"
        self._axis.semilogx(nonposy='clip', nonposx='clip')

    def setgraph_logy(self):
        "Set graph in xlogscale and adjusts plot (grid, ticks, legend)"
        self._axis.semilogy(nonposy='clip', nonposx='clip')

    def setgraph_loglog(self):
        "Set graph in xlogscale and adjusts plot (grid, ticks, legend)"
        self._axis.loglog(nonposy='clip', nonposx='clip')

    def cdfplotdata(self, list_data_name, **kwargs):
        "Method to be able to append data to the figure"
        cdfplotdata(list_data_name, figure=self, **kwargs)

    def ccdfplotdata(self, list_data_name, **kwargs):
        "Method to be able to append data to the figure"
        cdfplotdata(list_data_name, figure=self, cdf=False, **kwargs)

def cdfplotdata(list_data_name, figure=None, xlabel='x', loc='best',
                fs_legend='large', title = 'Empirical Distribution', logx=True,
                logy=False, cdf=True, dashes=True, plot_line=None,
                legend_ncol=1, markerscale=None, xlim=None, ylim=None,
                count_samples=True):
    "Plot the cdf of a list of names and data arrays"
    if not figure:
        figure = CdfFigure(xlabel=xlabel, title=title, markerscale=markerscale,
                           legend_fontsize=fs_legend, legend_ncol=legend_ncol)
    else:
        figure.title = title
        figure.xlabel = xlabel
        figure.legend_fontsize = fs_legend
        figure.legend_ncol = legend_ncol
        figure.markerscale = markerscale
    if not list_data_name:
        LOG.info("no data to plot")
        return figure
    if plot_line:
        for x_value, label in plot_line:
            figure.plot((x_value, x_value), [0, 1],
                             linewidth=2, color='red', label=label)
    for name, data in list_data_name:
        if cdf:
            figure.cdfplot(data, name=name, count_samples=count_samples)
        else:
            figure.ccdfplot(data, name=name, count_samples=count_samples)
            figure.ylabel = r'1 - P(X$\leq$x)'
    if logx and logy:
        figure.setgraph_loglog()
    elif logy:
        figure.setgraph_logy()
    elif logx:
        figure.setgraph_logx()
    figure.adjust_lines(dashes=dashes, leg_loc=loc)
    if xlim:
        figure._axis.set_xlim(xlim)
    if ylim:
        figure._axis.set_ylim(ylim)
    return figure

def cdfplot(in_file, col=0):
    "Plot the cdf of a column in file"
    data = np.loadtxt(in_file, usecols = [col])
    cdfplotdata(('Data', data))

def scatter_plot(data, title='Scatterplot', xlabel='X', ylabel='Y',
                 logx=False, logy=False):
    "Plot a scatter plot of data"
    figure = CdfFigure(title=title, xlabel=xlabel, ylabel=ylabel)
    x, y = zip(*data)
    figure.plot(x, y, linestyle='', marker='^', markersize=8,
             markeredgecolor='b', markerfacecolor='w')
    if logx and logy:
        figure.setgraph_loglog()
    elif logy:
        figure.setgraph_logy()
    elif logx:
        figure.setgraph_logx()
    figure.adjust_plot()
    return figure

def scatter_plot_multi(datas, title='Scatterplot', xlabel='X', ylabel='Y',
                 logx=False, logy=False):
    "Plot a scatter plot of dictionary"
    figure = CdfFigure(title=title, xlabel=xlabel, ylabel=ylabel)
    markers = cycle('^xo')
    colors = cycle('brm')
    transparent = colorConverter.to_rgba('w', alpha=1)
    total_nb = len([x for y in datas.values() for x in y])
    for label, data in sorted(datas.items()):
        x, y = zip(*data)
        figure.plot(x, y,
                    label=(r'%s: %d (\textbf{%d\%%})'
                           % (label, len(data), 100 *len(data) // total_nb)),
                    linestyle='', marker=markers.next(), markersize=8,
                    markeredgecolor=colors.next(), markerfacecolor=transparent)
    if logx and logy:
        figure.setgraph_loglog()
    elif logy:
        figure.setgraph_logy()
    elif logx:
        figure.setgraph_logx()
    figure.adjust_plot()
    return figure

def bin_plot(datas, title='Bin Plot', xlabel='X', ylabel='Y',
             logx=False, logy=False):
    "Plot a bin plot of dictionary"
    figure = CdfFigure(title=title, xlabel=xlabel, ylabel=ylabel)
#    linestyles = cycle(('-', '--'))
#    markers = cycle('^xo')
#    colors = cycle('brm')
#    for label, data in datas:
    left, width, height, yerr = zip(*datas)
    figure.bar(left, height, width, linewidth=0) #, yerr=yerr)
#                    linestyle=linestyles.next(), marker=markers.next(),
#                    markersize=6, markeredgecolor=colors.next(),
#                    markerfacecolor='w')
    if logx and logy:
        figure.setgraph_loglog()
    elif logy:
        figure.setgraph_logy()
    elif logx:
        figure.setgraph_logx()
    figure.adjust_plot()
    return figure

def main():
    'Program wrapper'
    usage = ('%prog [-c col -x x_label -y y_label -t title '
             '-n data_name --line line_value --linelabel line_label'
             '-lw line_width -fs fontsize --xmin x_min --xmax x_max'
             '[-g|-p]] data_file|-')
    parser = OptionParser(usage = usage, version='%prog ' + _VERSION)
    parser.add_option('-w', dest='out_file', default='cdf_plot.pdf',
                      help='output filename for output (default cdf_plot.pdf)')
    parser.add_option('-c', dest='col', type='int', default=0,
                      help='column in the file [default value = 0]')
    parser.add_option('-x', dest='xlabel', default='X', help='x label')
    parser.add_option('--line', dest='line', action='append', type='float',
                      help='plot a veritcal line in the CDF')
    parser.add_option('--xmin', dest='xmin', type='float', help='Min x value')
    parser.add_option('--xmax', dest='xmax', type='float', help='Max x value')
    parser.add_option('--linelabel', dest='linelabel',
                      action='append', help='line label')
    parser.add_option('-t', dest='title', default='Empirical Distribution',
                      help='graph title')
    parser.add_option('-n', dest='name', default='Data', help='data name')
    parser.add_option('-l', '--lw', dest='lw', type='int', default=2,
                      help='line width')
    parser.add_option('-f', '--fs', dest='fs', default='large',
                      help='font size')
    parser.add_option('--logx', dest='logx', action='store_true', default=True,
                      help='set logx axis (default)')
    parser.add_option('--nologx', dest='logx', action='store_false',
                      default=True, help='unset logx axis')
    parser.add_option('--logy', dest='logy', action='store_true',
                      default=False, help='set logy axis')
    parser.add_option('--nology', dest='logy', action='store_false',
                      default=False, help='unset logy axis (default)')
    parser.add_option('-d', '--delimiter',
                      help='delimiter in the input file (default any whitespace)')
    (options, arg_list) = parser.parse_args()
    if not arg_list or len(arg_list) != 1:
        parser.error('Must provide one filename')
    if arg_list[0] == '-':
        in_file = sys.stdin
    else:
        try:
            in_file = open(arg_list[0], 'r')
        except IOError:
            parser.error('File, %s, does not exist.' % arg_list[0])
    try:
        data = np.loadtxt(in_file, usecols = [options.col],
                          delimiter=options.delimiter)
    except ValueError, mes:
        LOG.exception('problem in importing data: %s' % mes)
    plot_line = []
    if options.line:
        if len(options.line) != len(options.linelabel):
            LOG.info('labels are not corresponding to lines: '
                     'skipping all labels')
            options.linelabel = ['label %d' % i
                                 for i in range(len(options.line))]
        try:
            map(float, options.line)
        except ValueError, mes:
            LOG.exception('problem for line value: %s' % mes)
        plot_line = zip(options.line, options.linelabel)
    xlim = None
    if options.xmin or options.xmax:
        xlim = [options.xmin, options.xmax]
    figure = cdfplotdata([(options.name, data)], figure=None,
                         xlabel=options.xlabel, loc='best',
                         fs_legend=options.fs, title=options.title,
                         plot_line=plot_line, logx=options.logx,
                         logy=options.logy, cdf=True, dashes=True,
                         legend_ncol=1, xlim=xlim, ylim=None)
                         #legend_ncol=1, xlim=[1, 1e8], ylim=None)
    figure.savefig(options.out_file)

if __name__ == '__main__':
    sys.exit(main())
