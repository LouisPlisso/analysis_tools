#!/usr/bin/python
"Compute arrival time of flows. Can be called directly."

import numpy as np 
import pylab

def compute_AT(data, _xlabel = 'Time (Minutes since beginning of capture)', 
               _ylabel = 'Nb of New Flows per Minute', 
               _title = 'Data', _lw = 0.3, _fs = 'x-large', _ls = '-', 
               _divide = 60.0): # 60.0 for min, 1.0 for sec
    "Compute and plot arrival time among a single column array of floats"
    assert (_divide == 1.0 or _divide == 60.0), 'bad input value'
    s = int(max(data)/_divide + 1)
    x = np.arange(s)
    y = np.zeros(s)
    prev = 0
    n = 0
    tot = 0
    for t in data:
        if int(t)/_divide == prev:
            n += 1
        else:
            y[prev] += n
            prev = int(t)/_divide
            n = 1
        tot += 1
    if _divide == 1.0:
        x = x / 60.0 # for plot in 
    pylab.plot(x, y, 'k', lw = _lw, drawstyle = 'steps', ls = _ls)
    ax = pylab.gca()
    for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(_fs)
    for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(_fs)
    pylab.xlabel(_xlabel, size = _fs)
    pylab.ylabel(_ylabel, size = _fs)
    pylab.title(_title, size = _fs)
    return np.mean(y),sum(y),tot

def main():
    from optparse import OptionParser
    usage ="%prog -r data_file {-c col | -b} [-t title -w output.pdf]"
    parser = OptionParser(usage = usage)
    parser.add_option("-r", dest = "file",
                      help = "input data file")
    parser.add_option("-c", dest = "col", type=int,
                      help = "init time column of data file")
    parser.add_option("-b", dest = "binary", action="store_true", default=False,
                      help = "indicate if file format is python binary")
    parser.add_option("-t", dest = "title", default="Data",
                      help = "title of graph")
    parser.add_option("-w", dest = "out_file", default=None,
                      help = "output graph file")
    (options, args) = parser.parse_args()

    if not ((options.file and options.col) or
            (options.file and options.binary)):
        parser.print_help()
        exit()
    
    if options.binary:
        data = np.load(options.file)
    else:
        data = pylab.loadtxt(options.file,usecols=[col])
    m,s,t = compute_AT(data, title=options.title)
    print m, s, t
    if options.out_file:
        pylab.savefig(options.out_file)

if __name__ == '__main__':
    main()
