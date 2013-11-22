#!/usr/bin/python

import sys
import pylab

fontsize='x-large'
linewidth=3

data=pylab.loadtxt(sys.argv[1], usecols=[int(sys.argv[2])])

n, bins, patches = pylab.hist(data, bins=len(data), normed=True, \
	color='k', cumulative=True, histtype='step', lw=linewidth, align='left')
pylab.xlabel('X', size=fontsize)
pylab.ylabel('F(X)', size=fontsize)
pylab.title('Empirical Distribution', size=fontsize)
pylab.grid(True)
pylab.semilogx()
ax = pylab.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
pylab.show()
