#!/usr/bin/env python
"Module to aggregate all pdf figures of a directory \
into a single latex file, and compile it."

import os
import sys
import re
from optparse import OptionParser

_VERSION = '1.0'

def latex_dir(outfile, dir, column=2):
    print dir
    outfile.write(r"""\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8x]{inputenc}
\usepackage{fancyheadings}
\def\goodgap{\hspace{\subfigtopskip}\hspace{\subfigbottomskip}}

\usepackage[pdftex]{graphicx}
\usepackage{subfigure,a4wide}

%%set dimensions of columns, gap between columns, and paragraph indent
\setlength{\textheight}{8in}
%%\setlength{\textheight}{9.3in}
\setlength{\voffset}{0.5in}
\setlength{\topmargin}{-0.55in}
%%\setlength{\topmargin}{0in}
\setlength{\headheight}{0.0in}
%%\setlength{\headsep}{0.0in}

%%\setlength{\textwidth}{7.43in}
\setlength{\textwidth}{7.10in}
%%\setlength{\textwidth}{6in}
\setlength{\hoffset}{-0.4in}
\setlength{\columnsep}{0.25in}

\setlength{\oddsidemargin}{0.0in}
\setlength{\evensidemargin}{0.0in}

%% more than .95 of text and figures
\def\topfraction{.95}
\def\floatpagefraction{.95}
\def\textfraction{.05}

\newcommand{\mydefaultheadersandfooters}
{
 \chead{\today}
 \rhead{\thepage}
 \lfoot{}
 \cfoot{}
 \rfoot{}
}


\title{Automatically generated latex for directory %(title)s}
\date{}
\begin{document}
\pagestyle{fancy}
\mydefaultheadersandfooters

\maketitle
\clearpage
""" % {'dir': dir, 'title': dir.replace('_', '\_')})
    files = os.listdir(os.getcwd() + '/' + dir)
    exclude_filename = outfile.name.split('/')[-1].replace('.tex', '.pdf')
    pattern = re.compile(r'(?!%s)\w+\.pdf' % exclude_filename)
    count = 0
    if column == 1:
        line_size = .99
    elif column == 2:
        line_size = .49
    else:
        print "invalid column size"
        raise
    nb_floats = 0
    for f in sorted(files):
        if pattern.match(f):
            nb_floats += 1
            if column == 1 or count % 2 == 0:
                outfile.write(r"\begin{figure}[h!] \
                        \begin{center}")
            outfile.write(r"\subfigure[]{\includegraphics[width=%f\textwidth,height=%f\textheight]{%s/%s}}"
                    % (line_size, .7*line_size, dir, f))

            if column == 1 or count % 2 != 0:
                outfile.write('\n' + r"\caption{}\end{center}\end{figure}" + '\n')
                if nb_floats >= 4:
                    outfile.write(r"\clearpage")
                    nb_floats = 0
            elif count % 2 == 0:
#                outfile.write('\goodgap')
                pass
            else:
                print "Double column and modulo is not working on count: %d" \
                    % count
                raise
            count += 1
    outfile.write('\n' + r"\caption{}\end{center}\end{figure}" + '\n')
    outfile.write(r"\end{document}")
    outfile.flush()

def main():
    usage = "%prog -d dir [-c nb_of_columns -w outtexfile]"

    parser = OptionParser(usage = usage)
    parser.add_option("-d", dest = "dir", type = "string",
                      help = "directory to search for pdf files")
    parser.add_option("-w", dest = "outtexfile", type = "string",
                      help = "output latex file (default is dir/latex_dir.tex)")
    parser.add_option("-c", dest = "column", type = "int", default = 2,
            help = "number of columns of latex file: 1 or 2")
    (options, args) = parser.parse_args()

    if not options.dir:
        print "no directory given"
        parser.print_help()
        exit()
    if not options.outtexfile:
        outfile = open(os.sep.join((options.dir, 'latex_dir.tex')), 'w')
    else:
        outfile = open(options.outtexfile, 'w')
    if options.column not in (1, 2):
        print "invalid number of columns"
        parser.print_help()
        exit()

    latex_dir(outfile, options.dir, options.column)

    #compile the tex file
    os.execlp('pdflatex', 'pdflatex', '-interaction=nonstopmode',
           '-output-directory', options.dir, outfile.name)

if __name__ == '__main__':
    sys.exit(main())

