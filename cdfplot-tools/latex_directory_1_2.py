#!/usr/bin/env python
"Module to aggregate all pdf figures of a directory \
into a single latex file, and compile it."

from __future__ import division, print_function
import os
import sys
import re
from optparse import OptionParser

_VERSION = '1.0'

def latex_dir(outfile_name, directory, column=2, eps=False):
    "Print latex source file"
    print(directory)
    with open(outfile_name, 'w') as outfile:
        outfile.write(r"""\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8x]{inputenc}
\usepackage{fancyhdr}
\def\goodgap{\hspace{\subfigtopskip}\hspace{\subfigbottomskip}}


\usepackage%(include_pdf_package_option)s{graphicx}

\usepackage{subfigure,a4wide}

%%set dimensions of columns, gap between columns, and paragraph indent
\setlength{\textheight}{8in}
%%\setlength{\textheight}{9.3in}
\setlength{\voffset}{0.5in}
\setlength{\topmargin}{-0.55in}
%%\setlength{\topmargin}{0in}
\setlength{\headheight}{12.0pt}
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
\author{%(login)s}
\begin{document}
\pagestyle{fancy}
\mydefaultheadersandfooters

\maketitle
\clearpage
""" % {'title': directory.replace('_', '\_'),
       'login': os.getenv('LOGNAME').capitalize(),
       'include_pdf_package_option': '' if eps else '[pdftex]'})
        files = os.listdir(os.getcwd() + '/' + directory)
#    exclude_filename = outfile.name.split('/')[-1].replace('.tex', '.pdf')
        exclude_filename = 'latex_dir_'
        pattern = re.compile(r'(?!%s)\S+\.%s' % (exclude_filename,
                                                 ('eps' if eps else 'pdf')))
        count = 0
        if column == 1:
            line_size = .99
        elif column == 2:
            line_size = .49
        else:
            print("invalid column size")
            raise
        nb_floats = 0
        for cur_file in sorted(files):
            if pattern.match(cur_file):
                nb_floats += 1
                if column == 1 or count % 2 == 0:
                    outfile.write(r"\begin{figure}[!ht]"
                                  r"\begin{center}")
                outfile.write(r"\subfigure[]{\includegraphics" +
                          r"[width=%f\textwidth,height=%f\textheight]{%s/%s}}"
                              % (line_size, .7*line_size, directory, cur_file))
                if column == 1 or count % 2 != 0:
                    outfile.write('\n' + r"\caption{}\end{center}\end{figure}"
                                  + '\n')
                    if nb_floats >= 4:
                        outfile.write(r"\clearpage")
                        nb_floats = 0
                elif count % 2 == 0:
#                outfile.write('\goodgap')
                    pass
                else:
                    print("Double column and modulo is not working on count: %d"
                          % count)
                    raise
                count += 1
        if count % 2 == 1:
            outfile.write('\n' + r"\caption{}\end{center}\end{figure}" + '\n')
        outfile.write(r"\end{document}")

def main():
    "Option parsing and launch latex_dir"
    usage = "%prog [-c nb_of_columns -w outtexfile] directory_list"
    parser = OptionParser(usage = usage)
    parser.add_option('-w', dest='outtexfile', type='string',
                      help='output latex file (default is dir/latex_dir.tex)')
    parser.add_option('-c', dest='column', type='int', default = 2,
                      help='number of columns of latex file: 1 or 2')
    parser.add_option('--eps', dest='eps', default=False, action='store_true',
                      help='use eps files instead of pdf')
    (options, args) = parser.parse_args()
    if not args:
        parser.print_help()
        exit(5)
    for directory in args:
        if not options.outtexfile:
            outfile_name = os.sep.join((directory,
                                        'latex_dir_%s.tex' % directory))
        else:
            outfile_name = options.outtexfile
        if options.column not in (1, 2):
            print("invalid number of columns")
            parser.print_help()
            exit(5)
        latex_dir(outfile_name, directory, options.column, eps=options.eps)
        #compile the tex file
        if options.eps:
            os.execlp('latex', 'latex', '-interaction=nonstopmode',
                      '-output-directory', directory, outfile_name)
        else:
            os.execlp('pdflatex', 'pdflatex', '-interaction=nonstopmode',
                      '-output-directory', directory, outfile_name)

if __name__ == '__main__':
    sys.exit(main())

