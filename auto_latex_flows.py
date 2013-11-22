#!/usr/bin/env python
"Module to generate a latex table dynamically out of flows stats."

import sys
import os
from optparse import OptionParser
import re
import numpy as np
from itertools import repeat

import INDEX_VALUES
import aggregate
import compute_AT

from streaming_tools import un_mismatch_dscp, format_title

def generate_general_table(datas, nb_cols = 4,
        file_name = 'general_tables', output_path = 'rapport/resume'):
    """Generate latex table of flows resume, the nb of cols is without
    description column.
    Launch as:
#    data = tools.load_hdf5_data.load_h5_file('hdf5/lzf_data.h5')
#    names = (('ADSL_juil_2008', data['ADSL_2008_GVB']),
#        ('FTTH_juil_2008', data['FTTH_2008_GVB']),
#        ('ADSL_nov_2009', data['ADSL_nov_2009_GVB']),
#        ('FTTH_nov_2009', data['FTTH_nov_2009_GVB']),
#        ('ADSL_dec_2009', data['ADSL_dec_2009_GVB']),
#        ('FTTH_dec_2009', data['FTTH_dec_2009_GVB']),
#        ('FTTH_fev_2010', data['FTTH_Montsouris_2010_02_07_GVB']),
#        ('FTTH_mar_2010', data['FTTH_mar_2010_GVB']))
#    tools.auto_latex_flows.generate_general_table(names)
    With new data (correctly labeled):
    datas = tools.load_hdf5_data.load_h5_file('flows/hdf5/traces_lzf.h5')
    tools.auto_latex_flows.generate_general_table(datas)
"""
#    flows = {}
#    for (order, (name, data_flow)) in enumerate(flows_name_list):
#        flows['%d_%s' % (order, name)] = data_flow

    template = [('App: %s', '', 's'),
            ('total_vol%s', 'Bytes', '.3g'),
            ('vol_up%s', 'Bytes', '.3g'),
            ('vol_down%s', 'Bytes', '.3g'),
            ('nb_client_down%s', 'Nb', 'd'),
            ('nb_client_down_1MB%s', 'Nb', 'd'),
            ('avg_vol_down_per_client%s', 'Bytes', '.3g'),
            ('nb_flows_down%s', 'Nb', '.4g'),
            ('nb_flows_down_1MB%s', 'Nb', '.4g'),
            ('avg_vol_down_per_flow%s', 'Bytes', '.3g')]
    formatting = [(name % replace, unit, form)
            for replace in ('', '_WEB', '_HTTP_STREAM', '_OTHER_STREAM')
            for (name, unit, form) in template]
    tables = {}
    traces = set(key.strip('_DIPCP').strip('_GVB') for key in datas)
    for trace in sorted(traces):
        tables[trace] = format_resume_table(trace,
                fetch_data_general(datas[trace + '_GVB']), formatting)
        #print >> open('%s/temp_file_%s.tex' % (output_path, name), 'w'), \
        #        tables[name]
    print_tables(tables, nb_cols, output_path, file_name)

def generate_new_table(flows, nb_cols = 3,
        file_name = 'new_tables', output_path = 'rapport/resume_new'):
    """Generate latex table of flows resume, the nb of cols is without
    description column.
    Launch as:
        flows_stream = cPickle.load(open('flows_gvb_streams.pickle'))
        tools.auto_latex_flows.generate_new_table(flows_stream)
"""
    template = [('App: %s', '', 's'),
#            ('total_vol%s', 'Bytes', '.3g'),
            ('vol_up%s', 'Bytes', '.3g'),
#            ('vol_down%s', 'Bytes', '.3g'),
            ('nb_client_down%s', 'Nb', 'd'),
            ('nb_client_down_1MB%s', 'Nb', 'd'),
            ('avg_vol_up_per_client%s', 'Bytes', '.3g'),
            ('nb_flows_down%s', 'Nb', '.4g'),
            ('nb_flows_down_1MB%s', 'Nb', '.4g'),
            ('ratio_nb_flows%s', '\%', 'd'),
            ('avg_vol_up_per_flow%s', 'Bytes', '.3g'),
            ('avg_nb_flows_per_client%s', 'Nb', '.2f'),
            ('avg_nb_flows_1MB_per_client%s', 'Nb', '.2f')]
    formatting = [(name % replace, unit, form)
            for replace in ('_HTTP_STREAM', )
            for (name, unit, form) in template]
    tables = {}
    traces = set(name for name in flows if name.find('DEEZER') == -1)
    for trace in sorted(traces):
        short_name = trace.replace('AS_YOUTUBE', 'YT'
                                  ).replace('AS_DAILYMOTION', 'DM'
                                  ).replace('AS_DEEZER', 'DZ'
                                  ).replace('AS_GOOGLE', 'GOO')
        tables[trace] = format_resume_table(short_name,
                fetch_data_general(flows[trace], filtered=True), formatting)
        #print >> open('%s/temp_file_%s.tex' % (output_path, name), 'w'), \
        #        tables[name]
    print_tables(tables, nb_cols, output_path, file_name)



def generate_streaming_table(datas, nb_cols = 4, flows_th = 10,
        file_name = 'streaming_tables', output_path = 'rapport/resume'):
    """Generate latex table of flows resume, the nb of cols is without
    description column.
    Launch as:
#    data = tools.load_hdf5_data.load_h5_file('hdf5/lzf_data.h5')
#    names = (('ADSL_2008', data['ADSL_2008_GVB']),
#        ('FTTH_2008', data['FTTH_2008_GVB']),
#        ('ADSL_nov_2009', data['ADSL_nov_2009_GVB']),
#        ('FTTH_nov_2009', data['FTTH_nov_2009_GVB']),
#        ('ADSL_dec_2009', data['ADSL_dec_2009_GVB']),
#        ('FTTH_dec_2009', data['FTTH_dec_2009_GVB']),
#        ('FTTH_fev_2010', data['FTTH_Montsouris_2010_02_07_GVB']),
#        ('FTTH_mar_2010', data['FTTH_mar_2010_GVB']))
#    tools.auto_latex_flows.generate_streaming_table(names)
    With new data (correctly labeled):
        datas = tools.load_hdf5_data.load_h5_file('flows/hdf5/traces_lzf.h5')
        tools.auto_latex_flows.generate_streaming_table(datas)
"""
    flows = {}
    traces = set(key.strip('_DIPCP').strip('_GVB') for key in datas)
    for trace in traces:
        data_flow = datas[trace + '_GVB']
        flows_tmp = {}
        (dscp_http_stream, _, _) = un_mismatch_dscp(data_flow)
        flows_tmp['flows_%s_stream' % trace] = data_flow.compress(
            data_flow['dscp'] == dscp_http_stream)
        flows_tmp['flows_%s_stream_down' % trace] = flows_tmp['flows_%s_stream'
                % trace].compress(flows_tmp['flows_%s_stream' % trace]
                        ['direction'] == INDEX_VALUES.DOWN)
        del flows_tmp['flows_%s_stream' % trace]
        flows['%s_YT' % trace] = flows_tmp['flows_%s_stream_down'
                % trace].compress([x['asBGP'] in INDEX_VALUES.AS_YOUTUBE
                    for x in flows_tmp['flows_%s_stream_down' % trace]])
        flows['%s_YT_EU' % trace] = flows_tmp['flows_%s_stream_down'
                % trace].compress([x['asBGP'] in INDEX_VALUES.AS_YOUTUBE_EU
                    for x in flows_tmp['flows_%s_stream_down' % trace]])
        flows['%s_GOO' % trace] = flows_tmp['flows_%s_stream_down'
                % trace].compress([x['asBGP'] in INDEX_VALUES.AS_GOOGLE
                    for x in flows_tmp['flows_%s_stream_down' % trace]])
        flows['%s_OTHER' % trace] = flows_tmp['flows_%s_stream_down'
                % trace].compress([x['asBGP'] not in
                    set(INDEX_VALUES.AS_YOUTUBE_EU +
                        INDEX_VALUES.AS_YOUTUBE + INDEX_VALUES.AS_GOOGLE)
                    for x in flows_tmp['flows_%s_stream_down' % trace]])
        del flows_tmp['flows_%s_stream_down' % trace]
        # remove non signicative flows
        for formater in ('%s_YT', '%s_YT_EU', '%s_GOO', '%s_OTHER'):
            if len(flows[formater % trace]) < flows_th:
                del flows[formater % trace]
        del flows_tmp
    #flows_name_len = len(flows_name_list)

    tables = {}
    formatting = ( ('vol_down', 'Bytes', '.3g'),
            ('nb_client', 'Nb', 'd'), ('nb_client_1MB', 'Nb', 'd'),
            ('nb_flow', 'Nb', 'd'), ('nb_flow_1MB', 'Nb', 'd'),
            ('mean_flow_size', 'Bytes', '.4g'),
            ('median_flow_size', 'Bytes', '.4g'),
            ('max_flow_size', 'Bytes', '.4g'),
            ('mean_flow_duration', 'Seconds', '.4g'),
            ('median_flow_duration', 'Seconds', '.4g'),
            ('max_flow_duration', 'Seconds', '.4g'),
            ('mean_flow_mean_rate', 'kb/s', '.4g'),
            ('median_flow_mean_rate', 'kb/s', '.4g'),
            ('max_flow_mean_rate', 'kb/s', '.4g'),
            ('mean_flow_mean_rate_1MB', 'kb/s', '.4g'),
            ('median_flow_mean_rate_1MB', 'kb/s', '.4g'),
            ('max_flow_mean_rate_1MB', 'kb/s', '.4g'),
            ('mean_flow_peak_rate', 'b/s', '.4g'),
            ('median_flow_peak_rate', 'b/s', '.4g'),
            ('max_flow_peak_rate', 'b/s', '.4g'),
            ('mean_flow_AR', 'cns/ks', '.4g'),
            ('mean_flow_100_AR_per_cl', 'cns/ks/cl', '.4g'))
    for name in sorted(flows.keys()):
        tables[name] = format_resume_table(name,
                fetch_data_http_stream_down(flows[name]), formatting)
        #print >> open('%s/temp_file_%s.tex' % (output_path, name), 'w'), \
        #        tables[name]
    print_tables(tables, nb_cols, output_path, file_name)

def print_tables(tables, nb_cols, output_path, file_name):
    """Prints the formatted table into multiple files and a global one to
    include."""
    new_tables = join_latex_table(tables, nb_cols)
    cwd = os.getcwd()
    all_tables = ""
    for (i, table) in enumerate(new_tables):
        all_tables = ''.join((all_tables, r"\begin{table}\begin{center}",'\n'))
        print >> open("%s/%s_%d.tex" % (output_path, file_name, i), 'w'), table
        all_tables = ''.join((all_tables, r"\input{%s/%s/%s_%d.tex}" \
                % (cwd, output_path, file_name, i),'\n'))
        all_tables = ''.join((all_tables, r"\end{center}\end{table}",'\n'))
    print >> open("%s/%s_%s.tex" % (output_path, file_name, 'all'), 'w'), \
            all_tables
    print "%d tables generated" % len(new_tables)

def get_dscp(app, in_flow, filtered=False):
    "Return DSCP value of selected application (based on special syntax)."
    if not filtered:
        (dscp_http_stream, dscp_other_stream, dscp_web) = un_mismatch_dscp(
            in_flow)
    else:
        (dscp_http_stream, dscp_other_stream, dscp_web) = tuple(repeat(
            in_flow[0]['dscp'], 3))
    value = {}
    value['_WEB'] = dscp_web
    value['_HTTP_STREAM'] = dscp_http_stream
    value['_OTHER_STREAM'] = dscp_other_stream
    try:
        return value[app]
    except KeyError:
        print '%s not defined as DSCP value' % app
        return -1

def fetch_data_general(in_flow, filtered=False):
    "Return a resume of interesting \
    flows characteristics."

#            ('nb_down_flows_1MB%s', 'Nb', '.4g'),
#            ('avg_vol_down_per_flow%s', 'Bytes', '.4g'),
#            ('avg_vol_down_per_client%s', 'Bytes', '.4g')]

    #new_flows = {}
    resume = {}
    for app in ('', '_WEB', '_HTTP_STREAM', '_OTHER_STREAM'):
        if app == '':
            new_flow = in_flow #['data%s' % app]
        else:
            dscp = get_dscp(app, in_flow, filtered=filtered)
            new_flow = in_flow.compress(in_flow['dscp'] == dscp)
            #['data%s' % app]
        resume['App: %s' % app] = ''
        new_flow = new_flow.view(np.recarray)
        new_flow_down = new_flow.compress(new_flow.direction
                == INDEX_VALUES.DOWN)
        new_flow_1MB = new_flow.compress(new_flow.l3Bytes > 10**6)
        new_flow_down_1MB = new_flow_down.compress(new_flow_down.l3Bytes
                > 10**6)
        vol_dir = aggregate.aggregate(new_flow, 'direction', 'l3Bytes', sum)
#        resume['vol_up%s' % app] = vol_dir[0][1]
#        resume['vol_down%s' % app] = vol_dir[1][1]
        try:
            resume['vol_up%s' % app] = vol_u = vol_dir[0][1]
        except IndexError:
            resume['vol_up%s' % app] = vol_u = float(0)
        try:
            resume['vol_down%s' % app] = vol_d = vol_dir[1][1]
        except IndexError:
            resume['vol_down%s' % app] = vol_d = float(0)
        resume['total_vol%s' % app] = vol_u + vol_d
        resume['nb_flows_down%s' % app] = nb_fl = len(new_flow_down)
        resume['nb_flows_down_1MB%s' % app] = nb_fl_1mb = len(new_flow_down_1MB)
        resume['ratio_nb_flows%s' % app] = int(100 * nb_fl_1mb / float(nb_fl)) \
                if (resume['nb_flows_down%s' % app] != 0) else 0
        resume['nb_client_down%s' % app] = nb_cl = len(np.unique(
            new_flow_down.client_id))
        resume['avg_vol_down_per_client%s' % app] = vol_d / nb_cl \
                if (nb_cl != 0) else 0
        resume['avg_vol_down_per_flow%s' % app] = vol_d / nb_fl \
                if (nb_fl != 0) else 0
        resume['avg_vol_up_per_client%s' % app] = vol_u / nb_cl \
                if (nb_cl != 0) else 0
        resume['avg_vol_up_per_flow%s' % app] = vol_u / nb_fl \
                if (nb_fl != 0) else 0
        resume['nb_client_down_1MB%s' % app] = len(np.unique(
            new_flow_down_1MB.client_id))
        resume['avg_nb_flows_per_client%s' % app] = nb_fl / float(nb_cl) \
                if nb_cl !=0 else 0
        resume['avg_nb_flows_1MB_per_client%s' % app] = nb_fl_1mb / float(nb_cl) \
                if nb_cl !=0 else 0
#    resume['nb_flow'] = len(flow)
#    resume['nb_client_1MB'] = len(np.unique(flows_1MB.client_id))
#    resume['nb_flow_1MB'] = len(flows_1MB)
#    resume['mean_flow_size'] = np.mean(flow.l3Bytes)
#    resume['median_flow_size'] = np.median(flow.l3Bytes)
#    resume['max_flow_size'] = np.int64(np.max(flow.l3Bytes))
#    resume['mean_flow_duration'] = np.mean(flow.duration)
#    resume['median_flow_duration'] = np.median(flow.duration)
#    resume['max_flow_duration'] = np.max(flow.duration)
#    resume['mean_flow_peak_rate'] = np.mean(80.0 * flow.peakRate)
#    resume['median_flow_peak_rate'] = np.median(80.0 * flow.peakRate)
#    resume['max_flow_peak_rate'] = np.max(80.0 * flow.peakRate)
#    mean_rate = [8*x['l3Bytes']/(1000.0*x['duration'])
#            for x in flow if x['duration']>0]
#    resume['mean_flow_mean_rate'] = np.mean(mean_rate)
#    resume['median_flow_mean_rate'] = np.median(mean_rate)
#    resume['max_flow_mean_rate'] = np.max(mean_rate)
#    mean_rate_1MB = [8*x['l3Bytes']/(1000.0*x['duration'])
#            for x in flow if x['duration']>0
#            and x['l3Bytes'] > 10**6]
#    resume['mean_flow_mean_rate_1MB'] = np.mean(mean_rate_1MB)
#    resume['median_flow_mean_rate_1MB'] = np.median(mean_rate_1MB)
#    resume['max_flow_mean_rate_1MB'] = np.max(mean_rate_1MB)
#    resume['mean_flow_AR'] = \
#            compute_AT.compute_AT(flow.initTime)[0]
#    resume['mean_flow_100_AR_per_cl'] = \
#            100 * resume['mean_flow_AR'] / resume['nb_client']
    return resume


def fetch_data_http_stream_down(flow):
    "Return a resume of interesting HTTP streaming \
    down flows characteristics."
    flow = flow.view(np.recarray)
    resume = {}
    flows_1MB = flow.compress(flow.l3Bytes > 10**6 )
    vol_dir = aggregate.aggregate(flow, 'direction', 'l3Bytes', sum)
    #resume['vol_up'] = vol_dir[0][1]
    resume['vol_down'] = vol_dir[0][1]
    #resume['total_vol'] = (resume['vol_down'] +
            #resume['vol_up'])
    resume['nb_client'] = len(np.unique(flow.client_id))
    resume['nb_flow'] = len(flow)
    resume['nb_client_1MB'] = len(np.unique(flows_1MB.client_id))
    resume['nb_flow_1MB'] = len(flows_1MB)
    resume['mean_flow_size'] = np.mean(flow.l3Bytes)
    resume['median_flow_size'] = np.median(flow.l3Bytes)
    resume['max_flow_size'] = np.int64(np.max(flow.l3Bytes))
    resume['mean_flow_duration'] = np.mean(flow.duration)
    resume['median_flow_duration'] = np.median(flow.duration)
    resume['max_flow_duration'] = np.max(flow.duration)
    resume['mean_flow_peak_rate'] = np.mean(80.0 * flow.peakRate)
    resume['median_flow_peak_rate'] = np.median(80.0 * flow.peakRate)
    resume['max_flow_peak_rate'] = np.max(80.0 * flow.peakRate)
    mean_rate = [8*x['l3Bytes']/(1000.0*x['duration'])
            for x in flow if x['duration']>0]
    resume['mean_flow_mean_rate'] = np.mean(mean_rate)
    resume['median_flow_mean_rate'] = np.median(mean_rate)
    resume['max_flow_mean_rate'] = np.max(mean_rate)
    mean_rate_1MB = [8*x['l3Bytes']/(1000.0*x['duration'])
            for x in flow if x['duration']>0
            and x['l3Bytes'] > 10**6]
    resume['mean_flow_mean_rate_1MB'] = np.mean(mean_rate_1MB)
    resume['median_flow_mean_rate_1MB'] = np.median(mean_rate_1MB)
    resume['max_flow_mean_rate_1MB'] = np.max(mean_rate_1MB)
    resume['mean_flow_AR'] = \
            compute_AT.compute_AT(flow.initTime)[0]
    resume['mean_flow_100_AR_per_cl'] = \
            100 * resume['mean_flow_AR'] / resume['nb_client']
    return resume

def format_resume_table(name, resume, formatting):
    """Formats a dictionnary of static description of a streaming capture into a
    latex tabular code."""
    title = format_title(name).split('\n')
    out = r"""\begin{tabular}{|l|c|}
\hline
& %s \\
\hline """ % title[1]
    out = ''.join((out, "\nDate & %s \\\\ \n\\hline" % title[0]))
    for (label, unit, formater) in formatting:
        if label not in resume.keys():
            raise Exception("Unknown label: %s" % label)
        out = ''.join((out, "\n{0} in {1} & {2:{3}} \\\\ \n".format(
            label.replace('_', ' '), unit, resume[label], formater)))
        out = ''.join((out, "\\hline\n"))
    return ''.join((out, r"\end{tabular}"))

def parse_latex_table(name, table):
    "Parse a latex tabular to retrieve header, data and description fields."
    cur_table = table.split('\n')
    cur_desc = []
    cur_data = []
    cur_header = cur_table[0]
    #parse only data: no begin or end table
    for line_nb in range(1, len(cur_table) - 1):
        line = cur_table[line_nb]
        if line == '':
            continue
        if line.startswith(r'\hline'):
            cur_desc.append(line)
            continue
        fields = re.split(r'&|\\\\', line)
        assert len(fields) == 3,  """Incorrect number of fields in table %s at
line %d""" % (name, line_nb)
        cur_desc.append(fields[0])
        cur_data.append(fields[1])
    else:
        #treat end of table
        cur_desc.append(cur_table[line_nb + 1])
    return (cur_header, cur_desc, cur_data)

def join_latex_table(tables, max_col=4):
    """Return tables with at most max_col data columns out of a dictionnary
    of latex tables. The tables must be of same shape with corresponding line
    titles."""
    data = []
    desc = None
    header = None
    #checks header and description column, and store data
    for name in sorted(tables.keys()):
        table = tables[name]
        (cur_header, cur_desc, cur_data) = parse_latex_table(name, table)
        if desc:
            assert desc == cur_desc
        else:
            desc = cur_desc
        if header:
            assert header == cur_header
        else:
            header = cur_header
        data.append(cur_data)
    return reconstruct_latex_table(header, desc, data, max_col)

def reconstruct_latex_table(header, desc, data, max_col):
    "Return the new tables out of data previously parsed and checked."
    new_tables = []
    table_offset = 0
    while table_offset < len(data):
        next_table_index = min(len(data), table_offset + max_col)
        #use + instead of ''.join() idiom for clarity
        fields = header.split('|')
        #assume header like: r'\begin{tabular}{|l|c|}'
        cols = next_table_index % max_col
        if cols == 0:
            cols = max_col
        new_table = '|'.join(fields[0:2]) + cols * ('|' + fields[2]) + '|' \
                + fields[3] + '\n'
        index_data = 0
        for line in desc:
            if line.startswith((r'\hline', '%')):
                new_table += line + '\n'
                continue
            #data line: first put description field
            new_table += line
            #all data table must have same size
            if index_data < len(data[0]):
                for j in range(table_offset, next_table_index):
                    new_table += '&' + data[j][index_data]
                index_data += 1
            new_table += r'\\' + '\n'
        table_offset = next_table_index
        new_tables.append(new_table)

    if len(data) % max_col > 0:
        new_tables_len = len(data) / max_col + 1
    else:
        new_tables_len = len(data) / max_col
    assert len(new_tables) == new_tables_len

    return new_tables

def separate_tabular(in_file):
    "Returns a list of tabulars containing only 2 columns: field and data."
    table = []
    #header must be in first line
    header = in_file.readline()
    #TODO: treat '||' separator
    # assume header as: \begin{tabular}{|l|c|c|c|c|c|c|c|c|}
    assert header.startswith(r"\begin{tabular}{|")
    assert header.endswith("|}\n")
    header = header.split('|')
    header_len = len(header)
    assert header_len  > 3
    for i in range(2, header_len - 1):
        table.append('|'.join([header[x] for x in (0, 1, i, header_len-1)])
                + '\n')
    for line in in_file.readlines():
        if line.startswith((r'\hline', '%')):
            for i in range(len(table)):
                table[i] += line + '\n'
            continue
        #TODO: multicols
        if line.find('multicols'):
            continue
        fields = re.split('[&$]', line)
        assert len(fields) == header_len - 2, "Expecting %d fields and getting \
only %d" % (header_len - 2, len(fields))
        for i in range(1, len(fields)):
            table[i - 1] += '&'.join([header[x] for x in (0, i)]) + r'\\' + '\n'
    return table


def main():
    "Program wrapper."
    usage = "%prog -r data_file [-n nb_cols -d dir]"

    parser = OptionParser(usage = usage)
    parser.add_option("-r", dest = "file", type = "string",
                      help = "input tex file")
    parser.add_option("-n", dest = "nb_cols", type = "int", default = 4,
            help = "number of columns of latex tabular")
    parser.add_option("-d", dest = "dir", type = "string", default = '.',
                      help = "directory to output tables")
    (options, _) = parser.parse_args()

    if not options.file:
        parser.print_help()
        return

    try:
        in_file = open(options.file, 'r')
    except IOError:
        print "File, %s, does not exist." % options.file
        parser.print_help()
        return 1

    tables = separate_tabular(in_file)
    #TODO: conversion to cope with existing code...
    dict_tables = dict(zip(range(len(tables)), tables))
    for (i, table) in enumerate(join_latex_table(dict_tables, options.nb_cols)):
        outfile = ''.join((options.dir, '/', options.file.replace('.', '_'),
            '_%d.tex' % i))
        open(outfile, 'w').write(table)

if __name__ == '__main__':
    sys.exit(main())
