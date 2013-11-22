#!/usr/bin/env python2.6
"""Module to load dipcp csv file and store it in python binary format.
Also give the possibility to filter a dipcp array according to some GVB fields.
"""

import sys
from optparse import OptionParser
import numpy as np
import re
from collections import defaultdict

import INDEX_VALUES
#from streaming_tools import un_mismatch_dscp
import streaming_tools


def return_cnx_id_GVB(GVB_line):
    "Return the connection ID of GVB flow."
    proto_nb2id = defaultdict(str, ((6, 'TCP'), (17, 'UDP')))
    if GVB_line['direction'] == 0:
        return (GVB_line['srcAddr'], GVB_line['srcPort'],
                GVB_line['dstAddr'], GVB_line['dstPort'],
                proto_nb2id[GVB_line['protocol']])
    elif GVB_line['direction'] == 1:
        return (GVB_line['dstAddr'], GVB_line['dstPort'],
                GVB_line['srcAddr'], GVB_line['srcPort'],
                proto_nb2id[GVB_line['protocol']])
    else:
        return None

def return_cnx_id_dipcp(dipcp_line):
    "Return the connection ID of dipcp flow."
    return (dipcp_line['FlowIPSource'],
            dipcp_line['FlowPortSource'],
            dipcp_line['FlowIPDest'],
            dipcp_line['FlowPortDest'],
            dipcp_line['LastPacketProtocol'])

def return_cnx_id_dipcp_reversed(dipcp_line):
    "Return the connection ID of dipcp flow."
    return (dipcp_line['FlowIPDest'],
            dipcp_line['FlowPortDest'],
            dipcp_line['FlowIPSource'],
            dipcp_line['FlowPortSource'],
            dipcp_line['LastPacketProtocol'])

def add_AS_dipcp(datas, dipcp_flows):
    """Add AS infos from GVB flows into dipcp flows
    input: dict of GVB (also dipcp) flows with AS data
        dict of dicpcp flows to add new data
    output:
        modify input dipcp flows by adding AS infos
    use as:
       datas = tools.load_hdf5_data.load_h5_file('flows/hdf5/traces_lzf.h5')
       dipcp_streaming = cPickle.load(open('dipcp_streaming_loss.pickle'))
       add_AS_dipcp(datas, dipcp_streaming)
    """
    traces = dipcp_flows.keys()
    for trace in traces:
        print "Processing trace: ", trace
        flow2as = {}
        GVB_flow = datas[trace + '_GVB']
        # operate only on streaming down so reduce table
        dscp_streaming, _, _ = streaming_tools.un_mismatch_dscp(GVB_flow)
        GVB_flow_down = GVB_flow.compress(
            GVB_flow['direction']==INDEX_VALUES.DOWN)
        GVB_flow_down_streaming = GVB_flow_down.compress(
            GVB_flow_down['dscp']==dscp_streaming)
        for GVB_line in GVB_flow_down_streaming:
            flow2as[return_cnx_id_GVB(GVB_line)] = (GVB_line['asBGP'],
                                                    GVB_line['asSrc'],
                                                    GVB_line['orgSrc'],
                                                    GVB_line['asDst'],
                                                    GVB_line['orgDst'])
        print "nb of GVB flows IDs: ", len(flow2as)
        dipcp_flow = dipcp_flows[trace].compress(
            dipcp_flows[trace]['TOS'] == 4 * dscp_streaming)
        old_dtype = dipcp_flow.dtype
        dtype_AS = np.dtype(old_dtype.descr + [('asBGP', np.uint16),
             ('asSrc', np.uint16),
             ('orgSrc', (np.str_, 32)),
             ('asDst', np.uint16),
             ('orgDst', (np.str_, 32))])
        dipcp_flow_AS = []
        print "nb of dipcp flows records before: ", len(dipcp_flow)
        dipcp_flows[trace] = np.array(
            [extend_fields_AS_dipcp(d, flow2as[return_cnx_id_dipcp(d)])
             for d in dipcp_flow],
            dtype=dtype_AS)
        print "nb of dipcp flows records after: ", len(dipcp_flow)

def extend_fields_AS_dipcp(d, new_fields):
    "Extend each line of array considered as list with both IP addresses."
    return tuple(list(d) + list(new_fields))



def filter_dipcp_array(dipcp_flows, GVB_flows,
                       field=None, value=None):
    """Return an array with dipcp flows corresponding to the value of field in
    GVB flow"""
    cnxs_list = set(return_cnx_id_GVB(x) for x in GVB_flows
                    if (field and x[field] == value) or not field)
    return np.array([x for x in dipcp_flows if
                     return_cnx_id_dipcp(x) in cnxs_list],
                    dtype=dipcp_flows.dtype, copy=True)

def filter_dipcp_dict(data, field=None, values=(None,)):
    "Wrapper to filter all flows according to a GVB field"
    traces = set(key.strip('_DIPCP').strip('_GVB') for key in data)
    filtered_data = {}
    for value in values:
        for trace in traces:
            filtered_data[trace] = filter_dipcp_array(data[trace + '_DIPCP'],
                    data[trace + '_GVB'])
            #, field=field, value=value)
    return filtered_data

def construct_dtype_and_converters(header, delimiter=INDEX_VALUES.sep_dipcp):
    dtype = []
    converters = {}
    header = header.rstrip().split(delimiter)
    for i, f in enumerate(header):
        f = re.sub('-[\d]+s-', '-', f)
        if f == '':
            t = '|S1' #(np.str_, 1)
            conv=np.str_
        elif  re.match('Time|\w*PacketDate|DIP[-\w]+Milli[sS]econds', f):
            # time format
            t = np.float_
            conv = lambda s: np.float_(s or 0)
        elif re.match('FlowIP|LastPacketIPSource', f):
            t = '|S16' #(np.str_, 16)
            conv=np.str_
        elif re.match('FlowEth', f):
            t = '|S17' #(np.str_, 17)
            conv=np.str_
        elif re.match('FlowPort', f):
            t = np.uint16
            conv = lambda s: np.uint16(s or 0)
        elif re.match('DIP[-\w]+Number-Packets-', f):
            t = np.uint32
            conv = lambda s: np.uint32(s or 0)
        elif re.match('DIP-Volume-Sum-Bytes-', f):
            t = np.uint64
            conv = lambda s: np.uint64(s or 0)
        elif re.match('DIP-Thp-Number-Kbps-|ts-', f):
            t = np.float_
            conv = lambda s: np.float_(s or 0)
        elif f == 'LastPacketProtocol':
            t = '|S5' #(np.str_, 5)
            conv=np.str_
        elif re.match('[\w]*LastTcpPacketType', f):
            t = '|S5' #(np.str_, 5)
            conv=np.str_
        elif re.match('SynCounter-', f):
            t = np.uint8
            conv = lambda s: np.uint8(s or 0)
        elif re.match('(?!Size)[A-Z][a-z]+-[1-9]', f):
            t = np.ubyte
            conv = lambda s: np.ubyte(s or 0)
        elif re.match('Size-[1-9]', f):
            t = np.uint16
            conv = lambda s: np.uint16(s or 0)
        elif f == 'TOS':
            t = np.ubyte
            conv = lambda s: np.ubyte(s or 0)
        elif f == 'LastPacketSize':
            t = np.uint16
            conv = lambda s: np.uint16(s or 0)
        elif re.match('DIP[-\w]+NbMes-', f):
            t = np.uint32
            conv = lambda s: np.uint32(s or 0)
        elif re.match('DIP[-\w]+Mean-', f):
            t = np.float_
            conv = lambda s: np.float_(s or 0)
        elif re.match('DIP[-\w]+(?:Min|Max)-ms-', f):
            t = np.float_
            conv = lambda s: np.float_(s or 0)
            # added for radius analysis
        elif f == 'IMSI':
            # IMSI and IP address both have 16 chars
            t = '|S16' #(np.str_, 16) #'|S16'
            conv=np.str_
        elif f == 'RAT':
            t = np.uint8
            conv = lambda s: np.uint8(s or 0)
        elif f == 'SGSN':
            t = '|S16' #(np.str_, 16)
            conv=np.str_
        elif f == 'IMEI':
            t = '|S48' #(np.str_, 48)
            conv=np.str_
        elif f == 'CellId':
            t = np.int_
            conv = lambda s: np.int_(s or 0)
        elif f == 'Constructor':
            t = '|S30'
            conv=np.str_
        elif f == 'DevType':
            t = '|S10'
            conv=np.str_
        elif re.match('DIP[-\w]+(?:Min|Max)-s-', f):
            t = np.float_
            conv = lambda s: np.float_(s or 0)
        else:
            raise Exception, "unknown format: " + f
        name = f
        while name in [fi for (fi, ty) in dtype]:
            name = '_'.join((name, 'TWICE'))
        dtype.append((name, t))
#        print t, type(t)
        converters[i] = conv
#        if type(t) is not tuple:
#            # automatic converter does not work for tuples
#            converters[i] = lambda s: t(s or 0)
#        else:
#            converters[i] = lambda s: str(s)
    return dtype, converters
#    return dtype

#def generic_converter(t):
#    try:
#        result = t(

def load_and_save_dipcp(in_file, dtype=None, converters=None, outfile=None,
        delimiter=INDEX_VALUES.sep_dipcp):
    "Load a file in dipcp format and store it in binary format."
    if in_file == '-':
        data = sys.stdin
    else:
        data = open(in_file, 'r')
    skip = 1
    if not(dtype and converters):
        #parse the header if not type specifyed
        dtype, converters = construct_dtype_and_converters(data.readline(),
                delimiter=delimiter)
        skip -= 1
    if not outfile:
        outfile = '%s' % '_'.join(in_file.split('.'))
    #use pytables
    np.save(outfile, np.loadtxt(data, dtype=dtype, converters=converters,
                                delimiter=delimiter, skiprows=skip))
#    test = np.load(outfile + '.npy')
#    print test.dtype
    data.close()

def main():
    usage = "%prog -r data_file_1 [-r data_file_2 ... -r data_file_n] \
[-w single_out_file]"

    parser = OptionParser(usage = usage)
    parser.add_option("-r", dest = "files", type = "string", action = "append",
                      help = "list of input dipcp csv files [- for stdin]")
    parser.add_option("-w", dest = "outfile", type = "string", default = None,
            help = "output file (python binary): use only for single file \
[DEFAULT=FILENAME.npy]")
    parser.add_option("-d", dest = "delimiter", default = None,
            help = "field delimiter")
    (options, args) = parser.parse_args()

    if not options.files:
        parser.print_help()
        return 1

    if options.outfile and len(options.files) > 1:
        parser.print_help()
        return 1

    if not options.outfile:
        options.outfile = '%s' % '_'.join(options.files[0].split('.'))

    for f in options.files:
        load_and_save_dipcp(f, outfile=options.outfile)

if __name__ == '__main__':
    sys.exit(main())

#flows_dipcp_FTTH_2008 = np.loadtxt(
#'dipcp_output_juill_2008_FTTH/20080701200002_20080701211711_Flows.csv',
#dtype=tools.INDEX_VALUES.dtype_dipcp, skiprows=1,
#delimiter=tools.INDEX_VALUES.sep_dipcp,
#converters=tools.INDEX_VALUES.converters_dipcp)
