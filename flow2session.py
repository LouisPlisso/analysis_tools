#!/usr/bin/env python
"""Module to construct sessions from flows
A session is an aggregation of flows overlaping or consecutive but with at most
N seconds (default: 30) of silence.
Modified for wired case: focus on streaming
"""

from __future__ import print_function
from optparse import OptionParser
import numpy as np
import pylab
from collections import defaultdict
import sys

import INDEX_VALUES

# default session gap
GAP = 600
# min duration of a flow (in case of incorrect decoding)
MIN_FLOW_DURATION = 1
OUT_SEPARATOR = ';'

DEBUG = False
ERRORS = defaultdict(int)

class StreamSession():
    "Streaming session class: keep track of streams quality"
    def __init__(self, s_client_id, f_init_time, c_len, c_dur, c_avg_br,
                 s_bytes, s_dur, f_skip):
        print('deprecated class')
        self.beg = f_init_time
        if c_avg_br != np.inf and c_avg_br > 0:
            self.end = f_init_time \
                    + float(s_bytes) / (1000 * float(c_avg_br) / 8)
        elif c_dur > 0 and c_len > 0:
            self.end = f_init_time \
                    + c_dur * float(s_bytes) / float(c_len)
        else:
            self.end = f_init_time + MIN_FLOW_DURATION
        self.client_id = s_client_id
        if s_dur > 0:
            self.thp_list = [float(s_bytes) / float(s_dur)]
        else:
            self.thp_list = [0]
        self.c_len_list = [c_len]
        self.c_dur_list = [c_dur]
#        self.rat_list = [f_rat]
#        self.hang_list = [f_hang]
        self.skip_list = [f_skip]
        self.tot_bytes = s_bytes

    def is_mine(self, client_id, f_init_time, s_bytes, c_avg_br, c_dur, c_len,
               gap):
        """Checks if the flow belongs to the session
        Flows must be sorted according to start time!!!
        """
        # must be the same client
        if self.client_id != client_id:
            return False
        # interpolation of stop date
        if c_avg_br > 0:
            stop = f_init_time + float(s_bytes) / (1000 * float(c_avg_br) / 8)
        elif c_dur > 0 and c_len > 0:
            stop = f_init_time  + c_dur * float(s_bytes) / float(c_len)
        else:
            ERRORS['stop_uncomputable'] += 1
            stop = f_init_time + MIN_FLOW_DURATION
        # disjoint and more than gap interval
        if f_init_time - self.end > gap or self.beg - stop > gap:
            return False
        # everything else is ok
        return True

    def update_session(self, f_init_time, c_len, c_dur, c_avg_br,
            s_bytes, s_dur, f_skip):
        "Extend session time and update stats, must check is_mine before call"
        self.beg = min(f_init_time, self.beg)
        if c_avg_br > 0:
            self.end = max(self.end, f_init_time
                           + float(s_bytes) / (1000 * float(c_avg_br) / 8))
        elif c_dur > 0 and c_len > 0:
            self.end = max(self.end, f_init_time \
                    + c_dur * float(s_bytes) / float(c_len))
        else:
            self.end = max(self.end, f_init_time + MIN_FLOW_DURATION)
        assert self.beg <= self.end, "error in interplation time: %f, %f\n" \
                % (self.beg, self.end)
        if s_dur > 0:
            self.thp_list.append(float(s_bytes) / s_dur)
        else:
            self.thp_list.append(0)
        self.c_len_list.append(c_len)
        self.c_dur_list.append(c_dur)
        self.skip_list.append(f_skip)
        self.tot_bytes += s_bytes

    def get_stats(self):
        "Returns sessions stats with throughput"
        return (self.client_id, self.beg, self.end,
                self.end - self.beg, self.tot_bytes, len(self.thp_list),
                min(self.thp_list), sum(self.thp_list) / len(self.thp_list))

class CnxStreamSession():
    "Streaming session class: keep track of streams quality"
    def __init__(self, record, client_field='Name'):
        self.client_field = client_field
        self.name = record[self.client_field]
        self.beg = float(record['StartDn'])
        self.end = CnxStreamSession.get_end_time(record)
        #self.end = int(record['StartDn']) + record['DurationDn']
        self.tot_bytes = record['ByteDn']
        self.nb_flows = 1

    @staticmethod
    def get_end_time(record):
        """Return the end time of a flow
        if possible interpolate it on the video rate"""
        if record.dtype == INDEX_VALUES.dtype_gvb_stream_indics:
            try:
                # video rate in b/s
                video_rate = (8 * record['Content-Length']
                              / record['Content-Duration'])
                duration = 8 * record['ByteDn'] / video_rate
            except ZeroDivisionError:
                duration = 1
        else:
            duration = record['DurationDn']
        return (float(record['StartDn'])
                + max(1, record['DurationDn'], duration))

    def is_mine(self, record, gap):
        """Checks if the flow belongs to the session
        Flows must be sorted according to start time!!!
        """
        # must be the same client
        if self.name != record[self.client_field]:
            return False
        # disjoint and more than gap interval
        if ((float(record['StartDn']) - self.end > gap)
            or (self.beg - CnxStreamSession.get_end_time(record) > gap)):
            return False
        # everything else is ok
        return True

    def update_session(self, record):
        "Extend session time and update stats, must check is_mine before call"
        self.beg = min(float(record['StartDn']), self.beg)
        self.end = max(self.end, CnxStreamSession.get_end_time(record))
        assert self.beg <= self.end, "error in interplation time: %f, %f\n" \
                % (self.beg, self.end)
        self.tot_bytes += record['ByteDn']
        self.nb_flows += 1

    def get_stats(self):
        "Returns sessions stats with throughput"
        return (self.name, self.beg, self.end,
                self.end - self.beg, self.tot_bytes, self.nb_flows)

    def print_stats(self, outfile):
        "Print sessions stats to file"
        print(OUT_SEPARATOR.join(map(str, self.get_stats())), file=outfile)

def clean_sessions(session_list, time, opened_file, gap=GAP):
    "Dumps old session to text file"
    removed_sessions = 0
    for session in session_list:
        if session.end - time > gap:
            session.print_stats(opened_file)
            session_list.remove(session)
            removed_sessions += 1
    if DEBUG:
        print(('%d sessions removed, remaining: %d' %
               (removed_sessions, len(session_list))), file=sys.stderr)

def process_cnx_sessions(data, out_file, gap=GAP, reset_errors=False,
                         client_field='Name'):
    "Process sessions for CnxStream files: new version"
#    for using datetime module:
#    datetime.datetime(*reversed(map(int, test[0]['Date'].split('/'))))
    assert (data.dtype == INDEX_VALUES.dtype_cnx_stream
            or data.dtype == INDEX_VALUES.dtype_cnx_stream_loss
            or data.dtype == INDEX_VALUES.dtype_stream_indics_tmp
            or data.dtype == INDEX_VALUES.dtype_gvb_stream_indics
            or data.dtype == INDEX_VALUES.dtype_rtt_stream_indics
            or data.dtype == INDEX_VALUES.dtype_all_stream_indics_final
            or data.dtype == INDEX_VALUES.dtype_all_stream_indics_final_good
            or data.dtype == INDEX_VALUES.dtype_all_stream_indics_final_tstat
            or data.dtype == INDEX_VALUES.dtype_all_stream_indics), (
                "not a valid data type: should be streaming cnx_stream")
    if reset_errors:
        ERRORS = defaultdict(int)
    dump_flows = 500
    dump_sessions = 500
    # first flow is just before midnight on the correct day
    date = data[0]['Date']
    data_date = np.sort(data.compress(data['Date'] == date), order='StartDn')
    len_data = len(data_date)
    with open(out_file, 'w') as opened_file:
        for client in np.unique(data_date[client_field]):
            data_client = np.sort(data_date.compress(
                data_date[client_field] == client), order='StartDn')
            record = data_client[0]
            current_session = CnxStreamSession(record,
                                               client_field=client_field)
            ERRORS['session_nb'] += 1
            for record in data_client[1:]:
                ERRORS['record_nb'] += 1
                if DEBUG and ERRORS['record_nb'] % dump_flows == 0:
                    print(('%d flows processed over %d'
                           % (ERRORS['record_nb'], len_data)), file=sys.stderr)
                if record['DurationDn'] < 0:
                    ERRORS['invalid_timestamps'] += 1
                    continue
                if current_session.is_mine(record, gap):
                    current_session.update_session(record)
                else:
                    # dumps previous session and create new one
                    current_session.print_stats(opened_file)
                    current_session = CnxStreamSession(record,
                                                   client_field=client_field)
                    ERRORS['session_nb'] += 1
                    if DEBUG and ERRORS['session_nb'] % dump_sessions == 0:
                        print('%d sessions' % ERRORS['session_nb'],
                              file=sys.stderr)
            # print last session
            current_session.print_stats(opened_file)
    print(ERRORS, file=sys.stderr)

def process_stream_session(data, gap=GAP, skip_nok=False,
                           client_field='dstAddr'):
    "Return the streaming sessions to file from a numpy array"
    assert (data.dtype == INDEX_VALUES.dtype_GVB_streaming
            or data.dtype == INDEX_VALUES.dtype_GVB_streaming_AS), \
            "not a valid data type: should be streaming GVB"
    session_list = []
    for client in np.unique(data[client_field]):
        if client == 0:
            ERRORS['record_no_' + client_field] += 1
            print(("Warning incorrect dstAddr at line: %d" %
                   ERRORS['record_nb']), file=sys.stderr)
            continue
        # adding flows in order is important
        data_client = np.sort(data.compress(
            data['dstAddr'] == client), order='initTime')
        record = data_client[0]
        current_session = StreamSession(record['dstAddr'],
                                        int(record['initTime']),
                                        record['Content-Length'],
                                        record['Content-Duration'],
                                        record['Content-Avg-Bitrate-kbps'],
                                        record['Session-Bytes'],
                                        record['Session-Duration'],
                                        record['nb_skips'])
        for record in data_client[1:]:
            ERRORS['record_nb'] += 1
            if record['valid'] != 'OK':
                ERRORS['record_nok'] += 1
                if DEBUG:
                    print("Warning record not checked line: %d" \
                          % ERRORS['record_nb'], file=sys.stderr)
                if skip_nok:
                    continue
            if record['Content-Avg-Bitrate-kbps'] < 1:
                ERRORS['record_low_bit_rate'] += 1
                continue
            if current_session.is_mine(record['dstAddr'],
                                       int(record['initTime']),
                                       record['Session-Bytes'],
                                       record['Content-Avg-Bitrate-kbps'],
                                       record['Content-Duration'],
                                       record['Content-Length'],
                                       gap):
                current_session.update_session(int(record['initTime']),
                                               record['Content-Length'],
                                               record['Content-Duration'],
                                           record['Content-Avg-Bitrate-kbps'],
                                               record['Session-Bytes'],
                                               record['Session-Duration'],
                                               record['nb_skips'])
            else:
                # session not found
                session_list.append(current_session)
                current_session = StreamSession(record['dstAddr'],
                                                int(record['initTime']),
                                                record['Content-Length'],
                                                record['Content-Duration'],
                                            record['Content-Avg-Bitrate-kbps'],
                                                record['Session-Bytes'],
                                                record['Session-Duration'],
                                                record['nb_skips'])
    print(ERRORS, file=sys.stderr)
    return np.array([session.get_stats() for session in session_list],
                            dtype=INDEX_VALUES.dtype_streaming_session)

def ip2int(ip):
    """Return the int value of IP address
    >>> ip2int('0.0.0.1')
    1
    >>> ip2int('0.0.1.0')
    256
    >>> ip2int('0.0.1.1')
    257
    >>> ip2int('255.255.255.255')
    4294967295
    """
    from re import match
    ret = 0
    match_ip = match("(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})", ip)
    if not match_ip:
        return 0
    for i in xrange(4):
        int_group = int(match_ip.groups()[i])
        assert int_group < 256, "Incorrect IP address"
        ret = (ret << 8) + int_group
    return ret

def get_header():
    "Print output file header"
    return OUT_SEPARATOR.join(("#", "ip2int", "DURATION", "NB", "MIN_THP",
        "AVG_THP", '\n'))

def plot_session_figs(in_file):
    """Plot some graphs on the generated session file:
    - thp vs duration
    - cdf of duration
    """
    sessions = np.load(in_file)
    # duration vs min thp
    pylab.clf()
    pylab.plot(sessions[:, 3], sessions[:, 1], 'k, ')
    pylab.loglog()
    axes = pylab.gca()
    pylab.grid()
    axes = pylab.gca()
    for tick in axes.xaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
    pylab.xlabel("Minimum throughput in kbps", size=16)
    for tick in axes.yaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
    pylab.ylabel("Session duration per client (sec)", size=16)
    pylab.savefig('%s_session_duration_vs_min_thp.pdf' % in_file)
    # nb flows vs avg thp
    pylab.clf()
    pylab.plot(sessions[:, 4], sessions[:, 2], 'k, ')
    pylab.loglog()
    pylab.grid()
#    axes = pylab.gca()
    pylab.xlabel("Average throughput in kbps", size=16)
    pylab.ylabel("Nb of flows", size=16)
    pylab.savefig('%s_session_nb_fl_vs_avg_thp.pdf' % in_file)
    pylab.clf()
    import cdfplot
    cdfplot.cdfplotdata(sessions[:, 1], _xlabel='Duration in seconds',
            _title='Session durations', _fs_legend='x-large')

def main():
    "Program wrapper"
    usage = "%prog -r flows_file [-t|-w out_file] [-s -p -g gap -k -n]"
    parser = OptionParser(usage = usage)
    parser.add_option("-r", dest = "flows_file", type = "string",
        action = "store", help = "input stream stats file in numpy form (.npy)")
    parser.add_option("-w", dest = "out_file", type = "string",
        action = "store", help = "output session file")
    parser.add_option("-g", dest = "gap", type = "int", default = GAP,
        action = "store", help = "interval between sessions (DEFAULT=%d)" % GAP)
    parser.add_option("-p", dest = "plot", action = "store_true",
            help = "flag to plot additionnal figures (DEFAULT=no)")
    parser.add_option("-s", dest = "stream_stats", action = "store_false",
            help = "flag to treat streaming stats (DEFAULT=yes) \
                      [toggle off if already computed]", default=True)
    parser.add_option("-k", dest = "header", action = "store_true",
            help = "flag to skip one line of header (DEFAULT=no)")
    parser.add_option("-n", dest = "skip", action = "store_true",
            help = "flag to skip nok records (DEFAULT=no)")
    parser.add_option("-t", dest = "cnx_stream", action = "store_true",
            help = "flag to calculate on cnx_streams files (DEFAULT=no)")
#    parser.add_option("-x", dest = "new", action = "store_true")
    (options, _) = parser.parse_args()
    if not options.flows_file:
        parser.print_help()
        exit()
#    if not options.stream_stats:
#        process_session(data, options.out_file, gap=options.gap,
#                skip=options.header)
#        if options.plot:
#            plot_session_figs(options.out_file)
#    else:
    if options.stream_stats:
        data = np.load(options.flows_file)
        if not options.cnx_stream:
            if not options.out_file:
                parser.print_help()
                exit()
            sessions_stats = process_stream_session(data, gap=options.gap,
                                                    skip_nok=options.skip)
            fmt = ('%s', '%f', '%f', '%f', '%d', '%d', '%f', '%f')
            np.save(open(options.out_file, 'w'),
                    sessions_stats)
            np.savetxt('.'.join((options.out_file.replace('.', '_'), 'gz')),
                       sessions_stats, fmt=OUT_SEPARATOR.join(fmt))
        else:
            process_cnx_sessions(data,
                             '.'.join((options.flows_file.replace('.', '_'),
                                       str(options.gap), 'txt')),
                             gap=options.gap)
#            fmt = ('%s', '%f', '%f', '%f', '%d', '%d')
    if options.plot:
        plot_session_figs(options.out_file)

if __name__ == '__main__':
#    print("run doctest")
#    from doctest import testmod
#    testmod()
    sys.exit(main())

