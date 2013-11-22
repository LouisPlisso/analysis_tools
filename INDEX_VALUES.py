"Types and constant values definitions."

import numpy as np
from functional import foldl
import re
from copy import copy

class Everything(object):
    "Class to always match (if itereated)"
    def __contains__(self, *args, **kwargs):
        return True

class Nothing(object):
    "Class to never match (if itereated)"
    def __contains__(self, *args, **kwargs):
        return False

UNKNOWN_ID = ('0_0', 'Default_00:00:00:00:00:00')
EPSILON = 1e-10

TRACE_LIST = ('2008_07_01_ADSL', '2008_07_01_FTTH',
              '2009_11_26_ADSL', '2009_11_26_FTTH',
              '2009_12_14_ADSL_R', '2009_12_14_FTTH',
              '2010_02_07_ADSL_R', '2010_02_07_FTTH')

# FTTH 2009 starts at 14h, put 13h here for security
# more margin in the time
TIME_START = {'2008_07_01_ADSL': 64800,
              '2008_07_01_FTTH': 64800,
              '2009_11_26_ADSL': 64800,
              '2009_11_26_FTTH': 64800,
              '2009_12_14_ADSL_R': 64800,
              '2009_12_14_FTTH': 46800,
              '2010_02_07_ADSL_R': 64800,
              '2010_02_07_FTTH': 64800}

# 4000 is large enough for 1 hour
# ADSL 2008 trace is longer
TIME_STOP = {'2008_07_01_ADSL': 86400,
             '2008_07_01_FTTH': 76000,
             '2009_11_26_ADSL': 76000,
             '2009_11_26_FTTH': 76000,
             '2009_12_14_ADSL_R': 76000,
             '2009_12_14_FTTH': TIME_START['2009_12_14_FTTH'] + 8000,
             '2010_02_07_ADSL_R': 76000,
             '2010_02_07_FTTH': 76000}

ACCESS_DEFAULT_RATES = [128, 512, 1024, 1024, 2048, 2048,
                        8192, 8192, 18432]
ACCESS_RATES_NB_RENNES_2009_12 = [7, 150.4838709677, 1,
                                   754.4193548387, 15.064516129,
                                   76.1612903226, 852.2903225806,
                                   0, 125.9677419355]
ACCESS_RATES_NB_RENNES_2010_02 = [7, 132.4285714286, 1, 688.8035714286,
                                  14.0535714286, 71.5714285714,
                                  735.6428571429, 0, 108.5714285714]
ACCESS_RATES_NB_MONT_2008_07 = [12, 177.2580645161, 2, 422.935483871,
                                25.6774193548, 14, 931.8494623656, 0,
                                287.688172043]
ACCESS_RATES_NB_MONT_2009_11 = [11, 169, 0, 449, 23, 17, 892, 0, 292]

assert (len(ACCESS_RATES_NB_RENNES_2010_02)
        == len(ACCESS_RATES_NB_RENNES_2009_12)
        == len(ACCESS_RATES_NB_MONT_2008_07)
        == len(ACCESS_RATES_NB_MONT_2009_11)
        == len(ACCESS_DEFAULT_RATES))

#data array index: no more used
#PROTOCOL = 0
#CLIENT = 1
#DIR = 2
#INIT = 3
#DUR = 4
#BYTES = 5
#PEAK = 6
#AS = 7
#DSCP = 8

#dscp values
DSCP_WEB = 1
DSCP_OTHER_STREAM = 10
DSCP_HTTP_STREAM = 11

DSCP_MARCIN_HTTP_STREAM = 8
DSCP_MARCIN_OTHER_STREAM = 9
DSCP_MARCIN_WEB = 11

#TOS values
TOS_WEB = 4 * DSCP_WEB
TOS_OTHER_STREAM = 4 * DSCP_OTHER_STREAM
TOS_HTTP_STREAM = 4 * DSCP_HTTP_STREAM

#dir values
DOWN = 1
UP = 0

#AS values
AS_ACRONOC = (19166,)
AS_DAILYMOTION = (41690,)
AS_GOOGLE = (15169,)
AS_LIMELIGHT = (22822,)
AS_LIMELIGHT_CHI = (38621,)
AS_LIMELIGHT_AUS = (38622,)
AS_YOUTUBE_EU = (43515,)
AS_YOUTUBE = (36561, 1901)
AS_DEEZER = (39605,)
AS_MICROSOFT = (8068, 8075)
AS_MICROSOFT_LIVE = (35106,)
AS_GLOBAL_CROSSING = (3549,)
AS_CABLE_WIRELESS = (1273,)
AS_TINET = (3257,)

AS_ALL_YOUTUBE = AS_YOUTUBE + AS_YOUTUBE_EU
AS_ALL_GOOGLE = AS_ALL_YOUTUBE + AS_GOOGLE
AS_ALL_LIMELIGHT = AS_LIMELIGHT + AS_LIMELIGHT_CHI + AS_LIMELIGHT_AUS
AS_ALL_MICROSOFT = AS_MICROSOFT + AS_MICROSOFT_LIVE

# for tstat-2.0
dtype_tstat = [
    ('Client_IP_address', (np.str_, 16)),
    ('Client_TCP_port', np.uint16),
    ('C_packets', np.uint32),
    ('C_RST_sent', np.uint32),
    ('C_ACK_sent', np.uint32),
    ('C_PURE_ACK_sent', np.uint32),
    ('C_unique_bytes', np.uint64),
    ('C_data_packets', np.uint32),
    ('C_data_bytes', np.uint64),
    ('C_rexmit_packets', np.uint32),
    ('C_rexmit_bytes', np.uint64),
    ('C_out_seq_packets', np.uint32),
    ('C_SYN_count', np.uint32),
    ('C_FIN_count', np.uint32),
    ('C_RFC1323_ws', np.uint8),
    ('C_RFC1323_ts', np.uint8),
    ('C_window_scale', np.uint8),
    ('C_SACK_req', np.uint8),
    ('C_SACK_sent', np.uint32),
    ('C_MSS', np.uint16),
    ('C_max_seg_size', np.uint16),
    ('C_min_seg_size', np.uint16),
    ('C_win_max', np.uint32),
    ('C_win_min', np.uint32),
    ('C_win_zero', np.uint32),
    ('C_cwin_max', np.uint32),
    ('C_cwin_min', np.uint32),
    ('C_initial_cwin', np.uint32),
    ('C_Average_rtt', np.float_),
    ('C_rtt_min', np.float_),
    ('C_rtt_max', np.float_),
    ('C_Stdev_rtt', np.float_),
    ('C_rtt_count', np.float_),
    ('C_ttl_min', np.float_),
    ('C_ttl_max', np.float_),
    ('C_rtx_RTO', np.uint32),
    ('C_rtx_FR', np.uint32),
    ('C_reordering', np.uint32),
    ('C_net_dup', np.uint32),
    ('C_unknown', np.uint32),
    ('C_flow_control', np.uint32),
    ('C_unnece_rtx_RTO', np.uint32),
    ('C_unnece_rtx_FR', np.uint32),
    ('C_diff_SYN_seqno', np.uint8),
    ('Server_IP_address', (np.str_, 16)),
    ('Server_TCP_port', np.uint16),
    ('S_packets', np.uint32),
    ('S_RST_sent', np.uint32),
    ('S_ACK_sent', np.uint32),
    ('S_PURE_ACK_sent', np.uint32),
    ('S_unique_bytes', np.uint64),
    ('S_data_packets', np.uint32),
    ('S_data_bytes', np.uint64),
    ('S_rexmit_packets', np.uint32),
    ('S_rexmit_bytes', np.uint64),
    ('S_out_seq_packets', np.uint32),
    ('S_SYN_count', np.uint32),
    ('S_FIN_count', np.uint32),
    ('S_RFC1323_ws', np.uint8),
    ('S_RFC1323_ts', np.uint8),
    ('S_window_scale', np.uint8),
    ('S_SACK_req', np.uint8),
    ('S_SACK_sent', np.uint32),
    ('S_MSS', np.uint16),
    ('S_max_seg_size', np.uint16),
    ('S_min_seg_size', np.uint16),
    ('S_win_max', np.uint32),
    ('S_win_min', np.uint32),
    ('S_win_zero', np.uint32),
    ('S_cwin_max', np.uint32),
    ('S_cwin_min', np.uint32),
    ('S_initial_cwin', np.uint32),
    ('S_Average_rtt', np.float_),
    ('S_rtt_min', np.float_),
    ('S_rtt_max', np.float_),
    ('S_Stdev_rtt', np.float_),
    ('S_rtt_count', np.float_),
    ('S_ttl_min', np.float_),
    ('S_ttl_max', np.float_),
    ('S_rtx_RTO', np.uint32),
    ('S_rtx_FR', np.uint32),
    ('S_reordering', np.uint32),
    ('S_net_dup', np.uint32),
    ('S_unknown', np.uint32),
    ('S_flow_control', np.uint32),
    ('S_unnece_rtx_RTO', np.uint32),
    ('S_unnece_rtx_FR', np.uint32),
    ('S_diff_SYN_seqno', np.uint8),
    ('Completion_time', np.float_),
    ('First_time', np.float_),
    ('Last_time', np.float_),
    ('C_first_payload', np.float_),
    ('S_first_payload', np.float_),
    ('C_last_payload', np.float_),
    ('S_last_payload', np.float_),
    ('Internal', np.uint8),
    ('Connection_type', np.uint32),
    ('P2P_type', np.uint8),
    ('P2P_subtype', np.uint8),
    ('ED2K_Data', np.uint32),
    ('ED2K_Signaling', np.uint32),
    ('ED2K_C2S', np.uint32),
    ('ED2K_C2C', np.uint32),
    ('ED2K_Chat', np.uint32),
    ('HTTP_type', np.uint32)]

# converters index for tstat substract 1 because index starts at 0
CLIENT_IP_ADDRESS = 1 - 1
CLIENT_TCP_PORT = 2 - 1
C_PACKETS = 3 - 1
C_RST_SENT = 4 - 1
C_ACK_SENT = 5 - 1
C_PURE_ACK_SENT = 6 - 1
C_UNIQUE_BYTES = 7 - 1
C_DATA_PACKETS = 8 - 1
C_DATA_BYTES = 9 - 1
C_REXMIT_PACKETS = 10 - 1
C_REXMIT_BYTES = 11 - 1
C_OUT_SEQ_PACKETS = 12 - 1
C_SYN_COUNT = 13 - 1
C_FIN_COUNT = 14 - 1
C_RFC1323_WS = 15 - 1
C_RFC1323_TS = 16 - 1
C_WINDOW_SCALE = 17 - 1
C_SACK_REQ = 18 - 1
C_SACK_SENT = 19 - 1
C_MSS = 20 - 1
C_MAX_SEG_SIZE = 21 - 1
C_MIN_SEG_SIZE = 22 - 1
C_WIN_MAX = 23 - 1
C_WIN_MIN = 24 - 1
C_WIN_ZERO = 25 - 1
C_CWIN_MAX = 26 - 1
C_CWIN_MIN = 27 - 1
C_INITIAL_CWIN = 28 - 1
C_AVERAGE_RTT = 29 - 1
C_RTT_MIN = 30 - 1
C_RTT_MAX = 31 - 1
C_STDEV_RTT = 32 - 1
C_RTT_COUNT = 33 - 1
C_TTL_MIN = 34 - 1
C_TTL_MAX = 35 - 1
C_RTX_RTO = 36 - 1
C_RTX_FR = 37 - 1
C_REORDERING = 38 - 1
C_NET_DUP = 39 - 1
C_UNKNOWN = 40 - 1
C_FLOW_CONTROL = 41 - 1
C_UNNECE_RTX_RTO = 42 - 1
C_UNNECE_RTX_FR = 43 - 1
C_DIFF_SYN_SEQNO = 44 - 1
SERVER_IP_ADDRESS = 45 - 1
SERVER_TCP_PORT = 46 - 1
S_PACKETS = 47 - 1
S_RST_SENT = 48 - 1
S_ACK_SENT = 49 - 1
S_PURE_ACK_SENT = 50 - 1
S_UNIQUE_BYTES = 51 - 1
S_DATA_PACKETS = 52 - 1
S_DATA_BYTES = 53 - 1
S_REXMIT_PACKETS = 54 - 1
S_REXMIT_BYTES = 55 - 1
S_OUT_SEQ_PACKETS = 56 - 1
S_SYN_COUNT = 57 - 1
S_FIN_COUNT = 58 - 1
S_RFC1323_WS = 59 - 1
S_RFC1323_TS = 60 - 1
S_WINDOW_SCALE = 61 - 1
S_SACK_REQ = 62 - 1
S_SACK_SENT = 63 - 1
S_MSS = 64 - 1
S_MAX_SEG_SIZE = 65 - 1
S_MIN_SEG_SIZE = 66 - 1
S_WIN_MAX = 67 - 1
S_WIN_MIN = 68 - 1
S_WIN_ZERO = 69 - 1
S_CWIN_MAX = 70 - 1
S_CWIN_MIN = 71 - 1
S_INITIAL_CWIN = 72 - 1
S_AVERAGE_RTT = 73 - 1
S_RTT_MIN = 74 - 1
S_RTT_MAX = 75 - 1
S_STDEV_RTT = 76 - 1
S_RTT_COUNT = 77 - 1
S_TTL_MIN = 78 - 1
S_TTL_MAX = 79 - 1
S_RTX_RTO = 80 - 1
S_RTX_FR = 81 - 1
S_REORDERING = 82 - 1
S_NET_DUP = 83 - 1
S_UNKNOWN = 84 - 1
S_FLOW_CONTROL = 85 - 1
S_UNNECE_RTX_RTO = 86 - 1
S_UNNECE_RTX_FR = 87 - 1
S_DIFF_SYN_SEQNO = 88 - 1
COMPLETION_TIME = 89 - 1
FIRST_TIME = 90 - 1
LAST_TIME = 91 - 1
C_FIRST_PAYLOAD = 92 - 1
S_FIRST_PAYLOAD = 93 - 1
C_LAST_PAYLOAD = 94 - 1
S_LAST_PAYLOAD = 95 - 1
INTERNAL = 96 - 1
CONNECTION_TYPE = 97 - 1
P2P_TYPE = 98 - 1
P2P_SUBTYPE = 99 - 1
ED2K_DATA = 100 - 1
ED2K_SIGNALING = 101 - 1
ED2K_C2S = 102 - 1
ED2K_C2C = 103 - 1
ED2K_CHAT = 104 - 1
HTTP_TYPE = 105 - 1

converters_tstat = {
#    ('Client_IP_address', (np.str_, 16)),
    CLIENT_TCP_PORT: lambda s: np.uint16(s or 0),
    C_PACKETS: lambda s: np.uint32(s or 0),
    C_RST_SENT: lambda s: np.uint32(s or 0),
    C_ACK_SENT: lambda s: np.uint32(s or 0),
    C_PURE_ACK_SENT: lambda s: np.uint32(s or 0),
    C_UNIQUE_BYTES: lambda s: np.uint64(s or 0),
    C_DATA_PACKETS: lambda s: np.uint32(s or 0),
    C_DATA_BYTES: lambda s: np.uint64(s or 0),
    C_REXMIT_PACKETS: lambda s: np.uint32(s or 0),
    C_REXMIT_BYTES: lambda s: np.uint64(s or 0),
    C_OUT_SEQ_PACKETS: lambda s: np.uint32(s or 0),
    C_SYN_COUNT: lambda s: np.uint32(s or 0),
    C_FIN_COUNT: lambda s: np.uint32(s or 0),
    C_RFC1323_WS: lambda s: np.uint8(s or 0),
    C_RFC1323_TS: lambda s: np.uint8(s or 0),
    C_WINDOW_SCALE: lambda s: np.uint8(s or 0),
    C_SACK_REQ: lambda s: np.uint8(s or 0),
    C_SACK_SENT: lambda s: np.uint32(s or 0),
    C_MSS: lambda s: np.uint16(s or 0),
    C_MAX_SEG_SIZE: lambda s: np.uint16(s or 0),
    C_MIN_SEG_SIZE: lambda s: np.uint16(s or 0),
    C_WIN_MAX: lambda s: np.uint32(s or 0),
    C_WIN_MIN: lambda s: np.uint32(s or 0),
    C_WIN_ZERO: lambda s: np.uint32(s or 0),
    C_CWIN_MAX: lambda s: np.uint32(s or 0),
    C_CWIN_MIN: lambda s: np.uint32(s or 0),
    C_INITIAL_CWIN: lambda s: np.uint32(s or 0),
    C_AVERAGE_RTT: lambda s: np.float_(s or 0),
    C_RTT_MIN: lambda s: np.float_(s or 0),
    C_RTT_MAX: lambda s: np.float_(s or 0),
    C_STDEV_RTT: lambda s: np.float_(s or 0),
    C_RTT_COUNT: lambda s: np.float_(s or 0),
    C_TTL_MIN: lambda s: np.float_(s or 0),
    C_TTL_MAX: lambda s: np.float_(s or 0),
    C_RTX_RTO: lambda s: np.uint32(s or 0),
    C_RTX_FR: lambda s: np.uint32(s or 0),
    C_REORDERING: lambda s: np.uint32(s or 0),
    C_NET_DUP: lambda s: np.uint32(s or 0),
    C_UNKNOWN: lambda s: np.uint32(s or 0),
    C_FLOW_CONTROL: lambda s: np.uint32(s or 0),
    C_UNNECE_RTX_RTO: lambda s: np.uint32(s or 0),
    C_UNNECE_RTX_FR: lambda s: np.uint32(s or 0),
    C_DIFF_SYN_SEQNO: lambda s: np.uint8(s or 0),
#    ('Server_IP_address', (np.str_, 16)),
    SERVER_TCP_PORT: lambda s: np.uint16(s or 0),
    S_PACKETS: lambda s: np.uint32(s or 0),
    S_RST_SENT: lambda s: np.uint32(s or 0),
    S_ACK_SENT: lambda s: np.uint32(s or 0),
    S_PURE_ACK_SENT: lambda s: np.uint32(s or 0),
    S_UNIQUE_BYTES: lambda s: np.uint64(s or 0),
    S_DATA_PACKETS: lambda s: np.uint32(s or 0),
    S_DATA_BYTES: lambda s: np.uint64(s or 0),
    S_REXMIT_PACKETS: lambda s: np.uint32(s or 0),
    S_REXMIT_BYTES: lambda s: np.uint64(s or 0),
    S_OUT_SEQ_PACKETS: lambda s: np.uint32(s or 0),
    S_SYN_COUNT: lambda s: np.uint32(s or 0),
    S_FIN_COUNT: lambda s: np.uint32(s or 0),
    S_RFC1323_WS: lambda s: np.uint8(s or 0),
    S_RFC1323_TS: lambda s: np.uint8(s or 0),
    S_WINDOW_SCALE: lambda s: np.uint8(s or 0),
    S_SACK_REQ: lambda s: np.uint8(s or 0),
    S_SACK_SENT: lambda s: np.uint32(s or 0),
    S_MSS: lambda s: np.uint16(s or 0),
    S_MAX_SEG_SIZE: lambda s: np.uint16(s or 0),
    S_MIN_SEG_SIZE: lambda s: np.uint16(s or 0),
    S_WIN_MAX: lambda s: np.uint32(s or 0),
    S_WIN_MIN: lambda s: np.uint32(s or 0),
    S_WIN_ZERO: lambda s: np.uint32(s or 0),
    S_CWIN_MAX: lambda s: np.uint32(s or 0),
    S_CWIN_MIN: lambda s: np.uint32(s or 0),
    S_INITIAL_CWIN: lambda s: np.uint32(s or 0),
    S_AVERAGE_RTT: lambda s: np.float_(s or 0),
    S_RTT_MIN: lambda s: np.float_(s or 0),
    S_RTT_MAX: lambda s: np.float_(s or 0),
    S_STDEV_RTT: lambda s: np.float_(s or 0),
    S_RTT_COUNT: lambda s: np.float_(s or 0),
    S_TTL_MIN: lambda s: np.float_(s or 0),
    S_TTL_MAX: lambda s: np.float_(s or 0),
    S_RTX_RTO: lambda s: np.uint32(s or 0),
    S_RTX_FR: lambda s: np.uint32(s or 0),
    S_REORDERING: lambda s: np.uint32(s or 0),
    S_NET_DUP: lambda s: np.uint32(s or 0),
    S_UNKNOWN: lambda s: np.uint32(s or 0),
    S_FLOW_CONTROL: lambda s: np.uint32(s or 0),
    S_UNNECE_RTX_RTO: lambda s: np.uint32(s or 0),
    S_UNNECE_RTX_FR: lambda s: np.uint32(s or 0),
    S_DIFF_SYN_SEQNO: lambda s: np.uint8(s or 0),
    COMPLETION_TIME: lambda s: np.float_(s or 0),
    FIRST_TIME: lambda s: np.float_(s or 0),
    LAST_TIME: lambda s: np.float_(s or 0),
    C_FIRST_PAYLOAD: lambda s: np.float_(s or 0),
    S_FIRST_PAYLOAD: lambda s: np.float_(s or 0),
    C_LAST_PAYLOAD: lambda s: np.float_(s or 0),
    S_LAST_PAYLOAD: lambda s: np.float_(s or 0),
    INTERNAL: lambda s: np.uint8(s or 0),
    CONNECTION_TYPE: lambda s: np.uint32(s or 0),
    P2P_TYPE: lambda s: np.uint8(s or 0),
    P2P_SUBTYPE: lambda s: np.uint8(s or 0),
    ED2K_DATA: lambda s: np.uint32(s or 0),
    ED2K_SIGNALING: lambda s: np.uint32(s or 0),
    ED2K_C2S: lambda s: np.uint32(s or 0),
    ED2K_C2C: lambda s: np.uint32(s or 0),
    ED2K_CHAT: lambda s: np.uint32(s or 0),
    HTTP_TYPE: lambda s: np.uint32(s or 0)
}

# for tstat-2.2
dtype_tstat2 = [
    ('Client_IP_address', (np.str_, 16)),
    ('Client_TCP_port', np.uint16),
    ('C_packets', np.uint32),
    ('C_RST_sent', np.uint32),
    ('C_ACK_sent', np.uint32),
    ('C_PURE_ACK_sent', np.uint32),
    ('C_unique_bytes', np.uint64),
    ('C_data_packets', np.uint32),
    ('C_data_bytes', np.uint64),
    ('C_rexmit_packets', np.uint32),
    ('C_rexmit_bytes', np.uint64),
    ('C_out_seq_packets', np.uint32),
    ('C_SYN_count', np.uint32),
    ('C_FIN_count', np.uint32),
    ('C_RFC1323_ws', np.uint8),
    ('C_RFC1323_ts', np.uint8),
    ('C_window_scale', np.uint8),
    ('C_SACK_req', np.uint8),
    ('C_SACK_sent', np.uint32),
    ('C_MSS', np.uint16),
    ('C_max_seg_size', np.uint16),
    ('C_min_seg_size', np.uint16),
    ('C_win_max', np.uint32),
    ('C_win_min', np.uint32),
    ('C_win_zero', np.uint32),
    ('C_cwin_max', np.uint32),
    ('C_cwin_min', np.uint32),
    ('C_initial_cwin', np.uint32),
    ('C_Average_rtt', np.float_),
    ('C_rtt_min', np.float_),
    ('C_rtt_max', np.float_),
    ('C_Stdev_rtt', np.float_),
    ('C_rtt_count', np.float_),
    ('C_ttl_min', np.float_),
    ('C_ttl_max', np.float_),
    ('C_rtx_RTO', np.uint32),
    ('C_rtx_FR', np.uint32),
    ('C_reordering', np.uint32),
    ('C_net_dup', np.uint32),
    ('C_unknown', np.uint32),
    ('C_flow_control', np.uint32),
    ('C_unnece_rtx_RTO', np.uint32),
    ('C_unnece_rtx_FR', np.uint32),
    ('C_diff_SYN_seqno', np.uint8),
    ('Server_IP_address', (np.str_, 16)),
    ('Server_TCP_port', np.uint16),
    ('S_packets', np.uint32),
    ('S_RST_sent', np.uint32),
    ('S_ACK_sent', np.uint32),
    ('S_PURE_ACK_sent', np.uint32),
    ('S_unique_bytes', np.uint64),
    ('S_data_packets', np.uint32),
    ('S_data_bytes', np.uint64),
    ('S_rexmit_packets', np.uint32),
    ('S_rexmit_bytes', np.uint64),
    ('S_out_seq_packets', np.uint32),
    ('S_SYN_count', np.uint32),
    ('S_FIN_count', np.uint32),
    ('S_RFC1323_ws', np.uint8),
    ('S_RFC1323_ts', np.uint8),
    ('S_window_scale', np.uint8),
    ('S_SACK_req', np.uint8),
    ('S_SACK_sent', np.uint32),
    ('S_MSS', np.uint16),
    ('S_max_seg_size', np.uint16),
    ('S_min_seg_size', np.uint16),
    ('S_win_max', np.uint32),
    ('S_win_min', np.uint32),
    ('S_win_zero', np.uint32),
    ('S_cwin_max', np.uint32),
    ('S_cwin_min', np.uint32),
    ('S_initial_cwin', np.uint32),
    ('S_Average_rtt', np.float_),
    ('S_rtt_min', np.float_),
    ('S_rtt_max', np.float_),
    ('S_Stdev_rtt', np.float_),
    ('S_rtt_count', np.float_),
    ('S_ttl_min', np.float_),
    ('S_ttl_max', np.float_),
    ('S_rtx_RTO', np.uint32),
    ('S_rtx_FR', np.uint32),
    ('S_reordering', np.uint32),
    ('S_net_dup', np.uint32),
    ('S_unknown', np.uint32),
    ('S_flow_control', np.uint32),
    ('S_unnece_rtx_RTO', np.uint32),
    ('S_unnece_rtx_FR', np.uint32),
    ('S_diff_SYN_seqno', np.uint8),
    ('Completion_time', np.float_),
    ('First_time', np.float_),
    ('Last_time', np.float_),
    ('C_first_payload', np.float_),
    ('S_first_payload', np.float_),
    ('C_last_payload', np.float_),
    ('S_last_payload', np.float_),
    ('First_time_abs', np.float_),
    ('Internal', np.uint8),
    ('Connection_type', np.uint32),
    ('P2P_type', np.uint8),
    ('P2P_subtype', np.uint8),
    ('ED2K_Data', np.uint32),
    ('ED2K_Signaling', np.uint32),
    ('ED2K_C2S', np.uint32),
    ('ED2K_C2C', np.uint32),
    ('ED2K_Chat', np.uint32),
    ('HTTP_type', np.uint32)]

# converters index for tstat substract 1 because index starts at 0
CLIENT_IP_ADDRESS2 = 1 - 1
CLIENT_TCP_PORT2 = 2 - 1
C_PACKETS2 = 3 - 1
C_RST_SENT2 = 4 - 1
C_ACK_SENT2 = 5 - 1
C_PURE_ACK_SENT2 = 6 - 1
C_UNIQUE_BYTES2 = 7 - 1
C_DATA_PACKETS2 = 8 - 1
C_DATA_BYTES2 = 9 - 1
C_REXMIT_PACKETS2 = 10 - 1
C_REXMIT_BYTES2 = 11 - 1
C_OUT_SEQ_PACKETS2 = 12 - 1
C_SYN_COUNT2 = 13 - 1
C_FIN_COUNT2 = 14 - 1
C_RFC1323_WS2 = 15 - 1
C_RFC1323_TS2 = 16 - 1
C_WINDOW_SCALE2 = 17 - 1
C_SACK_REQ2 = 18 - 1
C_SACK_SENT2 = 19 - 1
C_MSS2 = 20 - 1
C_MAX_SEG_SIZE2 = 21 - 1
C_MIN_SEG_SIZE2 = 22 - 1
C_WIN_MAX2 = 23 - 1
C_WIN_MIN2 = 24 - 1
C_WIN_ZERO2 = 25 - 1
C_CWIN_MAX2 = 26 - 1
C_CWIN_MIN2 = 27 - 1
C_INITIAL_CWIN2 = 28 - 1
C_AVERAGE_RTT2 = 29 - 1
C_RTT_MIN2 = 30 - 1
C_RTT_MAX2 = 31 - 1
C_STDEV_RTT2 = 32 - 1
C_RTT_COUNT2 = 33 - 1
C_TTL_MIN2 = 34 - 1
C_TTL_MAX2 = 35 - 1
C_RTX_RTO2 = 36 - 1
C_RTX_FR2 = 37 - 1
C_REORDERING2 = 38 - 1
C_NET_DUP2 = 39 - 1
C_UNKNOWN2 = 40 - 1
C_FLOW_CONTROL2 = 41 - 1
C_UNNECE_RTX_RTO2 = 42 - 1
C_UNNECE_RTX_FR2 = 43 - 1
C_DIFF_SYN_SEQNO2 = 44 - 1
SERVER_IP_ADDRESS2 = 45 - 1
SERVER_TCP_PORT2 = 46 - 1
S_PACKETS2 = 47 - 1
S_RST_SENT2 = 48 - 1
S_ACK_SENT2 = 49 - 1
S_PURE_ACK_SENT2 = 50 - 1
S_UNIQUE_BYTES2 = 51 - 1
S_DATA_PACKETS2 = 52 - 1
S_DATA_BYTES2 = 53 - 1
S_REXMIT_PACKETS2 = 54 - 1
S_REXMIT_BYTES2 = 55 - 1
S_OUT_SEQ_PACKETS2 = 56 - 1
S_SYN_COUNT2 = 57 - 1
S_FIN_COUNT2 = 58 - 1
S_RFC1323_WS2 = 59 - 1
S_RFC1323_TS2 = 60 - 1
S_WINDOW_SCALE2 = 61 - 1
S_SACK_REQ2 = 62 - 1
S_SACK_SENT2 = 63 - 1
S_MSS2 = 64 - 1
S_MAX_SEG_SIZE2 = 65 - 1
S_MIN_SEG_SIZE2 = 66 - 1
S_WIN_MAX2 = 67 - 1
S_WIN_MIN2 = 68 - 1
S_WIN_ZERO2 = 69 - 1
S_CWIN_MAX2 = 70 - 1
S_CWIN_MIN2 = 71 - 1
S_INITIAL_CWIN2 = 72 - 1
S_AVERAGE_RTT2 = 73 - 1
S_RTT_MIN2 = 74 - 1
S_RTT_MAX2 = 75 - 1
S_STDEV_RTT2 = 76 - 1
S_RTT_COUNT2 = 77 - 1
S_TTL_MIN2 = 78 - 1
S_TTL_MAX2 = 79 - 1
S_RTX_RTO2 = 80 - 1
S_RTX_FR2 = 81 - 1
S_REORDERING2 = 82 - 1
S_NET_DUP2 = 83 - 1
S_UNKNOWN2 = 84 - 1
S_FLOW_CONTROL2 = 85 - 1
S_UNNECE_RTX_RTO2 = 86 - 1
S_UNNECE_RTX_FR2 = 87 - 1
S_DIFF_SYN_SEQNO2 = 88 - 1
COMPLETION_TIME2 = 89 - 1
FIRST_TIME2 = 90 - 1
LAST_TIME2 = 91 - 1
C_FIRST_PAYLOAD2 = 92 - 1
S_FIRST_PAYLOAD2 = 93 - 1
C_LAST_PAYLOAD2 = 94 - 1
S_LAST_PAYLOAD2 = 95 - 1
FIRST_ABS2 = 96 - 1
INTERNAL2 = 97 - 1
CONNECTION_TYPE2 = 98 - 1
P2P_TYPE2 = 99 - 1
P2P_SUBTYPE2 = 100 - 1
ED2K_DATA2 = 101 - 1
ED2K_SIGNALING2 = 102 - 1
ED2K_C2S2 = 103 - 1
ED2K_C2C2 = 104 - 1
ED2K_CHAT2 = 105 - 1
HTTP_TYPE2 = 106 - 1

converters_tstat2 = {
#    ('Client_IP_address', (np.str_, 16)),
    CLIENT_TCP_PORT2: lambda s: np.uint16(s or 0),
    C_PACKETS2: lambda s: np.uint32(s or 0),
    C_RST_SENT2: lambda s: np.uint32(s or 0),
    C_ACK_SENT2: lambda s: np.uint32(s or 0),
    C_PURE_ACK_SENT2: lambda s: np.uint32(s or 0),
    C_UNIQUE_BYTES2: lambda s: np.uint64(s or 0),
    C_DATA_PACKETS2: lambda s: np.uint32(s or 0),
    C_DATA_BYTES2: lambda s: np.uint64(s or 0),
    C_REXMIT_PACKETS2: lambda s: np.uint32(s or 0),
    C_REXMIT_BYTES2: lambda s: np.uint64(s or 0),
    C_OUT_SEQ_PACKETS2: lambda s: np.uint32(s or 0),
    C_SYN_COUNT2: lambda s: np.uint32(s or 0),
    C_FIN_COUNT2: lambda s: np.uint32(s or 0),
    C_RFC1323_WS2: lambda s: np.uint8(s or 0),
    C_RFC1323_TS2: lambda s: np.uint8(s or 0),
    C_WINDOW_SCALE2: lambda s: np.uint8(s or 0),
    C_SACK_REQ2: lambda s: np.uint8(s or 0),
    C_SACK_SENT2: lambda s: np.uint32(s or 0),
    C_MSS2: lambda s: np.uint16(s or 0),
    C_MAX_SEG_SIZE2: lambda s: np.uint16(s or 0),
    C_MIN_SEG_SIZE2: lambda s: np.uint16(s or 0),
    C_WIN_MAX2: lambda s: np.uint32(s or 0),
    C_WIN_MIN2: lambda s: np.uint32(s or 0),
    C_WIN_ZERO2: lambda s: np.uint32(s or 0),
    C_CWIN_MAX2: lambda s: np.uint32(s or 0),
    C_CWIN_MIN2: lambda s: np.uint32(s or 0),
    C_INITIAL_CWIN2: lambda s: np.uint32(s or 0),
    C_AVERAGE_RTT2: lambda s: np.float_(s or 0),
    C_RTT_MIN2: lambda s: np.float_(s or 0),
    C_RTT_MAX2: lambda s: np.float_(s or 0),
    C_STDEV_RTT2: lambda s: np.float_(s or 0),
    C_RTT_COUNT2: lambda s: np.float_(s or 0),
    C_TTL_MIN2: lambda s: np.float_(s or 0),
    C_TTL_MAX2: lambda s: np.float_(s or 0),
    C_RTX_RTO2: lambda s: np.uint32(s or 0),
    C_RTX_FR2: lambda s: np.uint32(s or 0),
    C_REORDERING2: lambda s: np.uint32(s or 0),
    C_NET_DUP2: lambda s: np.uint32(s or 0),
    C_UNKNOWN2: lambda s: np.uint32(s or 0),
    C_FLOW_CONTROL2: lambda s: np.uint32(s or 0),
    C_UNNECE_RTX_RTO2: lambda s: np.uint32(s or 0),
    C_UNNECE_RTX_FR2: lambda s: np.uint32(s or 0),
    C_DIFF_SYN_SEQNO2: lambda s: np.uint8(s or 0),
#    ('Server_IP_address', (np.str_, 16)),
    SERVER_TCP_PORT2: lambda s: np.uint16(s or 0),
    S_PACKETS2: lambda s: np.uint32(s or 0),
    S_RST_SENT2: lambda s: np.uint32(s or 0),
    S_ACK_SENT2: lambda s: np.uint32(s or 0),
    S_PURE_ACK_SENT2: lambda s: np.uint32(s or 0),
    S_UNIQUE_BYTES2: lambda s: np.uint64(s or 0),
    S_DATA_PACKETS2: lambda s: np.uint32(s or 0),
    S_DATA_BYTES2: lambda s: np.uint64(s or 0),
    S_REXMIT_PACKETS2: lambda s: np.uint32(s or 0),
    S_REXMIT_BYTES2: lambda s: np.uint64(s or 0),
    S_OUT_SEQ_PACKETS2: lambda s: np.uint32(s or 0),
    S_SYN_COUNT2: lambda s: np.uint32(s or 0),
    S_FIN_COUNT2: lambda s: np.uint32(s or 0),
    S_RFC1323_WS2: lambda s: np.uint8(s or 0),
    S_RFC1323_TS2: lambda s: np.uint8(s or 0),
    S_WINDOW_SCALE2: lambda s: np.uint8(s or 0),
    S_SACK_REQ2: lambda s: np.uint8(s or 0),
    S_SACK_SENT2: lambda s: np.uint32(s or 0),
    S_MSS2: lambda s: np.uint16(s or 0),
    S_MAX_SEG_SIZE2: lambda s: np.uint16(s or 0),
    S_MIN_SEG_SIZE2: lambda s: np.uint16(s or 0),
    S_WIN_MAX2: lambda s: np.uint32(s or 0),
    S_WIN_MIN2: lambda s: np.uint32(s or 0),
    S_WIN_ZERO2: lambda s: np.uint32(s or 0),
    S_CWIN_MAX2: lambda s: np.uint32(s or 0),
    S_CWIN_MIN2: lambda s: np.uint32(s or 0),
    S_INITIAL_CWIN2: lambda s: np.uint32(s or 0),
    S_AVERAGE_RTT2: lambda s: np.float_(s or 0),
    S_RTT_MIN2: lambda s: np.float_(s or 0),
    S_RTT_MAX2: lambda s: np.float_(s or 0),
    S_STDEV_RTT2: lambda s: np.float_(s or 0),
    S_RTT_COUNT2: lambda s: np.float_(s or 0),
    S_TTL_MIN2: lambda s: np.float_(s or 0),
    S_TTL_MAX2: lambda s: np.float_(s or 0),
    S_RTX_RTO2: lambda s: np.uint32(s or 0),
    S_RTX_FR2: lambda s: np.uint32(s or 0),
    S_REORDERING2: lambda s: np.uint32(s or 0),
    S_NET_DUP2: lambda s: np.uint32(s or 0),
    S_UNKNOWN2: lambda s: np.uint32(s or 0),
    S_FLOW_CONTROL2: lambda s: np.uint32(s or 0),
    S_UNNECE_RTX_RTO2: lambda s: np.uint32(s or 0),
    S_UNNECE_RTX_FR2: lambda s: np.uint32(s or 0),
    S_DIFF_SYN_SEQNO2: lambda s: np.uint8(s or 0),
    COMPLETION_TIME2: lambda s: np.float_(s or 0),
    FIRST_TIME2: lambda s: np.float_(s or 0),
    LAST_TIME2: lambda s: np.float_(s or 0),
    C_FIRST_PAYLOAD2: lambda s: np.float_(s or 0),
    S_FIRST_PAYLOAD2: lambda s: np.float_(s or 0),
    C_LAST_PAYLOAD2: lambda s: np.float_(s or 0),
    S_LAST_PAYLOAD2: lambda s: np.float_(s or 0),
    FIRST_ABS2: lambda s: np.float_(s or 0),
    INTERNAL2: lambda s: np.uint8(s or 0),
    CONNECTION_TYPE2: lambda s: np.uint32(s or 0),
    P2P_TYPE2: lambda s: np.uint8(s or 0),
    P2P_SUBTYPE2: lambda s: np.uint8(s or 0),
    ED2K_DATA2: lambda s: np.uint32(s or 0),
    ED2K_SIGNALING2: lambda s: np.uint32(s or 0),
    ED2K_C2S2: lambda s: np.uint32(s or 0),
    ED2K_C2C2: lambda s: np.uint32(s or 0),
    ED2K_CHAT2: lambda s: np.uint32(s or 0),
    HTTP_TYPE2: lambda s: np.uint32(s or 0)
}


dtype_cnx_stream = [('Name1', (np.str_, 40)),
                    ('Name2', (np.str_, 40)),
                    ('Name', (np.str_, 80)),
                    ('LocAS', (np.str_, 7)),
                    ('LocAddr', (np.str_, 16)),
                    ('LocPort', np.uint16),
                    ('Protocol', (np.str_, 5)),
                    ('Application', (np.str_, 32)),
                    ('RemName', (np.str_, 180)),
                    ('Host_Referer', (np.str_, 90)),
                    ('RemCountry', (np.str_, 7)),
                    ('RemAS', (np.str_, 7)),
                    ('RemAddr', (np.str_, 16)),
                    ('RemPort', np.uint16),
                    ('Date', (np.str_, 10)),
                    ('StartTime', np.uint32), #(np.str_, 8)),
                    ('EndTime', np.uint32), #(np.str_, 8)),
                    ('Duration', np.float_),
                    ('nByte', np.uint64),
                    ('StartDn', np.uint32), #(np.str_, 8)),
                    ('DurationDn', np.float_),
                    ('ByteDn', np.uint64),
                    ('PacketDn', np.uint32),
                    ('Type', (np.str_, 25)),
                    ('Service', (np.str_, 25))]

dtype_cnx_stream_loss = copy(dtype_cnx_stream)
dtype_cnx_stream_loss.extend([
    ('LostUp', np.uint32),
    ('DesyncUp', np.uint32),
    ('LostDn', np.uint32),
    ('DesyncDn', np.uint32),
    ('WiFi', np.uint8)])

dtype_stream_indics_tmp = [
    # from cnx_stream
    ('Name', (np.str_, 80)),
    ('LocPort', np.uint16),
    ('RemPort', np.uint16),
    ('Date', (np.str_, 10)), # date of flow
    ('StartTime', np.uint32), # start of flow
    ('Host_Referer', (np.str_, 90)), # url
    ('RemName', (np.str_, 180)), # filename or cgi path
    ('StartDn', np.uint32), # start of download stream
    ('DurationDn', np.float_), # duration of download stream
    ('ByteDn', np.uint64), # vol of download stream
    ('PacketDn', np.uint32), # pkts of download stream
    ('Type', (np.str_, 25)), # clip, video...
    ('LostDn', np.uint32), # losses of download stream
    ('DesyncDn', np.uint32), # desync of download stream
    ('Application', (np.str_, 32)), # video rate
    ('Service', (np.str_, 25)), # type of video (heuristic)
    ('loss_ok', np.bool_), # indicate if loss info is reliable
    # from str_stats
    ('srcAddr', (np.str_, 16)),
    ('dstAddr', (np.str_, 16)),
    ('initTime', np.float_),
    ('Content-Type', (np.str_, 24)),
    ('Content-Length', np.uint32),
    ('Content-Duration', np.float_),
    ('Content-Avg-Bitrate-kbps', np.float_), #attention aux inf
    ('Session-Bytes', np.uint64),
    ('Session-Pkts', np.uint32),
    ('Session-Duration', np.float_),
    ('nb_skips', np.uint16),
    ('valid', (np.str_, 5)),
    ('asBGP', np.uint16)
]

dtype_gvb_stream_indics = copy(dtype_stream_indics_tmp)
dtype_gvb_stream_indics.extend([
    # from GVB stats
    ('protocol', np.ubyte),
    ('srcPort', np.uint16),
    ('dstPort', np.uint16),
    ('initTime_flow', np.float_),
    ('direction', np.ubyte),
    ('client_id', np.uint16),
    ('dscp', np.ushort),
    ('peakRate', np.uint32)
])

dtype_rtt_stream_indics = copy(dtype_gvb_stream_indics)
dtype_rtt_stream_indics.extend([
    # from dipcp
    ('DIP-RTT-NbMes-ms-TCP-Up', np.float_),
    ('DIP-RTT-NbMes-ms-TCP-Down', np.float_),
    ('DIP-RTT-Mean-ms-TCP-Up', np.float_),
    ('DIP-RTT-Mean-ms-TCP-Down', np.float_),
    ('DIP-RTT-Min-ms-TCP-Up', np.float_),
    ('DIP-RTT-Min-ms-TCP-Down', np.float_),
    ('DIP-RTT-Max-ms-TCP-Up', np.float_),
    ('DIP-RTT-Max-ms-TCP-Down', np.float_)
])

dtype_all_stream_indics_reorder = [
    # from cnx_stream
    ('Name', (np.str_, 80)),
    ('LocPort', np.uint16),
    ('RemPort', np.uint16),
    ('Date', (np.str_, 10)), # date of flow
    ('StartTime', np.uint32), # start of flow
    ('Host_Referer', (np.str_, 90)), # url
    ('RemName', (np.str_, 180)), # filename or cgi path
    ('StartDn', np.uint32), # start of download stream
    ('DurationDn', np.float_), # duration of download stream
    ('ByteDn', np.uint64), # vol of download stream
    ('PacketDn', np.uint32), # pkts of download stream
    ('Type', (np.str_, 25)), # clip, video...
    ('LostDn', np.uint32), # losses of download stream
    ('DesyncDn', np.uint32), # desync of download stream
    ('Application', (np.str_, 32)), # video rate
    ('Service', (np.str_, 25)), # type of video (heuristic)
    ('loss_ok', np.bool_), # indicate if loss info is reliable
    # from str_stats
    ('srcAddr', (np.str_, 16)), # server IP
    ('dstAddr', (np.str_, 16)), # customer IP
    ('initTime', np.float_),
    ('Content-Type', (np.str_, 24)),
    ('Content-Length', np.uint32),
    ('Content-Duration', np.float_),
    ('Content-Avg-Bitrate-kbps', np.float_), #attention aux inf
    ('Session-Bytes', np.uint64),
    ('Session-Pkts', np.uint32),
    ('Session-Duration', np.float_),
    ('nb_skips', np.uint16),
    ('valid', (np.str_, 5)),
    ('asBGP', np.uint16),
    # from GVB stats
    ('protocol', np.ubyte),
    ('srcPort', np.uint16),
    ('dstPort', np.uint16),
    ('initTime_flow', np.float_),
    ('direction', np.ubyte),
    ('client_id', np.uint16),
    ('dscp', np.ushort),
    ('peakRate', np.uint32),
    # from dipcp
    ('DIP-RTT-NbMes-ms-TCP-Up', np.uint32),
    ('DIP-RTT-NbMes-ms-TCP-Down', np.uint32),
    ('DIP-RTT-Mean-ms-TCP-Up', np.float_),
    ('DIP-RTT-Mean-ms-TCP-Down', np.float_),
    ('DIP-RTT-Min-ms-TCP-Up', np.float_),
    ('DIP-RTT-Min-ms-TCP-Down', np.float_),
    ('DIP-RTT-Max-ms-TCP-Up', np.float_),
    ('DIP-RTT-Max-ms-TCP-Down', np.float_),
    ('DIP-Volume-Number-Packets-Down', np.uint32),
    ('DIP-Volume-Number-Packets-Up', np.uint32),
    ('DIP-Volume-Sum-Bytes-Down', np.uint64),
    ('DIP-Volume-Sum-Bytes-Up', np.uint64),
    ('DIP-DSQ-NbMes-sec-TCP-Up', np.uint32),
    ('DIP-DSQ-NbMes-sec-TCP-Down', np.uint32),
    ('DIP-RTM-NbMes-sec-TCP-Up', np.uint32),
    ('DIP-RTM-NbMes-sec-TCP-Down', np.uint32),
    ('DIP-DST-Number-Milliseconds-Up', np.float_),
    ('DIP-DST-Number-Milliseconds-Down', np.float_),
    ('DIP-CLT-Number-Milliseconds-Up', np.float_),
    ('DIP-CLT-Number-Milliseconds-Down', np.float_),
]

dtype_all_stream_indics = [
    # from cnx_stream
    ('Name', (np.str_, 80)),
    ('LocPort', np.uint16),
    ('RemPort', np.uint16),
    ('Date', (np.str_, 10)), # date of flow
    ('StartTime', np.uint32), # start of flow
    ('Host_Referer', (np.str_, 90)), # url
    ('RemName', (np.str_, 180)), # filename or cgi path
    ('StartDn', np.uint32), # start of download stream
    ('DurationDn', np.float_), # duration of download stream
    ('ByteDn', np.uint64), # vol of download stream
    ('PacketDn', np.uint32), # pkts of download stream
    ('Type', (np.str_, 25)), # clip, video...
    ('LostDn', np.uint32), # losses of download stream
    ('DesyncDn', np.uint32), # desync of download stream
    ('Application', (np.str_, 32)), # video rate
    ('Service', (np.str_, 25)), # type of video (heuristic)
    ('loss_ok', np.bool_), # indicate if loss info is reliable
    # from str_stats
    ('srcAddr', (np.str_, 16)), # server IP
    ('dstAddr', (np.str_, 16)), # customer IP
    ('initTime', np.float_),
    ('Content-Type', (np.str_, 24)),
    ('Content-Length', np.uint32),
    ('Content-Duration', np.float_),
    ('Content-Avg-Bitrate-kbps', np.float_), #attention aux inf
    ('Session-Bytes', np.uint64),
    ('Session-Pkts', np.uint32),
    ('Session-Duration', np.float_),
    ('nb_skips', np.uint16),
    ('valid', (np.str_, 5)),
    ('asBGP', np.uint16),
    # from GVB stats
    ('protocol', np.ubyte),
    ('srcPort', np.uint16),
    ('dstPort', np.uint16),
    ('initTime_flow', np.float_),
    ('direction', np.ubyte),
    ('client_id', np.uint16),
    ('dscp', np.ushort),
    ('peakRate', np.uint32),
    # from dipcp
    ('DIP-RTT-NbMes-ms-TCP-Up', np.uint32),
    ('DIP-RTT-NbMes-ms-TCP-Down', np.uint32),
    ('DIP-RTT-Mean-ms-TCP-Up', np.float_),
    ('DIP-RTT-Mean-ms-TCP-Down', np.float_),
    ('DIP-RTT-Min-ms-TCP-Up', np.float_),
    ('DIP-RTT-Min-ms-TCP-Down', np.float_),
    ('DIP-RTT-Max-ms-TCP-Up', np.float_),
    ('DIP-RTT-Max-ms-TCP-Down', np.float_),
    ('DIP-Volume-Number-Packets-Down', np.uint32),
    ('DIP-Volume-Number-Packets-Up', np.uint32),
    ('DIP-Volume-Sum-Bytes-Down', np.uint64),
    ('DIP-Volume-Sum-Bytes-Up', np.uint64),
    ('DIP-DSQ-NbMes-sec-TCP-Up', np.uint32),
    ('DIP-DSQ-NbMes-sec-TCP-Down', np.uint32),
    ('DIP-RTM-NbMes-sec-TCP-Up', np.uint32),
    ('DIP-RTM-NbMes-sec-TCP-Down', np.uint32),
    ('DIP-DST-Number-Milliseconds-Up', np.float_),
    ('DIP-DST-Number-Milliseconds-Down', np.float_),
    ('DIP-CLT-Number-Milliseconds-Up', np.float_),
    ('DIP-CLT-Number-Milliseconds-Down', np.float_),
]

dtype_all_stream_indics_final = [
    # from cnx_stream
    ('Name', (np.str_, 80)),
    ('LocPort', np.uint16),
    ('RemPort', np.uint16),
    ('Date', (np.str_, 10)), # date of flow
    ('StartTime', np.uint32), # start of flow
    ('Host_Referer', (np.str_, 90)), # url
    ('RemName', (np.str_, 180)), # filename or cgi path
    ('StartDn', np.uint32), # start of download stream
    ('DurationDn', np.float_), # duration of download stream
    ('ByteDn', np.uint64), # vol of download stream
    ('PacketDn', np.uint32), # pkts of download stream
    ('Type', (np.str_, 25)), # clip, video...
    ('LostDn', np.uint32), # losses of download stream
    ('DesyncDn', np.uint32), # desync of download stream
    ('Application', (np.str_, 32)), # video rate
    ('Service', (np.str_, 25)), # type of video (heuristic)
    ('loss_ok', np.bool_), # indicate if loss info is reliable
    # from str_stats
    ('srcAddr', (np.str_, 16)), # server IP
    ('dstAddr', (np.str_, 16)), # customer IP
    ('initTime', np.float_),
    ('Content-Type', (np.str_, 24)),
    ('Content-Length', np.uint32),
    ('Content-Duration', np.float_),
    ('Content-Avg-Bitrate-kbps', np.float_), #attention aux inf
    ('Session-Bytes', np.uint64),
    ('Session-Pkts', np.uint32),
    ('Session-Duration', np.float_),
    ('nb_skips', np.uint16),
    ('valid', (np.str_, 5)),
    ('asBGP', np.uint16),
    # from GVB stats
    ('protocol', np.ubyte),
    ('srcPort', np.uint16),
    ('dstPort', np.uint16),
    ('initTime_flow', np.float_),
    ('direction', np.ubyte),
    ('client_id', np.uint16),
    ('dscp', np.ushort),
    ('peakRate', np.uint32),
    # from dipcp
    ('DIP-RTT-DATA-NbMes-ms-TCP-Up', np.uint32),
    ('DIP-RTT-DATA-NbMes-ms-TCP-Down', np.uint32),
    ('DIP-RTT-DATA-Mean-ms-TCP-Up', np.float_),
    ('DIP-RTT-DATA-Mean-ms-TCP-Down', np.float_),
    ('DIP-RTT-DATA-Min-ms-TCP-Up', np.float_),
    ('DIP-RTT-DATA-Min-ms-TCP-Down', np.float_),
    ('DIP-RTT-DATA-Max-ms-TCP-Up', np.float_),
    ('DIP-RTT-DATA-Max-ms-TCP-Down', np.float_),
    ('DIP-Volume-Number-Packets-Down', np.uint32),
    ('DIP-Volume-Number-Packets-Up', np.uint32),
    ('DIP-Volume-Sum-Bytes-Down', np.uint64),
    ('DIP-Volume-Sum-Bytes-Up', np.uint64),
    ('DIP-DSQ-NbMes-sec-TCP-Up', np.uint32),
    ('DIP-DSQ-NbMes-sec-TCP-Down', np.uint32),
    ('DIP-RTM-NbMes-sec-TCP-Up', np.uint32),
    ('DIP-RTM-NbMes-sec-TCP-Down', np.uint32),
    ('DIP-DST-Number-Milliseconds-Up', np.float_),
    ('DIP-DST-Number-Milliseconds-Down', np.float_),
    ('DIP-CLT-Number-Milliseconds-Up', np.float_),
    ('DIP-CLT-Number-Milliseconds-Down', np.float_),
    ('DIP-RTT-NbMes-ms-TCP-Up', np.uint32),
    ('DIP-RTT-NbMes-ms-TCP-Down', np.uint32),
    ('DIP-RTT-Mean-ms-TCP-Up', np.float_),
    ('DIP-RTT-Mean-ms-TCP-Down', np.float_),
    ('DIP-RTT-Min-ms-TCP-Up', np.float_),
    ('DIP-RTT-Min-ms-TCP-Down', np.float_),
    ('DIP-RTT-Max-ms-TCP-Up', np.float_),
    ('DIP-RTT-Max-ms-TCP-Down', np.float_),
]

dtype_all_stream_indics_final_good = copy(dtype_all_stream_indics_final)
dtype_all_stream_indics_final_good.extend([
    ('good', np.bool_)
])

dtype_all_stream_indics_final_tstat = copy(dtype_all_stream_indics_final_good)
dtype_all_stream_indics_final_tstat.extend(dtype_tstat)

dtype_all_stream_indics_rename = [
    # from cnx_stream
    ('Name', (np.str_, 80)),
    ('LocPort', np.uint16),
    ('RemPort', np.uint16),
    ('Date', (np.str_, 10)), # date of flow
    ('StartTime', np.uint32), # start of flow
    ('Host_Referer', (np.str_, 90)), # url
    ('RemName', (np.str_, 180)), # filename or cgi path
    ('StartDn', np.uint32), # start of download stream
    ('DurationDn', np.float_), # duration of download stream
    ('ByteDn', np.uint64), # vol of download stream
    ('PacketDn', np.uint32), # pkts of download stream
    ('Type', (np.str_, 25)), # clip, video...
    ('LostDn', np.uint32), # losses of download stream
    ('DesyncDn', np.uint32), # desync of download stream
    ('Application', (np.str_, 32)), # video rate
    ('Service', (np.str_, 25)), # type of video (heuristic)
    ('loss_ok', np.bool_), # indicate if loss info is reliable
    # from str_stats
    ('srcAddr', (np.str_, 16)), # server IP
    ('dstAddr', (np.str_, 16)), # customer IP
    ('initTime', np.float_),
    ('Content-Type', (np.str_, 24)),
    ('Content-Length', np.uint32),
    ('Content-Duration', np.float_),
    ('Content-Avg-Bitrate-kbps', np.float_), #attention aux inf
    ('Session-Bytes', np.uint64),
    ('Session-Pkts', np.uint32),
    ('Session-Duration', np.float_),
    ('nb_skips', np.uint16),
    ('valid', (np.str_, 5)),
    ('asBGP', np.uint16),
    # from GVB stats
    ('protocol', np.ubyte),
    ('srcPort', np.uint16),
    ('dstPort', np.uint16),
    ('initTime_flow', np.float_),
    ('direction', np.ubyte),
    ('client_id', np.uint16),
    ('dscp', np.ushort),
    ('peakRate', np.uint32),
    # from dipcp
    ('DIP-RTT-DATA-NbMes-ms-TCP-Up', np.uint32),
    ('DIP-RTT-DATA-NbMes-ms-TCP-Down', np.uint32),
    ('DIP-RTT-DATA-Mean-ms-TCP-Up', np.float_),
    ('DIP-RTT-DATA-Mean-ms-TCP-Down', np.float_),
    ('DIP-RTT-DATA-Min-ms-TCP-Up', np.float_),
    ('DIP-RTT-DATA-Min-ms-TCP-Down', np.float_),
    ('DIP-RTT-DATA-Max-ms-TCP-Up', np.float_),
    ('DIP-RTT-DATA-Max-ms-TCP-Down', np.float_),
    ('DIP-Volume-Number-Packets-Down', np.uint32),
    ('DIP-Volume-Number-Packets-Up', np.uint32),
    ('DIP-Volume-Sum-Bytes-Down', np.uint64),
    ('DIP-Volume-Sum-Bytes-Up', np.uint64),
    ('DIP-DSQ-NbMes-sec-TCP-Up', np.uint32),
    ('DIP-DSQ-NbMes-sec-TCP-Down', np.uint32),
    ('DIP-RTM-NbMes-sec-TCP-Up', np.uint32),
    ('DIP-RTM-NbMes-sec-TCP-Down', np.uint32),
    ('DIP-DST-Number-Milliseconds-Up', np.float_),
    ('DIP-DST-Number-Milliseconds-Down', np.float_),
    ('DIP-CLT-Number-Milliseconds-Up', np.float_),
    ('DIP-CLT-Number-Milliseconds-Down', np.float_),
]

dtype_cnx_h323 = ["guid",
                  "Name1",
                  "Name2",
                  "Caller_Name",
                  "Caller_n",
                  "Called_n",
                  "nMediaUp",
                  "nMediaDn",
                  "Status",
                  "Protocol",
                  "App",
                  "StartTime",
                  "LastTime",
                  "MediaTypeUp0",
                  "MediaStartUp0",
                  "MediaDurationUp0",
                  "MediaByteUp0",
                  "MediaPacketUp0",
                  "MediaTypeUp1",
                  "MediaStartUp1",
                  "MediaDurationUp1",
                  "MediaByteUp1",
                  "MediaPacketUp1",
                  "MediaTypeDn0",
                  "MediaStartDn0",
                  "MediaDurationDn0",
                  "MediaByteDn0",
                  "MediaPacketDn0",
                  "MediaTypeDn1",
                  "MediaStartDn1",
                  "MediaDurationDn1",
                  "MediaByteDn1",
                  "MediaPacketDn1",
                  "MediaIPDn",
                  "MediaASDn",
                  "MediaPortDn",
                  "MediaIPUp",
                  "MediaASUp",
                  "MediaPortUp",
                  "CalledName",
                  "CodecNameUp0",
                  "CodecNameUp1",
                  "CodecNameDn0",
                  "CodecNameDn1",
                  "PacketLostUp",
                  "PacketOOOUp",
                  "JitterUp",
                  "PacketLostDn",
                  "PacketOOODn",
                  "JitterDn",
                  "ProbLossUp0",
                  "ProbLossUp1",
                  "ProbLossDn0",
                  "ProbLossDn1",
                  "RAT"
                  ]
# converters index for cnx_stream
NAME1 = 0
NAME2 = 1
NAME = 2
LOCAS = 3
LOCADDR = 4
LOCPORT = 5
PROTOCOL = 6
APPLICATION = 7
REMNAME = 8
HOST_REFERER = 9
REMCOUNTRY = 10
REMAS = 11
REMADDR = 12
REMPORT = 13
DATE = 14
STARTTIME = 15
ENDTIME = 16
DURATION = 17
NBYTE = 18
STARTDN = 19
DURATIONDN = 20
BYTEDN = 21
PACKETDN = 22
TYPE = 23
SERVICE = 24
LOSTUP = 25
DESYNCUP = 26
LOSTDN = 27
DESYNCDN = 28
WIFI = 29

TIME_MATCHER = re.compile("\d{2}:\d{2}:\d{2}")
def time2nbsec(s):
    "Return the number of seconds since start of day"
    assert TIME_MATCHER.match(s), "incorrect time to format %s" % s
    return foldl(lambda t, acc: acc + 60 * t, 0, map(int, s.split(':')))

converters_cnx_stream = {
#    NAME1: lambda s: np.uint8(s or 0),
#    NAME2: lambda s: np.uint8(s or 0),
#    ('Name', np.str),
#    ('LocAS', np.str),
#    ('LocAddr', (np.str_, 16)),
    LOCPORT: lambda s: np.uint16(s or 0),
#    ('Protocol', (np.str_, 5)),
#    ('Application', np.str),
#    ('RemName', np.str),
#    ('Host_Referer', np.str),
#    ('RemCountry', (np.str_, 7)),
#    ('RemAS', (np.str_, 7)),
#    ('RemAddr', (np.str_, 16)),
    REMPORT: lambda s: np.uint16(s or 0),
#    ('Date', (np.str_, 10)),
    STARTTIME: lambda s: time2nbsec(s),
    ENDTIME: lambda s: time2nbsec(s),
    DURATION: lambda s: np.float_(s or 0),
    NBYTE: lambda s: np.uint64(s or 0),
    STARTDN: lambda s: time2nbsec(s),
    DURATIONDN: lambda s: np.float_(s or 0),
    BYTEDN: lambda s: np.uint64(s or 0),
    PACKETDN: lambda s: np.uint32(s or 0),
#    ('Type', np.str),
#    ('Service', np.str),
}

converters_cnx_stream_loss = {
#    NAME1: lambda s: np.uint8(s or 0),
#    NAME2: lambda s: np.uint8(s or 0),
#    ('Name', np.str),
#    ('LocAS', np.str),
#    ('LocAddr', (np.str_, 16)),
    LOCPORT: lambda s: np.uint16(s or 0),
#    ('Protocol', (np.str_, 5)),
#    ('Application', np.str),
#    ('RemName', np.str),
#    ('Host_Referer', np.str),
#    ('RemCountry', (np.str_, 7)),
#    ('RemAS', (np.str_, 7)),
#    ('RemAddr', (np.str_, 16)),
    REMPORT: lambda s: np.uint16(s or 0),
#    ('Date', (np.str_, 10)),
    STARTTIME: lambda s: time2nbsec(s),
    ENDTIME: lambda s: time2nbsec(s),
    DURATION: lambda s: np.float_(s or 0),
    NBYTE: lambda s: np.uint64(s or 0),
    STARTDN: lambda s: time2nbsec(s),
    DURATIONDN: lambda s: np.float_(s or 0),
    BYTEDN: lambda s: np.uint64(s or 0),
    PACKETDN: lambda s: np.uint32(s or 0),
#    ('Type', np.str),
#    ('Service', np.str),
    LOSTUP: lambda s: np.uint32(s or 0),
    DESYNCUP: lambda s: np.uint32(s or 0),
    LOSTDN: lambda s: np.uint32(s or 0),
    DESYNCDN: lambda s: np.uint32(s or 0),
    WIFI: lambda s: np.uint8(s or 0)}

#GVB file type with BGP info
dtype_GVB_BGP = [('protocol', np.ubyte),
             ('client_id', np.uint16),
             ('direction', np.ubyte),
             ('srcAddr', (np.str_, 16)),
             ('srcPort', np.uint16),
             ('dstAddr', (np.str_, 16)),
             ('dstPort', np.uint16),
             ('initTime', np.float_),
             ('nbPkt', np.uint32),
             ('l2Bytes', np.uint64),
             ('duration', np.float_),
             ('l3Bytes', np.uint64),
             ('minHopCount', np.ushort),
             ('maxHopCount', np.ushort),
             ('peakRate', np.uint32),
             ('dscp', np.ushort),
             ('flags', np.ushort),
             ('asBGP', np.uint16)]

#GVB file type with AS info
dtype_GVB_AS_down = [('protocol', np.ubyte),
             ('client_id', np.uint16),
             ('direction', np.ubyte),
             ('srcAddr', (np.str_, 16)),
             ('srcPort', np.uint16),
             ('dstAddr', (np.str_, 16)),
             ('dstPort', np.uint16),
             ('initTime', np.float_),
             ('nbPkt', np.uint32),
             ('l2Bytes', np.uint64),
             ('duration', np.float_),
             ('l3Bytes', np.uint64),
             ('minHopCount', np.ushort),
             ('maxHopCount', np.ushort),
             ('peakRate', np.uint32),
             ('dscp', np.ushort),
             ('flags', np.ushort),
             ('asSrc', np.uint16),
             ('orgSrc', (np.str_, 32))]

#GVB file type with AS
dtype_GVB_AS = [('protocol', np.ubyte),
             ('client_id', np.uint16),
             ('direction', np.ubyte),
             ('srcAddr', (np.str_, 16)),
             ('srcPort', np.uint16),
             ('dstAddr', (np.str_, 16)),
             ('dstPort', np.uint16),
             ('initTime', np.float_),
             ('nbPkt', np.uint32),
             ('l2Bytes', np.uint64),
             ('duration', np.float_),
             ('l3Bytes', np.uint64),
             ('minHopCount', np.ushort),
             ('maxHopCount', np.ushort),
             ('peakRate', np.uint32),
             ('dscp', np.ushort),
             ('flags', np.ushort),
             ('asSrc', np.uint16),
             ('orgSrc', (np.str_, 32)),
             ('asDst', np.uint16),
             ('orgDst', (np.str_, 32))]

#GVB BGP file type with AS
dtype_GVB_BGP_AS = [('protocol', np.ubyte),
             ('client_id', np.uint16),
             ('direction', np.ubyte),
             ('srcAddr', (np.str_, 16)),
             ('srcPort', np.uint16),
             ('dstAddr', (np.str_, 16)),
             ('dstPort', np.uint16),
             ('initTime', np.float_),
             ('nbPkt', np.uint32),
             ('l2Bytes', np.uint64),
             ('duration', np.float_),
             ('l3Bytes', np.uint64),
             ('minHopCount', np.ushort),
             ('maxHopCount', np.ushort),
             ('peakRate', np.uint32),
             ('dscp', np.ushort),
             ('flags', np.ushort),
             ('asBGP', np.uint16),
             ('asSrc', np.uint16),
             ('orgSrc', (np.str_, 32)),
             ('asDst', np.uint16),
             ('orgDst', (np.str_, 32))]

#NOT WORKING: print formatter for GVB file type with AS
#fmt_GVB_AS = ('%s',
#             '%u',
#             '%u',
#             '%s',
#             '%u',
#             '%s',
#             '%u',
#             '%f',
#             '%u',
#             '%u',
#             '%f',
#             '%u',
#             '%u',
#             '%u',
#             '%u',
#             '%u',
#             '%u',
#             '%u',
#             '%s',
#             '%u',
#             's')

#GVB file type
dtype_GVB = [('protocol', np.ubyte),
             ('client_id', np.uint16),
             ('direction', np.ubyte),
             ('srcAddr', (np.str_, 16)),
             ('srcPort', np.uint16),
             ('dstAddr', (np.str_, 16)),
             ('dstPort', np.uint16),
             ('initTime', np.float_),
             ('nbPkt', np.uint32),
             ('l2Bytes', np.uint64),
             ('duration', np.float_),
             ('l3Bytes', np.uint64),
             ('minHopCount', np.ushort),
             ('maxHopCount', np.ushort),
             ('peakRate', np.uint32),
             ('dscp', np.ushort),
             ('flags', np.ushort)]

sep_GVB = ' '

# GVB streaming type
#%src_ip dst_ip Init-Time Content-Type Content-Length Content-Duration
#Content-Avg-Bitrate-kbps Session-Bytes Session-Pkts Session-Duration
#buffer_sizes nb_hangs nb_skips Tracks valid
# beware tracks became hasAudio + hasVideo!
dtype_GVB_streaming = [
    ('srcAddr', (np.str_, 16)),
    ('dstAddr', (np.str_, 16)),
    ('initTime', np.float_),
    ('Content-Type', (np.str_, 24)),
    ('Content-Length', np.uint32),
    ('Content-Duration', np.float_),
    ('Content-Avg-Bitrate-kbps', np.float_), #attention aux inf
    ('Session-Bytes', np.uint64),
    ('Session-Pkts', np.uint32),
    ('Session-Duration', np.float_),
    ('buffer_sizes', (np.uint32, 100)),
    ('nb_hangs', (np.uint16, 100)),
    ('nb_skips', np.uint16),
    ('hasAudio', (np.str_, 1)),
    ('hasVideo', (np.str_, 1)),
    ('valid', (np.str_, 5))]

dtype_GVB_streaming_AS = copy(dtype_GVB_streaming)
dtype_GVB_streaming_AS.append(('asBGP', np.uint16))

# deprecated
dtype_streaming_session_old = [
    ('dstAddr', (np.str_, 16)),
    ('duration', np.float_),
    ('nb_streams', np.uint16),
    ('min_thp', np.float_),
    ('avg_thp', np.float_)
]

# streaming sessions type
dtype_streaming_session = [
    ('dstAddr', (np.str_, 16)),
    ('beg', np.float_),
    ('end', np.float_),
    ('duration', np.float_),
    ('session_bytes', np.uint64),
    ('nb_streams', np.uint16),
    ('min_thp', np.float_),
    ('avg_thp', np.float_)
]

dtype_streaming_session_other = [
    ('dstAddr', (np.str_, 80)), # name in fact
    ('beg', np.float_),
    ('end', np.float_),
    ('duration', np.float_),
    ('session_bytes', np.uint64),
    ('nb_streams', np.uint16)
]

# cnx stream session dtype
dtype_cnx_streaming_session = [
    ('Name', (np.str_, 80)),
    ('beg', np.float_),
    ('end', np.float_),
    ('duration', np.float_),
    ('tot_bytes', np.uint64),
    ('nb_flows', np.uint)
]

# dipcp file type
dtype_dipcp = [('Time', np.float_),
               ('FlowIPSource', (np.str_, 16)),
               ('FlowEthSource', (np.str_, 17)),
               ('FlowPortSource', np.uint16),
               ('FlowIPDest', (np.str_, 16)),
               ('FlowEthDest', (np.str_, 17)),
               ('FlowPortDest', np.uint16),
               ('FirstPacketDate', np.float_),
               ('LastPacketDate', np.float_),
               ('DIP-Duration-Sum-MilliSeconds-Effective', np.float_),
               ('DIP-Volume-Number-Packets-Down', np.uint32),
               ('DIP-Volume-Number-Packets-Up', np.uint32),
               ('DIP-Volume-Sum-Bytes-Down', np.uint64),
               ('DIP-Volume-Sum-Bytes-Up', np.uint64),
               ('DIP-Thp-Number-Kbps-Down', np.float_),
               ('DIP-Thp-Number-Kbps-Up', np.float_),
               ('DIP-Thp-Number-Kbps-Down-Effective', np.float_),
               ('DIP-Thp-Number-Kbps-Up-Effective', np.float_),
               ('LastPacketProtocol', (np.str_, 5)),
               ('PreviousLastTcpPacketType', (np.str_, 1)),
               ('LastTcpPacketType', (np.str_, 1)),
               ('SynCounter-Up', np.uint8),
               ('SynCounter-Down', np.uint8),
               ('Up-1', np.ubyte),
               ('Down-1', np.ubyte),
               ('Push-1', np.ubyte),
               ('Size-1', np.uint16),
               ('ts-1', np.float_),
               ('Up-2', np.ubyte),
               ('Down-2', np.ubyte),
               ('Push-2', np.ubyte),
               ('Size-2', np.uint16),
               ('ts-2', np.float_),
               ('Up-3', np.ubyte),
               ('Down-3', np.ubyte),
               ('Push-3', np.ubyte),
               ('Size-3', np.uint16),
               ('ts-3', np.float_),
               ('Up-4', np.ubyte),
               ('Down-4', np.ubyte),
               ('Push-4', np.ubyte),
               ('Size-4', np.uint16),
               ('ts-4', np.float_),
               ('Up-5', np.ubyte),
               ('Down-5', np.ubyte),
               ('Push-5', np.ubyte),
               ('Size-5', np.uint16),
               ('ts-5', np.float_),
               ('Up-6', np.ubyte),
               ('Down-6', np.ubyte),
               ('Push-6', np.ubyte),
               ('Size-6', np.uint16),
               ('ts-6', np.float_),
               ('Up-7', np.ubyte),
               ('Down-7', np.ubyte),
               ('Push-7', np.ubyte),
               ('Size-7', np.uint16),
               ('ts-7', np.float_),
               ('Up-8', np.ubyte),
               ('Down-8', np.ubyte),
               ('Push-8', np.ubyte),
               ('Size-8', np.uint16),
               ('ts-8', np.float_),
               ('Up-9', np.ubyte),
               ('Down-9', np.ubyte),
               ('Push-9', np.ubyte),
               ('Size-9', np.uint16),
               ('ts-9', np.float_),
               ('TOS', np.ubyte),
               ('LastPacketIPSource', (np.str_, 16)),
               ('LastPacketSize', np.uint16),
               ('DIP-DSQ-NbMes-sec-TCP-Up', np.uint32),
               ('DIP-DSQ-NbMes-sec-TCP-Down', np.uint32),
               ('DIP-RTM-NbMes-sec-TCP-Up', np.uint32),
               ('DIP-RTM-NbMes-sec-TCP-Down', np.uint32),
               ('DIP-DST-Number-Milliseconds-Up', np.float_),
               ('DIP-DST-Number-Milliseconds-Down', np.float_),
               ('DIP-CLT-Number-Milliseconds-Up', np.float_),
               ('DIP-CLT-Number-Milliseconds-Down', np.float_)]

dtype_dipcp_compat = [('Time', np.float_),
                      ('FlowIPSource', (np.str_, 16)),
                      ('FlowEthSource', (np.str_, 17)),
                      ('FlowPortSource', np.uint16),
                      ('FlowIPDest', (np.str_, 16)),
                      ('FlowEthDest', (np.str_, 17)),
                      ('FlowPortDest', np.uint16),
                      ('FirstPacketDate', np.float_),
                      ('LastPacketDate', np.float_),
                      ('DIP-Duration-Sum-MilliSeconds-Effective', np.float_),
                      ('DIP-Volume-Number-Packets-Down', np.uint32),
                      ('DIP-Volume-Number-Packets-Up', np.uint32),
                      ('DIP-Volume-Sum-Bytes-Down', np.uint64),
                      ('DIP-Volume-Sum-Bytes-Up', np.uint64),
                      ('DIP-Thp-Number-Kbps-Down', np.float_),
                      ('DIP-Thp-Number-Kbps-Up', np.float_),
                      ('DIP-Thp-Number-Kbps-Down-Effective', np.float_),
                      ('DIP-Thp-Number-Kbps-Up-Effective', np.float_),
                      ('LastPacketProtocol', (np.str_, 5)),
                      # don't ask why...
                      ('PreviousLastTcpPacketType', (np.str_, 5)),
                      ('LastTcpPacketType', (np.str_, 5)),
                      ('SynCounter-Up', np.uint8),
                      ('SynCounter-Down', np.uint8),
                      ('Up-1', np.ubyte),
                      ('Down-1', np.ubyte),
                      ('Push-1', np.ubyte),
                      ('Size-1', np.uint16),
                      ('ts-1', np.float_),
                      ('Up-2', np.ubyte),
                      ('Down-2', np.ubyte),
                      ('Push-2', np.ubyte),
                      ('Size-2', np.uint16),
                      ('ts-2', np.float_),
                      # why???
                      ('Up-1_TWICE', np.ubyte),
                      ('Down-3', np.ubyte),
                      ('Push-3', np.ubyte),
                      ('Size-3', np.uint16),
                      ('ts-3', np.float_),
                      ('Up-4', np.ubyte),
                      ('Down-4', np.ubyte),
                      ('Push-4', np.ubyte),
                      ('Size-4', np.uint16),
                      ('ts-4', np.float_),
                      ('Up-5', np.ubyte),
                      ('Down-5', np.ubyte),
                      ('Push-5', np.ubyte),
                      ('Size-5', np.uint16),
                      ('ts-5', np.float_),
                      ('Up-6', np.ubyte),
                      ('Down-6', np.ubyte),
                      ('Push-6', np.ubyte),
                      ('Size-6', np.uint16),
                      ('ts-6', np.float_),
                      ('Up-7', np.ubyte),
                      ('Down-7', np.ubyte),
                      ('Push-7', np.ubyte),
                      ('Size-7', np.uint16),
                      ('ts-7', np.float_),
                      ('Up-8', np.ubyte),
                      ('Down-8', np.ubyte),
                      ('Push-8', np.ubyte),
                      ('Size-8', np.uint16),
                      ('ts-8', np.float_),
                      ('Up-9', np.ubyte),
                      ('Down-9', np.ubyte),
                      ('Push-9', np.ubyte),
                      ('Size-9', np.uint16),
                      ('ts-9', np.float_),
                      ('TOS', np.ubyte),
                      ('LastPacketIPSource', (np.str_, 16)),
                      ('LastPacketSize', np.uint16),
                      ('DIP-RTT-NbMes-ms-TCP-Up', np.uint32),
                      ('DIP-RTT-NbMes-ms-TCP-Down', np.uint32),
                      ('DIP-RTT-Mean-ms-TCP-Up', np.float_),
                      ('DIP-RTT-Mean-ms-TCP-Down', np.float_),
                      ('DIP-RTT-Min-ms-TCP-Up', np.float_),
                      ('DIP-RTT-Min-ms-TCP-Down', np.float_),
                      ('DIP-RTT-Max-ms-TCP-Up', np.float_),
                      ('DIP-RTT-Max-ms-TCP-Down', np.float_),
                      ('DIP-DSQ-NbMes-sec-TCP-Up', np.uint32),
                      ('DIP-DSQ-NbMes-sec-TCP-Down', np.uint32),
                      ('DIP-RTM-NbMes-sec-TCP-Up', np.uint32),
                      ('DIP-RTM-NbMes-sec-TCP-Down', np.uint32),
                      ('DIP-DST-Number-Milliseconds-Up', np.float_),
                      ('DIP-DST-Number-Milliseconds-Down', np.float_),
                      ('DIP-CLT-Number-Milliseconds-Up', np.float_),
                      ('DIP-CLT-Number-Milliseconds-Down', np.float_),
                      # what's that?
                      ('f87', (np.str_, 1)),
]

dtype_dipcp_cnx_stream = copy(dtype_dipcp_compat)
dtype_dipcp_cnx_stream.extend(copy(dtype_cnx_stream_loss))
dtype_dipcp_cnx_stream_tstat2 = copy(dtype_dipcp_cnx_stream)
dtype_dipcp_cnx_stream_tstat2.extend(copy(dtype_tstat2))

sep_dipcp = ';'

#\(.*\)(\('[-A-Za-z0-9]+'\), np.\(uint\(?:8|32|64\)\))
#dipcp converter for missing values
#col indexes
TIME = 0
FLOWIPSOURCE = 1
FLOWETHSOURCE = 2
FLOWPORTSOURCE = 3
FLOWIPDEST = 4
FLOWETHDEST = 5
FLOWPORTDEST = 6
FIRSTPACKETDATE = 7
LASTPACKETDATE = 8
DIP_DURATION_SUM_MILLISECONDS_EFFECTIVE = 9
DIP_VOLUME_NUMBER_PACKETS_DOWN = 10
DIP_VOLUME_NUMBER_PACKETS_UP = 11
DIP_VOLUME_SUM_BYTES_DOWN = 12
DIP_VOLUME_SUM_BYTES_UP = 13
DIP_THP_NUMBER_KBPS_DOWN = 14
DIP_THP_NUMBER_KBPS_UP = 15
DIP_THP_NUMBER_KBPS_DOWN_EFFECTIVE = 16
DIP_THP_NUMBER_KBPS_UP_EFFECTIVE = 17
LASTPACKETPROTOCOL = 18
PREVIOUSLASTTCPPACKETTYPE = 19
LASTTCPPACKETTYPE = 20
SYNCOUNTER_UP = 21
SYNCOUNTER_DOWN = 22
UP_1 = 23
DOWN_1 = 24
PUSH_1 = 25
SIZE_1 = 26
TS_1 = 27
UP_2 = 28
DOWN_2 = 29
PUSH_2 = 30
SIZE_2 = 31
TS_2 = 32
UP_3 = 33
DOWN_3 = 34
PUSH_3 = 35
SIZE_3 = 36
TS_3 = 37
UP_4 = 38
DOWN_4 = 39
PUSH_4 = 40
SIZE_4 = 41
TS_4 = 42
UP_5 = 43
DOWN_5 = 44
PUSH_5 = 45
SIZE_5 = 46
TS_5 = 47
UP_6 = 48
DOWN_6 = 49
PUSH_6 = 50
SIZE_6 = 51
TS_6 = 52
UP_7 = 53
DOWN_7 = 54
PUSH_7 = 55
SIZE_7 = 56
TS_7 = 57
UP_8 = 58
DOWN_8 = 59
PUSH_8 = 60
SIZE_8 = 61
TS_8 = 62
UP_9 = 63
DOWN_9 = 64
PUSH_9 = 65
SIZE_9 = 66
TS_9 = 67
TOS = 68
LASTPACKETIPSOURCE = 69
LASTPACKETSIZE = 70
DIP_DSQ_NBMES_SEC_TCP_UP = 71
DIP_DSQ_NBMES_SEC_TCP_DOWN = 72
DIP_RTM_NBMES_SEC_TCP_UP = 73
DIP_RTM_NBMES_SEC_TCP_DOWN = 74
DIP_DST_NUMBER_MILLISECONDS_UP = 75
DIP_DST_NUMBER_MILLISECONDS_DOWN = 76
DIP_CLT_NUMBER_MILLISECONDS_UP = 77
DIP_CLT_NUMBER_MILLISECONDS_DOWN = 78

#Dictionnary for missing values
converters_dipcp = {
    TIME: lambda s: np.float(s or 0),
#               ('FlowIPSource', (np.str_, 16)),
#               ('FlowEthSource', (np.str_, 17)),
               FLOWPORTSOURCE: lambda s: np.uint16(s or 0),
#               ('FlowIPDest', (np.str_, 16)),
#               ('FlowEthDest', (np.str_, 17)),
               FLOWPORTDEST: lambda s: np.uint16(s or 0),
               FIRSTPACKETDATE: lambda s: np.float(s or 0),
               LASTPACKETDATE: lambda s: np.float(s or 0),
               DIP_DURATION_SUM_MILLISECONDS_EFFECTIVE: lambda s: np.float(s or 0),
               DIP_VOLUME_NUMBER_PACKETS_DOWN: lambda s: np.uint32(s or 0),
               DIP_VOLUME_NUMBER_PACKETS_UP: lambda s: np.uint32(s or 0),
               DIP_VOLUME_SUM_BYTES_DOWN: lambda s: np.uint64(s or 0),
               DIP_VOLUME_SUM_BYTES_UP: lambda s: np.uint64(s or 0),
               DIP_THP_NUMBER_KBPS_DOWN: lambda s: np.float(s or 0),
               DIP_THP_NUMBER_KBPS_UP: lambda s: np.float(s or 0),
               DIP_THP_NUMBER_KBPS_DOWN_EFFECTIVE: lambda s: np.float(s or 0),
               DIP_THP_NUMBER_KBPS_UP_EFFECTIVE: lambda s: np.float(s or 0),
#               ('LASTPACKETPROTOCOL', (np.str_, 5)),
#               ('PREVIOUSLASTTCPPACKETTYPE', np.str_),
#               ('LASTTCPPACKETTYPE', np.str_),
               SYNCOUNTER_UP: lambda s: np.uint8(s or 0),
               SYNCOUNTER_DOWN: lambda s: np.uint8(s or 0),
               UP_1: lambda s: np.ubyte(s or 0),
               DOWN_1: lambda s: np.ubyte(s or 0),
               PUSH_1: lambda s: np.ubyte(s or 0),
               SIZE_1: lambda s: np.uint16(s or 0),
               TS_1: lambda s: np.float_(s or 0),
               UP_2: lambda s: np.ubyte(s or 0),
               DOWN_2: lambda s: np.ubyte(s or 0),
               PUSH_2: lambda s: np.ubyte(s or 0),
               SIZE_2: lambda s: np.uint16(s or 0),
               TS_2: lambda s: np.float_(s or 0),
               UP_3: lambda s: np.ubyte(s or 0),
               DOWN_3: lambda s: np.ubyte(s or 0),
               PUSH_3: lambda s: np.ubyte(s or 0),
               SIZE_3: lambda s: np.uint16(s or 0),
               TS_3: lambda s: np.float_(s or 0),
               UP_4: lambda s: np.ubyte(s or 0),
               DOWN_4: lambda s: np.ubyte(s or 0),
               PUSH_4: lambda s: np.ubyte(s or 0),
               SIZE_4: lambda s: np.uint16(s or 0),
               TS_4: lambda s: np.float_(s or 0),
               UP_5: lambda s: np.ubyte(s or 0),
               DOWN_5: lambda s: np.ubyte(s or 0),
               PUSH_5: lambda s: np.ubyte(s or 0),
               SIZE_5: lambda s: np.uint16(s or 0),
               TS_5: lambda s: np.float_(s or 0),
               UP_6: lambda s: np.ubyte(s or 0),
               DOWN_6: lambda s: np.ubyte(s or 0),
               PUSH_6: lambda s: np.ubyte(s or 0),
               SIZE_6: lambda s: np.uint16(s or 0),
               TS_6: lambda s: np.float_(s or 0),
               UP_7: lambda s: np.ubyte(s or 0),
               DOWN_7: lambda s: np.ubyte(s or 0),
               PUSH_7: lambda s: np.ubyte(s or 0),
               SIZE_7: lambda s: np.uint16(s or 0),
               TS_7: lambda s: np.float_(s or 0),
               UP_8: lambda s: np.ubyte(s or 0),
               DOWN_8: lambda s: np.ubyte(s or 0),
               PUSH_8: lambda s: np.ubyte(s or 0),
               SIZE_8: lambda s: np.uint16(s or 0),
               TS_8: lambda s: np.float_(s or 0),
               UP_9: lambda s: np.ubyte(s or 0),
               DOWN_9: lambda s: np.ubyte(s or 0),
               PUSH_9: lambda s: np.ubyte(s or 0),
               SIZE_9: lambda s: np.uint16(s or 0),
               TS_9: lambda s: np.float_(s or 0),
               TOS: lambda s: np.ubyte(s or 0),
#               ('LASTPACKETIPSOURCE', (np.str_, 16)),
               LASTPACKETSIZE: lambda s: np.uint16(s or 0),
               DIP_DSQ_NBMES_SEC_TCP_UP: lambda s: np.uint32(s or 0),
               DIP_DSQ_NBMES_SEC_TCP_DOWN: lambda s: np.uint32(s or 0),
               DIP_RTM_NBMES_SEC_TCP_UP: lambda s: np.uint32(s or 0),
               DIP_RTM_NBMES_SEC_TCP_DOWN: lambda s: np.uint32(s or 0),
               DIP_DST_NUMBER_MILLISECONDS_UP: lambda s: np.float_(s or 0),
               DIP_DST_NUMBER_MILLISECONDS_DOWN: lambda s: np.float_(s or 0),
               DIP_CLT_NUMBER_MILLISECONDS_UP: lambda s: np.float_(s or 0),
               DIP_CLT_NUMBER_MILLISECONDS_DOWN: lambda s: np.float_(s or 0)}

