#!/usr/bin/env python

import numpy as np
import h5py

global_capture_list = ()

def create_h5py(flows, file):
    f = h5py.File(file, 'w')

    for d in ('2008', 'nov_2009', 'dec_2009'):
        for t in ('ADSL', 'FTTH'):
            group = f.create_group('%s_%s' % (t, d))
            if len(flows['flows_%s_%s_stream_down' % (t, d)])>0:
                group.create_dataset('GVB_%s_%s_HTTP_STREAM_DOWN' % (t, d),
                                     data=flows['flows_%s_%s_stream_down'
                                                % (t, d)],
                                     compression='lzf', chunks=True)
            if len(flows['dipcp_%s_%s_down_stream' % (t, d)])>0:
                group.create_dataset('dipcp_%s_%s_HTTP_STREAM_DOWN' % (t, d),
                                     data=flows['dipcp_%s_%s_down_stream'
                                                % (t, d)],
                                     compression='lzf', chunks=True)
            if len(flows['flows_%s_%s_down_yt' % (t, d)])>0:
                group.create_dataset('GVB_%s_%s_YT_STREAM_DOWN' % (t, d),
                                     data=flows['flows_%s_%s_down_yt' % (t, d)],
                                     compression='lzf', chunks=True)
            if len(flows['dipcp_%s_%s_down_yt' % (t, d)])>0:
                group.create_dataset('dipcp_%s_%s_YT_STREAM_DOWN' % (t, d),
                                     data=flows['dipcp_%s_%s_down_yt' % (t, d)],
                                     compression='lzf', chunks=True)
            if len(flows['flows_%s_%s_down_yt_eu' % (t, d)])>0:
                group.create_dataset('GVB_%s_%s_YT_EU_STREAM_DOWN' % (t, d),
                                     data=flows['flows_%s_%s_down_yt_eu'
                                                % (t, d)],
                                     compression='lzf', chunks=True)
            if len(flows['dipcp_%s_%s_down_yt_eu' % (t, d)])>0:
                group.create_dataset('dipcp_%s_%s_YT_EU_STREAM_DOWN' % (t, d),
                                     data=flows['dipcp_%s_%s_down_yt_eu'
                                                % (t, d)],
                                     compression='lzf', chunks=True)

            if len(flows['flows_%s_%s_stream_down_1MB' % (t, d)])>0:
                group.create_dataset('GVB_%s_%s_HTTP_STREAM_DOWN_1MB' % (t, d),
                                     data=flows['flows_%s_%s_stream_down_1MB'
                                                % (t, d)],
                                     compression='lzf', chunks=True)
            if len(flows['dipcp_%s_%s_down_stream_1MB' % (t, d)])>0:
                group.create_dataset('dipcp_%s_%s_HTTP_STREAM_DOWN_1MB'
                                     % (t, d),
                                     data=flows['dipcp_%s_%s_down_stream_1MB'
                                                % (t, d)],
                                     compression='lzf', chunks=True)
            if len(flows['flows_%s_%s_down_yt_1MB' % (t, d)])>0:
                group.create_dataset('GVB_%s_%s_YT_STREAM_DOWN_1MB' % (t, d),
                                     data=flows['flows_%s_%s_down_yt_1MB'
                                                % (t, d)],
                                     compression='lzf', chunks=True)
            if len(flows['dipcp_%s_%s_down_yt_1MB' % (t, d)])>0:
                group.create_dataset('dipcp_%s_%s_YT_STREAM_DOWN_1MB' % (t, d),
                                     data=flows['dipcp_%s_%s_down_yt_1MB'
                                                % (t, d)],
                                     compression='lzf', chunks=True)
            if len(flows['flows_%s_%s_down_yt_eu_1MB' % (t, d)])>0:
                group.create_dataset('GVB_%s_%s_YT_EU_STREAM_DOWN_1MB' % (t, d),
                                     data=flows['flows_%s_%s_down_yt_eu_1MB'
                                                % (t, d)],
                                     compression='lzf', chunks=True)
            if len(flows['dipcp_%s_%s_down_yt_eu_1MB' % (t, d)])>0:
                group.create_dataset('dipcp_%s_%s_YT_EU_STREAM_DOWN_1MB'
                                     % (t, d),
                                     data=flows['dipcp_%s_%s_down_yt_eu_1MB'
                                                % (t, d)],
                                     compression='lzf', chunks=True)

def main():
    f = h5py.File('/media/Data/elpy7391/streaming/hdf5/lzf_data.h5', 'w')

    ADSL_2008 = f.create_group("ADSL_2008")
    gvb_adsl_2008 = np.load('python_flows/flows_marked_GVB_juill_2008_ADSL_cut_BGP_AS.npy')
    ADSL_2008.create_dataset('GVB', data=gvb_adsl_2008,
                             compression='lzf', chunks=True)
    dipcp_adsl_2008 = np.load('python_flows/dipcp_flows_ADSL_juill_2008.npy')
    ADSL_2008.create_dataset('dipcp', data=dipcp_adsl_2008,
                             compression='lzf', chunks=True)
    ADSL_2008.attrs["type"] = "ADSL"
    ADSL_2008.attrs["place"] = "Montsouris"
    ADSL_2008.attrs["date"] = "2008-07-01"
    ADSL_2008.attrs["time"] = "early evening"
    ADSL_2008.attrs["start_time"] = "19:56"
    ADSL_2008.attrs["stop_time"] = "21:19"
    ADSL_2008.attrs["duration"] = "1h24"
    ADSL_2008.attrs["load"] = "medium"
    ADSL_2008.attrs["filter"] = "no"

    FTTH_2008 = f.create_group("FTTH_2008")
    gvb_ftth_2008 = np.load('python_flows/flows_marked_GVB_juill_2008_FTTH_BGP_AS.npy')
    FTTH_2008.create_dataset('GVB', data=gvb_ftth_2008,
                             compression='lzf', chunks=True)
    dipcp_ftth_2008 = np.load('python_flows/dipcp_flows_FTTH_juill_2008_TCP.npy')
    FTTH_2008.create_dataset('dipcp', data=dipcp_ftth_2008,
                             compression='lzf', chunks=True)
    FTTH_2008.attrs["type"] = "FTTH"
    FTTH_2008.attrs["place"] = "Montsouris"
    FTTH_2008.attrs["date"] = "2008-07-01"
    FTTH_2008.attrs["time"] = "early evening"
    FTTH_2008.attrs["start_time"] = "20:00"
    FTTH_2008.attrs["stop_time"] = "21:17"
    FTTH_2008.attrs["duration"] = "1h17"
    FTTH_2008.attrs["load"] = "medium"
    FTTH_2008.attrs["filter"] = "no"

    ADSL_nov_2009 = f.create_group("ADSL_nov_2009")
    gvb_adsl_nov_2009 = np.load('python_flows/flows_marked_GVB_nov_2009_ADSL_BGP_AS.npy')
    ADSL_nov_2009.create_dataset('GVB', data=gvb_adsl_nov_2009,
                                 compression='lzf', chunks=True)
    dipcp_adsl_nov_2009 = np.load('python_flows/dipcp_flows_ADSL_nov_2009.npy')
    ADSL_nov_2009.create_dataset('dipcp', data=dipcp_adsl_nov_2009,
                                 compression='lzf', chunks=True)
    ADSL_nov_2009.attrs["type"] = "ADSL"
    ADSL_nov_2009.attrs["place"] = "Montsouris"
    ADSL_nov_2009.attrs["date"] = "2009-11-26"
    ADSL_nov_2009.attrs["time"] = "early evening"
    ADSL_nov_2009.attrs["start_time"] = "20:00"
    ADSL_nov_2009.attrs["stop_time"] = "21:20"
    ADSL_nov_2009.attrs["duration"] = "1h20"
    ADSL_nov_2009.attrs["load"] = "high"
    ADSL_nov_2009.attrs["filter"] = "web + streaming"

    FTTH_nov_2009 = f.create_group("FTTH_nov_2009")
    gvb_ftth_nov_2009 = np.load('python_flows/flows_marked_GVB_nov_2009_FTTH_BGP_AS.npy')
    FTTH_nov_2009.create_dataset('GVB', data=gvb_ftth_nov_2009,
                                 compression='lzf', chunks=True)
    dipcp_ftth_nov_2009 = np.load('python_flows/dipcp_flows_FTTH_nov_2009.npy')
    FTTH_nov_2009.create_dataset('dipcp', data=dipcp_ftth_nov_2009,
                                 compression='lzf', chunks=True)
    FTTH_nov_2009.attrs["type"] = "FTTH"
    FTTH_nov_2009.attrs["place"] = "Montsouris"
    FTTH_nov_2009.attrs["date"] = "2009-11-26"
    FTTH_nov_2009.attrs["time"] = "early evening"
    FTTH_nov_2009.attrs["start_time"] = "20:00"
    FTTH_nov_2009.attrs["stop_time"] = "20:38"
    FTTH_nov_2009.attrs["duration"] = "0h38"
    FTTH_nov_2009.attrs["load"] = "high: capture loss!"
    FTTH_nov_2009.attrs["filter"] = "web + streaming"

    ADSL_dec_2009 = f.create_group("ADSL_dec_2009")
    gvb_adsl_dec_2009 = np.load('python_flows/flows_marked_GVB_dec_2009_ADSL_BGP_AS.npy')
    ADSL_dec_2009.create_dataset('GVB', data=gvb_adsl_dec_2009,
                                 compression='lzf', chunks=True)
    dipcp_adsl_dec_2009 = np.load('python_flows/dipcp_flows_ADSL_dec_2009.npy')
    ADSL_dec_2009.create_dataset('dipcp', data=dipcp_adsl_dec_2009,
                                 compression='lzf', chunks=True)
    ADSL_dec_2009.attrs["type"] = "ADSL"
    ADSL_dec_2009.attrs["place"] = "Rennes"
    ADSL_dec_2009.attrs["date"] = "2009-12-14"
    ADSL_dec_2009.attrs["time"] = "early evening"
    ADSL_dec_2009.attrs["start_time"] = "20:00"
    ADSL_dec_2009.attrs["stop_time"] = "21:00"
    ADSL_dec_2009.attrs["duration"] = "1h00"
    ADSL_dec_2009.attrs["load"] = "very high"
    ADSL_dec_2009.attrs["filter"] = "web + streaming"

    FTTH_dec_2009 = f.create_group("FTTH_dec_2009")
    gvb_ftth_dec_2009 = np.load('python_flows/flows_marked_GVB_dec_2009_FTTH_BGP_AS.npy')
    FTTH_dec_2009.create_dataset('GVB', data=gvb_ftth_dec_2009,
                                 compression='lzf', chunks=True)
    dipcp_ftth_dec_2009 = np.load('python_flows/dipcp_flows_FTTH_dec_2009.npy')
    FTTH_dec_2009.create_dataset('dipcp', data=dipcp_ftth_dec_2009,
                                 compression='lzf', chunks=True)
    FTTH_nov_2009.attrs["type"] = "FTTH"
    FTTH_nov_2009.attrs["place"] = "Montsouris"
    FTTH_nov_2009.attrs["date"] = "2009-11-26"
    FTTH_nov_2009.attrs["time"] = "early afternoon"
    FTTH_nov_2009.attrs["start_time"] = "14:11"
    FTTH_nov_2009.attrs["stop_time"] = "15:00"
    FTTH_nov_2009.attrs["duration"] = "0h49"
    FTTH_nov_2009.attrs["load"] = "medium"
    FTTH_nov_2009.attrs["filter"] = "web + streaming"


if __name__ == '__main__':
    main()

