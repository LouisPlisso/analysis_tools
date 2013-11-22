#!/usr/bin/env python
"Module to extract a resume of a preloaded GVB flow file\
filtered on HTTP streaming down."


#from optparse import OptionParser
import numpy as np

#import INDEX_VALUES
#import aggregate
import compute_AT

def fetch_data_http_stream_down(
    flows_ADSL_2008_stream_down_yt,
    flows_FTTH_2008_stream_down_yt,
    flows_ADSL_2009_stream_down_yt,
    flows_FTTH_2009_stream_down_yt,
    flows_ADSL_2008_stream_down_other,
    flows_FTTH_2008_stream_down_other,
    flows_ADSL_2009_stream_down_other,
    flows_FTTH_2009_stream_down_other):
    "Return a resume of interesting HTTP streaming \
    down flows characteristics."
    resume = {}

    #ADSL 2008: YT
    resume['nb_cl_adsl_2008_yt'] = len(np.unique(
        flows_ADSL_2008_stream_down_yt.client_id))
    resume['nb_fl_adsl_2008_yt'] = len(flows_ADSL_2008_stream_down_yt)
    flows_ADSL_2008_stream_down_yt_1MB = flows_ADSL_2008_stream_down_yt.compress(
        flows_ADSL_2008_stream_down_yt.l3Bytes > 10**6 )
    resume['nb_cl_adsl_2008_yt_1MB'] = len(np.unique(
        flows_ADSL_2008_stream_down_yt_1MB.client_id))
    resume['nb_fl_adsl_2008_yt_1MB'] = len(flows_ADSL_2008_stream_down_yt_1MB)
    resume['mean_fl_size_adsl_2008_yt'] = np.mean(
        flows_ADSL_2008_stream_down_yt.l3Bytes)
    resume['median_fl_size_adsl_2008_yt'] = np.median(
        flows_ADSL_2008_stream_down_yt.l3Bytes)
    resume['max_fl_size_adsl_2008_yt'] = max(
        flows_ADSL_2008_stream_down_yt.l3Bytes)
    resume['mean_fl_dur_adsl_2008_yt'] = np.mean(
        flows_ADSL_2008_stream_down_yt.duration)
    resume['median_fl_dur_adsl_2008_yt'] = np.median(
        flows_ADSL_2008_stream_down_yt.duration)
    resume['max_fl_dur_adsl_2008_yt'] = max(
        flows_ADSL_2008_stream_down_yt.duration)
    resume['mean_fl_peak_adsl_2008_yt'] = np.mean(
        80 * flows_ADSL_2008_stream_down_yt.peakRate)
    resume['median_fl_peak_adsl_2008_yt'] = np.median(
        80 * flows_ADSL_2008_stream_down_yt.peakRate)
    resume['max_fl_peak_adsl_2008_yt'] = max(
        80 * flows_ADSL_2008_stream_down_yt.peakRate)
    meanRate_adsl_2008_yt = [8*x['l3Bytes']/(1000.0*x['duration'])
                             for x in flows_ADSL_2008_stream_down_yt
                             if x['duration']>0]
    resume['mean_fl_meanRate_adsl_2008_yt'] = np.mean(meanRate_adsl_2008_yt)
    resume['median_fl_meanRate_adsl_2008_yt'] = np.median(meanRate_adsl_2008_yt)
    resume['max_fl_meanRate_adsl_2008_yt'] = max(meanRate_adsl_2008_yt)
    meanRate_adsl_2008_yt_1MB = [8*x['l3Bytes']/(1000.0*x['duration'])
                                 for x in flows_ADSL_2008_stream_down_yt
                                 if x['duration']>0
                                 and x['l3Bytes'] > 10**6]
    resume['mean_fl_meanRate_adsl_2008_yt_1MB'] = np.mean(
        meanRate_adsl_2008_yt_1MB)
    resume['median_fl_meanRate_adsl_2008_yt_1MB'] = np.median(
        meanRate_adsl_2008_yt_1MB)
    resume['max_fl_meanRate_adsl_2008_yt_1MB'] = max(meanRate_adsl_2008_yt_1MB)
    resume['mean_fl_AR_adsl_2008_yt'] = compute_AT.compute_AT(
        flows_ADSL_2008_stream_down_yt.initTime)[0]
    resume['mean_fl_100AR_per_cl_adsl_2008_yt'] \
        = 100 * resume['mean_fl_AR_adsl_2008_yt']/resume['nb_cl_adsl_2008_yt']
    #ADSL 2008: other
    resume['nb_cl_adsl_2008_other'] = len(
        np.unique(flows_ADSL_2008_stream_down_other.client_id))
    resume['nb_fl_adsl_2008_other'] = len(flows_ADSL_2008_stream_down_other)
    flows_ADSL_2008_stream_down_other_1MB = flows_ADSL_2008_stream_down_other.compress(
        flows_ADSL_2008_stream_down_other.l3Bytes > 10**6 )
    resume['nb_cl_adsl_2008_other_1MB'] = len(np.unique(
        flows_ADSL_2008_stream_down_other_1MB.client_id))
    resume['nb_fl_adsl_2008_other_1MB'] = len(
        flows_ADSL_2008_stream_down_other_1MB)
    resume['mean_fl_size_adsl_2008_other'] = np.mean(
        flows_ADSL_2008_stream_down_other.l3Bytes)
    resume['median_fl_size_adsl_2008_other'] = np.median(
        flows_ADSL_2008_stream_down_other.l3Bytes)
    resume['max_fl_size_adsl_2008_other'] = max(
        flows_ADSL_2008_stream_down_other.l3Bytes)
    resume['mean_fl_dur_adsl_2008_other'] = np.mean(
        flows_ADSL_2008_stream_down_other.duration)
    resume['median_fl_dur_adsl_2008_other'] = np.median(
        flows_ADSL_2008_stream_down_other.duration)
    resume['max_fl_dur_adsl_2008_other'] = max(
        flows_ADSL_2008_stream_down_other.duration)
    resume['mean_fl_peak_adsl_2008_other'] = np.mean(
        80 * flows_ADSL_2008_stream_down_other.peakRate)
    resume['median_fl_peak_adsl_2008_other'] = np.median(
        80 * flows_ADSL_2008_stream_down_other.peakRate)
    resume['max_fl_peak_adsl_2008_other'] = max(
        80 * flows_ADSL_2008_stream_down_other.peakRate)
    meanRate_adsl_2008_other = [8*x['l3Bytes']/(1000.0*x['duration'])
                             for x in flows_ADSL_2008_stream_down_other
                                if x['duration']>0]
    resume['mean_fl_meanRate_adsl_2008_other'] = np.mean(
        meanRate_adsl_2008_other)
    resume['median_fl_meanRate_adsl_2008_other'] = np.median(
        meanRate_adsl_2008_other)
    resume['max_fl_meanRate_adsl_2008_other'] = max(meanRate_adsl_2008_other)
    meanRate_adsl_2008_other_1MB = [8*x['l3Bytes']/(1000.0*x['duration'])
                                 for x in flows_ADSL_2008_stream_down_other
                                    if x['duration']>0
                                 and x['l3Bytes'] > 10**6]
    resume['mean_fl_meanRate_adsl_2008_other_1MB'] = np.mean(
        meanRate_adsl_2008_other_1MB)
    resume['median_fl_meanRate_adsl_2008_other_1MB'] = np.median(
        meanRate_adsl_2008_other_1MB)
    resume['max_fl_meanRate_adsl_2008_other_1MB'] = max(
        meanRate_adsl_2008_other_1MB)
    resume['mean_fl_AR_adsl_2008_other'] = compute_AT.compute_AT(
        flows_ADSL_2008_stream_down_other.initTime)[0]
    resume['mean_fl_100AR_per_cl_adsl_2008_other'] \
        = 100 * resume['mean_fl_AR_adsl_2008_other'] / resume['nb_cl_adsl_2008_other']


    #FTTH 2008: YT
    resume['nb_cl_ftth_2008_yt'] = len(np.unique(
        flows_FTTH_2008_stream_down_yt.client_id))
    resume['nb_fl_ftth_2008_yt'] = len(flows_FTTH_2008_stream_down_yt)
    flows_FTTH_2008_stream_down_yt_1MB = flows_FTTH_2008_stream_down_yt.compress(
        flows_FTTH_2008_stream_down_yt.l3Bytes > 10**6 )
    resume['nb_cl_ftth_2008_yt_1MB'] = len(np.unique(
        flows_FTTH_2008_stream_down_yt_1MB.client_id))
    resume['nb_fl_ftth_2008_yt_1MB'] = len(flows_FTTH_2008_stream_down_yt_1MB)
    resume['mean_fl_size_ftth_2008_yt'] = np.mean(
        flows_FTTH_2008_stream_down_yt.l3Bytes)
    resume['median_fl_size_ftth_2008_yt'] = np.median(
        flows_FTTH_2008_stream_down_yt.l3Bytes)
    resume['max_fl_size_ftth_2008_yt'] = max(
        flows_FTTH_2008_stream_down_yt.l3Bytes)
    resume['mean_fl_dur_ftth_2008_yt'] = np.mean(
        flows_FTTH_2008_stream_down_yt.duration)
    resume['median_fl_dur_ftth_2008_yt'] = np.median(
        flows_FTTH_2008_stream_down_yt.duration)
    resume['max_fl_dur_ftth_2008_yt'] = max(
        flows_FTTH_2008_stream_down_yt.duration)
    resume['mean_fl_peak_ftth_2008_yt'] = np.mean(
        80 * flows_FTTH_2008_stream_down_yt.peakRate)
    resume['median_fl_peak_ftth_2008_yt'] = np.median(
        80 * flows_FTTH_2008_stream_down_yt.peakRate)
    resume['max_fl_peak_ftth_2008_yt'] = max(
        80 * flows_FTTH_2008_stream_down_yt.peakRate)
    meanRate_ftth_2008_yt = [8*x['l3Bytes']/(1000.0*x['duration'])
                             for x in flows_FTTH_2008_stream_down_yt
                             if x['duration']>0]
    resume['mean_fl_meanRate_ftth_2008_yt'] = np.mean(meanRate_ftth_2008_yt)
    resume['median_fl_meanRate_ftth_2008_yt'] = np.median(meanRate_ftth_2008_yt)
    resume['max_fl_meanRate_ftth_2008_yt'] = max(meanRate_ftth_2008_yt)
    meanRate_ftth_2008_yt_1MB = [8*x['l3Bytes']/(1000.0*x['duration'])
                                 for x in flows_FTTH_2008_stream_down_yt
                                 if x['duration']>0
                                 and x['l3Bytes'] > 10**6]
    resume['mean_fl_meanRate_ftth_2008_yt_1MB'] = np.mean(
        meanRate_ftth_2008_yt_1MB)
    resume['median_fl_meanRate_ftth_2008_yt_1MB'] = np.median(
        meanRate_ftth_2008_yt_1MB)
    resume['max_fl_meanRate_ftth_2008_yt_1MB'] = max(meanRate_ftth_2008_yt_1MB)
    resume['mean_fl_AR_ftth_2008_yt'] = compute_AT.compute_AT(
        flows_FTTH_2008_stream_down_yt.initTime)[0]
    resume['mean_fl_100AR_per_cl_ftth_2008_yt'] \
        = 100 * resume['mean_fl_AR_ftth_2008_yt']/resume['nb_cl_ftth_2008_yt']
    #FTTH 2008: other
    resume['nb_cl_ftth_2008_other'] = len(np.unique(
        flows_FTTH_2008_stream_down_other.client_id))
    resume['nb_fl_ftth_2008_other'] = len(flows_FTTH_2008_stream_down_other)
    flows_FTTH_2008_stream_down_other_1MB = flows_FTTH_2008_stream_down_other.compress(
        flows_FTTH_2008_stream_down_other.l3Bytes > 10**6 )
    resume['nb_cl_ftth_2008_other_1MB'] = len(np.unique(
        flows_FTTH_2008_stream_down_other_1MB.client_id))
    resume['nb_fl_ftth_2008_other_1MB'] = len(
        flows_FTTH_2008_stream_down_other_1MB)
    resume['mean_fl_size_ftth_2008_other'] = np.mean(
        flows_FTTH_2008_stream_down_other.l3Bytes)
    resume['median_fl_size_ftth_2008_other'] = np.median(
        flows_FTTH_2008_stream_down_other.l3Bytes)
    resume['max_fl_size_ftth_2008_other'] = max(
        flows_FTTH_2008_stream_down_other.l3Bytes)
    resume['mean_fl_dur_ftth_2008_other'] = np.mean(
        flows_FTTH_2008_stream_down_other.duration)
    resume['median_fl_dur_ftth_2008_other'] = np.median(
        flows_FTTH_2008_stream_down_other.duration)
    resume['max_fl_dur_ftth_2008_other'] = max(
        flows_FTTH_2008_stream_down_other.duration)
    resume['mean_fl_peak_ftth_2008_other'] = np.mean(
        80 * flows_FTTH_2008_stream_down_other.peakRate)
    resume['median_fl_peak_ftth_2008_other'] = np.median(
        80 * flows_FTTH_2008_stream_down_other.peakRate)
    resume['max_fl_peak_ftth_2008_other'] = max(
        80 * flows_FTTH_2008_stream_down_other.peakRate)
    meanRate_ftth_2008_other = [8*x['l3Bytes']/(1000.0*x['duration'])
                             for x in flows_FTTH_2008_stream_down_other
                                if x['duration']>0]
    resume['mean_fl_meanRate_ftth_2008_other'] = np.mean(
        meanRate_ftth_2008_other)
    resume['median_fl_meanRate_ftth_2008_other'] = np.median(
        meanRate_ftth_2008_other)
    resume['max_fl_meanRate_ftth_2008_other'] = max(meanRate_ftth_2008_other)
    meanRate_ftth_2008_other_1MB = [8*x['l3Bytes']/(1000.0*x['duration'])
                                 for x in flows_FTTH_2008_stream_down_other
                                    if x['duration']>0
                                 and x['l3Bytes'] > 10**6]
    resume['mean_fl_meanRate_ftth_2008_other_1MB'] = np.mean(
        meanRate_ftth_2008_other_1MB)
    resume['median_fl_meanRate_ftth_2008_other_1MB'] = np.median(
        meanRate_ftth_2008_other_1MB)
    resume['max_fl_meanRate_ftth_2008_other_1MB'] = max(
        meanRate_ftth_2008_other_1MB)
    resume['mean_fl_AR_ftth_2008_other'] = compute_AT.compute_AT(
        flows_FTTH_2008_stream_down_other.initTime)[0]
    resume['mean_fl_100AR_per_cl_ftth_2008_other'] \
        = 100 * resume['mean_fl_AR_ftth_2008_other'] / resume['nb_cl_ftth_2008_other']


    #ADSL 2009: YT
    resume['nb_cl_adsl_2009_yt'] = len(np.unique(
        flows_ADSL_2009_stream_down_yt.client_id))
    resume['nb_fl_adsl_2009_yt'] = len(flows_ADSL_2009_stream_down_yt)
    flows_ADSL_2009_stream_down_yt_1MB = flows_ADSL_2009_stream_down_yt.compress(
        flows_ADSL_2009_stream_down_yt.l3Bytes > 10**6 )
    resume['nb_cl_adsl_2009_yt_1MB'] = len(np.unique(
        flows_ADSL_2009_stream_down_yt_1MB.client_id))
    resume['nb_fl_adsl_2009_yt_1MB'] = len(flows_ADSL_2009_stream_down_yt_1MB)
    resume['mean_fl_size_adsl_2009_yt'] = np.mean(
        flows_ADSL_2009_stream_down_yt.l3Bytes)
    resume['median_fl_size_adsl_2009_yt'] = np.median(
        flows_ADSL_2009_stream_down_yt.l3Bytes)
    resume['max_fl_size_adsl_2009_yt'] = max(
        flows_ADSL_2009_stream_down_yt.l3Bytes)
    resume['mean_fl_dur_adsl_2009_yt'] = np.mean(
        flows_ADSL_2009_stream_down_yt.duration)
    resume['median_fl_dur_adsl_2009_yt'] = np.median(
        flows_ADSL_2009_stream_down_yt.duration)
    resume['max_fl_dur_adsl_2009_yt'] = max(
        flows_ADSL_2009_stream_down_yt.duration)
    resume['mean_fl_peak_adsl_2009_yt'] = np.mean(
        80 * flows_ADSL_2009_stream_down_yt.peakRate)
    resume['median_fl_peak_adsl_2009_yt'] = np.median(
        80 * flows_ADSL_2009_stream_down_yt.peakRate)
    resume['max_fl_peak_adsl_2009_yt'] = max(
        80 * flows_ADSL_2009_stream_down_yt.peakRate)
    meanRate_adsl_2009_yt = [8*x['l3Bytes']/(1000.0*x['duration'])
                             for x in flows_ADSL_2009_stream_down_yt
                             if x['duration']>0]
    resume['mean_fl_meanRate_adsl_2009_yt'] = np.mean(meanRate_adsl_2009_yt)
    resume['median_fl_meanRate_adsl_2009_yt'] = np.median(meanRate_adsl_2009_yt)
    resume['max_fl_meanRate_adsl_2009_yt'] = max(meanRate_adsl_2009_yt)
    meanRate_adsl_2009_yt_1MB = [8*x['l3Bytes']/(1000.0*x['duration'])
                                 for x in flows_ADSL_2009_stream_down_yt
                                 if x['duration']>0
                                 and x['l3Bytes'] > 10**6]
    resume['mean_fl_meanRate_adsl_2009_yt_1MB'] = np.mean(
        meanRate_adsl_2009_yt_1MB)
    resume['median_fl_meanRate_adsl_2009_yt_1MB'] = np.median(
        meanRate_adsl_2009_yt_1MB)
    resume['max_fl_meanRate_adsl_2009_yt_1MB'] = max(meanRate_adsl_2009_yt_1MB)
    resume['mean_fl_AR_adsl_2009_yt'] = compute_AT.compute_AT(
        flows_ADSL_2009_stream_down_yt.initTime)[0]
    resume['mean_fl_100AR_per_cl_adsl_2009_yt'] \
        = 100 * resume['mean_fl_AR_adsl_2009_yt']/resume['nb_cl_adsl_2009_yt']
    #ADSL 2009: other
    resume['nb_cl_adsl_2009_other'] = len(np.unique(
        flows_ADSL_2009_stream_down_other.client_id))
    resume['nb_fl_adsl_2009_other'] = len(flows_ADSL_2009_stream_down_other)
    flows_ADSL_2009_stream_down_other_1MB = flows_ADSL_2009_stream_down_other.compress(
        flows_ADSL_2009_stream_down_other.l3Bytes > 10**6 )
    resume['nb_cl_adsl_2009_other_1MB'] = len(np.unique(
        flows_ADSL_2009_stream_down_other_1MB.client_id))
    resume['nb_fl_adsl_2009_other_1MB'] = len(
        flows_ADSL_2009_stream_down_other_1MB)
    resume['mean_fl_size_adsl_2009_other'] = np.mean(
        flows_ADSL_2009_stream_down_other.l3Bytes)
    resume['median_fl_size_adsl_2009_other'] = np.median(
        flows_ADSL_2009_stream_down_other.l3Bytes)
    resume['max_fl_size_adsl_2009_other'] = max(
        flows_ADSL_2009_stream_down_other.l3Bytes)
    resume['mean_fl_dur_adsl_2009_other'] = np.mean(
        flows_ADSL_2009_stream_down_other.duration)
    resume['median_fl_dur_adsl_2009_other'] = np.median(
        flows_ADSL_2009_stream_down_other.duration)
    resume['max_fl_dur_adsl_2009_other'] = max(
        flows_ADSL_2009_stream_down_other.duration)
    resume['mean_fl_peak_adsl_2009_other'] = np.mean(
        80 * flows_ADSL_2009_stream_down_other.peakRate)
    resume['median_fl_peak_adsl_2009_other'] = np.median(
        80 * flows_ADSL_2009_stream_down_other.peakRate)
    resume['max_fl_peak_adsl_2009_other'] = max(
        80 * flows_ADSL_2009_stream_down_other.peakRate)
    meanRate_adsl_2009_other = [8*x['l3Bytes']/(1000.0*x['duration'])
                             for x in flows_ADSL_2009_stream_down_other
                                if x['duration']>0]
    resume['mean_fl_meanRate_adsl_2009_other'] = np.mean(
        meanRate_adsl_2009_other)
    resume['median_fl_meanRate_adsl_2009_other'] = np.median(
        meanRate_adsl_2009_other)
    resume['max_fl_meanRate_adsl_2009_other'] = max(meanRate_adsl_2009_other)
    meanRate_adsl_2009_other_1MB = [8*x['l3Bytes']/(1000.0*x['duration'])
                                 for x in flows_ADSL_2009_stream_down_other
                                    if x['duration']>0
                                 and x['l3Bytes'] > 10**6]
    resume['mean_fl_meanRate_adsl_2009_other_1MB'] = np.mean(
        meanRate_adsl_2009_other_1MB)
    resume['median_fl_meanRate_adsl_2009_other_1MB'] = np.median(
        meanRate_adsl_2009_other_1MB)
    resume['max_fl_meanRate_adsl_2009_other_1MB'] = max(
        meanRate_adsl_2009_other_1MB)
    resume['mean_fl_AR_adsl_2009_other'] = compute_AT.compute_AT(
        flows_ADSL_2009_stream_down_other.initTime)[0]
    resume['mean_fl_100AR_per_cl_adsl_2009_other'] \
        = 100 * resume['mean_fl_AR_adsl_2009_other'] / resume['nb_cl_adsl_2009_other']


    #FTTH 2009: YT
    resume['nb_cl_ftth_2009_yt'] = len(
        np.unique(flows_FTTH_2009_stream_down_yt.client_id))
    resume['nb_fl_ftth_2009_yt'] = len(flows_FTTH_2009_stream_down_yt)
    flows_FTTH_2009_stream_down_yt_1MB = flows_FTTH_2009_stream_down_yt.compress(
        flows_FTTH_2009_stream_down_yt.l3Bytes > 10**6 )
    resume['nb_cl_ftth_2009_yt_1MB'] = len(
        np.unique(flows_FTTH_2009_stream_down_yt_1MB.client_id))
    resume['nb_fl_ftth_2009_yt_1MB'] = len(flows_FTTH_2009_stream_down_yt_1MB)
    resume['mean_fl_size_ftth_2009_yt'] = np.mean(
        flows_FTTH_2009_stream_down_yt.l3Bytes)
    resume['median_fl_size_ftth_2009_yt'] = np.median(
        flows_FTTH_2009_stream_down_yt.l3Bytes)
    resume['max_fl_size_ftth_2009_yt'] = max(
        flows_FTTH_2009_stream_down_yt.l3Bytes)
    resume['mean_fl_dur_ftth_2009_yt'] = np.mean(
        flows_FTTH_2009_stream_down_yt.duration)
    resume['median_fl_dur_ftth_2009_yt'] = np.median(
        flows_FTTH_2009_stream_down_yt.duration)
    resume['max_fl_dur_ftth_2009_yt'] = max(
        flows_FTTH_2009_stream_down_yt.duration)
    resume['mean_fl_peak_ftth_2009_yt'] = np.mean(
        80 * flows_FTTH_2009_stream_down_yt.peakRate)
    resume['median_fl_peak_ftth_2009_yt'] = np.median(
        80 * flows_FTTH_2009_stream_down_yt.peakRate)
    resume['max_fl_peak_ftth_2009_yt'] = max(
        80 * flows_FTTH_2009_stream_down_yt.peakRate)
    meanRate_ftth_2009_yt = [8*x['l3Bytes']/(1000.0*x['duration'])
                             for x in flows_FTTH_2009_stream_down_yt
                             if x['duration']>0]
    resume['mean_fl_meanRate_ftth_2009_yt'] = np.mean(meanRate_ftth_2009_yt)
    resume['median_fl_meanRate_ftth_2009_yt'] = np.median(meanRate_ftth_2009_yt)
    resume['max_fl_meanRate_ftth_2009_yt'] = max(meanRate_ftth_2009_yt)
    meanRate_ftth_2009_yt_1MB = [8*x['l3Bytes']/(1000.0*x['duration'])
                                 for x in flows_FTTH_2009_stream_down_yt
                                 if x['duration']>0 and x['l3Bytes'] > 10**6]
    resume['mean_fl_meanRate_ftth_2009_yt_1MB'] = np.mean(
        meanRate_ftth_2009_yt_1MB)
    resume['median_fl_meanRate_ftth_2009_yt_1MB'] = np.median(
        meanRate_ftth_2009_yt_1MB)
    resume['max_fl_meanRate_ftth_2009_yt_1MB'] = max(meanRate_ftth_2009_yt_1MB)
    resume['mean_fl_AR_ftth_2009_yt'] = compute_AT.compute_AT(
        flows_FTTH_2009_stream_down_yt.initTime)[0]
    resume['mean_fl_100AR_per_cl_ftth_2009_yt'] \
        = 100 * resume['mean_fl_AR_ftth_2009_yt']/resume['nb_cl_ftth_2009_yt']
    #FTTH 2009: other
    resume['nb_cl_ftth_2009_other'] = len(np.unique(
        flows_FTTH_2009_stream_down_other.client_id))
    resume['nb_fl_ftth_2009_other'] = len(flows_FTTH_2009_stream_down_other)
    flows_FTTH_2009_stream_down_other_1MB = flows_FTTH_2009_stream_down_other.compress(
        flows_FTTH_2009_stream_down_other.l3Bytes > 10**6 )
    resume['nb_cl_ftth_2009_other_1MB'] = len(np.unique(
        flows_FTTH_2009_stream_down_other_1MB.client_id))
    resume['nb_fl_ftth_2009_other_1MB'] = len(
        flows_FTTH_2009_stream_down_other_1MB)
    resume['mean_fl_size_ftth_2009_other'] = np.mean(
        flows_FTTH_2009_stream_down_other.l3Bytes)
    resume['median_fl_size_ftth_2009_other'] = np.median(
        flows_FTTH_2009_stream_down_other.l3Bytes)
    resume['max_fl_size_ftth_2009_other'] = max(
        flows_FTTH_2009_stream_down_other.l3Bytes)
    resume['mean_fl_dur_ftth_2009_other'] = np.mean(
        flows_FTTH_2009_stream_down_other.duration)
    resume['median_fl_dur_ftth_2009_other'] = np.median(
        flows_FTTH_2009_stream_down_other.duration)
    resume['max_fl_dur_ftth_2009_other'] = max(
        flows_FTTH_2009_stream_down_other.duration)
    resume['mean_fl_peak_ftth_2009_other'] = np.mean(
        80 * flows_FTTH_2009_stream_down_other.peakRate)
    resume['median_fl_peak_ftth_2009_other'] = np.median(
        80 * flows_FTTH_2009_stream_down_other.peakRate)
    resume['max_fl_peak_ftth_2009_other'] = max(
        80 * flows_FTTH_2009_stream_down_other.peakRate)
    meanRate_ftth_2009_other = [8*x['l3Bytes']/(1000.0*x['duration'])
                             for x in flows_FTTH_2009_stream_down_other
                                if x['duration']>0]
    resume['mean_fl_meanRate_ftth_2009_other'] = np.mean(
        meanRate_ftth_2009_other)
    resume['median_fl_meanRate_ftth_2009_other'] = np.median(
        meanRate_ftth_2009_other)
    resume['max_fl_meanRate_ftth_2009_other'] = max(meanRate_ftth_2009_other)
    meanRate_ftth_2009_other_1MB = [8*x['l3Bytes']/(1000.0*x['duration'])
                                 for x in flows_FTTH_2009_stream_down_other
                                    if x['duration']>0 and x['l3Bytes'] > 10**6]
    resume['mean_fl_meanRate_ftth_2009_other_1MB'] = np.mean(
        meanRate_ftth_2009_other_1MB)
    resume['median_fl_meanRate_ftth_2009_other_1MB'] = np.median(
        meanRate_ftth_2009_other_1MB)
    resume['max_fl_meanRate_ftth_2009_other_1MB'] = max(
        meanRate_ftth_2009_other_1MB)
    resume['mean_fl_AR_ftth_2009_other'] = compute_AT.compute_AT(
        flows_FTTH_2009_stream_down_other.initTime)[0]
    resume['mean_fl_100AR_per_cl_ftth_2009_other'] \
        = 100 * resume['mean_fl_AR_ftth_2009_other'] / resume['nb_cl_ftth_2009_other']

    return resume

def format_resume_http_tab(resume):
    return r"""\begin{tabular}{|l||c|c||c|c||c|c||c|c|}
\hline
&\multicolumn{2}{c||}{ADSL 2008}&
\multicolumn{2}{c||}{FTTH 2008}&\multicolumn{2}{c||}{ADSL 2009}
&\multicolumn{2}{c|}{FTTH 2009}\\
& YouTube & Other& YouTube & Other& YouTube & Other& YouTube & Other\\
\hline
\multicolumn{9}{l}{Nb Clients}\\
\hline
All & %(nb_cl_adsl_2008_yt)d & %(nb_cl_adsl_2008_other)d& %(nb_cl_ftth_2008_yt)d
& %(nb_cl_ftth_2008_other)d & %(nb_cl_adsl_2009_yt)d & %(nb_cl_adsl_2009_other)d
& %(nb_cl_ftth_2009_yt)d & %(nb_cl_ftth_2009_other)d\\
with 1 flow $>$1\,MB & %(nb_cl_adsl_2008_yt_1MB)d
& %(nb_cl_adsl_2008_other_1MB)d& %(nb_cl_ftth_2008_yt_1MB)d
& %(nb_cl_ftth_2008_other_1MB)d & %(nb_cl_adsl_2009_yt_1MB)d
& %(nb_cl_adsl_2009_other_1MB)d& %(nb_cl_ftth_2009_yt_1MB)d
& %(nb_cl_ftth_2009_other_1MB)d\\
\hline
\multicolumn{9}{l}{Nb Flows}\\
\hline
All & %(nb_fl_adsl_2008_yt)d & %(nb_fl_adsl_2008_other)d& %(nb_fl_ftth_2008_yt)d
& %(nb_fl_ftth_2008_other)d & %(nb_fl_adsl_2009_yt)d & %(nb_fl_adsl_2009_other)d
& %(nb_fl_ftth_2009_yt)d & %(nb_fl_ftth_2009_other)d\\
with 1 flow $>$1\,MB & %(nb_fl_adsl_2008_yt_1MB)d
& %(nb_fl_adsl_2008_other_1MB)d& %(nb_fl_ftth_2008_yt_1MB)d
& %(nb_fl_ftth_2008_other_1MB)d & %(nb_fl_adsl_2009_yt_1MB)d
& %(nb_fl_adsl_2009_other_1MB)d& %(nb_fl_ftth_2009_yt_1MB)d
& %(nb_fl_ftth_2009_other_1MB)d\\
\hline
\multicolumn{9}{l}{Flows Size in Bytes}\\
\hline
Mean & %(mean_fl_size_adsl_2008_yt).4g\,B
& %(mean_fl_size_adsl_2008_other).4g\,B & %(mean_fl_size_ftth_2008_yt).4g\,B
& %(mean_fl_size_ftth_2008_other).4g\,B & %(mean_fl_size_adsl_2009_yt).4g\,B
& %(mean_fl_size_adsl_2009_other).4g\,B & %(mean_fl_size_ftth_2009_yt).4g\,B
& %(mean_fl_size_ftth_2009_other).4g\,B \\
Median & %(median_fl_size_adsl_2008_yt).4g\,B
& %(median_fl_size_adsl_2008_other).4g\,B
& %(median_fl_size_ftth_2008_yt).4g\,B
& %(median_fl_size_ftth_2008_other).4g\,B
& %(median_fl_size_adsl_2009_yt).4g\,B
& %(median_fl_size_adsl_2009_other).4g\,B
& %(median_fl_size_ftth_2009_yt).4g\,B
& %(median_fl_size_ftth_2009_other).4g\,B \\
Max & %(max_fl_size_adsl_2008_yt).4g\,B
& %(max_fl_size_adsl_2008_other).4g\,B & %(max_fl_size_ftth_2008_yt).4g\,B
& %(max_fl_size_ftth_2008_other).4g\,B & %(max_fl_size_adsl_2009_yt).4g\,B
& %(max_fl_size_adsl_2009_other).4g\,B & %(max_fl_size_ftth_2009_yt).4g\,B
& %(max_fl_size_ftth_2009_other).4g\,B \\
\hline
\multicolumn{9}{l}{Flows Duration in Seconds}\\
\hline
Mean & %(mean_fl_dur_adsl_2008_yt).4g\,s & %(mean_fl_dur_adsl_2008_other).4g\,s
& %(mean_fl_dur_ftth_2008_yt).4g\,s & %(mean_fl_dur_ftth_2008_other).4g\,s
& %(mean_fl_dur_adsl_2009_yt).4g\,s & %(mean_fl_dur_adsl_2009_other).4g\,s
& %(mean_fl_dur_ftth_2009_yt).4g\,s & %(mean_fl_dur_ftth_2009_other).4g\,s \\
Median & %(median_fl_dur_adsl_2008_yt).4g\,s
& %(median_fl_dur_adsl_2008_other).4g\,s & %(median_fl_dur_ftth_2008_yt).4g\,s
& %(median_fl_dur_ftth_2008_other).4g\,s & %(median_fl_dur_adsl_2009_yt).4g\,s
& %(median_fl_dur_adsl_2009_other).4g\,s & %(median_fl_dur_ftth_2009_yt).4g\,s
& %(median_fl_dur_ftth_2009_other).4g\,s \\
Max & %(max_fl_dur_adsl_2008_yt).4g\,s & %(max_fl_dur_adsl_2008_other).4g\,s
& %(max_fl_dur_ftth_2008_yt).4g\,s & %(max_fl_dur_ftth_2008_other).4g\,s
& %(max_fl_dur_adsl_2009_yt).4g\,s & %(max_fl_dur_adsl_2009_other).4g\,s
& %(max_fl_dur_ftth_2009_yt).4g\,s & %(max_fl_dur_ftth_2009_other).4g\,s \\
\hline
\multicolumn{9}{l}{Flows Peak Rate in b/s (over 100ms)}\\
\hline
Mean & %(mean_fl_peak_adsl_2008_yt).4g\,b/s
& %(mean_fl_peak_adsl_2008_other).4g\,b/s
& %(mean_fl_peak_ftth_2008_yt).4g\,b/s
& %(mean_fl_peak_ftth_2008_other).4g\,b/s
& %(mean_fl_peak_adsl_2009_yt).4g\,b/s
& %(mean_fl_peak_adsl_2009_other).4g\,b/s
& %(mean_fl_peak_ftth_2009_yt).4g\,b/s
& %(mean_fl_peak_ftth_2009_other).4g\,b/s \\
Median & %(median_fl_peak_adsl_2008_yt).4g\,b/s
& %(median_fl_peak_adsl_2008_other).4g\,b/s
& %(median_fl_peak_ftth_2008_yt).4g\,b/s
& %(median_fl_peak_ftth_2008_other).4g\,b/s
& %(median_fl_peak_adsl_2009_yt).4g\,b/s
& %(median_fl_peak_adsl_2009_other).4g\,b/s
& %(median_fl_peak_ftth_2009_yt).4g\,b/s
& %(median_fl_peak_ftth_2009_other).4g\,b/s \\
Max & %(max_fl_peak_adsl_2008_yt).4g\,b/s
& %(max_fl_peak_adsl_2008_other).4g\,b/s & %(max_fl_peak_ftth_2008_yt).4g\,b/s
& %(max_fl_peak_ftth_2008_other).4g\,b/s & %(max_fl_peak_adsl_2009_yt).4g\,b/s
& %(max_fl_peak_adsl_2009_other).4g\,b/s & %(max_fl_peak_ftth_2009_yt).4g\,b/s
& %(max_fl_peak_ftth_2009_other).4g\,b/s \\
\hline
\multicolumn{9}{l}{Flows Mean Rate in kb/s}\\
\hline
Mean & %(mean_fl_meanRate_adsl_2008_yt).4g\,kb/s
& %(mean_fl_meanRate_adsl_2008_other).4g\,kb/s
& %(mean_fl_meanRate_ftth_2008_yt).4g\,kb/s
& %(mean_fl_meanRate_ftth_2008_other).4g\,kb/s
& %(mean_fl_meanRate_adsl_2009_yt).4g\,kb/s
& %(mean_fl_meanRate_adsl_2009_other).4g\,kb/s
& %(mean_fl_meanRate_ftth_2009_yt).4g\,kb/s
& %(mean_fl_meanRate_ftth_2009_other).4g\,kb/s \\
Median & %(median_fl_meanRate_adsl_2008_yt).4g\,kb/s
& %(median_fl_meanRate_adsl_2008_other).4g\,kb/s
& %(median_fl_meanRate_ftth_2008_yt).4g\,kb/s
& %(median_fl_meanRate_ftth_2008_other).4g\,kb/s
& %(median_fl_meanRate_adsl_2009_yt).4g\,kb/s
& %(median_fl_meanRate_adsl_2009_other).4g\,kb/s
& %(median_fl_meanRate_ftth_2009_yt).4g\,kb/s
& %(median_fl_meanRate_ftth_2009_other).4g\,kb/s \\
Max & %(max_fl_meanRate_adsl_2008_yt).4g\,kb/s
& %(max_fl_meanRate_adsl_2008_other).4g\,kb/s
& %(max_fl_meanRate_ftth_2008_yt).4g\,kb/s
& %(max_fl_meanRate_ftth_2008_other).4g\,kb/s
& %(max_fl_meanRate_adsl_2009_yt).4g\,kb/s
& %(max_fl_meanRate_adsl_2009_other).4g\,kb/s
& %(max_fl_meanRate_ftth_2009_yt).4g\,kb/s
& %(max_fl_meanRate_ftth_2009_other).4g\,kb/s \\
\hline
\multicolumn{9}{l}{Flows Arriva Rate in nb of flows per kilo Seconds}\\
\hline
All trace & %(mean_fl_AR_adsl_2008_yt).4g\,cns/ks
& %(mean_fl_AR_adsl_2008_other).4g\,cns/ks
& %(mean_fl_AR_ftth_2008_yt).4g\,cns/ks
& %(mean_fl_AR_ftth_2008_other).4g\,cns/ks
& %(mean_fl_AR_adsl_2009_yt).4g\,cns/ks
& %(mean_fl_AR_adsl_2009_other).4g\,cns/ks
& %(mean_fl_AR_ftth_2009_yt).4g\,cns/ks
& %(mean_fl_AR_ftth_2009_other).4g\,cns/ks \\
per nb of client & %(mean_fl_100AR_per_cl_adsl_2008_yt).4g\,cns/ks/cl
& %(mean_fl_100AR_per_cl_adsl_2008_other).4g\,cns/ks/cl
& %(mean_fl_100AR_per_cl_ftth_2008_yt).4g\,cns/ks/cl
& %(mean_fl_100AR_per_cl_ftth_2008_other).4g\,cns/ks/cl
& %(mean_fl_100AR_per_cl_adsl_2009_yt).4g\,cns/ks/cl
& %(mean_fl_100AR_per_cl_adsl_2009_other).4g\,cns/ks/cl
& %(mean_fl_100AR_per_cl_ftth_2009_yt).4g\,cns/ks/cl
& %(mean_fl_100AR_per_cl_ftth_2009_other).4g\,cns/ks/cl \\
\hline
\end{tabular}
""" % resume


#def main():
#    usage = "%prog -r data_file"
#
#    parser = OptionParser(usage = usage)
#    parser.add_option("-r", dest = "file", type = "string",
#                      help = "input data file")
#    (options, args) = parser.parse_args()
#
#    if not options.file:
#        parser.print_help()
#        exit()
#
#    fetch(options.file)
#
#if __name__ == '__main__':
#    main()
