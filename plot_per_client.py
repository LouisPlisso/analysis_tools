"Module to compute stats per client"


import pylab
import cdfplot
from collections import defaultdict
from INDEX_VALUES import Everything, Nothing
import INDEX_VALUES
import aggregate
from streaming_tools import format_title, un_mismatch_dscp

MIN_NB_FLOWS = 15

def format_as_title(trace):
    r"""Return a formatted string for the trace name
    >>> format_title('2010_02_07_FTTH_GVB_YOUTUBE')
    '2010/02/07 FTTH\nYOUTUBE'
    >>>
    """
    trace_name = trace.split('.')[0]
    trace_date, trace_type = trace_name.split('_GVB_')
    trace_date = '/'.join(trace_date.split('_')[:3]) + ' ' + \
            ' '.join(trace_date.split('_')[3:])
    trace_type = ' '.join(trace_type.split('_')).lstrip('GVB ')
    return '\n'.join((trace_date, trace_type))


def filter_down_traces(data,
        trace_type='GVB', list_exclude=None):
    """Return a dict of arrays downstream
    """
    data_down = {}
    for trace in (t for t in data.keys() if t.find(trace_type) >=0):
        data_down[trace] = filter_array(data[trace], 'direction',
                                             (INDEX_VALUES.DOWN,),
                                             list_exclude)
    return data_down

def filter_mix_traces_streaming(data,
        trace_type='GVB', list_exclude=None):
    """Return a dict of arrays filtered for streaming using the unmismatch
    tool
    """
    data_streaming = {}
    for trace in (t for t in data.keys() if t.find(trace_type) >=0):
        dscp_http_streaming, _, _ = un_mismatch_dscp(data[trace])
        data_streaming[trace] = filter_array(data[trace], 'dscp',
                                             (dscp_http_streaming,),
                                             list_exclude)
    return data_streaming


def filter_array(data, field, list_include, list_exclude):
    """Returns an array of the data filtered on field value that is in
    list_include and not in list_exclude
    """
    if list_include == None:
        tmp = data
    else:
        tmp = pylab.array([x for x in data if x[field] in list_include],
                dtype=data.dtype)
    if list_exclude == None:
        out = tmp
    else:
        out = pylab.array([x for x in tmp if x[field] not in list_exclude],
                dtype=data.dtype)
    return pylab.array(out, dtype=data.dtype)

def filter_all_large(datas, list_exclude=None):
    """Returns a dict of arrays (one for each item of dict data) filtered on
    large flows (>1MB)
    """
    data_large = {}
    for trace in datas:
        data = datas[trace]
        if trace.find('_GVB') > 0:
            data_large[trace] = data.compress(data['l3Bytes']>10**6)
        if trace.find('_DIPCP') > 0:
            data_large[trace] = data.compress(data['DIP-Volume-Sum-Bytes-Down']
                                              + data['DIP-Volume-Sum-Bytes-Up']
                                              >10**6)
    return data_large


def filter_all_traces(data, field, values,
        trace_type='GVB', list_exclude=None):
    """Returns a dict of arrays (one for each item of dict data) filtered on
    field with value
    data = tools.load_hdf5_data.load_h5_file('flows/hdf5/traces_lzf.h5')
    data_streaming = tools.plot_per_client.filter_all_traces(data, 'dscp',
        tools.INDEX_VALUES.DSCP_HTTP_STREAM)
    """
    data_streaming = {}
    if not getattr(values, '__iter__', None):
        values = (values,)
    for trace in (t for t in data if t.find(trace_type) >=0):
        data_streaming[trace] = filter_array(data[trace], field, values,
                list_exclude)
    return data_streaming

def filter_array_list(data, trace, list_include, list_exclude, field='asBGP'):
    """Returns a dict of arrays filtered with filter_array on AS number with
            list_exclude and list_include
        data_streaming = tools.plot_per_client.filter_array_list(data,
        '2010_02_07_ADSL_R_GVB', (((tools.INDEX_VALUES.DSCP_HTTP_STREAM,),
        'HTTP_STREAMING'),), rien, field='dscp')
            """
    tmp = {}
    if list_include == None:
        list_include = Everything()
    if list_exclude == None:
        list_exclude = Nothing()
    for as_list, as_name in list_include:
        tmp['%s_%s' % (trace, as_name)] = filter_array(data[trace],
            field, as_list, list_exclude)
    return tmp

def vol_per_client(data, as_list=None, as_excluded=None, on_list=False,
    field='l3Bytes', func=sum,
    output_path = 'rapport/client_ok', title='', prefix = ''#,
#    trace_list = ('ADSL_2008', 'FTTH_2008', 'ADSL_nov_2009', 'FTTH_nov_2009',
#        'ADSL_dec_2009', 'FTTH_dec_2009')
        ):
    """Plots volumes per clients according to AS match list:
    use * for all ASes.
    flag 'on_list' works only on AS_list (included AS) and AS_list elements are
    filters and names: see exemples
    Use as:
    data = tools.load_hdf5_data.load_h5_file('hdf5/lzf_data.h5')
    tools.plot_per_client.vol_per_client(data)
    tools.plot_per_client.vol_per_client(data,
        ('*', tools.INDEX_VALUES.AS_YOUTUBE))
    tools.plot_per_client.vol_per_client(data,
        as_excluded=tools.INDEX_VALUES.AS_YOUTUBE
        +tools.INDEX_VALUES.AS_YOUTUBE_EU,
        title='Other Streams', prefix='OTHER_')
    tools.plot_per_client.vol_per_client(data_streaming,
        as_list=((tools.INDEX_VALUES.AS_YOUTUBE, 'YOUTUBE'),
        (tools.INDEX_VALUES.AS_YOUTUBE_EU, 'YOUTUBE_EU')),
        title='YT and YT-EU Streams', prefix='YT_YT_EU_', on_list=True,
        output_path='rapport/client_ok')
    tools.plot_per_client.vol_per_client(data,
        as_list=((tools.INDEX_VALUES.AS_YOUTUBE, 'YOUTUBE'),
        (tools.INDEX_VALUES.AS_YOUTUBE_EU, 'YOUTUBE_EU'),
        (tools.INDEX_VALUES.AS_GOOGLE, 'GOOGLE')),
        title='YT and GOO Streams', prefix='YT_GOO', on_list=True,
        output_path='rapport/client_ok')
    """
    client_vol = {}
    # data collection
    args = []
    # TODO: AS list
    for trace in sorted([x for x in data.keys() if '_GVB' in x]):
        print 'process trace: ', trace
        filtered_data_dict = defaultdict(dict)
        if on_list:
            filtered_data_dict[trace] = filter_array_list(data, trace,
                    as_list, as_excluded)
        else:
            filtered_data_dict[trace][trace] = filter_array(data[trace],
                    'asBGP', as_list, as_excluded)
        for name in sorted(filtered_data_dict[trace]):
            filtered_data = filtered_data_dict[trace][name]
            # at least MIN_NB_FLOWS flows per data to plot
            if len(filtered_data) < MIN_NB_FLOWS:
                continue
            client_vol[name] = aggregate.aggregate(filtered_data,
                    'client_id', field, func)
            # construct plot args
            if as_list:
                title_name = format_as_title(name)
            else:
                title_name = format_title(name).rstrip(' GVB')
            args.append((title_name, client_vol[name]['aggregation']))
            # plot individual repartitions
            pylab.clf()
            cdfplot.repartplotdata(client_vol[name]['aggregation'],
                _title='%s Volume per Client for %s' % (title, trace),
                _ylabel='Percentage of Downstream Volume', _loc=0)
            cdfplot.setgraph_loglog()
            pylab.savefig(output_path
                + '/%s%s_repart_volume_per_client.pdf' % (prefix, trace))
    # plot CDF
    pylab.clf()
    cdfplot.cdfplotdataN(args, _title='%s Volume per Client' % title,
                         _xlabel='Downstream Volume in Bytes', _loc=0)
    pylab.savefig(output_path + '/%sCDF_volume_per_client.pdf' % prefix)

    # plot global repartition
    pylab.clf()
    cdfplot.repartplotdataN(args, _title='%s Volume per Client' % title,
            _ylabel='Percentage of Downstream Volume', _loc=0)
    pylab.savefig(output_path + '/%srepart_volume_per_client.pdf' % prefix)
