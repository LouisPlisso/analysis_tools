#!/usr/bin/env python
"Module to extract a resume of a preloaded GVB flow file."


from optparse import OptionParser
import numpy as np

import INDEX_VALUES
import aggregate

def extract_aggregated_field(data, key, ref):
    return [f['aggregation'] for f in data if f[key] == ref][0]

def fetch(file='flows_FTTH_2009.npy'):
    "Extract a resume of a preloaded GVB flow file and format it into latex tabular."
    flows = np.load(file).view(np.recarray )
    format_latex_tab(fetch_data(flows))
    

def fetch_data(flows):
    "Return a resume of GVB flow recarray in the form of a dictonnary."
    resume = {}
    vol_dir = aggregate.aggregate(flows, 'direction', 'l3Bytes', sum)
    resume['vol_up'] = vol_dir[0][1]
    resume['vol_down'] = vol_dir[1][1]
    resume['vol_tot'] = resume['vol_down'] + resume['vol_up']

    vol_dscp = aggregate.aggregate(flows, 'dscp', 'l3Bytes', sum)
    resume['vol_down_web'] = extract_aggregated_field(vol_dscp, 'dscp', INDEX_VALUES.DSCP_WEB)
    resume['vol_down_http_stream'] = extract_aggregated_field(vol_dscp, 'dscp', INDEX_VALUES.DSCP_HTTP_STREAM)
    resume['vol_down_other_stream'] = extract_aggregated_field(vol_dscp, 'dscp', INDEX_VALUES.DSCP_OTHER_STREAM)

    #to check nb of flow value
#    nb_flows_dir = aggregate.aggregate(flows, 'direction', 'client_id', len)
#    nb_flows_up = list(nb_flows_dir[0])[1]
#    nb_flows_down = list(nb_flows_dir[1])[1]
    
    flows_down = flows.compress(flows.direction == INDEX_VALUES.DOWN)
    resume['nb_down_flows_tot'] = np.shape(flows_down)[0]

    flows_down_web = flows_down.compress(flows_down.dscp == INDEX_VALUES.DSCP_WEB )
    flows_down_other_stream = flows_down.compress(flows_down.dscp == INDEX_VALUES.DSCP_OTHER_STREAM )
    flows_down_http_stream = flows_down.compress(flows_down.dscp == INDEX_VALUES.DSCP_HTTP_STREAM )
    resume['nb_down_flows_web'] = np.shape(flows_down_web)[0]
    resume['nb_down_flows_http_stream'] = np.shape(flows_down_http_stream)[0]
    resume['nb_down_flows_other_stream'] = np.shape(flows_down_other_stream)[0]

    resume['nb_clients_tot'] = np.shape(np.unique(flows_down.client_id))[0]
    resume['nb_clients_web'] = np.shape(np.unique(flows_down_web.client_id))[0]
    resume['nb_clients_http_stream'] = np.shape(np.unique(flows_down_http_stream.client_id))[0]
    resume['nb_clients_other_stream'] = np.shape(np.unique(flows_down_other_stream.client_id))[0]

    flows_down_1MB = flows_down.compress(flows_down.l3Bytes > 10**6 )
    flows_down_1MB_dscp = aggregate.aggregate(flows_down_1MB, 'dscp', 'l3Bytes', len)

    flows_down_1MB_web = flows_down_1MB.compress(flows_down_1MB.dscp 
                                                 == INDEX_VALUES.DSCP_WEB )
    flows_down_1MB_http_stream = flows_down_1MB.compress(flows_down_1MB.dscp 
                                                         == INDEX_VALUES.DSCP_HTTP_STREAM )
    flows_down_1MB_other_stream = flows_down_1MB.compress(flows_down_1MB.dscp 
                                                          == INDEX_VALUES.DSCP_OTHER_STREAM )

    resume['nb_clients_1MB_tot'] = np.shape(np.unique(flows_down_1MB.client_id))[0]
    resume['nb_clients_1MB_web'] = np.shape(np.unique(flows_down_1MB_web.client_id))[0]
    resume['nb_clients_1MB_http_stream'] = np.shape(np.unique(flows_down_1MB_http_stream.client_id))[0]
    resume['nb_clients_1MB_other_stream'] = np.shape(np.unique(flows_down_1MB_other_stream.client_id))[0]

    resume['nb_down_flows_1MB_tot'] = np.shape(flows_down_1MB)[0]
    resume['nb_down_flows_1MB_web'] = extract_aggregated_field(flows_down_1MB_dscp, 
                                                               'dscp', INDEX_VALUES.DSCP_WEB)
    resume['nb_down_flows_1MB_http_stream'] = extract_aggregated_field(flows_down_1MB_dscp, 
                                                                       'dscp', INDEX_VALUES.DSCP_HTTP_STREAM)
    resume['nb_down_flows_1MB_other_stream'] = extract_aggregated_field(flows_down_1MB_dscp, 
                                                                        'dscp', INDEX_VALUES.DSCP_OTHER_STREAM)
    
    return resume 


def modify_and_fetch_data_named(resume, flows, name):
    "Modify a dictonnary resume, to extend it with a GVB array with a specifier name. "
    vol_dir = aggregate.aggregate(flows, 'direction', 'l3Bytes', sum)
    resume['vol_up_%s' % name] = vol_dir[0][1]
    resume['vol_down_%s' % name] = vol_dir[1][1]
    resume['vol_tot_%s' % name] = resume['vol_down_%s' % name] + resume['vol_up_%s' % name]

    vol_dscp = aggregate.aggregate(flows, 'dscp', 'l3Bytes', sum)
    resume['vol_down_web_%s' % name] = extract_aggregated_field(vol_dscp, 'dscp', INDEX_VALUES.DSCP_WEB)
    resume['vol_down_http_stream_%s' % name] = extract_aggregated_field(vol_dscp, 'dscp', INDEX_VALUES.DSCP_HTTP_STREAM)
    resume['vol_down_other_stream_%s' % name] = extract_aggregated_field(vol_dscp, 'dscp', INDEX_VALUES.DSCP_OTHER_STREAM)

    flows_down = flows.compress(flows.direction == INDEX_VALUES.DOWN)
    resume['nb_down_flows_tot_%s' % name] = np.shape(flows_down)[0]

    flows_down_web = flows_down.compress(flows_down.dscp == INDEX_VALUES.DSCP_WEB )
    flows_down_other_stream = flows_down.compress(flows_down.dscp == INDEX_VALUES.DSCP_OTHER_STREAM )
    flows_down_http_stream = flows_down.compress(flows_down.dscp == INDEX_VALUES.DSCP_HTTP_STREAM )
    resume['nb_down_flows_web_%s' % name] = np.shape(flows_down_web)[0]
    resume['nb_down_flows_http_stream_%s' % name] = np.shape(flows_down_http_stream)[0]
    resume['nb_down_flows_other_stream_%s' % name] = np.shape(flows_down_other_stream)[0]

    resume['vol_down_per_flow_tot_%s' % name] = resume['vol_down_%s' % name] / resume['nb_down_flows_tot_%s' % name]
    resume['vol_down_per_flow_web_%s' % name] = resume['vol_down_web_%s' % name] / resume['nb_down_flows_web_%s' % name]
    resume['vol_down_per_flow_http_stream_%s' % name] = resume['vol_down_http_stream_%s' % name] / resume['nb_down_flows_http_stream_%s' % name]
    resume['vol_down_per_flow_other_stream_%s' % name] = resume['vol_down_other_stream_%s' % name] / resume['nb_down_flows_other_stream_%s' % name]

    resume['nb_clients_tot_%s' % name] = np.shape(np.unique(flows_down.client_id))[0]
    resume['nb_clients_web_%s' % name] = np.shape(np.unique(flows_down_web.client_id))[0]
    resume['nb_clients_http_stream_%s' % name] = np.shape(np.unique(flows_down_http_stream.client_id))[0]
    resume['nb_clients_other_stream_%s' % name] = np.shape(np.unique(flows_down_other_stream.client_id))[0]

    resume['vol_down_per_client_tot_%s' % name] = resume['vol_down_%s' % name] / resume['nb_clients_tot_%s' % name]
    resume['vol_down_per_client_web_%s' % name] = resume['vol_down_web_%s' % name] / resume['nb_clients_web_%s' % name]
    resume['vol_down_per_client_http_stream_%s' % name] = resume['vol_down_http_stream_%s' % name] / resume['nb_clients_http_stream_%s' % name]
    resume['vol_down_per_client_other_stream_%s' % name] = resume['vol_down_other_stream_%s' % name] / resume['nb_clients_other_stream_%s' % name]


    flows_down_1MB = flows_down.compress(flows_down.l3Bytes > 10**6 )
    flows_down_1MB_dscp = aggregate.aggregate(flows_down_1MB, 'dscp', 'l3Bytes', len)

    flows_down_1MB_web = flows_down_1MB.compress(flows_down_1MB.dscp 
                                                 == INDEX_VALUES.DSCP_WEB )
    flows_down_1MB_http_stream = flows_down_1MB.compress(flows_down_1MB.dscp 
                                                         == INDEX_VALUES.DSCP_HTTP_STREAM )
    flows_down_1MB_other_stream = flows_down_1MB.compress(flows_down_1MB.dscp 
                                                          == INDEX_VALUES.DSCP_OTHER_STREAM )

    resume['nb_clients_1MB_tot_%s' % name] = np.shape(np.unique(flows_down_1MB.client_id))[0]
    resume['nb_clients_1MB_web_%s' % name] = np.shape(np.unique(flows_down_1MB_web.client_id))[0]
    resume['nb_clients_1MB_http_stream_%s' % name] = np.shape(np.unique(flows_down_1MB_http_stream.client_id))[0]
    resume['nb_clients_1MB_other_stream_%s' % name] = np.shape(np.unique(flows_down_1MB_other_stream.client_id))[0]

    resume['nb_down_flows_1MB_tot_%s' % name] = np.shape(flows_down_1MB)[0]
    resume['nb_down_flows_1MB_web_%s' % name] = extract_aggregated_field(flows_down_1MB_dscp, 
                                                               'dscp', INDEX_VALUES.DSCP_WEB)
    resume['nb_down_flows_1MB_http_stream_%s' % name] = extract_aggregated_field(flows_down_1MB_dscp, 
                                                                       'dscp', INDEX_VALUES.DSCP_HTTP_STREAM)
    resume['nb_down_flows_1MB_other_stream_%s' % name] = extract_aggregated_field(flows_down_1MB_dscp, 
                                                                        'dscp', INDEX_VALUES.DSCP_OTHER_STREAM)
    
    return resume 

def format_latex_tab_multi(resume):
    "Formats the resume dictionnary into a latex tabular."
    string = r"""\begin{center}
\begin{tabular}{|l|c|c|c|c|}
\hline
& ADSL 2008 & FTTH 2008 & ADSL 2009 & FTTH 2009\\
\hline
Total Volume & %(vol_tot_adsl_2008)4.2e\,Bytes& %(vol_tot_ftth_2008)4.2e\,Bytes& %(vol_tot_adsl_2009)4.2e\,Bytes& %(vol_tot_ftth_2009)4.2e\,Bytes\\
Up Volume & %(vol_up_adsl_2008)4.2e\,Bytes& %(vol_up_ftth_2008)4.2e\,Bytes& %(vol_up_adsl_2009)4.2e\,Bytes& %(vol_up_ftth_2009)4.2e\,Bytes\\
Down Volume& %(vol_down_adsl_2008)4.2e\,Bytes& %(vol_down_ftth_2008)4.2e\,Bytes& %(vol_down_adsl_2009)4.2e\,Bytes& %(vol_down_ftth_2009)4.2e\,Bytes\\
Nb of down flows & %(nb_down_flows_tot_adsl_2008).4g& %(nb_down_flows_tot_ftth_2008).4g& %(nb_down_flows_tot_adsl_2009).4g& %(nb_down_flows_tot_ftth_2009).4g\\
Average Vol Down/Flow & %(vol_down_per_flow_tot_adsl_2008)4.2e\,Bytes& %(vol_down_per_flow_tot_ftth_2008)4.2e\,Bytes& %(vol_down_per_flow_tot_adsl_2009)4.2e\,Bytes& %(vol_down_per_flow_tot_ftth_2009)4.2e\,Bytes\\
Nb of down flows $>$ 1\,MBytes & %(nb_down_flows_1MB_tot_adsl_2008).4g& %(nb_down_flows_1MB_tot_ftth_2008).4g& %(nb_down_flows_1MB_tot_adsl_2009).4g& %(nb_down_flows_1MB_tot_ftth_2009).4g\\
Nb of clients & %(nb_clients_tot_adsl_2008).4g& %(nb_clients_tot_ftth_2008).4g& %(nb_clients_tot_adsl_2009).4g& %(nb_clients_tot_ftth_2009).4g\\
Average Vol Down/Client & %(vol_down_per_client_tot_adsl_2008)4.2e\,Bytes& %(vol_down_per_client_tot_ftth_2008)4.2e\,Bytes& %(vol_down_per_client_tot_adsl_2009)4.2e\,Bytes& %(vol_down_per_client_tot_ftth_2009)4.2e\,Bytes\\
Nb of client with down flows $>$ 1\,MBytes & %(nb_clients_1MB_tot_adsl_2008).4g& %(nb_clients_1MB_tot_ftth_2008).4g& %(nb_clients_1MB_tot_adsl_2009).4g& %(nb_clients_1MB_tot_ftth_2009).4g\\
\hline
\multicolumn{5}{c}{Web Traffic}\\
\hline
Nb of clients& %(nb_clients_web_adsl_2008).4g& %(nb_clients_web_ftth_2008).4g& %(nb_clients_web_adsl_2009).4g& %(nb_clients_web_ftth_2009).4g\\
Volume Down & %(vol_down_web_adsl_2008)4.2e\,Bytes& %(vol_down_web_ftth_2008)4.2e\,Bytes& %(vol_down_web_adsl_2009)4.2e\,Bytes& %(vol_down_web_ftth_2009)4.2e\,Bytes\\
Average Vol Down/Client & %(vol_down_per_client_web_adsl_2008)4.2e\,Bytes& %(vol_down_per_client_web_ftth_2008)4.2e\,Bytes& %(vol_down_per_client_web_adsl_2009)4.2e\,Bytes& %(vol_down_per_client_web_ftth_2009)4.2e\,Bytes\\
Down flows& %(nb_down_flows_web_adsl_2008).4g& %(nb_down_flows_web_ftth_2008).4g& %(nb_down_flows_web_adsl_2009).4g& %(nb_down_flows_web_ftth_2009).4g\\
Average Vol Down/Flow & %(vol_down_per_flow_web_adsl_2008)4.2e\,Bytes& %(vol_down_per_flow_web_ftth_2008)4.2e\,Bytes& %(vol_down_per_flow_web_adsl_2009)4.2e\,Bytes& %(vol_down_per_flow_web_ftth_2009)4.2e\,Bytes\\
Nb of down flows $>$ 1\,MBytes & %(nb_down_flows_1MB_web_adsl_2008).4g& %(nb_down_flows_1MB_web_ftth_2008).4g& %(nb_down_flows_1MB_web_adsl_2009).4g& %(nb_down_flows_1MB_web_ftth_2009).4g\\
Nb of client with down flows $>$ 1\,MBytes & %(nb_clients_1MB_web_adsl_2008).4g& %(nb_clients_1MB_web_ftth_2008).4g& %(nb_clients_1MB_web_adsl_2009).4g& %(nb_clients_1MB_web_ftth_2009).4g\\
\hline
\multicolumn{5}{c}{HTTP Streaming Traffic}\\
\hline
Nb of clients& %(nb_clients_http_stream_adsl_2008).4g& %(nb_clients_http_stream_ftth_2008).4g& %(nb_clients_http_stream_adsl_2009).4g& %(nb_clients_http_stream_ftth_2009).4g\\
Volume Down& %(vol_down_http_stream_adsl_2008)4.2e\,Bytes & %(vol_down_http_stream_ftth_2008)4.2e\,Bytes & %(vol_down_http_stream_adsl_2009)4.2e\,Bytes & %(vol_down_http_stream_ftth_2009)4.2e\,Bytes \\
Average Vol Down/Client & %(vol_down_per_client_http_stream_adsl_2008)4.2e\,Bytes& %(vol_down_per_client_http_stream_ftth_2008)4.2e\,Bytes& %(vol_down_per_client_http_stream_adsl_2009)4.2e\,Bytes& %(vol_down_per_client_http_stream_ftth_2009)4.2e\,Bytes\\
Down Flows& %(nb_down_flows_http_stream_adsl_2008).4g& %(nb_down_flows_http_stream_ftth_2008).4g& %(nb_down_flows_http_stream_adsl_2009).4g& %(nb_down_flows_http_stream_ftth_2009).4g\\
Average Vol Down/Flow & %(vol_down_per_flow_http_stream_adsl_2008)4.2e\,Bytes& %(vol_down_per_flow_http_stream_ftth_2008)4.2e\,Bytes& %(vol_down_per_flow_http_stream_adsl_2009)4.2e\,Bytes& %(vol_down_per_flow_http_stream_ftth_2009)4.2e\,Bytes\\
Nb of down flows $>$ 1\,MBytes & %(nb_down_flows_1MB_http_stream_adsl_2008).4g& %(nb_down_flows_1MB_http_stream_ftth_2008).4g& %(nb_down_flows_1MB_http_stream_adsl_2009).4g& %(nb_down_flows_1MB_http_stream_ftth_2009).4g\\
Nb of client with down flows $>$ 1\,MBytes & %(nb_clients_1MB_http_stream_adsl_2008).4g& %(nb_clients_1MB_http_stream_ftth_2008).4g& %(nb_clients_1MB_http_stream_adsl_2009).4g& %(nb_clients_1MB_http_stream_ftth_2009).4g\\
\hline
\multicolumn{5}{c}{Other Streaming Traffic}\\
\hline
Nb of clients& %(nb_clients_other_stream_adsl_2008).4g& %(nb_clients_other_stream_ftth_2008).4g& %(nb_clients_other_stream_adsl_2009).4g& %(nb_clients_other_stream_ftth_2009).4g\\
Volume Down& %(vol_down_other_stream_adsl_2008)4.2e\,Bytes & %(vol_down_other_stream_ftth_2008)4.2e\,Bytes & %(vol_down_other_stream_adsl_2009)4.2e\,Bytes & %(vol_down_other_stream_ftth_2009)4.2e\,Bytes \\
Average Vol Down/Client & %(vol_down_per_client_other_stream_adsl_2008)4.2e\,Bytes& %(vol_down_per_client_other_stream_ftth_2008)4.2e\,Bytes& %(vol_down_per_client_other_stream_adsl_2009)4.2e\,Bytes& %(vol_down_per_client_other_stream_ftth_2009)4.2e\,Bytes\\
Down Flows& %(nb_down_flows_other_stream_adsl_2008).4g& %(nb_down_flows_other_stream_ftth_2008).4g& %(nb_down_flows_other_stream_adsl_2009).4g& %(nb_down_flows_other_stream_ftth_2009).4g\\
Average Vol Down/Flow & %(vol_down_per_flow_other_stream_adsl_2008)4.2e\,Bytes& %(vol_down_per_flow_other_stream_ftth_2008)4.2e\,Bytes& %(vol_down_per_flow_other_stream_adsl_2009)4.2e\,Bytes& %(vol_down_per_flow_other_stream_ftth_2009)4.2e\,Bytes\\
Nb of down flows $>$ 1\,MBytes & %(nb_down_flows_1MB_other_stream_adsl_2008).4g& %(nb_down_flows_1MB_other_stream_ftth_2008).4g& %(nb_down_flows_1MB_other_stream_adsl_2009).4g& %(nb_down_flows_1MB_other_stream_ftth_2009).4g\\
Nb of client with down flows $>$ 1\,MBytes & %(nb_clients_1MB_other_stream_adsl_2008).4g& %(nb_clients_1MB_other_stream_ftth_2008).4g& %(nb_clients_1MB_other_stream_adsl_2009).4g& %(nb_clients_1MB_other_stream_ftth_2009).4g\\
\hline
\end{tabular}
\end{center}
""" % resume
    #return or print: one must choose
    return string

def format_latex_tab(resume):
    "Formats the resume dictionnary into a latex tabular."
    print r"""\begin{center}
\begin{tabular}{|c|c|}
\hline
Total Volume & %(vol_tot)4.2e\,Bytes\\
Up Volume & %(vol_up)4.2e\,Bytes\\
Down Volume& %(vol_down)4.2e\,Bytes\\
Nb of down flows & %(nb_down_flows_tot).4g\\
Nb of down flows $>$ 1\,MBytes & %(nb_down_flows_1MB_tot).4g\\
Nb of clients & %(nb_clients_tot).4g\\
Nb of client with down flows $>$ 1\,MBytes & %(nb_clients_1MB_tot).4g\\
\hline
\multicolumn{2}{c}{Web Traffic}\\
\hline
Nb of clients& %(nb_clients_web).4g\\
Volume Down & %(vol_down_web)4.2e\,Bytes\\
Down flows& %(nb_down_flows_web).4g\\
Nb of down flows $>$ 1\,MBytes & %(nb_down_flows_1MB_web).4g\\
Nb of client with down flows $>$ 1\,MBytes & %(nb_clients_1MB_web).4g\\
\hline
\multicolumn{2}{c}{HTTP Streaming Traffic}\\
\hline
Nb of clients& %(nb_clients_http_stream).4g\\
Volume Down& %(vol_down_http_stream)4.2e\,Bytes \\
Down Flows& %(nb_down_flows_http_stream).4g\\
Nb of down flows $>$ 1\,MBytes & %(nb_down_flows_1MB_http_stream).4g\\
Nb of client with down flows $>$ 1\,MBytes & %(nb_clients_1MB_http_stream).4g\\
\hline
\multicolumn{2}{c}{Other Streaming Traffic}\\
\hline
Nb of clients& %(nb_clients_other_stream).4g\\
Volume Down&  %(vol_down_other_stream)4.2e\,Bytes \\
Down Flows&  %(nb_down_flows_other_stream).4g\\
Nb of down flows $>$ 1\,MBytes & %(nb_down_flows_1MB_other_stream).4g\\
Nb of client with down flows $>$ 1\,MBytes & %(nb_clients_1MB_other_stream).4g\\
\hline
\end{tabular}
\end{center}
""" % resume


def main():
    usage = "%prog -r data_file"

    parser = OptionParser(usage = usage)
    parser.add_option("-r", dest = "file", type = "string",
                      help = "input data file") 
    (options, args) = parser.parse_args()

    if not options.file:
        parser.print_help()
        exit()
	
    fetch(options.file)

if __name__ == '__main__':
    main()
