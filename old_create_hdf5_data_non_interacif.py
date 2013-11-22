import numpy as np

import h5py

f = h5py.File('hdf5/data_streaming.h5', 'w')

ADSL_2008 = f.create_group("ADSL_Montsouris_2008_07_01")
# retreve: ADSL_2008 = f['ADSL_Montsouris_2008_07_01']
gvb_adsl_2008 = np.load('python_flows/flows_marked_GVB_juill_2008_ADSL_cut_BGP_AS.npy')
ADSL_2008.create_dataset('GVB', data=gvb_adsl_2008)
dipcp_adsl_2008 = np.load('python_flows/dipcp_flows_ADSL_juill_2008.npy')
ADSL_2008.create_dataset('dipcp', data=dipcp_adsl_2008)


FTTH_2008 = f.create_group("FTTH_Montsouris_2008_07_01")
# retreve: FTTH_2008 = f['FTTH_Montsouris_2008_07_01']
gvb_ftth_2008 = np.load('python_flows/flows_marked_GVB_juill_2008_FTTH_BGP_AS.npy')
FTTH_2008.create_dataset('GVB', data=gvb_ftth_2008)
dipcp_ftth_2008 = np.load('python_flows/dipcp_flows_FTTH_juill_2008_TCP.npy')
FTTH_2008.create_dataset('dipcp', data=dipcp_ftth_2008)


ADSL_nov_2009 = f.create_group("ADSL_Montsouris_2009_11_26")
gvb_adsl_nov_2009 = np.load('python_flows/flows_marked_GVB_nov_2009_ADSL_BGP_AS.npy')
ADSL_nov_2009.create_dataset('GVB', data=gvb_adsl_nov_2009)
dipcp_adsl_nov_2009 = np.load('python_flows/dipcp_flows_ADSL_nov_2009.npy')
ADSL_nov_2009.create_dataset('dipcp', data=dipcp_adsl_nov_2009)


FTTH_nov_2009 = f.create_group("FTTH_Montsouris_2009_11_26")
gvb_ftth_nov_2009 = np.load('python_flows/flows_marked_GVB_nov_2009_FTTH_BGP_AS.npy')
FTTH_nov_2009.create_dataset('GVB', data=gvb_ftth_nov_2009)
dipcp_ftth_nov_2009 = np.load('python_flows/dipcp_flows_FTTH_nov_2009.npy')
FTTH_nov_2009.create_dataset('dipcp', data=dipcp_ftth_nov_2009)


ADSL_dec_2009 = f.create_group("ADSL_Rennes_2009_12_14")
gvb_adsl_dec_2009 = np.load('python_flows/flows_marked_GVB_dec_2009_ADSL_BGP_AS.npy')
ADSL_dec_2009.create_dataset('GVB', data=gvb_adsl_dec_2009)
dipcp_adsl_dec_2009 = np.load('python_flows/dipcp_flows_ADSL_dec_2009.npy')
ADSL_dec_2009.create_dataset('dipcp', data=dipcp_adsl_dec_2009)


FTTH_dec_2009 = f.create_group("FTTH_Montsouris_2009_12_14")
gvb_ftth_dec_2009 = np.load('python_flows/flows_marked_GVB_dec_2009_FTTH_BGP_AS.npy')
FTTH_dec_2009.create_dataset('GVB', data=gvb_ftth_dec_2009)
dipcp_ftth_dec_2009 = np.load('python_flows/dipcp_flows_FTTH_dec_2009.npy')
FTTH_dec_2009.create_dataset('dipcp', data=dipcp_ftth_dec_2009)



