"""Retrieve As information out of GeoIP (MaxMind) binding.
Beware: database location is hardcoded!"""

import GeoIP
import re
import numpy as np
import INDEX_VALUES

#WARNING: hard coded
GAS=GeoIP.open('/home/louis/streaming/flows/AS/GeoIPASNum.dat',
	       GeoIP.GEOIP_STANDARD)

#ugly but more efficient: compile only once
REGEXP = re.compile('(AS([0-9]+).*)')

def extend_fields_AS_down(d):
    "Extend each line of array considered as list with src IP addresses."
    fields = list(d)
    src = GAS.org_by_addr(d['srcAddr'])
    if src != None:
        fields.extend(list(REGEXP.match(src).group(2,1)))
    else:
        fields.extend([0, 'Not found'])
    return tuple(fields)

def extend_array_AS_down(flows_array):
    "Return a new array with AS information upstream."
    return np.array([extend_fields_AS_down(d) for d in flows_array],
		    dtype=INDEX_VALUES.dtype_GVB_AS_down)


def extend_fields_AS(d):
    "Extend each line of array considered as list with both IP addresses."
    fields = list(d)
    src = GAS.org_by_addr(d['srcAddr'])
    if src != None:
        fields.extend(list(REGEXP.match(src).group(2,1)))
    else:
        fields.extend([0, 'Not found'])
    dst = GAS.org_by_addr(d['dstAddr'])
    if dst != None:
        fields.extend(list(REGEXP.match(dst).group(2,1)))
    else:
        fields.extend([0, 'Not found'])
    return tuple(fields)

def extend_array_AS(flows_array):
    "Return a new array with AS information on both sides."
    return np.array([extend_fields_AS(d) for d in flows_array],
		    dtype=INDEX_VALUES.dtype_GVB_AS)

def extend_array_BGP_AS(flows_array):
    "Return a new array with AS information on both sides."
    return np.array([extend_fields_AS(d) for d in flows_array],
		    dtype=INDEX_VALUES.dtype_GVB_BGP_AS)



#test_flows=np.loadtxt('test/flows_ftth_nov.head',
#dtype=INDEX_VALUES.dtype_GVB,skiprows=1).view(np.recarray)

#np.array(zip(test_flows,[[GAS.org_by_addr(src),GAS.org_by_addr(dst)]
#for src,dst in zip(test_flows.srcAddr,test_flows.dstAddr)]))
#[(f, GAS.org_by_addr(f['srcAddr']), GAS.org_by_addr(f['dstAddr']))
#for f in test_flows]
