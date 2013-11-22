"""Defines functions to aggregate arrays.
Last function aggregate should be generic.
"""

from __future__ import division, print_function
import matplotlib.mlab
import numpy as np

from collections import defaultdict
from itertools import izip

def aggregate_mean(data, idx_agg, idx_field):
    result = {}
    n = {}
    for d in data:
        agg = d[idx_agg]
        result[agg] = result.get(agg, 0) + d[idx_field]
        n[agg] = n.get(agg, 0) + 1
    out = []
    for agg in n.keys():
        out.append(result[agg]/n[agg])
    return out

def aggregate_sum(data, idx_agg, idx_field):
    result = defaultdict(int)
    for d in data:
        agg = d[idx_agg]
        result[agg] += d[idx_field]
    return zip(result.keys(), result.values())


#def aggregate_sum_nb(data, key_name, field_name):
#    result = {}
#    for d in np.unique(data[key_name]):
#        tmp = [s[field_name] for s in data if s[key_name] == d]
#        result[d]=(sum(tmp), len(tmp))
#    return zip(result.keys(), result.values())

def aggregate_sum_nb(data, key, field):
    "Returns a dict of nb of occurences and sum of data aggregated with key"
    agg_sum = defaultdict(int)
    agg_nb = defaultdict(int)
    for row in data:
        d = row[key]
        agg_nb[d] += 1
        agg_sum[d] += row[field]
    return dict(izip(agg_nb.keys(), izip(agg_nb.values(), agg_sum.values())))



def aggregate_nb(data, key_name):
    #return [(d, len([x for x in data if x[key_name] == d])) for d in
    #    np.unique(data[key_name])]
    result = defaultdict(int)
    for value in data[key_name]:
        result[value] += 1
    return result

def get_sample(data, key, searched_value):
    "Return first item on selected field value."
    for value in  data:
        if value[key] == searched_value:
            return value



def aggregate_min_max(data, key_name, field_name):
    first = {}
    last = {}
    result = {}
    max_value = max(data.field_name)
    min_value = min(data.field_name)
    for d in data:
        agg = d.key_name
        #result item stores (min values, max values)
        prev = result.get(agg, (max_value, min_value))
        result[agg] = (min(prev[0], d.field_name),
		       max(prev[1], d.field_name))
    return zip(result.keys(), result.values())

def aggregate(data, key, value, func):
    "Aggregate data.value according to data.key \
    by applying func to each key."
    if len(data) > 0:
        return matplotlib.mlab.rec_groupby(data, (key,),
				       ((value, func ,'aggregation'),))
    else:
        # TODO: check dtype
        return np.array([],
               dtype=[(key, data.dtype[key]),
                   ('aggregation', data.dtype[value])])

#    data_per_key = {}
#    for k,v in zip(data[key], data[value]):
#        if k not in data_per_key.keys():
#            data_per_key[k]=[]
#        data_per_key[k].append(v)
#    return [(k,func(data_per_key[k])) for k in data_per_key.keys()]

