"Module to load hd5 files"

import numpy as np
import h5py


def load_h5_dataset(dataset):
    "Retruns a numpy array of the hdf5 dataset."
    temp = np.zeros(dtype=dataset.dtype, shape=dataset.shape)
    temp = dataset[...]
    return temp

def load_h5_datasets(h5py_file, fullpath=True):
    "Returns a dataset dictonary with all datasets of hdf5 file, \
    and print group attributes."
    datasets = {}
    for group in h5py_file.values():
        print 'Loading ' + group.name.strip('/') + ' with attributes:'
        for name, value in group.attrs.iteritems():
            print "\t" + name + ":", value
        for dataset in group.values():
            if fullpath:
                data_name = dataset.name.strip('/').replace('/','_')
            else:
                data_name = dataset.name.split('/')[-1]
            datasets[data_name] = load_h5_dataset(dataset)
    return datasets

def load_h5_file(hdf5_file='/home/louis/streaming/hdf5/lzf_data.h5',
        fullpath=True):
    "Returns a dataset dictonary contained in the hdf5 file"
    h5py_file = h5py.File(hdf5_file, 'r')
    return load_h5_datasets(h5py_file, fullpath=fullpath)
