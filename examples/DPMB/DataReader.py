import os
#
import h5py
import numpy
#
import Cloudless.examples.DPMB.h5_functions as h5
import Cloudless.examples.DPMB.remote_functions as rf
import Cloudless.examples.DPMB.helper_functions as hf


class DataReader:

    def __init__(self, filename, global_data_indices, dir='', cache_size=100000):
        print "__init__"
        #
        hdf5_filename = h5.get_h5_name_from_pkl_name(filename)
        hdf5_full_filename = os.path.join(dir, hdf5_filename)
        f = h5py.File(hdf5_full_filename, 'r')
        #
        self.cache = dict()
        self.cache_size = cache_size
        self.f = f
        self.global_data_indices = numpy.array(global_data_indices)
        self.masked_indices = numpy.array(global_data_indices)

    def __del__(self):
        self.f.close()
        print "__del__"

    def set_mask_ordering(self, mask_ordering):
        global_data_indices = self.global_data_indices
        #
        masked_indices = global_data_indices[mask_ordering]
        print masked_indices
        #
        self.masked_indices = masked_indices

    def get_indices_to_cache(self, local_missed_index):
        cache_size = self.cache_size
        masked_indices = self.masked_indices
        #
        base = (local_missed_index / cache_size) * cache_size
        end = base + cache_size
        local_indices = range(base, end)
        global_indices = masked_indices[base:end]
        #
        return local_indices, global_indices

    def get_global_indices_from_f(self, global_indices_to_get):
        f = self.f
        #
        sorted_indices = numpy.sort(global_indices_to_get)
        inverse_indices = numpy.argsort(numpy.argsort(global_indices_to_get))
        # return f['xs'].value[sorted_indices][inverse_indices]
        # FIXME: REVERT
        return f['xs'].value[sorted_indices][inverse_indices][:,:6]

    def cache_global_indices(self, local_indices, global_indices_to_cache):
        xs = self.get_global_indices_from_f(global_indices_to_cache)
        cache = dict(zip(local_indices, xs))
        #
        self.cache = cache

    def get_from_cache(self, idx):
        cache = self.cache
        #
        if idx not in cache:
            local_indices, global_indices = self.get_indices_to_cache(idx)
            self.cache_global_indices(local_indices, global_indices)
            cache = self.cache
        #
        return cache[idx]

    def __getitem__(self, idx, verbose=False):
        if numpy.isscalar(idx):
            with hf.Timer('get_from_cache', verbose=verbose):
                ret = self.get_from_cache(idx)
        else:
            with hf.Timer('get_global_indices_from_f', verbose=verbose):
                ret = self.get_global_indices_from_f(idx)
        return ret

if __name__ == '__main__':
    # filename = 'tiny_image_problem_nImages_320000_nPcaTrain_10000.pkl.gz'
    filename = 'tiny_image_problem_nImages_1000000_nPcaTrain_400000.pkl.gz'
    dir = 'Data'
    num_values = 10
    #
    if 'dr' in locals(): del dr
    dr = DataReader(filename, range(num_values), dir=dir)
    print '\n'.join(map(str, zip(range(10), dr[numpy.arange(0,10)])))
    numpy.random.seed(0)
    permutation = numpy.random.permutation(range(10))
    dr.set_mask_ordering(permutation)
    for idx in range(10):
        print (idx, dr[idx])
