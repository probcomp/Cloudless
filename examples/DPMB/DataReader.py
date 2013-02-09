import os
#
import h5py
import numpy
#
import Cloudless.examples.DPMB.h5_functions as h5
# hadoop chokes on this import
# import matplotlib.mlab


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
        self.local_access_ordering = numpy.arange(len(global_data_indices))

    def __del__(self):
        self.f.close()
        print "__del__"

    def __len__(self):
        return len(self.global_data_indices)

    def set_local_access_ordering(self, local_access_ordering):
        self.local_access_ordering = local_access_ordering

    def get_indices_to_cache(self, local_missed_index):
        cache_size = self.cache_size
        local_access_ordering = self.local_access_ordering
        global_data_indices = self.global_data_indices
        #
        which_local_missed_index = numpy.nonzero(local_access_ordering==local_missed_index)[0][0]
        start = (which_local_missed_index / cache_size) * cache_size
        end = start + cache_size
        local_indices = local_access_ordering[start:end]
        global_indices = global_data_indices[local_indices]
        #
        return local_indices, global_indices

    def get_global_indices_from_f(self, global_indices_to_get):
        f = self.f
        #
        sorted_indices = numpy.sort(global_indices_to_get)
        inverse_indices = numpy.argsort(numpy.argsort(global_indices_to_get))
        return f['xs'].value[sorted_indices][inverse_indices]

    def cache_global_indices(self, local_indices, global_indices_to_cache):
        xs = self.get_global_indices_from_f(global_indices_to_cache)
        cache = dict(zip(local_indices, xs))
        #
        self.cache = cache

    def get_from_cache(self, idx):
        cache = self.cache
        #
        if idx not in cache:
            print 'cache miss'
            local_indices, global_indices = self.get_indices_to_cache(idx)
            self.cache_global_indices(local_indices, global_indices)
            cache = self.cache
        #
        return cache[idx]

    def __getitem__(self, idx):
        if numpy.isscalar(idx):
            ret = self.get_from_cache(idx)
        else:
            ret = self.get_global_indices_from_f(idx)
        return ret

if __name__ == '__main__':
    # filename = 'tiny_image_problem_nImages_320000_nPcaTrain_10000.pkl.gz'
    filename = 'tiny_image_problem_nImages_1000000_nPcaTrain_400000.pkl.gz'
    dir = 'Data'
    num_values = 100
    cache_size = num_values/10
    #
    if 'dr' in locals(): del dr
    dr = DataReader(filename, range(num_values), dir=dir, cache_size=cache_size)
    numpy.random.seed(0)
    permutation = numpy.random.permutation(num_values)
    dr.set_local_access_ordering(permutation)
    print 'dr.local_access_ordering:', dr.local_access_ordering
    for permutation_idx in range(cache_size-5, cache_size+5):
        data_idx = permutation[permutation_idx]
        print permutation_idx, data_idx, dr[data_idx][:6]
