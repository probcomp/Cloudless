from __future__ import division
from numpy import *
import bino_dpm
import numpy as np
import kmeans

def run(data_reader, **kwargs):
    seed = kwargs.get('seed', 0)
    rng = random.RandomState(seed)
    n_max = len(data_reader)
    n = int(.1*n_max)
    permutation = rng.permutation(n_max)
    data_reader.set_local_access_ordering(permutation)
    data = empty((n, 256), np.int8)
    for i in range(n):
        data[i] = data_reader[i]
    sub_n, n_iters, alpha_set, time = bino_dpm.run(data, **kwargs)
    return alpha_set[-1]

def kmeans(data, k):
    return kmeans.cluster(data, k)
