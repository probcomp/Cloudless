from __future__ import division
from numpy import *
import bino_dpm

rng = random.RandomState(0)

def run(data_reader):
    n_max = len(data_reader)
    n = int(.1*n_max)
    permutation = rng.permutation(n_max)
    data_reader.set_local_access_ordering(permutation)
    data = empty((n, 256), int)
    for i in range(n):
        data[i] = data_reader[i]
    alpha = bino_dpm.run(data)
    return alpha
