#http://docs.cython.org/src/userguide/source_files_and_compilation.html#pyximport
# import pyximport
# pyximport.install()
# import pyx_functions as pf

import cython
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double log(double)

def cluster_vector_joint_helper(np.ndarray[np.int32_t,ndim=1] data
                                ,np.ndarray[np.float64_t,ndim=1] column_sums
                                ,np.ndarray[np.float64_t,ndim=1] betas
                                ,int count,int num_els):
    
    cdef int idx
    cdef double curr_beta,curr_denominator,curr_column_sum,data_term
    data_term = 0
    for idx in range(num_els):

        if data[idx] == 0:
                data_term += log(count - column_sums[idx] + betas[idx]) - log(count + 2.0*betas[idx])
        else:
                data_term += log(column_sums[idx] + betas[idx]) - log(count + 2.0*betas[idx])
    return data_term
