#http://docs.cython.org/src/userguide/source_files_and_compilation.html#pyximport
# import pyximport
# pyximport.install()
# import pyx_functions as pf

import cython
import numpy as np
cimport numpy as np

def cluster_vector_joint_helper(np.ndarray[np.int32_t,ndim=1] data
                                ,np.ndarray[np.float64_t,ndim=1] column_sums
                                ,np.ndarray[np.float64_t,ndim=1] betas
                                ,int count,int num_els):
    
    cdef int idx
    cdef double curr_beta,curr_denominator,curr_column_sum,data_term
    data_term = 0
    for idx in range(num_els):
        curr_beta = betas[idx]
        curr_column_sum = column_sums[idx]
        curr_denominatnor = np.log(count + 2*curr_beta)
        if data[idx] == 0:
            curr_numerator = np.log(count - curr_column_sum + curr_beta)
        else:
            curr_numerator = np.log(curr_column_sum + curr_beta)
        data_term += curr_numerator - curr_denominator
    return data_term
