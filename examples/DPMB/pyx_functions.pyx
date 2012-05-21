#http://docs.cython.org/src/userguide/source_files_and_compilation.html#pyximport
# import pyximport
# pyximport.install()
# import pyx_functions as pf

import cython
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double log(double)
cdef extern from "math.h":
    double lgamma(double)

def cluster_vector_joint_helper(np.ndarray[np.int32_t,ndim=1] data
                                ,np.ndarray[np.float64_t,ndim=1] column_sums
                                ,np.ndarray[np.float64_t,ndim=1] betas
                                ,int count,int num_els):
    
    cdef int idx
    cdef double curr_beta,curr_denominator,curr_column_sum,data_term
    data_term = 0
    for idx in range(num_els):

        if data[idx] == 0:
                data_term += log(count - column_sums[idx] + betas[idx]) \
                    - log(count + 2.0*betas[idx])
        else:
                data_term += log(column_sums[idx] + betas[idx]) \
                    - log(count + 2.0*betas[idx])
    return data_term

def calc_beta_conditional_helper(np.ndarray[np.float64_t,ndim=1] S_list
                                 ,np.ndarray[np.float64_t,ndim=1] R_list
                                 ,np.ndarray[np.float64_t,ndim=1] grid
                                 ,np.float64_t score
                                 ):
    cdef int list_len = S_list.shape[0]
    cdef int grid_len = grid.shape[0]
    cdef double s,r,curr_score
    cdef np.ndarray ret_arr = np.zeros([1, grid_len], dtype=np.float64)
    print ret_arr
    for grid_idx,test_beta in enumerate(grid):
        curr_score = score
        for S,R in zip(S_list,R_list):
            curr_score += lgamma(2*test_beta) \
                - 2*lgamma(test_beta) \
                + lgamma(S+test_beta) \
                + lgamma(R+test_beta) \
                - lgamma(S+R+2*test_beta)
        ret_arr[0,grid_idx] = curr_score
    return ret_arr

            
    
