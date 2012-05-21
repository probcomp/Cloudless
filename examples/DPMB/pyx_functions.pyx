#http://docs.cython.org/src/userguide/source_files_and_compilation.html#pyximport
# import pyximport
# pyximport.install()
# import pyx_functions as pf

import cython
import numpy as np
cimport numpy as np

def cluster_vector_joint(vector,cluster,state):
    try_new = True
    alpha = state.alpha
    numVectors = len(state.get_all_vectors())
    if cluster is None or cluster.count() == 0:
        ##if the cluster would be empty without the vector, then its a special case
        alpha_term = np.log(alpha) - np.log(numVectors-1+alpha)
        data_term = state.num_cols*np.log(.5)
    else:
        alpha_term = np.log(cluster.count()) - np.log(numVectors-1+alpha)
        if not try_new:
            boolIdx = np.array(vector.data,dtype=type(True))
            numerator1 = boolIdx * np.log(cluster.column_sums + state.betas)
            numerator2 = (~boolIdx) * np.log(
                cluster.count() - cluster.column_sums + state.betas)
            denominator = np.log(cluster.count() + 2*state.betas)
            data_term = (numerator1 + numerator2 - denominator).sum()
        else:
            data_term = cluster_vector_joint_helper(
                np.array(vector.data)
                ,np.array(cluster.column_sums)
                ,np.array(state.betas)
                ,cluster.count(),len(state.betas))
    retVal = alpha_term + data_term
    return retVal,alpha_term,data_term

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
