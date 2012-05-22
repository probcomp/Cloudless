#http://docs.cython.org/src/userguide/source_files_and_compilation.html#pyximport
# import pyximport
# pyximport.install()
# import pyx_functions as pf

# cython: profile=True
import cython
import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double log(double)
cdef extern from "math.h":
    double exp(double)
cdef extern from "math.h":
    double lgamma(double)

@cython.boundscheck(False)
cdef int renormalize_and_sample(
    np.ndarray[np.float64_t,ndim=1] conditionals
    ,double randv):
    #
    cdef np.ndarray[np.float64_t,ndim=1] scaled
    cdef np.ndarray[np.float64_t,ndim=1] p_vec
    cdef double maxv
    cdef double logZ
    cdef int draw
    #
    maxv = max(conditionals)
    scaled = conditionals - maxv
    logZ = reduce(np.logaddexp, scaled)
    p_vec = np.exp(scaled - logZ)    
    draw = 0
    while True:
        if randv < p_vec[draw]:
            return draw
        else:
            randv = randv - p_vec[draw]
            draw += 1

@cython.boundscheck(False)
def calculate_cluster_conditional(state,vector,randv):
    ##vector should be unassigned

    cdef np.ndarray[np.float64_t,ndim=1] betas = state.betas
    cdef np.ndarray[np.int32_t,ndim=1] data = np.array(vector.data)
    cdef int num_clusters = len(state.cluster_list)
    cdef int num_cols = len(betas)
    cdef int state_num_vectors = len(state.get_all_vectors())
    cdef double alpha = state.alpha
    cdef double score = state.score
    #
    cdef np.ndarray[np.float64_t,ndim=1] column_sums
    cdef int cluster_num_vectors
    cdef int cluster_idx
    cdef int column_idx
    cdef double alpha_term,data_term
    cdef double curr_beta,curr_denominator,curr_column_sum
    #
    conditionals = np.ndarray((num_clusters+1,),dtype=np.float64)
    for cluster_idx from 0 <= cluster_idx < num_clusters:
        # cluster_vector_joint
        cluster = state.cluster_list[cluster_idx]
        cluster_num_vectors = len(cluster.vector_list)
        column_sums = cluster.column_sums
        #
        alpha_term = log(cluster_num_vectors) - log(state_num_vectors-1+alpha)
        data_term = 0
        for column_idx from 0 <= column_idx < num_cols:
            if data[column_idx] == 0:
                    data_term += \
                        log( cluster_num_vectors \
                                 - column_sums[column_idx] \
                                 + betas[column_idx]) \
                        - log(cluster_num_vectors \
                                  + 2.0*betas[column_idx])
            else:
                    data_term += \
                    log(column_sums[column_idx] + betas[column_idx]) \
                        - log(cluster_num_vectors + 2.0*betas[column_idx])
        conditionals[cluster_idx] = score + alpha_term + data_term

    # last run for new cluster
    conditionals[num_clusters] = state.score \
        + log(alpha) - log(state_num_vectors-1+alpha) \
        + num_cols*log(.5)

    if randv is None:
        return conditionals
    else:
        return conditionals,renormalize_and_sample(conditionals,randv)

@cython.boundscheck(False)
def cluster_vector_joint(vector,cluster,state):

    cdef int num_cols = len(state.betas)
    cdef double alpha = state.alpha
    cdef int state_num_vectors = len(state.get_all_vectors())
    cdef int cluster_num_vectors = len(cluster.vector_list) \
        if cluster is not None else 0
    cdef double alpha_term,data_term

    if cluster_num_vectors == 0:
        alpha_term = log(alpha) - log(state_num_vectors-1+alpha)
        data_term = num_cols * log(.5)
        return alpha_term+data_term,alpha_term,data_term

    cdef np.ndarray[np.float64_t,ndim=1] betas = state.betas
    cdef np.ndarray[np.int32_t,ndim=1] data = np.array(vector.data)
    cdef np.ndarray[np.float64_t,ndim=1] column_sums = cluster.column_sums
    #
    alpha_term = log(cluster_num_vectors) - log(state_num_vectors-1+alpha)
    data_term = 0
    for column_idx from 0 <= column_idx < num_cols:
        if data[column_idx] == 0:
                data_term += \
                    log( cluster_num_vectors \
                             - column_sums[column_idx] \
                             + betas[column_idx]) \
                    - log(cluster_num_vectors \
                              + 2.0*betas[column_idx])
        else:
                data_term += \
                log(column_sums[column_idx] + betas[column_idx]) \
                    - log(cluster_num_vectors + 2.0*betas[column_idx])
    return alpha_term + data_term,alpha_term,data_term


@cython.boundscheck(False)
def calc_beta_conditional_helper(
    state
    ,np.ndarray[np.float64_t,ndim=1] grid
    ,int col_idx):

    cdef np.float64_t score = state.score
    cdef int num_clusters = len(state.cluster_list)
    cdef int grid_len = grid.shape[0]
    cdef double S,curr_score,test_beta
    cdef int N
    cdef np.ndarray ret_arr = np.zeros([1, grid_len], dtype=np.float64)
    cdef int grid_idx
    cdef int cluster_idx

    for grid_idx from 0 <= grid_idx < grid_len:
        test_beta = grid[grid_idx]
        curr_score = score
        for cluster_idx from 0 <= cluster_idx < num_clusters:
            S = state.cluster_list[cluster_idx].column_sums[col_idx]
            N = len(state.cluster_list[cluster_idx].vector_list)
            curr_score += lgamma(2*test_beta) \
                - 2*lgamma(test_beta) \
                + lgamma(S+test_beta) \
                + lgamma(N-S+test_beta) \
                - lgamma(N+2*test_beta)
        ret_arr[0,grid_idx] = curr_score
    return ret_arr

            
    
