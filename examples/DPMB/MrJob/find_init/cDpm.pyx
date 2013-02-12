#cython: boundscheck=False
#cython: wraparound=False
#cython: profile=True
#cython infer_types=True

import numpy as np
cimport numpy as np
from libc.math cimport log, exp
from libcpp.vector cimport vector
from scipy import stats, special
#import helpers
cimport gsl

ctypedef unsigned int uint


cpdef int test(int x):
    y = x + 1
    return y

cdef class GSLState:
    cdef gsl.gsl_rng* rng

    def __cinit__(self):
        self.rng = gsl.gsl_rng_alloc(gsl.gsl_rng_taus)

    cpdef set_seed(self, uint seed):
        gsl.gsl_rng_set(self.rng, seed)

    cpdef int sample_discrete(self, w):
        cdef np.ndarray[double, ndim=1] w_c = np.ascontiguousarray(w)
        cdef gsl.gsl_ran_discrete_t* d=gsl.gsl_ran_discrete_preproc(w_c.shape[0], <double *>w_c.data)

        cdef uint idx = gsl.gsl_ran_discrete(self.rng, d)
        return idx

    cpdef double sample_uniform(self):
        return gsl.gsl_rng_uniform(self.rng)

rng_g = GSLState()
rng_g.set_seed(0)

cpdef np.ndarray calc_alpha_llh(np.ndarray[double, ndim=1] alpha, double shape, double scale, int n_clusters, int n):
    cdef uint n_alpha = alpha.shape[0]
    cdef np.ndarray[double, ndim=1] out = np.empty(n_alpha)
    cdef uint i
    cdef double prior, lh
    for i in range(n_alpha):
        prior = (shape-1)*log(alpha[i])+-alpha[i]/scale
        lh = gsl.gsl_sf_lngamma(alpha[i])+n_clusters*log(alpha[i])-gsl.gsl_sf_lngamma(alpha[i]+n)
        #lh = 0
        out[i] = prior+lh
    return out


cpdef inline uint discrete_sample(vector[double] w):
    cdef uint n=w.size()
    cdef np.ndarray[double, ndim=1] c = np.empty(n)
    cdef uint i
    c[0] = w[0]
    cdef double m
    for i in range(1, n):
        c[i] = c[i-1]+w[i]
        # if c[i-1] > w[i]:
        #     m = c[i-1]
        # else:
        #     m = w[i]
        # c[i] = log(exp(c[i-1]-m) + exp(w[i]-m))+m
    cdef double z= c[n-1]
    #cdef double r=log(rng.sample_uniform()) + z
    cdef double r=z*rng_g.sample_uniform()
    cdef int idx
    for i in range(n):
        if c[i]>r:
            idx = i
            break
    return idx

cdef double neg_inf = -np.inf
cdef vector[double] p
p.reserve(10000)

cdef  uint sample_c_idx(uint n_clusters, vector[int] &counts, np.ndarray[int, ndim=2] data,
                       vector[vector[int]] &n_succ, vector[vector[int]] &n_fail, double alpha,
                       double beta, uint i):
    cdef uint j, count, new_c_i, dim, d
    cdef double prior, llh, alpha_prime, beta_prime
    cdef int x_val
    p.clear()
    dim = data.shape[1]
    for j in range(n_clusters):
        count = counts[j]
        if count==0:
            p.push_back(neg_inf)
            continue
        prior = log(count)
        llh = 0
        for d in range(dim):
            x_val = data[i, d]
            alpha_prime = beta + n_succ[j][d]
            beta_prime = beta + n_fail[j][d]
            llh += log(alpha_prime/(alpha_prime+beta_prime)*x_val + beta_prime/(alpha_prime+beta_prime)*(1-x_val))
        #p[j] = prior + llh
        p.push_back(prior+llh)
    p.push_back(log(alpha) + log(.5)*dim)#llh_new_cluster(data, beta, beta, i)
    for j in range(n_clusters+1):
        p[j] = exp(p[j])
    new_c_i = discrete_sample(p)
    return new_c_i

cdef  uint sample_c(uint i, np.ndarray[int, ndim=1] c, np.ndarray[int, ndim=2] data, double alpha, double beta,
              vector[int] &counts, vector[vector[int]] &n_succ, vector[vector[int]] &n_fail):
    cdef uint n_clusters, new_c_i, j, count, d, dim
    cdef double prior, llh
    dim = data.shape[1]
    counts[c[i]] -= 1
    n_clusters = counts.size()
    for d in range(dim):
        n_succ[c[i]][d] -= data[i, d]
        n_fail[c[i]][d] -= (1-data[i, d])
    new_c_i = sample_c_idx(n_clusters, counts, data, n_succ, n_fail, alpha, beta, i)
    if new_c_i < n_clusters:
        counts[new_c_i] += 1
        for d in range(dim):
            n_succ[new_c_i][d] += data[i, d]
            n_fail[new_c_i][d] += (1-data[i, d])
    else:
        new_c_i = n_clusters
        counts.push_back(1)
        n_succ.push_back(data[i])
        n_fail.push_back(1-data[i])

    return new_c_i

cpdef np.ndarray sample_cs(np.ndarray[int, ndim=1] c, np.ndarray[int, ndim=2] data,
                           double alpha, double beta):
    cdef uint n_clusters, dim, i, n
    cdef np.ndarray[int, ndim=1] cluster_list, new_c
    cdef uint new_c_i, j, count, d
    cdef double prior, llh
    cdef vector[int] counts
    cdef vector[vector[int]] n_succ, n_fail
    n = len(c)
    cluster_list, new_c = np.unique(c, return_inverse=True)
    n_clusters = len(cluster_list)
    dim = data.shape[1]
    counts = np.bincount(new_c)
    #n_succ = np.empty((n_clusters, dim), int)
    #n_fail = np.empty((n_clusters, dim), int)
    for i in range(n_clusters):
        n_succ.push_back(np.sum(data[new_c==i]==1, 0))
        n_fail.push_back(np.sum(data[new_c==i]==0, 0))
    for i in range(n):
        new_c[i] = sample_c(i, new_c, data, alpha, beta, counts, n_succ, n_fail)
    return new_c

def set_seed(seed):
    rng_g.set_seed(seed)