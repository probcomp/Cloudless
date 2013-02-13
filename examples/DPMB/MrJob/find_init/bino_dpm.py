from __future__ import division
from numpy import *
import scaffold
import helpers
from helpers import discrete_sample
from scipy import special, stats
import logging
import cDpm
from matplotlib.pyplot import *
logging.basicConfig(level=logging.DEBUG)
import datetime

class State(scaffold.State):
    __slots__ = ['c', 'alpha', 'beta']


def sample_latent_dp(params, data_params, rng):
    s = State()
    n, dim = data_params['n'], data_params['dim']
    alpha = rng.gamma(params['alpha_shape'], params['alpha_scale'])
    if 'alpha_start' in params:
        alpha = params['alpha_start']
    s.alpha = alpha
    s.beta = rng.gamma(params['beta_shape'], params['beta_scale'])
    s.c = cDpm.sample_crp(n, alpha)
    return s


def sample_data_dp(state, params, dp, rng):
    n, dim = dp['n'], dp['dim']
    x = zeros((n, dim), int)
    cluster_list, cluster_ids = unique(state.c, return_inverse=True)
    n_clusters = len(cluster_list)
    beta = state.beta
    clusters = rng.beta(beta, beta, size=(n_clusters, dim))
    for j in range(n_clusters):
        in_j = cluster_ids==j
        x[in_j] = (rng.rand(sum(in_j), dim) < clusters[j]).astype(int)
    return x


def calc_alpha_llh(alpha, shape, scale, n_clusters, n):
    prior = stats.gamma.logpdf(alpha, shape, scale)
    lh = special.gammaln(alpha) + n_clusters * log(alpha) - special.gammaln(alpha + n)
    return prior + lh


def sample_alpha(n_clusters, n, shape, scale, grid):
    llh = cDpm.calc_alpha_llh(grid, shape, scale, n_clusters, n)
    #idx = cDpm.discrete_sample(llh, True, rng_g)
    idx = cDpm.discrete_sample(exp(llh-amax(llh))) #todo risk of overflow
    return grid[idx]


def sample_beta(shape, scale, rng):
    return rng.gamma(shape, scale)


class Chain(scaffold.Chain):
    def __init__(self, **kwargs):
        scaffold.Chain.__init__(self, **kwargs)
        self.alpha_grid = linspace(.01, 600, 4000) #todo: make adapative

    def start_state(self, params, dp, rng):
        return self.sample_latent(params, dp, rng)

    def sample_latent(self, params, dp, rng):
        return sample_latent_dp(params, dp, rng)

    def sample_data(self, state, params, dp, rng):
        return sample_data_dp(state, params, dp, rng)

    def transition(self, state, params, data, rng):
        s = State()
        s.c = cDpm.sample_cs(state.c, data, state.alpha, state.beta)
        s.alpha = sample_alpha(len(unique(s.c)), len(s.c), params['alpha_shape'],
                               params['alpha_scale'], self.alpha_grid)
        s.beta = sample_beta(params['beta_shape'], params['beta_scale'], rng)
        return s

chain = Chain(alpha_shape=5, alpha_scale=1, beta_shape=1, beta_scale=1, seed=1, max_iters=100)
dp = dict(n=50, dim=10)
g_funcs = [lambda state: state.alpha, lambda state: len(unique(state.c))]

def test(mode='geweke'):
    cDpm.set_seed(0)
    if mode!='geweke':
        s = chain.sample_state(dp)
        chain.data = s.data
        states = chain._run(dict(store_state=True))
        alpha = [_.alpha for _ in states]
        hist(alpha, 20)
        title("%r"%s.alpha)
    else:
        g = chain.geweke_test(2000, dp, g_funcs, 1000)
        scaffold.plot_test(g)
        return g


rng = random.RandomState(0)

def k_mean_from_alpha(alpha, n):
    return helpers.expected_tables(alpha, n)

def run(data, alpha_shape=5, alpha_scale=5, seed=0, sub_n=None, n_iters=None):
    run_params = dict(alpha_shape=alpha_shape, alpha_scale=alpha_scale, beta_shape=1, beta_scale=1, seed=seed)
    rng = random.RandomState(seed)
    cDpm.set_seed(seed)
    rng.shuffle(data)
    alpha = rng.gamma(alpha_shape, alpha_scale)
    n_data = len(data)
    if sub_n is None:
        sub_n = [100, 1000, 10000, 100000]
        n_iters = [100, 10, 5, 2]
    dim = data.shape[1]

    alpha_set = [alpha]
    times = []
    for j, (n_iter, n) in enumerate(zip(n_iters, sub_n)):
        times.append(datetime.datetime.now())
        logging.info("Running on subset  %d", n)
        params = run_params.copy()
        params['alpha_start'] = alpha
        s = chain.sample_latent(params, dict(n=n, dim=dim), rng)
        data_sub = data[:n]
        alpha_hist = empty(n_iter)
        n_iters_actual=zeros(len(n_iters), int)
        for i in range(n_iter):
            if i%1==0:
                logging.info("Iteration %d", i)
            try:
                s = chain.transition(s, chain.params, data_sub, rng)
            except KeyboardInterrupt:
                logging.info('breaking')
                break
            alpha_hist[i] = s.alpha
            n_iters_actual[j] += 1
        alpha = mean(alpha_hist[-1])
        alpha_set.append(alpha)
        if n > n_data:
            break
    times.append(datetime.datetime.now())
    return sub_n, n_iters_actual, alpha_set, times