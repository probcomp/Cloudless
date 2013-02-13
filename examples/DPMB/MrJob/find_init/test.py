from __future__ import division
import cDpm
from matplotlib.pylab import *
from numpy import *
import bino_dpm
import datasources
from pandas import *
import tables

def make_synthetic(n, n_clusters, alpha=1, beta=1, dim=256, seed=0):
    clusters = []
    rng = random.RandomState(seed)
    p = rng.beta(alpha, beta, size=(n_clusters, dim))
    for j in range(n_clusters):
        cluster = datasources.BinomialCluster(p[j])
        clusters.append(cluster)
    params = dict(n_points=n, clusters=clusters, weights=ones(n_clusters), seed=seed)
    fm = datasources.FiniteMixture(**params)
    fm._load_data()
    return fm

def load_tiny_images():
    f = tables.openFile('problem.h5')
    n = f.getNode('/xs')
    data = n[:]
    f.close()
    return data

def produce_df(sub_n, n_iters, alpha_set, times):
    t_cum=[]
    for i in range(1, len(times)):
        d = times[i]-times[i-1]
        t_cum.append(d.total_seconds())
    df = DataFrame(dict(n=sub_n, iters=n_iters, alpha=alpha_set[1:], t_cum=t_cum))
    df['t_per_iter'] = df.t_cum/df.iters
    df['t_per_n'] = df.t_per_iter/df.n
    df['k_mean'] = [bino_dpm.k_mean_from_alpha(alpha, 1000000) for alpha in df.alpha]
    return df
