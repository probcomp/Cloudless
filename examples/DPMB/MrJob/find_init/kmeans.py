from __future__ import division
from numpy import *
from scipy import spatial
from collections import namedtuple
import helpers
from matplotlib.pyplot import *
import matplotlib as mpl
from datasources import FiniteMixture, GaussianCluster

rng = random


def kmeans_pp_seeds(data, k):
    n = len(data)
    seeds = empty(k, int)
    seeds[0] = rng.randint(n)
    for i in range(1, k):
        d = spatial.distance.cdist(data, data[seeds[:i]], 'sqeuclidean')
        d_closest = amin(d, 1)
        new_seed = helpers.discrete_sample(d_closest, rng=rng)
        seeds[i] = new_seed
    return data[seeds]


Clustering = namedtuple('Clustering', ['clusters', 'c', 'data', 'k'])


def cluster(data, k):
    clusters = kmeans_pp_seeds(data, k)
    c = zeros(k, int)
    while True:
        c_old = copy(c)
        d = spatial.distance.cdist(data, clusters, 'sqeuclidean')
        c = argmin(d, 1)
        for i in range(k):
            clusters[i] = mean(data[c == i], 0)
        if all(c == c_old):
            break
    return Clustering(clusters=clusters, c=c, data=data, k=k)


def show_clustering(cl):
    colors = mpl.rcParams['axes.color_cycle']
    for i in range(cl.k):
        scatter(cl.data[cl.c == i, 0], cl.data[cl.c == i, 1], c=colors[i], edgecolor='None')
        hold(True)
        scatter(cl.clusters[i, 0], cl.clusters[i, 1], s=100, c=colors[i], marker='x')
    grid()


def test():
    c1 = GaussianCluster([3, 3], eye(2))
    c2 = GaussianCluster([-1, -1], eye(2) * .5)
    fm = FiniteMixture()
    data = fm.load(dict(clusters=[c1, c2], weights=[.5, .5], n_points=20), rng)
    cl = cluster(data, 2)
    show_clustering(cl)



