from __future__ import division
from numpy import *
from scipy import spatial
from collections import namedtuple
import helpers
from matplotlib.pyplot import *
import matplotlib as mpl
from datasources import FiniteMixture, GaussianCluster
import itertools
import logging
import cDpm
rng = random
distance_metric='hamming'


def kmeans_pp_seeds(data, k):
    n = len(data)
    seeds = empty(k, int)
    seeds[0] = rng.randint(n)
    d_closest = empty(n, np.double)
    for i in range(1, k):
        #d = spatial.distance.cdist(data, data[seeds[:i]], distance_metric)
        #d_closest = amin(d, 1)
        cDpm.find_closest_a(data, data[seeds[:i]].astype(np.double), d_closest)
        new_seed = helpers.discrete_sample(d_closest, rng=rng)
        seeds[i] = new_seed
    return data[seeds]


Clustering = namedtuple('Clustering', ['clusters', 'c', 'data', 'k'])


def cluster(data, k):
    clusters = kmeans_pp_seeds(data, k).astype(np.double)
    c = zeros(len(data), np.int)
    for iter in itertools.count():
        c_old = copy(c)
        #d = spatial.distance.cdist(data, clusters, distance_metric)
        #c = argmin(d, 1)
        logging.info("Iteration %d" % iter)
        cDpm.find_closest_arg(data, clusters, c)
        for i in range(k):
            clusters[i] = mean(data[c == i].astype(np.double), 0)
        n_changed = sum(c!=c_old)
        if n_changed==0:
            break
        logging.info("Num changed: %d" % n_changed)
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
    c2 = GaussianCluster([-1, -1], eye(2) * .5) #needs to change to boolean
    fm = FiniteMixture()
    data = fm.load(dict(clusters=[c1, c2], weights=[.5, .5], n_points=20), rng)
    cl = cluster(data, 2)
    show_clustering(cl)
