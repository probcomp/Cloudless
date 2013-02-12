"""
Implementation of some common procedurally-generated datasets
"""

from __future__ import division
from matplotlib.patches import Ellipse
from matplotlib.pylab import *
from numpy import asarray, random, empty
from scaffold import ProceduralDataSource, ParameterException
import helpers


class GaussianCluster:
    """
    A Gaussian cluster parameterized by a mean vector and covariance matrix
    """

    def __init__(self, mu=None, cov=None):
        self.mu = None
        self.cov = None
        if mu is not None:
            self.mu = asarray(mu, 'd')
        if cov is not None:
            self.cov = asarray(cov, 'd')

    dim = property(lambda self: len(self.mu))

    def sample_points(self, n, rng=random):
        """
        Sample points from the cluster

        :param n: The number of points
        :param rng: The random-number generator
        :type n: int
        :return: An *n* x *dim* array. Each row is a point; each column is a dimension.
        """
        return rng.multivariate_normal(self.mu, self.cov, size=int(n))

    def __hash__(self):
        return hash(str(self.mu) + str(self.cov))

    def __str__(self):
        return str(self.mu) + "," + str(self.cov)




class FiniteMixture(ProceduralDataSource):
    """
    Implements a finite mixture model.
    """

    def __init__(self, **kwargs):
        super(FiniteMixture, self).__init__(**kwargs)

    def load(self, params, rng):
        """
        Loads the latent variables and data

        Expected parameter keys:

        n_points
         Number of points in the dataset

        clusters
         A list of clusters of type :py:class:`Cluster`

        weights
         A list of mixing weights for each cluster in *clusters*
        """
        try:
            n_points = params['n_points']
            clusters = params['clusters']
            weights = asarray(params['weights'])
            self.clusters = clusters
        except KeyError as error:
            raise ParameterException("Required finite mixture parameter not passed in: %r" % error)
        dim = clusters[0].dim
        self.c = helpers.discrete_sample(weights, n_points, rng)
        data = empty((n_points, dim))
        for i, cluster in enumerate(clusters):
            idx = self.c == i
            n_in_cluster = int(sum(idx))
            data[idx] = cluster.sample_points(n_in_cluster, rng)
        return data

    def points_in_cluster(self, c):
        return self.data[self.c == c]

    dim = property(lambda self: len(self.clusters[0].dim))

    def show(self):
        colors = helpers.circlelist(['red', 'blue', 'orange', 'green', 'yellow'])
        for c in range(len(self.clusters)):
            cluster = self.clusters[c]
            x = self.points_in_cluster(c)
            scatter(x[:, 0], x[:, 1], color=colors[c])
            width = cluster.cov[0, 0] * 2
            height = cluster.cov[1, 1] * 2
            e = Ellipse(cluster.mu, width, height, alpha=.5, color=colors[c])
            gca().add_artist(e)

    def llh_pred(self, x):
        pass #todo: implement this


class EmptyData(ProceduralDataSource):
    def __init__(self, **kwargs):
        super(EmptyData, self).__init__(**kwargs)

    def load_data(self):
        self.data = empty(0)


class BinomialCluster:
    """
    A cluster of binary data. Each dimension has a separate probability of being True.

     Parameterized by a vector 'p', where p[d] is the probability that the 'd' coordinate is true.
    """
    def __init__(self, p=None):
        self.p = asarray(p, float)

    dim = property(lambda self: len(self.p))

    def sample_points(self, n, rng=random):
        x = empty((n, self.dim), int)
        for i in range(n):
            x[i] = rng.rand(self.dim) < self.p
        return x

    def __hash__(self):
        return hash(str(self.p))