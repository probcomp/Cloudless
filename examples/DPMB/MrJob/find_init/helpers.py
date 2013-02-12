"""
A set of utility functions used in various modules.
"""

from __future__ import division
from numpy import *
import logging
import cStringIO
import matplotlib.pylab as plt
import tempfile
import subprocess
from scipy import stats

logger = logging.getLogger('scaffold')
[logger.removeHandler(h) for h in logger.handlers] #for handling module reloading
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s: %(message)s',
                              datefmt='%H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)


class ParameterException(BaseException):
    """
    Exception type for when an expected key is missing from the parameter dictionary of a parameterized algorithm
    """
    pass


def discrete_sample(w, n=1, rng=random, log_mode=False, flatten=True):
    """
    Sample from a general  discrete distribution.

    :param w: A list of weights of each discrete outcome. Does not need to be normalized.
    :param  n: The number of samples to return.
    :param rng: The random number generator to use (e.g. as returned by *random.RandomState*)
    :param log_mode: If *True*, interpret *w* as the log of the true weights. If *False*, interpret *w* as the literal weights. Default *False*.
    :param flatten: If *True* and n=1, returns a scalar instead of an array. Default *True*.


    :return: A list of *n* integers, corresponding to the indices of *w* that were chosen.
    """
    w = asarray(w, 'd')
    #seterr(over='raise', under='raise')
    if log_mode:
        c = logaddexp.accumulate(w)
        c -= c[-1]
        r = log(rng.rand(n))
        value = searchsorted(c, r)
    else:
        c = cumsum(w)
        c /= c[-1]
        r = rng.rand(n)
        value = searchsorted(c, r)
    if n == 1 and flatten:
        return value[0]
    else:
        return value

def save_fig_to_str():
    """
    Returns a string representing the current figure, in PDF format. Useful for creating a figure on a remote process and marshaling it back to the client.

    Example::

        plot([1,2], [3,4])
        s = save_fig_to_str()
        f = open('myfile.pdf', 'wb')
        f.write(s)


    :return: A string of bytes in PDF format.
    """
    buffer = cStringIO.StringIO()
    plt.savefig(buffer, format='pdf')
    return buffer.getvalue()


def show_fig(fig):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False)
    f.write(fig)
    f.close()
    subprocess.call(['open', f.name]) #todo: only works on OS X


class frozendict(dict):
    """
    A hashable dictionary which can be used as the key to other dictionaries
    """
    def get_key(self):
        d = []
        for k in sorted(self.keys()):
            if isinstance(self[k], list):
                d.append((k, self[k]))
            else:
                d.append((k, self[k]))
        return tuple(d)

    def __hash__(self):
        return hash(self.get_key())


class circlelist(list):
    """
    A list with modular arithmetic indexing semantics.

    Example::

       l = circlelist([1, 2, 3])
       print l[10]

    will print l[10%3]=2
    """
    def __getitem__(self, idx):
        return list.__getitem__(self, idx % len(self))

def estimate_mcmc_var(g, bin_size=10):
    """
    Estimate the variance of a population parameter, given autocorrelated samples of the parameter

    :param g: Numpy 1d array
    :return: Scalar variance estimate
    """

    n = len(g)
    n_burn = int(.2*n)
    g_steady = g[n_burn:]
    n = len(g_steady)
    #todo: the  bin size should be automatically calculated
    bins = array_split(g_steady, int(n/bin_size))
    means = [mean(bin) for bin in bins]
    v = var(means)
    return v/len(bins)

def qq_plot(s1, s2):
    """
    Produce a quantile-quantile plot

    :param s1: A one-dimensional series
    :param s2: A one-dimensional series
    :return:
    """
    p = linspace(.1, 99, 100)
    q1 = [stats.scoreatpercentile(s1, _) for _ in p]
    q2 = [stats.scoreatpercentile(s2, _) for _ in p]
    plt.scatter(q1, q2, edgecolor='None')
    plt.grid()
    a = plt.axis()
    lower = min([a[0], a[2]])
    upper = max([a[1], a[3]])
    return plt.plot([lower, upper], [lower, upper], color='red')

def expected_tables(alpha, n):
    i = arange(n)
    return sum(alpha/(alpha+i))

def bar_chart(x):
    plt.bar(range(len(x)), x, align='center')
    plt.grid()

