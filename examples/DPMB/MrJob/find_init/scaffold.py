"""
Classes for representing the state and operators of an iterative algorithm.

These classes are meant to be inherited from as needed.
"""
from __future__ import division
import time
import itertools
from copy import deepcopy
from numpy import *
from helpers import ParameterException
import helpers
import abc
from scipy import stats
from collections import namedtuple
import matplotlib.pyplot as plt

#todo: add underscore to methods meant for internal use
#todo: clean up API for follow the prior

class JLogger:
    """
    Hack because picloud is complaining about pickling the standard python logger
    """

    def debug(self, str):
        print str

    def info(self, str):
        print str


logger = JLogger()


class State(object):
    """
    Represents all state variables of the algorithm at a particular iteration.

    Derived classes must use slots for storing their instance variables, rather than relying on :py:attr:`self.__dict__`.

    At a minimum, a state object will have the following attributes:

    iter
     The iteration number of the algorithm that this state corresponds to. State 0 corresponds to the initial state
     of the algorithm, before any transitions have been applied. The last state is the state that
     caused :py:meth:`Chain.do_stop` to return *True*.

    time
     The time (in seconds since epoch) that this state was created. Mainly used to assess runtime of algorithms.

    data
      A numpy array of the data associated with the state. Only used for Geweke testing, when the data is different on each iteration. In other cases, data is stored only once in the :py:class:`Chain` object.
    """

    __slots__ = ['iter', 'time', 'data']
    # Slots are used for memory efficiency
    # The 'data' slot is only used for follow-the-prior testing

    def __init__(self):
        self.iter, self.time, self.data = None, None, None

    def summarize(self):
        """
        Perform any work on computing summary statistics or visualizations of this iteration.
        Typically executing at end of an MCMC run.

        Main purpose is to allow for remote computation of state summary, rather than having the state pulled
        back to the client and then having the client create visualizations.

        **Warning**: Should depend *only* on the instance variables defined in the state object.
        """
        pass

    def __getstate__(self):
        d = dict(iter=self.iter, time=self.time, data=self.data)
        state = {}
        for k in self.__slots__:
            state[k] = getattr(self, k)
        d.update(state)
        return d

    def __setstate__(self, state):
        for k, v in state.iteritems():
            setattr(self, k, v)

    def sample_data(self, n_data):
        pass


registry = {}


class RegisteredClass(abc.ABCMeta):
    def __init__(cls, name, bases, attrs):
        abc.ABCMeta.__init__(cls, name, bases, attrs)
        if name in registry:
            logger.debug("Name conflict in registered class")
        registry[name] = cls


GewekeTest = namedtuple('GewekeTest', ['p', 'z', 'cs', 'mc', 'g'])


def plot_test(results):
    """
    Plots the results of a Geweke test.
    :param results: The geweke test results, as returned by :py:meth:`Chain.geweke_test`.
    :raise:
    """
    if True:
    #if isinstance(results, GewekeTest):
        g = results.g
        n_tests = g.shape[0]
        fig, plts = plt.subplots(3, n_tests, squeeze=False)
        for i in range(n_tests):
            plt.sca(plts[0, i])
            g1 = g[i, :, 0]
            g2 = g[i, :, 1]
            helpers.qq_plot(g1, g2)
            plt.xlabel('Ground')
            plt.ylabel('Inferred')
            plt.sca(plts[1, i])
            plt.hist(g1)
            plt.title('Ground')
            plt.sca(plts[2, i])
            plt.hist(g2)
            plt.title('Inferred')
        fig.subplots_adjust(hspace=1)

    else:
        raise BaseException("Result not recognized")


class Chain(object):
    """
    Provides the actual implementation of a Markovian  algorithm. When you subclass this, you should at least implement
    *transition* and *start_state*. You can optionally also implement *sample_data* and *sample_latent* to enable
    Geweke testing. You can optionally implement *do_stop* to implement a custom stopping rule.

     Each of the methods you override takes an argument calls *params*. This stands for 'parameters' and is a Python dictionary containing the
     parameters you passed to the Experiment object. They also take a method called *rng*. This is a numpy random number generator object,
     created based on the method seed of the experiment. When generating random values, you should only use the *rng* object. This guarentees your experiments will be reproducible.

     Some of the methods take a *data_params* object. This object contains the number of datapoints and the dimensionality of the data the chain is being run (in the 'n' key and 'dim' key, respectively).
    """

    __metaclass__ = RegisteredClass

    def __init__(self, **kwargs):
        """
        :param kwargs: A set of parameters controlling the inference algorithm. Keys:

        seed (Default 0)
         The random seed used for the iterative algorithm

        max_runtime (Default 1800 seconds)
         The maximum amount of time the algorithm will be allowed to run before being terminated (in seconds)

        max_iters (Default 1000)
         The maximum number of iterations the algorithm will be allowed to run before being terminated.

        All other keys are passed as the *params* argument to the derived class.
        """
        self.params = kwargs
        self.seed = kwargs.get('seed', 0)
        self.max_runtime = kwargs.get('max_runtime', 60 * 30)
        self.max_iters = kwargs.get('max_iters', 1000)
        self.rng = random.RandomState(self.seed)
        self.data = None
        self.follow_prior = kwargs.get('follow_prior', False)
        self.start_time = None
        self.end_time = None


    @abc.abstractmethod
    def transition(self, state, params, data, rng):
        """
        Implementation of the transition operator. Expected to be implemented in a user-derived subclass.

        :param state: The current state of the Markov algorithm

        :param params: A dict of the parameters of the chain

        :param data: A numpy array representing the data

        :param rng: The random number generator object

        :return: The next state of the Markov Algorithm
        """
        pass

    def _get_net_runtime(self):
        return time.time() - self.start_time

    net_runtime = property(_get_net_runtime)

    def _should_stop(self, state):
        if self.do_stop(state):
            return True
        if self.net_runtime > self.max_runtime:
            return True
        if state.iter >= self.max_iters:
            return True
        return False


    def do_stop(self, state):
        """
        method that decides when the iterative algorithm should terminate

        :param state: Current state

        :return: *True* if the algorithm should terminate. *False* otherwise.
        """
        return False

    @abc.abstractmethod
    def start_state(self, params, data_params, rng):
        """
        User-defined method that is expected to return the starting state of the algorithm.

        :param params: A dict of the parameters of the chain
        :param data_params: A dict describing the shape of the data. Typically includes *n*, the number of data points, and *dim*, the dimensionality of the data.
        :param rng: The random number generator

        :return: The initial state of the algorithm
        """
        pass

    def _attach_state_metadata(self, state, iter):
        state.iter = iter
        state.time = time.time()

    def sample_data(self, state, params, data_params, rng):
        """
        User-defined method that is meant to resample the 'data' the chain is being trained on, as a function of the
        latent state. Useful for testing.

        :param state: The latent state of the Markov chain
        :param params: A dict of the parameters of the chain
        :param data_params: A dict describing the shape of the data to be returned. Typically includes *n*, the number of data points, and *dim*, the dimensionality of the data
        :param rng: Random number generator object
        :return: A numpy array of the data
        """

        logger.debug('Resampling data method not implemented')

    def sample_latent(self, params, data_params, rng):
        """
        User defined method that is meant to sample the latent state of the Markov chain from the prior. Useful for
        testing.

        :param params: A dict of the parameters of the chain.
        :param data_params: A dict of the shape of the data.
        :param rng: The random number generator object
        :return: A :py:class:`State`-derived object that contains a sample from the latent state.
        """
        pass

    def sample_state(self, data_params):
        """
        Returns a sample from the prior.

        :param data_params: A dict of the shape of the data to be returned
        :return:
        """
        s = self.sample_latent(self.params, data_params, self.rng)
        s.data = self.sample_data(s, self.params, data_params, self.rng)
        return s

    def _run(self, params=None):
        """
        Actually executes the algorithm. Starting with the state returned  by :py:meth:`start_state`,
        continues to call :py:meth:`transition` to retrieve subsequent states of the algorithm,
        until :py:meth:`do_stop` indicates the algorithm should terminate.

        :return: A list of :py:class:`State` objects, representing the state of the algorithm
        at the start of each iteration. **Exception**: The last state is the list is the state at
        the end of the last iteration.
        """
        if params is None:
            params = {}
        store_state = params.get('store_state', True)
        logger.debug('Running chain')
        if self.data is None:
            raise ParameterException("Data source not set when trying to run chain")
        states = []
        self.start_time = time.time()
        state = self.start_state(self.params, dict(n=len(self.data), dim=len(self.data.shape)), self.rng)
        self._attach_state_metadata(state, 0)
        for iter in itertools.count():
            if iter % 50 == 0:
                logger.debug("Chain running iteration %d" % iter)
            if store_state:
                states.append(state)
            new_state = self.transition(state, self.params, self.data, self.rng)
            self._attach_state_metadata(new_state, iter + 1)
            if self._should_stop(new_state):
                states.append(new_state)
                break
            state = new_state
        logger.debug("Chain complete, now summarizing states")
        for state in states:
            state.summarize()
        logger.debug("States summarized")
        self.end_time = time.time()
        if store_state:
            return states
        else:
            return new_state


    def geweke_test(self, M, data_params, g_funcs, log_freq=100):
        """
        Runs a Geweke test for a specified set of test functions.
        :param M: Number of iterations of the chain to run. Larger values will result in a more accurate test.
        :param data_params: The shape of the data to train on
        :param g_funcs: A list of test functions. Each function is expected to take in a state, and return a scalar value.
        :return: A :py:class:`GewekeTest` object containing a full description of the results of the test.
        """
        mc_sim = []
        for i in range(M):
            state = self.sample_state(data_params)
            mc_sim.append(state)
        cs_sim = []
        theta = self.start_state(self.params, data_params, self.rng)
        for i in range(M):
            if i % log_freq == 0:
                logger.debug("Iteration %d" % i)
            y = self.sample_data(theta, self.params, data_params, self.rng)
            theta = self.transition(theta, self.params, y, self.rng)
            theta.data = y
            cs_sim.append(deepcopy(theta))
        n_tests = len(g_funcs)
        g_vals = empty((n_tests, M, 2))
        p = empty(n_tests)
        z = empty(n_tests)
        for i in range(n_tests):
            for m in range(M):
                g_vals[i, m, 0] = g_funcs[i](mc_sim[m])
                g_vals[i, m, 1] = g_funcs[i](cs_sim[m])
            g_mean_mc = mean(g_vals[i, :, 0])
            g_mean_cs = mean(g_vals[i, :, 1])
            g_var_mc = var(g_vals[i, :, 0]) / M
            g_var_cs = helpers.estimate_mcmc_var(g_vals[i, :, 1])
            z[i] = (g_mean_mc - g_mean_cs) / sqrt((g_var_mc + g_var_cs))
            p[i] = 2 * stats.norm(0, 1).cdf(-abs(z[i]))
        return GewekeTest(p=p, z=z, cs=cs_sim, mc=mc_sim, g=g_vals)


    def get_data(self, state):
        """
        Returns the data the chain was trained on.
        """
        return self.data

    def summarize(self, history):
        """
        Return a summary of *history*, which will be computed on the cloud and then cached for local use.
        """
        pass


class DataSource(object):
    """
    Represents datasets that have been procedurally generated. Intended to be inherited from by users.

    Specifically, the *load* method needs to be implemented by the end user.
    """

    __metaclass__ = RegisteredClass

    def __init__(self, **kwargs):
        """
        Initializes the data source by setting its parameters. Note that data is not actually generated until *load*
        is called. This division is meant to allow for a client to set parameters, while the actual data is generated
        on the cloud rather than uploaded.

        :param kwargs: A set of parameters controlling the data source. At a minimum, keys should include

        seed
         An integer specifying the random seed

        test_fraction
         What fraction of the data in the dataset should be used as held-out test data, as opposed to training data
          for the inference algorithms

        """
        self.seed = kwargs.get('seed', 0)
        self.rng = random.RandomState(self.seed)
        self.data = None
        self.test_fraction = kwargs.get('test_fraction', .2)
        self.params = kwargs #todo: deal with unhashable parameters automatically
        self.loaded = False


    @abc.abstractmethod
    def load(self, params, rng):
        """
        Loads the data into memory.
        :param params: A python dict containing the parameters of the data source, as given to the Experiment object.
        :param rng: Random number generator to be used for generating synthetic datasets.
        """
        pass

    def _load_data(self):
        """
        Load/generate the data into memory
        """
        if self.loaded:
            logger.debug("Dataset is trying to load after already being loaded")
            return
        self.data = self.load(self.params, self.rng)
        if self.data is None:
            raise BaseException("Datasouce 'load_data' method failed to create data attribute")
        self._split_data(self.test_fraction)
        self.loaded = True

    def _get_train_data(self):
        """

        :return: Training data
        """
        return self.data[self.train_idx]

    def _get_test_data(self):
        """

        :return: Held-out test data
        """
        return self.data[self.test_idx]

    train_data = property(_get_train_data)
    test_data = property(_get_test_data)

    def size(self):
        """

        :return: The number of data points currently in the dataset
        """
        return len(self.data)

    def _split_data(self, test_fraction):
        """
        Splits the data into a training dataset and test dataset. Meant for internal use only.

        :param test_fraction: Fraction of data to put in the test training set.
        1-test_fraction is put into the training set.
        :type test_fraction: float
        """
        n = self.size()
        n_test = int(test_fraction * n)
        idx = arange(n)
        self.rng.shuffle(idx)
        self.test_idx = idx[0:n_test]
        self.train_idx = idx[n_test:]


class ProceduralDataSource(DataSource):
    def __init__(self, **kwargs):
        super(ProceduralDataSource, self).__init__(**kwargs)

    @abc.abstractmethod
    def llh_pred(self, x):
        """
        For procedural datasets, the log-likelihood of generating the data in *x* given the latent variables of the model
        """
        pass

    def branch(self, seed):
        """
        Returns a new data source that has the same parameters as the current datasource, but a new seed. Useful for
        creating many synthetic datasets that have the distribution.
        :param seed:
        :return:
        """
        params = self.params.copy()
        params['seed'] = seed
        new_src = type(self)(**params)
        return new_src
