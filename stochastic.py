import Cloudless.longjob
import numpy.random
import numpy

import copy

class StochasticInferenceProblem:
    #
    # NOTE: This is a slight departure from iSIMs, for expediency
    #
    # Params contain things like:
    # - number of topics [[latent/hypothesis params]]
    # - number of documents and words per document [[observable params]]
    # 
    # A state is a dictionary with:
    # - latent variables (name->val)
    # - observable variables (name->val)
    # - bookkeeping variables, (name->val, but name starts with "__")
    #   used for scores, sufficient statistics, etc
    #
    # This also takes:
    # - a latents_seed
    # - an observables_seed
    # - a test_observables_seed [[FIXME, where does that go?]]
    
    # sample from a prior over latent variables
    def sample_latents(self, state, params):
        pass

    # sample from a prior over observables, given the latents
    def sample_observables(self, state, params):
        pass

    # check that the observables satisfy a given constraint
    def check_constraint(self, state, params):
        pass

    # enforce the constraint
    #
    # data is a dictionary of state variable, forced value pairs
    def enforce_constraint(self, state, params, data):
        pass

    # evaluate the joint probability
    def evaluate_log_joint_latents(self, state):
        pass

    def evaluate_log_constraint_marginal_prob_given_latents(self, state):
        pass

    def evaluate_test_metrics(self, state):
        pass

    def render_state(self, state):
        pass

# FIXME: Make into iSIM machine, with proper setting of parameters, etc
#
# Adds all the bookkeeping variables, by putting them into the state dictionary
# which then gets summarized in the longjob history
# __score
# __latent_score
# __data_prob
#
# Can do (mode):
# - Follow prior test
# - Recover from prior (and compare predictive)
# - Inference (where the contents = data)
#
# __test_ll etc if defined, and flagged to be used

class MarkovChain():
    def __init__(self, stochastic_inference_problem, mode, observables = None):
        self.stochastic_inference_problem = stochastic_inference_problem
        self.is_testing = is_testing
        self.observables = observables

    def evaluate_log_joint_probability(self, state):
        return self.stochastic_inference_problem.evaluate_log_joint_probability(state)

    def get_summary(self, state):
        return self.stochastic_inference_problem.get_summary(state)

    def initialize_state(self, state):
        self.stochastic_inference_problem.sample_latents(state)

        if self.is_testing:
            self.stochastic_inference_problem.sample_observables(state)
        else:
            for (k, v) in self.observables.items():
                state[k] = v

    # iterate a transition kernel over latent variables
    #
    # returns proposal diagnostics
    def transition_latent(self, state):
        pass

    # iterate a transition kernel over observables
    #
    # returns proposal diagnostics, if applicable
    def transition_observables(self, state):
        self.stochastic_inference_problem.sample_observables(state)
        pass

class MarkovChainIterativeJob:
    """
    Makes it easy to generate an iterative job from a Markov Chain

    Typical mode:
    - One IterativeJob for following the prior given params
    - One IterativeJob for some particular synthetic recovery experiments
    - One IterativeJob for some 

    Stores 
    """
    pass

class MarkovChainDiagnostics:
    """
    Makes it easy to plot runs for each chain, and also standard aggregates

    Takes a function from parameters to colors, linestyles, and markers,
    so plots can be auto generated.
    
    By default, a state is summarized by all its scalar variables,
    including "__" variables (like scores, suffstats, etc). Each gets
    its own plot, versus runtime.
    """
    def __init__(self, job_runner, get_style_from_params_proc):
        self.job_runner = job_runner
        self.get_style_from_params_proc = get_style_from_params_proc

    def generate_all_plots(self, 'base_dir'):
        pass

def renormalize_and_sample(logpstar_vec):
    maxv = max(logpstar_vec)
    scaled = [logpstar - maxv for logpstar in logpstar_vec]
    logZ = reduce(numpy.logaddexp, scaled)
    logp_vec = [s - logZ for s in scaled]
    randv = numpy.random.random()
    for (i, logp) in enumerate(logp_vec):
        p = numpy.exp(logp)
        if randv < p:
            return i
        else:
            randv = randv - p
