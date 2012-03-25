import Cloudless.longjob
import numpy.random
import numpy

class StochasticInferenceProgram:
    # sample from a prior over latent variables
    def sample_latents(self, state):
        pass

    # sample from a prior over observables, given the latents
    def sample_observables(self, state):
        pass

    # check that the observables satisfy a given constraint
    def check_constraint(self, state):
        pass

    # get the log joint probability of a state
    def evaluate_log_joint_probability(self, state):
        pass

    # get a summary dictionary from the state
    def get_summary(self, state):
        pass

class MarkovChain():
    def __init__(self, stochastic_inference_program, is_testing = True, observables = None):
        self.stochastic_inference_program = stochastic_inference_program
        self.is_testing = is_testing
        self.observables = observables

    def evaluate_log_joint_probability(self, state):
        return self.stochastic_inference_program.evaluate_log_joint_probability(state)

    def get_summary(self, state):
        return self.stochastic_inference_program.get_summary(state)

    def initialize_state(self, state):
        self.stochastic_inference_program.sample_latents(state)

        if self.is_testing:
            self.stochastic_inference_program.sample_observables(state)
        else:
            for (k, v) in self.observables.items():
                state[k] = v

    # iterate a transition kernel over latent variables
    def transition_latent(self, state):
        pass

    # iterate a transition kernel over observables
    def transition_observables(self, state):
        self.stochastic_inference_program.sample_observables(state)

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

class MarkovChainLongjob(Cloudless.longjob.Longjob):
    def __init__(self, markov_chain, max_iters):
        self.markov_chain = markov_chain
        self.state = {}
        self.markov_chain.initialize_state(self.state)
        self.iter = 0
        self.max_iters = max_iters
        self.final = None
    def iterate(self):
        if self.iter >= self.max_iters:
            self.final = self.state
            return self.state

        # we're not stopping, so keep going

        if self.markov_chain.is_testing:
            if self.iter % 2 is 0:
                self.markov_chain.transition_latent(self.state)
            else:
                self.markov_chain.transition_observables(self.state)
        else:
            self.markov_chain.transition_latent(self.state)

        self.iter += 1
        return None

    def get_summary(self):
        return self.markov_chain.get_summary(self.state)
