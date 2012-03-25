import numpy.random
import numpy
import Cloudless
from Cloudless.stochastic import *

class OrWithNoise(StochasticInferenceProgram):
    def __init__(self, N, p=0.5, eps=None, epsvals = None):
        self.N = N
        self.p = p
        self.eps = eps
        if epsvals is None:
            self.epsvals = numpy.arange(0, 1, 0.1)
        else:
            self.epsvals = epsvals

    def sample_latents(self, state):
        if self.eps is None:
            state["eps"] = numpy.random.random()

        for i in range(self.N):
            state["x_" + str(i)] = (numpy.random.random() < self.p)

    def sample_observables(self, state):
        isOrTrue = False
        
        for i in range(self.N):
            if state["x_" + str(i)]:
                isOrTrue = True

        if isOrTrue:
            state["y"] = numpy.random.random() < (1.0 - self.eps)
        else:
            state["y"] = numpy.random.random() < self.eps

    def evaluate_log_joint_probability(self, state):
        # start with log(Pr[state] = 1.0) = 0.0, as no random
        # choices have been incorporated yet
        out = 0.0

        # prior on eps is uniform, so no contribution

        # add in prior on flips
        flips = [state["x_" + str(i)] for i in range(self.N)]
        num_heads = reduce(lambda count, x: count+1 if x else count, flips, 0)
        num_tails = self.N - num_heads
        out += num_heads * numpy.log(self.p)
        out += num_tails * numpy.log(1.0 - self.p)

        noiseless_or = True if num_heads > 0 else False
        
        eps = state["eps"] if self.eps is None else self.eps

        if noiseless_or and state["y"] or not noiseless_or and not state["y"]:
            out += numpy.log(1.0 - eps)
        else:
            out += numpy.log(eps)

        return out

    def get_summary(self, state):
        return {"logscore": self.evaluate_log_joint_probability(state),
                "eps": state["eps"] if self.eps is None else self.eps,
                "num_heads": sum([1 if x else 0 for x in [state["x_" + str(i)] for i in range(self.N)]])}
        
class OrWithNoise_ChurchGibbsNoise(MarkovChain):
    def __init__(self, orwithnoise, is_testing = True, observables = None):
        MarkovChain.__init__(self, orwithnoise, is_testing, observables)
        self.N = orwithnoise.N
        self.p = orwithnoise.p
        self.eps = orwithnoise.eps
        self.epsvals = orwithnoise.epsvals

    def transition_latent(self, state):
        # pick a variable at random from the state
        
        choices = list(state.keys())
        del choices[choices.index("y")]
        chosen_var = choices[numpy.random.randint(len(choices))]
        
        # if the name is eps, do Gibbs
        if chosen_var is "eps":
            print "chose eps"
            assert self.eps is None #if not true, we shouldn't have chosen this var

            # enumerate eps values and score each under the joint

            logpstar_vec = []
            for new_eps in self.epsvals:
                state["eps"] = new_eps
                logpstar_vec.append(self.evaluate_log_joint_probability(state))

            # renormalize, sample, and store the new state
            eps_idx = renormalize_and_sample(logpstar_vec)
            state["eps"] = self.epsvals[eps_idx]

        else:
            # otherwise, do MH from the prior, following Church:
            log_p_old = self.evaluate_log_joint_probability(state)

            # pick a flip at random
            idx = numpy.random.randint(self.N)
            var_name = "x_" + str(idx)
            old_val = state[var_name]

            # propose a new value, calculating the qs and the p
            new_val = numpy.random.random() < self.p

            log_q_forward = numpy.log(self.p) if new_val else numpy.log(1.0 - self.p)
            log_q_reverse = numpy.log(self.p) if old_val else numpy.log(1.0 - self.p)
            state[var_name] = new_val

            log_p_new = self.evaluate_log_joint_probability(state)

            # accept/reject via MH
            accept_val = min(1.0, numpy.exp(log_p_old + log_q_reverse - log_p_new - log_q_forward))

            if accept_val >= 1.0 or numpy.random.random() < accept_val:
                # print "accepted"
                # keep the move
                pass
            else:
                # print "rejected"
                # undo the move
                state[var_name] = old_val

def make_orwithnoise_job(N, p=0.5, eps=None, iters=100):
    own = OrWithNoise(N, p, eps)
    own_mc = OrWithNoise_ChurchGibbsNoise(own, is_testing = False, observables = {"y": True})
    mclongjob = MarkovChainLongjob(own_mc, iters)
    return mclongjob

mclongjob = make_orwithnoise_job(5, p=0.1, eps=None, iters=100)
while mclongjob.iterate() is None:
    print mclongjob.get_summary()
