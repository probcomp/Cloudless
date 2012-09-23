import cloud
import stochastic
import longjob

import numpy.random
import numpy
import scipy
import scipy.special

cloud.start_simulator()

def test_foo():
    return True

# Test the CRP mixture stochastic inference problem prior

# - basic SIP
#
#   - is the probability of all-apart alpha^N, for randomly generated
#     problems?
#
#   - does the score accumulator work right for re-assigning datapoints?
#
#   - does the score accumulator work right for re-setting the hyperparams?
#
#   - does my state renderer work OK?
#
# - stochastic components
#
#   - does my 1D slice sampler work?
#
#   - does my 1D griddy Gibbs via inverse CDF work?
#
#   - do they agree? (they should use an iterator)
#
#   - does my 1D regular old discretized Gibbs work?
#
# - Markov chain
#
#   - does the overall Gibbs sampler follow the prior, anecdotally?
#
#   - do the default plotting diagnostics produce legit plots?
#
#   - does the automatic follow-the-prior test work okay?
#
#   - can I recover from data generated from the DP prior, anecdotally?
#
#   - what do the inference runs look like? can I recover the test metrics?
#
# - overall framework:
#
#   - what does the synthetic recovery graph look like?
#
# - Can I add a Gaussian datatype?
#
# - Can I add a CRP datatype?
#
# - Can I make it into CrossCat?

# NOTES:
# - latent seed
# - data seed (maybe unused)
# - test data seed (maybe unused)
#
# - anything needed to actually parameterize the problem, including
#   - constants in the background
#   - the shape of the data that's expected
#   - data generation parameters, for test data

# - CRP via Gibbs-From-Prior with Bernoulli data
#
# - CRP via data generated from DP prior, showing convergence
#
# - CRP via data generated from balanced finite mixtures, varying the
#   number of true clusters and plotting the recovered, for a couple
#   different sizes of dataset
