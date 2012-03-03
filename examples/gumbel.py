#
# FIXME: figure notes
#
# a) exponentials, with samples, and vertical bar
# b) gammas, with samples, and vertical bar
# c) robustness plot, like precision
# 
# - imprecise energies
# - stochastic process has noise
# - max/min has noise
#
# - sigmoid on serial or binary search tree
# 
# - gaussian to list being minned, but then what sigma and truncation?
# - smooth, in a bayesian sense, towards the uniform distribution
#
# simulate a discrete choice by noising up a gumbel
#
# approach:
# - generate the dice locally
# - farm out the process of simulating a bunch from a gumbel and calculating
#   the resulting average KL
# - plot the result

#block 1: remote definitions, suitable for re-evaluation after git push and plugin
#         re-install
import numpy.random
import numpy
import itertools
import Cloudless
import Cloudless.base
import Cloudless.memo
reload(Cloudless)
reload(Cloudless.base)
reload(Cloudless.memo)
from IPython.parallel import *
import matplotlib
matplotlib.use('Agg')
import pylab
###########################################

# block 2: configure remote nodes
Cloudless.base.remote_mode()
Cloudless.base.remote_exec('import numpy')
Cloudless.base.remote_exec('import numpy.random')
Cloudless.base.remote_exec('import itertools')
###########################################

# block 3: helper functions
def entropy(p_vec):
    out = 0.0
    for p in p_vec:
        if p == 0:
            pass
        else:
            out += p * numpy.log(p)
    return -1 * out
Cloudless.base.remote_procedure('entropy', entropy)

def D_KL(p_vec, q_vec):
    out = 0.0
    for (p, q) in zip(p_vec, q_vec):
        if p == 0:
            pass
        elif q == 0:
            # FIXME: is this the cleanest? do we want to smooth?
            return numpy.inf
        else:
            out += p * (numpy.log(p) - numpy.log(q))

    return out
Cloudless.base.remote_procedure('D_KL', D_KL)
###########################################

# block 4: generate dice locally
# parameters for the test dice (note: test dice generated locally)
DIE_K       = 10
DIE_ALPHAS  = [10**(float(x) / float(DIE_K)) for x in [-4, -1, 0, 1, 4]]
DIE_REPEATS = 10

print "Generating alphas..."
nested_alphas = [[alpha for x in range(DIE_K)] for alpha in DIE_ALPHAS]

print "Generating " + str(len(nested_alphas) * DIE_REPEATS) + " dice..."
nested_dice   = [numpy.random.mtrand.dirichlet(alpha_vec, DIE_REPEATS) for alpha_vec in nested_alphas]

print "Flattening dice..."
all_dice      = list(itertools.chain.from_iterable(nested_dice))

# now we have dice, locally. 
###########################################

# block 5: calculate entropies remotely
# calculate entropies remotely
mem_entropy = Cloudless.memo.AsyncMemoize('entropy', ['die'], entropy, override = True)
for die in all_dice:
    mem_entropy(die)
###########################################

# block 6: helper functions
# FIXME: stop ignoring the noise level
def gumbel_sample(p_vec, noise_level = 0.0):
    import numpy
    import numpy.random
    e_max = numpy.log(max(p_vec))
    e_vec = [numpy.log(p) - e_max for p in p_vec]

    # do gamma noise
    gamma_vec = [numpy.random.gamma(1.0 + noise_level, 1.0 / numpy.exp(e)) for e in e_vec]
    return gamma_vec.index(min(gamma_vec))

    #g_vec = [numpy.random.gumbel(e, 1.0 + noise_level) for e in e_vec]
    #return g_vec.index(max(g_vec))
    
    # FIXME can also try min-gamma noise

    # poisson spiking, each with the rate of e
    # choose the earliest spike:
    # argmin(exp(a), exp(b)) is 0 with probability a/a+b by poisson process
    # argmax(gumbel(log(a)), gumbel(log(b))) is 0 with probability a/a+b
Cloudless.base.remote_procedure('gumbel_sample', gumbel_sample)

def die_kl(p_vec, num_samples = 1000, noise_level = 0.0, sanity=False):
    counts = [0 for i in range(len(p_vec))]
    samples = [gumbel_sample(p_vec, noise_level) for i in range(num_samples)]
    for s in samples:
        counts[s] += 1
    empirical_vec = [float(count) / float(num_samples) for count in counts]
    # FIXME: Should we use JS, or smoothed KL in the reverse, rather than
    #        this non-intuitive KL?

    if sanity:
        # save the histogram(s) to disk
        pylab.figure()
        ind = range(len(p_vec))
        ind2 = [idx + 0.35 for idx in ind]
        r1 = pylab.bar(ind, p_vec, 0.35, color = 'r')
        r2 = pylab.bar(ind2, empirical_vec, 0.35, color='y')
        
        pylab.ylabel('Pr')
        pylab.legend( (r1[0], r2[0]), ('True', 'Empirical') )
        print "TRUE: " + str(p_vec) + " SUM: " + str(sum(p_vec))
        print "SANITY: " + str(empirical_vec) + "SUM: " + str(sum(empirical_vec))
        pylab.savefig('sanity.png')

    return D_KL(empirical_vec, p_vec)
Cloudless.base.remote_procedure('die_kl', die_kl)
###########################################

# block: define core experimental procedure
# for a given die, compute the avg KL for a range of noise levels
def raw_avg_kl_of_die_with_noise(p_vec, noise_level, sanity=False):
    avg_kl = 0

    for r in range(NOISE_REPEATS):
        avg_kl += die_kl(p_vec, noise_level = noise_level, sanity=sanity)

    avg_kl = float(avg_kl) / float(NOISE_REPEATS)
    return avg_kl

avg_kl_of_die_with_noise = Cloudless.memo.AsyncMemoize('avg_kl', ['p_vec', 'noise_level'], raw_avg_kl_of_die_with_noise, override = True)
###########################################

# block: trigger experiments
NOISE_LEVELS = [0, 0.2, 0.5, 1, 2, 5, 10]
NOISE_REPEATS = 1

print "Getting average KL for " + str(len(all_dice) * len(NOISE_LEVELS) * NOISE_REPEATS) + " trials..."
# for each die/noise level pair, get the average kl
for die in all_dice:
    for noise in NOISE_LEVELS:
        avg_kl_of_die_with_noise(die, noise)
#print "Sanity check on first die"
#raw_avg_kl_of_die_with_noise(all_dice[0], NOISE_LEVELS[0], sanity=True)
###########################################

# block: plot robustness based on available results
def plot_robustness(name):
    xs = []
    ys = []
    zs = []

    for (k, v) in avg_kl_of_die_with_noise.iter():
        die = k[0]
        noise = k[1]
        H = mem_entropy(die)
        if H is not None:
            akl = v
            xs.append(noise)
            ys.append(H)
            zs.append(akl)

    if len(xs) > 0:
        pylab.figure()
        pylab.hexbin(xs, ys, C=zs)
        cb = pylab.colorbar()
        pylab.xlabel('Noise level')
        pylab.ylabel('Entropy')
        cb.set_label('KL divergence')
        pylab.show()
#    pylab.savefig(name)
    
print "plotting!"
plot_robustness('robustness.png')
