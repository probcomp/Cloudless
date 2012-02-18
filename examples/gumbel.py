# <nbformat>2</nbformat>

# simulate a discrete choice by noising up a gumbel
#
# approach:
# - generate the dice locally
# - farm out the process of simulating a bunch from a gumbel and calculating
#   the resulting average KL
# - plot the result

# <codecell>

import numpy.random
import itertools
import Cloudless
import Cloudless.memo
from IPython.parallel import *
import pylab

# </codecell>
# <codecell>
def entropy(p_vec):
    out = 0.0
    for p in p_vec:
        if p == 0:
            pass
        else:
            out += p * numpy.log(p)
    return out
# </codecell>
# <codecell>
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
# </codecell>
# <codecell>

# parameters for the test dice (note: test dice generated locally)
DIE_K       = 3
DIE_ALPHAS  = [10**(float(x) / float(DIE_K)) for x in [-1, 0, 1]]
DIE_REPEATS = 10

# </codecell>
# <codecell>

print "Generating alphas..."
nested_alphas = [[alpha for x in range(DIE_K)] for alpha in DIE_ALPHAS]

# </codecell>
# <codecell>

print "Generating dice..."
nested_dice   = [numpy.random.mtrand.dirichlet(alpha_vec, DIE_REPEATS) for alpha_vec in nested_alphas]

# </codecell>
# <codecell>

print "Flattening dice..."
all_dice      = list(itertools.chain.from_iterable(nested_dice))

# </codecell>
# <codecell>

# now we have dice, locally. 

# calculate entropies locally:
mem_entropy = Cloudless.memo.Memoize(entropy)
for die in all_dice:
    mem_entropy(die)

# </codecell>
# <codecell>

# FIXME: packaging issues with ipython.parallel

# </codecell>
# <codecell>

# FIXME: stop ignoring the noise level
@require('numpy', 'numpy.random')
def gumbel_sample(p_vec, noise_level = 0.0):
    import numpy
    import numpy.random
    e_max = numpy.log(max(p_vec))
    e_vec = [numpy.log(p) - e_max for p in p_vec]
    
    g_vec = [e + numpy.random.gumbel() for e in e_vec]

    # FIXME: choose more ecological noise model
    # gaussian noise, but truncated. maybe log-normal? exp?
    g_raw = list(g_vec)
    for (i, g) in enumerate(g_vec):
        offset = noise_level * numpy.random.normal()
        if g+offset < 0:
            g_vec[i] = 0
        else:
            g_vec[i] = g + offset

    # poisson spiking, each with the rate of e
    # choose the earliest spike

    return g_vec.index(max(g_vec))

# </codecell>
# <codecell>

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
        r1 = pylab.bar(range(len(p_vec)), p_vec, 0.35, color = 'r')
        r2 = pylab.bar(range(len(p_vec)), empirical_vec, 0.35, color='y')
        pylab.ylabel('Pr')
        pylab.legend( (r1[0], r2[0]), ('True', 'Empirical') )
        print "TRUE: " + str(p_vec) + " SUM: " + str(sum(p_vec))
        print "SANITY: " + str(empirical_vec) + "SUM: " + str(sum(empirical_vec))
        pylab.savefig('sanity.png')

    return D_KL(empirical_vec, p_vec)

# </codecell>
# <codecell>

NOISE_LEVELS = [0,1]
NOISE_REPEATS = 1
# </codecell>
# <codecell>

# for a given die, compute the avg KL for a range of noise levels
def raw_avg_kl_of_die_with_noise(p_vec, noise_level, sanity=False):
    avg_kl = 0

    for r in range(NOISE_REPEATS):
        avg_kl += die_kl(p_vec, noise_level = noise_level, sanity=sanity)

    avg_kl = float(avg_kl) / float(NOISE_REPEATS)
    return avg_kl
# </codecell>
# <codecell>

avg_kl_of_die_with_noise = Cloudless.memo.Memoize(raw_avg_kl_of_die_with_noise)
# </codecell>
# <codecell>

print "Getting average KL..."
# for each die/noise level pair, get the average kl
for die in all_dice:
    for noise in NOISE_LEVELS:
        avg_kl_of_die_with_noise(die, noise)
# </codecell>
# <codecell>

#print "Sanity check on first die"
#raw_avg_kl_of_die_with_noise(all_dice[0], NOISE_LEVELS[0], sanity=True)

def plot_robustness(name):
    xs = []
    ys = []
    zs = []

    for (k, v) in avg_kl_of_die_with_noise.iter():
        die = k[0]
        noise = k[1]
        H = entropy(die)
        akl = v
        xs.append(noise)
        ys.append(H)
        zs.append(akl)

    pylab.figure()
    pylab.hexbin(xs, ys, C=zs)
    cb = pylab.colorbar()
    pylab.xlabel('Noise level')
    pylab.ylabel('Entropy')
    cb.set_label('KL divergence')
    pylab.show()
    #pylab.savefig(name)
# </codecell>
# <codecell>
    
print "plotting!"
plot_robustness('robustness.png')

# </codecell>
