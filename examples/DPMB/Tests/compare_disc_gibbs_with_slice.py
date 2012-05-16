import matplotlib
matplotlib.use('Agg')
import pylab
import numpy as np
import scipy.special as ss
#
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB.DPMB as dm
reload(dm)

GEN_SEED = 1
NUM_COLS = 8
NUM_ROWS = 8
INIT_ALPHA = None
INIT_BETAS = None
INIT_X = None
EVERY_N = 1
NUM_ITERS = 7000
ALPHA_MIN = 1E-2
ALPHA_MAX = 1E4
state = ds.DPMB_State(
    gen_seed=GEN_SEED
    ,num_cols=NUM_COLS
    ,num_rows=NUM_ROWS
    ,init_alpha=INIT_ALPHA
    ,init_betas=INIT_BETAS
    ,init_z=None
    ,init_x=INIT_X
    ,alpha_min = ALPHA_MIN
    ,alpha_max = ALPHA_MAX
    )

logprob_disc_gibbs = hf.create_alpha_lnPdf(state)
grid = state.get_alpha_grid()

disc_gibbs_probs = [logprob_disc_gibbs(alpha) for alpha in grid]
norm_prob = hf.log_conditional_to_norm_prob(disc_gibbs_probs)
hf.bar_helper(x=np.log10(grid),y=norm_prob,v_line=np.log10(state.alpha))
pylab.savefig("disc_gibbs")
pylab.close()

n_slice_samples = 100000
state.alpha_fail_count = 0
slice_samples = []
new_alpha = state.alpha
for idx in range(n_slice_samples):
    new_alpha = hf.slice_sample_alpha(state,new_alpha)
    slice_samples.append(new_alpha)

counts,bins,patches = pylab.hist(np.log10(slice_samples)
                                 ,bins=np.log10(grid),normed=True)
pylab.close()
hf.bar_helper(x=bins[:-1],y=counts/sum(counts),v_line=np.log10(state.alpha))
pylab.savefig("slice")
pylab.close()

pretty_list = lambda y : map(lambda x : "%.2f" % x,y)
zip(pretty_list(norm_prob),pretty_list(grid))
zip(pretty_list(counts/sum(counts)),pretty_list(10**bins[:-1]))
