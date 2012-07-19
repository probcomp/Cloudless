#!python
import argparse
import os
import sys
##
import matplotlib
matplotlib.use('Agg')
import pylab
import numpy
##
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless.examples.DPMB.DPMB as dm
reload(dm)
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB.settings as settings
reload(settings)


# parse some args
parser = argparse.ArgumentParser('Evaluate log probs of gibbs init\'d and '
                                 'intuitive states for a synthetic problem')
parser.add_argument('--rand_seed',default=0,type=int)
args,unkown_args = parser.parse_known_args()
rand_seed = args.rand_seed
random_state = hf.generate_random_state(rand_seed)

# read the problem
problem_file_str = os.path.join(settings.data_dir,'structured_problem.pkl.gz')
data_dict = rf.unpickle(problem_file_str)
data = data_dict['data']
inverse_permutation_indices_list = data_dict['inverse_permutation_indices_list']
zs_to_permute = data_dict['zs_to_permute']
num_rows = data.shape[0]
num_cols = data.shape[1]
num_splits = data_dict['num_splits']


def plot_state(state,new_zs,permutation_indices=None,fh=None,save_str=None):
    if permutation_indices is None:
        permutation_indices = numpy.argsort(new_zs)
    x_values = state.getXValues()
    #
    h_lines = []
    for z_index in numpy.unique(new_zs):
        h_lines.append(sum(new_zs==z_index))
    h_lines = numpy.array(h_lines).cumsum()
    #
    hf.plot_data(
        data=numpy.array(x_values)[permutation_indices],
        h_lines=h_lines,
        fh=fh)
    #
    if save_str is not None:
        pylab.savefig(save_str)
        pylab.close()

def get_aris(z_indices):
    # ari according to EACH ground truth state for a SINGLE input configuration
    ari_list = []
    for inverse_permutation_indices in inverse_permutation_indices_list:
        ground_truth_zs = zs_to_permute[
            numpy.argsort(inverse_permutation_indices)]
        new_aris = hf.calc_ari(z_indices,ground_truth_zs)
        ari_list.append(new_aris)
    return ari_list

def plot_states(zs_to_plot,save_str=None):
    fh = pylab.figure()
    title_str = 'plot of gibbs init\'d and intuitive configurations'
    pylab.title(title_str)
    for plot_idx,zs in enumerate(zs_to_plot):
        ari_list = get_aris(zs)
        ari_str = ','.join(['%.2f' % ari for ari in ari_list])
        #
        pylab.subplot(330+plot_idx+1)
        pylab.title(ari_str)
        plot_state(state,zs,fh=fh)
    #
    pylab.subplots_adjust(hspace=.5)
    if save_str is None:
        save_str = 'states_'+str(rand_seed)
    pylab.savefig(save_str)
    pylab.close()

def calc_state_logp(zs):
    state = ds.DPMB_State(
        gen_seed=0,
        num_cols=num_cols,
        num_rows=num_rows,
        init_x=data,
        init_z=hf.canonicalize_list(zs)[0])
    #
    alpha_logps,lnPdf,grid = hf.calc_alpha_conditional(state)
    alpha_log_prob = reduce(numpy.logaddexp,numpy.array(alpha_logps))
    #
    beta_log_probs = []
    for col_idx in range(num_cols):
        beta_logps,lnPdf,grid = hf.calc_beta_conditional(state,col_idx)
        beta_log_prob = reduce(numpy.logaddexp,numpy.array(beta_logps))
        beta_log_probs.append(beta_log_prob)
    #
    return alpha_log_prob + sum(beta_log_probs)

def plot_histograms(zs_list,save_str=None):
    state_logps = []
    for zs in zs_list:
        state_logps.append(calc_state_logp(zs))
    state_logps = numpy.array(state_logps)
    #
    xvals = range(len(state_logps))
    fh = pylab.figure()
    pylab.title('theoretical log probabilities')
    hf.bar_helper(xvals,state_logps,fh=fh)
    if save_str is None:
        save_str = 'histogram_rand_seed_'+str(rand_seed)
    pylab.savefig(save_str)
    return state_logps

# create some gibbs init states as a baseline for performance 
num_gibbs_init_states = 9 - len(inverse_permutation_indices_list) - 2
gibbs_zs = []
for gibbs_idx in range(num_gibbs_init_states):
    gibbs_seed = random_state.randint(sys.maxint)
    # create a state for a gibb init score/ari/log_prob
    state = ds.DPMB_State(
        gen_seed=gibbs_seed,
        num_cols=num_cols,
        num_rows=num_rows,
        init_x=data)
    gibbs_zs.append(state.getZIndices())

# proof that these permutation indices work
for idx,inverse_permutation_indices \
        in enumerate(inverse_permutation_indices_list):
    plot_state(state,zs_to_permute,
               permutation_indices=inverse_permutation_indices,
               save_str=str(idx)
               )

# create the zs to analyze: gibbs_init, all apart, all together, ground truth
zs_to_plot = gibbs_zs[:]
zs_to_plot.append([0 for x in xrange(num_rows)]) # all togehter
if num_rows < 2048: # takes forever if too large
    zs_to_plot.append(range(num_rows)) # all apart
else:
    zs_to_plot.append([0 for x in xrange(num_rows)]) # all togehter
for idx,inverse_permutation_indices \
        in enumerate(inverse_permutation_indices_list):
    permute_indices = numpy.argsort(inverse_permutation_indices)
    zs_to_plot.append(zs_to_permute[permute_indices].tolist())

# plot top states, histograms of actual vs theoretical state frequencies
plot_states(zs_to_plot,'states_with_intuitive_clusterings_'+str(rand_seed))
state_logps = plot_histograms(
    zs_to_plot,
    'histograms_with_intuitive_clusterings_'+str(rand_seed)
    )
