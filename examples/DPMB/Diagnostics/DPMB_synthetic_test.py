#!python
import argparse
import os
from collections import Counter
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


# parse some arguments
parser = argparse.ArgumentParser('Run inference on a synthetic problem')
parser.add_argument('inf_seed',type=int)
parser.add_argument('--gen_seed',default=0,type=int)
parser.add_argument('--num_iters',default=2000,type=int)
parser.add_argument('-intermediate_plots',action='store_true')
args,unkown_args = parser.parse_known_args()
#
inf_seed = args.inf_seed
gen_seed = args.gen_seed
num_iters = args.num_iters

# read the problem
problem_file_str = os.path.join(settings.data_dir,'structured_problem.pkl.gz')
data_dict = rf.unpickle(problem_file_str)
data = data_dict['data']
inverse_permutation_indices_list = data_dict['inverse_permutation_indices_list']
zs_to_permute = data_dict['zs_to_permute']
num_rows = data.shape[0]
num_cols = data.shape[1]
num_splits = data_dict['num_splits']


def plot_timeseries():
    cols_per_split = num_cols/num_splits
    # reduced_summaries = summaries[2:] # omit first few to reduce range
    reduced_summaries = summaries # FIXME : change this back to [2:]
    reduced_betas = numpy.array([
            summary['betas']
            for summary in reduced_summaries
            ])
    ari_mat = numpy.array([
            summary['ari_list'] 
            for summary in reduced_summaries
            ])
    score = [
        summary["score"] 
        for summary in reduced_summaries
        ]
    #
    pylab.figure()
    ax1 = pylab.subplot(211 + 100*num_splits)
    pylab.plot(ari_mat)
    pylab.xlabel('iter')
    pylab.ylabel('independent aris')
    pylab.subplot(212 + 100*num_splits,sharex=ax1)
    pylab.plot(score,color='k')
    pylab.xlabel('iter')
    pylab.ylabel('model score')
    #
    for betas_idx in range(num_splits):
        start_idx = betas_idx*cols_per_split
        end_idx = (betas_idx+1)*cols_per_split
        pylab.subplot(213 + 100*num_splits + betas_idx,sharex=ax1)
        pylab.plot(reduced_betas[:,start_idx:end_idx],linewidth=.5)
        pylab.xlabel('iter')
        pylab.ylabel('betas: ' + str(start_idx) + ':' + str(end_idx))

    pylab.savefig('ari_score_betas_inf_seed_'+str(inf_seed))
    pylab.close()
    #
    return ari_mat

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

def plot_top_states(top_zs,save_str=None):
    fh = pylab.figure()
    title_str = 'Count of particular samples: total samples=' \
        + str(args.num_iters)
    pylab.title(title_str)
    for plot_idx,zs_str in enumerate(top_zs):
        count = z_indices_count.get(zs_str,0)
        zs = eval(zs_str)
        ari_list = []
        for inverse_permutation_indices in inverse_permutation_indices_list:
            ari_list.append(hf.calc_ari(
                    zs,zs_to_permute[numpy.argsort(inverse_permutation_indices)]))
        #
        pylab.subplot(330+plot_idx+1)
        ari_str = ','.join(['%.2f' % ari for ari in ari_list])
        pylab.title(str(count) + ' ; ' + ari_str)
        plot_state(state,zs,fh=fh)
    #
    pylab.subplots_adjust(hspace=.5)
    if save_str is None:
        save_str = 'top_states_'+str(inf_seed)
    pylab.savefig(save_str)
    pylab.close()

def plot_histograms(zs_strs,save_str=None):
    state_logps = []
    state_counts = []
    for zs_str in zs_strs:
        state_logps.append(calc_state_logp(eval(zs_str)))
        # fill in zeros with tiny value so histogram is spaced correctly
        state_counts.append(z_indices_count.get(zs_str,1.E-6))
    state_logps = numpy.array(state_logps)[numpy.argsort(state_counts)]
    state_counts = numpy.sort(state_counts)
    xvals = xrange(len(state_logps),0,-1) # xrange(len(state_logps)) # 
    #
    fh = pylab.figure()
    pylab.subplot(211)
    pylab.title('theoretical frequencies')
    hf.bar_helper(xvals,numpy.exp(state_logps - state_logps[-1]),fh=fh)
    pylab.xlabel('state rank')
    pylab.subplot(212)
    pylab.title('empirical frequencies')
    hf.bar_helper(xvals, state_counts/float(state_counts[-1]),fh=fh)
    pylab.xlabel('state rank')
    #
    pylab.subplots_adjust(hspace=.5)
    if save_str is None:
        save_str = 'histogram_inf_seed_'+str(inf_seed)
    pylab.savefig(save_str)
    return state_logps

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

def get_aris(z_indices):
    ari_list = []
    for inverse_permutation_indices in inverse_permutation_indices_list:
        # ground_truth_zs = zs_to_permute[inverse_permutation_indices]
        ground_truth_zs = zs_to_permute[numpy.argsort(inverse_permutation_indices)]
        new_aris = hf.calc_ari(z_indices,ground_truth_zs)
        ari_list.append(new_aris)
    return ari_list


# set up inference machinery
state = ds.DPMB_State(
    gen_seed=0,
    num_cols=num_cols,
    num_rows=num_rows,
    init_x=data)
#
transitioner = dm.DPMB(
    inf_seed=inf_seed,
    state=state,
    infer_alpha=True,
    infer_beta=True)

# proof that these permutation indices work
for idx,inverse_permutation_indices \
        in enumerate(inverse_permutation_indices_list):
    plot_state(state,zs_to_permute,
               permutation_indices=inverse_permutation_indices,
               save_str=str(idx)
               )

# save initial state
z_indices_count = dict()
summaries = []
#
z_indices = state.getZIndices()
z_indices_count[str(z_indices)] = z_indices_count.get(str(z_indices),0) + 1
next_summary = transitioner.extract_state_summary()
next_summary['ari_list'] = get_aris(z_indices)
summaries.append(next_summary)

# run inference
transition_orders = []
for iter_num in range(num_iters):
    transition_order = transitioner.transition()
    transition_orders.append(transition_order)
    z_indices = state.getZIndices()
    z_indices_count[str(z_indices)] = z_indices_count.get(str(z_indices),0) + 1
    next_summary = transitioner.extract_state_summary()
    next_summary['ari_list'] = get_aris(z_indices)
    summaries.append(next_summary)
    #
    if iter_num % 100 == 0 and iter_num != 0:
        hf.printTS('Done iter ' + str(iter_num))
        if args.intermediate_plots:
            plot_timeseries()

# final anlaysis  
ari_mat = plot_timeseries()
sort_counts_func = lambda x,y: int(numpy.sign(z_indices_count[x]-z_indices_count[y]))
top_zs = sorted(z_indices_count.keys(),sort_counts_func)[-9:][::-1]
summaries[-1]['top_zs'] = top_zs
rf.pickle((summaries,ari_mat),'summaries.pkl.gz')

# plot top states, histograms of actual vs theoretical state frequencies
plot_top_states(top_zs)
state_logps = plot_histograms(top_zs)

# create new top_zs with intuitively chosen states
top_zs_copy = top_zs[:-(2+len(inverse_permutation_indices_list))]
top_zs_copy.append(str([0 for x in xrange(num_rows)])) # all togehter
if num_rows < 1025:
    top_zs_copy.append(str(range(num_rows))) # all apart
else:
    top_zs_copy.append(str([0 for x in xrange(num_rows)])) # all togehter
for idx,inverse_permutation_indices \
        in enumerate(inverse_permutation_indices_list):
    permute_indices = numpy.argsort(inverse_permutation_indices)
    top_zs_copy.append(str(zs_to_permute[permute_indices].tolist()))
#
plot_top_states(top_zs_copy,'states_with_intuitive_clusterings_'+str(inf_seed))
state_logps = plot_histograms(top_zs_copy,'histograms_with_intuitive_clusterings_'+str(inf_seed))