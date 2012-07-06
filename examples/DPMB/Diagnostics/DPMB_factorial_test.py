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

data_dict = rf.unpickle(os.path.join(settings.data_dir,'factorial_problem.pkl.gz'))
data = data_dict['data']
inverse_permutation_indices_list = data_dict['inverse_permutation_indices_list']
zs_to_permute = data_dict['zs_to_permute']
num_rows = data.shape[0]
num_cols = data.shape[1]

parser = argparse.ArgumentParser('Run inference on a factorial problem')
parser.add_argument('inf_seed',type=int)
parser.add_argument('--gen_seed',default=0,type=int)
parser.add_argument('--num_iters',default=2000,type=int)
parser.add_argument('-no_intermediate_plots',action='store_true')
args,unkown_args = parser.parse_known_args()

inf_seed = args.inf_seed
gen_seed = args.gen_seed
num_iters = args.num_iters

def do_plot():
    ari_mat = numpy.array([summary['ari_list'] for summary in summaries[2:]])
    #
    pylab.figure()
    ax1 = pylab.subplot(411)
    pylab.plot(ari_mat)
    pylab.xlabel('iter')
    pylab.ylabel('independent aris')
    pylab.subplot(412,sharex=ax1)
    pylab.plot([summary["score"] for summary in summaries[2:]],color='k')
    pylab.xlabel('iter')
    pylab.ylabel('model score')
    #
    temp_betas = numpy.array([summary['betas'] for summary in summaries[2:]])
    pylab.subplot(413,sharex=ax1)
    pylab.plot(temp_betas[:,:4],color='b',linewidth=.5)
    pylab.xlabel('iter')
    pylab.ylabel('betas')
    pylab.subplot(414,sharex=ax1)
    pylab.plot(temp_betas[:,4:],color='g',linewidth=.5)
    pylab.xlabel('iter')
    pylab.ylabel('betas')
    #
    pylab.savefig('ari_score_betas_inf_seed_'+str(inf_seed))
    pylab.close()
    #
    return ari_mat

def save_state(state,new_zs,permutation_indices=None,fh=None,save_str=None):
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

state = ds.DPMB_State(
    gen_seed=0,
    num_cols=num_cols,
    num_rows=num_rows,
    beta_min=1.E-1,
    beta_max=1.E1,
    init_x=data)

# proof that these permutation indices work
for idx,inverse_permutation_indices \
        in enumerate(inverse_permutation_indices_list):
    new_zs = zs_to_permute
    save_state(state,new_zs,permutation_indices=inverse_permutation_indices,save_str=str(idx))

transitioner = dm.DPMB(
    inf_seed=inf_seed,
    state=state,
    infer_alpha=True,
    infer_beta=True)

z_indices_count = dict()
summaries = []
next_summary = transitioner.extract_state_summary()
#
ari_list = []
z_indices = state.getZIndices()
z_indices_count[str(z_indices)] = 1
for inverse_permutation_indices in inverse_permutation_indices_list:
    ari_list.append(hf.calc_ari(
            z_indices,zs_to_permute[inverse_permutation_indices]))
next_summary['ari_list'] = ari_list
summaries.append(next_summary)

transition_orders = []
for iter_num in range(num_iters):
    transition_order = transitioner.transition()
    transition_orders.append(transition_order)
    next_summary = transitioner.extract_state_summary()
    ari_list = []
    z_indices = state.getZIndices()
    z_indices_count[str(z_indices)] = z_indices_count.get(str(z_indices),0) + 1
    for inverse_permutation_indices in inverse_permutation_indices_list:
        ari_list.append(hf.calc_ari(
                z_indices,zs_to_permute[numpy.argsort(inverse_permutation_indices)]))
    next_summary['ari_list'] = ari_list
    summaries.append(next_summary)
    if False and 250 < iter_num and iter_num < 270:
        if iter_num % 1 == 0:
            state.plot(save_str='state_'+str(iter_num))
    if iter_num % 100 == 0 and iter_num != 0:
        hf.printTS('Done iter ' + str(iter_num))
        if not args.no_intermediate_plots:
            do_plot()

ari_mat = do_plot()
top_zs = sorted(z_indices_count.keys(),lambda x,y: int(numpy.sign(z_indices_count[x]-z_indices_count[y])))[-9:]
summaries[-1]['top_zs'] = top_zs
rf.pickle((summaries,ari_mat),'summaries.pkl.gz')

fh = pylab.figure()
pylab.title('Count of particular samples: total samples=' + str(args.num_iters))
for plot_idx in range(9):
    zs_str = top_zs[-plot_idx]
    count = z_indices_count[zs_str]
    zs = eval(zs_str)
    ari_list = []
    for inverse_permutation_indices in inverse_permutation_indices_list:
        ari_list.append(hf.calc_ari(
                zs,zs_to_permute[numpy.argsort(inverse_permutation_indices)]))
    #
    pylab.subplot(330+plot_idx)
    ari_str = ','.join(['%.2f' % ari for ari in ari_list])
    pylab.title(str(count) + ' ; ' + ari_str)
    save_state(state,zs,fh=fh)

pylab.subplots_adjust(hspace=.5)
pylab.savefig('top_states_'+str(inf_seed))
pylab.close()


state_logps = []
for zs_str in top_zs[-9:]:
    zs = eval(zs_str)
    state = ds.DPMB_State(
        gen_seed=0,
        num_cols=num_cols,
        num_rows=num_rows,
        beta_min=1.E-1,
        beta_max=1.E1,
        init_x=data,
        init_z=zs)

    alpha_logps,lnPdf,grid = hf.calc_alpha_conditional(state)
    alpha_log_prob = reduce(
        numpy.logaddexp,
        numpy.array(alpha_logps)
        )
    #
    beta_log_probs = []
    for col_idx in range(num_cols):

        beta_logps,lnPdf,grid = hf.calc_beta_conditional(state,col_idx)
        beta_log_prob = reduce(
            numpy.logaddexp,
            numpy.array(beta_logps)
            )
        beta_log_probs.append(beta_log_prob)

    print '%3.2g' % numpy.exp(alpha_log_prob), \
        '%3.2g' % numpy.exp(sum(beta_log_probs)), \
        '%3.2g' % numpy.exp(alpha_log_prob + sum(beta_log_probs))
    state_logps.append(alpha_log_prob + sum(beta_log_probs))

transition_orders = numpy.array(transition_orders)
state_probs = numpy.exp(state_logps)
state_counts = numpy.array([z_indices_count[zs] for zs in top_zs[-9:]])

fh = pylab.figure()
pylab.subplot(211)
pylab.title('theoretical frequencies')
hf.bar_helper(xrange(len(state_logps)),numpy.exp(state_logps - state_logps[-1]),fh=fh)
pylab.subplot(212)
pylab.title('empirical frequencies')
hf.bar_helper(xrange(len(state_logps)), state_counts/float(state_counts[-1]),fh=fh)
pylab.savefig('histogram_inf_seed_'+str(inf_seed))


print
print "order of transitions occurring"
print "transition 0"
print Counter(transition_orders[:,0])
print "transition 1"
print Counter(transition_orders[:,1])
print "transition 2"
print Counter(transition_orders[:,2])
