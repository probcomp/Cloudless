#!python
import argparse
import os
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
args,unkown_args = parser.parse_known_args()

inf_seed = args.inf_seed
gen_seed = args.gen_seed
num_iters = args.num_iters

state = ds.DPMB_State(
    gen_seed=0,
    num_cols=num_cols,
    num_rows=num_rows,
    beta_min=1.E-2,
    beta_max=1.E2,
    init_x=data)

# proof that these permutation indices work
for idx,inverse_permutation_indices \
        in enumerate(inverse_permutation_indices_list):
    hf.plot_data(
        data=numpy.array(state.getXValues())[inverse_permutation_indices])
    pylab.savefig(str(idx))
    pylab.close()

transitioner = dm.DPMB(
    inf_seed=inf_seed,
    state=state,
    infer_alpha=True,
    infer_beta=True)

summaries = []
next_summary = transitioner.extract_state_summary()
#
ari_list = []
for inverse_permutation_indices in inverse_permutation_indices_list:
    ari_list.append(hf.calc_ari(
        state.getZIndices(),zs_to_permute[inverse_permutation_indices]))
next_summary['ari_list'] = ari_list
summaries.append(next_summary)

for iter_num in range(num_iters):
    transitioner.transition()
    next_summary = transitioner.extract_state_summary()
    ari_list = []
    for inverse_permutation_indices in inverse_permutation_indices_list:
        ari_list.append(hf.calc_ari(
            state.getZIndices(),zs_to_permute[numpy.argsort(inverse_permutation_indices)]))
    next_summary['ari_list'] = ari_list
    summaries.append(next_summary)
    if False and 250 < iter_num and iter_num < 270:
        if iter_num % 1 == 0:
            state.plot(save_str='state_'+str(iter_num))
    if iter_num % 100 == 0 and iter_num != 0:
        hf.printTS('Done iter ' + str(iter_num))
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
        temp_betas = numpy.log10([summary['betas'] for summary in summaries[2:]])
        pylab.subplot(413,sharex=ax1)
        pylab.plot(temp_betas[:,:4],color='b',linewidth=.5)
        pylab.xlabel('iter')
        pylab.ylabel('log10 betas')
        pylab.subplot(414,sharex=ax1)
        pylab.plot(temp_betas[:,4:],color='g',linewidth=.5)
        pylab.xlabel('iter')
        pylab.ylabel('log10 betas')
        #
        pylab.savefig('ari_score_betas_inf_seed_'+str(inf_seed))
        pylab.close()

rf.pickle((summaries,ari_mat),'summaries.pkl.gz')


