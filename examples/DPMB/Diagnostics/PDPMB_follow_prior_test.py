import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as pylab
import datetime
import gc
#
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless.examples.DPMB.PDPMB_State as pds
reload(pds)
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB.PDPMB as pdm
reload(pdm)

# generate 1000 PDPMB_States
# - make histograms of
#   alpha
#   beta_0
#   number of datapoints in cluster 0 (how to determine vector 0?
#   total number of clusters

run_spec = rf.gen_run_spec()
dataset_spec = run_spec["dataset_spec"]

if False:
    sample_alpha_list = []
    sample_beta_0_list = []
    sample_num_clusters_list = []
    for gen_seed in range(1000):
        pstate = pds.PDPMB_State(
            gen_seed=gen_seed
            ,num_cols=dataset_spec["num_cols"]
            ,num_rows=dataset_spec["num_rows"]
            ,num_nodes=4
#            ,init_alpha=dataset_spec["gen_alpha"]
#            ,init_betas=dataset_spec["gen_betas"]
            ,init_z = dataset_spec["gen_z"])
        sample_alpha_list.append(pstate.alpha)
        sample_beta_0_list.append(pstate.betas[0])
        sample_num_clusters_list.append(len(pstate.get_cluster_list()))


    pylab.figure()
    pylab.subplot(411)
    pylab.hist(np.log(sample_alpha_list),normed=True)
    pylab.title("alpha (log10)")
    #
    pylab.subplot(412)
    pylab.hist(np.log(sample_beta_0_list),normed=True)
    pylab.title("beta_0 (log10)")
    #
    pylab.subplot(414)
    pylab.hist(sample_num_clusters_list,normed=True)
    pylab.title("num_clusters")
    #
    pylab.subplots_adjust(hspace=.4)
    pylab.savefig("pdpmb_gen_prior.png")
    
#
# initialize a state from the prior
# run a modified follow_the_prior_transition:
#   - calls your original transition
#   - then removes all the vectors from the state and generates 8 new ones
#
# (implement
# run that chain for 1000 steps, forming a histogram of every 50 states it reaches, on alpha, beta_0, num_datapoints_in_cluster_0, total number of clusters
#
# plot those histograms in a 2 cols x 4 rows chart, one column for prior samples, one column for results from the Markov chain
#
# can automate "are these two histograms similar enough?" by normalizing them into probability distribution estimates, and running a Kolmogorov-Smirnof test
# but for starters, just eyeballing is enough

if True and "pmodel" not in locals():

    start_ts = datetime.datetime.now()
    EVERY_N = 1
    NUM_ITERS = 1000
    INIT_X = None
    NUM_COLS = 8
    NUM_ROWS = 32
    NUM_NODES = 5
    ALPHA_MAX = 1E4
    ALPHA_MIN = 1E-1
    pstate = pds.PDPMB_State(
        gen_seed=0
        ,num_cols=NUM_COLS
        ,num_rows=NUM_ROWS
        ,num_nodes=NUM_NODES
        ,init_alpha=None
        ,init_betas=None
        ,init_z = ("balanced",2)
        ,alpha_max = ALPHA_MAX
        ,alpha_min = ALPHA_MIN
        )
    pmodel = pdm.PDPMB(
        inf_seed=0
        ,state=pstate
        ,infer_alpha = True
        ,infer_beta = True)

    ##
    chain_alpha_list = []
    chain_beta_0_list = []
    chain_cluster_0_count_list = []
    chain_num_clusters_list = []
    data_size_list = []

if True:

    for iter_num in range(200): # range(NUM_ITERS):
        true_iter_num = len(chain_alpha_list)
        print "iter num : " + str(true_iter_num)
        pmodel.transition_x()
        pmodel.transition()
        
        if true_iter_num % EVERY_N == 0: ## must do this after inference
            cluster_list_len = len(pstate.get_cluster_list())
            chain_alpha_list.append(pstate.alpha)
            chain_beta_0_list.append(pstate.betas[0])
            chain_num_clusters_list.append(cluster_list_len)
            print "alpha: " + str(pstate.alpha)
            print "betas[0]: " + str(pstate.betas[0])
            print "num clusters: " + str(cluster_list_len)

        data_size_list.append([len(model.state.vector_list) for model in pmodel.state.model_list])

        save_str = "state_"+str(true_iter_num)+"_parent"
        pmodel.state.plot(show=False,save_str=save_str)
        for model_idx,model in enumerate(pmodel.state.model_list):
            save_str = "state_"+str(true_iter_num)+"_child_"+str(model_idx)
            pmodel.state.model_list[model_idx].state.plot(show=False,save_str=save_str)

        pylab.figure()
        pylab.subplot(411)
        pylab.hist(np.log10(chain_alpha_list),normed=True)
        pylab.title("alpha (log10)")
        #
        pylab.subplot(412)
        pylab.hist(np.log10(chain_beta_0_list),normed=True)
        pylab.title("beta_0 (log10)")
        #
        pylab.subplot(413)
        pylab.title("num_iters: " + str(true_iter_num))
        # pylab.hist(chain_cluster_0_count_list,normed=True)
        # pylab.title("chain_cluster_0_count_list")
        # pylab.savefig("chain_cluster_0_count_list.png")
        #
        pylab.subplot(414)
        pylab.hist(chain_num_clusters_list,normed=True)
        pylab.title("num_clusters")
        #
        pylab.subplots_adjust(hspace=.5)
        pylab.savefig("pdpmb_hist_" + str(true_iter_num))
        pylab.close('all')
        gc.collect()

        print "Time delta: ",datetime.datetime.now()-start_ts

def force_alpha(pmodel,alpha_val):
    pmodel.state.setAlpha(lambda x:0,alpha_val)

    if True:
        pmodel.transition_z()
        pmodel.transition_beta()
        pmodel.transition_alpha()
        pmodel.transition_node_assignments()
    else:
        pmodel.transition()

    pmodel.transition_x()

def display_info(pmodel):
    pmodel.state.plot(show=False,save_str="one_off")
    print pmodel.state.alpha
    print [model.state.alpha for model in pmodel.state.model_list]
    [pmodel.state.model_list[idx].state.plot(show=False,save_str="submodel_state_"+str(idx)) for idx in range(len(pmodel.state.model_list))]
    pylab.close('all')

def transition_alpha_helper(model,alpha=None):
    model.state.cluster_list = model.state.get_cluster_list()
    logp_list,lnPdf,grid = hf.calc_alpha_conditional(model.state)
    if alpha is None:
        alpha_idx = hf.renormalize_and_sample(model.random_state, logp_list)
        alpha = grid[alpha_idx]
    model.state.removeAlpha(lnPdf)
    model.state.setAlpha(lnPdf,alpha)
    # empty everything that was just used to mimic DPMB_State
    model.state.cluster_list = None

def force_beta(pmodel,beta_val):
    pmodel.state.cluster_list = pmodel.state.get_cluster_list()
    for col_idx in range(pmodel.state.num_cols):
        logp_list, lnPdf, grid = hf.calc_beta_conditional(pmodel.state,col_idx)
        pmodel.state.removeBetaD(lnPdf,col_idx)
        pmodel.state.setBetaD(lnPdf,col_idx,beta_val)
    pmodel.state.cluster_list = None


if False:
    while any([len(model.state.cluster_list)>1 for model in pmodel.state.model_list]):
        pmodel.state.setAlpha(lambda x:0,.01)
        pmodel.transition_z()
        pmodel.transition_beta()
        display_info(pmodel)

    for model in pmodel.state.model_list:
        model.state.debug_move_cluster = True 

    pmodel.transition_node_assignments()
    display_info(pmodel)

# pmodel.state.setAlpha(lambda x:0,.1)
# pmodel.transition_z()
# pmodel.transition_beta()
# display_info(pmodel)

# pmodel.transition_node_assignments()
# display_info(pmodel)
