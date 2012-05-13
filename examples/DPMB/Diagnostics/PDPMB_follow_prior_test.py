import matplotlib
matplotlib.use('Agg')
import numpy as np
import numpy.random as nr
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
            ,init_gammas=[1.0/dataset_spec["num_cols"]
                         for idx in range(dataset_spec["num_cols"])]
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

if True:
    sample_alpha_list = []
    sample_beta_0_list = []
    sample_num_clusters_list = []
    start_ts = datetime.datetime.now()
    EVERY_N = 1
    NUM_ITERS = 3000
    INIT_X = None
    INIT_BETAS = None
    INIT_ALPHA = None # FIXME : should be None
    NUM_NODES = 1
    pstate = pds.PDPMB_State(
        gen_seed=0
        ,num_cols=dataset_spec["num_cols"]
        ,num_rows=8
        ,num_nodes=NUM_NODES
        ,init_gammas=[1.0/NUM_NODES
                     for idx in range(NUM_NODES)]
        ,init_alpha=INIT_ALPHA
        ,init_betas=INIT_BETAS
        ,init_z = ("balanced",2))
    pmodel = pdm.PDPMB(
        0
        ,pstate
        ,run_spec["infer_do_alpha_inference"]
        ,run_spec["infer_do_betas_inference"])
    sample_alpha_list.append(pstate.alpha)
    sample_beta_0_list.append(pstate.betas[0])
    sample_num_clusters_list.append(len(pstate.get_cluster_list()))

    ##
    chain_alpha_list = []
    chain_beta_0_list = []
    chain_cluster_0_count_list = []
    chain_num_clusters_list = []
    for iter_num in range(NUM_ITERS):
        print "iter num : " + str(iter_num)
        pmodel.transition()
        # temp = raw_input("blocking: ---- ")
        # pylab.close('all')
        
        if iter_num % EVERY_N == 0: ## must do this after inference
            cluster_list_len = len(pstate.get_cluster_list())
            chain_alpha_list.append(pstate.alpha)
            chain_beta_0_list.append(pstate.betas[0])
            chain_num_clusters_list.append(cluster_list_len)
            print "alpha: " + str(pstate.alpha)
            print "betas[0]: " + str(pstate.betas[0])
            print "num clusters: " + str(cluster_list_len)

        rand_state = nr.mtrand.RandomState(iter_num)
        seed_list = [int(x) for x in rand_state.tomaxint(len(pstate.model_list))]
        for gamma_i,model,gen_seed in zip(
                pstate.gammas,pstate.model_list,seed_list):
            prior_zs = model.state.getZIndices()
            # FIXME : What to use for gen_seed here?
            state = ds.DPMB_State(gen_seed=gen_seed
                                  ,num_cols=dataset_spec["num_cols"]
                                  ,num_rows=len(prior_zs)
                                  ,init_alpha=INIT_ALPHA
                                  ,init_betas=INIT_BETAS
                                  ,init_z=prior_zs
                                  ,init_x=INIT_X
                                  ,alpha_min=gamma_i*pstate.alpha_min
                                  ,alpha_max=gamma_i*pstate.alpha_max
                                  )
            model.state = state

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
        pylab.title("num_iters: " + str(iter_num))
        # pylab.hist(chain_cluster_0_count_list,normed=True)
        # pylab.title("chain_cluster_0_count_list")
        # pylab.savefig("chain_cluster_0_count_list.png")
        #
        pylab.subplot(414)
        pylab.hist(chain_num_clusters_list,normed=True)
        pylab.title("num_clusters")
        #
        pylab.subplots_adjust(hspace=.5)
        pylab.savefig("pdpmb_follow_prior.png")
        pylab.close()
        gc.collect()

    print "Time delta: ",datetime.datetime.now()-start_ts
