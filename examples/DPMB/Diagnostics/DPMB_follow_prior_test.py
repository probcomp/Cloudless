import matplotlib
##matplotlib.use('Agg')
import numpy as np
import numpy.random as nr
import matplotlib.pylab as pylab
import datetime
import re
import gc
#
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB.DPMB as dm
reload(dm)

##separted out form other tests beacuse it takes a while


# follow the prior:
#
# D=8
# N=8
# gen_alpha = None
# gen_betas = None s
# gen_z = None
# gen_x = None
#
# generate 1000 DPMB_State s with these parameters
# - make histograms of
#   alpha
#   beta_0
#   number of datapoints in cluster 0
#   total number of clusters

if False:
    sample_alpha_list = []
    sample_beta_0_list = []
    sample_cluster_0_count_list = []
    sample_num_clusters_list = []
    for gen_seed in range(1000):
        state = ds.DPMB_State(gen_seed=gen_seed,num_cols=8,num_rows=8,init_alpha=None,init_betas=None,init_z=None,init_x=None)
        sample_alpha_list.append(state.alpha)
        sample_beta_0_list.append(state.betas[0])
        sample_cluster_0_count_list.append(state.cluster_list[0].count())
        sample_num_clusters_list.append(len(state.cluster_list))

    pylab.figure()
    pylab.subplot(411)
    pylab.hist(np.log(sample_alpha_list),normed=True)
    pylab.title("alpha (log10)")
    pylab.subplot(412)
    hist(np.log(sample_beta_0_list),normed=True)
    pylab.title("beta_0 (log10)")
    pylab.subplot(413)
    hist(sample_cluster_0_count_list,normed=True)
    pylab.title("cluster_0_count")
    pylab.subplot(414)
    hist(sample_num_clusters_list,normed=True)
    pylab.title("num_clusters")

    import cPickle
    var_names = ["sample_alpha_list","sample_beta_0_list","sample_cluster_0_count_list","sample_num_clusters_list"]
    for var_name in var_names:
        with open(var_name+".pkl","wb") as fh:
            cPickle.dump(eval(var_name),fh)
    
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

pylab.show()
if True:
    start_ts = datetime.datetime.now()
    GEN_SEED = 1
    NUM_COLS = 8
    NUM_ROWS = 8
    INIT_ALPHA = None
    INIT_BETAS = None
    INIT_X = None
    EVERY_N = 1
    NUM_ITERS = 100
    state = ds.DPMB_State(gen_seed=GEN_SEED,num_cols=NUM_COLS,num_rows=NUM_ROWS,init_alpha=INIT_ALPHA,init_betas=INIT_BETAS,init_z=None,init_x=INIT_X)
    model = dm.DPMB(state=state,inf_seed=1,infer_alpha=True,infer_beta=True)
    ##
    chain_alpha_list = []
    chain_beta_0_list = []
    chain_cluster_0_count_list = []
    chain_num_clusters_list = []
    for iter_num in range(NUM_ITERS):

        transition_func_list = [model.transition_beta,model.transition_z,model.transition_alpha]
        transition_func_list = nr.permutation(transition_func_list)
        for func_idx,transition_func_handle in enumerate(transition_func_list):
            func_str = re.findall("method DPMB.(\w+)",str(transition_func_handle))[0]
            print func_str
            transition_func_handle()
            save_str = "state_"+str(iter_num)+"_"+str(func_idx)+"_"+str(func_str)
            model.state.plot(show=False,save_str=save_str,title_append=func_str)

        # model.transition(random_order=True)
        # model.state.plot()

        # pylab.close('all')
        
        if iter_num % EVERY_N == 0: ## must do this after inference
            chain_alpha_list.append(state.alpha)
            chain_beta_0_list.append(state.betas[0])
            chain_cluster_0_count_list.append(state.cluster_list[0].count())
            chain_num_clusters_list.append(len(state.cluster_list))

        prior_zs = np.sort(state.getZIndices()).tolist() ## could there be an issue with inference over canonical clustering? permuate the data?
        prior_alpha = state.alpha
        prior_betas = state.betas
        #
        state = ds.DPMB_State(gen_seed=iter_num,num_cols=NUM_COLS,num_rows=NUM_ROWS,init_alpha=prior_alpha,init_betas=prior_betas,init_z=prior_zs,init_x=INIT_X)
        model.state = state

        print "Time delta: ",datetime.datetime.now()-start_ts

        pylab.figure()
        pylab.subplot(411)
        pylab.hist(np.log10(chain_alpha_list),normed=True)
        pylab.title("alpha (log10)")
        pylab.subplot(412)
        pylab.hist(np.log10(chain_beta_0_list),normed=True)
        pylab.title("beta_0 (log10)")
        pylab.subplot(413)
        pylab.hist(chain_cluster_0_count_list,normed=True)
        pylab.title("chain_cluster_0_count_list")
        pylab.subplot(414)
        pylab.hist(chain_num_clusters_list,normed=True)
        pylab.title("num_clusters")
        #
        pylab.subplots_adjust(hspace=.5)
        pylab.savefig("hist_" + str(iter_num))
        # temp = raw_input("blocking: ---- ")
        pylab.close('all')
        gc.collect()

    import cPickle
    var_names = ["chain_alpha_list","chain_beta_0_list","chain_cluster_0_count_list","chain_num_clusters_list"]
    for var_name in var_names:
        with open(var_name+".pkl","wb") as fh:
            cPickle.dump(eval(var_name),fh)
