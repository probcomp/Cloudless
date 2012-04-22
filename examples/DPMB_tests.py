import DPMB_plotutils as dp
reload(dp)
import DPMB_helper_functions as hf
reload(hf)
import DPMB_State as ds
reload(ds)
import DPMB as dm
reload(dm)
import numpy as np
import matplotlib.pylab as pylab


# - start with an empty state
# - add a bunch of vectors and remove them (perhaps even randomly firing those off)
# - check that the score returns to zero afterwards
#

# initialize to a state
if False:
    state = ds.DPMB_State(gen_seed=0,num_cols=32,num_rows=1000,init_alpha=1,init_betas=np.repeat(.01,32),init_z=1,init_x=None)
    score_before_removal = state.score
    # remove all the vectors
    while len(state.vector_list) > 0:
        state.remove_vector(state.vector_list[0])
    score_after_removal = state.score
    # check that score was not 0 before but now (practically) zero
    assert score_before_removal < -1 and abs(score_after_removal) < 1E-6, "DPMB_test.py fails score re-zeroing test 1"


# - start with a balanced state
# - assign and deassign vectors to manually chosen clusters
# - then undo all those moves
# - check that the score is the same as the initial score
#

# initialize to a state
if False:
    state = ds.DPMB_State(gen_seed=1,num_cols=32,num_rows=1000,init_alpha=1,init_betas=np.repeat(.01,32),init_z=("balanced",100),init_x=None)
    score_before_addition = state.score
    # generate a bunch of new data
    NUM_NEW_VECTORS = 100
    for vector_idx in range(NUM_NEW_VECTORS):
        state.generate_vector()
    score_after_addition = state.score
    # remove all the new data
    for vector_idx in range(NUM_NEW_VECTORS):
        state.remove_vector(state.vector_list[-1])
    score_after_removal = state.score
    ##check that a significant change was made to score but it found its way back to where it started
    assert abs(score_before_addition-score_after_addition) > 1 and abs(score_before_addition-score_after_removal) < 1E-6, "DPMB_test.py fails score re-zeroing test 2"


# - construct a state that's empty of data, but has, say, 4 columns, with beta_d = 1000 (so the clusters are almost perfectly even)
# - generate a vector
# - test that its conditional probability, as in cluster_conditional, is very close to the CRP prior
#

# initialize to a state
if False:
    state = ds.DPMB_State(gen_seed=1,num_cols=32,num_rows=1000,init_alpha=1,init_betas=np.repeat(.01,32),init_z=None,init_x=None)
    # generate a bunch of data vectors, removing them after creating but recording the cluster index
    empirical_distribution = []
    NUM_NEW_VECTORS = 1000
    for vector_idx in range(NUM_NEW_VECTORS):
        state.generate_vector()
        assigned_cluster_idx = state.cluster_list.index(state.vector_list[-1].cluster)
        empirical_distribution.append(assigned_cluster_idx)
        state.remove_vector(state.vector_list[-1])

    # determine what the theoretical distribution was
    theoretical_distribution = []
    for cluster_idx,cluster in enumerate(state.cluster_list):
        theoretical_distribution.extend(np.repeat(cluster_idx,len(cluster.vector_list)))

    # just plot until find a good function for KL Divergence or likewise
    pylab.figure()
    pylab.subplot(211)
    pylab.hist(empirical_distribution,len(np.unique(empirical_distribution)))
    pylab.title("empirical distribution")
    pylab.subplot(212)
    pylab.hist(theoretical_distribution,len(np.unique(theoretical_distribution)))
    pylab.title("theoretical distribution")


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

alpha_list = []
beta_0_list = []
cluster_0_count_list = []
num_clusters_list = []
for gen_seed in range(1000):
    state = ds.DPMB_State(gen_seed=gen_seed,num_cols=8,num_rows=8,init_alpha=None,init_betas=None,init_z=None,init_x=None)
    alpha_list.append(state.alpha)
    beta_0_list.append(state.betas[0])
    cluster_0_count_list.append(state.cluster_list[0].count())
    num_clusters_list.append(len(state.cluster_list))

pylab.figure()
pylab.subplot(411)
pylab.hist(np.log(alpha_list))
pylab.title("alpha (log10)")
pylab.subplot(412)
hist(np.log(beta_0_list))
pylab.title("beta_0 (log10)")
pylab.subplot(413)
hist(cluster_0_count_list)
pylab.title("cluster_0_count")
pylab.subplot(414)
hist(num_clusters_list)
pylab.title("num_clusters")

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

start_ts = datetime.datetime.now()
GEN_SEED = 1
NUM_COLS = 8
NUM_ROWS = 8
INIT_ALPHA = None
INIT_BETAS = None
INIT_X = None
EVERY_N = 50
NUM_ITERS = 1000
state = ds.DPMB_State(gen_seed=GEN_SEED,num_cols=NUM_COLS,num_rows=NUM_ROWS,init_alpha=INIT_ALPHA,init_betas=INIT_BETAS,init_z=None,init_x=INIT_X)
model = dm.DPMB(state=state,inf_seed=0,infer_alpha=True,infer_beta=True)
##
alpha_list = []
beta_0_list = []
num_datapoints_in_cluster_0_list = []
num_clusters_list = []
for iter_num in range(NUM_ITERS):
    model.transition()
    if iter_num % EVERY_N == 0: ## must do this after inference
        alpha_list.append(state.alpha)
        beta_0_list.append(state.betas[0])
        num_datapoints_in_cluster_0_list.append(state.cluster_list[0].count())
        num_clusters_list.append(len(state.cluster_list))

    prior_zs = np.sort(state.getZIndices()).tolist() ## could there be an issue with inference over canonical clustering? permuate the data?
    state = ds.DPMB_State(gen_seed=iter_num,num_cols=NUM_COLS,num_rows=NUM_ROWS,init_alpha=INIT_ALPHA,init_betas=INIT_BETAS,init_z=prior_zs,init_x=INIT_X)
    model.state = state

print "Time delta: ",datetime.datetime.now()-start_ts

pylab.figure()
pylab.subplot(411)
pylab.hist(np.log(alpha_list))
pylab.title("alpha (log10)")
pylab.subplot(412)
hist(np.log(beta_0_list))
pylab.title("beta_0 (log10)")
pylab.subplot(413)
hist(num_datapoints_in_cluster_0_list)
pylab.title("num_datapoints_in_cluster_0")
pylab.subplot(414)
hist(num_clusters_list)
pylab.title("num_clusters")

# for the 8 by 8 data matrix that we started to study today:
# - create a balanced 4-cluster problem with beta_d = 0.05 or so (so very clean data)
# - initialize to all-in-one and all-apart
# - show me the sequence, for each, of states and conditional probabilities on alpha, beta_1, z (each one in a single .png, so I can step through 3 slideshows, watching the state evolve from initialization to whatever it is, including the red bars for alpha, beta_1, z)



# can you cluster 100 digits from MNIST, anecdotally, with 10 digits from each class? or even just pick 3 classes, 10 digits each? show me the z plots versus the true clusters
