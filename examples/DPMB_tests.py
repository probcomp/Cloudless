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
if True:
    state = ds.DPMB_State(gen_seed=0,num_cols=32,num_rows=1000,init_alpha=1,init_betas=np.repeat(.01,32),init_z=1,init_x=None)
    score_before_removal = state.score
    # remove all the vectors
    while len(state.vector_list) > 0:
        state.remove_vector(state.vector_list[0])
    score_after_removal = state.score
    # check that score was not 0 before but now (practically) zero
    assert score_before_removal < -1 and abs(score_after_removal) < 1E-6, "DPMB_test.py fails score re-zeroing test 1"

print "Passed initialize to empty state, add, remove test"

# - start with a balanced state
# - assign and deassign vectors to manually chosen clusters
# - then undo all those moves
# - check that the score is the same as the initial score
#

# initialize to a state
if True:
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

print "Passed initialize to balanced, add, remove test"


# - construct a state that's empty of data, but has, say, 4 columns, with beta_d = 1000 (so the clusters are almost perfectly even)
# - generate a vector
# - test that its conditional probability, as in cluster_conditional, is very close to the CRP prior
#

# initialize to a state
if True:
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
    pylab.hist(empirical_distribution,len(np.unique(empirical_distribution)),normed=True)
    pylab.title("empirical distribution")
    pylab.subplot(212)
    pylab.hist(theoretical_distribution,len(np.unique(theoretical_distribution)),normed=True)
    pylab.title("theoretical distribution")




# for the 8 by 8 data matrix that we started to study today:
# - create a balanced 4-cluster problem with beta_d = 0.05 or so (so very clean data)
# - initialize to all-in-one and all-apart
# - show me the sequence, for each, of states and conditional probabilities on alpha, beta_1, z (each one in a single .png, so I can step through 3 slideshows, watching the state evolve from initialization to whatever it is, including the red bars for alpha, beta_1, z)



# can you cluster 100 digits from MNIST, anecdotally, with 10 digits from each class? or even just pick 3 classes, 10 digits each? show me the z plots versus the true clusters
