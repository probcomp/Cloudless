# - construct a state that's empty of data, but has, say, 4 columns, with beta_d = 1000 (so the clusters are almost perfectly even)
# - generate a vector
# - test that its conditional probability, as in cluster_conditional, is very close to the CRP prior
#
# - start with an empty state
# - add a bunch of vectors and remove them (perhaps even randomly firing those off)
# - check that the score returns to zero afterwards
#
# - start with a balanced state
# - assign and deassign vectors to manually chosen clusters
# - then undo all those moves
# - check that the score is the same as the initial score
#
#

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

# for the 8 by 8 data matrix that we started to study today:
# - create a balanced 4-cluster problem with beta_d = 0.05 or so (so very clean data)
# - initialize to all-in-one and all-apart
# - show me the sequence, for each, of states and conditional probabilities on alpha, beta_1, z (each one in a single .png, so I can step through 3 slideshows, watching the state evolve from initialization to whatever it is, including the red bars for alpha, beta_1, z)



# can you cluster 100 digits from MNIST, anecdotally, with 10 digits from each class? or even just pick 3 classes, 10 digits each? show me the z plots versus the true clusters
