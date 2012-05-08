import datetime,sys,pdb
##
import pylab
import matplotlib
import numpy as np
import numpy.random as nr
import scipy.special as ss
##
import DPMB_State as ds

# FIXME: do generate_from_prior test (to make Ryan happy)

####################
# PROBABILITY FUNCTIONS
def renormalize_and_sample(logpstar_vec,verbose=False):
    p_vec = log_conditional_to_norm_prob(logpstar_vec)
    randv = nr.random()
    for (i, p) in enumerate(p_vec):
        if randv < p:
            if verbose:
                print " - hash of seed is " + str(hash(str(nr.get_state())))
                print " - draw,probs: ",i,np.array(np.log(p_vec)).round(2)
            return i
        else:
            randv = randv - p

def log_conditional_to_norm_prob(logp_list):
    maxv = max(logp_list)
    scaled = [logpstar - maxv for logpstar in logp_list]
    logZ = reduce(np.logaddexp, scaled)
    logp_vec = [s - logZ for s in scaled]
    return np.exp(logp_vec)

def cluster_vector_joint(vector,cluster,state):
    # FIXME: Is np.log(0.5) correct? (Probably, since it comes from symmetry plus beta_d < 1.0.) How does
    # this relate to the idea that we mix out of all-in-one by first finding the prior over zs (by first
    # washing out the effect of the data, by raising the betas so high that all clusters are nearly perfectly
    # likely to be noisy)
    alpha = state.alpha
    numVectors = len(state.get_all_vectors())
    if cluster is None or cluster.count() == 0:
        ##if the cluster would be empty without the vector, then its a special case
        alpha_term = np.log(alpha) - np.log(numVectors-1+alpha)
        data_term = state.num_cols*np.log(.5)
        retVal =  alpha_term + data_term
    else:
        boolIdx = np.array(vector.data,dtype=type(True))
        alpha_term = np.log(cluster.count()) - np.log(numVectors-1+alpha)
        numerator1 = boolIdx * np.log(cluster.column_sums + state.betas)
        numerator2 = (~boolIdx) * np.log(cluster.count() - cluster.column_sums + state.betas)
        denominator = np.log(cluster.count() + 2*state.betas)
        data_term = (numerator1 + numerator2 - denominator).sum()
        retVal = alpha_term + data_term
    if not np.isfinite(retVal):
        pdb.set_trace()
    if hasattr(state,"print_predictive") and state.print_predictive:
        mean_p = np.exp(numerator1 + numerator2 - denominator).mean().round(2) if "numerator1" in locals() else .5
        print retVal.round(2),alpha_term.round(2),data_term.round(2),state.vector_list.index(vector),state.cluster_list.index(cluster),mean_p
    if hasattr(state,"debug_predictive") and state.debug_predictive:
        pdb.set_trace()
        temp = 1 ## if this isn't here, debug start in return and can't see local variables?
    return retVal,alpha_term,data_term

def create_alpha_lnPdf(state):
    lnProdGammas = sum([ss.gammaln(cluster.count()) for cluster in state.cluster_list])
    lnPdf = lambda alpha: (ss.gammaln(alpha) + len(state.cluster_list)*np.log(alpha)
                           - ss.gammaln(alpha+len(state.vector_list)) + lnProdGammas)
    return lnPdf

def create_beta_lnPdf(state,col_idx):
    S_list = [cluster.column_sums[col_idx] for cluster in state.cluster_list]
    R_list = [len(cluster.vector_list) - cluster.column_sums[col_idx] for cluster in state.cluster_list]
    beta_d = state.betas[col_idx]
    lnPdf = lambda beta_d: sum([ss.gammaln(2*beta_d) - 2*ss.gammaln(beta_d)
                                + ss.gammaln(S+beta_d) + ss.gammaln(R+beta_d)
                                - ss.gammaln(S+R+2*beta_d) for S,R in zip(S_list,R_list)])
    return lnPdf

def calc_alpha_conditional(state):
    ##save original value, should be invariant
    original_alpha = state.alpha
    ##
    grid = state.get_alpha_grid()
    lnPdf = create_alpha_lnPdf(state)
    logp_list = []
    for test_alpha in grid:
        state.removeAlpha(lnPdf)
        state.setAlpha(lnPdf,test_alpha)
        logp_list.append(state.score)
    ##
    state.removeAlpha(lnPdf)
    state.setAlpha(lnPdf,original_alpha)
    ##
    return logp_list,lnPdf,grid

def calc_beta_conditional(state,col_idx):
    lnPdf = create_beta_lnPdf(state,col_idx)
    grid = state.get_beta_grid()
    logp_list = []
    ##
    original_beta = state.betas[col_idx]
    for test_beta in grid:
        state.removeBetaD(lnPdf,col_idx)
        state.setBetaD(lnPdf,col_idx,test_beta)
        logp_list.append(state.score)
    ##put everything back how you found it
    state.removeBetaD(lnPdf,col_idx)
    state.setBetaD(lnPdf,col_idx,original_beta)
    ##
    return logp_list,lnPdf,grid

def calculate_cluster_conditional(state,vector):
    ##vector should be unassigned
    ##new_cluster is auto appended to cluster list
    ##and pops off when vector is deassigned

    # FIXME: if there is already an empty cluster (either because deassigning didn't clear it out,
    #        or for some other reason), then we'll have a problem here. maybe a crash, maybe just
    #        incorrect probabilities.
    new_cluster = ds.Cluster(state)
    state.cluster_list.append(new_cluster)
    ##
    conditionals = []
    for cluster in state.cluster_list:
        cluster.assign_vector(vector)
        conditionals.append(state.score)
        cluster.deassign_vector(vector)
    ##
    return conditionals

def mle_alpha(clusters,points_per_cluster,max_alpha=100):
    mle = 1+np.argmax([ss.gammaln(alpha) + clusters*np.log(alpha) - ss.gammaln(clusters*points_per_cluster+alpha) for alpha in range(1,max_alpha)])
    return mle

def mhSample(initVal,nSamples,lnPdf,sampler):
    samples = [initVal]
    priorSample = initVal
    for counter in range(nSamples):
        unif = nr.rand()
        proposal = sampler(priorSample)
        thresh = np.exp(lnPdf(proposal) - lnPdf(priorSample)) ## presume symmetric
        if np.isfinite(thresh) and unif < min(1,thresh):
            samples.append(proposal)
        else:
            samples.append(priorSample)
        priorSample = samples[-1]
    return samples

####################
# UTILITY FUNCTIONS
def plot_data(data,fh=None,h_lines=None,title_str=None,interpolation="nearest",**kwargs):
    if fh is None:
        fh = pylab.figure()
    pylab.imshow(data,interpolation=interpolation,cmap=matplotlib.cm.binary,**kwargs)
    if h_lines is not None:
        xlim = fh.get_axes()[0].get_xlim()
        pylab.hlines(h_lines-.5,*xlim,color="red",linewidth=3)
    if title_str is not None:
        pylab.title(title_str)
    return fh

def bar_helper(x,y,fh=None,v_line=None,title_str=None,which_id=0):
    if fh is None:
        fh = pylab.figure()
    pylab.bar(x,y,width=min(np.diff(x)))
    if v_line is not None:
        pylab.vlines(v_line,*fh.get_axes()[which_id].get_ylim(),color="red",linewidth=3)
    if title_str is not None:
        pylab.ylabel(title_str)
    return fh

def printTS(printStr):
    print datetime.datetime.now().strftime("%H:%M:%S") + " :: " + printStr
    sys.stdout.flush()

def listCount(listIn):
    return dict([(currValue,sum(np.array(listIn)==currValue)) for currValue in np.unique(listIn)])


####################
# SEED FUNCTIONS
def set_seed(seed):
    if type(seed) == tuple:
        nr.set_state(seed)
    elif type(seed) == int:
        nr.seed(seed)
    else:
        raise Exception("Bad argument to set_seed: " + str(seed)) 

def get_seed():
    return nr.get_state()

####################
# ARI FUNCTIONS
def calc_ari(group_idx_list_1,group_idx_list_2):
    ##https://en.wikipedia.org/wiki/Rand_index#The_contingency_table
    ##presumes group_idx's are canonicaized
    Ns,As,Bs = gen_contingency_data(group_idx_list_1,group_idx_list_2)
    n_choose_2 = choose_2_sum(np.array([len(group_idx_list_1)]))
    cross_sums = choose_2_sum(Ns[Ns>1])
    a_sums = choose_2_sum(As)
    b_sums = choose_2_sum(Bs)
    return ((n_choose_2*cross_sums - a_sums*b_sums)
            /(.5*n_choose_2*(a_sums+b_sums) - a_sums*b_sums))

def choose_2_sum(x):
    return sum(x*(x-1)/2.0)
            
def count_dict_overlap(dict1,dict2):
    overlap = 0
    for key in dict1:
        if key in dict2:
            overlap += 1
    return overlap

def gen_contingency_data(group_idx_list_1,group_idx_list_2):
    group_idx_dict_1 = {}
    for list_idx,group_idx in enumerate(group_idx_list_1):
        group_idx_dict_1.setdefault(group_idx,{})[list_idx] = None
    group_idx_dict_2 = {}
    for list_idx,group_idx in enumerate(group_idx_list_2):
        group_idx_dict_2.setdefault(group_idx,{})[list_idx] = None
    ##
    Ns = np.ndarray((len(group_idx_dict_1.keys()),len(group_idx_dict_2.keys())))
    for key1,value1 in group_idx_dict_1.iteritems():
        for key2,value2 in group_idx_dict_2.iteritems():
            Ns[key1,key2] = count_dict_overlap(value1,value2)
    As = Ns.sum(axis=1)
    Bs = Ns.sum(axis=0)
    return Ns,As,Bs