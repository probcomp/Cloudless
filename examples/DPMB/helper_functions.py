import datetime,sys,pdb
##
from numpy.random import RandomState
import pylab
import matplotlib
import numpy as np
import scipy.special as ss
##
import DPMB_State as ds

import pyximport
pyximport.install()
import pyx_functions as pf

def transition_single_z(vector,random_state):
    cluster = vector.cluster
    state = cluster.state
    #
    vector.cluster.deassign_vector(vector)
    score_vec = calculate_cluster_conditional(state,vector)
    # FIXME : printing score_vec to be able to compare output of optimized and non-optimized routines
    #         remove when done
    print score_vec
    draw = renormalize_and_sample(random_state, score_vec)
    #
    cluster = None
    if draw == len(state.cluster_list):
        cluster = state.generate_cluster_assignment(force_new = True)
    else:
        cluster = state.cluster_list[draw]
    cluster.assign_vector(vector)
    
####################
# PROBABILITY FUNCTIONS
def renormalize_and_sample(random_state,logpstar_vec):
    p_vec = log_conditional_to_norm_prob(logpstar_vec)
    randv = random_state.uniform()
    for (i, p) in enumerate(p_vec):
        if randv < p:
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
    alpha = state.alpha
    numVectors = len(state.get_all_vectors())
    count = cluster.count() if cluster is not None else 0
    
    if count == 0:
        # if the cluster would be empty without the vector, then its a special case
        alpha_term = np.log(alpha) - np.log(numVectors-1+alpha)
        data_term = state.num_cols*np.log(.5)
    else:
        alpha_term = np.log(cluster.count()) - np.log(numVectors-1+alpha)
        data_term = pf.cluster_vector_joint_helper(
            #np.array(vector.data)
            vector.data
            ,cluster.column_sums
            ,state.betas
            ,cluster.count()
            )
    retVal = alpha_term + data_term

    # retVal,alpha_term,data_term = pf.cluster_vector_joint_helper_2(
    #     alpha
    #     ,numVectors
    #     ,state.num_cols
    #     ,vector.data
    #     ,cluster.column_sums if count != 0 else None
    #     ,state.betas
    #     ,count
    #     )
    return retVal,alpha_term,data_term

def create_alpha_lnPdf(state):
    lnProdGammas = 0 # FIXME : decide whether to entirely remove this
    # lnProdGammas = sum([ss.gammaln(cluster.count()) 
    #                     for cluster in state.cluster_list])
    lnPdf = lambda alpha: ss.gammaln(alpha) \
        + len(state.cluster_list)*np.log(alpha) \
        - ss.gammaln(alpha+len(state.vector_list)) \
        + lnProdGammas
    return lnPdf

def create_beta_lnPdf(state,col_idx):
    S_list = [cluster.column_sums[col_idx] for cluster in state.cluster_list]
    R_list = [len(cluster.vector_list) - cluster.column_sums[col_idx] \
                  for cluster in state.cluster_list]
    beta_d = state.betas[col_idx]
    lnPdf = lambda beta_d: sum([ss.gammaln(2*beta_d) - 2*ss.gammaln(beta_d)
                                + ss.gammaln(S+beta_d) + ss.gammaln(R+beta_d)
                                - ss.gammaln(S+R+2*beta_d) 
                                for S,R in zip(S_list,R_list)])
    return lnPdf

def slice_sample_alpha(state,init=None):
    logprob = create_alpha_lnPdf(state)
    lower = state.alpha_min
    upper = state.alpha_max
    init = state.alpha if init is None else init
    slice = np.log(state.random_state.uniform()) + logprob(init)
    while True:
        a = state.random_state.uniform()*(upper-lower) + lower
        if slice < logprob(a):
            break;
        elif a < init:
            lower = a
        elif a > init:
            upper = a
        else:
            raise Exception('Slice sampler for alpha shrank to zero.')
    return a

def slice_sample_beta(state,col_idx,init=None):
    logprob = create_beta_lnPdf(state,col_idx)
    lower = state.beta_min
    upper = state.beta_max
    init = state.betas[col_idx] if init is None else init
    slice = np.log(state.random_state.uniform()) + logprob(init)
    while True:
        a = state.random_state.uniform()*(upper-lower) + lower
        if slice < logprob(a):
            break;
        elif a < init:
            lower = a
        elif a > init:
            upper = a
        else:
            raise Exception('Slice sampler for beta shrank to zero.')
    return a

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
    original_beta = state.betas[col_idx]
    ##
    state.removeBetaD(lnPdf,col_idx)
    S_list = [cluster.column_sums[col_idx] for cluster in state.cluster_list]
    R_list = [len(cluster.vector_list) - cluster.column_sums[col_idx] \
                  for cluster in state.cluster_list]

    logp_arr = pf.calc_beta_conditional_helper(
        # np.array(S_list)
        # ,np.array(R_list)
        S_list
        ,R_list
        ,grid
        ,state.score
        )
    logp_list = logp_arr.tolist()[0]
    ##
    state.setBetaD(lnPdf,col_idx,original_beta)
    return logp_list,lnPdf,grid

def calculate_cluster_conditional(state,vector):
    ##vector should be unassigned
    ##new_cluster is auto appended to cluster list
    ##and pops off when vector is deassigned

    # FIXME : uncomment when done debuggin
    # new_cluster = ds.Cluster(state)
    # state.cluster_list.append(new_cluster)

    ##
    conditionals = []
    for cluster in state.cluster_list + [None]:
        scoreDelta,alpha_term,data_term = cluster_vector_joint(vector,cluster,state)
        conditionals.append(scoreDelta + state.score)
    ##
    # FIXME : uncomment when done debuggin
    # new_cluster.state.cluster_list.remove(new_cluster)
    # new_cluster.state = None
    
    return conditionals

def calculate_node_conditional(pstate,cluster):
    conditionals = pstate.mus
    return conditionals

def mle_alpha(clusters,points_per_cluster,max_alpha=100):
    mle = 1+np.argmax([ss.gammaln(alpha) + clusters*np.log(alpha) 
                       - ss.gammaln(clusters*points_per_cluster+alpha) 
                       for alpha in range(1,max_alpha)])
    return mle

def mhSample(initVal,nSamples,lnPdf,sampler,random_state):
    samples = [initVal]
    priorSample = initVal
    for counter in range(nSamples):
        unif = random_state.uniform()
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
def plot_data(data,fh=None,h_lines=None,title_str=None
              ,interpolation="nearest",**kwargs):
    if fh is None:
        fh = pylab.figure()
    pylab.imshow(data,interpolation=interpolation
                 ,cmap=matplotlib.cm.binary,**kwargs)
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
        pylab.vlines(v_line,*fh.get_axes()[which_id].get_ylim()
                     ,color="red",linewidth=3)
    if title_str is not None:
        pylab.ylabel(title_str)
    return fh

def printTS(printStr):
    print datetime.datetime.now().strftime("%H:%M:%S") + " :: " + printStr
    sys.stdout.flush()

def listCount(listIn):
    return dict([(currValue,sum(np.array(listIn)==currValue)) 
                 for currValue in np.unique(listIn)])

def delta_since(start_dt):
    try: ##older datetime modules don't have .total_seconds()
        delta = (datetime.datetime.now()-start_dt).total_seconds()
    except Exception, e:
        delta = (datetime.datetime.now()-start_dt).seconds()
    return delta
    
####################
# SEED FUNCTIONS
def generate_random_state(seed):
    random_state = RandomState()
    if type(seed) == tuple:
        random_state.set_state(seed)
    elif type(seed) == int:
        random_state.seed(seed)
    elif type(seed) == RandomState:
        random_state = seed
    else:
        raise Exception("Bad argument to generate_random_state: " + str(seed)) 
    return random_state

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
