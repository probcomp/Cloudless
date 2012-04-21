import datetime,numpy as np,numpy.random as nr,scipy.special as ss,sys
import DPMB_State as ds
import DPMB as dm
##
import pdb

##per vikash's outline: https://docs.google.com/document/d/16iLc2jjtw7Elxy22wyM_TsSPYwW0KOErWKIpFX5k8Y8/edit
# FIXME: later, when we support predictive accuracy assessments, include train/test split stuff here
def gen_problem(gen_seed, gen_cols, gen_rows, gen_alpha, gen_beta, gen_z):
    state = ds.DPMB_State(gen_seed, gen_cols, gen_rows, gen_alpha, gen_beta, gen_z, None)
    ##return state.get_flat_dictionary() ## all you need is test_train_split unless ??
    ##N_test isn't saved in _State initialization.  Unless it is, need to extract test_train_split (at least in addition to flat_dictionary)
    ##alternatively, whoever runs gen_problem should know state generation conditions and test_train_split's return value is simpler to work with
    return state.test_train_split()
##DONE? #FIXME: finish get_flat_dictionary from DPMB_State (mirroring clone)

##run_spec = (dataset_spec,inf_seed, ?num_iters?, init_method,infer_alpha,infer_beta)
##if runspec contains
def infer(run_spec):
    dataset_spec = run_spec["dataset_spec"]
    infer_spec = run_spec["infer_spec"]
    num_iter = run_spec["num_iter"]
    state = ds.DPMB_State(**datset_spec)
    model = dm.DPMB(state=state,**infer_spec)
    ret_list = []
    ##capture initial states
    ret_list.append({"summary":model.extract_state_summary,"flat_dict":state.get_flat_dictionary()})
    for iter_num in range(num_iter):
        model.transition()
        ret_list.append({"summary":model.extract_state_summary,"flat_dict":state.get_flat_dictionary()})
    return ret_list
    # returns a list of dictionaries, one per iter. each dict contains:
    #   - a dict of timing for each kernel
    #   - a state as a flattened dictionary
    # look for max iterations and xs (the training data, for initializing the DPMB_State that inference will be done on) inside run_spec
    # FIXME: Complete
    pass

def gen_sample(inf_seed, train_data, num_iters, init_method, infer_alpha=None, infer_beta=None, gen_state_with_data=None, paramDict=None):
    model = dm.DPMB(inf_seed=inf_seed)
    state = ds.DPMB_State(model,paramDict=paramDict,dataset={"xs":train_data},init_method=init_method,infer_alpha=infer_alpha,infer_beta=infer_beta) ##z's are generated from CRP if not passed
    ##capture the initial state
    init_latents = model.reconstitute_latents()
    init_state = {
        "stats":model.extract_state_summary()
        ##,"predicitive_prob":None if gen_state_with_data is None else test_model(gen_state_with_data["test_data"],init_latents)
        ,"ari":None if gen_state_with_data is None else calc_ari(gen_state_with_data["gen_state"]["zs"],init_latents["zs"])
        }
    ##
    stats = []
    for iter_num in range(num_iters):
        model.transition()
        stats.append(model.extract_state_summary())
        if gen_state_with_data is not None:
            latents = model.reconstitute_latents()
            ##stats[-1]["predictive_prob"] = test_model(gen_state_with_data["test_data"],latents)
            stats[-1]["ari"] = calc_ari(gen_state_with_data["gen_state"]["zs"],latents["zs"])
    return {"state":model.reconstitute_latents(),"stats":stats,"init_state":init_state}

def extract_measurement(which_measurement, one_runs_data):
    # measurement can be:
    # "num_clusters"
    # "alpha"
    # "beta"
    # ("ari", z_indices_vec)
    # work by reconstituting states from the flat dictionary list in one_runs_data[i]["flat_state"] and then applying the desired accessor
    pass

# FIXME: a state should know how to plot itself. calling that method, on a state, should dump out the figure, to a specified filename

def plot_measurement(memoized_infer, which_measurement, which_dataset):
    # FIXME: trawl through memoized_infer.iter(), finding the datasets that match
    # all_runs, finding the datasets matching which_dataset, and then make the pair of plots for which_measurement
    # by first extracting the measurements from memoized_infer 
    pass

def test_model(test_data, sampled_state):
    # computes the sum (not average) predictive probability of the the test_data
    sampled_prob = 0
    for data in test_data:
        sampled_prob += test_model_helper(data,sampled_state["thetas"],sampled_state["phis"])
    return sampled_prob

def test_model_helper(data,thetas,phis):
    boolIdx = np.array(data,dtype=type(True))
    runSum = 0
    for theta,phi in zip(thetas,phis):
        runSum += np.exp(np.log(phi) + np.log(theta[boolIdx]).sum() + np.log(1-theta[~boolIdx]).sum())
    return np.log(runSum)

def calc_ari(group_idx_list_1,group_idx_list_2):
    ##https://en.wikipedia.org/wiki/Rand_index#The_contingency_table
    ##presumes group_idx's are numbered sequentially starting at 0
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

def renormalize_and_sample(logpstar_vec,verbose=False):
  maxv = max(logpstar_vec)
  scaled = [logpstar - maxv for logpstar in logpstar_vec]
  logZ = reduce(np.logaddexp, scaled)
  logp_vec = [s - logZ for s in scaled]
  randv = nr.random()
  for (i, logp) in enumerate(logp_vec):
      p = np.exp(logp)
      if randv < p:
          if verbose:
              print i,np.array(logp_vec).round(2)
          return i
      else:
          randv = randv - p

def cluster_predictive(vector,cluster,state):
    alpha = state.alpha
    numVectors = state.numVectorsDyn() ##this value changes when generating the data
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
        print retVal.round(2),alpha_term.round(2),data_term.round(2),vector.vectorIdx,cluster.cluster_idx,mean_p
    if hasattr(state,"debug_predictive") and state.debug_predictive:
        pdb.set_trace()
        temp = 1 ## if this isn't here, debug start in return and can't see local variables?
    return retVal,alpha_term,data_term

def create_alpha_lnPdf(state):
    lnProdGammas = sum([ss.gammaln(cluster.count()) for cluster in state.cluster_list])
    lnPdf = lambda alpha: (ss.gammaln(alpha) + state.numClustersDyn()*np.log(alpha)
                           - ss.gammaln(alpha+state.numVectorsDyn()) + lnProdGammas)
    return lnPdf

def create_beta_lnPdf(state,col_idx):
    S_list = [cluster.column_sums[col_idx] for cluster in state.cluster_list]
    R_list = [len(cluster.vectorIdxList) - cluster.column_sums[col_idx] for cluster in state.cluster_list]
    beta_d = state.betas[col_idx]
    lnPdf = lambda beta_d: sum([ss.gammaln(2*beta_d) - 2*ss.gammaln(beta_d)
                                + ss.gammaln(S+beta_d) + ss.gammaln(R+beta_d)
                                - ss.gammaln(S+R+2*beta_d) for S,R in zip(S_list,R_list)])
    return lnPdf

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

def printTS(printStr):
    print datetime.datetime.now().strftime("%H:%M:%S") + " :: " + printStr
    sys.stdout.flush()

def listCount(listIn):
    return dict([(currValue,sum(np.array(listIn)==currValue)) for currValue in np.unique(listIn)])


