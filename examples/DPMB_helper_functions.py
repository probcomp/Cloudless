import datetime,numpy as np,numpy.random as nr,scipy.special as ss,sys
import DPMB_State as ds
import DPMB as dm
##
import pdb

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


##per vikash's outline: https://docs.google.com/document/d/16iLc2jjtw7Elxy22wyM_TsSPYwW0KOErWKIpFX5k8Y8/edit
def gen_dataset(gen_seed, gen_rows, gen_cols, gen_alpha, gen_beta, zDims=None):
    state = ds.DPMB_State(parent=None,paramDict={"numVectors":gen_rows,"numColumns":gen_cols,"alpha":gen_alpha,"betas":np.repeat(gen_beta,gen_cols)})
    state.create_data(gen_seed,zDims=zDims)
    train_data = state.getXValues()
    gen_state = {"zs":state.getZIndices(),"thetas":state.getThetas(),"phis":state.getPhis()}
    ##
    state.sample_xs()
    state.refresh_counts()
    test_data = state.getXValues()
    return {"observables":train_data,"gen_state":gen_state,"test_data":test_data}
        
def gen_sample(inf_seed, train_data, num_iters, init_method, infer_alpha=None, infer_beta=None, gen_state_with_data=None, paramDict=None):
    model = dm.DPMB(inf_seed=inf_seed)
    state = ds.DPMB_State(model,paramDict=paramDict,dataset={"xs":train_data},init_method=init_method) ##z's are generated from CRP if not passed
    ##is this still necessary?
    state.refresh_counts(np.repeat(0,len(state.getZIndices())))
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
        data_term = state.numColumns*np.log(.5)
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
        print retVal.round(2),alpha_term.round(2),data_term.round(2),vector.vectorIdx,cluster.clusterIdx,mean_p
    if hasattr(state,"debug_predictive") and state.debug_predictive:
        pdb.set_trace()
        temp = 1 ## if this isn't here, debug start in return and can't see local variables?
    return retVal,alpha_term,data_term
