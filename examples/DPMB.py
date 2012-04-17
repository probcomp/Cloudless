#!python
import datetime,matplotlib.pylab as plt,numpy as np,numpy.random as nr,scipy.special as ss,sys
import DPMB_State as ds
import pdb
reload(ds)


class DPMB():
    def __init__(self,paramDict=None,state=None,seed=None):
        if seed is None:
            seed = nr.randint(sys.maxint)
        nr.seed(int(np.clip(seed,0,np.inf)))
        ##
        if paramDict is not None:
            if "__builtins__" in paramDict: paramDict.pop("__builtins__")
            self.__dict__.update(paramDict)
        ##
        self.state = ds.DPMB_State(self,paramDict) if state is None else state

    def sample_alpha(self):
        self.state.alpha = nr.gamma(self.state.gamma_k,self.state.gamma_theta)

    def sample_betas(self):
        self.state.betas = nr.gamma(self.state.gamma_k,self.state.gamma_theta,(self.state.numColumns,))
        self.state.betas = np.clip(self.state.betas,self.state.clipBeta[0],self.state.clipBeta[1])

    def sample_zs(self):
        self.state.sample_zs()
        
    def sample_xs(self):
        self.state.sample_xs()

    def reconstitute_thetas(self):
        thetas = np.array([cluster.column_sums/float(len(cluster.vectorIdxList)) for cluster in self.state.cluster_list])
        return thetas

    def reconstitute_phis(self):
        phis = np.array([float(len(cluster.vectorIdxList))/self.state.numVectorsDyn() for cluster in self.state.cluster_list])
        return phis

    def reconstitute_latents(self):
        return {"thetas":self.reconstitute_thetas(),"phis":self.reconstitute_phis(),"zs":self.state.getZIndices()}
    
    def remove_cluster_assignment(self,vectorIdx):
        vector = self.state.xs[vectorIdx]
        cluster = vector.cluster
        cluster.remove_vector(vector)
        
    def assign_vector_to_cluster(self,vectorIdx,clusterIdx):
        vector = self.state.xs[vectorIdx]
        cluster = self.state.cluster_list[clusterIdx] if clusterIdx<self.state.numClustersDyn() else ds.Cluster(self.state)
        cluster.add_vector(vector)
        
    def calculate_cluster_conditional(self,vectorIdx):
        ##vector should be unassigned
        vector = self.state.xs[vectorIdx]
        ##new_cluster is auto appended to cluster list
        ##and pops off when vector is deassigned
        new_cluster = ds.Cluster(self.state)
        conditionals = []
        for cluster in self.state.cluster_list:
            cluster.add_vector(vector)
            conditionals.append(self.state.score)
            cluster.remove_vector(vector)
        return conditionals
    
    def transition_alpha_discrete_gibbs(self,n_grid=10):
        self.state.timing.setdefault("alpha",{})["start"] = datetime.datetime.now()
        ##
        lnProdGammas = sum([ss.gammaln(cluster.count()) for cluster in self.state.cluster_list])
        lnPdf = lambda alpha: (ss.gammaln(alpha) + self.state.numClustersDyn()*np.log(alpha)
                               - ss.gammaln(alpha+self.state.numVectorsDyn()) + lnProdGammas)
        ##
        logp_list = []
        grid = 10.0**np.array(range(-2,-2+n_grid))
        for test_alpha in grid:
            self.state.removeAlpha(lnPdf)
            self.state.setAlpha(lnPdf,test_alpha)
            logp_list.append(self.state.score)
        alpha_idx = renormalize_and_sample(logp_list)
        self.state.removeAlpha(lnPdf)
        self.state.setAlpha(lnPdf,grid[alpha_idx])
        ##
        self.state.infer_alpha_count += 1
        self.state.timing["alpha"]["stop"] = datetime.datetime.now()
        self.state.timing["alpha"]["delta"] = (self.state.timing["alpha"]["stop"]-self.state.timing["alpha"]["start"]).total_seconds()

    def transition_beta_discrete_gibbs(self,n_grid=6):
        self.state.timing.setdefault("beta",{})["start"] = datetime.datetime.now()
        ##
        grid = 10.0**np.array(range(-2,-2+n_grid))
        for colIdx in range(self.state.numColumns):
            S_list = [cluster.column_sums[colIdx] for cluster in self.state.cluster_list]
            R_list = [len(cluster.vectorIdxList) - cluster.column_sums[colIdx] for cluster in self.state.cluster_list]
            beta_d = self.state.betas[colIdx]
            lnPdf = lambda beta_d: sum([ss.gammaln(2*beta_d) - 2*ss.gammaln(beta_d)
                                        + ss.gammaln(S+beta_d) + ss.gammaln(R+beta_d)
                                        - ss.gammaln(S+R+2*beta_d) for S,R in zip(S_list,R_list)])
            logp_list = []
            ##
            for test_beta in grid:
                self.state.removeBetaD(lnPdf,colIdx)
                self.state.setBetaD(lnPdf,colIdx,test_beta)
                logp_list.append(self.state.score)
            beta_idx = renormalize_and_saple(logp_list)
            self.state.removeBetaD(lnPdf)
            self.state.setBetaD(lnPdf,colIdx,grid[beta_idx])
        self.state.infer_betas_count += 1
        self.state.timing["beta"]["stop"] = datetime.datetime.now()
        self.state.timing["beta"]["delta"] = self.state.timing["beta"]["stop"]-self.state.timing["beta"]["start"]
        return self.state.betas


    def transition_alpha_mh(self):
        self.state.timing.setdefault("alpha",{})["start"] = datetime.datetime.now()
        initVal = self.state.alpha
        nSamples = 1000
        lnProdGammas = sum([ss.gammaln(cluster.count()) for cluster in self.state.cluster_list])
        lnPdf = lambda alpha: (ss.gammaln(alpha) + self.state.numClustersDyn()*np.log(alpha)
                               - ss.gammaln(alpha+self.state.numVectorsDyn()) + lnProdGammas)
        sampler = lambda x: np.clip(x + nr.normal(0.0,.1),1E-10,np.inf)
        samples = mhSample(initVal,nSamples,lnPdf,sampler)
        newAlpha = samples[-1]
        if np.isfinite(lnPdf(newAlpha)):
            self.state.removeAlpha(lnPdf)
            self.state.setAlpha(lnPdf,newAlpha)
            self.state.infer_alpha_count += 1
        else:
            print "NOT using newAlpha: " + str(newAlpha)
        self.state.timing["alpha"]["stop"] = datetime.datetime.now()
        self.state.timing["alpha"]["delta"] = (self.state.timing["alpha"]["stop"]-self.state.timing["alpha"]["start"]).total_seconds()
        return samples
        
    def transition_beta_mh(self):
        self.state.timing.setdefault("beta",{})["start"] = datetime.datetime.now()
        nSamples = 100
        sampler = lambda x: np.clip(x + nr.normal(0.0,.1),1E-10,np.inf)
        for colIdx in range(self.state.numColumns):
            initVal = self.state.betas[colIdx]
            S_list = [cluster.column_sums[colIdx] for cluster in self.state.cluster_list]
            R_list = [len(cluster.vectorIdxList) - cluster.column_sums[colIdx] for cluster in self.state.cluster_list]
            beta_d = self.state.betas[colIdx]
            lnPdf = lambda beta_d: sum([ss.gammaln(2*beta_d) - 2*ss.gammaln(beta_d)
                                        + ss.gammaln(S+beta_d) + ss.gammaln(R+beta_d)
                                        - ss.gammaln(S+R+2*beta_d) for S,R in zip(S_list,R_list)])
            samples = mhSample(initVal,nSamples,lnPdf,sampler)
            newBetaD = samples[-1]
            if np.isfinite(lnPdf(newBetaD)):
                self.state.removeBetaD(lnPdf,colIdx)
                self.state.setBetaD(lnPdf,colIdx,newBetaD)
            else:
                print "NOT using beta_d " + str((colIdx,newBetaD)) 
        self.state.infer_betas_count += 1
        self.state.timing["beta"]["stop"] = datetime.datetime.now()
        self.state.timing["beta"]["delta"] = self.state.timing["beta"]["stop"]-self.state.timing["beta"]["start"]
        return self.state.betas

    def transition_alpha(self):
        if self.state.infer_alpha == "MH":
            self.transition_alpha_mh()
        elif self.state.infer_alpha == "DISCRETE_GIBBS":
            self.transition_beta_discrete_gibbs()
        else:
            print "state.infer_alpha: " + str(self.state.infer_alpha) + " not understood"

    def transition_beta(self):
        if self.state.infer_beta == "MH":
            self.transition_beta_mh()
        elif self.state.infer_beta == "DISCRETE_GIBBS":
            self.transition_beta_discrete_gibbs()
        else:
            print "state.infer_beta: " + str(self.state.infer_beta) + " not understood"
            
    def transition_z(self):
        self.state.timing.setdefault("zs",{})["start"] = datetime.datetime.now()
        for vectorIdx in range(self.state.numVectors):
            prior_cluster_idx = self.state.zs[vectorIdx].clusterIdx
            self.remove_cluster_assignment(vectorIdx)
            conditionals = self.calculate_cluster_conditional(vectorIdx)
            clusterIdx = renormalize_and_sample(conditionals)
            if hasattr(self.state,"print_conditionals") and self.state.print_conditionals:
                print clusterIdx,(conditionals-max(conditionals)).round(2)
            if hasattr(self.state,"debug_conditionals") and self.state.debug_conditionals:
                pdb.set_trace()
            if hasattr(self.state,"print_cluster_switch") and self.state.print_cluster_switch and prior_cluster_idx != clusterIdx:
                print "New cluster assignement: ",str(vectorIdx),str(prior_cluster_idx),str(clusterIdx)
            self.assign_vector_to_cluster(vectorIdx,clusterIdx)
            if hasattr(self.state,"vectorIdx_break") and vectorIdx== self.state.vectorIdx_break:
                pdb.set_trace()
        self.state.infer_z_count += 1
        self.state.timing["zs"]["stop"] = datetime.datetime.now()
        self.state.timing["zs"]["delta"] = self.state.timing["zs"]["stop"]-self.state.timing["zs"]["start"]

    def transition(self,numSteps=1):
        for counter in range(numSteps):
            printTS("Starting iteration: " + str(self.state.infer_z_count))
            ##
            if self.state.infer_alpha is not None:
                if self.state.verbose:
                    print "PRE transition_alpha score: " + str(self.state.score)
                self.transition_alpha()
            ##
            if self.state.infer_beta is not None:
                if self.state.verbose:
                        print "PRE transition_beta score: " + str(self.state.score)
                self.transition_beta()
            ##
            if self.state.verbose:
                if False:
                    print "alpha: " + str(self.state.alpha)
                    print "betas: " + str(self.state.betas)
                print "PRE transition_z score: " + str(self.state.score)
            self.transition_z()
            ##
            if self.state.verbose:
                print "Cycle end score: " + str(self.state.score)
                print
            
    def randomize_z(self):
        tempCrp = ds.CRP(self.state.alpha,self.state.numVectors)
        self.state.refresh_counts(tempCrp.zs)

    def compareClustering(self,originalZs,reverse=False):
        if reverse:
            actualValues = np.array(self.state.getZIndices())
            testValues = np.array(originalZs)
        else:
            actualValues = np.array(originalZs)
            testValues = np.array(self.state.getZIndices())
        countDict = dict([(actualValue,listCount(testValues[actualValues==actualValue])) for actualValue in np.unique(actualValues)])
        for actualValue in np.sort(countDict.keys()):
            print str(actualValue) + ": " + str(sum(countDict[actualValue].values())) + " elements"
            print countDict[actualValue]
            print

    def extract_state_summary(self):
        return {
            "hypers":{"alpha":self.state.alpha,"betas":self.state.betas}
            ,"score":self.state.score
            ,"numClusters":self.state.numClustersDyn()
            ,"timing":self.state.timing if len(self.state.timing.keys())>0 else {"zs":{"delta":0},"alpha":{"delta":0},"beta":{"delta":0}}
            ,"infer_z_count":self.state.infer_z_count
            }

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
def gen_dataset(gen_seed, rows, cols, alpha, beta, zDims=None):
    state = ds.DPMB_State(None,{"numVectors":rows,"numColumns":cols,"alpha":alpha,"betas":np.repeat(beta,cols)})
    state.create_data(gen_seed,zDims=zDims)
    train_data = state.getXValues()
    gen_state = {"zs":state.getZIndices(),"thetas":state.getThetas(),"phis":state.getPhis()}
    ##
    state.sample_xs()
    state.refresh_counts()
    test_data = state.getXValues()
    return {"observables":train_data,"gen_state":gen_state,"test_data":test_data}
        
def gen_sample(inf_seed, train_data, num_iters, init_method, hyper_method=None, gen_state_with_data=None, paramDict=None):
    model = DPMB(paramDict=paramDict,state=None,seed=inf_seed)
    ##will have to pass init_method so that alpha can be set from prior (if so specified)
    state = ds.DPMB_State(model,paramDict=paramDict,dataset={"xs":train_data},init_method=init_method) ##z's are generated from CRP if not passed
    state.refresh_counts(np.repeat(0,len(state.getZIndices())))
    ##capture the initial state
    init_state = model.extract_state_summary()
    latents = model.reconstitute_latents()
    init_predictive_prob = None ## test_model(gen_state_with_data["test_data"],latents)
    init_ari = calc_ari(gen_state_with_data["gen_state"]["zs"],latents["zs"])
    ##
    stats = []
    for iter_num in range(num_iters):
        model.transition()
        stats.append(model.extract_state_summary())
        if gen_state_with_data is not None:
            latents = model.reconstitute_latents()
            stats[-1]["predictive_prob"] = test_model(gen_state_with_data["test_data"],latents)
            stats[-1]["ari"] = calc_ari(gen_state_with_data["gen_state"]["zs"],latents["zs"])
    return {"state":model.reconstitute_latents(),"stats":stats,"init_state":{"stats":init_state,"predicitive_prob":init_predictive_prob,"ari":init_ari}}

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

def renormalize_and_sample(logpstar_vec):
  maxv = max(logpstar_vec)
  scaled = [logpstar - maxv for logpstar in logpstar_vec]
  logZ = reduce(np.logaddexp, scaled)
  logp_vec = [s - logZ for s in scaled]
  randv = nr.random()
  for (i, logp) in enumerate(logp_vec):
      p = np.exp(logp)
      if randv < p:
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
        mean_p = np.exp(numerator1 + numerator2 - denominator).mean() if "numerator1" in locals() else 0
        print retVal.round(2),alpha_term.round(2),data_term.round(2),vector.vectorIdx,cluster.clusterIdx,mean_p
    if hasattr(state,"debug_predictive") and state.debug_predictive:
        pdb.set_trace()
        temp = 1 ## if this isn't here, debug start in return and can't see local variables?
    return retVal,alpha_term,data_term

def plot_state(state,gen_state=None,interpolation="nearest",**kwargs):
    ##sort by attributed state and then gen_state if available
    if gen_state is not None:
        mult_factor = np.round(np.log10(len(gen_state["phis"])))
        sort_by = np.array(mult_factor * state.getZIndices() + gen_state["zs"],dtype=int)
    else:
        sort_by = state.getZIndices()
    import pylab
    pylab.ion()
    fh = pylab.figure()
    pylab.imshow(state.getXValues()[np.argsort(sort_by)],interpolation=interpolation,**kwargs)
    ##
    xlim = fh.get_axes()[0].get_xlim()
    h_lines = np.array([cluster.count() for cluster in state.cluster_list]).cumsum()
    pylab.hlines(h_lines-.5,*xlim)
