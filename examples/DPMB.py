#!python
import datetime,matplotlib.pylab as plt,numpy as np,numpy.random as nr,scipy.special as ss,sys
import DPMB_State as ds
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
        
    def transition_alpha(self):
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
        self.state.timing["alpha"]["delta"] = self.state.timing["alpha"]["stop"]-self.state.timing["alpha"]["start"]
        return samples
        
    def transition_betas(self):
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

    def transition_z(self):
        self.state.timing.setdefault("zs",{})["start"] = datetime.datetime.now()
        for vectorIdx in range(self.state.numVectors):
            self.remove_cluster_assignment(vectorIdx)
            conditionals = self.calculate_cluster_conditional(vectorIdx)
            clusterIdx = renormalize_and_sample(conditionals)
            if hasattr(self.state,"debug") and self.state.debug:
                print clusterIdx,conditionals-max(conditionals)
                import pdb
                pdb.set_trace()
            self.assign_vector_to_cluster(vectorIdx,clusterIdx)
        self.state.infer_z_count += 1
        self.state.timing["zs"]["stop"] = datetime.datetime.now()
        self.state.timing["zs"]["delta"] = self.state.timing["zs"]["stop"]-self.state.timing["zs"]["start"]

    def transition(self,numSteps=1):
        for counter in range(numSteps):
            printTS("Starting iteration: " + str(self.state.infer_z_count))
            ##
            if self.state.inferAlpha:
                if self.state.verbose:
                    print "PRE transition_alpha score: " + str(self.state.score)
                self.transition_alpha()
            ##
            if self.state.inferBetas:
                if self.state.verbose:
                        print "PRE transition_betas score: " + str(self.state.score)
                self.transition_betas()
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
            ,"timing":self.state.timing
            }
    
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
        
def gen_sample(inf_seed, train_data, num_iters, prior_or_gibbs_init, hyper_method, gen_state_with_data=None, paramDict=None):
    model = DPMB(paramDict=paramDict,state=None,seed=inf_seed)
    ##will have to pass prior_or_gibbs_init so that alpha can be set from prior (if so specified)
    state = ds.DPMB_State(model,paramDict=paramDict,dataset={"xs":train_data},prior_or_gibbs_init=prior_or_gibbs_init) ##z's are generated from CRP if not passed
    init_num_clusters = state.numClustersDyn()
    stats = []
    for iter_num in range(num_iters):
        model.transition()
        stats.append(model.extract_state_summary())
        if gen_state_with_data is not None:
            latents = model.reconstitute_latents()
            stats[-1]["predictive_prob"] = test_model(gen_state_with_data["test_data"],latents)
            stats[-1]["ari"] = calc_ari(gen_state_with_data["gen_state"]["zs"],latents["zs"])
    return {"state":model.reconstitute_latents(),"stats":stats,"init_num_clusters":init_num_clusters}

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
