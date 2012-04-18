#!python
import datetime,numpy as np,numpy.random as nr,scipy.special as ss,sys
import DPMB_State as ds
reload(ds)
import DPMB_helper_functions as hf
reload(hf)
##
import pdb


class DPMB():
    def __init__(self,paramDict=None,state=None,inf_seed=None):
        if inf_seed is None:
            inf_seed = nr.randint(sys.maxint)
        nr.seed(int(np.clip(inf_seed,0,np.inf)))
        ##
        if paramDict is not None:
            if "__builtins__" in paramDict: paramDict.pop("__builtins__")
            self.__dict__.update(paramDict)
        ##
        self.state = state ## ds.DPMB_State(self,paramDict) if state is None else state

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
    
    def transition_alpha_discrete_gibbs(self):
        self.state.timing.setdefault("alpha",{})["start"] = datetime.datetime.now()
        ##
        lnProdGammas = sum([ss.gammaln(cluster.count()) for cluster in self.state.cluster_list])
        lnPdf = lambda alpha: (ss.gammaln(alpha) + self.state.numClustersDyn()*np.log(alpha)
                               - ss.gammaln(alpha+self.state.numVectorsDyn()) + lnProdGammas)
        ##
        low_val = self.state.infer_alpha["low_val"]
        high_val = self.state.infer_alpha["high_val"]
        n_grid = self.state.infer_alpha["n_grid"]
        grid = 10.0**np.linspace(np.log10(low_val),np.log10(high_val),n_grid) ##endpoint should be set by MLE of all data in its own cluster?
        ##
        logp_list = []
        for test_alpha in grid:
            self.state.removeAlpha(lnPdf)
            self.state.setAlpha(lnPdf,test_alpha)
            logp_list.append(self.state.score)
        alpha_idx = hf.renormalize_and_sample(logp_list,self.state.verbose)
        self.state.removeAlpha(lnPdf)
        self.state.setAlpha(lnPdf,grid[alpha_idx])
        ##
        self.state.infer_alpha_count += 1
        self.state.timing["alpha"]["stop"] = datetime.datetime.now()
        try:
            self.state.timing["alpha"]["delta"] = (self.state.timing["alpha"]["stop"]-self.state.timing["alpha"]["start"]).total_seconds()
        except Exception, e:
            self.state.timing["alpha"]["delta"] = (self.state.timing["alpha"]["stop"]-self.state.timing["alpha"]["start"]).seconds

    def transition_beta_discrete_gibbs(self):
        self.state.timing.setdefault("beta",{})["start"] = datetime.datetime.now()
        ##
        low_val = self.state.infer_beta["low_val"]
        high_val = self.state.infer_beta["high_val"]
        n_grid = self.state.infer_beta["n_grid"]
        grid = 10.0**np.linspace(np.log10(low_val),np.log10(high_val),n_grid) ##endpoint should be set by MLE of all data in its own cluster?
        ##
        logp_list = []
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
            beta_idx = hf.renormalize_and_sample(logp_list)
            self.state.removeBetaD(lnPdf,colIdx)
            self.state.setBetaD(lnPdf,colIdx,grid[beta_idx])
        self.state.infer_betas_count += 1
        self.state.timing["beta"]["stop"] = datetime.datetime.now()
        try:
            self.state.timing["beta"]["delta"] = (self.state.timing["beta"]["stop"]-self.state.timing["beta"]["start"]).total_seconds()
        except Exception, e:
            self.state.timing["beta"]["delta"] = (self.state.timing["beta"]["stop"]-self.state.timing["beta"]["start"]).seconds
        return self.state.betas


    def transition_alpha_mh(self):
        self.state.timing.setdefault("alpha",{})["start"] = datetime.datetime.now()
        initVal = self.state.alpha
        nSamples = 1000
        lnProdGammas = sum([ss.gammaln(cluster.count()) for cluster in self.state.cluster_list])
        lnPdf = lambda alpha: (ss.gammaln(alpha) + self.state.numClustersDyn()*np.log(alpha)
                               - ss.gammaln(alpha+self.state.numVectorsDyn()) + lnProdGammas)
        sampler = lambda x: np.clip(x + nr.normal(0.0,.1),1E-10,np.inf)
        samples = hf.mhSample(initVal,nSamples,lnPdf,sampler)
        newAlpha = samples[-1]
        if np.isfinite(lnPdf(newAlpha)):
            self.state.removeAlpha(lnPdf)
            self.state.setAlpha(lnPdf,newAlpha)
            self.state.infer_alpha_count += 1
        else:
            print "NOT using newAlpha: " + str(newAlpha)
        self.state.timing["alpha"]["stop"] = datetime.datetime.now()
        try:
            self.state.timing["alpha"]["delta"] = (self.state.timing["alpha"]["stop"]-self.state.timing["alpha"]["start"]).total_seconds()
        except Exception, e:
            self.state.timing["alpha"]["delta"] = (self.state.timing["alpha"]["stop"]-self.state.timing["alpha"]["start"]).seconds
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
            samples = hf.mhSample(initVal,nSamples,lnPdf,sampler)
            newBetaD = samples[-1]
            if np.isfinite(lnPdf(newBetaD)):
                self.state.removeBetaD(lnPdf,colIdx)
                self.state.setBetaD(lnPdf,colIdx,newBetaD)
            else:
                print "NOT using beta_d " + str((colIdx,newBetaD)) 
        self.state.infer_betas_count += 1
        self.state.timing["beta"]["stop"] = datetime.datetime.now()
        try:
            self.state.timing["beta"]["delta"] = (self.state.timing["beta"]["stop"]-self.state.timing["beta"]["start"]).total_seconds()
        except Exception, e:
            self.state.timing["beta"]["delta"] = (self.state.timing["beta"]["stop"]-self.state.timing["beta"]["start"]).seconds
        return self.state.betas

    def transition_alpha(self):
        if self.state.verbose:
            print "PRE transition_alpha score: " + str(self.state.score)
        if "method" not in self.state.infer_alpha:
            print "state.infer_alpha does NOT have a \"method\" key"
            return
        if "MH" == self.state.infer_alpha["method"]:
            self.transition_alpha_mh()
        elif "DISCRETE_GIBBS" == self.state.infer_alpha["method"]:
            self.transition_alpha_discrete_gibbs()
        else:
            print "state.infer_alpha[\"method\"]: " + self.state.infer_alpha["method"] + " not understood"

    def transition_beta(self):
        if self.state.verbose:
            print "PRE transition_beta score: " + str(self.state.score)
        if "method" not in self.state.infer_beta:
            print "state.infer_beta does NOT have a \"method\" key"
            return
        if "MH" == self.state.infer_beta["method"]:
            self.transition_beta_mh()
        elif "DISCRETE_GIBBS" == self.state.infer_beta["method"]:
            self.transition_beta_discrete_gibbs()
        else:
            print "state.infer_beta[\"method\"]: " + self.state.infer_beta["method"] + " not understood"
            
    def transition_z(self):
        if self.state.verbose:
            print "PRE transition_z score: " + str(self.state.score)
        self.state.timing.setdefault("zs",{})["start"] = datetime.datetime.now()
        for vectorIdx in range(self.state.numVectors):
            prior_cluster_idx = self.state.zs[vectorIdx].clusterIdx
            self.remove_cluster_assignment(vectorIdx)
            conditionals = self.calculate_cluster_conditional(vectorIdx)
            clusterIdx = hf.renormalize_and_sample(conditionals)
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
            hf.printTS("Starting iteration: " + str(self.state.infer_z_count))
            ##
            self.transition_z()
            if type(self.state.infer_alpha) == dict:
                self.transition_alpha()
            if type(self.state.infer_beta) == dict:
                self.transition_beta()
            ##
            if self.state.verbose:
                print "Cycle end score: " + str(self.state.score)
                print "alpha: " + str(self.state.alpha)
                print "mean beta: " + str(self.state.betas.mean())
                print "empirical phis: ",self.reconstitute_latents()["phis"]
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
        countDict = dict([(actualValue,hf.listCount(testValues[actualValues==actualValue])) for actualValue in np.unique(actualValues)])
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
