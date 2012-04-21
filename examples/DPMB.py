#!python
import datetime,numpy as np,numpy.random as nr,scipy.special as ss,sys
import DPMB_State as ds
reload(ds)
import DPMB_helper_functions as hf
reload(hf)
##
import pdb


class DPMB():
    def __init__(self,state,inf_seed):
        nr.seed(int(np.clip(inf_seed,0,np.inf))) ##who's random seed is used where?  And does it even matter (consider true inf_seed to be f(inf_seed,gen_seed))?
        self.state = state

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
        
    def assign_vector_to_cluster(self,vectorIdx,cluster_idx):
        vector = self.state.xs[vectorIdx]
        cluster = self.state.cluster_list[cluster_idx] if cluster_idx<self.state.numClustersDyn() else ds.Cluster(self.state)
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
        start_dt = datetime.datetime.now()
        ##
        grid = self.state.get_alph_grid()
        lnPdf = hf.create_alpha_lnPdf(self)
        logp_list = []
        for test_alpha in grid:
            self.state.removeAlpha(lnPdf)
            self.state.setAlpha(lnPdf,test_alpha)
            logp_list.append(self.state.score)
        alpha_idx = hf.renormalize_and_sample(logp_list,self.state.verbose)
        self.state.removeAlpha(lnPdf)
        self.state.setAlpha(lnPdf,grid[alpha_idx])
        ##
        try: ##older datetime modules don't have .total_seconds()
            self.state.timing["alpha"] = (datetime.datetime.now()-start_dt).total_seconds()
        except Exception, e:
            self.state.timing["alpha"] = (datetime.datetime.now()-start_dt).seconds()

    def transition_beta_discrete_gibbs(self):
        start_dt = datetime.datetime.now()
        ##
        grid = self.state.get_beta_grid()
        logp_list = []
        for colIdx in range(self.state.numColumns):
            lnPdf = hf.create_beta_lnPdf(self,colIdx)
            logp_list = []
            ##
            for test_beta in grid:
                self.state.removeBetaD(lnPdf,colIdx)
                self.state.setBetaD(lnPdf,colIdx,test_beta)
                logp_list.append(self.state.score)
            beta_idx = hf.renormalize_and_sample(logp_list)
            self.state.removeBetaD(lnPdf,colIdx)
            self.state.setBetaD(lnPdf,colIdx,grid[beta_idx])
        ##
        try: ##older datetime modules don't have .total_seconds()
            self.state.timing["betas"] = (datetime.datetime.now()-start_dt).total_seconds()
        except Exception, e:
            self.state.timing["betas"] = (datetime.datetime.now()-start_dt).seconds()

    def transition_alpha_mh(self):
        start_dt = datetime.datetime.now()
        ##
        initVal = self.state.alpha
        nSamples = 1000
        lnPdf = hf.create_alpha_lnPdf(self)
        sampler = lambda x: np.clip(x + nr.normal(0.0,.1),1E-10,np.inf)
        samples = hf.mhSample(initVal,nSamples,lnPdf,sampler)
        newAlpha = samples[-1]
        if np.isfinite(lnPdf(newAlpha)):
            self.state.removeAlpha(lnPdf)
            self.state.setAlpha(lnPdf,newAlpha)
        else:
            print "NOT using newAlpha: " + str(newAlpha)
        ##
        ##
        try: ##older datetime modules don't have .total_seconds()
            self.state.timing["alpha"] = (datetime.datetime.now()-start_dt).total_seconds()
        except Exception, e:
            self.state.timing["alpha"] = (datetime.datetime.now()-start_dt).seconds()
        
    def transition_beta_mh(self):
        start_dt = datetime.datetime.now()
        ##
        nSamples = 100
        sampler = lambda x: np.clip(x + nr.normal(0.0,.1),1E-10,np.inf)
        for colIdx in range(self.state.numColumns):
            lnPdf = hf.create_beta_lnPdf(self,colIdx)
            initVal = self.state.betas[colIdx]
            samples = hf.mhSample(initVal,nSamples,lnPdf,sampler)
            newBetaD = samples[-1]
            if np.isfinite(lnPdf(newBetaD)):
                self.state.removeBetaD(lnPdf,colIdx)
                self.state.setBetaD(lnPdf,colIdx,newBetaD)
            else:
                print "NOT using beta_d " + str((colIdx,newBetaD))
        ##
        try: ##older datetime modules don't have .total_seconds()
            self.state.timing["betas"] = (datetime.datetime.now()-start_dt).total_seconds()
        except Exception, e:
            self.state.timing["betas"] = (datetime.datetime.now()-start_dt).seconds()

    def transition_alpha(self):
        if self.state.verbose:
            print "PRE transition_alpha score: ",self.state.score
        if self.infer_alpha == "GIBBS"
            self.transition_alpha_discrete_gibbs()
        elif self.infer_alpha == "FIXED":
            self.state.timing["alpha"] = 0 ##ensure last value not carried forward
        else:
            print "infer_alpha: ",infer_alpha," not understood"

    def transition_beta(self):
        if self.state.verbose:
            print "PRE transition_beta score: ",self.state.score
        if self.infer_beta == "GIBBS"
            self.transition_beta_discrete_gibbs()
        elif self.infer_beta == "FIXED":
            self.state.timing["betas"] = 0 ##ensure last value not carried forward
        else:
            print "infer_beta: ",infer_beta," not understood"
            
    def transition_z(self):
        if self.state.verbose:
            print "PRE transition_z score: ",self.state.score
        start_dt = datetime.datetime.now()
        ##
        for vectorIdx in range(self.state.numVectors):
            prior_cluster_idx = self.state.zs[vectorIdx].cluster_idx
            self.remove_cluster_assignment(vectorIdx)
            conditionals = self.calculate_cluster_conditional(vectorIdx)
            cluster_idx = hf.renormalize_and_sample(conditionals)
            ##
            if hasattr(self.state,"print_conditionals") and self.state.print_conditionals:
                print cluster_idx,(conditionals-max(conditionals)).round(2)
            if hasattr(self.state,"debug_conditionals") and self.state.debug_conditionals:
                pdb.set_trace()
            if hasattr(self.state,"print_cluster_switch") and self.state.print_cluster_switch and prior_cluster_idx != cluster_idx:
                print "New cluster assignement: ",str(vectorIdx),str(prior_cluster_idx),str(cluster_idx)
            if hasattr(self.state,"vectorIdx_break") and vectorIdx== self.state.vectorIdx_break:
                pdb.set_trace()
            ##
            self.assign_vector_to_cluster(vectorIdx,cluster_idx)
        ##
        try: ##older datetime modules don't have .total_seconds()
            self.state.timing["zs"] = (datetime.datetime.now()-start_dt).total_seconds()
        except Exception, e:
            self.state.timing["zs"] = (datetime.datetime.now()-start_dt).seconds()

    def transition(self,numSteps=1):
        for counter in range(numSteps):
            hf.printTS("Starting iteration: " + str(self.state.infer_z_count))
            ##
            self.transition_z()
            self.transition_alpha() ##may do nothing if infer_alpha == "FIXED"
            self.transition_beta() ##may do nothing if infer_beta == "FIXED"
            ##
            if self.state.verbose:
                print "Cycle end score: ",self.state.score
                print "alpha: ",self.state.alpha
                print "mean beta: ",self.state.betas.mean()
                print "empirical phis: ",self.reconstitute_phis()
                print
            
    def extract_state_summary(self):
        return {
            "hypers":{"alpha":self.state.alpha,"betas":self.state.betas}
            ,"score":self.state.score
            ,"numClusters":self.state.numClustersDyn()
            ,"timing":self.state.timing if len(self.state.timing.keys())>0 else {"zs":{"delta":0},"alpha":{"delta":0},"beta":{"delta":0}}
            ,"state":self.state.clone()
            }

    def sample_zs(self):
        raise Exception("not implemented")
        
    def sample_xs(self):
        raise Exception("not implemented")
