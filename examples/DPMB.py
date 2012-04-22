#!python
import datetime,numpy as np,numpy.random as nr,scipy.special as ss,sys
import DPMB_State as ds
reload(ds)
import DPMB_helper_functions as hf
reload(hf)
##
import pdb


class DPMB():
    def __init__(self,inf_seed,state,infer_alpha,infer_beta):
        nr.seed(int(np.clip(inf_seed,0,np.inf))) ##who's random seed is used where?  And does it even matter (consider true inf_seed to be f(inf_seed,gen_seed))?
        self.state = state
        self.infer_alpha = infer_alpha
        self.infer_beta = infer_beta
        ##
        self.transition_z_count = 0

    def calculate_cluster_conditional(self,vector):
        ##vector should be unassigned
        ##new_cluster is auto appended to cluster list
        ##and pops off when vector is deassigned

        # FIXME: if there is already an empty cluster (either because deassigning didn't clear it out,
        #        or for some other reason), then we'll have a problem here. maybe a crash, maybe just
        #        incorrect probabilities.
        new_cluster = ds.Cluster(self.state)
        self.state.cluster_list.append(new_cluster)
        
        conditionals = []
        for cluster in self.state.cluster_list:
            cluster.assign_vector(vector)
            conditionals.append(self.state.score)
            cluster.deassign_vector(vector)
            
        return conditionals
    
    def transition_alpha_discrete_gibbs(self):
        start_dt = datetime.datetime.now()
        ##
        grid = self.state.get_alpha_grid()
        lnPdf = hf.create_alpha_lnPdf(self.state)
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
        for colIdx in range(self.state.num_cols):
            lnPdf = hf.create_beta_lnPdf(self.state,colIdx)
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

    def transition_alpha(self):
        if self.state.verbose:
            print "PRE transition_alpha score: ",self.state.score
        if self.infer_alpha:
            self.transition_alpha_discrete_gibbs()
        elif self.infer_alpha:
            self.state.timing["alpha"] = 0 ##ensure last value not carried forward

    def transition_beta(self):
        if self.state.verbose:
            print "PRE transition_beta score: ",self.state.score
        if self.infer_beta:
            self.transition_beta_discrete_gibbs()
        elif self.infer_beta:
            self.state.timing["betas"] = 0 ##ensure last value not carried forward
            
    def transition_z(self):
        self.transition_z_count += 1
        if self.state.verbose:
            print "PRE transition_z score: ",self.state.score
        start_dt = datetime.datetime.now()
        ##
        # for each vector
        for vector in self.state.get_all_vectors():
            # FIXME: randomize order?
            
            # deassign it
            vector.cluster.deassign_vector(vector)
            
            # calculate the conditional
            score_vec = self.calculate_cluster_conditional(vector)

            # sample an assignment
            draw = hf.renormalize_and_sample(score_vec)

            cluster = None
            if draw == len(self.state.cluster_list):
                cluster = self.state.generate_cluster_assignment(force_new = True)
            else:
                cluster = self.state.cluster_list[draw]

            # assign it
            cluster.assign_vector(vector)

            # debug print out states:
            print " --- " + str(self.state.getZIndices())
            print "     " + str([cluster.count() for cluster in self.state.cluster_list])

        ##
        try: ##older datetime modules don't have .total_seconds()
            self.state.timing["zs"] = (datetime.datetime.now()-start_dt).total_seconds()
        except Exception, e:
            self.state.timing["zs"] = (datetime.datetime.now()-start_dt).seconds()

    def transition_x():
        # regenerate new vector values, preserving the exact same clustering
        # create a new state, where you force init_z to be the current markov_chain, but you don't pass in init_x
        # then copy out the data vectors from this new state (getXValues())
        # then you replace your state's vector's data fields with these values
        # then you manually or otherwise recalculate the counts and the score --- write a full_score_and_count refresh
        # or do the state-swapping thing, where you switch points
        
        pass
    
    def transition(self,numSteps=1, regen_data=False):
        for counter in range(numSteps):
            self.transition_z()
            self.transition_alpha() ##may do nothing if infer_alpha == "FIXED"
            self.transition_beta() ##may do nothing if infer_beta == "FIXED"
            if regen_data:
                self.transition_x()
            ##
            if self.state.verbose:
                hf.printTS("Starting iteration: ", self.infer_z_count)
                print "Cycle end score: ",self.state.score
                print "alpha: ",self.state.alpha
                print "mean beta: ",self.state.betas.mean()
                print "empirical phis: ",self.reconstitute_phis()
                print
            
    def extract_state_summary(self):
        print "Z: " + str(self.state.getZIndices())
        print "score: " + str(self.state.score)
        print "num_clusters: " + str(len(self.state.cluster_list))
        
        return {
            "hypers":{"alpha":self.state.alpha,"betas":self.state.betas}
            ,"score":self.state.score
            ,"numClusters":len(self.state.cluster_list)
            ,"timing":self.state.timing
            ,"state":self.state.get_flat_dictionary()
            }
