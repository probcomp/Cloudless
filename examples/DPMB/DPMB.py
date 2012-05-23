#!python
import datetime
import numpy as np
import scipy.special as ss
import sys
#
import DPMB_State as ds
reload(ds)
import helper_functions as hf
reload(hf)
##
import pdb

import pyximport
pyximport.install()
import pyx_functions as pf


class DPMB():
    def __init__(self,inf_seed,state,infer_alpha,infer_beta):
        self.random_state = hf.generate_random_state(inf_seed)
        self.state = state
        self.infer_alpha = infer_alpha
        self.infer_beta = infer_beta
        ##
        self.transition_count = 0
        self.time_seatbelt_hit = False
        self.ari_seatbelt_hit = False
    
    def transition_alpha_slice(self,time_seatbelt=None):
        start_dt = datetime.datetime.now()
        if self.check_time_seatbelt(time_seatbelt):
            return # don't transition
        new_alpha = hf.slice_sample_alpha(self.state)
        logp_list,lnPdf,grid = hf.calc_alpha_conditional(self.state)
        self.state.removeAlpha(lnPdf)
        self.state.setAlpha(lnPdf,new_alpha)
        self.state.timing["alpha"] = hf.delta_since(start_dt)
        self.state.timing["run_sum"] += self.state.timing["alpha"]

    def transition_beta_slice(self,time_seatbelt=None):
        start_dt = datetime.datetime.now()
        ##
        for col_idx in range(self.state.num_cols):
            delta_t = hf.delta_since(start_dt)
            if self.check_time_seatbelt(time_seatbelt,delta_t):
                break
            new_beta = hf.slice_sample_beta(self.state,col_idx)
            logp_list, lnPdf, grid = hf.calc_beta_conditional(self.state,col_idx)
            self.state.removeBetaD(lnPdf,col_idx)
            self.state.setBetaD(lnPdf,col_idx,new_beta)
        self.state.timing["betas"] = hf.delta_since(start_dt)
        self.state.timing["run_sum"] += self.state.timing["betas"]

    def transition_alpha_discrete_gibbs(self,time_seatbelt=None):
        start_dt = datetime.datetime.now()
        if self.check_time_seatbelt(time_seatbelt):
            return # don't transition
        ##
        logp_list,lnPdf,grid = hf.calc_alpha_conditional(self.state)
        alpha_idx = pf.renormalize_and_sample(
            np.array(logp_list),self.random_state.uniform())
        self.state.removeAlpha(lnPdf)
        self.state.setAlpha(lnPdf,grid[alpha_idx])
        ##
        self.state.timing["alpha"] = hf.delta_since(start_dt)
        self.state.timing["run_sum"] += self.state.timing["alpha"]

    def transition_beta_discrete_gibbs(self,time_seatbelt=None):
        start_dt = datetime.datetime.now()
        ##
        for col_idx in range(self.state.num_cols):

            delta_t = (datetime.datetime.now() - start_dt).total_seconds()
            if self.check_time_seatbelt(time_seatbelt,delta_t):
                break

            logp_list, lnPdf, grid = hf.calc_beta_conditional(self.state,col_idx)
            beta_idx = pf.renormalize_and_sample(
                np.array(logp_list),self.random_state.uniform())
            self.state.removeBetaD(lnPdf,col_idx)
            self.state.setBetaD(lnPdf,col_idx,grid[beta_idx])

        self.state.timing["betas"] = hf.delta_since(start_dt)
        self.state.timing["run_sum"] += self.state.timing["betas"]

    def transition_alpha(self,time_seatbelt=None):
        if self.state.verbose:
            print "PRE transition_alpha score: ",self.state.score
        if self.infer_alpha:
            self.transition_alpha_discrete_gibbs(time_seatbelt=None)
        else:
            self.state.timing["alpha"] = 0 ##ensure last value not carried forward

    def transition_beta(self,time_seatbelt=None):
        if self.state.verbose:
            print "PRE transition_beta score: ",self.state.score
        if self.infer_beta:
            self.transition_beta_discrete_gibbs(time_seatbelt)
        else:
            self.state.timing["betas"] = 0 ##ensure last value not carried forward
            
    def transition_z(self,time_seatbelt=None):
        if self.state.verbose:
            print "PRE transition_z score: ",self.state.score
        start_dt = datetime.datetime.now()

        for vector in self.random_state.permutation(list(self.state.get_all_vectors())):
            delta_t = (datetime.datetime.now() - start_dt).total_seconds()
            if self.check_time_seatbelt(time_seatbelt,delta_t):
                break
            
            hf.transition_single_z(vector,self.random_state)

        # debug print out states:
        if self.state.verbose:
            # print " --- " + str(self.state.getZIndices())
            print "     " + str([len(cluster.vector_list) 
                                 for cluster in self.state.cluster_list])
        ##
        self.state.timing["zs"] = hf.delta_since(start_dt)
        self.state.timing["run_sum"] += self.state.timing["zs"]

    def transition_x(self):
        # regenerate new vector values, preserving the exact same clustering
        # create a new state, where you force init_z to be the current markov_chain, but you don't pass in init_x
        # then copy out the data vectors from this new state (getXValues())
        # then you replace your state's vector's data fields with these values
        # then you manually or otherwise recalculate the counts and the score --- write a full_score_and_count refresh
        # or do the state-swapping thing, where you switch points

        prior_timing = self.state.timing

        state = ds.DPMB_State(gen_seed=self.random_state # use inference seed
                              ,num_cols=self.state.num_cols
                              ,num_rows=len(self.state.vector_list)
                              ,init_alpha=self.state.alpha
                              ,init_betas=self.state.betas
                              ,init_z=self.state.getZIndices()
                              ,init_x=None
                              ,alpha_min=self.state.alpha_min
                              ,alpha_max=self.state.alpha_max
                              ,beta_min=self.state.beta_min
                              ,beta_max=self.state.beta_max
                              ,grid_N=self.state.grid_N
                              )
        self.state = state
        self.state.timing = prior_timing
    
    def transition(self,numSteps=1, regen_data=False
                   ,time_seatbelt=None,ari_seatbelt=None
                   ,true_zs=None,random_order=True):

        transition_func_list = [self.transition_beta
                                ,self.transition_z
                                ,self.transition_alpha]
        if random_order :
            transition_func_list = self.random_state.permutation(
                transition_func_list)

        for counter in range(numSteps):
            for transition_func in transition_func_list:
                transition_func(time_seatbelt=time_seatbelt)

            if regen_data:
                self.transition_x()
                
            self.check_time_seatbelt(time_seatbelt)
            self.check_ari_seatbelt(ari_seatbelt,true_zs)

            if self.state.verbose:
                hf.printTS("Done iteration: ", self.transition_count)
                print "Cycle end score: ",self.state.score
                print "alpha: ",self.state.alpha
                print "mean beta: ",self.state.betas.mean()

            if self.ari_seatbelt_hit or self.time_seatbelt_hit:
                return {"ari_seatbelt_hit":self.ari_seatbelt_hit
                        ,"time_seatbelt_hit":self.time_seatbelt_hit}

            self.transition_count += 1

        return None

    def check_time_seatbelt(self,time_seatbelt=None,delta_t=0):
        if time_seatbelt is None:
            return self.time_seatbelt_hit
        self.time_seatbelt_hit = self.state.timing["run_sum"] \
            + delta_t > time_seatbelt
        return self.time_seatbelt_hit

    def check_ari_seatbelt(self,ari_seatbelt=None,true_zs=None):
        if ari_seatbelt is None or true_zs is None:
            return self.ari_seatbelt_hit
        self.ari_seatbelt_hit = hf.calc_ari(
            self.state.getZIndices(),true_zs) > ari_seatbelt
        return self.ari_seatbelt_hit

    def extract_state_summary(self,true_zs=None,send_zs=False
                              ,verbose_state=False,test_xs=None):
        
        state_dict = {
            "alpha":self.state.alpha
            ,"betas":self.state.betas.copy()
            ,"score":self.state.score
            ,"num_clusters":len(self.state.cluster_list)
            ,"cluster_counts":[len(cluster.vector_list) 
                               for cluster in self.state.cluster_list]
            ,"timing":self.state.get_timing()
            ,"inf_seed":self.random_state.get_state()
            }

        if true_zs is not None:
            state_dict["ari"] = hf.calc_ari(true_zs,self.state.getZIndices())
        else:
            state_dict["ari"] = None

        if verbose_state or send_zs:
            state_dict["zs"] = self.state.getZIndices()
        if verbose_state:
            state_dict["xs"] = self.state.getXValues()

        if test_xs is not None:
            state_dict["test_lls"] = self.state.score_test_set(test_xs)
        return state_dict
