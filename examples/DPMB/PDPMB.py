#!python

import numpy as np
import scipy.special as ss
import pylab
import sys
import datetime
#
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB.DPMB as dm
reload(dm)
#
import pdb

class PDPMB():
    def __init__(self,inf_seed,state,infer_alpha,infer_beta):
        self.random_state = hf.generate_random_state(inf_seed)
        self.state = state
        self.infer_alpha = infer_alpha
        self.infer_beta = infer_beta
        ##
        self.transition_count = 0
        self.transition_count = 0
        self.time_seatbelt_hit = False
        self.ari_seatbelt_hit = False
        ##
        self.transition_alpha = self.transition_alpha_discrete_gibbs
        self.transition_beta = self.transition_beta_discrete_gibbs

    def transition_alpha_slice(self,time_seatbelt=None):
        start_dt = datetime.datetime.now()
        self.state.cluster_list = self.state.get_cluster_list()
        #
        new_alpha = hf.slice_sample_alpha(self.state)
        logp_list,lnPdf,grid = hf.calc_alpha_conditional(self.state)
        self.state.removeAlpha(lnPdf)
        self.state.setAlpha(lnPdf,new_alpha)
        #
        self.state.cluster_list = None
        self.state.timing["alpha"] = hf.delta_since(start_dt)
        self.state.timing["run_sum"] += self.state.timing["alpha"]

    def transition_beta_slice(self,time_seatbelt=None):
        start_dt = datetime.datetime.now()
        self.state.cluster_list = self.state.get_cluster_list()
        #
        for col_idx in range(self.state.num_cols):
            delta_t = hf.delta_since(start_dt)
            new_beta = hf.slice_sample_beta(self.state,col_idx)
            logp_list, lnPdf, grid = hf.calc_beta_conditional(self.state,col_idx)
            self.state.removeBetaD(lnPdf,col_idx)
            self.state.setBetaD(lnPdf,col_idx,new_beta)
        #
        self.state.cluster_list = None
        self.state.timing["betas"] = hf.delta_since(start_dt)
        self.state.timing["run_sum"] += self.state.timing["betas"]

    def transition_alpha_discrete_gibbs(self,time_seatbelt=None):
        start_dt = datetime.datetime.now()
        if self.check_time_seatbelt(time_seatbelt):
            return # don't transition
        self.state.cluster_list = self.state.get_cluster_list()
        #
        logp_list,lnPdf,grid = hf.calc_alpha_conditional(self.state)
        alpha_idx = hf.renormalize_and_sample(self.random_state, logp_list)
        self.state.removeAlpha(lnPdf)
        self.state.setAlpha(lnPdf,grid[alpha_idx])
        # empty everything that was just used to mimic DPMB_State
        self.state.cluster_list = None
        self.state.timing["alpha"] = hf.delta_since(start_dt)
        self.state.timing["run_sum"] += self.state.timing["alpha"]

    def transition_beta_discrete_gibbs(self,time_seatbelt=None):
        start_dt = datetime.datetime.now()
        self.state.cluster_list = self.state.get_cluster_list()
        #
        for col_idx in range(self.state.num_cols):

            delta_t = (datetime.datetime.now() - start_dt).total_seconds()
            if self.check_time_seatbelt(time_seatbelt,delta_t):
                break

            logp_list, lnPdf, grid = hf.calc_beta_conditional(self.state,col_idx)
            beta_idx = hf.renormalize_and_sample(self.random_state, logp_list)
            self.state.removeBetaD(lnPdf,col_idx)
            self.state.setBetaD(lnPdf,col_idx,grid[beta_idx])
        # empty everything that was just used to mimic DPMB_State
        self.state.cluster_list = None
        self.state.timing["betas"] = hf.delta_since(start_dt)
        self.state.timing["run_sum"] += self.state.timing["betas"]

    def transition_single_node_assignment(self, cluster):
        node_log_prob_list = hf.calculate_node_conditional(self.state,cluster)
        draw = hf.renormalize_and_sample(self.random_state, node_log_prob_list)
        to_state = self.state.model_list[draw].state
        self.state.move_cluster(cluster,to_state)

    def get_cluster_list_list(self):
        cluster_list_list = [] #all the clusters in the model
        for model in self.state.model_list:
            cluster_list_list.append(model.state.cluster_list[:])
        return cluster_list_list

    def transition_node_assignments(self,time_seatbelt=None):
        start_dt = datetime.datetime.now()
        cluster_list_list = self.get_cluster_list_list()

        print "pre transition node"
        for state_idx,cluster_list in enumerate(cluster_list_list):
            print "state #" + str(state_idx) \
                + " has " + str(len(cluster_list)) + " clusters"
            print "     " + str([cluster.count() for cluster in cluster_list])

        for cluster_list in cluster_list_list[:]:
            for cluster in cluster_list[:]:
                self.transition_single_node_assignment(cluster)

        print "post transition node"
        cluster_list_list = self.get_cluster_list_list()
        for state_idx,cluster_list in enumerate(cluster_list_list):
            print "state #" + str(state_idx) \
                + " has " + str(len(cluster_list)) + " clusters"
            print "     " + str([cluster.count() for cluster in cluster_list])

        self.state.timing["nodes"] = hf.delta_since(start_dt)
        self.state.timing["run_sum"] += self.state.timing["nodes"]


    def transition_z(self,time_seatbelt=None):
        self.state.timing["each_zs"] = []
        start_dt = datetime.datetime.now()

        for model in self.state.model_list:

            delta_t = (datetime.datetime.now() - start_dt).total_seconds()
            if self.check_time_seatbelt(time_seatbelt,delta_t):
                break

            model.transition_z()
            individual_z_time = model.state.timing["zs"]
            self.state.timing["each_zs"].append(individual_z_time)
        self.state.timing["compute_zs"] = sum(self.state.timing["each_zs"])
        self.state.timing["zs"] = max(self.state.timing["each_zs"])

    def transition_x(self):
        for model in self.state.model_list:
            model.transition_x()

    def transition(self,time_seatbelt=None,ari_seatbelt=None,true_zs=None,exclude_list=None):
        import sets
        exclude_set = sets.Set(exclude_list)
        transition_type_list = [self.transition_z
                                ,self.transition_alpha
                                ,self.transition_beta
                                ,self.transition_node_assignments]
        for transition_type in self.random_state.permutation(
        #for transition_type in ( # FIXME : change back to permutation later
            transition_type_list):

            if transition_type in exclude_set:
                continue

            transition_type(time_seatbelt=time_seatbelt)

    def check_time_seatbelt(self,time_seatbelt=None,delta_t=0):
        if time_seatbelt is None:
            return self.time_seatbelt_hit
        self.time_seatbelt_hit = self.state.timing["run_sum"] + delta_t > time_seatbelt
        return self.time_seatbelt_hit

    def check_ari_seatbelt(self,ari_seatbelt=None,true_zs=None):
        if ari_seatbelt is None or true_zs is None:
            return self.ari_seatbelt_hit
        self.ari_seatbelt_hit = hf.calc_ari(self.state.getZIndices(),true_zs) > ari_seatbelt
        return self.ari_seatbelt_hit

    def extract_state_summary(self,true_zs=None,send_zs=False,verbose_state=False,test_xs=None):

        psuedo_state = self.state.create_single_state()
        
        state_dict = {
            "alpha":self.state.alpha
            ,"betas":self.state.betas.copy()
            ,"score":psuedo_state.score
            ,"num_clusters":len(psuedo_state.cluster_list)
            ,"cluster_counts":[cluster.count() 
                               for cluster in psuedo_state.cluster_list]
            ,"timing":self.state.get_timing()
            ,"inf_seed":self.random_state.get_state()
            }

        if true_zs is not None:
            state_dict["ari"] = hf.calc_ari(true_zs,psuedo_state.getZIndices())
        else:
            state_dict["ari"] = None

        if verbose_state or send_zs:
            state_dict["zs"] = psuedo_state.getZIndices()
        if verbose_state:
            state_dict["xs"] = self.state.getXValues()

        if test_xs is not None:
            state_dict["test_lls"] = psuedo_state.score_test_set(test_xs)
        return state_dict
