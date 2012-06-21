#!python
import datetime
import sets
import sys
#
import numpy as np
import scipy.special as ss
import pylab
#
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB.DPMB as dm
reload(dm)
import pyx_functions as pf
reload(pf)
#
import pdb


class PDPMB():
    def __init__(self,inf_seed,state,infer_alpha
                 ,infer_beta,hypers_every_N=1):
        self.random_state = hf.generate_random_state(inf_seed)
        self.state = state
        self.infer_alpha = infer_alpha
        self.infer_beta = infer_beta
        self.hypers_every_N = hypers_every_N
        ##
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
        alpha_idx = pf.renormalize_and_sample(
            np.array(logp_list),self.random_state.uniform())
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

            delta_t = hf.delta_since(start_dt)
            if self.check_time_seatbelt(time_seatbelt,delta_t):
                break

            logp_list, lnPdf, grid = hf.calc_beta_conditional(self.state,col_idx)
            beta_idx = pf.renormalize_and_sample(
                np.array(logp_list),self.random_state.uniform())
            self.state.removeBetaD(lnPdf,col_idx)
            self.state.setBetaD(lnPdf,col_idx,grid[beta_idx])
        # empty everything that was just used to mimic DPMB_State
        self.state.cluster_list = None
        self.state.score = None
        self.state.timing["betas"] = hf.delta_since(start_dt)
        self.state.timing["run_sum"] += self.state.timing["betas"]

    def transition_single_node_assignment(self, cluster):
        node_log_prob_list = hf.calculate_node_conditional(self.state,cluster)
        draw = pf.renormalize_and_sample(
            np.array(node_log_prob_list),self.random_state.uniform())
        to_state = self.state.model_list[draw].state
        self.state.move_cluster(cluster,to_state)

    def get_global_cluster_list(self):
        global_cluster_list = [] #all the clusters in the model
        for model in self.state.model_list:
            global_cluster_list.extend(model.state.cluster_list[:])
        return global_cluster_list

    def transition_node_assignments(self,time_seatbelt=None):
        start_dt = datetime.datetime.now()
        global_cluster_list = self.get_global_cluster_list()

        for cluster in global_cluster_list[:]:
            self.transition_single_node_assignment(cluster)

        self.state.timing["nodes"] = hf.delta_since(start_dt)
        self.state.timing["run_sum"] += self.state.timing["nodes"]

    def transition_z(self,time_seatbelt=None):
        self.state.timing["each_zs"] = []
        start_dt = datetime.datetime.now()
        for model in self.state.model_list:
            model.transition_z()
            individual_z_time = model.state.timing["zs"]
            self.state.timing["each_zs"].append(individual_z_time)
            # don't break here like in DPMB, wait till all individual models complete and break after
            # since they are suppposed run in parallel
        self.state.timing["zs"] = max(self.state.timing["each_zs"])
        self.state.timing["compute_zs"] = sum(self.state.timing["each_zs"])
        self.state.timing["run_sum"] += self.state.timing["zs"]

    def transition_x(self):
        for model in self.state.model_list:
            model.transition_x()

    def transition(self,time_seatbelt=None,ari_seatbelt=None,true_zs=None,exclude_list=None):
        exclude_set = sets.Set(exclude_list)
        hyper_inference_set = sets.Set([
            self.transition_alpha
            ,self.transition_beta
            ,self.transition_node_assignments
            ])
        transition_type_list = [
            self.transition_z
            ,self.transition_alpha
            ,self.transition_beta
            ,self.transition_node_assignments
            ]
        self.transition_count += 1
        transition_hypers = self.transition_count % self.hypers_every_N == 0
        self.state.timing = self.state.create_fresh_timing()

        for transition_type in self.random_state.permutation(
            transition_type_list
            ):
            if transition_type in exclude_set:
                continue
            if transition_type in hyper_inference_set \
                    and not transition_hypers:
                continue
            transition_type(time_seatbelt=time_seatbelt)

        self.check_time_seatbelt(time_seatbelt)
        self.check_ari_seatbelt(ari_seatbelt,true_zs)

        if self.ari_seatbelt_hit or self.time_seatbelt_hit:
            return {"ari_seatbelt_hit":self.ari_seatbelt_hit
                    ,"time_seatbelt_hit":self.time_seatbelt_hit}

    def check_time_seatbelt(self,time_seatbelt=None,delta_t=0):
        if self.time_seatbelt_hit or time_seatbelt is None:
            return self.time_seatbelt_hit
        self.time_seatbelt_hit = self.state.timing["run_sum"] + delta_t > time_seatbelt
        return self.time_seatbelt_hit

    def check_ari_seatbelt(self,ari_seatbelt=None,true_zs=None):
        if self.ari_seatbelt_hit or True or ari_seatbelt is None or true_zs is None:
            # FIXME : how to implement ari seatbelt without a second reconstitute state?
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
            ,"cluster_counts":[len(cluster.vector_list)
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
