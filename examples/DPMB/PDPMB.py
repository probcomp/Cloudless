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

    def transition_gamma(self):
        start_dt = datetime.datetime.now()
        #
        node_sizes = [len(model.state.vector_list) 
                      for model in self.state.model_list]
        modified_prior = np.array(node_sizes)+self.state.alpha
        self.state.gammas = self.random_state.dirichlet(
            modified_prior,1).tolist()[0]
        #
        self.state.timing["gamma"] = hf.delta_since(start_dt)
        self.state.timing["run_sum"] += self.state.timing["gamma"]

    def transition_alpha(self):
        start_dt = datetime.datetime.now()
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

    def transition_beta(self):
        start_dt = datetime.datetime.now()
        self.state.cluster_list = self.state.get_cluster_list()
        #
        for col_idx in range(self.state.num_cols):
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

    def transition_node_assignments(self):
        start_dt = datetime.datetime.now()
        cluster_list_list = [] #all the clusters in the model
        for model in self.state.model_list:
            cluster_list_list.append(model.state.cluster_list)

        print "pre transition node"
        for state_idx,cluster_list in enumerate(cluster_list_list):
            print "state #" + str(state_idx) \
                + " has " + str(len(cluster_list)) + " clusters"
            print "     " + str([cluster.count() for cluster in cluster_list])
            for cluster in cluster_list:
                self.transition_single_node_assignment(cluster)
        print "post transition node"
        for state_idx,cluster_list in enumerate(cluster_list_list):
            print "state #" + str(state_idx) \
                + " has " + str(len(cluster_list)) + " clusters"
            print "     " + str([cluster.count() for cluster in cluster_list])

        self.state.timing["nodes"] = hf.delta_since(start_dt)
        self.state.timing["run_sum"] += self.state.timing["nodes"]


    def transition_z(self):
        self.state.timing["zs"] = 0
        for model in self.state.model_list:
            model.transition_z()
            self.state.timing["zs"] += model.state.timing["zs"]

    def transition_x(self):
        for model in self.state.model_list:
            model.transition_x()

    def transition(self,exclude_list=None):
        import sets
        exclude_set = sets.Set(exclude_list)
        transition_type_list = [self.transition_z,self.transition_alpha
                                ,self.transition_beta,self.transition_gamma
                                ,self.transition_node_assignments]
        #for transition_type in self.random_state.permutation(
        for transition_type in ( # FIXME : change back to permutation later
            transition_type_list):

            if transition_type in exclude_set:
                continue

            transition_type()
