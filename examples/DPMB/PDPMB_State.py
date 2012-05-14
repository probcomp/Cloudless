#!python
import numpy as np
import scipy.special as ss
import pylab,sys
import datetime
#
import Cloudless.examples.DPMB.PDPMB as pdm
reload(pdm)
import Cloudless.examples.DPMB.DPMB as dm
reload(dm)
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
##
import pdb


class PDPMB_State():
    def __init__(self,gen_seed,num_cols,num_rows
                 ,num_nodes,init_gammas=None
                 ,init_alpha=None,init_betas=None
                 ,init_z=None,init_x=None
                 ,alpha_min=.01,alpha_max=1E4
                 ,beta_min=.01,beta_max=1E4
                 ,grid_N=100):
        self.random_state = hf.generate_random_state(gen_seed)
        self.num_cols = num_cols
        self.num_nodes = num_nodes
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.grid_N = grid_N
        ##
        self.timing = {"alpha":0,"betas":0,"zs":0,"nodes":0,"gamma":0,"run_sum":0}
        self.verbose = False
        self.clip_beta = [1E-2,1E10]
        # FIXME : setting seed becomes complicated when child states 
        #         will also do seed operations

        # generate the data from a DPMB_State
        state = ds.DPMB_State(gen_seed,
                              self.num_cols,
                              num_rows,
                              init_alpha = init_alpha,
                              init_betas = init_betas,
                              init_z = init_z,
                              init_x = init_x)
        self.vector_list = state.vector_list
        ##
        # note: no score modification here, because of uniform hyperpriors
        self.initialize_alpha(init_alpha)
        self.initialize_betas(init_betas)
        self.initialize_gammas(init_gammas)
        ##
        self.score = 0.0 #initially empty score

        # deal out data to states
        log_gammas = np.log(self.gammas)
        node_data_indices = [[] for node_idx in range(self.num_nodes)]
        for data_idx in range(len(self.vector_list)):
            draw = hf.renormalize_and_sample(self.random_state,log_gammas)
            node_data_indices[draw].append(data_idx)
        # now create the child states
        self.model_list = []
        seed_list = [int(x) for x in self.random_state.tomaxint(num_nodes)]
        for state_idx,gen_seed in enumerate(seed_list):
            node_data = []
            for index in node_data_indices[state_idx]:
                node_data.append(self.vector_list[index].data)

            alpha_i = self.alpha * self.gammas[state_idx]
            state = ds.DPMB_State(
                gen_seed=gen_seed,num_cols=self.num_cols,num_rows=len(node_data)
                ,init_alpha=alpha_i,init_betas=self.betas
                ,init_x=node_data)
            model = dm.DPMB(state=state,inf_seed=gen_seed
                            ,infer_alpha=False,infer_beta=False)
            self.model_list.append(model)

    def removeAlpha(self,lnPdf):
        self.alpha = None
        self.score = None

    def setAlpha(self,lnPdf,test_alpha):
        self.alpha = test_alpha
        for model,gamma_i in zip(self.model_list,self.gammas):
            lnPdf = hf.create_alpha_lnPdf(model.state)
            model.state.removeAlpha(lnPdf)
            model.state.setAlpha(lnPdf, self.alpha*gamma_i)
        self.score = lnPdf(test_alpha)

    def removeBetaD(self,lnPdf,col_idx):
        self.betas[col_idx] = None
        self.score = None
        
    def setBetaD(self,lnPdf,col_idx,beta_val):
        beta_val = np.clip(beta_val,self.clip_beta[0],self.clip_beta[1])
        self.betas[col_idx] = beta_val
        for model,gamma_i in zip(self.model_list,self.gammas):
            lnPdf = hf.create_beta_lnPdf(model.state,col_idx)
            model.state.removeBetaD(lnPdf,col_idx)
            model.state.setBetaD(lnPdf,col_idx,beta_val)
        self.score = lnPdf(beta_val)

    def get_cluster_list(self):
        cluster_list = []
        for model in self.model_list:
            cluster_list.extend(model.state.cluster_list)
        return cluster_list

    def N_score_component(self):
        counts = np.array([len(model.state.vector_list) 
                           for model in self.model_list])
        first_part = ss.gammaln(sum(counts)+1) - sum(ss.gammaln(counts+1))
        second_part = sum(counts*np.log(self.gammas))
        N_score = first_part + second_part
        return N_score,first_part,second_part

    def gamma_score_component(self):
        alpha = self.alpha # /self.num_nodes
        first_part = (ss.gammaln(float(alpha)*self.num_nodes)
                      - self.num_nodes*ss.gammaln(float(alpha)))
        second_part = (alpha-1) * sum(np.log(self.gammas))
        gamma_score = first_part + second_part
        return gamma_score,first_part,second_part

    def create_single_state(self):
        single_state = None
        cluster_idx = 0
        xs = []
        zs = []
        for model in self.model_list:
            state = model.state
            xs.extend(state.getXValues())
            temp_zs = state.getZIndices()
            if len(temp_zs) == 0:
                continue
            max_zs = temp_zs[-1]
            zs.extend(np.array(temp_zs) + cluster_idx)
            cluster_idx += max_zs
        single_state = ds.DPMB_State(
            gen_seed=0,num_cols=self.num_cols,num_rows=self.num_rows
            ,init_alpha=self.alpha,init_betas=self.betas
            ,init_x=xs,init_z=zs)
        return single_state

    def initialize_alpha(self,init_alpha):
        if init_alpha is not None:
            self.alpha = init_alpha
        else:
            self.alpha = 10.0**self.random_state.uniform(
                np.log10(self.alpha_min),np.log10(self.alpha_max))
            
    def initialize_betas(self,init_betas):
        if init_betas is not None:
            self.betas = np.array(init_betas).copy()
        else:
            self.betas = 10.0**self.random_state.uniform(
                np.log10(self.beta_min),np.log10(self.beta_max),self.num_cols)
        pass

    def initialize_gammas(self,init_gammas):
        if init_gammas is not None:
            self.gammas = np.array(init_gammas).copy()
        else:
            self.gammas = self.random_state.dirichlet(
                np.repeat(self.alpha,num_nodes),1).tolist()[0]
        pass

    def pop_cluster(self,cluster):
        data_list = []
        state = cluster.state
        for vector in cluster.vector_list:
            data_list.append(vector.data)
            state.remove_vector(vector)
        return data_list

    def add_cluster(self,state,data_list):
        new_cluster = state.generate_cluster_assignment(force_new=True)
        for data in data_list:
            state.generate_vector(data=data,cluster=new_cluster)

    def find_state_idx(self,state):
        for state_idx,model in enumerate(self.model_list):
            if state == model.state:
                return state_idx
        return None

    def move_cluster(self,cluster,to_state):
        if cluster.state is to_state:
            return # don't bother if from_state == to_state
        from_idx = self.find_state_idx(cluster.state)
        to_idx = self.find_state_idx(to_state)
        data_list = self.pop_cluster(cluster)
        self.add_cluster(to_state,data_list)

    def get_alpha_grid(self):
        ##endpoint should be set by MLE of all data in its own cluster?
        grid = 10.0**np.linspace(np.log10(self.alpha_min),np.log10(self.alpha_max),self.grid_N) 
        return grid
    
    def get_beta_grid(self):
        ##endpoint should be set by MLE of all data in its own cluster?
        grid = 10.0**np.linspace(np.log10(self.beta_min),np.log10(self.beta_max),self.grid_N) 
        return grid

    def get_timing(self):
        return self.timing.copy()
