#!python
import numpy as np
import numpy.random as nr
import scipy.special as ss
import pylab,sys
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
    def __init__(self,init_alpha,init_betas,init_gammas,init_x,gen_seed
                 ,num_nodes
                 ,alpha_min=.01,alpha_max=1E4,beta_min=.01,beta_max=1E4
                 ,grid_N=100):
        self.init_x = init_x
        self.num_cols = len(init_gammas)
        self.num_rows = len(init_x)
        self.num_nodes = num_nodes
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.grid_N = grid_N
        ##
        self.timing = {"alpha":0,"betas":0,"zs":0,"run_sum":0}
        self.verbose = False
        self.clip_beta = [1E-2,1E10]
        hf.set_seed(gen_seed)
        ##
        # note: no score modification here, because of uniform hyperpriors
        self.initialize_alpha(init_alpha)
        self.initialize_betas(init_betas)
        ##
        self.score = 0.0 #initially empty score

        # deal out data to states
        self.gammas = nr.dirichlet(np.repeat(self.alpha,num_nodes),1).tolist()[0]
        log_gammas = np.log(self.gammas)
        node_data_indices = [[] for node_idx in range(self.num_nodes)]
        for data_idx in range(self.num_rows):
            draw = hf.renormalize_and_sample(log_gammas)
            node_data_indices[draw].append(data_idx)
        # now create the child states
        self.model_list = []
        for state_idx in range(num_nodes):
            num_rows_i = len(node_data_indices[state_idx])
            alpha_i = self.alpha * self.gammas[state_idx]
            state = ds.DPMB_State(
                gen_seed=0,num_cols=self.num_cols,num_rows=num_rows_i
                ,init_alpha=alpha_i,init_betas=self.betas)
            model = dm.DPMB(state=state,inf_seed=0
                            ,infer_alpha=False,infer_beta=False)
            self.model_list.append(model)

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
            self.alpha = 10.0**nr.uniform(
                np.log10(alpha_min),np.log10(alpha_max))
            
    def initialize_betas(self,init_betas):
        if init_betas is not None:
            self.betas = np.array(init_betas).copy()
        else:
            self.betas = 10.0**nr.uniform(
                np.log10(beta_min),np.log10(beta_max),self.num_cols)
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
        print "movinng cluster from " + str(from_idx) + " to " + str(to_idx)
        data_list = self.pop_cluster(cluster)
        self.add_cluster(to_state,data_list)

    def transition_single_node_assignment(self, cluster):
        node_log_prob_list = hf.calculate_node_conditional(self,cluster)
        draw = hf.renormalize_and_sample(node_log_prob_list)
        to_state = self.model_list[draw].state
        self.move_cluster(cluster,to_state)

    def transition_node_assignments(self):
        cluster_list_list = [] #all the clusters in the model
        for model in self.model_list:
            cluster_list_list.append(model.state.cluster_list)

        for state_idx,cluster_list in enumerate(cluster_list_list):
            print "state #" + str(state_idx) + " has " + str(len(cluster_list)) + " clusters"
            for cluster in cluster_list:
                self.transition_single_node_assignment(cluster)

    def transition(self):
        for model in self.model_list:
            model.transition()
        self.transition_node_assignments()
        # transition alpha
        # transition betas

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

    def removeAlpha(self,lnPdf):
        scoreDelta = lnPdf(self.alpha)
        self.modifyScore(-scoreDelta)

    def setAlpha(self,lnPdf,alpha):
        scoreDelta = lnPdf(alpha)
        self.modifyScore(scoreDelta)        
        self.alpha = alpha

    def removeBetaD(self,lnPdf,colIdx):
        scoreDelta = lnPdf(self.betas[colIdx])
        self.modifyScore(-scoreDelta)        

    def setBetaD(self,lnPdf,colIdx,newBetaD):
        newBetaD = np.clip(newBetaD,self.clip_beta[0],self.clip_beta[1])
        scoreDelta = lnPdf(newBetaD)
        self.modifyScore(scoreDelta)        
        self.betas[colIdx] = newBetaD

    def modifyScore(self,scoreDelta):
        if not np.isfinite(scoreDelta):
            pdb.set_trace()
        self.score += scoreDelta
