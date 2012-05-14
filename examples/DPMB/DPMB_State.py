#!python
import numpy as np
import pylab
import sys
#
import DPMB as dm
reload(dm)
import helper_functions as hf
reload(hf)
##
import pdb


class DPMB_State():
    def __init__(self,gen_seed,num_cols,num_rows,init_alpha=None,init_betas=None
                 ,init_z=None,init_x=None,decanon_indices=None
                 ,alpha_min=.01,alpha_max=1E4,beta_min=.01,beta_max=1E4
                 ,grid_N=100):
        self.random_state = hf.generate_random_state(gen_seed)
        self.num_cols = num_cols
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.grid_N = grid_N
        ##
        self.timing = {"alpha":0,"betas":0,"zs":0,"run_sum":0}
        self.verbose = False
        self.clip_beta = [1E-2,1E10]
        ##
        # note: no score modification here, because of uniform hyperpriors
        self.initialize_alpha(init_alpha)
        self.initialize_betas(init_betas)
        ##
        self.score = 0.0 #initially empty score
        self.cluster_list = [] #all the Cluster s in the model
        self.vector_list = [] #contains all added (possibly unassigned) vectors, in order
        
        # now deal with init_z and init_x specs:

        # FIXME: address issue of Gibbs-type initialization, e.g. to get sharper
        #        results for the convergence of the sampler
        for R in range(num_rows):
            cluster = None
            if init_z is None:
                cluster = self.generate_cluster_assignment()
            elif type(init_z) == int and init_z == 1: ##all in one cluster
                cluster = self.generate_cluster_assignment(force_last=True)
            elif init_z == "N": ##all apart
                cluster = self.generate_cluster_assignment(force_new=True)
            elif isinstance(init_z, tuple) and init_z[0] == "balanced":
                num_clusters = init_z[1]
                mod_val = num_rows / num_clusters
                if mod_val == 0 : 
                    import pdb
                    pdb.set_trace()
                if R % mod_val == 0:
                    # create a new cluster
                    cluster = self.generate_cluster_assignment(force_new=True)
                else:
                    # use the last cluster
                    cluster = self.generate_cluster_assignment(force_last=True)
            elif isinstance(init_z, list) or isinstance(init_z,np.ndarray):
                if init_z[R] >= len(self.cluster_list):
                    cluster = self.generate_cluster_assignment(force_new=True)
                else:
                    cluster = self.cluster_list[init_z[R]]
            else:
                raise Exception("invalid init_z: " + str(init_z))

            # now we have the cluster. handle the data
            if init_x is None:
                vector = self.generate_vector(cluster = cluster)
            else:
                vector = self.generate_vector(data = init_x[R], cluster = cluster)

        if decanon_indices is not None:
            new_cluster_list = []
            for index in decanon_indices:
                new_cluster_list.append(self.cluster_list[index])
            self.cluster_list = new_cluster_list
            

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
    
    def generate_cluster_assignment(self, force_last=False, force_new=False):
        if force_new:
            draw = len(self.cluster_list)
        elif force_last:
            draw = max(0,len(self.cluster_list) - 1)
        else:
            unnorm_vec = [cluster.count() 
                          for cluster in self.cluster_list
                          ] + [self.alpha]
            draw = hf.renormalize_and_sample(

                self.random_state, np.log(unnorm_vec))

        if draw == len(self.cluster_list):
            # create a new cluster and assign it
            cluster = Cluster(self)
            self.cluster_list.append(cluster)
            return cluster
        else:
            return self.cluster_list[draw]

    # sample a cluster from the CRP, possibly resulting in a new one being generated
    # if force_last is True, always returns the last generated cluster in the model
    # if force_new is True, always returns a new cluster
    # force_last and force_new can only both be True if there are no other clusters
    #
    # initializing all apart requires repeated calling with force_new=True
    # initializing all together requires repeated calling with force_last = True
    # initializing from the prior requires no arguments

    # Given:
    # data=None, cluster=None: sample cluster from CRP, sample data from predictive of that cluster.
    # data=None, cluster=c: sample data from c's predictive and incorporate it into c
    # data=dvec, cluster=c: incorporate dvec into c
    #
    # FIXME: Support the case below for GIbbs-type initialization only:
    # data=dvec, cluster=None: sample cluster from the conditional posterior given the data
    def generate_vector(self, data=None, cluster=None):
        if cluster is None:
            cluster = self.generate_cluster_assignment()

        vector = Vector(self.random_state, cluster, data) ## FIXME : does this need to be copied? np.array(data).copy())
        self.vector_list.append(vector)
        cluster.assign_vector(vector)
        return vector

    # remove the given vector from the model, destroying its cluster if needed
    def remove_vector(self, vector):
        # first we deassign it
        vector.cluster.deassign_vector(vector)
        vector.cluster = None ##deassign does this as well
        # now we remove it
        self.vector_list.remove(vector)

    def calculate_log_predictive(self, vector):
        assert vector.cluster is None,("Tried calculate_log_predictive on a" +
                                       " vector already in model. Not kosher!")
        clusters = list(self.cluster_list) + [None]
        log_joints = [hf.cluster_vector_joint(vector, cluster, self)[0]
                      for cluster in clusters]
        log_marginal_on_vector = reduce(np.logaddexp, log_joints)
        return log_marginal_on_vector
    
    def generate_and_score_test_set(self, N_test):
        xs = []
        lls = []
        #
        for i in range(N_test):
            test_vec = self.generate_vector()
            xs.append(test_vec.data)
            self.remove_vector(test_vec)
            lls.append(self.calculate_log_predictive(test_vec))
        #
        return xs,lls

    def score_test_set(self, test_xs):
        lls = []
        #
        for data in test_xs:
            test_vec = self.generate_vector(data=data)
            self.remove_vector(test_vec)
            lls.append(self.calculate_log_predictive(test_vec))
        #
        return lls
                                                            
    def clone(self):
        # FIXME: can't rely on perfect random seed tracking right now. a future pass should make states modified by a journal of operations,
        #        and encapsulate a random source, so that we can control its precise state (and reduce all nondeterminism down to the underlying
        #        nondeterminism of things like Python data structures).
        return DPMB_State(self.gen_seed, self.num_cols, len(self.vector_list), self.alpha, self.betas, self.getZIndices(), self.getXValues(),
                          self.alpha_min, self.alpha_max, self.beta_min, self.beta_max, self.grid_N)

    def get_flat_dictionary(self):
        ##init_* naming is used, but its not really init
        ##makes sense when needed for state creation
        ##but otherwise only if you want to resume inference
        return {"gen_seed":self.gen_seed, "num_cols":self.num_cols, "num_rows": len(self.vector_list), "alpha":self.alpha
                , "betas":self.betas.copy(), "zs":self.getZIndices(), "xs":self.getXValues()
                , "alpha_min":self.alpha_min, "alpha_max":self.alpha_max, "beta_min":self.beta_min, "beta_max":self.beta_max
                , "grid_N":self.grid_N} ## , "N_test":self.N_test} ## N_test isn't save, should it be?
            
    def get_alpha_grid(self):
        ##endpoint should be set by MLE of all data in its own cluster?
        grid = 10.0**np.linspace(np.log10(self.alpha_min),np.log10(self.alpha_max),self.grid_N) 
        return grid
    
    def get_beta_grid(self):
        ##endpoint should be set by MLE of all data in its own cluster?
        grid = 10.0**np.linspace(np.log10(self.beta_min),np.log10(self.beta_max),self.grid_N) 
        return grid

    def get_all_vectors(self):
        return self.vector_list
            
    def get_timing(self):
        return self.timing.copy()

    def getZIndices(self):
        ##CANONICAL zs
        z_indices = []
        next_id = 0
        cluster_ids = {}
        for v in self.get_all_vectors():
            if v.cluster not in cluster_ids:
                cluster_ids[v.cluster] = next_id
                next_id += 1

            z_indices.append(cluster_ids[v.cluster])
        return z_indices

    def get_decanonicalizing_indices(self):
        ##CANONICAL zs
        z_indices = []
        next_id = 0
        cluster_ids = {}
        for v in self.get_all_vectors():
            if v.cluster not in cluster_ids:
                cluster_ids[v.cluster] = next_id
                next_id += 1

            z_indices.append(cluster_ids[v.cluster])

        decanon_indices = []
        for cluster in self.cluster_list:
            decanon_indices.append(cluster_ids[cluster])
        return decanon_indices

    def getXValues(self):
        data = []
        for vec in self.get_all_vectors():
            data.append(vec.data)
        return data

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
            # pdb.set_trace()
            pass # FIXME : for testing gamma = [1,0,0...]
                 #          remove when done
        self.score += scoreDelta

    def plot(self,which_plots=None,which_handles=None,title_append=None,gen_state=None,show=True,save_str=None,**kwargs):
        which_plots = ["data","alpha","beta","cluster"] if which_plots is None else which_plots
        if which_handles is None:
            which_handles = np.repeat(None,len(which_plots))
        handle_lookup = dict(zip(which_plots,which_handles))
        ## try orientation argument to histogram
        if show:
            pylab.ion()
        else:
            pylab.ioff()

        fh1 = None
        fh = pylab.figure()
        pylab.subplot(411)
        if "data" in which_plots or True:
            ##sort by attributed state and then gen_state if available
            if gen_state is not None:
                mult_factor = np.round(np.log10(len(gen_state["phis"])))
                sort_by = np.array(mult_factor * self.getZIndices() + gen_state["zs"],dtype=int)
            else:
                sort_by = self.getZIndices()
            ##plot the data
                ##FIXME : Am I canonicalizing properly
            data = np.array(self.getXValues())[np.argsort(sort_by)]
            h_lines = []
            z_indices = self.getZIndices()
            for z_index in np.unique(z_indices):
                h_lines.append(sum(z_indices==z_index))
            h_lines = np.array(h_lines).cumsum()
            title_str = "Data" if title_append is None else "Data" + ": " + title_append
            ##fh = handle_lookup["data"]
            fh1 = hf.plot_data(data=data,fh=fh,h_lines=h_lines,title_str=title_str)

        fh2 = None
        pylab.subplot(412)
        if "alpha" in which_plots or True:
            logp_list,lnPdf,grid = hf.calc_alpha_conditional(self)
            norm_prob = hf.log_conditional_to_norm_prob(logp_list)
            title_str = "alpha" if title_append is None else "alpha" + ": " + title_append
            ##fh = handle_lookup["alpha"]
            fh2 = hf.bar_helper(x=np.log10(grid),fh=fh,y=norm_prob,v_line=np.log10(self.alpha),title_str=title_str,which_id=1)

        fh3 = None
        pylab.subplot(413)
        if "beta" in which_plots or True:
            beta_idx = 0
            logp_list,lnPdf,grid = hf.calc_beta_conditional(self,beta_idx)
            norm_prob = hf.log_conditional_to_norm_prob(logp_list)
            title_str  = "Beta" if title_append is None else "Beta" + ": " + title_append
            ##fh = handle_lookup["beta"]
            fh3 = hf.bar_helper(x=np.log10(grid),y=norm_prob,fh=fh,v_line=np.log10(self.betas[beta_idx]),title_str=title_str,which_id=2)
            
        fh4 = None
        pylab.subplot(414)
        if ("cluster" in which_plots or True) and len(self.vector_list)>1:
            vector = self.vector_list[0]
            cluster = vector.cluster
            cluster_idx = self.getZIndices()[self.vector_list.index(vector)] ##ALWAYS GO THROUGH getZIndices
            ##
            # calculate the conditional
            cluster.deassign_vector(vector)

            score_vec = hf.calculate_cluster_conditional(self,vector)
            # score_vec is a list of scores, in the order of cluster_list.
            
            if cluster.state is None: ##handle singleton
                cluster = self.generate_cluster_assignment(self,force_new=True)
                cluster.assign_vector(vector)
            else:
                cluster.assign_vector(vector)                
            ##
            # sort score vec according to the order of the clusters from zindices

            next_id = 0
            cluster_ids = {}
            for v in self.get_all_vectors():
                if v.cluster not in cluster_ids:
                    cluster_ids[v.cluster] = next_id
                    next_id += 1

            # cluster_ids has a map from cluster references to z indices
            # but if a singleton was created and not chosen ("autopopped")
            # it is missing

            # move the scores according to the ordering we've found
            sorted_scores = [0 for x in range(len(score_vec))]
            for (i, cluster) in enumerate(self.cluster_list):
                sorted_scores[cluster_ids[cluster]] = score_vec[i]
                
            if len(vector.cluster.vector_list) > 1:
                # we need to put the singleton cluster's score at the end
                sorted_scores[-1] = score_vec[-1]

            norm_prob = hf.log_conditional_to_norm_prob(sorted_scores)
            
            title_str  = "Cluster cond" if title_append is None else "Cluster cond" + ": " + title_append
            ##fh = handle_lookup["cluster"]
            try:
                fh4 = hf.bar_helper(x=np.arange(len(norm_prob))-.5,y=norm_prob,fh=fh,v_line=cluster_idx,title_str=title_str,which_id=3)
            except Exception, e:
                pdb.set_trace()
                print 1
                

        if save_str is not None:
            pylab.savefig(save_str)

        return fh1,fh2,fh3,fh4


class Vector():
    def __init__(self,random_state,cluster,data=None):
        if cluster is None:
            raise Exception("creating a vector without a cluster")
        
        self.cluster = cluster
        if data is None:
            # reconstitute theta from the sufficient statistics for the cluster right now
            num_heads_vec = cluster.column_sums
            N_cluster = cluster.count()
            betas_vec = self.cluster.state.betas
            thetas = [float(num_heads_d + beta_d) / float(N_cluster + 2.0 * beta_d) for (num_heads_d, beta_d) in zip(num_heads_vec, betas_vec)]
            self.data = [random_state.binomial(1, theta) for theta in thetas]
        else:
            self.data = data

class Cluster():
    def __init__(self, state):
        self.state = state
        self.column_sums = np.zeros(self.state.num_cols)
        self.vector_list = []
        
    def count(self):
        return len(self.vector_list)

    def assign_vector(self,vector):
        scoreDelta,alpha_term,data_term = hf.cluster_vector_joint(vector,self,self.state)
        self.state.modifyScore(scoreDelta)
        ##
        self.vector_list.append(vector)
        self.column_sums += vector.data
        vector.cluster = self
        
    def deassign_vector(self,vector):
        vector.cluster = None
        self.column_sums -= vector.data
        self.vector_list.remove(vector)
        ##
        scoreDelta,alpha_term,data_term = hf.cluster_vector_joint(vector,self,self.state)
        self.state.modifyScore(-scoreDelta)
        if self.count() == 0:  ##must remove (self) cluster if necessary
            self.state.cluster_list.remove(self)
            self.state = None
