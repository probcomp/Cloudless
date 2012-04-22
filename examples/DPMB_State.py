#!python
import matplotlib.mlab as mlab,numpy as np, numpy.random as nr,sys
import DPMB as dm
reload(dm)
import DPMB_helper_functions as hf
reload(hf)
##
import pdb


class DPMB_State():
    def __init__(self,gen_seed,num_cols,num_rows,init_alpha=None,init_betas=None,init_z=None,init_x=None
                 ,alpha_min=.01,alpha_max=1E4,beta_min=.01,beta_max=1E4,grid_N=100):

        self.gen_seed = gen_seed
        self.num_cols = num_cols
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.grid_N = grid_N
        ##
        self.timing = {"alpha":0,"betas":0,"zs":0}
        self.verbose = False
        self.clip_beta = [1E-2,1E10]
        nr.seed(int(np.clip(gen_seed,0,np.inf)))
        ##
        # note: no score modification here, because of uniform hyperpriors
        self.alpha = init_alpha if init_alpha is not None else nr.uniform(alpha_min,alpha_max)
        self.betas = init_betas if init_betas is not None else nr.uniform(beta_min,beta_max,self.num_cols)
        ##
        self.score = 0.0 #initially empty score
        self.cluster_list = [] #all the Cluster s in the model
        self.vector_list = [] #contains all added (possibly unassigned) vectors, in order
        
        # now deal with init_z and init_x specs:

        # FIXME: address issue of Gibbs-type initialization, e.g. to get sharper
        #        results for the convergence of the sampler
        for r in range(num_rows):
            cluster = None
            if init_z is None:
                cluster = self.generate_cluster_assignment()
            elif init_z == 1: ##all in one cluster
                cluster = self.generate_cluster_assignment(force_last=True)
            elif init_z == "N": ##all apart
                cluster = self.generate_cluster_assignment(force_new=True)
            elif isinstance(init_z, tuple) and init_z[0] == "balanced":
                num_clusters = init_z[1]
                if r % num_clusters == 0:
                    # create a new cluster
                    cluster = self.generate_cluster_assignment(force_new=True)
                else:
                    # use the last cluster
                    cluster = self.generate_cluster_assignment(force_last=True)
            elif isinstance(init_z, list):
                if init_z[r] > len(self.cluster_list):
                    cluster = self.generate_cluster_assignment(force_new=True)
                else:
                    cluster = self.cluster_list[init_z[r]]
            else:
                raise Exception("invalid init_z: " + str(init_z))

            # now we have the cluster. handle the data
            if init_x is None:
                vector = self.generate_vector(cluster = cluster)
            else:
                vector = self.generate_vector(data = init_x[r], cluster = cluster)

    # sample a cluster from the CRP, possibly resulting in a new one being generated
    # if force_last is True, always returns the last generated cluster in the model
    # if force_new is True, always returns a new cluster
    # force_last and force_new can only both be True if there are no other clusters
    #
    # initializing all apart requires repeated calling with force_new=True
    # initializing all together requires repeated calling with force_last = True
    # initializing from the prior requires no arguments
    def generate_cluster_assignment(self, force_last=False, force_new=False):
        if force_new:
            draw = len(self.cluster_list)
        elif force_last:
            draw = max(0,len(self.cluster_list) - 1)
        else:
            unnorm_vec = [cluster.count() for cluster in self.cluster_list] + [self.alpha]
            draw = hf.renormalize_and_sample(np.log(unnorm_vec))

        if draw == len(self.cluster_list):
            # create a new cluster and assign it
            cluster = Cluster(self)
            self.cluster_list.append(cluster)
            return cluster
        else:
            return self.cluster_list[draw]

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

        vector = Vector(cluster, data)
        self.vector_list.append(vector)
        cluster.assign_vector(vector)
        return vector

    # remove the given vector from the model, destroying its cluster if needed
    def remove_vector(self, vector):
        # first we deassign it
        vector.cluster.deassign_vector(vector)
        vector.cluster = None
        # now we remove it
        self.vector_list.remove(vector)

    def calculate_log_predictive(self, vector):
        if vector.cluster is not None:
            raise Exception("Tried to do this for a vector already in some model. Not kosher!")

        clusters = list(self.cluster_list) + [None]
        log_joints = [hf.cluster_vector_joint(vector, cluster, self) for cluster in clusters]
        log_marginal_on_vector = reduce(np.logaddexp, log_joints)
        return log_marginal_on_vector
    
    def generate_and_score_test_set(self, N_test):
        xs = []
        lls = []

        for i in range(N_test):
            test_vec = self.generate_vector()
            xs.append(test_vec.data)
            self.remove_vector(test_vec)
            lls.append(self.calculate_log_predictive(test_vec))

        return xs,lls
    
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
                , "betas":self.betas, "zs":self.getZIndices(), "xs":self.getXValues()
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
            
    def getZIndices(self):
        z_indices = []
        next_id = 0
        cluster_ids = {}
        for v in self.get_all_vectors():
            if v.cluster not in cluster_ids:
                cluster_ids[v.cluster] = next_id
                next_id += 1

            z_indices.append(cluster_ids[v.cluster])
        return z_indices

    def getXValues(self):
        data = []
        for vec in self.get_all_vectors():
            data.append(vec.data)
    
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

class Vector():
    def __init__(self,cluster,data=None):
        if cluster is None:
            raise Exception("creating a vector without a cluster")
        
        self.cluster = cluster
        if data is None:
            # reconstitute theta from the sufficient statistics for the cluster right now
            num_heads_vec = cluster.column_sums
            N_cluster = cluster.count()
            betas_vec = self.cluster.state.betas
            thetas = [float(num_heads_d + beta_d) / float(N_cluster + 2.0 * beta_d) for (num_heads_d, beta_d) in zip(num_heads_vec, betas_vec)]
            self.data = [np.random.binomial(1, theta) for theta in thetas]
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
