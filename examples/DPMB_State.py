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
                 ,alpha_min=.01,alpha_max=1E4,beta_min=.01,beta_max=1E4,grid_N=100,N_test=0):
        self.gen_seed = gen_seed
        self.num_cols = num_cols
        self.num_rows = num_rows + N_test ##this is fine as long as states generating test data are not then used to initialize inference
        self.init_alpha = init_alpha
        self.init_betas = init_betas
        self.init_z = init_z
        self.init_x = init_x
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
        self.zs = [] #will eventually get a list of Cluster references

        self.init_z_func()

        self.xs = [] #will eventually get a list of Vector references

        self.init_x_func()

    def clone(self):
        ##I don't think this is quite correct
        ##To maintain seed state after generation, should just recreate the state, so pass self.init_z, not self.getZIndices()
        return DPMB_State(self.gen_seed, self.num_cols, self.num_rows, self.alpha, self.betas, self.getZIndices(), self.getXValues(),
                          self.alpha_min, self.alpha_max, self.beta_min, self.beta_max, self.grid_N)
    ##so perhaps test_train_split should be replaced with gen_test, which uses clone to create a new state, create N_test more values and keep only the last N_test of z,x

    def get_flat_dictionary(self):
        ##init_* naming is used, but its not really init
        ##makes sense when needed for state creation
        ##but otherwise only if you want to resume inference
        return {"gen_seed":self.gen_seed, "num_cols":self.num_cols, "num_rows":self.num_rows, "init_alpha":self.alpha
                , "init_betas":self.betas, "init_z":self.getZIndices(), "init_x":self.getXValues()
                , "alpha_min":self.alpha_min, "alpha_max":self.alpha_max, "beta_min":self.beta_min, "beta_max":self.beta_max
                , "grid_N":self.grid_N} ## , "N_test":self.N_test} ## N_test isn't save, should it be?
            
    def test_train_split(self, N_test):
        assert N_test > 0
        assert N_test <= self.num_rows
        
        all_data = self.getXValues()
        out = {}
        out["test_data"] = all_data[-N_test:]
        out["train_data"] = all_data[:-N_test]
        out["train_zs"] = self.getZIndices()[:-N_test]
        return out
    
    def init_z_func(self):
        ##
        if self.init_z is None: ##sample
            zs = CRP(self.alpha,self.num_rows).zs
        elif self.init_z == 1: ##all in one
            zs = np.repeat(0,self.num_rows)
        elif self.init_z == "N": ##all apart
            zs = range(self.num_rows)
        elif type(self.init_z)==tuple and self.init_z[0]=="balanced":
            num_clusters = init_z[1]
            zs = np.repeat(range(num_clusters),np.int(np.ceil(float(self.num_rows)/num_clusters))[:self.num_rows])
            zs = np.random.permutation(zs)
        elif isinstance(self.init_z, list):
            zs = self.init_z
        else:  ## init_method has a method key but it didn't match known entries
            raise Exception("invalid init_z passed to DPMB_State.create_data")

        num_clusters = len(np.unique(zs))
        for cluster_idx in range(num_clusters):
            ##must initialize clusters first to get correct labeling of xs, else index skip results in error
            Cluster(self) ## links self to state
        # now self.cluster_list is initialized!
        
        for cluster_idx in zs:
            self.zs.append(self.cluster_list[cluster_idx])

        
            
    def init_x_func(self):
        ##
        xs = self.init_x if self.init_x is not None else np.repeat(None,self.num_rows)
        ##
        for (cluster, vector_data) in zip(self.zs, xs):
            cluster.create_vector(vector_data)

    def get_alpha_grid(self):
        ##endpoint should be set by MLE of all data in its own cluster?
        grid = 10.0**np.linspace(np.log10(self.alpha_min),np.log10(self.alpha_max),self.grid_N) 

    def get_beta_grid(self):
        ##endpoint should be set by MLE of all data in its own cluster?
        grid = 10.0**np.linspace(np.log10(self.beta_min),np.log10(self.beta_max),self.grid_N) 

    def numClustersDyn(self):
        return len(self.cluster_list)

    def numVectorsDyn(self):
        return len(self.xs)

    def getZIndices(self):
        return np.array([vector.cluster.cluster_idx if vector.cluster is not None else None for vector in self.xs])

    def getXValues(self):
        return np.array([vector.data for vector in self.xs])
    
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

class CRP():
    def __init__(self,alpha=1,numSamples=0):
        self.alpha = alpha
        self.zs = []
        self.counts = []
        self.indexes = []
        if numSamples>0:
            self.sample(numSamples)
            
    def sample(self,numSamples=1):
        for currNDraw in range(numSamples):
            drawN = len(self.zs)
            modCounts = np.array(np.append(self.counts,self.alpha),dtype=type(1.0))
            draw = hf.renormalize_and_sample(np.log(modCounts))
            if(draw==len(self.counts)):
                self.counts.append(1)
                self.indexes.append([drawN])
            else:
                self.counts[draw] += 1
                self.indexes[draw].append(drawN)
            self.zs.append(draw)
        return self


class Vector():
    def __init__(self,cluster,data=None):
        self.cluster = cluster
        self.data = [np.random.binomial(1,theta) for theta in self.cluster.thetas] if data is None else data
        if self.cluster is not None:
            self.vectorIdx = len(self.cluster.parent.xs)
            self.cluster.parent.xs.append(self)


class Cluster():
    def __init__(self,parent):
        self.parent = parent

        self.genThetas()
        self.column_sums = np.zeros(self.parent.num_cols)
        self.vectorIdxList = []
        ##
        self.cluster_idx = len(self.parent.cluster_list)
        self.parent.cluster_list.append(self)

    def genThetas(self,recurrences=0):
        self.thetas = np.squeeze(np.array([np.random.beta(beta,beta,1) for beta in self.parent.betas]).T)
        if sum([not np.isfinite(theta) for theta in self.thetas])>0:
            print "genThetas recurrences: ",recurrences+1
            self.genThetas(recurrences+1)
        
    def count(self):
        return len(self.vectorIdxList)
        
    def create_vector(self,data=None):
        vector = Vector(self,data)
        self.add_vector(vector)

    def add_vector(self,vector):
        scoreDelta,alpha_term,data_term = hf.cluster_predictive(vector,self,self.parent)
        self.parent.modifyScore(scoreDelta)
        ##
        vector.cluster = self
        vectorIdx = vector.vectorIdx
        self.vectorIdxList.append(vectorIdx)
        self.column_sums += vector.data
        self.parent.zs[vectorIdx] = self
        self.workingVectorIdx = None
        self.workingClusterIdx = None
        
    def remove_vector(self,vector):
        self.workingVectorIdx = vector.vectorIdx
        self.workingClusterIdx = vector.cluster.cluster_idx
        vector.cluster = None
        vectorIdx = vector.vectorIdx
        self.vectorIdxList.remove(vectorIdx)
        self.column_sums -= vector.data
        self.parent.zs[vectorIdx] = None
        ##
        scoreDelta,alpha_term,data_term = hf.cluster_predictive(vector,self,self.parent)
        self.parent.modifyScore(-scoreDelta)
        if self.count() == 0:  ##must remove (self) cluster if necessary
            replacementCluster = self.parent.cluster_list.pop()
            if self.cluster_idx != len(self.parent.cluster_list):
                replacementCluster.cluster_idx = self.cluster_idx
                self.parent.cluster_list[self.cluster_idx] = replacementCluster
