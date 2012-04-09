#!python
import matplotlib.mlab as mlab,numpy as np, numpy.random as nr,sys
import DPMB as dm
reload(dm)

class DPMB_State():
    def __init__(self,parent,paramDict=None,dataset=None,prior_or_gibbs_init=None):
        self.parent = parent
        if self.parent is not None:
            parent.state = self
        self.reset_data()
        self.timing = {}
        ##default values
        self.verbose = False
        self.inferAlpha = True
        self.inferBetas = True
        self.clipBeta = [1E-2,1E10]
        self.gamma_k = 1
        self.gamma_theta = 1
        ##
        if dataset is not None:
            self.init_from_dataset(dataset,prior_or_gibbs_init)
        ##
        if paramDict is not None:
            if "__builtins__" in paramDict: paramDict.pop("__builtins__")
            self.__dict__.update(paramDict)

    def init_from_dataset(self,dataset,prior_or_gibbs_init):
        self.reset_data()
        self.numColumns = len(dataset["xs"][0])
        self.numVectors = len(dataset["xs"])
        ##alpha,beta must be set before loading data to keep track of score
        if prior_or_gibbs_init is not None:
            self.alpha = prior_or_gibbs_init["alpha"]
            self.betas = prior_or_gibbs_init["betas"]
        else:
            self.parent.sample_alpha()
            self.parent.sample_betas()
        if "zs" in dataset:
            tempZs = dataset["zs"] ##this should not often be the case
        else:
            tempZs = CRP(self.alpha,self.numVectors).zs
        ##
        numClusters = len(np.unique(tempZs))
        for clusterIdx in range(numClusters):
            ##need to initialize clusters first to ensure correct labeling of xs
            ##else, need to modify cluster creation to create null clusters
            ##in the event of an index skip
            Cluster(self) ## links self to state
        for clusterIdx,vector_data in zip(tempZs,dataset["xs"]):
            cluster = self.cluster_list[clusterIdx]
            self.zs.append(cluster)
            cluster.create_vector(vector_data)
    
    def reset_data(self):
        self.zs = []
        self.xs = []
        self.cluster_list = []
        self.score = 0.0
        self.workingVectorIdx = None
        self.workingClusterIdx = None
        self.infer_alpha_count = 0
        self.infer_betas_count = 0
        self.infer_z_count = 0

    def create_data(self,seed=None):
        if seed is None:
            seed = nr.randint(sys.maxint)
        nr.seed(int(np.clip(seed,0,np.inf)))
        self.reset_data()
        self.sample_zs()
        self.sample_xs()

    def sample_zs(self):
        self.zs = []
        tempCrp = CRP(self.alpha,self.numVectors)
        for clusterIdx in tempCrp.zs:
            cluster = self.cluster_list[clusterIdx] if clusterIdx<self.numClustersDyn() else Cluster(self)
            self.zs.append(cluster)
    
    def sample_xs(self):
        self.xs = []
        for cluster in self.zs:
            cluster.create_vector()
    
    def refresh_counts(self,new_zs=None):
        tempZs = [vector.cluster.clusterIdx for vector in self.xs] if new_zs is None else new_zs
        tempXs = self.xs
        self.reset_data()
        numClusters = len(np.unique(tempZs))
        for clusterIdx in range(numClusters):
            Cluster(self)
        ##
        for vector,clusterIdx in zip(tempXs,tempZs):
            cluster = self.cluster_list[clusterIdx] if clusterIdx<self.numClustersDyn() else Cluster(self)
            self.zs.append(cluster)
            self.xs.append(vector)
            cluster.add_vector(vector)

    def fixBrokenLinks(self):
        if self.workingVectorIdx is not None:
            vector = self.xs[workingVectorIdx]
            clusterIdx = self.workingClusterIdx
            cluster = self.cluster_list[clusterIdx] if clusterIdx<self.numClustersDyn() else Cluster(self)
            vector.cluster = cluster
        badIndices = mlab.find(np.array([index is None for index in self.getZIndices()]))
        print "Fixing bad indices: " + str(badIndices)
        for badIdx in badIndices:
            self.xs[badIdx].cluster = self.cluster_list[0]
        self.refresh_counts()

    def numClustersDyn(self):
        return len(self.cluster_list)

    def numVectorsDyn(self):
        return len(self.xs)

    def getZIndices(self):
        return np.array([vector.cluster.clusterIdx if vector.cluster is not None else None for vector in self.xs])

    def getXValues(self):
        return np.array([vector.data for vector in self.xs])
    
    def getThetas(self): ##true thetas
        return np.array([cluster.thetas for cluster in self.cluster_list])

    def getPhis(self): ##?true? phis
        phis = np.array([float(len(cluster.vectorIdxList))/self.numVectorsDyn() for cluster in self.cluster_list])
        return phis


    def removeAlpha(self,lnPdf):
        scoreDelta = lnPdf(self.alpha)
        self.modifyScore(-scoreDelta)
        ##self.score -= scoreDelta

    def setAlpha(self,lnPdf,alpha):
        scoreDelta = lnPdf(alpha)
        self.modifyScore(scoreDelta)        
        ##self.score += scoreDelta
        self.alpha = alpha

    def removeBetaD(self,lnPdf,colIdx):
        scoreDelta = lnPdf(self.betas[colIdx])
        self.modifyScore(-scoreDelta)        
        ##self.score -= scoreDelta

    def setBetaD(self,lnPdf,colIdx,newBetaD):
        newBetaD = np.clip(newBetaD,self.clipBeta[0],self.clipBeta[1])
        scoreDelta = lnPdf(newBetaD)
        ##self.score += scoreDelta
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
            ##should I be determining the draw in some other way
            draw = mlab.find(nr.multinomial(1,modCounts/sum(modCounts)))[0]
            if(draw==len(self.counts)):
                self.counts.append(1)
                self.indexes.append([drawN])
            else:
                self.counts[draw] += 1
                self.indexes[draw].append(drawN)
            self.zs.append(draw)


class Vector():
    def __init__(self,cluster,data=None):
        self.cluster = cluster
        self.data = [np.random.binomial(1,theta) for theta in self.cluster.thetas] if data is None else data
        self.vectorIdx = len(self.cluster.parent.xs)
        self.cluster.parent.xs.append(self)


class Cluster():
    def __init__(self,parent):
        self.parent = parent
        if hasattr(parent,"betas"):
            self.genThetas()
        else:
            self.thetas = np.repeat(np.nan,self.parent.numColumns)
        self.column_sums = np.zeros(self.parent.numColumns)
        self.vectorIdxList = []
        ##
        self.clusterIdx = len(self.parent.cluster_list)
        self.parent.cluster_list.append(self)

    def genThetas(self):
        self.thetas = np.squeeze(np.array([np.random.beta(beta,beta,1) for beta in self.parent.betas]).T)
        if sum([not np.isfinite(theta) for theta in self.thetas])>0:
            self.genThetas()
        
    def count(self):
        return len(self.vectorIdxList)
        
    def create_vector(self,data=None):
        vector = Vector(self,data)
        self.add_vector(vector)

    def add_vector(self,vector):
        scoreDelta = cluster_predictive(vector,self,self.parent)
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
        self.workingClusterIdx = vector.cluster.clusterIdx
        vector.cluster = None
        vectorIdx = vector.vectorIdx
        self.vectorIdxList.remove(vectorIdx)
        self.column_sums -= vector.data
        self.parent.zs[vectorIdx] = None
        ##
        scoreDelta = cluster_predictive(vector,self,self.parent)
        self.parent.modifyScore(-scoreDelta)
        if self.count() == 0:  ##must remove (self) cluster if necessary
            replacementCluster = self.parent.cluster_list.pop()
            if self.clusterIdx != len(self.parent.cluster_list):
                replacementCluster.clusterIdx = self.clusterIdx
                self.parent.cluster_list[self.clusterIdx] = replacementCluster


##if the cluster would be empty without the vector, then its a special case
def cluster_predictive(vector,cluster,state):
    alpha = state.alpha
    numVectors = state.numVectorsDyn() ##this value changes when generating the data
    if cluster is None or cluster.count() == 0:
        retVal = np.log(alpha) - np.log(numVectors-1+alpha) - state.numColumns*np.log(2)
    else:
        boolIdx = np.array(vector.data,dtype=type(True))
        firstFactor = np.log(cluster.count()) - np.log(numVectors-1+alpha)
        secondNumerator1 = cluster.column_sums[boolIdx] + state.betas[boolIdx]
        secondNumerator2 = (cluster.count() - cluster.column_sums[~boolIdx]) + state.betas[~boolIdx]
        secondDenominator = cluster.count() + 2*state.betas
        secondFactor = np.log(secondNumerator1).sum() + np.log(secondNumerator2).sum() - np.log(secondDenominator).sum()
        retVal = firstFactor + secondFactor
    if not np.isfinite(retVal):
        pdb.set_trace()
    return retVal
