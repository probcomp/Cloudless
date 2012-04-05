#!python
import numpy as np, numpy.random as nr, matplotlib.mlab as mlab
import pdb

class DPMB_State():
    def __init__(self,parent,paramDict=None):
        self.parent = parent
        self.reset_data()
        ##
        if paramDict is None: return
        if "__builtins__" in paramDict: paramDict.pop("__builtins__")
        self.__dict__.update(paramDict)

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
        
    def create_data(self):
        self.reset_data()
        self.parent.sample_zs()
        self.parent.sample_xs()
        
    def refresh_counts(self,new_zs=None):
        tempZs = [vector.cluster.clusterIdx for vector in self.xs] if new_zs is None else new_zs
        tempXs = self.xs
        self.reset_data()
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
        return [vector.cluster.clusterIdx if vector.cluster is not None else None for vector in self.xs]

    def getThetas(self):
        return [cluster.thetas.copy() for cluster in self.zs]

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
    def __init__(self,cluster):
        self.cluster = cluster
        self.data = [np.random.binomial(1,theta) for theta in self.cluster.thetas]
        self.vectorIdx = len(self.cluster.parent.xs)
        self.cluster.parent.xs.append(self)


class Cluster():
    def __init__(self,parent):
        self.parent = parent
        self.genThetas()
        self.column_sums = np.zeros(np.shape(self.parent.betas))
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
        
    def create_vector(self):
        vector = Vector(self)
        self.add_vector(vector)

    def add_vector(self,vector):
        scoreDelta = cluster_predictive(vector,self,self.parent)
        self.parent.modifyScore(scoreDelta)
        ##self.parent.score += scoreDelta
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
        ##self.parent.score -= scoreDelta
        ##must remove (self) cluster if necessary
        if self.count() == 0:
            replacementCluster = self.parent.cluster_list.pop()
            if self.clusterIdx != len(self.parent.cluster_list):
                replacementCluster.clusterIdx = self.clusterIdx
                self.parent.cluster_list[self.clusterIdx] = replacementCluster
        ##[cluster.clusterIdx for cluster in self.cluster_list] == range(self.numClustersDyn()


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
