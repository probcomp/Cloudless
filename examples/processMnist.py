#!python
import gzip,cPickle,os,platform
import numpy as np
import DPMB as dm
reload(dm)


##some general parameters
settingsDict = dict()
exec open(settingsFile) in settingsDict
baseDir = settingsDict["linuxBaseDir"] if (platform.system().lower()=='freebsd' or platform.system().lower()=='linux') else settingsDict["windowsBaseDir"]
fileStr = os.path.join(baseDir,dataDirSuffix,"mnist.pkl.gz")
settingsFile = "DPMB_settings.py"
pixelThresh = .5


##http://deeplearning.net/tutorial/logreg.html
def load_data(dataset):
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)        
    print '... loading data'
    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return (train_set, valid_set, test_set)
##
##if I want to read in the raw data, need to start with this
##http://stackoverflow.com/questions/1035340/reading-binary-file-in-python
##fileStr = "c:/dpmb.data/train-labels-idx1-ubyte"
##with open(fileStr,"rb") as f:


##read in the data
if "train_set" not in locals():
    (train_set, valid_set, test_set) = load_data(fileStr)
else:
    print "train_set already loaded"
##trainset[0] is the data
##trainset[1] is the labels
##trainset[0][i] is a vector of length 784 (28x28)


##alter settings as necessary
settingsDict["numColumns"] = np.shape(train_set[0])[1]
##settingsDict["numVectors"] = 1000 ## shape(train_set[0])[0] ## 
originalZs = train_set[1][:settingsDict["numVectors"]]


##initialize model
model = dm.DPMB(settingsDict)
model.sample_alpha()
model.sample_betas()
model.sample_zs()
model.sample_xs()
for vector in model.state.xs:
    vector.data = np.array(train_set[0][vector.vectorIdx]>pixelThresh,dtype=type(1))
##zs are effectively randomized    
##still need to refresh the counts
model.state.refresh_counts()

##run some transitions  
print "Transitioning alpha,beta to ensure hypers are in a reasonable state given the data"
alphas_considered = model.transition_alpha()
working_betas = model.transition_betas()
zSnapshotList = []
##
model.transition(model.nGibbsSteps)
zSnapshotList.append(np.array(model.state.getZIndices()))
print "What are the actual numbers for my labels?"
model.compareClustering(originalZs,reverse=True)
# print "What are my labels for the actual numbers?"
# model.compareClustering(originalZs)
