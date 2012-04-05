#!python
import gzip,cPickle,os,platform
import numpy as np
import DPMB as dm
reload(dm)
import DPMB_State as ds
reload(ds)


##some general parameters
settingsFile = "DPMB_settings.py"
settingsDict = dict()
exec open(settingsFile) in settingsDict
baseDir = settingsDict["linuxBaseDir"] if (platform.system().lower()=='freebsd' or platform.system().lower()=='linux') else settingsDict["windowsBaseDir"]
fileStr = os.path.join(baseDir,settingsDict["dataDirSuffix"],"mnist.pkl.gz")
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
##read in the data
if "train_set" not in locals():
    (train_set, valid_set, test_set) = load_data(fileStr)
    ##trainset[0] is the data
    ##trainset[1] is the labels
    ##trainset[0][i] is a vector of length 784 (28x28)
else:
    print "train_set already loaded"


##alter settings as necessary
settingsDict["numColumns"] = np.shape(train_set[0])[1]
##settingsDict["numVectors"] = 1000 ## shape(train_set[0])[0] ## 
originalZs = train_set[1][:settingsDict["numVectors"]]


##initialize model
dm.printTS("creating model")
model = dm.DPMB(paramDict=settingsDict,state=None)
dm.printTS("creating dataset")
dataset = {"zs":originalZs,"xs":train_set[0][:settingsDict["numVectors"],]}
dm.printTS("creating state")
state = ds.DPMB_State(model,paramDict=settingsDict,dataset=dataset)
dm.printTS("done creating model,dataset,state")

import pdb
pdb.set_trace()
##zs are not randomized
##still need to randomize

##verify counts are correct

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
