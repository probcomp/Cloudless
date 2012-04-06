#!python
import numpy as np
import DPMB as dm
reload(dm)
import DPMB_State as ds
reload(ds)

if False:
    settingsFile = "DPMB_settings.py"
    settingsDict = dict()
    exec open(settingsFile) in settingsDict
    model = dm.DPMB(settingsDict)
    ##sample hypers
    model.sample_alpha()
    model.sample_betas()
    ##create data
    model.state.create_data()


    print "Original data score: " + str(model.state.score)
    originalAlpha = model.state.alpha
    originalBetas = model.state.betas.copy()
    originalThetas = model.state.getThetas()
    originalZs = np.array(model.state.getZIndices())
    ##
    model.randomize_z() ##counts refreshed in randomization
    print "Randomized cluster score: " + str(model.state.score)
    print


    ##run some transitions  
    zSnapshotList = []
    ##
    model.transition(model.nGibbsSteps)
    zSnapshotList.append(np.array(model.state.getZIndices()))
    print "What are the actual numbers for my labels?"
    model.compareClustering(originalZs,reverse=True)
else:  ## basic.py style
    dataset = dm.gen_dataset(0,1000,20,1,1)
    datasetMod = dataset.copy()
    datasetMod["zs"] = ds.CRP(numSamples=len(datasetMod["zs"])).zs
    sample = dm.gen_sample(0,datasetMod,5,None,None)
    print [stats["score"] for stats in sample["stats"]] 
