#!python
import numpy as np
import DPMB as dm
reload(dm)


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
