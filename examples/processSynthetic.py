#!python


import numpy as np
import DPMB as dm
reload(dm)
import DPMB_State as ds
reload(ds)


clusters = 10
points_per_cluster = 100
num_iters = 5
gen_seed = 2
num_sims = 1
##below are fairly static values
cols = 10
beta = .1
infer_hypers = False
alpha = 1 ## dm.mle_alpha(clusters=clusters,points_per_cluster=points_per_cluster) ##
##
inf_seed = 0


paramDict = {"inferAlpha":infer_hypers,"inferBetas":infer_hypers,"alpha":alpha,"beta":beta}
gen_state_with_data = dm.gen_dataset(gen_seed,None,cols,alpha,beta,np.repeat(points_per_cluster,clusters))
##
##gen_sample_output = dm.gen_sample(inf_seed, gen_state_with_data["observables"], num_iters,{"alpha":alpha,"betas":np.repeat(.1,cols)}
##                                  ,None,paramDict=paramDict,gen_state_with_data=gen_state_with_data)
train_data = gen_state_with_data["observables"]
prior_or_gibbs_init = {"alpha":alpha,"betas":np.repeat(.1,cols)}
hyper_method=None
model = dm.DPMB(paramDict=paramDict,state=None,seed=inf_seed)
##will have to pass prior_or_gibbs_init so that alpha can be set from prior (if so specified)
state = ds.DPMB_State(model,paramDict=paramDict,dataset={"xs":train_data},prior_or_gibbs_init=prior_or_gibbs_init) ##z's are generated from CRP if not passed
state.refresh_counts(np.repeat(0,len(state.getZIndices())))
init_num_clusters = state.numClustersDyn()
stats = []
state.debug_conditionals = False
state.print_conditionals = True
state.debug_predictive = False
state.print_predictive = True
state.print_cluster_switch = True
state.vectorIdx_break = 100
print "generative state"
gen_state_with_data["gen_state"]
print "empirical latents"
model.reconstitute_latents()
print "observables"
model.state.getXValues()
model.transition()

for iter_num in range(num_iters):
    model.transition()
    stats.append(model.extract_state_summary())
    if gen_state_with_data is not None:
        latents = model.reconstitute_latents()
        stats[-1]["predictive_prob"] = dm.test_model(gen_state_with_data["test_data"],latents)
        stats[-1]["ari"] = dm.calc_ari(gen_state_with_data["gen_state"]["zs"],latents["zs"])
        if len(stats)>1 and stats[-1]["ari"]!=stats[-2]["ari"]:
            print stats[-1]["ari"]
        if stats[-1]["ari"]==1:
            break

gen_sample_output = {"state":model.reconstitute_latents(),"stats":stats,"init_num_clusters":init_num_clusters}

predictive_prob = dm.test_model(gen_state_with_data["observables"],gen_sample_output["state"])



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
elif False:  ## basic.py style
    gen_state_with_data = dm.gen_dataset(0,1000,20,1,1)
    sample = dm.gen_sample(0,gen_state_with_data,5,None,None,900)
    print [stats["score"] for stats in sample["stats"]] 
    print sample["stats"][0]["predictive_prob"]["gen_prob"]
    print [stats["predictive_prob"]["sampled_prob"] for stats in sample["stats"]]

