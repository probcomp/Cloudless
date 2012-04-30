#!python

import numpy as np
import DPMB as dm
reload(dm)
import DPMB_State as ds
reload(ds)
import DPMB_helper_functions as hf
reload(hf)


inf_seed = 0
low_val = .01
high_val = 1E4
##
gen_seed = 0
gen_beta = .1
cols = 256
num_sims = 5 ##propogates to inf_seed
##
alpha_list = [low_val,high_val]
beta_list = [low_val,high_val]
clusters_list = [10]
points_per_cluster_list = [100]
infer_alpha_list = [None,{"method":"DISCRETE_GIBBS","low_val":low_val,"high_val":high_val,"n_grid":100}]
infer_beta_list = [None,{"method":"DISCRETE_GIBBS","low_val":low_val,"high_val":high_val,"n_grid":100}]
init_method_str_list = ["all_together","all_separate","sample_prior"]
num_iters_list = [10]
##alpha = 1 ## dm.mle_alpha(clusters=clusters,points_per_cluster=points_per_cluster) ##


alpha = alpha_list[1]
beta = beta_list[0]
clusters = clusters_list[0]
points_per_cluster = points_per_cluster_list[0]
infer_alpha = infer_alpha_list[1]
infer_beta = infer_beta_list[0]
init_method_str = init_method_str_list[0]
num_iters = num_iters_list[0]


##want to vary alpha,beta according to low_val,high_val
paramDict = {
    "alpha":alpha
    ,"beta":.1
    ,"print_predictive":True
    ,"betas":np.repeat(beta,cols)
    ,"debug_conditionals":False
    ,"print_conditionals":False
    ,"debug_predictive":False
    ,"print_predictive":False
    ,"print_cluster_switch":True
    ,"vectorIdx_break":None
    ,"verbose":True
    }
gen_state_with_data = hf.gen_dataset(gen_seed,gen_rows=None,gen_cols=cols,gen_alpha=1.0,gen_beta=gen_beta,zDims=np.repeat(points_per_cluster,clusters))
##gen_sample_output = hf.gen_sample(inf_seed, gen_state_with_data["observables"], num_iters,{"alpha":alpha,"betas":np.repeat(.1,cols)}
##                                  ,None,paramDict=paramDict,gen_state_with_data=gen_state_with_data)
train_data = gen_state_with_data["observables"]
init_method = {"method":init_method_str}
model = dm.DPMB(inf_seed=inf_seed)
##will have to pass init_method so that alpha can be set from prior (if so specified)
state = ds.DPMB_State(model,paramDict=paramDict,dataset={"xs":train_data},init_method=init_method,infer_alpha=infer_alpha,infer_beta=infer_beta) ##z's are generated from CRP if not passed


init_num_clusters = state.numClustersDyn()
print "generative state"
for k,v in gen_state_with_data["gen_state"].iteritems():
    print k
    print v.round(2)    

print "empirical latents"
for k,v in model.reconstitute_latents().iteritems():
    print k
    print v.round(2)

print "observables"
model.state.getXValues()


model.transition_z()


stats = []
for iter_num in range(num_iters):
    model.transition()
    stats.append(model.extract_state_summary())
    if gen_state_with_data is not None:
        latents = model.reconstitute_latents()
        stats[-1]["predictive_prob"] = hf.test_model(gen_state_with_data["test_data"],latents)
        stats[-1]["ari"] = hf.calc_ari(gen_state_with_data["gen_state"]["zs"],latents["zs"])
        if len(stats)>1 and stats[-1]["ari"]!=stats[-2]["ari"]:
            print stats[-1]["ari"]
        if stats[-1]["ari"]==1:
            break


gen_sample_output = {"state":model.reconstitute_latents(),"stats":stats,"init_num_clusters":init_num_clusters}
predictive_prob = hf.test_model(gen_state_with_data["observables"],gen_sample_output["state"])


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
    gen_state_with_data = hf.gen_dataset(0,1000,20,1,1)
    sample = hf.gen_sample(0,gen_state_with_data,5,None,None,900)
    print [stats["score"] for stats in sample["stats"]] 
    print sample["stats"][0]["predictive_prob"]["gen_prob"]
    print [stats["predictive_prob"]["sampled_prob"] for stats in sample["stats"]]

