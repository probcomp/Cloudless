import DPMB_basic as b

low_val=.01
high_val=1E4
packed_params = {
    "path":"Plots"
    ,"CLUSTERS":10
    ,"POINTS_PER_CLUSTER":10
    ,"NUM_ITERS":10
    ,"INFER_ALPHA":[None,{"method":"DISCRETE_GIBBS","low_val":low_val,"high_val":high_val,"n_grid":100}][1]
    ,"INFER_BETA":[None,{"method":"DISCRETE_GIBBS","low_val":low_val,"high_val":high_val,"n_grid":100}][0]
    ,"INIT_METHOD":["all_together","all_separate","sample_prior"][1]
    ##BELOW ARE FAIRLY STATIC VALUES
    ,"ALPHA":100 ## hf.mle_alpha(clusters=CLUSTERS,points_per_cluster=POINTS_PER_CLUSTER) ## 
    ,"BETA":.1
    ,"COLS":256
    ,"GEN_SEED":0
    ,"NUM_SIMS":3
}

for CLUSTERS in [2,16,64]:
    for POINTS_PER_CLUSTER in [2,16,64]:
        packed_params["CLUSTERS"] = CLUSTERS
        packed_params["POINTS_PER_CLUSTER"] = POINTS_PER_CLUSTER
        b.run(packed_params=packed_params,**packed_params)
