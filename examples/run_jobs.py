#!python
import numpy as np

import Cloudless.examples.DPMB_basic as b
reload(b)
##
def create_dict(ALPHA=1,CLUSTERS=10,POINTS_PER_CLUSTER=10,NUM_ITERS=20):
    low_val=.01
    high_val=1E4
    cols = 256
    beta = .1
    betas = np.repeat(beta,cols)
    packed_params = {
        "path":"Plots"
        ,"CLUSTERS":CLUSTERS
        ,"POINTS_PER_CLUSTER":POINTS_PER_CLUSTER
        ,"NUM_ITERS":NUM_ITERS
        ,"INFER_ALPHA":[None,{"method":"DISCRETE_GIBBS","low_val":low_val,"high_val":high_val,"n_grid":100}][1]
        ,"INFER_BETA":[None,{"method":"DISCRETE_GIBBS","low_val":low_val,"high_val":high_val,"n_grid":100}][0]
        ,"INIT_METHOD":[{"method":"all_together","alpha":ALPHA,"betas":betas}
                        ,{"method":"all_separate","alpha":ALPHA,"betas":betas}
                        ,{"method":"sample_prior","alpha":ALPHA,"betas":betas}
                        ][0]
        ##BELOW ARE FAIRLY STATIC VALUES
        ,"ALPHA":ALPHA ## hf.mle_alpha(clusters=CLUSTERS,points_per_cluster=POINTS_PER_CLUSTER) ## 
        ,"BETA":beta
        ,"COLS":cols
        ,"GEN_SEED":0
        ,"NUM_SIMS":5
    }
    return packed_params
##
low_val = .01
high_val = 1E4
alpha = 1
CLUSTERS_list = [2,16,64]
POINTS_PER_CLUSTER_list = [2,16,64]
INFER_ALPHA_list = [None,{"method":"DISCRETE_GIBBS","low_val":low_val,"high_val":high_val,"n_grid":100}]
INFER_BETA_list = [None,{"method":"DISCRETE_GIBBS","low_val":low_val,"high_val":high_val,"n_grid":100}]
INIT_METHOD_list = [{"method":"all_together","alpha":alpha,"betas":np.repeat(.1,256)}
                    ,{"method":"all_separate","alpha":alpha,"betas":np.repeat(.1,256)}
                    ,{"method":"sample_prior","alpha":alpha,"betas":np.repeat(.1,256)}]


if "job_list" not in locals():
    job_list = []
    done_list = []
    for CLUSTERS in CLUSTERS_list:
        for POINTS_PER_CLUSTER in POINTS_PER_CLUSTER_list:
            for INIT_METHOD in INIT_METHOD_list:
                for INFER_ALPHA in INFER_ALPHA_list:
                    packed_params = create_dict(CLUSTERS=CLUSTERS,POINTS_PER_CLUSTER=POINTS_PER_CLUSTER,ALPHA=alpha)
                    packed_params["INIT_METHOD"] = INIT_METHOD
                    packed_params["INFER_ALPHA"] = INFER_ALPHA
                    job_list.append((b.queue_jobs(packed_params=packed_params,**packed_params),packed_params))

job_list,done_list = b.filter_plottable(job_list,done_list,y_vars=["ari","num_clusters"])
##to rerun, must 'del job_list'
