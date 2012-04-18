#!python


import Cloudless.examples.DPMB_basic as b
reload(b)
##
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
                if CLUSTERS == 64 and POINTS_PER_CLUSTER == 64:
                    continue
                packed_params = b.create_dict()
                packed_params["CLUSTERS"] = CLUSTERS
                packed_params["POINTS_PER_CLUSTER"] = POINTS_PER_CLUSTER
                packed_params["INIT_METHOD"]["alpha"] = 1;packed_params["alpha"] = 1
                job_list.append((b.queue_jobs(packed_params=packed_params,**packed_params),packed_params))

job_list,done_list = b.filter_plottable(job_list,done_list,y_vars=["ari","num_clusters"])
##to rerun, must 'del job_list'
