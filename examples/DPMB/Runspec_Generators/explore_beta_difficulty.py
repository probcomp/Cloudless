import numpy as np
import itertools

def gen_default_run_spec(num_clusters, vectors_per_cluster
                         , num_cols=256, beta_d=1.0):
    dataset_spec = {}
    dataset_spec["gen_seed"] = 0
    dataset_spec["num_cols"] = num_cols
    dataset_spec["num_rows"] = num_clusters*vectors_per_cluster
    dataset_spec["gen_alpha"] = 3.0 #FIXME: could make it MLE alpha later
    dataset_spec["gen_betas"] = np.repeat(beta_d, dataset_spec["num_cols"])
    dataset_spec["gen_z"] = ("balanced", num_clusters)
    dataset_spec["N_test"] = num_clusters*vectors_per_cluster/4
    #
    run_spec = {}
    run_spec["dataset_spec"] = dataset_spec
    run_spec["num_iters"] = 10
    run_spec["num_nodes"] = 1
    run_spec["infer_seed"] = 0
    run_spec["infer_init_alpha"] = 3.0
    run_spec["infer_init_betas"] = np.repeat(beta_d, dataset_spec["num_cols"])
    run_spec["infer_do_alpha_inference"] = True
    run_spec["infer_do_betas_inference"] = True
    run_spec["infer_init_z"] = None
    run_spec["hypers_every_N"] = 1
    run_spec["time_seatbelt"] = 60
    run_spec["ari_seatbelt"] = None
    run_spec["verbose_state"] = False
    #
    return run_spec

NUM_DATASETS = 2
NUM_RUNS = 2
#
GEN_SEED_LIST = range(NUM_DATASETS)
INFER_SEED_LIST = range(NUM_RUNS)
NUM_NODES_LIST = [1] # [1,4,16]
HYPERS_EVERY_N_LIST = [1] # [4,16]
NUM_CLUSTERS_LIST = [4,16,64,256]
VECTORS_PER_CLUSTER_LIST = [4,16,64,256]
#
param_iter = itertools.product(
    GEN_SEED_LIST,
    INFER_SEED_LIST,
    NUM_NODES_LIST,
    HYPERS_EVERY_N_LIST,
    NUM_CLUSTERS_LIST,
    VECTORS_PER_CLUSTER_LIST,
)

ALL_RUN_SPECS = []
for (gen_seed,
     infer_seed,
     num_nodes,
     hypers_every_N,
     num_clusters,
     vectors_per_cluster) \
     in param_iter:
        run_spec = gen_default_run_spec(num_clusters,vectors_per_cluster)
        run_spec["dataset_spec"]["gen_seed"] = gen_seed
        run_spec["infer_seed"] = infer_seed
        run_spec["num_nodes"] = num_nodes
        run_spec["hypers_every_N"] = hypers_every_N \
            if num_nodes != 1 else 1
        ALL_RUN_SPECS.append(run_spec)

# for num_nodes in NUM_NODES_LIST:
#     for infer_seed in range(NUM_RUNS):
#         for gen_seed in range(NUM_DATASETS):
#             for hypers_every_N in HYPERS_EVERY_N_LIST:
#                 run_spec = gen_default_run_spec()
#                 run_spec["num_nodes"] = num_nodes
#                 run_spec["infer_seed"] = infer_seed
#                 run_spec["dataset_spec"]["gen_seed"] = gen_seed
#                 run_spec["hypers_every_N"] = hypers_every_N \
#                     if num_nodes != 1 else 1
#                 ALL_RUN_SPECS.append(run_spec)
