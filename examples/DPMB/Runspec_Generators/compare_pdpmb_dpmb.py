import numpy as np

NUM_CLUSTERS = 4
VECTORS_PER_CLUSTER = 32
beta_d = 1.0
def gen_default_run_spec():
    dataset_spec = {}
    dataset_spec["gen_seed"] = 0
    dataset_spec["num_cols"] = 16
    dataset_spec["num_rows"] = NUM_CLUSTERS*VECTORS_PER_CLUSTER
    dataset_spec["gen_alpha"] = 3.0 #FIXME: could make it MLE alpha later
    dataset_spec["gen_betas"] = np.repeat(beta_d, dataset_spec["num_cols"])
    dataset_spec["gen_z"] = ("balanced", NUM_CLUSTERS)
    dataset_spec["N_test"] = NUM_CLUSTERS*VECTORS_PER_CLUSTER/4
    #
    run_spec = {}
    run_spec["dataset_spec"] = dataset_spec
    run_spec["num_iters"] = 100
    run_spec["num_nodes"] = 1
    run_spec["infer_seed"] = 0
    run_spec["infer_init_alpha"] = 3.0
    run_spec["infer_init_betas"] = np.repeat(beta_d, dataset_spec["num_cols"])
    run_spec["infer_do_alpha_inference"] = True
    run_spec["infer_do_betas_inference"] = True
    run_spec["infer_init_z"] = None
    run_spec["hypers_every_N"] = 1
    run_spec["time_seatbelt"] = 30
    run_spec["ari_seatbelt"] = None
    run_spec["verbose_state"] = True # FIXME : change back when done debugging
    #
    return run_spec

NUM_RUNS = 1
NUM_DATASETS = 3
NUM_NODES_LIST = [1] # [1,4,16]
HYPERS_EVERY_N_LIST = [4,16]
#
ALL_RUN_SPECS = []
for num_nodes in NUM_NODES_LIST:
    for infer_seed in range(NUM_RUNS):
        for gen_seed in range(NUM_DATASETS):
            for hypers_every_N in HYPERS_EVERY_N_LIST:
                run_spec = gen_default_run_spec()
                run_spec["num_nodes"] = num_nodes
                run_spec["infer_seed"] = infer_seed
                run_spec["dataset_spec"]["gen_seed"] = gen_seed
                run_spec["hypers_every_N"] = hypers_every_N \
                    if num_nodes != 1 else 1
                ALL_RUN_SPECS.append(run_spec)
