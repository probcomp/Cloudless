import numpy as np
import os
import argparse
import cPickle

parser = argparse.ArgumentParser(description='Generate ALL_RUN_SPECS and store in a pkl file')
parser.add_argument('runspec_file_str',type=str)
args = parser.parse_args()
runspec_file_str = args.runspec_file_str

NUM_CLUSTERS = 4
VECTORS_PER_CLUSTER = 64
def gen_default_run_spec():
    dataset_spec = {}
    dataset_spec["gen_seed"] = 0
    dataset_spec["num_cols"] = 16
    dataset_spec["num_rows"] = NUM_CLUSTERS*VECTORS_PER_CLUSTER
    dataset_spec["gen_alpha"] = 3.0 #FIXME: could make it MLE alpha later
    dataset_spec["gen_betas"] = np.repeat(0.1, dataset_spec["num_cols"])
    dataset_spec["gen_z"] = ("balanced", NUM_CLUSTERS)
    dataset_spec["N_test"] = NUM_CLUSTERS*VECTORS_PER_CLUSTER/8
    ##
    run_spec = {}
    run_spec["dataset_spec"] = dataset_spec
    run_spec["num_iters"] = 100
    run_spec["num_nodes"] = 1
    run_spec["infer_seed"] = 0
    run_spec["infer_init_alpha"] = None
    run_spec["infer_init_betas"] = np.repeat(0.1, dataset_spec["num_cols"])
    run_spec["infer_do_alpha_inference"] = True
    run_spec["infer_do_betas_inference"] = True
    run_spec["infer_init_z"] = None
    run_spec["hypers_every_N"] = 1
    run_spec["time_seatbelt"] = 300
    run_spec["ari_seatbelt"] = None
    run_spec["verbose_state"] = False
    #
    return run_spec

NUM_RUNS = 3
NUM_DATASETS = 3
NUM_NODES_LIST = [1,4,16]
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
                run_spec["gen_seed"] = gen_seed
                run_spec["hypers_every_N"] = hypers_every_N \
                    if num_nodes != 1 else 1
                ALL_RUN_SPECS.append(run_spec)

try:
    with open(runspec_file_str,"wb") as fh:
        cPickle.dump(ALL_RUN_SPECS,fh)
except Exception, e:
    print "Couldn't write ALL_RUN_SPECS to " + runspec_file_str
    print str(e)
