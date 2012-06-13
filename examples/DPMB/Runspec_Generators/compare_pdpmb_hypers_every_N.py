import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)

NUM_ITERS = 50
NUM_RUNS = 3
NUM_NODES_LIST = [5]
HYPERS_EVERY_N_LIST = [1,2,5]
#
ALL_RUN_SPECS = []
for num_nodes in NUM_NODES_LIST:
    for infer_seed in range(NUM_RUNS):
        for hypers_every_N in HYPERS_EVERY_N_LIST:
            run_spec = rf.gen_default_run_spec()
            run_spec["num_iters"] = NUM_ITERS
            run_spec["num_nodes"] = num_nodes
            run_spec["infer_seed"] = infer_seed
            run_spec["time_seatbelt"] = 60
            run_spec["hypers_every_N"] = hypers_every_N
            ALL_RUN_SPECS.append(run_spec)
