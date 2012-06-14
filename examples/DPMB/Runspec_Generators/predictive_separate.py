import numpy as np
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)


NUM_RUNS = 2
NUM_DATASETS = 2
NUM_NODES_LIST = [4]
HYPERS_EVERY_N_LIST = [4]
#
ALL_RUN_SPECS = []
for infer_seed in range(NUM_RUNS):
    for gen_seed in range(NUM_DATASETS):
        run_spec = run_spec = rf.gen_default_run_spec(
            num_clusters = 256,
            vectors_per_cluster = 64,
            num_cols = 64,
            beta_d = 3.0,
            )
        run_spec["num_iters"] = 1000
        run_spec["num_nodes"] = 4
        run_spec["time_seatbelt"] = 1200
        run_spec["hypers_every_N"] = 4
        run_spec["infer_seed"] = infer_seed
        run_spec["dataset_spec"]["gen_seed"] = gen_seed
        ALL_RUN_SPECS.append(run_spec)


# cd /usr/local/lib/python2.7/dist-packages/Cloudless/examples/DPMB/Runspec_Generators/
# mkdir ~/separate
# python ../create_pickled_runspecs.py predictive_separate.py ~/predictive_separate_runspec.pkl
# python -i ../run_runspecs_from_pkl.py ~/predictive_separate_runspec.pkl --save_dir ~/separate/ --pkl_file_str ~/separate/saved_runs.pkl --infer_name infer_separate --infer_func_str rf.infer_separate

# cd /usr/local/lib/python2.7/dist-packages/Cloudless/examples/DPMB/Runspec_Generators/
# mkdir ~/not_separate
# python ../create_pickled_runspecs.py predictive_separate.py ~/predictive_separate_runspec.pkl
# python -i ../run_runspecs_from_pkl.py ~/predictive_separate_runspec.pkl --save_dir ~/not_separate/ --pkl_file_str ~/not_separate/saved_runs.pkl --infer_name infer_not_separate --infer_func_str rf.infer
