import numpy as np
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)

# python -i n_seperate_states.py --num_nodes 2 --num_clusters 8 --num_rows 32 --num_iters 15 --plot_states --num_cols 8 --beta_d 0.1

run_spec = rf.gen_default_run_spec(
    num_clusters = 256,
    vectors_per_cluster = 4,
    num_cols = 8,
    beta_d = 0.1,
    )
run_spec["num_iters"] = 15
run_spec["num_nodes"] = 2

ALL_RUN_SPECS = [run_spec]

# cd /usr/local/Cloudless/examples/DPMB/Runspec_Generators
# python ../create_pickled_runspecs.py test_predictive_separate.py test_predictive_separate_runspec.pkl
# python -i ../run_runspecs_from_pkl.py test_predictive_separate_runspec.pkl --infer_name infer_separate --infer_func_str rf.infer_separate --not_remote

