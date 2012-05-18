import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as pylab
#
import Cloudless
reload(Cloudless)
import Cloudless.memo
reload(Cloudless.memo)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
#
if True:
    Cloudless.base.remote_mode()
    Cloudless.base.remote_exec('import Cloudless.examples.DPMB_remote_functions as rf')
    Cloudless.base.remote_exec('reload(rf)')

pkl_file_str = "big_mixed_pickled_jobs.pkl"
which_measurements=["predictive","ari","num_clusters","score"]

def gen_default_run_spec():
    dataset_spec = {}
    dataset_spec["gen_seed"] = 0
    dataset_spec["num_cols"] = 16
    dataset_spec["num_rows"] = 64*64
    dataset_spec["gen_alpha"] = 3.0 #FIXME: could make it MLE alpha later
    dataset_spec["gen_betas"] = np.repeat(0.1, dataset_spec["num_cols"])
    dataset_spec["gen_z"] = ("balanced", 64)
    dataset_spec["N_test"] = 256
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
    run_spec["time_seatbelt"] = 1200
    run_spec["ari_seatbelt"] = .95
    run_spec["verbose_state"] = False
    #
    return run_spec

NUM_RUNS = 3
NUM_DATASETS = 3
NUM_NODES_LIST = [1,3,9]
#
ALL_RUN_SPECS = []
for num_nodes in NUM_NODES_LIST:
    for infer_seed in range(NUM_RUNS):
        for gen_seed in range(NUM_DATASETS):
            for hypers_every_N in NUM_NODES_LIST:
                run_spec = gen_default_run_spec()
                run_spec["num_nodes"] = num_nodes
                run_spec["infer_seed"] = infer_seed
                run_spec["gen_seed"] = gen_seed
                run_spec["hypers_every_N"] = hypers_every_N \
                    if num_nodes != 1 else 1
                ALL_RUN_SPECS.append(run_spec)

# now request the inference
memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], rf.infer, override=False)
print "Created memoizer"

for run_spec in ALL_RUN_SPECS:
    memoized_infer(run_spec)

rf.try_plots(memoized_infer,which_measurements=which_measurements)
rf.pickle_if_done(memoized_infer,file_str=pkl_file_str)
