import matplotlib
matplotlib.use('Agg')
import numpy as np
import numpy.random as nr
import matplotlib.pylab as pylab
import datetime
import re
import gc
#
import Cloudless
reload(Cloudless)
import Cloudless.memo
reload(Cloudless.memo)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)

NUM_CLUSTERS = 32
dataset_spec = {}
dataset_spec["gen_seed"] = 0
dataset_spec["num_cols"] = 16
dataset_spec["num_rows"] = 32*32
dataset_spec["gen_alpha"] = 3.0 #FIXME: could make it MLE alpha later
dataset_spec["gen_betas"] = np.repeat(0.1, dataset_spec["num_cols"])
dataset_spec["gen_z"] = ("balanced", NUM_CLUSTERS)
dataset_spec["N_test"] = 128
#
problem = rf.gen_problem(dataset_spec)

ALL_RUN_SPECS = []
num_iters = 10
num_nodes = 2
count = 0
for infer_seed in range(1):
    for infer_init_alpha in [1.0]: #note: we're never trying sample-alpha-from-prior-for-init
        for infer_init_betas in [np.repeat(0.1, dataset_spec["num_cols"])]:
            for infer_do_alpha_inference in [True, False]:
                for infer_do_betas_inference in [True, False]:
                    for infer_init_z in [None, 1, "N"]:
                        run_spec = {}
                        run_spec["num_iters"] = num_iters
                        run_spec["infer_seed"] = infer_seed
                        run_spec["infer_init_alpha"] = infer_init_alpha
                        run_spec["infer_init_betas"] = infer_init_betas.copy()
                        run_spec["infer_do_alpha_inference"] = infer_do_alpha_inference
                        run_spec["infer_do_betas_inference"] = infer_do_betas_inference
                        run_spec["infer_init_z"] = infer_init_z
                        run_spec["dataset_spec"] = dataset_spec
                        ##
                        run_spec["num_nodes"] = num_nodes
                        run_spec["time_seatbelt"] = 1200
                        run_spec["ari_seatbelt"] = .95
                        ALL_RUN_SPECS.append(run_spec) ## this seems to make the comparison fail copy.deepcopy(run_spec))

print "Generated " + str(len(ALL_RUN_SPECS)) + " run specs!"
# until success:
#ALL_RUN_SPECS = ALL_RUN_SPECS[:1]

print "Running inference on " + str(len(ALL_RUN_SPECS)) + " problems..."

# now request the inference
memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], rf.infer, override=True) #FIXME once we've debugged, we can eliminate this override

print "Created memoizer"

# for run_spec in ALL_RUN_SPECS:
#     memoized_infer(run_spec)
run_spec = ALL_RUN_SPECS[0]
memoized_infer(run_spec)

run_spec_filter = None ## lambda x: x["infer_init_z"] is None ## 

rf.try_plots(memoized_infer,which_measurements=["predictive"])
rf.pickle_if_done(memoized_infer,file_str="pickled_jobs.pkl")
