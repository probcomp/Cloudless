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
#
Cloudless.base.remote_mode()
Cloudless.base.remote_exec('import Cloudless.examples.DPMB_remote_functions as rf')
Cloudless.base.remote_exec('reload(rf)')

pkl_file_str = "compare_pdpmb_hypers_every_N_pickled_jobs.pkl"

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

# now request the inference
memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], rf.infer, override=False)
print "Created memoizer"

for run_spec in ALL_RUN_SPECS:
    memoized_infer(run_spec)

rf.try_plots(memoized_infer,which_measurements=["predictive","ari","num_clusters","score"])
rf.pickle_if_done(memoized_infer,file_str=pkl_file_str)
