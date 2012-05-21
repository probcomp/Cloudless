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
if False:
    Cloudless.base.remote_mode()
    Cloudless.base.remote_exec('import Cloudless.examples.DPMB_remote_functions as rf')
    Cloudless.base.remote_exec('reload(rf)')

pkl_file_str = "test_predictive_pickled_jobs.pkl"

ALL_RUN_SPECS = []
#
run_spec = rf.gen_default_run_spec()
run_spec["num_iters"] = 5
run_spec["time_seatbelt"] = 60
problem = rf.gen_problem(run_spec["dataset_spec"])
print "Created problem"

# now request the inference
memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], rf.infer, override=False)
print "Created memoizer"

memoized_infer(run_spec)

rf.try_plots(memoized_infer,which_measurements=["predictive","ari","num_clusters","score"])
rf.pickle_if_done(memoized_infer,file_str=pkl_file_str)
