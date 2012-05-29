import matplotlib
matplotlib.use('Agg')
import numpy as np
import numpy.random as nr
import matplotlib.pylab as pylab
import datetime
import re
import gc
import argparse
import os
#
import Cloudless
reload(Cloudless)
import Cloudless.memo
reload(Cloudless.memo)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)


# load up some arguments
parser = argparse.ArgumentParser(description='A test run that plots predictive, among other things')
parser.add_argument('--num_cols',default=16,type=int)
parser.add_argument('--num_rows',default=64*64,type=int)
parser.add_argument('--num_clusters',default=64,type=int)
parser.add_argument('--num_iters',default=1000,type=int)
parser.add_argument('--num_nodes',default=5,type=int)
parser.add_argument('--time_seatbelt',default=60,type=int)
parser.add_argument('--pkl_file_str',default=os.path.expanduser("~/test_predictive_pickled_jobs.pkl"),type=str)
parser.add_argument('--remote',action='store_true')
#
args = parser.parse_args()


if args.remote:
    Cloudless.base.remote_mode()
    Cloudless.base.remote_exec('import Cloudless.examples.DPMB_remote_functions as rf')
    Cloudless.base.remote_exec('reload(rf)')


run_spec = rf.gen_default_run_spec(args.num_cols)
run_spec["dataset_spec"]["num_rows"] = args.num_rows
run_spec["dataset_spec"]["gen_z"] = ("balanced",args.num_clusters)
run_spec["num_iters"] = args.num_iters
run_spec["num_nodes"] = args.num_nodes
run_spec["hypers_every_N"] = args.num_nodes
run_spec["time_seatbelt"] = args.time_seatbelt
problem = rf.gen_problem(run_spec["dataset_spec"])
print "Created problem"

# now request the inference
memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], rf.infer, override=False)
print "Created memoizer"

memoized_infer(run_spec)

rf.try_plots(memoized_infer,which_measurements=["predictive","ari","num_clusters","score"])
rf.pickle_if_done(memoized_infer,file_str=args.pkl_file_str)
