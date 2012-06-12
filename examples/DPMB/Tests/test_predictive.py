import matplotlib
matplotlib.use('Agg')
import numpy as np
import argparse
import os
#
import Cloudless
reload(Cloudless)
import Cloudless.memo
reload(Cloudless.memo)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)

default_save_dir = os.path.expanduser("~/Run/")
default_pkl_file_str = "test_predictive_pickled_jobs.pkl.gz"
# load up some arguments
parser = argparse.ArgumentParser(description='A test run that plots predictive, among other things')
parser.add_argument('--num_cols',default=256,type=int)
parser.add_argument('--num_rows',default=32*32,type=int)
parser.add_argument('--num_clusters',default=32,type=int)
parser.add_argument('--beta_d',default=3.0,type=float)
parser.add_argument('--balanced',default=-1,type=int)
parser.add_argument('--num_iters',default=1000,type=int)
parser.add_argument('--num_nodes',default=1,type=int)
parser.add_argument('--time_seatbelt',default=60,type=int)
parser.add_argument(
    '--save_dir',
    default=default_save_dir,
    type=str,
    )
parser.add_argument(
    '--pkl_file_str',
    default=default_pkl_file_str,
    type=str,
    )
parser.add_argument('--remote',action='store_true')
#
args = parser.parse_args()
pkl_file_str = os.path.join(args.save_dir,args.pkl_file_str) \
    if args.pkl_file_str == default_pkl_file_str else args.pkl_file_str

if args.remote:
    Cloudless.base.remote_mode()
    Cloudless.base.remote_exec('import Cloudless.examples.DPMB_remote_functions as rf')
    Cloudless.base.remote_exec('reload(rf)')

run_spec = rf.gen_default_run_spec(
    num_clusters = args.num_clusters,
    vectors_per_cluster = args.num_rows/args.num_clusters,
    num_cols = args.num_cols,
    beta_d = args.beta_d
    )
run_spec["dataset_spec"]["num_rows"] = args.num_rows
run_spec["dataset_spec"]["gen_z"] = ("balanced",args.num_clusters)
run_spec["num_iters"] = args.num_iters
run_spec["num_nodes"] = args.num_nodes
run_spec["hypers_every_N"] = args.num_nodes
run_spec["time_seatbelt"] = args.time_seatbelt
run_spec["infer_init_z"] = None if args.balanced == -1 else ("balanced",args.balanced)
run_spec["N_test"] = max(64,run_spec["dataset_spec"]["num_rows"]/16)
problem = rf.gen_problem(run_spec["dataset_spec"])
print "Created problem"

# now request the inference
memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], rf.infer, override=False)
print "Created memoizer"

if os.path.isfile(pkl_file_str):
    print "Using pickled results"
    rf.unpickle_asyncmemoize(memoized_infer,pkl_file_str)
else:
    print "Running inference"
    memoized_infer(run_spec)
    rf.try_plots(
        memoized_infer,
        which_measurements=["predictive","ari","num_clusters","score"],
        save_dir=args.save_dir,
        )
    rf.pickle_if_done(memoized_infer,file_str=pkl_file_str)

gibbs_init_dur = memoized_infer.memo.values()[0][0]["timing"]["init"]
print "gibbs_init_dur: " + str(gibbs_init_dur)

cluster_counts = []
z_diff_times = []
for summary in memoized_infer.memo.values()[0][1:]:
    micro_z_timing = summary["timing"]["micro_z_timing"]
    cluster_counts.extend(micro_z_timing["cluster_counts"][1:])
    z_diff_times.extend(np.diff(micro_z_timing["z_cumulative_time"]))

cluster_counts = np.array(cluster_counts)
z_diff_times = np.array(z_diff_times)

rf.timing_plots(
    cluster_counts,
    z_diff_times,
    args,
    save_dir=args.save_dir,
    )
