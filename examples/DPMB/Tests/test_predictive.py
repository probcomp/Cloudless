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
from scipy.stats import linregress
#
import Cloudless
reload(Cloudless)
import Cloudless.memo
reload(Cloudless.memo)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)


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
parser.add_argument('--pkl_file_str',default=os.path.expanduser("~/test_predictive_pickled_jobs.pkl"),type=str)
parser.add_argument('--remote',action='store_true')
#
args = parser.parse_args()


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

if os.path.isfile(args.pkl_file_str):
    print "Using pickled results"
    rf.unpickle_asyncmemoize(memoized_infer,args.pkl_file_str)
else:
    print "Running inference"
    memoized_infer(run_spec)
    rf.try_plots(memoized_infer,which_measurements=["predictive","ari","num_clusters","score"])
    rf.pickle_if_done(memoized_infer,file_str=args.pkl_file_str)

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

box_input = {}
for cluster_count,diff_time in zip(cluster_counts,z_diff_times):
    box_input.setdefault(cluster_count,[]).append(diff_time)

median_times = []
for cluster_count in np.sort(box_input.keys()):
    median_times.append(np.median(box_input[cluster_count]))

slope,intercept,r_value,p_value,stderr = linregress(
    np.sort(box_input.keys())
    ,median_times)
title_str = "slope = " + ("%.3g" % slope) \
    + "; intercept = " + ("%.3g" % intercept) \
    + "; R^2 = " + ("%.5g" % r_value**2)

num_cols = args.num_cols
num_rows = args.num_rows
cutoff = cluster_counts.max()/3
box_every_n = max(1,len(box_input)/10)

pylab.figure()
pylab.plot(cluster_counts,z_diff_times,'x')
pylab.title(title_str)
pylab.xlabel("num_clusters")
pylab.ylabel("single-z scan time (seconds)")
fig_str = "scatter_scan_times_num_cols_"+str(num_cols)+"_num_rows_"+str(num_rows)
pylab.savefig(os.path.expanduser("~/"+fig_str))
#
pylab.figure()
pylab.boxplot(box_input.values()[::box_every_n]
              ,positions=box_input.keys()[::box_every_n]
              ,sym="")
pylab.title(title_str)
pylab.xlabel("num_clusters")
pylab.ylabel("single-z scan time (seconds)")
fig_str = "boxplot_scan_times_num_cols_"+str(num_cols)+"_num_rows_"+str(num_rows)
pylab.savefig(os.path.expanduser("~/"+fig_str))
pylab.close()
#
try:
    pylab.figure()
    pylab.hexbin(cluster_counts[cluster_counts<cutoff],z_diff_times[cluster_counts<cutoff])
    pylab.title(title_str)
    pylab.xlabel("num_clusters")
    pylab.ylabel("single-z scan time (seconds)")
    pylab.colorbar()
    fig_str = "hexbin_scan_times_num_cols_"+str(num_cols)+"_num_rows_"+str(num_rows)+"_lt_"+str(cutoff)
    pylab.savefig(os.path.expanduser("~/"+fig_str))
except Exception, e:
    print e
#
try:
    pylab.figure()
    pylab.hexbin(cluster_counts[cluster_counts>cutoff],z_diff_times[cluster_counts>cutoff])
    pylab.title(title_str)
    pylab.xlabel("num_clusters")
    pylab.ylabel("single-z scan time (seconds)")
    pylab.colorbar()
    fig_str = "hexbin_scan_times_num_cols_"+str(num_cols)+"_num_rows_"+str(num_rows)+"_gt_"+str(cutoff)
    pylab.savefig(os.path.expanduser("~/"+fig_str))
except Exception, e:
    print e

