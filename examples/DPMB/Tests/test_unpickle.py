import matplotlib
matplotlib.use('Agg')
import os
import sys
import argparse
#
import numpy as np
import matplotlib.pylab as pylab
#
import Cloudless
reload(Cloudless)
import Cloudless.memo
reload(Cloudless.memo)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)

parser = argparse.ArgumentParser(
    description='Unpickle an async_memo.memo and populate one called memoized_infer')
parser.add_argument(
    '--pkl_file_str',
    default=os.path.expanduser("~/Run/saved_runs.pkl.gz"),
    type=str,
    )
parser.add_argument(
    '--save_dir',
    default=os.path.expanduser("~/Run/"),
    type=str,
    )
args,unknown_args = parser.parse_known_args()

memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], rf.infer, override=False)
rf.unpickle_asyncmemoize(memoized_infer,args.pkl_file_str)

def report():
    memoized_infer.report_status()

def plot(which_measurements=None):
    if which_measurements is None:
        which_measurements = ["predictive","num_clusters","score"]
    rf.try_plots(
        memoized_infer,
        which_measurements=which_measurements,
        save_dir=save_dir,
        )

def pickle():
    memoized_infer.advance()
    rf.pickle_asyncmemoize(memoized_infer,file_str=pkl_file_str)

def pickle_if_done():
    rf.pickle_if_done(memoized_infer,file_str=pkl_file_str)

