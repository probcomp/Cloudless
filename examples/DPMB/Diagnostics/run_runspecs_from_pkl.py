import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as pylab
import os
import argparse
import cPickle
#
import Cloudless
reload(Cloudless)
import Cloudless.memo
reload(Cloudless.memo)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)

# load up some arguments
parser = argparse.ArgumentParser(description='Run ALL_RUN_SPECS from a pkl file')
parser.add_argument('runspec_file_str',type=str)
parser.add_argument('--time_seatbelt',default=-1,type=int)
parser.add_argument('--pkl_file_str',default=os.path.expanduser("~/run_from_pkl_pickled_jobs.pkl.gz"),type=str)
parser.add_argument('--infer_name',default="infer",type=str)
parser.add_argument('--not_remote',action='store_true')
args = parser.parse_args()
#
runspec_file_str = args.runspec_file_str
pkl_file_str = args.pkl_file_str
infer_name = args.infer_name
time_seatbelt = args.time_seatbelt
remote = not args.not_remote

try:
    with open(runspec_file_str,"rb") as fh:
        ALL_RUN_SPECS = cPickle.load(fh)
except Exception, e:
    print "Couldn't load ALL_RUN_SPECS from " + runspec_file_str
    print str(e)

if remote:
    Cloudless.base.remote_mode()
    Cloudless.base.remote_exec('import Cloudless.examples.DPMB_remote_functions as rf')
    Cloudless.base.remote_exec('reload(rf)')

# now request the inference
memoized_infer = Cloudless.memo.AsyncMemoize(
    infer_name
    , ["run_spec"]
    , rf.infer
    , override=False
    )
print "Created memoizer"

for run_spec in ALL_RUN_SPECS:
    if time_seatbelt != -1:
        run_spec["time_seatbelt"] = time_seatbelt
    memoized_infer(run_spec)


def report():
    memoized_infer.report_status()

def plot():
    which_measurements=["predictive","ari","num_clusters","score"]
    rf.try_plots(memoized_infer,which_measurements=which_measurements)

def pickle():
    memoized_infer.advance()
    rf.pickle_asyncmemoize(memoized_infer,file_str=pkl_file_str)

def pickle_if_done():
    rf.pickle_if_done(memoized_infer,file_str=pkl_file_str)
