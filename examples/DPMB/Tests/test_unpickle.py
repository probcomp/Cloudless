import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as pylab
import os
import sys
#
import Cloudless
reload(Cloudless)
import Cloudless.memo
reload(Cloudless.memo)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)

base_dir = "/usr/local/Cloudless/examples/DPMB/Diagnostics/"
pkl_file_str = "bigger_mixed_pickled_jobs.pkl"

if len(sys.argv) > 1:
    pkl_file_str = sys.argv[1]

which_measurements=["predictive","ari","num_clusters","score"]

memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], rf.infer, override=False)
rf.unpickle_asyncmemoize(memoized_infer,os.path.join(base_dir,pkl_file_str))

summaries = memoized_infer.memo.values()[0]
for summary in summaries:
    print summary["timing"]

rf.try_plots(memoized_infer,which_measurements=which_measurements)
