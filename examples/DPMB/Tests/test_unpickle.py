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

pkl_file_str = "/usr/local/Cloudless/examples/DPMB/Diagnostics/psuedo_parallel_pickled_jobs.pkl"

memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], rf.infer, override=False)
rf.unpickle_asyncmemoize(memoized_infer,pkl_file_str)

summaries = memoized_infer.memo.values()[0]
for summary in summaries:
    print summary["timing"]

rf.try_plots(memoized_infer,which_measurements=["predictive"])
