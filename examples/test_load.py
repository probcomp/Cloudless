import DPMB_plotutils as dp
reload(dp)
import DPMB_helper_functions as hf
reload(hf)
import DPMB_State as ds
reload(ds)
import numpy as np
import matplotlib.pylab as pylab

import sys
sys.path.append("c://")
import Cloudless
reload(Cloudless)
import Cloudless.memo
reload(Cloudless.memo)

import cPickle
with open("pickled_jobs.pkl") as fh:
    pickled_jobs = cPickle.load(fh)


memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], hf.infer, override=True) #FIXME once we've debugged, we can eliminate this override
##
memoized_infer.memo = pickled_jobs["memo"]
ALL_RUN_SPECS = pickled_jobs["ALL_RUN_SPECS"]

run_spec = ALL_RUN_SPECS[0]
target_problem = run_spec["problem"]
hf.plot_measurement(memoized_infer, "num_clusters", target_problem)
