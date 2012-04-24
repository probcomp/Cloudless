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
with open("pickled_jobs.pkl","rb") as fh:
    pickled_memo = cPickle.load(fh)

[key in memoized_infer.memo for key in memoized_infer.memo.keys()]

ALL_RUN_SPECS = [eval(run_spec)[0] for run_spec in memoized_infer.memo.keys()]
new_memo = dict(zip([str((run_spec,)) for run_spec in ALL_RUN_SPECS],memoized_infer.memo.values()))


memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], hf.infer, override=True) #FIXME once we've debugged, we can eliminate this override
##
memoized_infer.memo = new_memo

run_spec = ALL_RUN_SPECS[0]
target_problem = run_spec["problem"]
hf.plot_measurement(memoized_infer, "num_clusters", target_problem)

