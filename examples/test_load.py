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

if "memoized_infer" not in locals():
    memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], hf.infer, override=True) #FIXME once we've debugged, we can eliminate this override
    ALL_RUN_SPECS = hf.unpickle_asyncmemoize(memoized_infer,"pickled_jobs.pkl")

run_spec = ALL_RUN_SPECS[0]
target_problem = run_spec["problem"]
hf.plot_measurement(memoized_infer, ("ari",target_problem["zs"]), target_problem
                    ,save_str="1.png")
hf.plot_measurement(memoized_infer, ("ari",target_problem["zs"]), target_problem,by_time=False
                    ,save_str="2.png")                    
