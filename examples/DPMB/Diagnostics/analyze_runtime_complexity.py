import matplotlib
matplotlib.use('Agg')
import numpy as np
import numpy.random as nr
import matplotlib.pylab as pylab
from scipy.stats import linregress
#
import Cloudless
reload(Cloudless)
import Cloudless.memo
reload(Cloudless.memo)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
#
Cloudless.base.remote_mode()
Cloudless.base.remote_exec('import Cloudless.examples.DPMB_remote_functions as rf')
Cloudless.base.remote_exec('reload(rf)')

ALL_RUN_SPECS = []
for num_cols in [64,128,256]:
    for num_rows in [32768,65536]:
        run_spec = rf.gen_default_run_spec(num_cols)
        run_spec["dataset_spec"]["num_rows"] = num_rows
        run_spec["dataset_spec"]["gen_z"] = ("balanced",256)
        run_spec["time_seatbelt"] = 1800
        run_spec["infer_init_z"] = 1
        ALL_RUN_SPECS.append(run_spec)

# now request the inference
memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], rf.infer, override=False)
print "Created memoizer"

for run_spec in ALL_RUN_SPECS:
    memoized_infer(run_spec)

# rf.try_plots(memoized_infer,which_measurements=["predictive","ari","num_clusters","score"])
# rf.pickle_if_done(memoized_infer,file_str=args.pkl_file_str)
