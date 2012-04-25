import cPickle,matplotlib,sys
matplotlib.use('Agg')
import DPMB_plotutils as dp
reload(dp)
import DPMB_State as ds
reload(ds)
import DPMB_helper_functions as hf
reload(hf)
import numpy as np
import matplotlib.pylab as pylab
##
if sys.platform == "win32":
    sys.path.append("c:/")
    
import Cloudless
reload(Cloudless)
import Cloudless.memo
reload(Cloudless.memo)


# block 2
# configure remote nodes
# TODO: Clean up naming of load balanced vs direct views
if sys.platform != "win32":
    Cloudless.base.remote_mode()
    Cloudless.base.remote_exec('import Cloudless.examples.DPMB_plotutils as dp')
    Cloudless.base.remote_exec('reload(dp)')
    Cloudless.base.remote_exec('import Cloudless.examples.DPMB_State as ds')
    Cloudless.base.remote_exec('reload(ds)')
    Cloudless.base.remote_exec('import Cloudless.examples.DPMB_helper_functions as hf')
    Cloudless.base.remote_exec('reload(hf)')
    Cloudless.base.remote_exec('import numpy as np')
    Cloudless.base.remote_exec('import matplotlib.pylab as pylab')
    
import Cloudless.examples.DPMB_plotutils as dp
reload(dp)
import Cloudless.examples.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB_helper_functions as hf
reload(hf)
import numpy as np
import matplotlib.pylab as pylab

config_file_str = "~/config_test_states.py"
if os.path.isfile(config_file_str):
    execfile(config_file_str) ##generate ALL_RUN_SPECS
else:
    print "must have configuration file at ~/config_test_states.py"
    sys.exit()

print "Running inference on " + str(len(ALL_RUN_SPECS)) + " problems..."

# now request the inference
memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], hf.infer, override=False)

print "Created memoizer"

for run_spec in ALL_RUN_SPECS:
    memoized_infer(run_spec)

run_spec_filter = None ## lambda x: x["infer_init_z"] is None ## 

hf.try_plots(memoized_infer)
hf.pickle_if_done(memoized_infer,file_str="pickled_jobs.pkl")
