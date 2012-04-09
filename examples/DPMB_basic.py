# block 1
# remote definitions, written for re-evaluation
import Cloudless.base
reload(Cloudless.base) # to make it easy to develop locally
import Cloudless.memo
reload(Cloudless.memo) # to make it easy to develop locally
import matplotlib
matplotlib.use('Agg')
import pylab
from IPython.parallel import *


# block 2
# configure remote nodes
# TODO: Clean up naming of load balanced vs direct views
Cloudless.base.remote_mode()
Cloudless.base.remote_exec('import Cloudless.examples.DPMB as dm')
Cloudless.base.remote_exec('import Cloudless.examples.DPMB_State as ds')
Cloudless.base.remote_exec('import time')
Cloudless.base.remote_exec('import numpy as np')
import Cloudless.examples.DPMB as dm
import Cloudless.examples.DPMB_State as ds
import time
import numpy as np


# block 3
def raw_testjob(gen_seed,inf_seed,rows,cols,alpha,beta,num_iters):
    gen_state_with_data = dm.gen_dataset(gen_seed,rows,cols,alpha,beta)
    gen_sample_output = dm.gen_sample(inf_seed, gen_state_with_data["observables"], num_iters,None,None,num_train=int(np.floor(.9*rows)))
    predictive_prob = test_model(gen_state_with_data,gen_sample_output["state"],num_train=int(np.floor(.9*rows)))
    return gen_sample_output,predictive_prob
# make memoized job (re-eval if the job code changes, or to reset cache)
testjob = Cloudless.memo.AsyncMemoize("testjob", ["gen_seed","inf_seed","rows","cols","alpha","beta","num_iters"], raw_testjob, override=True)


# block 4
# set constants (re-eval to change the scope of the plot)
XRANGE = 10
# request the computation (re-eval if e.g. the range changes)
for x in range(XRANGE):
    testjob(0,x,1000,20,1,1,10)


# block 5
# get plot data locally (re-eval to get more data)
status = testjob.report_status()
xs = []
ys = []
for (k, v) in testjob.iter():
    xs.append(np.array([x["timing"]["zs"]["delta"].total_seconds() for x in v["stats"]]).cumsum())
    ys.append(np.array([x["score"] for x in v["stats"]]))

# block 6
# make a plot (iterate on this block to fix layout/display issues)
pylab.figure()
pylab.plot(np.array(xs).T, np.array(ys).T)
pylab.xlabel('Time Elapsed')
pylab.ylabel('log score')
pylab.show()
pylab.savefig('basic-job-results.png')
##
print sample["stats"][0]["predictive_prob"]["gen_prob"]
print [stats["predictive_prob"]["sampled_prob"] for stats in sample["stats"]]

# block 7
# examine the exceptions for the jobs that failed
testjob.report_status(verbose=True)
