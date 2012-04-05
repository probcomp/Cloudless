# block 1
# remote definitions, written for re-evaluation
import Cloudless.base
reload(Cloudless.base) # to make it easy to develop locally
import Cloudless.memo
reload(Cloudless.memo) # to make it easy to develop locally
matplotlib.use('Agg')
import matplotlib
import pylab
from IPython.parallel import *


# block 2
# configure remote nodes
# TODO: Clean up naming of load balanced vs direct views
Cloudless.base.remote_mode()
Cloudless.base.remote_exec('import Cloudless.examples.DPMB as dm')
Cloudless.base.remote_exec('import time')
Cloudless.base.remote_exec('import numpy')
import Cloudless.examples.DPMB as dm
import time
import numpy


# block 3
def raw_testjob(gen_seed,inf_seed,rows,cols,alpha,beta,num_iters):
    dataset = dm.gen_dataset(gen_seed,rows,cols,alpha,beta)
    return dm.gen_sample(inf_seed, dataset, num_iters,None,None)
# make memoized job (re-eval if the job code changes, or to reset cache)
testjob = Cloudless.memo.AsyncMemoize("testjob", ["gen_seed","inf_seed","rows","cols","alpha","beta","num_iters"], raw_testjob, override=True)


# block 4
# set constants (re-eval to change the scope of the plot)
XRANGE = 4
# request the computation (re-eval if e.g. the range changes)
for x in range(XRANGE):
    testjob(0,0,10,10*(x+1),1,1,3)


# block 5
# get plot data locally (re-eval to get more data)
status = testjob.report_status()
xs = []
ys = []
for (k, v) in testjob.iter():
    xs.append(k[3]/3.0)
    deltaT = sum([x["timing"]["zs"]["stop"]-x["timing"]["zs"]["start"] for x in v["stats"]])
    ys.append(deltaT.total_seconds())


# block 6
# make a plot (iterate on this block to fix layout/display issues)
pylab.figure()
pylab.scatter(xs, ys)
pylab.xlabel('X')
pylab.ylabel('Y')
pylab.show()
pylab.savefig('basic-job-results.png')


# block 7
# examine the exceptions for the jobs that failed
testjob.report_status(verbose=True)
