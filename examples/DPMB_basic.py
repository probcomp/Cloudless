# block 1
# must be run before matplotlib,pylab imports
matplotlib.use('Agg')

# block 2
# remote definitions, written for re-evaluation
import Cloudless.base
reload(Cloudless.base) # to make it easy to develop locally
import Cloudless.memo
reload(Cloudless.memo) # to make it easy to develop locally
import matplotlib
import pylab
from IPython.parallel import *

# block 3
# configure remote nodes
# TODO: Clean up naming of load balanced vs direct views
Cloudless.base.remote_mode()
Cloudless.base.remote_exec('import Cloudless.examples.DPMB as dm')
Cloudless.base.remote_exec('import time')
Cloudless.base.remote_exec('import numpy')
import Cloudless.examples.DPMB as dm
import time
import numpy

# block 4
# definition of the job (re-eval to change code)
def sampler(gen_seed,inf_seed,rows,cols,alpha,beta,num_iters):
    dataset = dm.gen_dataset(gen_seed,rows,cols,alpha,beta)
    sample = dm.gen_sample(inf_seed, dataset, num_iters,None,None)
    return sample

#TODO: make this a decorator; think carefully about dependencies, reloading
Cloudless.base.remote_procedure('sampler', sampler)

# block 5
def raw_testjob(gen_seed,inf_seed,rows,cols,alpha,beta,num_iters):
    time.sleep(numpy.random.uniform(1))
    return sampler(gen_seed,inf_seed,rows,cols,alpha,beta,num_iters)

# make memoized job (re-eval if the job code changes, or to reset cache)
testjob = Cloudless.memo.AsyncMemoize("testjob", ["gen_seed","inf_seed","rows","cols","alpha","beta","num_iters"], raw_testjob, override=True)

# block 6
# set constants (re-eval to change the scope of the plot)
XRANGE = 4

# request the computation (re-eval if e.g. the range changes)
for x in range(XRANGE):
    testjob(0,0,10,10*(x+1),1,1,3)

# block 7
# get plot data locally (re-eval to get more data)
status = testjob.report_status()
xs = []
ys = []
for (k, v) in testjob.iter():
    xs.append(k[0])
    deltaT = sum([x["timing"]["zs"]["stop"]-x["timing"]["zs"]["start"] for x in v["stats"]])
    ys.append(deltaT)

# block 8
# make a plot (iterate on this block to fix layout/display issues)
pylab.figure()
pylab.scatter(xs, ys)
pylab.xlabel('X')
pylab.ylabel('Y')
pylab.show()
pylab.savefig('basic-job-results.png')

# block 9
# examine the exceptions for the jobs that failed
testjob.report_status(verbose=True)
