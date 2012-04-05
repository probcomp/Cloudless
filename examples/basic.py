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
Cloudless.base.remote_exec('import numpy.random')
Cloudless.base.remote_exec('import time')
import numpy.random
import time

# block 4
# definition of the job (re-eval to change code)
def helper(x):
    if numpy.random.uniform() < 0.3:
        raise Exception('we should detect this exception')
    else:
        return x + numpy.random.normal(0, 1.0)

#TODO: make this a decorator; think carefully about dependencies, reloading
Cloudless.base.remote_procedure('helper', helper)

# block 5
def raw_testjob(x):
    time.sleep(numpy.random.uniform(1))
    return helper(x)

# make memoized job (re-eval if the job code changes, or to reset cache)
testjob = Cloudless.memo.AsyncMemoize("testjob", ["x"], raw_testjob, override=True)

# block 6
# set constants (re-eval to change the scope of the plot)
XRANGE = 100

# request the computation (re-eval if e.g. the range changes)
for x in range(XRANGE):
    testjob(x)

# block 7
# get plot data locally (re-eval to get more data)
status = testjob.report_status()
xs = []
ys = []
for (k, v) in testjob.iter():
    xs.append(k[0])
    ys.append(v)

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

