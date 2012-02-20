# remote definitions, written for re-evaluation
import Cloudless.base
reload(Cloudless.base) # to make it easy to develop locally
import Cloudless.memo
reload(Cloudless.memo) # to make it easy to develop locally
import pylab
from IPython.parallel import *

# configure remote nodes
# TODO: Clean up naming of load balanced vs direct views
Cloudless.base.remote_mode()
Cloudless.base.remote_exec('import numpy.random')
Cloudless.base.remote_exec('import time')
import numpy.random
import time

# definition of the job (re-eval to change code)
def helper(x):
    if numpy.random.uniform() < 0.3:
        raise Exception('we should detect this exception')
    else:
        return x + numpy.random.normal(0, 1.0)

#TODO: make this a decorator; think carefully about dependencies, reloading
Cloudless.base.remote_procedure('helper', helper)

def raw_testjob(x):
    time.sleep(numpy.random.uniform(1))
    return helper(x)

# make memoized job (re-eval if the job code changes, or to reset cache)
testjob = Cloudless.memo.AsyncMemoize("testjob", ["x"], raw_testjob, override=True)

# set constants (re-eval to change the scope of the plot)
XRANGE = 100

# request the computation (re-eval if e.g. the range changes)
for x in range(XRANGE):
    testjob(x)

# get plot data locally (re-eval to get more data)
status = testjob.report_status()
xs = []
ys = []
for (k, v) in testjob.iter():
    xs.append(k[0])
    ys.append(v)

# make a plot (iterate on this block to fix layout/display issues)
pylab.figure()
pylab.scatter(xs, ys)
pylab.xlabel('X')
pylab.ylabel('Y')
pylab.show()

# examine the exceptions for the jobs that failed
testjob.report_status(verbose=True)

