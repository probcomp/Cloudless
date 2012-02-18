import Cloudless.base
import Cloudless.memo
import pylab
from IPython.parallel import *

# paste in job definition
@require('numpy.random', 'time')
def raw_testjob(x):
    # FIXME: imports shouldn't be needed
    import numpy.random
    import time
    time.sleep(numpy.random.uniform(1))
    return x + numpy.random.normal(0, 1.0)

# make memoized job (and/or reset memoizer)
testjob = Cloudless.memo.AsyncMemoize(raw_testjob, Cloudless.base.get_view())

# set constants
XRANGE = 50

# request the computation
for x in range(XRANGE):
    testjob(x)

# get plot data locally
xs = []
ys = []
for (k, v) in testjob.iter():
    xs.append(k[0])
    ys.append(v)

# make a plot
pylab.figure()
pylab.scatter(xs, ys)
pylab.xlabel('X')
pylab.ylabel('Y')
pylab.show()
