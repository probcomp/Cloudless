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
Cloudless.base.remote_exec('reload(dm)')
Cloudless.base.remote_exec('import Cloudless.examples.DPMB_State as ds')
Cloudless.base.remote_exec('reload(ds)')
Cloudless.base.remote_exec('import time')
Cloudless.base.remote_exec('import numpy as np')
import Cloudless.examples.DPMB as dm
reload(dm)
import Cloudless.examples.DPMB_State as ds
reload(ds)
import time
import numpy as np


# block 3
def raw_testjob(gen_seed,inf_seed,clusters,points_per_cluster,num_iters,hold_out_ratio,cols,alpha,beta,infer_hypers):
    paramDict = {"inferAlpha":infer_hypers,"inferBetas":infer_hypers}
    raise Exception("must figure out what to do with num_train")
    gen_state_with_data = dm.gen_dataset(gen_seed,None,cols,alpha,beta,np.repeat(points_per_cluster,clusters))
    gen_sample_output = dm.gen_sample(inf_seed, gen_state_with_data["observables"], num_iters,None,None,paramDict=paramDict)
    predictive_prob = dm.test_model(gen_state_with_data["observables"],gen_sample_output["state"])
    return gen_sample_output,predictive_prob
# make memoized job (re-eval if the job code changes, or to reset cache)
testjob = Cloudless.memo.AsyncMemoize("testjob", ["gen_seed","inf_seed","clusters","points_per_cluster","num_iters","hold_out_ratio","cols","alpha","beta","infer_hypers"], raw_testjob, override=True)


# block 4
# set constants (re-eval to change the scope of the plot)
CLUSTERS = 10
POINTS_PER_CLUSTER = 10
NUM_ITERS = 10
GEN_SEED = 2
NUM_SIMS = 10
##BELOW ARE FAIRLY STATIC VALUES
COLS = 256
BETA = .1
INFER_HYPERS = False
ALPHA = 10
# request the computation (re-eval if e.g. the range changes)
for inf_seed in range(NUM_SIMS):
    testjob(GEN_SEED,inf_seed,CLUSTERS,POINTS_PER_CLUSTER,NUM_ITERS,COLS,ALPHA,BETA,INFER_HYPERS)


# block 5
# get plot data locally (re-eval to get more data)
status = testjob.report_status()
time_delta = []
log_score = []
predictive_prob = []
DATASET = dm.gen_dataset(GEN_SEED,None,COLS,ALPHA,BETA,np.repeat(POINTS_PER_CLUSTER,CLUSTERS))
true_prob = test_model({"observables":DATASET["test_data"]},DATASET["gen_state"])
##
for (k, v) in testjob.iter():
    z_delta = np.array([x["timing"]["zs"]["delta"].total_seconds() for x in v[0]["stats"]]).cumsum()
    if "alpha" in v[0]["stats"][0]["timing"]:
        alpha_delta = np.array([x["timing"]["alpha"]["delta"].total_seconds() for x in v[0]["stats"]]).cumsum()
    else:
        alpha_delta = np.zeros(np.shape(z_delta))
    if "beta" in v[0]["stats"][0]["timing"]:
        beta_delta = np.array([x["timing"]["beta"]["delta"].total_seconds() for x in v[0]["stats"]]).cumsum()
    else:
        beta_delta = np.zeros(np.shape(z_delta))
    time_delta.append(z_delta+alpha_delta+beta_delta)
    log_score.append(np.array([x["score"] for x in v[0]["stats"]]))
    predictive_prob.append(np.array([x["predictive_prob"] for x in v[0]["stats"]]))    


# block 6
# make a plot (iterate on this block to fix layout/display issues)
fh = pylab.figure()
pylab.plot(np.array(time_delta).T, np.array(predictive_prob).T)
pylab.hlines(true_prob,*fh.get_axes()[0].get_xlim(),colors='r',linewidth=3)
pylab.title("RxC: " + str(ROWS) + "x" + str(COLS) + "; NUM_ITERS: " + str(NUM_ITERS) + "; ALPHA: " + str(ALPHA))
pylab.xlabel('Time Elapsed (seconds)')
pylab.ylabel('predictive_prob')
pylab.show()
pylab.savefig('predictive_prob_by_time.png')
##
pylab.figure()
pylab.plot(np.array(time_delta).T, np.array(log_score).T)
pylab.title("RxC: " + str(ROWS) + "x" + str(COLS) + "; NUM_ITERS: " + str(NUM_ITERS) + "; ALPHA: " + str(ALPHA))
pylab.xlabel('Time Elapsed (seconds)')
pylab.ylabel('log_score')
pylab.show()
pylab.savefig('log_score_by_time.png')
##
##
fh = pylab.figure()
pylab.plot(repeat(range(NUM_ITERS),NUM_SIMS).reshape(NUM_ITERS,NUM_SIMS), np.array(predictive_prob).T)
pylab.hlines(true_prob,*fh.get_axes()[0].get_xlim(),colors='r',linewidth=3)
pylab.title("RxC: " + str(ROWS) + "x" + str(COLS) + "; NUM_ITERS: " + str(NUM_ITERS) + "; ALPHA: " + str(ALPHA))
pylab.xlabel('Iteration number')
pylab.ylabel('predictive_prob')
pylab.show()
pylab.savefig('predictive_prob_by_iter.png')
##
pylab.figure()
pylab.plot(repeat(range(NUM_ITERS),NUM_SIMS).reshape(NUM_ITERS,NUM_SIMS), np.array(log_score).T)
pylab.title("RxC: " + str(ROWS) + "x" + str(COLS) + "; NUM_ITERS: " + str(NUM_ITERS) + "; ALPHA: " + str(ALPHA))
pylab.xlabel('Iteration number')
pylab.ylabel('log_score')
pylab.show()
pylab.savefig('log_score_by_iter.png')


# block 7
# examine the exceptions for the jobs that failed
testjob.report_status(verbose=True)
cluster_dist = np.sort(dm.gen_dataset(GEN_SEED,None,COLS,ALPHA,BETA,np.repeat(POINTS_PER_CLUSTER,CLUSTERS))["gen_state"]["phis"])
print str(sum(cluster_dist.cumsum()>.10)) + " clusters comprise 90% of data; " + str(sum(cluster_dist.cumsum()>.30)) + " clusters comprise 70% of data"
##testjob.terminate_pending() ## uncomment to kill non-running jobs
