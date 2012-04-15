# block 1
# remote definitions, written for re-evaluation
import Cloudless.base
reload(Cloudless.base) # to make it easy to develop locally
import Cloudless.memo
reload(Cloudless.memo) # to make it easy to develop locally
import matplotlib
##matplotlib.use('Agg')
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
# make a plot (iterate on this block to fix layout/display issues)
def do_plots(x_vars=None,y_vars=None):
    x_vars = x_vars if x_vars is not None else ["TIME","ITER"]
    y_vars = y_vars if y_vars is not None else ["ari","predictive_prob","log_score"]
    CLUSTER_STR = str(POINTS_PER_CLUSTER) + "*" + str(CLUSTERS)
    TITLE_STR = "{Clusters*Points}xC: " + CLUSTER_STR + "x" + str(COLS) + "; NUM_ITERS: " + str(NUM_ITERS) + "; ALPHA: " + str(ALPHA)
    TIME_LABEL = 'Time Elapsed (seconds)'
    ITER_LABEL = 'Iteration number'
    TIME_ARR = np.array(time_delta).T
    ITER_ARR = np.repeat(range(NUM_ITERS),NUM_SIMS).reshape(NUM_ITERS,NUM_SIMS)
    ##
    for y_var_str in y_vars:
        for x_var_str in x_vars:
            fh = pylab.figure()
            pylab.plot(locals()[x_var_str+"_ARR"], np.array(globals()[y_var_str]).T)
            if y_var_str=="predictive_prob":
                pylab.hlines(true_prob,*fh.get_axes()[0].get_xlim(),colors='r',linewidth=3)
            pylab.title(TITLE_STR)
            pylab.xlabel(locals()[x_var_str+"_LABEL"])
            pylab.ylabel(y_var_str)
            pylab.show()
            pylab.savefig(y_var_str+'_by_' + x_var_str.lower() + '.png')


# block 4
def raw_testjob(gen_seed,inf_seed,clusters,points_per_cluster,num_iters,cols,alpha,beta,infer_hypers):
    paramDict = {"inferAlpha":infer_hypers,"inferBetas":infer_hypers,"alpha":alpha,"beta":beta}
    gen_state_with_data = dm.gen_dataset(gen_seed,None,cols,alpha,beta,np.repeat(points_per_cluster,clusters))
    gen_sample_output = dm.gen_sample(inf_seed, gen_state_with_data["observables"], num_iters,{"alpha":alpha,"betas":np.repeat(.1,cols)}
                                      ,None,paramDict=paramDict,gen_state_with_data=gen_state_with_data)
    predictive_prob = None ## dm.test_model(gen_state_with_data["observables"],gen_sample_output["state"]) ##
    return gen_sample_output,predictive_prob

# make memoized job (re-eval if the job code changes, or to reset cache)
testjob = Cloudless.memo.AsyncMemoize("testjob", ["gen_seed","inf_seed","clusters","points_per_cluster","num_iters","cols","alpha","beta","infer_hypers"], raw_testjob, override=True)


# block 5
# set constants (re-eval to change the scope of the plot)
CLUSTERS = 10
POINTS_PER_CLUSTER = 100
NUM_ITERS = 20
GEN_SEED = 0
NUM_SIMS = 3
##BELOW ARE FAIRLY STATIC VALUES
COLS = [20, 40, 80]
BETA = .1
INFER_HYPERS = False
ALPHA = 1 ## dm.mle_alpha(clusters=CLUSTERS,points_per_cluster=POINTS_PER_CLUSTER) ## 
INF_SEED = 0
# request the computation (re-eval if e.g. the range changes)
for cols in COLS:
    testjob(GEN_SEED,INF_SEED,CLUSTERS,POINTS_PER_CLUSTER,NUM_ITERS,cols,ALPHA,BETA,INFER_HYPERS)


# block 6
# get plot data locally (re-eval to get more data)
status = testjob.report_status()
time_delta = []
log_score = []
predictive_prob = []
ari = []
num_clusters = []
init_num_clusters = []
##DATASET = dm.gen_dataset(GEN_SEED,None,COLS,ALPHA,BETA,np.repeat(POINTS_PER_CLUSTER,CLUSTERS))
true_prob = None ## dm.test_model(DATASET["test_data"],DATASET["gen_state"]) ##
##
for (k, v) in testjob.iter():
    z_delta = np.array([x["timing"]["zs"]["delta"].total_seconds() for x in v[0]["stats"]]).cumsum()
    if "alpha" in v[0]["stats"][-1]["timing"]:
        alpha_delta = np.array([x["timing"]["alpha"]["delta"] for x in v[0]["stats"]]).cumsum()
    else:
        alpha_delta = np.zeros(np.shape(z_delta))
    if "beta" in v[0]["stats"][-1]["timing"]:
        beta_delta = np.array([x["timing"]["beta"]["delta"] for x in v[0]["stats"]]).cumsum()
    else:
        beta_delta = np.zeros(np.shape(z_delta))
    time_delta.append(z_delta+alpha_delta+beta_delta)
    log_score.append(np.array([x["score"] for x in v[0]["stats"]]))
    predictive_prob.append(np.array([x["predictive_prob"] for x in v[0]["stats"]]))    
    ari.append(np.array([x["ari"] for x in v[0]["stats"]]))    
    num_clusters.append(np.array([x["numClusters"] for x in v[0]["stats"]]))    
    init_num_clusters.append(v[0]["init_state"]["stats"]["numClusters"])


# block 7
do_plots()


# block 8
# examine the exceptions for the jobs that failed
testjob.report_status(verbose=True)
cluster_dist = np.sort(dm.gen_dataset(GEN_SEED,None,COLS,ALPHA,BETA,np.repeat(POINTS_PER_CLUSTER,CLUSTERS))["gen_state"]["phis"])
print str(sum(cluster_dist.cumsum()>.10)) + " clusters comprise 90% of data; " + str(sum(cluster_dist.cumsum()>.30)) + " clusters comprise 70% of data"
##testjob.terminate_pending() ## uncomment to kill non-running jobs
