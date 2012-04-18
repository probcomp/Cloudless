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
Cloudless.base.remote_exec('import Cloudless.examples.DPMB_helper_functions as hf')
Cloudless.base.remote_exec('reload(hf)')
Cloudless.base.remote_exec('import time')
Cloudless.base.remote_exec('import numpy as np')
import Cloudless.examples.DPMB as dm
reload(dm)
import Cloudless.examples.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB_helper_functions as hf
reload(hf)
import time
import numpy as np


# block 4
def raw_testjob(gen_seed,inf_seed,clusters,points_per_cluster,num_iters,cols,alpha,beta,init_method,infer_alpha,infer_beta):
    paramDict = {"infer_alpha":infer_alpha,"infer_beta":infer_beta,"alpha":alpha,"beta":beta}
    gen_state_with_data = hf.gen_dataset(gen_seed=gen_seed,gen_rows=None,gen_cols=cols,gen_alpha=alpha,gen_beta=beta,zDims=np.repeat(points_per_cluster,clusters))
    gen_sample_output = hf.gen_sample(inf_seed=inf_seed,train_data=gen_state_with_data["observables"]
                                      , num_iters=num_iters,init_method=init_method,infer_alpha=infer_alpha
                                      , infer_beta=infer_beta,paramDict=paramDict,gen_state_with_data=gen_state_with_data)
    predictive_prob = hf.test_model(gen_state_with_data["observables"],gen_sample_output["state"])
    return gen_sample_output,predictive_prob


def run(GEN_SEED=None):
    # block 3
    # make a plot (iterate on this block to fix layout/display issues)
    def do_plots(x_vars=None,y_vars=None,path=None):
        x_vars = x_vars if x_vars is not None else ["TIME","ITER"]
        path = path if path is not None else ""
        CLUSTER_STR = str(POINTS_PER_CLUSTER) + "*" + str(CLUSTERS)
        TITLE_STR = "{Clusters*Points}xCols: " + CLUSTER_STR + "x" + str(COLS) + "; NUM_ITERS: " + str(NUM_ITERS) + "; ALPHA: " + str(ALPHA)
        TIME_LABEL = 'Time Elapsed (seconds)'
        ITER_LABEL = 'Iteration number'
        TIME_ARR = np.array(time_delta).T
        ITER_ARR = np.repeat(range(NUM_ITERS),NUM_SIMS).reshape(NUM_ITERS,NUM_SIMS)
        ##
        for y_var_str in y_vars:
            for x_var_str in x_vars:
                fh = pylab.figure()
                pylab.plot(locals()[x_var_str+"_ARR"], np.array(y_vars[y_var_str]).T)
                if y_var_str=="predictive_prob" and true_prob is not None:
                    pylab.hlines(true_prob,*fh.get_axes()[0].get_xlim(),colors='r',linewidth=3)
                pylab.title(TITLE_STR)
                pylab.xlabel(locals()[x_var_str+"_LABEL"])
                pylab.ylabel(y_var_str)
                pylab.ion()
                pylab.show()
                ##
                inf_str = INFER_HYPERS if INFER_HYPERS is not None else "A="+str(ALPHA),"B="+str(BETA)
                config_prefix = ",".join(inf_str.__add__(("CL="+str(CLUSTERS),"PPC="+str(POINTS_PER_CLUSTER),"CO="+str(COLS),"GEN_SEED="+str(GEN_SEED)))) 
                variable_infix = "_" + y_var_str+'_by_' + x_var_str.lower()
                import os
                file_name_prefix = os.path.join(path,config_prefix + variable_infix)
                pylab.savefig(file_name_prefix  + '.png')
                ##
                import cPickle
                pklDataFile = file_name_prefix + ".pkl"
                with open(pklDataFile,"wb") as fh:
                    cPickle.dump(sim_params,fh,-1)
    # block 5
    # set constants (re-eval to change the scope of the plot)
    import sys
    path = "" if len(sys.argv)<4 else sys.argv[3]
    CLUSTERS = 10 if len(sys.argv)<2 else int(sys.argv[1])
    POINTS_PER_CLUSTER = 10 if len(sys.argv)<3 else int(sys.argv[2])
    NUM_ITERS = 10
    ##BELOW ARE FAIRLY STATIC VALUES
    INFER_HYPERS = None
    COLS = 256
    ALPHA = 100 ## hf.mle_alpha(clusters=CLUSTERS,points_per_cluster=POINTS_PER_CLUSTER) ## 
    BETA = .1
    GEN_SEED = 0 if GEN_SEED is None else GEN_SEED
    NUM_SIMS = 3
    ##
    low_val = .01
    high_val = 1E4
    INFER_ALPHA = [None,{"method":"DISCRETE_GIBBS","low_val":low_val,"high_val":high_val,"n_grid":100}][0]
    INFER_BETA = [None,{"method":"DISCRETE_GIBBS","low_val":low_val,"high_val":high_val,"n_grid":100}][0]
    INIT_METHOD = ["all_together","all_separate","sample_prior"][1]
    ##
    # make memoized job (re-eval if the job code changes, or to reset cache)
    testjob = Cloudless.memo.AsyncMemoize("testjob", ["gen_seed","inf_seed","clusters","points_per_cluster","num_iters","cols","alpha","beta","init_method","infer_alpha","infer_beta"], raw_testjob, override=True)
    # request the computation (re-eval if e.g. the range changes)
    for inf_seed in range(NUM_SIMS):
        testjob(GEN_SEED,inf_seed,CLUSTERS,POINTS_PER_CLUSTER,NUM_ITERS,COLS,ALPHA,BETA,INIT_METHOD,INFER_ALPHA,INFER_BETA)
    sim_params = {
        "CLUSTERS":CLUSTERS
        ,"POINTS_PER_CLUSTER":POINTS_PER_CLUSTER
        ,"NUM_ITERS":NUM_ITERS
        ,"INFER_HYPERS":INFER_HYPERS
        ,"COLS":COLS
        ,"ALPHA":ALPHA
        ,"BETA":BETA
        ,"GEN_SEED":GEN_SEED
        ,"NUM_SIMS":NUM_SIMS
    }
    # block 6
    # get plot data locally (re-eval to get more data)
    import time ##when run as script, must wait for all data to be ready
    while testjob.report_status()["waiting"]!=0:
        time.sleep(1)
    ##
    status = testjob.report_status()
    time_delta = []
    log_score = []
    predictive_prob = []
    ari = []
    num_clusters = []
    init_num_clusters = []
    ##DATASET = hf.gen_dataset(GEN_SEED,None,COLS,ALPHA,BETA,np.repeat(POINTS_PER_CLUSTER,CLUSTERS))
    true_prob = None ## hf.test_model(DATASET["test_data"],DATASET["gen_state"]) ## 
    ##
    for (k, v) in testjob.iter():
        z_delta = np.array([x["timing"]["zs"]["delta"].total_seconds() for x in v[0]["stats"]]).cumsum()
        if "alpha" in v[0]["stats"][-1]["timing"]:
            alpha_delta = np.array([x["timing"]["alpha"]["delta"] for x in v[0]["stats"]]).cumsum()
        else:
            alpha_delta = np.zeros(np.shape(z_delta))
        ##
        if "beta" in v[0]["stats"][-1]["timing"]:
            beta_delta = np.array([x["timing"]["beta"]["delta"] for x in v[0]["stats"]]).cumsum()
        else:
            beta_delta = np.zeros(np.shape(z_delta))
        ##
        time_delta.append(z_delta+alpha_delta+beta_delta)
        log_score.append(np.array([x["score"] for x in v[0]["stats"]]))
        predictive_prob.append(np.array([x["predictive_prob"] for x in v[0]["stats"]]))
        ari.append(np.array([x["ari"] for x in v[0]["stats"]]))
        num_clusters.append(np.array([x["numClusters"] for x in v[0]["stats"]]))
        init_num_clusters.append(v[0]["init_state"]["stats"]["numClusters"])
    # block 7
    y_vars = {"ari":ari}
    do_plots(y_vars=y_vars,path=path)
