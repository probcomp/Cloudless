#!python

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


# block 3
# make a plot (iterate on this block to fix layout/display issues)
def do_plots(x_vars=None,y_vars=None,time_delta=None,true_prob=None,ari=None
             ,path=None,CLUSTERS=None,POINTS_PER_CLUSTER=None,NUM_ITERS=None,INFER_ALPHA=None,INFER_BETA=None
             ,INIT_METHOD=None,ALPHA=None,BETA=None,COLS=None,GEN_SEED=None,NUM_SIMS=None,packed_params=None):
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
            pylab.plot(locals()[x_var_str+"_ARR"], np.array(locals()[y_var_str]).T)
            if y_var_str=="predictive_prob" and true_prob is not None:
                pylab.hlines(true_prob,*fh.get_axes()[0].get_xlim(),colors='r',linewidth=3)
            pylab.title(TITLE_STR)
            pylab.xlabel(locals()[x_var_str+"_LABEL"])
            pylab.ylabel(y_var_str)
            pylab.ion()
            pylab.show()
            ##
            if INFER_ALPHA is None:
                INF_ALPHA_STR = "A="+str(ALPHA)
            elif type(INFER_ALPHA)==dict and "method" in INFER_ALPHA and INFER_ALPHA["method"] == "DISCRETE_GIBBS":
                INF_ALPHA_STR = "A=DiscG"
            else:
                print "Unkown INFER_ALPHA value"
                return
            if INFER_BETA is None:
                INF_BETA_STR = "B="+str(BETA)
            elif type(INFER_BETA)==dict and "method" in INFER_BETA and INFER_BETA["method"] == "DISCRETE_GIBBS":
                INF_BETA_STR = "A=DiscG"
            else:
                print "Unkown INFER_BETA value"
                return
            inf_str = INF_ALPHA_STR,INF_BETA_STR
            if type(INIT_METHOD) != dict or "method" not in INIT_METHOD or "sample_prior" == INIT_METHOD["method"]:
                init_str = "init=P"
            elif "all_together" == INIT_METHOD["method"]:
                init_str = "init=1"
            elif "all_separate" == INIT_METHOD["method"]:
                init_str = "init=N"
            ##PRESUME COLS is always 256 and GEN_SEED is always 0
            config_prefix = ",".join(inf_str.__add__((init_str,"CL="+str(CLUSTERS),"PPC="+str(POINTS_PER_CLUSTER)))) 
            variable_infix = "_" + y_var_str+'_by_' + x_var_str.lower()
            import os
            file_name_prefix = os.path.join(path,config_prefix + variable_infix)
            pylab.savefig(file_name_prefix  + '.png')
            ##
            import cPickle
            pklDataFile = file_name_prefix + ".pkl"
            with open(pklDataFile,"wb") as fh:
                cPickle.dump(packed_params,fh,-1)


def queue_jobs(path=None,CLUSTERS=None,POINTS_PER_CLUSTER=None,NUM_ITERS=None,INFER_ALPHA=None,INFER_BETA=None,INIT_METHOD=None,ALPHA=None,BETA=None,COLS=None,GEN_SEED=None,NUM_SIMS=None,packed_params=None):
    # block 5
    # set constants (re-eval to change the scope of the plot)
    ##
    # make memoized job (re-eval if the job code changes, or to reset cache)
    testjob = Cloudless.memo.AsyncMemoize("testjob", ["gen_seed","inf_seed","clusters","points_per_cluster","num_iters","cols","alpha","beta","init_method","infer_alpha","infer_beta"], raw_testjob, override=True)
    # request the computation (re-eval if e.g. the range changes)
    for inf_seed in range(NUM_SIMS):
        testjob(GEN_SEED,inf_seed,CLUSTERS,POINTS_PER_CLUSTER,NUM_ITERS,COLS,ALPHA,BETA,INIT_METHOD,INFER_ALPHA,INFER_BETA)
    return testjob

def can_plot(testjob):
    return testjob.report_status()["waiting"]==0

def extract_and_plot(testjob,path=None,CLUSTERS=None,POINTS_PER_CLUSTER=None,NUM_ITERS=None,INFER_ALPHA=None,INFER_BETA=None,INIT_METHOD=None,ALPHA=None,BETA=None,COLS=None,GEN_SEED=None,NUM_SIMS=None,packed_params=None):
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
    do_plots(y_vars=y_vars,ari=ari,time_delta=time_delta,packed_params=packed_params,**packed_params)

def filter_plottable(job_list,done_list):
    jobs_ready = []
    jobs_not_ready = []
    for testjob,packed_params in job_list:
        if can_plot(testjob):
            jobs_ready.append((testjob,packed_params))
        else:
            jobs_not_ready.append((testjob,packed_params))
    ##
    for testjob,packed_params in jobs_ready:
        extract_and_plot(testjob,packed_params=packed_params,**packed_params)
    return jobs_not_ready,np.append(done_list,jobs_ready)

def create_dict():
    low_val=.01
    high_val=1E4
    packed_params = {
        "path":"Plots"
        ,"CLUSTERS":10
        ,"POINTS_PER_CLUSTER":10
        ,"NUM_ITERS":10
        ,"INFER_ALPHA":[None,{"method":"DISCRETE_GIBBS","low_val":low_val,"high_val":high_val,"n_grid":100}][1]
        ,"INFER_BETA":[None,{"method":"DISCRETE_GIBBS","low_val":low_val,"high_val":high_val,"n_grid":100}][0]
        ,"INIT_METHOD":[{"method":"all_together"},{"method":"all_separate"},{"method":"sample_prior"}][1]
        ##BELOW ARE FAIRLY STATIC VALUES
        ,"ALPHA":100 ## hf.mle_alpha(clusters=CLUSTERS,points_per_cluster=POINTS_PER_CLUSTER) ## 
        ,"BETA":.1
        ,"COLS":256
        ,"GEN_SEED":0
        ,"NUM_SIMS":3
    }
    return packed_params
