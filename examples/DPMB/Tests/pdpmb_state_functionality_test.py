#!python

import numpy as np
#
import remote_functions as rf
import PDPMB_State as pds

num_iters = 50 # 10 
chunk_iter = 27 # 3
def gen_run_spec():
    dataset_spec = {}
    dataset_spec["gen_seed"] = 0
    dataset_spec["num_cols"] = 16
    dataset_spec["num_rows"] = 32
    dataset_spec["gen_alpha"] = 1.0 #FIXME: could make it MLE alpha later
    dataset_spec["gen_betas"] = np.repeat(0.1, dataset_spec["num_cols"])
    dataset_spec["gen_z"] = ("balanced", 10)
    dataset_spec["N_test"] = 10
    ##
    run_spec = {}
    run_spec["num_iters"] = num_iters
    run_spec["infer_seed"] = 0
    run_spec["infer_init_alpha"] = 1.0
    run_spec["infer_init_betas"] = np.repeat(0.1, dataset_spec["num_cols"])
    run_spec["infer_do_alpha_inference"] = True
    run_spec["infer_do_betas_inference"] = True
    run_spec["infer_init_z"] = None
    run_spec["dataset_spec"] = dataset_spec
    run_spec["time_seatbelt"] = 600
    run_spec["ari_seatbelt"] = None
    run_spec["verbose_state"] = False
    #
    return run_spec

run_spec = gen_run_spec()
problem = rf.gen_problem(run_spec["dataset_spec"])
init_x = problem["xs"]
num_cols = 10
pds = pdf.PDPMB_State(
    init_alpha=1,init_betas=np.repeat(1.0,num_cols)
    ,init_gammas=np.repeat(1.0/num_cols,num_cols)
    ,init_x=init_x,gen_seed=0,num_nodes=2)
