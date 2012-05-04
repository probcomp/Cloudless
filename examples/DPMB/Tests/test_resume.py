import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as pylab
import numpy as np
##
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB.DPMB as dm
reload(dm)
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
##
import Cloudless
reload(Cloudless)
import Cloudless.memo
reload(Cloudless.memo)
##
if sys.platform == "win32":
    sys.path.append("c:/")

##create a dataset_spec->problem->run_spec
##infer with xs returned
##compare xs

num_iters = 11
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
    run_spec["verbose_state"] = True

    return run_spec

run_spec = gen_run_spec()
one_job_value = hf.infer(run_spec)

problem = hf.gen_problem(dataset_spec)
inf_xs_list = [inf["xs"] for inf in one_job_value if "xs" in inf]
gen_matches_inf = [(np.array(problem["xs"])==np.array(inf_xs)).all()
                   for inf_xs in inf_xs_list]

assert all(gen_matches_inf), "inference not run on correct xs!"
print "Inference XS match generated XS"

##create a dataset_spec->problem->run_spec
##infer 20 with xs returned
##infer 10 with xs returned -> infer 10 with xs returned
##compare all xs
##compare zs of second 10 and 20

import copy
num_runs = 3
num_iters_list = np.repeat(num_iters/num_runs,num_runs)
num_iters_list[-1] = num_iters - sum(num_iters_list[:-1])
run_spec_list = []
job_list = []
for sub_num_iters in num_iters_list:
    sub_run_spec = gen_run_spec()
    sub_run_spec["num_iters"] = sub_num_iters
    if len(job_list) != 0:
        sub_run_spec["infer_seed"] = copy.deepcopy(job_list[-1][-1]["inf_seed"])
        sub_run_spec["infer_init_alpha"] = copy.deepcopy(job_list[-1][-1]["alpha"])
        sub_run_spec["infer_init_betas"] = copy.deepcopy(job_list[-1][-1]["betas"])
        sub_run_spec["infer_init_z"] = copy.deepcopy(job_list[-1][-1]["zs"])
        sub_run_spec["decanon_indices"] = job_list[-1][-1]["decanon_indices"]
    run_spec_list.append(sub_run_spec)
    job_list.append(hf.infer(sub_run_spec))
    assert "break" not in job_list[-1][-1], "Prior inference had seatbelt break, not appropriate for verifying inference continutity"

    
assert all(np.array(one_job_value[-1]["zs"]) == np.array(job_list[-1][-1]["zs"])), "Inference didn't match!"
print "Inference resume matched!"
