import sys
import matplotlib
import thread
matplotlib.use('Agg')
import matplotlib.pylab as pylab
import numpy as np
import time
##
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB.DPMB as dm
reload(dm)
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
##
import Cloudless
reload(Cloudless)
import Cloudless.memo
reload(Cloudless.memo)
##
if sys.platform == "win32":
    sys.path.append("c:/")

if sys.platform != "win32" and True:
    Cloudless.base.remote_mode()
    Cloudless.base.remote_exec('import matplotlib.pylab as pylab')
    Cloudless.base.remote_exec('import numpy as np')
    Cloudless.base.remote_exec('import Cloudless.examples.DPMB.DPMB_State as ds')
    Cloudless.base.remote_exec('reload(ds)')
    Cloudless.base.remote_exec('import Cloudless.examples.DPMB.DPMB as dm')
    Cloudless.base.remote_exec('reload(dm)')
    Cloudless.base.remote_exec('import Cloudless.examples.DPMB.helper_functions as hf')
    Cloudless.base.remote_exec('reload(hf)')
    Cloudless.base.remote_exec('import Cloudless.examples.DPMB.remote_functions as rf')
    Cloudless.base.remote_exec('reload(rf)')
    Cloudless.base.remote_exec('import Cloudless')
    Cloudless.base.remote_exec('reload(Cloudless)')
    Cloudless.base.remote_exec('import Cloudless.memo')
    Cloudless.base.remote_exec('reload(Cloudless.memo)')    
else:
    print "!!!NOT REMOTE!!!"

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

memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], rf.infer, override=True)
lock = thread.allocate_lock()

num_threads = 2
run_spec_list = []
run_spec_list.append(gen_run_spec())
for run_spec_infer_seed in range(1,1+num_threads):
    run_spec = gen_run_spec()
    run_spec["infer_seed"] = run_spec_infer_seed
    run_spec_list.append(run_spec)

cj_list = []
for run_spec in run_spec_list:
    cj = rf.Chunked_Job(run_spec,asyncmemo=memoized_infer
                        ,chunk_iter=chunk_iter,lock=lock)
    cj_list.append(cj)

# for cj in cj_list:
#     time.sleep(1)
#     cj.start()

while not all([cj.done for cj in cj_list]):
    time.sleep(10)
    memoized_infer.advance()
    for cj in cj_list:
        cj.evolve_chain()

# for cj in cj_list:
#     cj.join()

one_job_value = rf.infer(run_spec_list[0]) # make it look like cj_list[0]

# check that inference xs are same as generated xs
problem = rf.gen_problem(run_spec_list[0]["dataset_spec"])
inf_xs_list = [summary["xs"] for summary in 
               cj_list[0].consolidated_data if "xs" in summary]
gen_matches_inf = [(np.array(problem["xs"])==np.array(inf_xs)).all()
                   for inf_xs in inf_xs_list]
#
assert all(gen_matches_inf), "inference not run on correct xs!"
print "Inference XS match generated XS"


# check that one_job, chunked_job solutions match
assert all(np.array(one_job_value[-1]["zs"]) == np.array(cj_list[0].consolidated_data[-1]["zs"])), "Inference didn't match!"
print "Inference resume matched!"


key = "test_lls"
for cj in cj_list:
    print [np.log(sum(np.exp(list(x[key])))) for x in cj.consolidated_data]

print [np.log(sum(np.exp(list(x[key])))) for x in one_job_value]
