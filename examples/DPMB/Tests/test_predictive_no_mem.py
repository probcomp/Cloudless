import matplotlib
matplotlib.use('Agg')
import numpy as np
import numpy.random as nr
import matplotlib.pylab as pylab
import datetime
import re
import gc
#
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)

from Cloudless.examples.DPMB.remote_functions import *

run_spec = rf.gen_default_run_spec()
run_spec["dataset_spec"]["num_rows"] = 64*64
run_spec["dataset_spec"]["gen_z"] = ("balanced",64)
run_spec["num_iters"] = 5
run_spec["time_seatbelt"] = 600

dataset_spec = run_spec["dataset_spec"]
problem = gen_problem(dataset_spec)
verbose_state = run_spec.get("verbose_state",False)
decanon_indices = run_spec.get("decanon_indices",None)
num_nodes = run_spec.get("num_nodes",1)
hypers_every_N = run_spec.get("hypers_every_N",1)
time_seatbelt = run_spec.get("time_seatbelt",None)
ari_seatbelt = run_spec.get("ari_seatbelt",None)
#
if verbose_state:
    print "doing run: "
    for (k, v) in run_spec.items():
        if k.find("seed") != -1:
            print "   " + "hash(" + str(k) + ")" + " ---- " + str(hash(str(v)))
        else:
            print "   " + str(k) + " ---- " + str(v)
#
state_kwargs = {}
model_kwargs = {}
print "initializing"
if num_nodes == 1:
    state_type = ds.DPMB_State
    model_type = dm.DPMB
    state_kwargs = {"decanon_indices":decanon_indices}
else:
    state_type = pds.PDPMB_State
    model_type = pdm.PDPMB
    state_kwargs = {"num_nodes":num_nodes}
    model_kwargs = {"hypers_every_N":hypers_every_N}

inference_state = state_type(dataset_spec["gen_seed"],
                             dataset_spec["num_cols"],
                             dataset_spec["num_rows"],
                             init_alpha=run_spec["infer_init_alpha"],
                             init_betas=run_spec["infer_init_betas"],
                             init_z=run_spec["infer_init_z"],
                             init_x = problem["xs"],
                             **state_kwargs
                             )

print "...initialized"
transitioner = model_type(
    inf_seed = run_spec["infer_seed"],
    state = inference_state,
    infer_alpha = run_spec["infer_do_alpha_inference"],
    infer_beta = run_spec["infer_do_betas_inference"],
    **model_kwargs
    )
#
summaries = []
summaries.append(
    transitioner.extract_state_summary(
        true_zs=problem["zs"]
        ,verbose_state=verbose_state
        ,test_xs=problem["test_xs"]))
#
print "saved initialization"
#
last_valid_zs = None
decanon_indices = None
for i in range(run_spec["num_iters"]):
    transition_return = transitioner.transition(
        time_seatbelt=time_seatbelt
        ,ari_seatbelt=ari_seatbelt
        ,true_zs=problem["zs"]) # true_zs necessary for seatbelt 
    hf.printTS("finished doing iteration" + str(i))
    next_summary = transitioner.extract_state_summary(
        true_zs=problem["zs"]
        ,verbose_state=verbose_state
        ,test_xs=problem["test_xs"])
    if transition_return is not None:
        summaries[-1]["break"] = transition_return
        summaries[-1]["failed_info"] = next_summary
        break
    summaries.append(next_summary)
    hf.printTS("finished saving iteration" + str(i))
    if hasattr(transitioner.state,"getZIndices"):
        last_valid_zs = transitioner.state.getZIndices()
        decanon_indices = transitioner.state.get_decanonicalizing_indices()
summaries[-1]["last_valid_zs"] = last_valid_zs
summaries[-1]["decanon_indices"] = decanon_indices
