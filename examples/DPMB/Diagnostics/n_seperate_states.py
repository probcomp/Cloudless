import datetime
import os
#
import matplotlib
matplotlib.use('Agg')
import numpy as np
#
import Cloudless.examples.DPMB.DPMB as dm
reload(dm)
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB.PDPMB as pdm
reload(pdm)
import Cloudless.examples.DPMB.PDPMB_State as pds
reload(pds)
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless
reload(Cloudless)

parser = rf.gen_default_arg_parser()
parser.add_argument('--plot_states',action='store_true')
args = parser.parse_args()
#
run_spec = rf.gen_runspec_from_argparse(args)
plot_problem = rf.gen_problem(
    run_spec["dataset_spec"],
    save_str="generative_state",
    )
problem = rf.gen_problem(run_spec["dataset_spec"])
print "Created problem"

# guts of rf.infer
dataset_spec = run_spec["dataset_spec"]
verbose_state = run_spec.get("verbose_state",False)
decanon_indices = run_spec.get("decanon_indices",None)
num_nodes = run_spec.get("num_nodes",2) # default to more than 1 node
hypers_every_N = run_spec.get("hypers_every_N",1)
time_seatbelt = run_spec.get("time_seatbelt",None)
ari_seatbelt = run_spec.get("ari_seatbelt",None)

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

# create a p_{model,state} using the parameters of a true PDPMB run
init_start_ts = datetime.datetime.now()
inference_state = state_type(dataset_spec["gen_seed"],
                             dataset_spec["num_cols"],
                             dataset_spec["num_rows"],
                             init_alpha=run_spec["infer_init_alpha"],
                             init_betas=run_spec["infer_init_betas"],
                             init_z=run_spec["infer_init_z"],
                             init_x = problem["xs"],
                             **state_kwargs
                             )
init_delta_seconds = hf.delta_since(init_start_ts)
#
print "...initialized"
transitioner = model_type(
    inf_seed = run_spec["infer_seed"],
    state = inference_state,
    infer_alpha = run_spec["infer_do_alpha_inference"],
    infer_beta = run_spec["infer_do_betas_inference"],
    **model_kwargs
    )

# evolve individual models, at each step 
#   record individual state's progress (num_clusters,score)
#          to verify its doing something meaningful
#   pull data back to PDPMB_State to determine predictive, ARI 

master_summaries = []
node_summaries = []
master_summaries.append(
    transitioner.extract_state_summary(
        true_zs=problem["zs"]
        ,verbose_state=verbose_state
        ,test_xs=problem["test_xs"]))
master_summaries[-1]["timing"]["init"] = init_delta_seconds
#
node_summaries_element = []
for node in inference_state.model_list:
    node_summaries_element.append(node.extract_state_summary())
node_summaries.append(node_summaries_element)
print "saved initialization"

last_valid_zs = None
decanon_indices = None
for iter_idx in range(run_spec["num_iters"]):
    # never actually transition transitioner
    # only inference_state.model_list elements

    transitioner_return_list = []
    for node in inference_state.model_list:
        transitioner_return_list.append(
            node.transition(time_seatbelt=time_seatbelt))
        if transitioner_return_list[-1] is not None:
            break
    # check for seatbelt violations in any of the individual states
    if len(filter(None,transitioner_return_list)) > 0:
        break
    hf.printTS("finished doing iteration " + str(iter_idx))

    # plot states?
    if args.plot_states:
        for node_idx,node in enumerate(inference_state.model_list):
            save_str = "_".join([
                    "child_state_idx",
                    str(node_idx),
                    "iter_idx",
                    str(iter_idx)
                    ])
            node.state.plot(save_str=save_str)
    # get node summaries
    node_summaries_element = []
    for node in inference_state.model_list:
        node_summaries_element.append(node.extract_state_summary())
    time_elapsed = max([
            summary["timing"].get("run_sum",0)
            for summary in node_summaries_element
            ])
    node_summaries.append(node_summaries_element)
    # get master summary 
    next_summary = transitioner.extract_state_summary(
        true_zs=problem["zs"]
        ,verbose_state=verbose_state
        ,test_xs=problem["test_xs"])
    time_elapsed_str = "%.1f" % time_elapsed
    hf.printTS("time elapsed: " + time_elapsed_str)
    master_summaries.append(next_summary)
    hf.printTS("finished saving iteration " + str(iter_idx))
    if hasattr(transitioner.state,"getZIndices"):
        last_valid_zs = transitioner.state.getZIndices()
        decanon_indices = transitioner.state.get_decanonicalizing_indices()
master_summaries[-1]["last_valid_zs"] = last_valid_zs
master_summaries[-1]["decanon_indices"] = decanon_indices


# stuff an AsyncMemoize to create plots
memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], rf.infer, override=False)
memoized_infer.memo[str((run_spec,))] = master_summaries
memoized_infer.args[str((run_spec,))] = (run_spec,)

rf.pickle_if_done(memoized_infer,file_str="n_seperate_states_results.pkl")
rf.try_plots(
    memoized_infer,
    which_measurements=["predictive","ari","num_clusters","score"],
    save_dir="./"
    )
