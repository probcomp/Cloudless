#!python
import os
import datetime
from collections import Counter
#
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas
from matplotlib.mlab import find
#
import Cloudless.examples.DPMB.settings as settings
reload(settings)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless.examples.DPMB.s3_helper as s3h
reload(s3h)
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB.DPMB as dm
reload(dm)
import Cloudless.examples.DPMB.PDPMB_State as pds
reload(pds)
import Cloudless.examples.DPMB.PDPMB as pdm
reload(pdm)

problem_file = os.path.join(settings.data_dir,settings.cifar_100_problem_file)
image_dir = os.path.join(settings.data_dir,settings.cifar_100_image_dir)
clustering_dir = os.path.join(settings.data_dir,settings.clustering_dir)
s3h.S3_helper().verify_file(settings.cifar_100_problem_file)
#
pkl_data = rf.unpickle(problem_file)
init_x = pkl_data["subset_xs"]
true_zs,ids = hf.canonicalize_list(pkl_data["subset_zs"])
test_xs = pkl_data['test_xs']
image_indices = pkl_data['chosen_indices']
#
beta_d = 2.0
dataset_spec = {}
dataset_spec["pkl_file"] = problem_file
dataset_spec["gen_seed"] = 0
dataset_spec["num_cols"] = 256
dataset_spec["num_rows"] = len(true_zs)
dataset_spec["gen_alpha"] = 1.0
dataset_spec["gen_betas"] = np.repeat(beta_d, dataset_spec["num_cols"])
dataset_spec["gen_z"] = true_zs
init_z = None # gibbs-type init
# init_z = dataset_spec["gen_z"] # ground truth init
#
problem = {}
problem["zs"] = dataset_spec["gen_z"]
problem["xs"] = init_x
problem["test_xs"] = test_xs
#
state = ds.DPMB_State(dataset_spec["gen_seed"],
                      dataset_spec["num_cols"],
                      dataset_spec["num_rows"],
                      init_alpha=dataset_spec["gen_alpha"],
                      init_betas=dataset_spec["gen_betas"],
                      init_z=dataset_spec["gen_z"],
                      init_x = init_x)

run_spec = {}
run_spec["dataset_spec"] = dataset_spec
run_spec["num_iters"] = 0
run_spec["infer_seed"] = 0
run_spec["infer_init_alpha"] = 1.0
run_spec["infer_init_betas"] = dataset_spec["gen_betas"].copy()
run_spec["infer_init_z"] = init_z
run_spec["infer_do_alpha_inference"] = True
run_spec["infer_do_betas_inference"] = True

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
summaries[-1]["timing"]["init"] = init_delta_seconds
#
print "saved initialization"
#
last_valid_zs = None
decanon_indices = None


def do_transitions(num_transitions):
    for i in xrange(num_transitions):
        transition_return = transitioner.transition(
            time_seatbelt=time_seatbelt
            ,ari_seatbelt=ari_seatbelt
            ,true_zs=problem["zs"]) # true_zs necessary for seatbelt 
        hf.printTS("finished doing iteration " + str(i))
        next_summary = transitioner.extract_state_summary(
            true_zs=problem["zs"]
            ,verbose_state=verbose_state
            ,test_xs=problem["test_xs"])
        time_elapsed_str = "%.1f" % next_summary["timing"].get("run_sum",0)
        hf.printTS("time elapsed: " + time_elapsed_str)
        if transition_return is not None:
            summaries[-1]["break"] = transition_return
            summaries[-1]["failed_info"] = next_summary
            break
        summaries.append(next_summary)
        hf.printTS("finished saving iteration " + str(i))
        if hasattr(transitioner.state,"getZIndices"):
            last_valid_zs = transitioner.state.getZIndices()
            decanon_indices = transitioner.state.get_decanonicalizing_indices()
        #
        empty_links()
        link_helper(zs=last_valid_zs)
        write_helper(zs=last_valid_zs)
        print "mean test_ll: " + str(np.mean(summaries[-1]['test_lls']))
        if transitioner.transition_count % 10 == 0:
            pkl_summaries_helper()
    summaries[-1]["last_valid_zs"] = last_valid_zs
    summaries[-1]["decanon_indices"] = decanon_indices

def plot_full_state(which_betas=None):
    if which_betas is None:
        which_betas = xrange(len(inference_state.betas))
    for beta_idx in which_betas:
        save_str = "cifar_init_state_beta" + str(beta_idx) + "_vector" + str(beta_idx)
        inference_state.plot(
            which_plots=["alpha","beta","cluster"],
            save_str=save_str,
            show=False,
            beta_idx=beta_idx,
            vector_idx=beta_idx
            )

def write_state(filename,data=None):
    if data is None:
        data = summaries[-1]['last_valid_zs']
    pandas.Series(data,image_indices).to_csv(filename)

def do_stats():
    print "scan times: "
    print str(np.diff([summary["timing"]["run_sum"] for summary in summaries]))
    print "cluster number trajectory: "
    str(np.diff([summary['num_clusters'] for summary in summaries]))
    print "mean test_ll: "
    print ["%.3f" % np.mean(summary['test_lls']) for summary in summaries]
    #
    num_clusters = summaries[-1]['num_clusters']
    print "cluster counts: " + str(zip(xrange(num_clusters),summaries[-1]['cluster_counts']))
    print "mean test_ll: " + str(np.mean(summaries[-1]['test_lls']))
    
def link_helper(zs=None,image_indices=image_indices):
    if zs is None:
        zs = summaries[-1]["last_valid_zs"]
    series = pandas.Series(zs,image_indices)
    hf.create_links(series,image_dir,clustering_dir)

def empty_links():
    if len(os.listdir(clustering_dir)) != 0:
        os.system("rm -rf " + os.path.join(clustering_dir,"*"))

def write_helper(zs=None):
    if zs is None:
        zs = summaries[-1]["last_valid_zs"]
    filename = "cifar_100_state_iter"+str(transitioner.transition_count)+".csv"
    write_state(filename,zs)

def pkl_summaries_helper():
    filename = "cifar_100_summaries_iter"+str(transitioner.transition_count)+".pkl.gz"
    rf.pickle(summaries,filename)

if True:
    do_transitions(1)
    do_stats()
    plot_full_state(range(10))
    empty_links()
    link_helper()
    write_helper()
    pkl_summaries_helper()

# import Cloudless.examples.DPMB.helper_functions as hf
# import Cloudless.examples.DPMB.settings as settings
# hf.create_links("cifar_10_state_iter70.csv",settings.cifar_100_image_dir,settings.clustering_dir)
