import matplotlib
matplotlib.use('Agg')
import DPMB_plotutils as dp
reload(dp)
import DPMB_State as ds
reload(ds)
import DPMB_helper_functions as hf
reload(hf)
import numpy as np
import matplotlib.pylab as pylab
##
import sys
import Cloudless
reload(Cloudless)
import Cloudless.memo
reload(Cloudless.memo)


# block 2
# configure remote nodes
# TODO: Clean up naming of load balanced vs direct views
Cloudless.base.remote_mode()
Cloudless.base.remote_exec('import Cloudless.examples.DPMB_plotutils as dp')
Cloudless.base.remote_exec('reload(dp)')
Cloudless.base.remote_exec('import Cloudless.examples.DPMB_State as ds')
Cloudless.base.remote_exec('reload(ds)')
Cloudless.base.remote_exec('import Cloudless.examples.DPMB_helper_functions as hf')
Cloudless.base.remote_exec('reload(hf)')
Cloudless.base.remote_exec('import numpy as np')
Cloudless.base.remote_exec('import matplotlib.pylab as pylab')
import Cloudless.examples.DPMB_plotutils as dp
reload(dp)
import Cloudless.examples.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB_helper_functions as hf
reload(hf)
import numpy as np
import matplotlib.pylab as pylab


ALL_DATASET_SPECS = []

for num_clusters in [2,8,32]:##[2**(j+1) for j in [2]]:
    dataset_spec = {}
    dataset_spec["gen_seed"] = 0
    dataset_spec["num_cols"] = 128
    dataset_spec["num_rows"] = 256
    dataset_spec["gen_alpha"] = 1.0 #FIXME: could make it MLE alpha later
    dataset_spec["gen_betas"] = np.repeat(0.1, dataset_spec["num_cols"])
    dataset_spec["gen_z"] = ("balanced", num_clusters)
    dataset_spec["N_test"] = 5
    ALL_DATASET_SPECS.append(dataset_spec)

print "Generated " + str(len(ALL_DATASET_SPECS)) + " dataset specs!"

ALL_PROBLEMS = []
for dataset_spec in ALL_DATASET_SPECS:
    problem = hf.gen_problem(dataset_spec)
    ALL_PROBLEMS.append(problem)

print "Generated " + str(len(ALL_PROBLEMS)) + " problems!"

# now we have, in ALL_PROBLEMs, the dataset specs, along with training
# data, test data, the generating zs for the training data, and the average
# log probability of the test data under the generating model

# NOTE: Can clean up using itertools.product()
# http://docs.python.org/library/itertools.html#itertools.product
ALL_RUN_SPECS = []
num_iters = 200
count = 0
for problem in ALL_PROBLEMS:
    for infer_seed in range(1):
        for infer_init_alpha in [1.0]: #note: we're never trying sample-alpha-from-prior-for-init
            for infer_init_betas in [np.repeat(0.1, dataset_spec["num_cols"])]:
                for infer_do_alpha_inference in [True, False]:
                    for infer_do_betas_inference in [True, False]:
                        for infer_init_z in [1, "N", None]:
                            run_spec = {}
                            run_spec["num_iters"] = num_iters
                            run_spec["infer_seed"] = infer_seed
                            run_spec["infer_init_alpha"] = infer_init_alpha
                            run_spec["infer_init_betas"] = infer_init_betas
                            run_spec["infer_do_alpha_inference"] = infer_do_alpha_inference
                            run_spec["infer_do_betas_inference"] = infer_do_betas_inference
                            run_spec["infer_init_z"] = infer_init_z
                            run_spec["problem"] = problem
                            ALL_RUN_SPECS.append(run_spec)

print "Generated " + str(len(ALL_RUN_SPECS)) + " run specs!"
# until success:
#ALL_RUN_SPECS = ALL_RUN_SPECS[:1]

print "Running inference on " + str(len(ALL_RUN_SPECS)) + " problems..."

# now request the inference
memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], hf.infer, override=True) #FIXME once we've debugged, we can eliminate this override

print "Created memoizer"

for run_spec in ALL_RUN_SPECS:
    memoized_infer(run_spec)

# now you can interactively call
for problem_idx,target_problem in enumerate(ALL_PROBLEMS):
    col_str = "_cols" + str(target_problem["dataset_spec"]["num_cols"])
    row_str = "_rows" + str(target_problem["dataset_spec"]["num_rows"])
    # FIXME : temporary workaround for gen_z bug
    ##cluster_str = "_clusters" + str(target_problem["dataset_spec"]["gen_z"][1]) ##
    ##Must override for now till generation issue is fixed
    gen_num_clusters = target_problem["dataset_spec"]["gen_z"][1]
    gen_num_rows = len(target_problem["zs"])
    cluster_str = "_clusters" + str(gen_num_rows/gen_num_clusters) ## 
    config_str = (col_str + row_str + cluster_str)
    hf.plot_measurement(memoized_infer, "num_clusters", target_problem,save_str="num_clusters_" + config_str + ".png",title_str="num_clusters")
    hf.plot_measurement(memoized_infer, ("ari", target_problem["zs"]), target_problem,save_str="ari_" + config_str + ".png",title_str="ari")
    
#hf.plot_measurement(memoized_infer, "predictive", target_problem)

# with open("pickled_jobs.pkl","wb") as fh:
#     cPickle.dump({"memo":memoized_infer.memo,"ALL_RUN_SPECS":ALL_RUN_SPECS},fh)
