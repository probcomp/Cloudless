import cPickle,matplotlib,sys
matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as pylab
##
if sys.platform == "win32":
    sys.path.append("c:/")
    
import Cloudless
reload(Cloudless)
import Cloudless.memo
reload(Cloudless.memo)
import Cloudless.examples.DPMB_helper_functions as hf
reload(hf)
import numpy as np
import matplotlib.pylab as pylab


ALL_DATASET_SPECS = []

for num_clusters in [4]: ## [2**(j+1) for j in [1,2,3,4,5]]:
    dataset_spec = {}
    dataset_spec["gen_seed"] = 0
    dataset_spec["num_cols"] = 32
    dataset_spec["num_rows"] = 1000
    dataset_spec["gen_alpha"] = 1.0 #FIXME: could make it MLE alpha later
    dataset_spec["gen_betas"] = np.repeat(0.1, dataset_spec["num_cols"])
    dataset_spec["gen_z"] = ("balanced", num_clusters)
    dataset_spec["N_test"] = 10
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
num_iters = 1000
count = 0
for problem in ALL_PROBLEMS:
    for infer_seed in range(1):
        for infer_init_alpha in [1.0]: #note: we're never trying sample-alpha-from-prior-for-init
            for infer_init_betas in [np.repeat(0.1, dataset_spec["num_cols"])]:
                for infer_do_alpha_inference in [True]:
                    for infer_do_betas_inference in [True]:
                        for infer_init_z in [None]:
                            run_spec = {}
                            run_spec["num_iters"] = num_iters
                            run_spec["infer_seed"] = infer_seed
                            run_spec["infer_init_alpha"] = infer_init_alpha
                            run_spec["infer_init_betas"] = infer_init_betas.copy()
                            run_spec["infer_do_alpha_inference"] = infer_do_alpha_inference
                            run_spec["infer_do_betas_inference"] = infer_do_betas_inference
                            run_spec["infer_init_z"] = infer_init_z
                            run_spec["problem"] = problem
                            ##
                            run_spec["time_seatbelt"] = 600
                            run_spec["ari_seatbelt"] = .9
                            ALL_RUN_SPECS.append(run_spec) ## this seems to make the comparison fail copy.deepcopy(run_spec))

print "Generated " + str(len(ALL_RUN_SPECS)) + " run specs!"
# until success:
#ALL_RUN_SPECS = ALL_RUN_SPECS[:1]

print "Running inference on " + str(len(ALL_RUN_SPECS)) + " problems..."

# now request the inference
memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], hf.infer, override=True) #FIXME once we've debugged, we can eliminate this override

print "Created memoizer"

for run_spec in ALL_RUN_SPECS:
    memoized_infer(run_spec)

run_spec_filter = None ## lambda x: x["infer_init_z"] is None ## 

hf.try_plots(memoized_infer)
hf.pickle_if_done(memoized_infer,file_str="pickled_jobs.pkl")
