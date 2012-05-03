import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as pylab
import numpy as np
##
import DPMB_plotutils as dp
reload(dp)
import DPMB_State as ds
reload(ds)
import DPMB as dm
reload(dm)
import DPMB_helper_functions as hf
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

dataset_spec = {}
dataset_spec["gen_seed"] = 0
dataset_spec["num_cols"] = 32
dataset_spec["num_rows"] = 1000
dataset_spec["gen_alpha"] = 1.0 #FIXME: could make it MLE alpha later
dataset_spec["gen_betas"] = np.repeat(0.1, dataset_spec["num_cols"])
dataset_spec["gen_z"] = ("balanced", num_clusters)
dataset_spec["N_test"] = 10

problem = hf.gen_problem(dataset_spec)

run_spec = {}
run_spec["num_iters"] = 20
run_spec["infer_seed"] = 0
run_spec["infer_init_alpha"] = 1.0
run_spec["infer_init_betas"] = np.repeat(0.1, dataset_spec["num_cols"])
run_spec["infer_do_alpha_inference"] = True
run_spec["infer_do_betas_inference"] = True
run_spec["infer_init_z"] = None
run_spec["problem"] = problem
##
run_spec["time_seatbelt"] = 600
run_spec["ari_seatbelt"] = .9


memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], hf.infer, override=True) #FIXME once we've debugged, we can eliminate this override

print "Created memoizer"

for run_spec in ALL_RUN_SPECS:
    memoized_infer(run_spec)

run_spec_filter = None ## lambda x: x["infer_init_z"] is None ## 

hf.try_plots(memoized_infer)
hf.pickle_if_done(memoized_infer,file_str="pickled_jobs.pkl")


##create a dataset_spec->problem->run_spec
##infer 20 with xs returned
##infer 10 with xs returned -> infer 10 with xs returned
##compare all xs
##compare zs of second 10 and 20


# state = ds.DPMB_State(gen_seed=1,num_cols=32,num_rows=1000,init_alpha=1,init_betas=np.repeat(.01,32),init_z=("balanced",100),init_x=None)
# score_before_addition = state.score
# # generate a bunch of new data
# NUM_NEW_VECTORS = 100
# for vector_idx in range(NUM_NEW_VECTORS):
#     state.generate_vector()
# score_after_addition = state.score
# # remove all the new data
# for vector_idx in range(NUM_NEW_VECTORS):
#     state.remove_vector(state.vector_list[-1])
# score_after_removal = state.score
# ##check that a significant change was made to score but it found its way back to where it started
# assert abs(score_before_addition-score_after_addition) > 1 and abs(score_before_addition-score_after_removal) < 1E-6, "DPMB_test.py fails score re-zeroing test 2"

# print "Passed initialize to balanced, add, remove test"
