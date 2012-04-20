import DPMB_plotutils as dp
reload(dp)
import DPMB_helper_functions as hf
reload(hf)
import DPMB_State as ds
reload(ds)
import numpy as np
import matplotlib.pylab as pylab

ALL_DATASET_SPECS = []

for num_clusters in [2**(j+1) for j in range(4)]:
    dataset_spec = {}
    dataset_spec["gen_seed"] = 0
    dataset_spec["num_cols"] = 16
    dataset_spec["num_rows"] = 32
    dataset_spec["init_alpha"] = 1.0 #FIXME: could make it MLE alpha later
    dataset_spec["init_beta"] = np.repeat(0.01, dataset_spec["num_cols"])
    dataset_spec["init_z"] = ("balanced", num_clusters)
    ALL_DATASET_SPECS.append(dataset_spec)

memoized_gen_problem = Cloudless.memo.AsyncMemoize(...) #FIXME
# generate the actual datasets, by mapping dataset_spec through gen_problem
for dataset_spec in ALL_DATASET_SPEC:
    memoized_gen_problem(**dataset_spec)
    
ALL_RUN_SPECS = []
for dataset_spec in ALL_DATSET_SPECS:
    for inf_seed in range(5):
        xs = memoized_gen_problem(dataset_spec)["xs"]
        # store xs in run_spec. this is the training data that the given run will be run on
        pass
    # FIXME: complete iteration over different possibilities, mirroring the above

# now request the inference
memoized_infer = Cloudless.memo.AsyncMemoize(...) #FIXME
for run_spec in ALL_RUN_SPECS:
    memoized_infer(run_spec)

# now you can interactively call
plot_measurement(memoized_infer, "num_clusters", ALL_DATASET_SPECS[0])
plot_measurement(memoized_infer, ("ari", memoized_gen_problem(ALL_DATASET_SPECS[0])["zs"]), ALL_DATASET_SPECS[0])
                                  
DATASET_SPEC = (GEN_SEED, COLS, ROWS, GEN_ALPHA, GEN_BETA, GEN_Z, GEN_X)
state = ds.DPMB_State(*DATASET_SPEC)
dp.plot_state(state)

# hf.gen_problem(*DATASET_SPEC)
# return {"observables":train_data,"gen_state":gen_state,"test_data":test_data}


# for NUM_CLUSTERS in [8]: ## [2,4,8,16,32]:
#     POINTS_PER_CLUSTER = DATAPOINTS/NUM_CLUSTERS
#     gen_state_with_data = hf.gen_dataset(gen_seed=GEN_SEED, gen_rows=None, gen_cols=COLS, gen_alpha=10
#                                          , gen_beta=GEN_BETA, zDims=np.repeat(POINTS_PER_CLUSTER,NUM_CLUSTERS))
#     state = ds.DPMB_State(None,paramDict={"alpha":1,"betas":np.repeat(1.0,COLS)}
#                           ,dataset={"xs":gen_state_with_data["observables"]},init_method=None
#                           ,infer_alpha={"method":"DISCRETE_GIBBS","low_val":low_val,"high_val":high_val,"n_grid":100}
#                           ,infer_beta={"method":"DISCRETE_GIBBS","low_val":low_val,"high_val":high_val,"n_grid":100})
    
#     state.refresh_counts(gen_state_with_data["gen_state"]["zs"])
#     fh1,fh2,fh3 = dp.plot_state(state=state)
#     fh1.title("NUM_CLUSTERS="+str(NUM_CLUSTERS))

# after you make model.extract_state_summary() contain the state
# sequence for every iteration (remember to copy! otherwise you may
# see overwriting), then write something very similar to the above,
# but using gen_sample instead of gen_dataset, and pulling the state
# out of the summary at iteration 0 (after initialization, but before
# the first gibbs scan).
#
# the goal is to test that all the initialization glue really works.
#
# future: could be interesting to look at after one iteration too.
#
# at the end: running this test script should do the "draw datasets
# according to truth, and also after initialization" and dump the results
# into a directory.
