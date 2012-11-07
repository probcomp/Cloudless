import numpy
#
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB.DPMB as dm
reload(dm)


# some settings
data_dir = '/tmp/programmatic_mrjob_b8fa4ff6c6'
summary_base = 'summary_numnodes4_seed0_iternum'
iter_num = 1

# read in problem_data
problem = rf.unpickle('problem.pkl.gz', data_dir)
true_zs = problem['true_zs']
test_xs = problem['test_xs']
init_x = problem['xs']
num_rows = len(init_x)
num_cols = len(init_x[0])

# read in summary data
auto_summary = rf.unpickle(summary_base + str(iter_num) + '.pkl.gz', data_dir)
init_alpha = auto_summary['alpha']
init_betas = auto_summary['betas']
list_of_x_indices = auto_summary['list_of_x_indices']
# post process summary data
zs = numpy.ndarray((num_rows,), dtype=int)
for cluster_idx, cluster_x_indices in enumerate(list_of_x_indices):
    for x_index in cluster_x_indices:
        zs[x_index] = cluster_idx
zs, other = hf.canonicalize_list(zs)
init_z = zs

# generate a state summary
state = ds.DPMB_State(gen_seed=0,
                      num_cols=num_cols,
                      num_rows=num_rows,
                      init_alpha=init_alpha,
                      init_betas=init_betas,
                      init_z=init_z,
                      init_x=init_x,
                      )
transitioner = dm.DPMB(0, state, False, False)
summary = transitioner.extract_state_summary(true_zs=true_zs, test_xs=test_xs)

# compare values
compare_fields = ['ari', 'score', 'test_lls']
print "field, is_equal, auto_value, post_value"
for field in compare_fields:
    auto_value = auto_summary[field]
    post_value = summary[field]
    print field, auto_value == post_value, auto_value, post_value
