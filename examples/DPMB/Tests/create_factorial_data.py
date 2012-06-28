#!python
import sys
#
import numpy
#
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)


def gen_data(gen_seed,num_clusters,num_cols,num_rows):
    state = ds.DPMB_State(
        gen_seed=gen_seed,
        num_cols=num_cols,
        num_rows=num_rows,
        init_z=('balanced',num_clusters),
        init_betas = np.repeat(0.1,num_cols)
    )
    return state.getXValues()

def factorial(gen_seed,num_clusters,num_cols,num_rows,n_splits):
    numpy.random.seed(gen_seed)
    data_list = []
    inverse_permutation_indices_list = []
    for data_idx in xrange(n_splits):
        data_i = gen_data(
            gen_seed=numpy.random.randint(sys.maxint),
            num_clusters=num_clusters,
            num_cols=num_cols/n_splits,
            num_rows=num_rows
            )
        permutation_indices = numpy.random.permutation(xrange(num_rows))
        inverse_permutation_indices = argsort(permutation_indices)
        inverse_permutation_indices_list.append(inverse_permutation_indices)
        data_list.append(numpy.array(data_i)[permutation_indices])
    data = numpy.hstack(data_list)

    # this is just to visualize, data is already generated
    for state_idx in xrange(n_splits):
        state = ds.DPMB_State(
            gen_seed=numpy.random.randint(sys.maxint),
            num_cols=num_cols,
            num_rows=num_rows,
            init_z=('balanced',num_clusters),
            init_x=data[inverse_permutation_indices_list[state_idx]]
            )
        state.plot()
        title(state_idx)
        
    return data,inverse_permutation_indices_list

num_rows = 32
num_cols = 32
gen_seed = 0
num_clusters = 4

data,permutation_indices = factorial(gen_seed,num_clusters,num_cols,num_rows,4)
