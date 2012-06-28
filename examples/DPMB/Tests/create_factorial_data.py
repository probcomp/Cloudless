#!python
import argparse
import sys
#
import matplotlib
matplotlib.use('Agg')
import numpy
#
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)


def gen_data(gen_seed,num_clusters,num_cols,num_rows):
    state = ds.DPMB_State(
        gen_seed=gen_seed,
        num_cols=num_cols,
        num_rows=num_rows,
        init_z=('balanced',num_clusters),
        init_betas = numpy.repeat(0.1,num_cols)
    )
    return state.getXValues()

def factorial(gen_seed,num_clusters,num_cols,num_rows,num_splits,
              plot=False,image_save_str=None):
    numpy.random.seed(gen_seed)
    data_list = []
    inverse_permutation_indices_list = []
    for data_idx in xrange(num_splits):
        data_i = gen_data(
            gen_seed=numpy.random.randint(sys.maxint),
            num_clusters=num_clusters,
            num_cols=num_cols/num_splits,
            num_rows=num_rows
            )
        permutation_indices = numpy.random.permutation(xrange(num_rows))
        inverse_permutation_indices = numpy.argsort(permutation_indices)
        inverse_permutation_indices_list.append(inverse_permutation_indices)
        data_list.append(numpy.array(data_i)[permutation_indices])
    data = numpy.hstack(data_list)

    # this is just to visualize, data is already generated
    if image_save_str is not None or plot:
        for state_idx in xrange(num_splits):
            state = ds.DPMB_State(
                gen_seed=numpy.random.randint(sys.maxint),
                num_cols=num_cols,
                num_rows=num_rows,
                init_z=('balanced',num_clusters),
                init_x=data[inverse_permutation_indices_list[state_idx]]
                )
            save_str = None
            if image_save_str is not None:
                save_str = image_save_str + '_' + str(state_idx)
            state.plot(save_str=save_str)
        
    return data,inverse_permutation_indices_list


parser = argparse.ArgumentParser('Create a factorial problem')
parser.add_argument('gen_seed',type=int)
parser.add_argument('num_cols',type=int)
parser.add_argument('num_rows',type=int)
parser.add_argument('num_clusters',type=int)
parser.add_argument('num_splits',type=int)
parser.add_argument('--pkl_file',default='factorial_problem.pkl.gz',type=str)
parser.add_argument('--image_save_str',default=None,type=str)
args,unkown_args = parser.parse_known_args()

data,inverse_permutation_indices = factorial(
    gen_seed=args.gen_seed,
    num_cols=args.num_cols,
    num_rows=args.num_rows,
    num_clusters=args.num_clusters,
    num_splits=args.num_splits,
    image_save_str=args.image_save_str)

rf.pickle(
    {'data':data,'inverse_permutation_indices':inverse_permutation_indices},
    args.pkl_file
    )
