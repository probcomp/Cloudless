#!python
import os
import argparse
#
import numpy
#
import Cloudless.examples.DPMB.Tests.create_synthetic_data as csd
reload(csd)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.settings as settings
reload(settings)


parser = argparse.ArgumentParser('Create a synthetic MrJob problem')
parser.add_argument('gen_seed',type=int)
parser.add_argument('num_rows',type=int)
parser.add_argument('num_cols',type=int)
parser.add_argument('num_clusters',type=int)
parser.add_argument('--beta_d',default=1.0,type=float)
args,unkown_args = parser.parse_known_args()

# generate some settings
gen_seed = args.gen_seed
num_rows = args.num_rows
num_cols = args.num_cols
num_clusters = args.num_clusters
beta_d = args.beta_d
image_save_str = 'balanced_data'
#
pkl_file = '_'.join([
        'clean_balanced_data',
        'rows', str(num_rows), 
        'cols', str(num_cols),
        'pkl.gz'
        ])

# create the data
data, inverse_permuatation_indices_list = csd.make_clean_data(
    gen_seed=gen_seed,
    num_rows=num_rows,
    num_cols=num_cols,
    num_clusters=num_clusters,
    beta_d=beta_d,
    image_save_str=image_save_str,
    )

all_indices = xrange(num_rows)
random_state = numpy.random.RandomState(gen_seed)
test_fraction = .1
breakpoint = int(num_rows * test_fraction)
random_indices = random_state.permutation(all_indices)
test_indices = random_indices[:breakpoint]
train_indices = random_indices[breakpoint:]
#
test_xs = data[test_indices]
xs = data[train_indices]

# set up pickle variable
pkl_vals = {
    'xs':xs,
    'test_xs':test_xs,
    'num_clusters':num_clusters,
    'beta_d':beta_d,
    'gen_seed':gen_seed,
    }

# actually pickle
pkl_full_file = os.path.join(settings.data_dir, pkl_file)
rf.pickle(pkl_vals, pkl_full_file)
