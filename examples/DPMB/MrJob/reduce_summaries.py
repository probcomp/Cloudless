#!python
import os
import argparse
#
import Cloudless.examples.DPMB.MrJob.consolidate_summaries as cs
reload(cs)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.settings as S
reload(S)


# some settings
init_filename = 'summary_numnodes4_seed0_iternum-1.pkl.gz'
reduced_summaries_name = S.files.reduced_summaries_name

# parse some args
parser = argparse.ArgumentParser('')
parser.add_argument('--top_dir', default='.', type=str)
parser.add_argument('--data_dir_prefix', default='programmatic_', type=str)
args = parser.parse_args()
top_dir = args.top_dir
data_dir_prefix = args.data_dir_prefix

# helper functions
is_data_dir = lambda x: x.startswith(data_dir_prefix)

# determine directories to process
all_dirs = os.listdir(top_dir)
data_dirs = filter(is_data_dir, all_dirs)


# process directories
for data_dir in data_dirs:
    full_data_dir = os.path.join(top_dir, data_dir)
    summaries_dict, numnodes1_parent_list = cs.read_summaries(
        [full_data_dir],
        init_filename=init_filename
        )
    reduced_summaries_dict = cs.extract_reduced_summaries(
        summaries_dict, cs.reduced_summary_extract_func_tuples)
    rf.pickle(reduced_summaries_dict, reduced_summaries_name, dir=full_data_dir)
