#!python
import os
import hashlib
import argparse
#
import Cloudless.examples.DPMB.settings as S
reload(S)
import Cloudless.examples.DPMB.h5_functions as h5
reload(h5)
import Cloudless.examples.DPMB.s3_helper as s3h
reload(s3h)
import Cloudless.examples.DPMB.MrJob.seed_inferer as si
reload(si)
import Cloudless.examples.DPMB.Tests.create_synthetic_data as csd
reload(csd)


# parse some arguments
parser_description = 'create problem to run on emr'
parser = argparse.ArgumentParser(description=parser_description)
# problem settings
parser.add_argument('--dont_push_to_s3', action='store_true')
parser.add_argument('gen_seed', type=int)
parser.add_argument('num_rows', type=int)
parser.add_argument('num_cols', type=int)
parser.add_argument('num_clusters', type=int)
parser.add_argument('beta_d', type=float)
#
# args = parser.parse_args(['0', '2048', '256', '32', '1.0'])
args = parser.parse_args()

# problem settings
gen_seed = args.gen_seed
num_rows = args.num_rows
num_cols = args.num_cols
num_clusters = args.num_clusters
beta_d = args.beta_d
push_to_s3 = not args.dont_push_to_s3
#
# non passable settings
data_dir = S.data_dir
problem_filename = S.files.problem_filename
h5_filename = h5.get_h5_name_from_pkl_name(problem_filename)
image_save_str = S.files.gen_state_image_save_str
init_filename = S.files.crp_init_filename
run_dir_prefix = S.files.run_dir_prefix
parameters_filename = S.files.parameters_filename
ignore_params = ['dont_push_to_s3']

# helper_functions
get_hexdigest = lambda variable: \
    hashlib.sha224(str(variable)).hexdigest()[:10]
def pop_args(in_dict, ignore_params):
    out_dict = in_dict.copy()
    for ignore_param in ignore_params:
        out_dict.pop(ignore_param)
    return out_dict
dict_to_hexdigest = lambda in_dict: get_hexdigest(in_dict)
create_args = lambda num_iters, num_nodes: [
    '--jobconf', 'mapred.map.tasks=2',
    '--num-iters', str(num_iters),
    '--num-nodes', str(num_nodes),
    '--problem-file', problem_filename,
    '--run_dir', run_dir,
    seed_full_filename,
    ]

parameters = pop_args(vars(args), ignore_params)
# determine some path variables
hex_digest = dict_to_hexdigest(parameters)
run_dir = run_dir_prefix + hex_digest
run_full_dir = os.path.join(data_dir, run_dir)
seed_full_filename = os.path.join(run_full_dir, 'fake_seed_for_gen.txt')
init_full_filename = os.path.join(run_full_dir, init_filename)
parameters_full_filename = os.path.join(run_full_dir, parameters_filename)
#
summary_bucket_dir = S.s3.summary_bucket_dir
run_bucket_dir = os.path.join(summary_bucket_dir, run_dir)    
#
print run_full_dir
try:
    os.makedirs(run_full_dir)
except OSError, ose:
    print ose

# create the problem
problem, problem_filename = csd.pkl_mrjob_problem(
    gen_seed, num_rows, num_cols, num_clusters, beta_d,
    problem_filename=problem_filename,
    image_save_str=image_save_str, dir='.')
# save the initial parameters
with open(parameters_filename, 'w') as fh:
    for key, value in parameters.iteritems():
        line = str(key) + ' = ' + str(value) + '\n'
        fh.write(line)
# now push files to hdfs
for filename in [problem_filename, h5_filename, parameters_filename]:
    hadoop_fs_cmd_1, hadoop_fs_cmd_2 = si.get_hadoop_fs_cmd(
        filename, dest_dir_suffix=run_dir)
    os.system(hadoop_fs_cmd_1)
    os.system(hadoop_fs_cmd_2)
# do I need to push to s3 so that crp init can read?

# gibbs init to be used by all subsequent inference
init_args = ['--init-file', init_filename, '--file', problem_filename, '--file', h5_filename]
init_num_iters = 0
init_num_nodes = 5 # FIXME: this is what you vary to make crp init quicker
init_args.extend(create_args(init_num_iters, init_num_nodes))
os.system('printf "' + str(gen_seed) + '\n" > ' + seed_full_filename)
mr_job = si.MRSeedInferer(args=init_args)
with mr_job.make_runner() as runner:
    runner.run()

# for hadoop
source_full_dir = os.path.join('/user/sgeadmin/', run_dir)
hadoop_fs_cmd = ' '.join(['hadoop', 'fs', '-get', source_full_dir, '.'])
os.system(hadoop_fs_cmd)
run_bucket_dir = os.path.join(summary_bucket_dir, run_dir)
s3 = s3h.S3_helper(bucket_dir=run_bucket_dir, local_dir=run_dir)
for filename in \
        [problem_filename, h5_filename, parameters_filename, init_filename]:
    s3.put_s3(filename)

print run_dir
