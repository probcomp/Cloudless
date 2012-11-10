#!python
import os
import hashlib
import argparse
#
import Cloudless.examples.DPMB.settings as S
reload(S)
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
parser.add_argument('--push_to_s3', action='store_false')
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
push_to_s3 = args.push_to_s3
#
# non passable settings
data_dir = S.data_dir
image_save_str = S.files.gen_state_image_save_str
init_filename = S.files.gibbs_init_filename
run_dir_prefix = S.files.run_dir_prefix
parameters_filename = S.files.parameters_filename

# helper_functions
get_hexdigest = lambda variable: \
    hashlib.sha224(str(variable)).hexdigest()[:10]
def pop_runspec_args(in_dict):
    out_dict = in_dict.copy()
    return out_dict
args_to_hexdigest = lambda args: get_hexdigest(vars(args))
create_args = lambda num_iters, num_nodes: [
    '--jobconf', 'mapred.map.tasks=2',
    '--num-iters', str(num_iters),
    '--num-nodes', str(num_nodes),
    '--problem-file', problem_filename,
    '--run_dir', run_dir,
    seed_full_filename,
    ]

# determine some path variables
hex_digest = args_to_hexdigest(args)
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
    image_save_str=image_save_str, dir=run_full_dir)

# actually create the inference problem init state
os.system('printf "' + str(gen_seed) + '\n" > ' + seed_full_filename)
if not os.path.isfile(init_full_filename):
    # gibbs init to be used by all subsequent inference
    init_args = ['--gibbs-init-file', init_filename]
    init_num_iters = 0
    init_num_nodes = 1
    init_args.extend(create_args(init_num_iters, init_num_iters))
    mr_job = si.MRSeedInferer(args=init_args)
    with mr_job.make_runner() as runner:
        runner.run()
else:
    print '!!!using prior init!!!'

# save the initial parameters
parameters = vars(args)
with open(parameters_full_filename, 'w') as fh:
    for key, value in parameters.iteritems():
        line = str(key) + ' = ' + str(value) + '\n'
        fh.write(line)

if push_to_s3:
    s3 = s3h.S3_helper(bucket_dir=run_bucket_dir, local_dir=run_full_dir)
    s3.put_s3(init_filename)
    s3.put_s3(parameters_filename)
    s3.put_s3(problem_filename)
