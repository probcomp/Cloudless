#!python
import os
from collections import namedtuple
import hashlib
import argparse
#
import matplotlib
matplotlib.use('Agg')
#
import Cloudless.examples.DPMB.settings as S
reload(S)
import Cloudless.examples.DPMB.s3_helper as s3h
reload(s3h)
import Cloudless.examples.DPMB.MrJob.seed_inferer as si
reload(si)
import Cloudless.examples.DPMB.Tests.create_synthetic_data as csd
reload(csd)
import Cloudless.examples.DPMB.MrJob.consolidate_summaries as cs
reload(cs)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)


# parse some arguments
parser_description = 'programmatically run mrjob on a synthetic problem'
parser = argparse.ArgumentParser(description=parser_description)
# problem settings
parser.add_argument('--infer_seed', type=int, default=0)
parser.add_argument('--num_iters_per_step', type=int, default=1)
parser.add_argument('--push_to_s3', action='store_true')
parser.add_argument('gen_seed', type=int)
parser.add_argument('num_rows', type=int)
parser.add_argument('num_cols', type=int)
parser.add_argument('num_clusters', type=int)
parser.add_argument('beta_d', type=float)
# inference settings
parser.add_argument('num_iters', type=int)
parser.add_argument('num_nodes_list', nargs='+', type=int)
#
# args = parser.parse_args(['0', '2048', '256', '32', '1.0', '3', '1', '4'])
args = parser.parse_args()

# problem settings
infer_seed = args.infer_seed
num_iters_per_step = args.num_iters_per_step
gen_seed = args.gen_seed
num_rows = args.num_rows
num_cols = args.num_cols
num_clusters = args.num_clusters
beta_d = args.beta_d
push_to_s3 = args.push_to_s3
#
# inference settings
num_iters = args.num_iters
num_nodes_list = args.num_nodes_list
#
# non passable settings
data_dir = S.data_dir
seed_filename = S.files.seed_filename
image_save_str = S.files.gen_state_image_save_str
gibbs_init_filename = S.files.gibbs_init_filename
run_dir_prefix = S.files.run_dir_prefix
parameters_filename = S.files.parameters_filename
reduced_summaries_name = S.files.reduced_summaries_name

# determine data dir
get_hexdigest = lambda variable: \
    hashlib.sha224(str(variable)).hexdigest()[:10]
def pop_runspec_args(in_dict):
    out_dict = in_dict.copy()
    out_dict.pop('num_iters')
    out_dict.pop('num_nodes_list')
    out_dict.pop('infer_seed')
    out_dict.pop('num_iters_per_step')
    return out_dict
args_to_hexdigest = lambda args: get_hexdigest(pop_runspec_args(vars(args)))

# FIXME: should omit num_iters, num_nodes_list from hexdigest
hex_digest = args_to_hexdigest(args)
run_dir = run_dir_prefix + hex_digest
run_full_dir = os.path.join(data_dir, run_dir)
print run_full_dir
try:
    os.makedirs(run_full_dir)
except OSError, ose:
    pass

# create the problem and seed file
problem, problem_filename = csd.pkl_mrjob_problem(
    gen_seed, num_rows, num_cols, num_clusters, beta_d,
    image_save_str=image_save_str, dir=run_full_dir)
s3h.ensure_dir(run_full_dir)
seed_full_filename = os.path.join(run_full_dir, seed_filename)
os.system('printf "' + str(infer_seed) + '\n" > ' + seed_full_filename)

# helper functions
create_args = lambda num_iters, num_nodes: [
    '--jobconf', 'mapred.map.tasks=' + str(num_nodes + 1),
    # may need to specify mapred.map.tasks greater than num_nodes
    '--num-iters', str(num_iters),
    '--num-iters-per-step', str(num_iters_per_step),
    '--num-nodes', str(num_nodes),
    '--problem-file', problem_filename,
    '--run_dir', run_dir,
    seed_full_filename,
    ]

gibbs_init_full_filename = os.path.join(run_full_dir, gibbs_init_filename)
if not os.path.isfile(gibbs_init_full_filename):
    # gibbs init to be used by all subsequent inference
    gibbs_init_args = ['--gibbs-init-file', gibbs_init_filename]
    init_num_iters = 0
    init_num_nodes = 1
    gibbs_init_args.extend(create_args(init_num_iters, init_num_iters))
    mr_job = si.MRSeedInferer(args=gibbs_init_args)
    with mr_job.make_runner() as runner:
        runner.run()
else:
    print '!!!using prior gibbs_init!!!'

# save the initial parameters
parameters = vars(args)
parameters_full_filename = os.path.join(run_full_dir, parameters_filename)
with open(parameters_full_filename, 'w') as fh:
    for key, value in parameters.iteritems():
        line = str(key) + ' = ' + str(value) + '\n'
        fh.write(line)

if push_to_s3:
    summary_bucket_dir = S.s3.summary_bucket_dir
    # FIXME : does data_dir have /tmp prefix?  
    run_bucket_dir = os.path.join(summary_bucket_dir, run_dir)
    s3 = s3h.S3_helper(bucket_dir=run_bucket_dir, local_dir=run_full_dir)
    s3.put_s3(gibbs_init_filename)
    s3.put_s3(parameters_filename)
    s3.put_s3(problem_filename)

# now run for each num_nodes
for num_nodes in num_nodes_list:
    print 'starting num_nodes = ' + str(num_nodes)
    infer_args = ['--resume-file', gibbs_init_filename]
    if push_to_s3:
        infer_args.extend(['--push_to_s3'])
    infer_args.extend(create_args(num_iters, num_nodes))
    mr_job = si.MRSeedInferer(args=infer_args)
    with mr_job.make_runner() as runner:
        runner.run()

# xlabel = 'time (seconds)'
# # summarize the data
# # is seed always zero in the filename?
# init_filename = 'summary_numnodes' + str(num_nodes) + '_seed' + str(infer_seed) + '_iternum-1.pkl.gz'
# summaries_dict, numnodes1_parent_list = cs.read_summaries([data_dir], init_filename=init_filename)
# title = cs.title_from_parameters(parameters)
# cs.plot_summaries(summaries_dict, problem=problem,
#                   title=title, xlabel=xlabel, plot_dir=data_dir)
# reduced_summaries_dict = cs.extract_reduced_summaries(
#     summaries_dict, cs.reduced_summary_extract_func_tuples)
# rf.pickle(reduced_summaries_dict, reduced_summaries_name, dir=data_dir)
