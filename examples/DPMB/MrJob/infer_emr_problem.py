#!python
import os
import argparse
#
import Cloudless.examples.DPMB.settings as S
reload(S)
import Cloudless.examples.DPMB.s3_helper as s3h
reload(s3h)
import Cloudless.examples.DPMB.MrJob.seed_inferer as si
reload(si)


# parse some arguments
parser_description = 'programmatically run mrjob on a synthetic problem'
parser = argparse.ArgumentParser(description=parser_description)
# emr settings
parser.add_argument('--dont_push_to_s3', action='store_true')
parser.add_argument('--job_flow_id', type=str, default=None)
# FIXME: add EMR option, default true
# inference settings
parser.add_argument('--infer_seed', type=int, default=0)
parser.add_argument('--num_iters_per_step', type=int, default=1)
parser.add_argument('num_iters', type=int)
parser.add_argument('run_dir', type=str)
parser.add_argument('num_nodes_list', nargs='+', type=int)
#
# args = parser.parse_args(['--infer_seed', '0', '--num_iters_per_step', '2', '3', 'programmatic_mrjob_74ffa9b56d', '2'])
args = parser.parse_args()

# emr settings
push_to_s3 = not args.dont_push_to_s3
job_flow_id = args.job_flow_id
# inference settings
infer_seed = args.infer_seed
num_iters = args.num_iters
run_dir = args.run_dir
num_iters_per_step = args.num_iters_per_step
num_nodes_list = args.num_nodes_list

# non passable settings
data_dir = S.path.data_dir
problem_filename = S.files.problem_filename
seed_filename = S.files.seed_filename
gibbs_init_filename = S.files.gibbs_init_filename
#
seed_full_filename = os.path.join(data_dir, run_dir, seed_filename)
os.system('printf "' + str(infer_seed) + '\n" > ' + seed_full_filename)

# helper functions
def create_args(num_iters, num_nodes, push_to_s3=True, job_flow_id=None):
    emr_args = [
        '-r', 'emr', '--pool-wait-minutes', '30'
        ]
    if push_to_s3:
        emr_args.extend(['--push_to_s3'])
    if job_flow_id != None:
        emr_args.extend(['--emr-job-flow-id', job_flow_id])
    #
    other_args = [
        '--jobconf', 'mapred.map.tasks=' + str(num_nodes + 1),
        # may need to specify mapred.map.tasks greater than num_nodes
        '--num-iters', str(num_iters),
        '--num-iters-per-step', str(num_iters_per_step),
        '--num-nodes', str(num_nodes),
        '--problem-file', problem_filename,
        '--resume-file', gibbs_init_filename,
        '--run_dir', run_dir,
        seed_full_filename,
        ]
    arg_list = []
    arg_list.extend(emr_args)
    arg_list.extend(other_args)
    return arg_list

# now run for each num_nodes
for num_nodes in num_nodes_list:
    print 'starting num_nodes = ' + str(num_nodes)
    infer_args = create_args(num_iters, num_nodes, push_to_s3, job_flow_id)
    mr_job = si.MRSeedInferer(args=infer_args)
    with mr_job.make_runner() as runner:
        runner.run()
