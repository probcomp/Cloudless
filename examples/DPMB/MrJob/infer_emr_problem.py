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
# inference settings
parser.add_argument('--infer_seed', type=int, default=0)
parser.add_argument('--num_iters_per_step', type=int, default=1)
parser.add_argument('num_iters', type=int)
parser.add_argument('run_dir', type=str)
parser.add_argument('num_nodes_list', nargs='+', type=int)
#
parser.add_argument('--num-ec2-instances', type=int, default=1)
parser.add_argument('--ec2-instance-type', type=str, default='c1.xlarge')
parser.add_argument('--ec2-master-instance-type', type=str, default='c1.xlarge')
parser.add_argument('--pool-emr-job-flows', action='store_true')
#
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
#
instance_count_helper = lambda instance_count: instance_count + 1 \
    if instance_count != 1 else 1
num_ec2_instances = str(instance_count_helper(args.num_ec2_instances))
ec2_instance_type = args.ec2_instance_type
ec2_master_instance_type = args.ec2_master_instance_type
pool_emr_job_flows = args.pool_emr_job_flows

# non passable settings
data_dir = S.path.data_dir
problem_filename = S.files.problem_filename
seed_filename = S.files.seed_filename
gibbs_init_filename = S.files.gibbs_init_filename
#
s3h.ensure_dir(os.path.join(data_dir, run_dir))
seed_full_filename = os.path.join(data_dir, run_dir, seed_filename)
os.system('printf "' + str(infer_seed) + '\n" > ' + seed_full_filename)

data_dir = S.path.data_dir
create_hadoop_get = lambda run_dir, filename: "".join([
        "/home/hadoop/bin/hadoop fs -get s3://mitpcp-dpmb/tiny_image_summaries/",
        run_dir, "/", filename, " ", data_dir, "/", run_dir, "/"])
here_cmd = 'echo "`date`: pulled down ' + run_dir + '" >> /tmp/steps'
conf_str='printf "runners:\n  local:\n    base_tmp_dir: /mnt/\n" > ~/.mrjob.conf'
hadoop_cmds = [
    'mkdir ' + os.path.join(data_dir, run_dir),
    create_hadoop_get(run_dir, "problem.h5"),
    create_hadoop_get(run_dir, "problem.pkl.gz"),
    here_cmd,
    conf_str,
    ]
hadoop_cmd = '; '.join(hadoop_cmds)

# helper functions
def create_args(num_iters, num_nodes, push_to_s3=True, job_flow_id=None):
    emr_args = ['-r', 'emr', '--verbose', '--jobconf', 'base_tmp_dir=' + data_dir]
    if push_to_s3:
        emr_args.extend(['--push_to_s3'])
    if job_flow_id is not None:
        emr_args.extend([
                '--emr-job-flow-id', job_flow_id,
                '--pool-wait-minutes', '600'
                ])
        emr_args.extend(['--setup-cmd', hadoop_cmd])
    else:
        emr_args.extend(['--enable-emr-debugging'])
        emr_args.extend(['--bootstrap-cmd', hadoop_cmd])
        bootstrap_full_filename = os.path.join(S.path.base_dir, 'bootstrap.sh')
        emr_args.extend(['--bootstrap-action', bootstrap_full_filename])
        emr_args.extend(['--num-ec2-instances', num_ec2_instances])
        emr_args.extend(['--ec2-instance-type', ec2_instance_type])
        emr_args.extend(['--ec2-master-instance-type', ec2_master_instance_type])
    if pool_emr_job_flows:
        emr_args.extend(['--pool-emr-job-flows'])
    #
# check
# http://wiki.apache.org/hadoop/HowManyMapsAndReduces
# for how to control # mappers, # reducers
    other_args = [
        # '--jobconf', 'mapred.task.default.maxvmem=-1',
        # '--jobconf', 'mapred.child.java.opts=-Xmx7G',
        # '--jobconf', 'mapred.map.tasks=' + str(4),
        '--jobconf', 'mapred.reduce.tasks=1',
        # '--jobconf', 'mapred.reduce.max.attempts=1',
        # '--jobconf', 'mapred.map.max.attempts=1',
        '--jobconf', 'mapred.task.timeout=60000000',
        '--jobconf', 'mapred.tasktracker.map.tasks.maximum=4',
        '--jobconf', 'mapred.reduce.tasks.speculative.execution=False',
        '--ami-version', '2.1.1',
        #
        # may need to specify mapred.map.tasks greater than num_nodes
        '--num-iters', str(num_iters),
        '--num-iters-per-step', str(num_iters_per_step),
        '--num-nodes', str(num_nodes),
        '--problem-file', problem_filename,
        # '--resume-file', gibbs_init_filename,
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
    print 'args passed: ' + ' '.join(map(str, infer_args))
    mr_job = si.MRSeedInferer(args=infer_args)
    with mr_job.make_runner() as runner:
        runner.run()


# python_cmds = [
#     "import Cloudless.examples.DPMB.s3_helper as s3h",
#     "my_s3 = s3h.S3_helper(bucket_dir=\"tiny_image_summaries/" + run_dir + "\", local_dir=\"/tmp/" + run_dir + "\")",
#     "my_s3.verify_file(\"problem.h5\")",
#     "my_s3.verify_file(\"problem.pkl.gz\")",
# ]
# python_cmd = '; '.join(python_cmds)
# python_cmd = "python -c '" + python_cmd + "'"
