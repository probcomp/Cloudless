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
import Cloudless.examples.DPMB.MrJob.seed_inferer as si
reload(si)
import Cloudless.examples.DPMB.Tests.create_synthetic_data as csd
reload(csd)
import Cloudless.examples.DPMB.MrJob.consolidate_summaries as cs
reload(cs)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)


# parse some arguments
parser_description = 'programmatically run mrjob on a synthetic problem'
parser = argparse.ArgumentParser(description=parser_description)
# problem settings
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
args = parser.parse_args(['0', '2048', '256', '32', '1.0', '10', '4'])

gen_seed = args.gen_seed
num_rows = args.num_rows
num_cols = args.num_cols
num_clusters = args.num_clusters
beta_d = args.beta_d
#
# inference settings
num_iters = args.num_iters
num_nodes_list = args.num_nodes_list
#
# non passable settings
base_dir = S.data_dir
seed_filename = 'seed_list.txt'
image_save_str = 'mrjob_problem_gen_state'
gibbs_init_filename = 'gibbs_init.pkl.gz'
data_dir_prefix = 'programmatic_mrjob_'
parameters_filename = 'run_parameters.txt'
reduced_summaries_name = 'reduced_summaries.pkl.gz'
problem_filename = 'problem.pkl.gz'

# determine data dir
get_hexdigest = lambda variable: \
    hashlib.sha224(str(variable)).hexdigest()[:10]
# FIXME: should omit num_iters, num_nodes_list from hexdigest
hex_digest = get_hexdigest(vars(args))
data_dir = data_dir_prefix + hex_digest
data_dir = os.path.join(base_dir, data_dir)
print data_dir
try:
    os.makedirs(data_dir)
except OSError, ose:
    pass

create_args = lambda num_iters, num_nodes: [
    '--jobconf', 'mapred.map.tasks=' + str(num_nodes + 1),
    # may need to specify mapred.map.tasks greater than num_nodes
    '--num-iters', str(num_iters),
    '--num-nodes', str(num_nodes),
    '--problem-file', problem_filename,
    '--data_dir', data_dir,
    seed_full_filename,
    ]

# create the problem and seed file
problem_filename = 'problem.pkl.gz'
try:
    problem = rf.unpickle(problem_filename, dir=data_dir)
    seed_full_filename = os.path.join(data_dir, seed_filename)
except Exception, e:
    # create problem
    problem, problem_filename = csd.pkl_mrjob_problem(
        gen_seed, num_rows, num_cols, num_clusters, beta_d,
        image_save_str=image_save_str, dir=data_dir)
    seed_full_filename = os.path.join(data_dir, seed_filename)
    os.system('printf "0\n" > ' + seed_full_filename)
    #
    # gibbs init
    gibbs_init_args = ['--gibbs-init-file', gibbs_init_filename]
    gibbs_init_args.extend(create_args(0, 1))
    mr_job = si.MRSeedInferer(args=gibbs_init_args)
    # mr_job.init(0,0).next()
    with mr_job.make_runner() as runner:
        runner.run()

num_nodes = num_nodes_list[0]
print 'starting num_nodes = ' + str(num_nodes)
infer_args = ['--resume-file', gibbs_init_filename]
infer_args.extend(create_args(num_iters, num_nodes))
mr_job = si.MRSeedInferer(args=infer_args)

gen_seed_str = str(gen_seed)
init_yielder = mr_job.init(gen_seed_str, gen_seed_str)
run_key, consolidated_data = init_yielder.next()

flatten = lambda data: [y for x in data for y in x]

for iter_idx in range(num_iters):
    # distribute
    distribute_yielder = mr_job.distribute_data(run_key, consolidated_data)
    distribute_out_tuples = [out_tuple for out_tuple in distribute_yielder]
    # infer
    infer_out_list = []
    for distribute_out_tuple in distribute_out_tuples:
        run_key, distribute_state_out = distribute_out_tuple
        infer_step = mr_job.infer(run_key, distribute_state_out)
        run_key, infer_out = infer_step.next()
        infer_out_list.append(infer_out)
    # consolidate
    run_key, consolidated_data = mr_job.consolidate_data(
        run_key, infer_out_list).next()

# save the initial parameters
parameters = vars(args)
parameters_full_filename = os.path.join(data_dir, parameters_filename)
with open(parameters_full_filename, 'w') as fh:
    for key, value in parameters.iteritems():
        line = str(key) + ' = ' + str(value) + '\n'
        fh.write(line)

xlabel = 'time (seconds)'
# summarize the data
summaries_dict, numnodes1_parent_list = cs.read_summaries([data_dir])
title = cs.title_from_parameters(parameters)
cs.plot_summaries(summaries_dict, problem=problem,
                  title=title, xlabel=xlabel, plot_dir=data_dir)
reduced_summaries_dict = cs.extract_reduced_summaries(
    summaries_dict, cs.reduced_summary_extract_func_tuples)
rf.pickle(reduced_summaries_dict, reduced_summaries_name, dir=data_dir)


# flatten = lambda data: [y for x in data for y in x]
# temp_zs = hf.list_of_x_indices_to_zs(summary['list_of_x_indices'])
# temp_xs = problem['xs'][flatten(summary['list_of_x_indices'])]
# num_cols = len(temp_xs[0])
# num_rows = len(temp_xs)
# temp_state = ds.DPMB_State(gen_seed=0, num_cols=num_cols, num_rows=num_rows, init_alpha=master_alpha, init_betas=betas, init_z=temp_zs, init_x=temp_xs)
# numpy.mean(temp_state.score_test_set(problem['test_xs']))

