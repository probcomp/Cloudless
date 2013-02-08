#!/usr/bin/python
import datetime
import os
import hashlib
from collections import namedtuple
#
import numpy
from mrjob.job import MRJob
from mrjob.protocol import PickleProtocol, RawValueProtocol
#
import Cloudless.examples.DPMB.remote_functions as rf
import Cloudless.examples.DPMB.DPMB_State as ds
import Cloudless.examples.DPMB.helper_functions as hf
import Cloudless.examples.DPMB.settings as settings
import Cloudless.examples.DPMB.s3_helper as s3h
import Cloudless.examples.DPMB.DataReader as DataReader

summary_bucket_dir = settings.s3.summary_bucket_dir
problem_bucket_dir = settings.s3.problem_bucket_dir
data_dir = settings.path.data_dir
default_problem_file = settings.files.problem_filename
default_resume_file = None
is_summary_file = lambda filename: \
    filename.startswith('summary') and filename.endswith('.pkl.gz')

def create_pickle_file_str(num_nodes, seed_str, iter_num, hypers_every_N=1):
    file_str = '_'.join([
        'summary',
        'numnodes' + str(num_nodes),
        'seed' + seed_str,
        'he' + str(hypers_every_N), # FIXME: need to make this conform to name filtering everywhere
        'iternum' + str(iter_num) + '.pkl.gz',
        ])
    return file_str

get_hexdigest = lambda variable: hashlib.sha224(str(variable)).hexdigest()
master_state_tuple = namedtuple(
    'master_state_tuple',
    ' master_suffstats list_of_list_of_x_indices last_valid_zs '
    ' master_alpha betas master_inf_seed iter_num '
    )
child_state_tuple = namedtuple(
    'child_state_tuple',
    ' child_suffstats list_of_x_indices x_indices zs '
    ' master_alpha betas master_inf_seed iter_num '
    ' child_inf_seed child_gen_seed child_counter '
    ' iter_start_dt '
    )

def run_hadoop_fs_cmd(filename, dest_dir_prefix='/user/sgeadmin/', dest_dir_suffix='', which='put'):
    dest_full_dir = os.path.join(dest_dir_prefix, dest_dir_suffix)
    dest_full_file = os.path.join(dest_full_dir, filename)
    which = '-' + which
    if which == '-put':
        hadoop_fs_cmd_1 = ' '.join(['hadoop', 'fs', '-mkdir', dest_full_dir])
        hadoop_fs_cmd_2 = ' '.join(['hadoop', 'fs', which, filename,
                                    dest_full_file])
        os.system(hadoop_fs_cmd_1)
        os.system(hadoop_fs_cmd_2)
    elif which == '-get':
        os.system('rm ' + filename)
        hadoop_fs_cmd_1 = ' '.join(['hadoop', 'fs', which, dest_full_file,
                                    filename])
        os.system(hadoop_fs_cmd_1)
    else:
        pass

class MRSeedInferer(MRJob):

    INPUT_PROTOCOL = RawValueProtocol
    INTERNAL_PROTOCOL = PickleProtocol
    OUTPUT_PROTOCOL = PickleProtocol

    def configure_options(self):
        super(MRSeedInferer, self).configure_options()
        self.add_passthrough_option('--num-iters-per-step', type='int',
                                    default=1)
        self.add_passthrough_option('--num-iters', type='int', default=8)
        self.add_passthrough_option('--num-nodes', type='int', default=4)
        self.add_passthrough_option('--time-seatbelt', type='int', default=None)
        self.add_passthrough_option('--push_to_s3', action='store_true')
        self.add_passthrough_option('--run_dir', type='str', default='')
        self.add_passthrough_option('--problem-file', type='str',
                                    default=default_problem_file)
        self.add_passthrough_option('--resume-file',type='str', default=None)
        self.add_passthrough_option('--init-file',type='str', default=None)
            

    def load_options(self, args):
        super(MRSeedInferer, self).load_options(args=args)
        self.num_iters = self.options.num_iters
        self.num_nodes = self.options.num_nodes
        self.num_iters_per_step = self.options.num_iters_per_step
        self.hypers_every_N = self.options.num_iters_per_step
        self.num_steps = self.num_iters
        # time_seatbelt only works on single node inference
        self.time_seatbelt = self.options.time_seatbelt
        self.push_to_s3 = self.options.push_to_s3
        self.run_dir = self.options.run_dir
        self.problem_file = self.options.problem_file
        self.resume_file = self.options.resume_file
        self.init_file = self.options.init_file
        if self.num_nodes == 1 and self.num_iters !=0:
            # override values
            self.num_steps = 1
            self.num_iters_per_step = self.num_iters
        self.run_bucket_dir = os.path.join(summary_bucket_dir, self.run_dir)

    def init(self, key, run_key):
        start_dt = datetime.datetime.now()
        master_inf_seed = int(run_key)
        num_nodes = self.num_nodes
        run_dir = self.run_dir
        problem_file = self.problem_file
        resume_file = self.resume_file
        init_file = self.init_file
        num_iters_per_step = self.num_iters_per_step
        hypers_every_N = self.hypers_every_N
        run_bucket_dir = self.run_bucket_dir

        problem_hexdigest = None
        with hf.Timer('init/resume') as init_resume_timer:
            if resume_file:
                summary = rf.unpickle(resume_file)
                if len(summary['lolo_x_indices']) != num_nodes:
                    lolo_x_indices = summary['lolo_x_indices']
                    alpha = summary['alpha']
                    lolo_x_indices, master_inf_seed = hf.reshape_superclusters(
                        lol_x_indices, num_nodes, alpha, master_inf_seed)
                    summary['lolo_x_indices'] = lolo_x_indices
            else:
                run_spec = rf.gen_default_cifar_run_spec(
                    problem_file=problem_file,
                    infer_seed=master_inf_seed,
                    num_iters=0 # no inference, just init
                   )
                run_spec['dataset_spec']['data_dir'] = '.'
                problem = rf.gen_problem(run_spec['dataset_spec'])
                # gen dummy state to get alphas, betas
                dummy_state = ds.DPMB_State(
                    gen_seed=run_spec['dataset_spec']['gen_seed'],
                    num_cols=run_spec['dataset_spec']['num_cols'], num_rows=10)
                # MLE alpha
                logp_list, lnPdf, grid = hf.calc_alpha_conditional(dummy_state)
                init_alpha = grid[numpy.argmax(logp_list)]
                #
                init_betas = dummy_state.betas
                n_draws = len(problem['xs'])
                mus = numpy.repeat(1./num_nodes, num_nodes)
                lolo_x_indices, random_state = \
                    hf.crp_init_superclusters_experimental(
                    init_alpha, mus, dummy_state.random_state, n_draws
                    )
                flat_lolo_x_indices = hf.flatten(lolo_x_indices)
                num_clusters = len(flat_lolo_x_indices)
                cluster_counts = map(len, flat_lolo_x_indices)
                flat_x_indices, zs = rf.list_of_x_indices_to_xs_and_zs(
                    flat_lolo_x_indices)
                timing = {
                    'alpha':0, 'betas':0, 'zs':0,'run_sum':0,
                    'timestamp':datetime.datetime.now(),
                    'iter_start_dt':datetime.datetime.now(),
                    'iter_end_dt':datetime.datetime.now(),
                    }
                # supercluster_CRP_INIT HERE
                # gen suffstats from supercluster here
                summary = {
                    'alpha':init_alpha,
                    'betas':init_betas,
                    'num_clusters':num_clusters,
                    'cluster_counts':cluster_counts,
                    'timing':timing,
                    'inf_seed':master_inf_seed,
                    'suffstats':None, # do I need to build this here?
                    'lolo_x_indices':lolo_x_indices,
                    'init_num_nodes':num_nodes,
                    }
                summary['last_valid_zs'] = zs
                summary['list_of_x_indices'] = flat_lolo_x_indices
                #
                problem_hexdigest = get_hexdigest(problem)
                # FIXME : infer will pickle over this
                if init_file is None:
                    init_file = create_pickle_file_str(
                        num_nodes, run_key, str(-1), hypers_every_N)
                # FIXME: should only pickle if it wasn't read
                rf.pickle(summary, init_file, dir='')
                run_bucket_dir = os.path.join(summary_bucket_dir, run_dir)
                s3 = s3h.S3_helper(bucket_dir=run_bucket_dir, local_dir='.')
                s3.put_s3(init_file, do_print=False)

        summary['problem_hexdigest'] = problem_hexdigest
        summary['timing'].update({
            'start_time':start_dt,
            'infer_problem_delta_t':init_resume_timer.elapsed_secs,
            })
        #
        #
        # pull out the values to pass on
        lolo_x_indices = summary.get('lolo_x_indices', None)
        last_valid_zs = summary.get('last_valid_zs', summary.get('zs', None))
        master_alpha = summary['alpha']
        betas = summary['betas']
        master_suffstats = None
        iter_num = summary.get('iter_num', 0)
        master_state = master_state_tuple(
            master_suffstats, lolo_x_indices, last_valid_zs,
            master_alpha, betas,
            master_inf_seed, iter_num,
            )
        yield run_key, master_state

    def distribute_data(self, run_key, master_state):
        iter_start_dt = datetime.datetime.now()
        num_nodes = self.num_nodes
        # pull variables out of master_state
        lolo_x_indices = master_state.list_of_list_of_x_indices
        master_alpha = master_state.master_alpha
        betas = master_state.betas
        master_inf_seed = master_state.master_inf_seed
        iter_num = master_state.iter_num
        #
        mus = numpy.repeat(1./num_nodes, num_nodes)
        random_state = hf.generate_random_state(master_inf_seed)
        # GIBBS SAMPLE SUPERCLUSTERS WITH DIRICHLET MULTINOMIAL
        lolo_x_indices, random_state = hf.gibbs_on_superclusters(
            lolo_x_indices, mus, master_alpha, random_state)
        #
        # generate child state info
        gen_seed_list = map(int, random_state.tomaxint(num_nodes))
        inf_seed_list = map(int, random_state.tomaxint(num_nodes))
        node_info_tuples = zip(gen_seed_list, inf_seed_list, lolo_x_indices)
        #
        # actually distribute
        for child_counter, node_info_tuple in enumerate(node_info_tuples):
            (child_gen_seed, child_inf_seed, child_list_of_x_indices) = \
                node_info_tuple
            xs, zs = rf.list_of_x_indices_to_xs_and_zs(child_list_of_x_indices)
            #
            # child inference builds own state, don't need suffstats, just zs
            child_suffstats_out = None
            child_state = child_state_tuple(
                child_suffstats_out, child_list_of_x_indices, xs, zs,
                master_alpha, betas, random_state, iter_num,
                child_inf_seed, child_gen_seed, child_counter,
                iter_start_dt)
            yield run_key, child_state

    def fake_infer(self, run_key, child_state_in):
        hf.echo_date('enter fake infer')
        hf.echo_date('exit fake infer')
        yield run_key, child_state_in
        
    def infer(self, run_key, child_state_in):
        hf.echo_date('enter infer')
        num_nodes = self.num_nodes
        run_dir = self.run_dir
        problem_file = self.problem_file
        num_iters_per_step = self.num_iters_per_step
        run_bucket_dir = self.run_bucket_dir
        #
        x_indices = child_state_in.x_indices
        zs = child_state_in.zs
        master_alpha = child_state_in.master_alpha
        betas = child_state_in.betas
        master_inf_seed = child_state_in.master_inf_seed
        iter_num = child_state_in.iter_num
        child_inf_seed = child_state_in.child_inf_seed
        child_gen_seed = child_state_in.child_gen_seed
        child_counter = child_state_in.child_counter
        iter_start_dt = child_state_in.iter_start_dt
        #
        if len(x_indices) == 0:
            hf.echo_date('!!!empty child!!!')
            new_iter_num = iter_num + 1
            last_valid_zs = []
            x_indices = []
            list_of_x_indices = []
            child_suffstats_out = ds.suffstats_tuple(0,0,[],[])
            child_state_out = child_state_tuple(
                child_suffstats_out, list_of_x_indices, x_indices, last_valid_zs,
                master_alpha, betas, master_inf_seed, new_iter_num,
                None, None, None, iter_start_dt
                )
            hf.echo_date('exit infer')
            yield run_key, child_state_out
        else:
            run_spec = rf.run_spec_from_child_state_info(
                zs, master_alpha, betas, child_inf_seed, child_gen_seed,
                num_iters_per_step, num_nodes)

            # sub_problem_xs = rf.get_xs_subset_from_h5(
            #     problem_file, x_indices, dir='.')
            sub_problem_xs = DataReader.DataReader(problem_file, x_indices,
                                                   dir='.')
            hf.echo_date('infer(): read problem')
            sub_problem = {'xs':sub_problem_xs, 'zs':zs, 'test_xs':None}
            # actually infer
            get_child_pkl_file = lambda child_iter_num: create_pickle_file_str(
                num_nodes, run_key+'_child'+str(child_counter), child_iter_num)
            child_summaries = None
            if num_nodes == 1:
                sub_problem['true_zs'] = None
                sub_problem['test_xs'] = None
                run_spec['infer_do_alpha_inference'] = True
                run_spec['infer_do_betas_inference'] = True
                # set up for intermediate results to be pickled on the fly
                def single_node_post_infer_func(iter_idx, state, last_summary,
                                                data_dir=''):
                    iter_num = iter_idx + 1
                    pkl_file = get_child_pkl_file(iter_num)
                    rf.pickle(last_summary, pkl_file, dir='')
                    run_bucket_dir = os.path.join(summary_bucket_dir, run_dir)
                    s3 = s3h.S3_helper(bucket_dir=run_bucket_dir, local_dir='.')
                    s3.put_s3(pkl_file, do_print=False)

                child_summaries = rf.infer(
                    run_spec, sub_problem, send_zs=True,
                    post_infer_func=single_node_post_infer_func)
            else:
                hf.echo_date('infer(): entering rf.infer()')
                child_summaries = rf.infer(run_spec, sub_problem)

            hf.echo_date('infer(): done rf.infer')
            last_valid_zs = child_summaries[-1]['last_valid_zs']
            last_valid_list_of_x_indices = \
                child_summaries[-1]['last_valid_list_of_x_indices']
            list_of_x_indices = child_state_in.list_of_x_indices
            list_of_x_indices = [
                [x_indices[idx] for idx in cluster_indices]
                for cluster_indices in last_valid_list_of_x_indices
                ]
            child_suffstats_out = child_summaries[-1]['suffstats']
            new_iter_num = iter_num + 1
            child_state_out = child_state_tuple(
                child_suffstats_out, list_of_x_indices, x_indices, last_valid_zs,
                master_alpha, betas, master_inf_seed, new_iter_num,
                None, None, None, iter_start_dt
                )
            hf.echo_date('exit infer')
            yield run_key, child_state_out

    def consolidate_data(self, run_key, child_infer_output_generator):
        hf.echo_date('enter consolidate')
        start_dt = datetime.datetime.now()
        num_nodes = self.num_nodes
        run_dir = self.run_dir
        problem_file = self.problem_file
        num_iters_per_step = self.num_iters_per_step
        hypers_every_N = self.hypers_every_N
        run_bucket_dir = self.run_bucket_dir

        zs_list = []
        list_of_list_of_x_indices = []
        child_suffstats_list = []
        #
        # consolidate data from child states
        child_state_counter = 0
        for child_state_out in child_infer_output_generator:
            child_suffstats_list.append(child_state_out.child_suffstats)
            zs_list.append(child_state_out.zs)
            list_of_list_of_x_indices.append(child_state_out.list_of_x_indices)
            child_state_counter += 1
        hf.echo_date('received ' + str(child_state_counter) + ' child states')

        # all master_alpha, betas, master_inf_seed are all the same, use last
        master_alpha = child_state_out.master_alpha
        betas = child_state_out.betas
        master_inf_seed = child_state_out.master_inf_seed
        iter_num = child_state_out.iter_num
        iter_start_dt = child_state_out.iter_start_dt
        master_suffstats = hf.consolidate_suffstats(child_suffstats_list)

        dummy_state = ds.DPMB_State(0, 10, 10)
        alpha_grid = dummy_state.get_alpha_grid()
        beta_grid = dummy_state.get_beta_grid()
        random_state = hf.generate_random_state(master_inf_seed)
        # sample alpha
        
        with hf.Timer('alpha_inference') as alpha_inference_timer:
            alpha_draw, random_state = hf.sample_alpha_suffstats(
                master_suffstats, alpha_grid, random_state)
        # sample betas
        with hf.Timer('beta_inference') as beta_inference_timer:
            beta_draws, random_state = hf.sample_betas_suffstats(
                master_suffstats, beta_grid, random_state)
        hf.echo_date('done transitioning')

        # FIXME : NEED TO CREATE PSUEDO SUMMARY HERE
        # FIXME : NEED TO CREATE PSUEDO SUMMARY HERE
        # FIXME : NEED TO CREATE PSUEDO SUMMARY HERE
        
        # extract summary
        summary = {
            'alpha':alpha_draw,
            'betas':numpy.array(beta_draws),
            'num_clusters':len(master_suffstats.list_of_cluster_suffstats),
            'cluster_counts':[cs.num_vectors for cs in
                              master_suffstats.list_of_cluster_suffstats],
            'timing':{
                'alpha':alpha_inference_timer.elapsed_secs,
                'betas':beta_inference_timer.elapsed_secs,
                'zs':0,'run_sum':0,
                },
            'inf_seed':random_state,
            'suffstats':master_suffstats,
            }
        # FIXME : need to flatten list_of_list_of_x_indices
        flat_lolo_x_indices = hf.flatten(list_of_list_of_x_indices)
        num_clusters = len(flat_lolo_x_indices)
        cluster_counts = map(len, flat_lolo_x_indices)
        flat_x_indices, zs = rf.list_of_x_indices_to_xs_and_zs(
            flat_lolo_x_indices)

        summary['last_valid_zs'] = zs
        summary['list_of_x_indices'] = flat_lolo_x_indices
        summary['lolo_x_indices'] = list_of_list_of_x_indices
        summary['timing']['timestamp'] = datetime.datetime.now()
        summary['iter_num'] = iter_num
        iter_end_dt = datetime.datetime.now()
        summary['timing']['iter_start_dt'] = iter_start_dt
        summary['timing']['iter_end_dt'] = iter_end_dt
        #
        # save pkl'ed summary locally, push to s3 if appropriate
        pkl_file = create_pickle_file_str(num_nodes, run_key, iter_num,
                                          hypers_every_N=hypers_every_N)
        master_suffstats = summary.pop('suffstats')
        rf.pickle(summary, pkl_file, dir='')
        run_bucket_dir = os.path.join(summary_bucket_dir, run_dir)
        s3 = s3h.S3_helper(bucket_dir=run_bucket_dir, local_dir='.')
        s3.put_s3(pkl_file, do_print=False)
        hf.echo_date('done pickling summary')
        #
        # format summary to pass out 
        last_valid_zs = summary['last_valid_zs']
        master_alpha = summary['alpha']
        betas = summary['betas']
        master_inf_seed = random_state
        master_suffstats = None
        master_state = master_state_tuple(
            master_suffstats, list_of_list_of_x_indices, last_valid_zs, master_alpha, betas,
            master_inf_seed=master_inf_seed, iter_num=iter_num)
        hf.echo_date('exit consolidate')
        yield run_key, master_state

    def push_to_s3_step(self, key, value):
        run_dir = self.run_dir
        #
        source_full_dir = os.path.join('/user/sgeadmin/', run_dir)
        # FIXME: make sure run_dir doesn't exist, else hadoop fs -get fails
        os.system(' '.join(['rm -rf', run_dir]))
        hadoop_fs_cmd = ' '.join(['hadoop', 'fs', '-get', source_full_dir, '.'])
        os.system(hadoop_fs_cmd)
        run_bucket_dir = os.path.join(summary_bucket_dir, run_dir)
        s3 = s3h.S3_helper(bucket_dir=run_bucket_dir, local_dir=run_dir)
        filename_list = os.listdir(run_dir)
        filename_list = filter(is_summary_file, filename_list)
        for filename in filename_list:
            s3.put_s3(filename, do_print=False)
        yield key, value

    def steps(self):
        step_list = [self.mr(self.init)]
        infer_step = [
            self.mr(self.distribute_data),
            self.mr(self.infer, self.consolidate_data),
            ]
        step_list.extend(infer_step * self.num_steps)
        # if self.push_to_s3:
        #     step_list.append(self.mr(self.push_to_s3_step))
        return step_list

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--problem-file', type=str, default=default_problem_file)
    parser.add_argument('--resume-file',type=str, default=None)
    parser.add_argument('--push_to_s3', action='store_true')
    args, unknown_args = parser.parse_known_args()
    run_dir = args.run_dir
    problem_filename = args.problem_file
    resume_file = args.resume_file
    push_to_s3 = args.push_to_s3
    #
    if resume_file is not None:
        rf.verify_file_helper(resume_file, run_dir, local_dir='')
    if run_dir is not None:
        rf.verify_problem_helper(run_dir, local_dir='',
                                 problem_filename=problem_filename)
    #
    MRSeedInferer.run()
