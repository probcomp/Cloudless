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
import Cloudless.examples.DPMB.pyx_functions as pf
import Cloudless.examples.DPMB.DPMB_State as ds
import Cloudless.examples.DPMB.DPMB as dm
import Cloudless.examples.DPMB.h5_functions as h5
import Cloudless.examples.DPMB.helper_functions as hf
import Cloudless.examples.DPMB.s3_helper as s3h
import Cloudless.examples.DPMB.settings as settings


summary_bucket_dir = settings.s3.summary_bucket_dir
problem_bucket_dir = settings.s3.problem_bucket_dir
data_dir = settings.path.data_dir
#
# problem_file = settings.tiny_image_problem_file
# problem_file = 'tiny_image_problem_nImages_320000_nPcaTrain_10000.pkl.gz'
default_problem_file = settings.files.problem_filename
default_resume_file = None
postpone_scoring = True

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
        self.add_passthrough_option('--gibbs-init-file',type='str', default=None)
            

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
        self.gibbs_init_file = self.options.gibbs_init_file
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
        gibbs_init_file = self.gibbs_init_file
        num_iters_per_step = self.num_iters_per_step
        hypers_every_N = self.hypers_every_N
        run_bucket_dir = self.run_bucket_dir

        run_full_dir = os.path.join(data_dir, run_dir)
        s3h.verify_problem_local(run_dir, problem_file)
        #
        # gibbs init or resume 
        problem_hexdigest = None
        with hf.Timer('init/resume') as init_resume_timer:
            if resume_file:
                summary = sh3.verify_file_helper(resume_file, run_dir,
                                                 unpickle=True)
            else:
                run_spec = rf.gen_default_cifar_run_spec(
                    problem_file=problem_file,
                    infer_seed=master_inf_seed,
                    num_iters=0 # no inference, just init
                   )
                # FIXME: should I pass permute=False here?
                run_spec['dataset_spec']['data_dir'] = run_full_dir
                problem = rf.gen_problem(run_spec['dataset_spec'])
                init_save_str = 'gibbs_init_state'
                init_save_str = os.path.join(run_full_dir, init_save_str)
                summary = rf.infer(
                    run_spec, problem, init_save_str=init_save_str)[-1]
                problem_hexdigest = get_hexdigest(problem)
                # FIXME : infer will pickle over this
                if gibbs_init_file is None:
                    gibbs_init_file = create_pickle_file_str(
                        num_nodes, run_key, str(-1), hypers_every_N)
                # FIXME: should only pickle if it wasn't read
                rf.pickle(summary, gibbs_init_file, dir=run_full_dir)
                if self.push_to_s3:
                    s3 = s3h.S3_helper(
                        bucket_dir=run_bucket_dir, local_dir=run_full_dir)
                    s3.put_s3(gibbs_init_file)

        summary['problem_hexdigest'] = problem_hexdigest
        summary['timing'].update({
            'start_time':start_dt,
            'infer_problem_delta_t':init_resume_timer.elapsed_secs,
            })
        #
        #
        # pull out the values to pass on
        list_of_x_indices = summary.get('list_of_x_indices', None)
        list_of_x_indices = summary.get('last_valid_list_of_x_indices',
                                        list_of_x_indices)
        last_valid_zs = summary.get('last_valid_zs', summary.get('zs', None))
        master_alpha = summary['alpha']
        betas = summary['betas']
        master_suffstats = summary['suffstats']
        iter_num = summary.get('iter_num', 0)
        master_state = master_state_tuple(
            master_suffstats, list_of_list_of_x_indices, last_valid_zs,
            master_alpha, betas,
            master_inf_seed, iter_num,
            )
        yield run_key, master_state

    def distribute_data(self, run_key, master_state):
        iter_start_dt = datetime.datetime.now()
        num_nodes = self.num_nodes
        # pull variables out of master_state
        lol_of_x_indices = master_state.list_of_list_of_x_indices
        master_alpha = master_state.master_alpha
        betas = master_state.betas
        master_inf_seed = master_state.master_inf_seed
        iter_num = master_state.iter_num
        #
        mus = numpy.repeat(1./num_nodes, num_nodes)
        master_inf_seed = hf.generate_random_state(master_inf_seed)
        # GIBBS SAMPLE SUPERCLUSTERS WITH DIRICHLET MULTINOMIAL
        lol_of_x_indices, master_inf_seed = hf.gibbs_on_superclusters(
            lol_of_x_indices, mus, master_alpha, master_inf_seed)
        #
        # generate child state info
        gen_seed_list = map(int, random_state.tomaxint(num_nodes))
        inf_seed_list = map(int, random_state.tomaxint(num_nodes))
        node_info_tuples = zip(gen_seed_list, inf_seed_list, lol_of_x_indices)
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
                master_alpha, betas, master_inf_seed, iter_num,
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
        run_spec = rf.run_spec_from_child_state_info(
            zs, master_alpha, betas, child_inf_seed, child_gen_seed,
            num_iters_per_step, num_nodes)

        run_full_dir = os.path.join(data_dir, run_dir)
        s3h.verify_problem_local(run_dir, problem_file)
        sub_problem_xs = rf.get_xs_subset_from_h5(
            problem_file, x_indices, dir=run_full_dir)
        hf.echo_date('infer(): read problem')
        sub_problem = {'xs':sub_problem_xs, 'zs':zs, 'test_xs':None}
        # actually infer
        get_child_pkl_file = lambda child_iter_num: create_pickle_file_str(
            num_nodes, run_key+'_child'+str(child_counter), child_iter_num)
        child_summaries = None
        if num_nodes == 1:
            true_zs, test_xs = None, None
            if not postpone_scoring:
                orig_problem = rf.unpickle(
                    problem_file, dir=run_full_dir, check_hdf5=False)
                true_zs = orig_problem.get('true_zs', None)
                if true_zs is not None:
                    true_zs = [true_zs[x_index] for x_index in x_indices]
                test_xs = numpy.array(orig_problem['test_xs'], dtype=numpy.int32)
            sub_problem['true_zs'] = true_zs
            sub_problem['test_xs'] = test_xs
            run_spec['infer_do_alpha_inference'] = True
            run_spec['infer_do_betas_inference'] = True
            #
            # set up for intermediate results to be pickled on the fly
            def single_node_post_infer_func(iter_idx, state, last_summary,
                                            data_dir=run_full_dir):
                iter_num = iter_idx + 1
                pkl_file = get_child_pkl_file(iter_num)
                rf.pickle(last_summary, pkl_file, dir=run_full_dir)
                if self.push_to_s3:
                    s3 = s3h.S3_helper(bucket_dir=run_bucket_dir,
                                       local_dir=run_full_dir)
                    s3.put_s3(pkl_file)
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
        x_indices_list = []
        list_of_list_of_x_indices = []
        child_suffstats_list = []
        #
        # consolidate data from child states
        child_state_counter = 0
        for child_state_out in child_infer_output_generator:
            child_suffstats_list.append(child_state_out.suffstats)
            zs_list.append(child_state_out.zs)
            x_indices_list.append(child_state_out.x_indices)
            list_of_list_of_x_indices.extend(child_state_out.list_of_x_indices)
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
        # sample alpha
        alpha_logps, alpha_lnPdf, alpha_grid = \
            hf.calc_alpha_conditional_suffstats(master_suffstats, alpha_grid)
        master_inf_seed = hf.generate_random_state(master_inf_seed)
        alpha_randv = master_inf_seed.uniform()
        alpha_draw_idx = pf.renormalize_and_sample(alpha_logps, alpha_randv)
        alpha_draw = alpha_grid[alpha_draw_idx] #!!!!
        # sample betas
        beta_draws = []
        for col_idx in range(len(betas)):
            SR_array = numpy.array([
                (cs.column_sums[col_idx], cs.num_vectors-cs.column_sums[col_idx])
                for cs in master_suffstats
                ])
            S_array = SR_array[:, 0]
            R_array = SR_array[:, 1]
            beta_logps = pf.calc_beta_conditional_suffstat_helper(
                S_array, R_array, beta_grid)
            beta_randv = master_inf_seed.uniform()
            beta_draw_dix = pf.renormalize_and_sample(beta_logs, beta_randv)
            beta_draw = beta_grid[beta_draw_idx]
            beta_draws.append(beta_draw)
        
        hf.echo_date('done transitioning')

        # extract summary
        with hf.Timer('score_delta') as score_delta_timer:
            summary = transitioner.extract_state_summary(
                true_zs=true_zs,
                test_xs=test_xs,
                )
        summary['last_valid_zs'] = transitioner.state.getZIndices()
        summary['list_of_x_indices'] = transitioner.state.get_list_of_x_indices()
        summary['timing'].update({
            'timestamp':datetime.datetime.now(),
            'consolidate_delta':consolidate_delta,
            'score_delta':score_delta_timer.elapsed_secs,
            })
        summary['iter_num'] = iter_num
        iter_end_dt = datetime.datetime.now()
        summary['timing']['iter_start_dt'] = iter_start_dt
        summary['timing']['iter_end_dt'] = iter_end_dt

        #
        # save pkl'ed summary locally, push to s3 if appropriate
        pkl_file = create_pickle_file_str(num_nodes, run_key, iter_num,
                                          hypers_every_N=hypers_every_N)
        with hf.Timer('pickling summary', verbose=True):
            rf.pickle(summary, pkl_file, dir=run_full_dir)
        if self.push_to_s3:
            s3 = s3h.S3_helper(bucket_dir=run_bucket_dir, local_dir=run_full_dir)
            s3.put_s3(pkl_file)
            # s3h.S3_helper(bucket_dir=summary_bucket_dir).put_s3(pkl_file)
        hf.echo_date('done pickling summary')

        #
        # format summary to pass out 
        last_valid_zs = summary['last_valid_zs']
        master_alpha = summary['alpha']
        betas = summary['betas']
        master_inf_seed = summary['inf_seed']
        master_state = master_state_tuple(
            list_of_x_indices, last_valid_zs, master_alpha, betas,
            master_inf_seed=master_inf_seed, iter_num=iter_num)
        hf.echo_date('exit consolidate')
        yield run_key, master_state

    def steps(self):
        step_list = [self.mr(self.init)]
        infer_step = [
            self.mr(self.distribute_data),
            self.mr(self.infer, self.consolidate_data),
            ]
        step_list.extend(infer_step * self.num_steps)
        return step_list

if __name__ == '__main__':
    MRSeedInferer.run()
