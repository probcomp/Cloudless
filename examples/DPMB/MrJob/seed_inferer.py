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
reload(rf)
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB.DPMB as dm
reload(dm)
import Cloudless.examples.DPMB.h5_functions as h5
reload(h5)
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless.examples.DPMB.s3_helper as s3h
reload(s3h)
import Cloudless.examples.DPMB.settings as settings
reload(settings)


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
    ' list_of_x_indices last_valid_zs '
    ' master_alpha betas master_inf_seed iter_num '
    )
child_state_tuple = namedtuple(
    'child_state_tuple',
    'list_of_x_indices x_indices zs '
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
        problem_full_file = os.path.join(run_full_dir, problem_file)
        if not os.path.isfile(problem_full_file):
            s3 = s3h.S3_helper(bucket_dir=run_bucket_dir, local_dir=run_full_dir)
            s3.verify_file(problem_file)
            h5_file = h5.get_h5_name_from_pkl_name(problem_file)
            s3.verify_file(h5_file)
        #
        # gibbs init or resume 
        problem_hexdigest = None
        with hf.Timer('init/resume') as init_resume_timer:
            if resume_file:
                resume_full_file = os.path.join(run_full_dir, resume_file)
                if not os.path.isfile(resume_full_file):
                    s3 = s3h.S3_helper(bucket_dir=run_bucket_dir,
                                       local_dir=run_full_dir)
                    s3.verify_file(resume_file)
                summary = rf.unpickle(resume_file, dir=run_full_dir)
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
        summary['problem_hexdigest'] = problem_hexdigest
        summary['timing'].update({
            'start_time':start_dt,
            'infer_problem_delta_t':init_resume_timer.elapsed_secs,
            })
        #
        # FIXME : infer will pickle over this
        if gibbs_init_file is None:
            gibbs_init_file = create_pickle_file_str(num_nodes, run_key, str(-1),
                                                     hypers_every_N)
        # FIXME: should only pickle if it wasn't read
        rf.pickle(summary, gibbs_init_file, dir=run_full_dir)
        if self.push_to_s3:
            s3 = s3h.S3_helper(bucket_dir=run_bucket_dir, local_dir=run_full_dir)
            s3.put_s3(gibbs_init_file)
        #
        # pull out the values to pass on
        list_of_x_indices = summary.get('list_of_x_indices', None)
        list_of_x_indices = summary.get('last_valid_list_of_x_indices',
                                        list_of_x_indices)
        last_valid_zs = summary.get('last_valid_zs', summary.get('zs', None))
        master_alpha = summary['alpha']
        betas = summary['betas']
        # master_inf_seed = summary['inf_seed'] # FIXME: set above so retain?
        iter_num = summary.get('iter_num', 0)
        master_state = master_state_tuple(
            list_of_x_indices, last_valid_zs, master_alpha, betas,
            master_inf_seed, iter_num,
            )
        yield run_key, master_state

    def distribute_data(self, run_key, master_state):
        iter_start_dt = datetime.datetime.now()
        child_counter = 0
        num_nodes = self.num_nodes
        # pull variables out of master_state
        list_of_x_indices = master_state.list_of_x_indices
        init_z = master_state.last_valid_zs
        master_alpha = master_state.master_alpha
        betas = master_state.betas
        master_inf_seed = master_state.master_inf_seed
        iter_num = master_state.iter_num
        #
        # generate child state info
        num_clusters = len(list_of_x_indices)
        node_info_tuples, random_state = rf.gen_cluster_dest_nodes(
            master_inf_seed, num_nodes, num_clusters)
        #
        # actually distribute
        for child_counter, node_info in enumerate(node_info_tuples):
            cluster_indices, child_inf_seed, child_gen_seed = node_info
            child_list_of_x_indices = \
                [list_of_x_indices[idx] for idx in cluster_indices]
            xs, zs = rf.list_of_x_indices_to_xs_and_zs(child_list_of_x_indices)
            #
            child_state = child_state_tuple(
                child_list_of_x_indices, xs, zs,
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
        # FIXME : write a new routine to read only those xs necessary
        # FIXME : look to 80MM TinyImages reader

        run_full_dir = os.path.join(data_dir, run_dir)
        problem_full_file = os.path.join(run_full_dir, problem_file)
        h5_full_file = h5.get_h5_name_from_pkl_name(problem_file)
        if not os.path.isfile(problem_full_file) or not os.path.isfile(h5_full_file):
            s3 = s3h.S3_helper(bucket_dir=run_bucket_dir, local_dir=run_full_dir)
            s3.verify_file(problem_file)
            h5_file = h5.get_h5_name_from_pkl_name(problem_file)
            s3.verify_file(h5_file)
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
                    # s3h.S3_helper(bucket_dir=summary_bucket_dir).put_s3(pkl_file)
                state.plot()
                save_str_base = '_'.join([
                        'infer_state',
                        'numnodes', str(num_nodes),
                        'child0',
                        'iter', str(iter_num)
                        ])
                save_str = os.path.join(run_full_dir, save_str_base)
                state.plot(save_str=save_str)
                save_str = os.path.join(run_full_dir, 'just_state_' + save_str_base)
                state.plot(save_str=save_str,
                                        which_plots=['just_data'])
            #
            child_summaries = rf.infer(
                run_spec, sub_problem, send_zs=True,
                post_infer_func=single_node_post_infer_func)
            # # FIXME: for now, pickle after the fact
            # for child_iter_num, child_summary in enumerate(child_summaries):
            #     pkl_file = get_child_pkl_file(child_iter_num)
            #     rf.pickle(child_summary, pkl_file, dir=data_dir)
            #     if self.push_to_s3:
            #         s3h.S3_helper(bucket_dir=summary_bucket_dir).put_s3(pkl_file)
        else:
            # FIXME : to robustify, should be checking for failure conditions
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
        new_iter_num = iter_num + 1
        child_state_out = child_state_tuple(
            list_of_x_indices, x_indices, last_valid_zs,
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
        list_of_x_indices = []
        #
        # consolidate data from child states
        child_state_counter = 0
        for child_state_out in child_infer_output_generator:
             zs_list.append(child_state_out.zs)
             x_indices_list.append(child_state_out.x_indices)
             list_of_x_indices.extend(child_state_out.list_of_x_indices)
             child_state_counter += 1
        hf.echo_date('received ' + str(child_state_counter) + ' child states')

        # all master_alpha, betas, master_inf_seed are all the same, use last
        master_alpha = child_state_out.master_alpha
        betas = child_state_out.betas
        master_inf_seed = child_state_out.master_inf_seed
        iter_num = child_state_out.iter_num
        iter_start_dt = child_state_out.iter_start_dt

        # format for use in singular state generation 
        jumbled_zs = rf.consolidate_zs(zs_list)
        x_indices = [y for x in x_indices_list for y in x]
        zs = [
            jumbled_zs[reorder_index]
            for reorder_index in numpy.argsort(x_indices)
            ]
        zs, cluster_idx = hf.canonicalize_list(zs)
        hf.echo_date('canonicalized zs')
        #
        run_full_dir = os.path.join(data_dir, run_dir)
        problem_full_file = os.path.join(run_full_dir, problem_file)
        if not os.path.isfile(problem_full_file):
            s3 = s3h.S3_helper(bucket_dir=run_bucket_dir, local_dir=run_full_dir)
            s3.verify_file(problem_file)
            h5_file = h5.get_h5_name_from_pkl_name(problem_file)
            s3.verify_file(h5_file)

        problem = rf.unpickle(problem_file, dir=run_full_dir)
        init_x = numpy.array(problem['xs'], dtype=numpy.int32)
        true_zs, test_xs = None, None
        if not postpone_scoring:
            if 'true_zs' in problem:
                true_zs = problem['true_zs']
                true_zs, cluster_idx = hf.canonicalize_list(true_zs)
            test_xs = numpy.array(problem['test_xs'], dtype=numpy.int32)
        hf.echo_date('read problem')

        # create singleton state
        consolidated_state = ds.DPMB_State(
            gen_seed=0, # FIXME : random variate from master_inf_seed?
            num_cols=len(betas),
            num_rows=len(zs),
            init_alpha=master_alpha,
            init_betas=betas,
            init_z=zs,
            init_x=init_x
            )
        consolidate_delta = hf.delta_since(start_dt)
        hf.echo_date('done creating consolidated state')
        #
        # run hyper inference
        transitioner = dm.DPMB(master_inf_seed, consolidated_state,
                               infer_alpha=True, infer_beta=True)
        # FIXME: is this right?
        random_state = hf.generate_random_state(master_inf_seed)
        transition_types = [
            transitioner.transition_beta, transitioner.transition_alpha]
        for transition_type in random_state.permutation(transition_types):
            transition_type_str = str(transition_type)
            hf.echo_date('transitioning type: ' + transition_type_str)
            transition_type()
        hf.echo_date('done transitioning')
        #

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

        hf.echo_date('faking plotting')
        #
        # save intermediate state plots
        # save_str_base = '_'.join([
        #         'infer_state',
        #         'numnodes', str(num_nodes),
        #         'iter', str(iter_num)
        #         ])
        # save_str = os.path.join(run_full_dir, save_str_base)
        # consolidated_state.plot(save_str=save_str)
        # save_str = os.path.join(run_full_dir, 'just_state_' + save_str_base)
        # consolidated_state.plot(save_str=save_str, which_plots=['just_data'])

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

    def s3_push_step(self, run_key, master_state_in):
        problem_file = self.problem_file
        run_dir = self.run_dir
        num_nodes = self.num_nodes
        hypers_every_N = self.num_iters_per_step
        run_bucket_dir = self.run_bucket_dir
        iter_num = master_state_in.iter_num
        seed_str = run_key
        run_full_dir = os.path.join(data_dir, run_dir)
        #
        # FIXME: presuming that s3 path is only one dir
        s3 = s3h.S3_helper(bucket_dir=run_bucket_dir, local_dir=run_full_dir)
        for iter_num in numpy.append(-1, range(1, iter_num + 1)):
            pkl_filename = create_pickle_file_str(
                num_nodes, seed_str, iter_num, hypers_every_N=hypers_every_N)
            pkl_full_filename = os.path.join(run_full_dir, pkl_filename)
            if os.path.isfile(pkl_full_filename):
                s3.put_s3(pkl_filename)
        # who pushes up problem, run_parameters?
        # whoever created them and set data_dir
        yield run_key, master_state_in

    def steps(self):
        step_list = [self.mr(self.init)]
        infer_step = [
            self.mr(self.distribute_data),
            self.mr(self.infer, self.consolidate_data),
            # self.mr(self.fake_infer, self.consolidate_data),
            ]
        step_list.extend(infer_step * self.num_steps)
        # if self.push_to_s3:
        #     step_list.extend([self.mr(self.s3_push_step)])
        return step_list

if __name__ == '__main__':
    MRSeedInferer.run()
