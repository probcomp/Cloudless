#!python
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
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless.examples.DPMB.s3_helper as s3h
reload(s3h)
import Cloudless.examples.DPMB.settings as settings
reload(settings)
# importing csmp will create a structured problem
# import Cloudless.examples.DPMB.MrJob.create_synthetic_mrjob_problem as csmp

data_dir = settings.data_dir
summary_bucket_dir = settings.s3.summary_bucket_dir
problem_bucket_dir = settings.s3.problem_bucket_dir
#
# problem_file = settings.tiny_image_problem_file
problem_file = 'tiny_image_problem_nImages_320000_nPcaTrain_10000.pkl.gz'
resume_file = None

create_pickle_file_str = lambda num_nodes, seed_str, iter_num : \
    '_'.join([
        'summary',
        'numnodes' + str(num_nodes),
        'seed' + seed_str,
        'iternum' + str(iter_num) + '.pkl.gz'
        ])
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
    )

class MRSeedInferer(MRJob):

    INPUT_PROTOCOL = RawValueProtocol
    INTERNAL_PROTOCOL = PickleProtocol
    OUTPUT_PROTOCOL = PickleProtocol

    def configure_options(self):
        super(MRSeedInferer, self).configure_options()
        self.add_passthrough_option('--num-steps', type='int', default=None)
        self.add_passthrough_option('--num-iters', type='int', default=8)
        self.add_passthrough_option('--num-nodes', type='int', default=4)
        self.add_passthrough_option('--time-seatbelt', type='int', default=None)

    def load_options(self, args):
        super(MRSeedInferer, self).load_options(args=args)
        self.num_steps = self.options.num_steps
        self.num_iters = self.options.num_iters
        self.num_nodes = self.options.num_nodes
        # time_seatbelt only works on single node inference
        self.time_seatbelt = self.options.time_seatbelt
        if self.num_steps is None:
            if self.num_nodes == 1 and self.num_iters !=0:
                self.num_steps = 1
            else:
                self.num_steps = self.num_iters/self.num_nodes
        #
        self.num_iters_per_step = self.num_iters/self.num_steps \
            if self.num_steps != 0 else 0

    def init(self, key, run_key):
        start_dt = datetime.datetime.now()
        master_infer_seed = int(run_key)
        num_nodes = self.num_nodes
        s3h.S3_helper(bucket_dir=problem_bucket_dir).verify_file(problem_file)
        #
        # gibbs init or resume 
        problem_hexdigest = None
        with hf.Timer('init/resume') as init_resume_timer:
            if resume_file:
                summary = rf.unpickle(resume_file, dir=data_dir)
            else:
                run_spec = rf.gen_default_cifar_run_spec(
                    problem_file=problem_file,
                    infer_seed=master_infer_seed,
                    num_iters=0 # no inference, just init
                   )
                # FIXME: should I pass permute=False here?
                problem = rf.gen_problem(run_spec['dataset_spec'])
                summary = rf.infer(run_spec, problem)[-1]
                problem_hexdigest = get_hexdigest(problem)
        summary['problem_hexdigest'] = problem_hexdigest
        summary['timing'] = {
            'start_time':start_dt,
            'infer_problem_delta_t':init_resume_timer.elapsed_secs,
            }
        #
        # FIXME : infer will pickle over this
        pickle_file = create_pickle_file_str(num_nodes, run_key, str(-1))
        rf.pickle(summary, pickle_file, dir=data_dir)
        s3h.S3_helper(bucket_dir=summary_bucket_dir).put_s3(pickle_file)
        #
        # pull out the values to pass on
        list_of_x_indices = summary.get('last_valid_list_of_x_indices', None)
        last_valid_zs = summary.get('last_valid_zs', summary.get('zs', None))
        master_alpha = summary['alpha']
        betas = summary['betas']
        master_inf_seed = summary['inf_seed']
        iter_num = summary.get('iter_num', 0)
        master_state = master_state_tuple(
            list_of_x_indices, last_valid_zs, master_alpha, betas,
            master_inf_seed, iter_num)
        yield run_key, master_state

    def distribute_data(self, run_key, master_state):
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
                child_inf_seed, child_gen_seed, child_counter)
            yield run_key, child_state

    def infer(self, run_key, child_state_in):
        num_nodes = self.num_nodes
        num_iters_per_step = self.num_iters_per_step
        x_indices = child_state_in.x_indices
        zs = child_state_in.zs
        master_alpha = child_state_in.master_alpha
        betas = child_state_in.betas
        master_inf_seed = child_state_in.master_inf_seed
        iter_num = child_state_in.iter_num
        child_inf_seed = child_state_in.child_inf_seed
        child_gen_seed = child_state_in.child_gen_seed
        child_counter = child_state_in.child_counter
        #
        run_spec = rf.run_spec_from_child_state_info(
            zs, master_alpha, betas, child_inf_seed, child_gen_seed,
            num_iters_per_step, num_nodes)
        # FIXME : write a new routine to read only those xs necessary
        # FIXME : look to 80MM TinyImages reader
        orig_problem = rf.unpickle(problem_file, dir=data_dir)
        problem_xs = numpy.array(orig_problem['xs'], dtype=numpy.int32)
        init_x = [
            orig_problem['xs'][x_index]
            for x_index in x_indices
            ]
        sub_problem = {'xs':init_x, 'zs':zs, 'test_xs':None}
        # actually infer
        get_child_pkl_file = lambda child_iter_num: create_pickle_file_str(
            num_nodes, run_key+'_child'+str(child_counter), child_iter_num)
        child_summaries = None
        if num_nodes == 1:
            # FIXME : keep time seatbelt for single node version?
            run_spec['time_seatbelt'] = self.time_seatbelt # FIXME
            #
            sub_problem['test_xs'] = \
                numpy.array(orig_problem['test_xs'], dtype=numpy.int32)
            run_spec['infer_do_alpha_inference'] = True
            run_spec['infer_do_betas_inference'] = True
            # FIXME : would be nice if intermediate results were pickled
            # FIXME : else, no way to tell how much progress has been made
            child_summaries = rf.infer(run_spec, sub_problem, send_zs=True)
            # FIXME: for now, pickle after the fact
            for child_iter_num, child_summary in enumerate(child_summaries):
                pkl_file = get_child_pkl_file(child_iter_num)
                rf.pickle(child_summary, pkl_file, dir=data_dir)
                s3h.S3_helper(bucket_dir=summary_bucket_dir).put_s3(pkl_file)
        else:
            # FIXME : to robustify, should be checking for failure conditions
            child_summaries = rf.infer(run_spec, sub_problem)

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
            None, None, None
            )
        yield run_key, child_state_out

    def consolidate_data(self, run_key, child_infer_output_generator):
        start_dt = datetime.datetime.now()
        num_nodes = self.num_nodes
        zs_list = []
        x_indices_list = []
        list_of_x_indices = []
        #
        for child_state_out in child_infer_output_generator:
             zs_list.append(child_state_out.zs)
             x_indices_list.append(child_state_out.x_indices)
             list_of_x_indices.extend(child_state_out.list_of_x_indices)
        # all master_alpha, betas, master_inf_seed fall are the same
        master_alpha = child_state_out.master_alpha
        betas = child_state_out.betas
        master_inf_seed = child_state_out.master_inf_seed
        iter_num = child_state_out.iter_num

        # 
        jumbled_zs = rf.consolidate_zs(zs_list)
        x_indices = [y for x in x_indices_list for y in x]
        zs = [
            jumbled_zs[reorder_index]
            for reorder_index in numpy.argsort(x_indices)
            ]
        #
        problem = rf.unpickle(problem_file, dir=data_dir)
        init_x = numpy.array(problem['xs'], dtype=numpy.int32)
        true_zs, cluster_idx = None, None
        if 'zs' in  problem:
            true_zs, cluster_idx = hf.canonicalize_list(problem['zs'])

        # FIXME: is this (NEW) canonicaliziation necessary?
        zs, cluster_idx = hf.canonicalize_list(zs) # FIXME

        test_xs = numpy.array(problem['test_xs'], dtype=numpy.int32)
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
        transitioner = dm.DPMB(master_inf_seed, consolidated_state,
                               infer_alpha=True, infer_beta=True)
        # FIXME: is this right?
        random_state = hf.generate_random_state(master_inf_seed)
        transition_types = [
            transitioner.transition_beta, transitioner.transition_alpha]
        for transition_type in random_state.permutation(transition_types):
            transition_type()
        with hf.Timer('score_delta') as score_delta_timer:
            summary = transitioner.extract_state_summary(
                true_zs=true_zs,
                test_xs=test_xs)
        summary['last_valid_zs'] = transitioner.state.getZIndices()
        summary['list_of_x_indices'] = transitioner.state.get_list_of_x_indices()
        summary['timing'] = {
            'timestamp':datetime.datetime.now(),
            'consolidate_delta':consolidate_delta,
            'score_delta':score_delta_timer.elapsed_secs,
            }
        summary['iter_num'] = iter_num
        #
        pkl_file = create_pickle_file_str(num_nodes, run_key, iter_num)
        rf.pickle(summary, pkl_file, dir=data_dir)
        s3h.S3_helper(bucket_dir=summary_bucket_dir).put_s3(pkl_file)
        #
        last_valid_zs = summary['last_valid_zs']
        master_alpha = summary['alpha']
        betas = summary['betas']
        master_inf_seed = summary['inf_seed']
        master_state = master_state_tuple(
            list_of_x_indices, last_valid_zs, master_alpha, betas,
            master_inf_seed=master_inf_seed, iter_num=iter_num)
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
