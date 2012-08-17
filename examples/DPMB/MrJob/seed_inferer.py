#!python
import datetime
import os
import hashlib
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
import Cloudless.examples.DPMB.settings as settings
reload(settings)


# problem_file = settings.cifar_100_problem_file
problem_file = settings.tiny_image_problem_file

create_pickle_file = lambda num_nodes, seed_str, iter_num : \
    '_'.join([
        'summary',
        'numnodes' + str(num_nodes),
        'seed' + seed_str,
        'iternum' + str(iter_num) + '.pkl.gz'
        ])

class MRSeedInferer(MRJob):

    INPUT_PROTOCOL = RawValueProtocol
    INTERNAL_PROTOCOL = PickleProtocol
    OUTPUT_PROTOCOL = PickleProtocol # RawValueProtocol # 

    def configure_options(self):
        super(MRSeedInferer, self).configure_options()
        self.add_passthrough_option('--num-steps',type='int',default=None)
        self.add_passthrough_option('--num-iters',type='int',default=8)
        self.add_passthrough_option('--num-nodes',type='int',default=4)

    def load_options(self, args):
        super(MRSeedInferer, self).load_options(args=args)
        self.num_steps = self.options.num_steps
        self.num_iters = self.options.num_iters
        self.num_nodes = self.options.num_nodes
        if self.num_steps is None:
            if self.num_nodes == 1 and self.num_iters !=0:
                self.num_steps = 1
            else:
                self.num_steps = self.num_iters/self.num_nodes
        #
        self.num_iters_per_step = self.num_iters/self.num_steps \
            if self.num_steps != 0 else 0

    def init(self, key, run_key):
        master_infer_seed = int(run_key)
        run_spec = rf.gen_default_cifar_run_spec(
            problem_file=problem_file,
            infer_seed=master_infer_seed,
            num_iters=0 # no inference, just init
           )
        gen_problem_start = datetime.datetime.now()
        problem = rf.gen_problem(run_spec['dataset_spec'])
        gen_problem_delta_t = hf.delta_since(gen_problem_start)
        infer_problem_start = datetime.datetime.now()
        summaries = rf.infer(run_spec, problem)
        infer_problem_delta_t = hf.delta_since(infer_problem_start)
        # pickle up summary
        summary = summaries[-1]
        summary['problem_hexdigest'] = hashlib.sha224(str(problem)).hexdigest()
        summary['timing'] = {'start_time':datetime.datetime.now(),
                             'gen_problem_delta_t':gen_problem_delta_t,
                             'infer_problem_delta_t':infer_problem_delta_t}
        iter_num = 0
        # FIXME : infer will pickle over this
        pickle_file = create_pickle_file(self.num_nodes, run_key, str(-1))
        pickle_full_file = os.path.join(settings.data_dir,pickle_file)
        rf.pickle(summary, pickle_full_file)
        # pull out the values to pass on
        last_valid_zs = summary['last_valid_zs']
        master_alpha = summary['alpha']
        betas = summary['betas']
        master_inf_seed = summary['inf_seed']
        yielded_tuple = [last_valid_zs, master_alpha, betas,
                         master_inf_seed, iter_num]
        yield run_key, yielded_tuple

    def distribute_data(self, run_key, yielded_tuple):
        print 'distribute data'
        child_counter = 0
        (init_z, master_alpha, betas, master_inf_seed, iter_num) = yielded_tuple
        # does distribute_data permute random_state?
        node_data_indices, node_zs, child_gen_seed_list, \
            child_inf_seed_list, master_inf_seed = \
            rf.distribute_data(inf_seed=master_inf_seed,
                               num_nodes=self.num_nodes,
                               init_z=init_z)
        for x_indices, zs, child_gen_seed, child_inf_seed in zip(
            node_data_indices, node_zs, child_gen_seed_list, child_inf_seed_list):
            if len(zs) == 0:
                continue
            yielded_tuple = (x_indices, zs, child_gen_seed, child_inf_seed,
                             master_alpha, betas, master_inf_seed, iter_num,
                             child_counter)
            child_counter += 1
            yield run_key, yielded_tuple

    def infer(self, run_key, model_specs):
        print 'infer'
        x_indices, zs, child_gen_seed, child_inf_seed, \
                   master_alpha, betas, master_inf_seed, iter_num, child_counter \
                   = model_specs
        run_spec = rf.run_spec_from_model_specs(model_specs, self)
        # FIXME : Perhaps problem can be generated in run_spec_from_model_specs?
        orig_problem = rf.unpickle(os.path.join(settings.data_dir, problem_file))
        problem_xs = numpy.array(orig_problem['xs'],dtype=numpy.int32)
        init_x = [
            orig_problem['xs'][x_index]
            for x_index in x_indices
            ]
        sub_problem = {'xs':init_x,'zs':zs,'test_xs':None}
        # actually infer
        child_summaries = None
        if self.num_nodes == 1:
            # FIXME : can I just modify run_spec and use rf.infer
            # FIXME: only distinction is how frequently summaries are pickled
            sub_problem['test_xs'] = \
                numpy.array(orig_problem['test_xs'],dtype=numpy.int32)
            run_spec['infer_do_alpha_inference'] = True
            run_spec['infer_do_betas_inference'] = True
            child_summaries = rf.infer(run_spec, sub_problem)
            for child_iter_num,child_summary in enumerate(child_summaries):
                pickle_str = os.path.join(
                    settings.data_dir,
                    create_pickle_file(
                        self.num_nodes, run_key+'_child'+str(child_counter),
                        child_iter_num)
                    )
                rf.pickle(child_summary, pickle_str)
        else:
            child_summaries = rf.infer(run_spec, sub_problem)
        #
        last_valid_zs = child_summaries[-1]['last_valid_zs']
        new_iter_num = iter_num + 1
        # FIXME : to robustify, should be checking for failure conditions
        # FIXME : here is where you pass timing if desired
        yielded_tuple = (last_valid_zs, x_indices,
                         master_alpha, betas,
                         master_inf_seed, new_iter_num)
        yield run_key, yielded_tuple

    def consolidate_data(self, run_key, yielded_val_generator):
        print 'consolidate data'
        start_dt = datetime.datetime.now()
        zs_list = []
        x_indices_list = []
        for (child_zs, child_x_indices, master_alpha, betas,
             master_inf_seed, iter_num) in \
                yielded_val_generator:
            zs_list.append(child_zs)
            x_indices_list.append(child_x_indices)
            # master_alpha, betas, master_inf_seed fall through
        jumbled_zs = rf.consolidate_zs(zs_list)
        x_indices = [y for x in x_indices_list for y in x]
        zs = [
            jumbled_zs[reorder_index]
            for reorder_index in numpy.argsort(x_indices)
            ]
        #
        problem = rf.unpickle(os.path.join(settings.data_dir, problem_file))
        init_x = numpy.array(problem['xs'],dtype=numpy.int32)
        # true_zs,cluster_idx = hf.canonicalize_list(cifar['zs'])
        true_zs, cluster_idx = None, None
        if problem.get('zs',None) is not None:
            true_zs, cluster_idx = hf.canonicalize_list(problem['zs'])
        test_xs = numpy.array(problem['test_xs'],dtype=numpy.int32)
        consolidated_state = ds.DPMB_State(
            gen_seed=0, # FIXME : generate random variate from master_inf_seed?
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
        for transition_type in numpy.random.permutation([
                transitioner.transition_beta, transitioner.transition_alpha]):
            transition_type()
        start_dt = datetime.datetime.now()
        summary = transitioner.extract_state_summary(
            true_zs=true_zs,
            test_xs=test_xs)
        score_delta = hf.delta_since(start_dt)
        summary['last_valid_zs'] = transitioner.state.getZIndices()
        summary['timing'] = {
            'timestamp':datetime.datetime.now(),
            'consolidate_delta':consolidate_delta,
            'score_delta':score_delta
            }
        #
        pickle_str = os.path.join(
            settings.data_dir,
            create_pickle_file(self.num_nodes, run_key, iter_num)
            )
        rf.pickle(summary, pickle_str)
        #
        last_valid_zs = summary['last_valid_zs']
        master_alpha = summary['alpha']
        betas = summary['betas']
        master_inf_seed = summary['inf_seed']
        yielded_tuple = [last_valid_zs, master_alpha, betas,
                         master_inf_seed, iter_num]
        yield run_key, yielded_tuple

    def steps(self):
        ret_list = [self.mr(self.init)]
        infer_step = [
            self.mr(self.distribute_data),
            self.mr(self.infer, self.consolidate_data)
            ]
        ret_list.extend(infer_step * self.num_steps)
        return ret_list

if __name__ == '__main__':
    MRSeedInferer.run()
