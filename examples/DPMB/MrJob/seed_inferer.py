#!python
import datetime
import os
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
problem_file = settings.cifar_100_bpr_problem_file
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
            if self.num_nodes == 1:
                self.num_steps = 1
            else:
                self.num_steps = self.num_iters/self.num_nodes
        #
        self.num_iters_per_step = self.num_iters/self.num_steps

    def init(self, key, run_key):
        master_infer_seed = int(run_key)
        run_spec = rf.gen_default_cifar_run_spec(
            problem_file=problem_file,
            infer_seed=master_infer_seed,
            num_iters=0 # just gibbs init 
           )
        problem = rf.gen_problem(run_spec['dataset_spec'])
        summaries = rf.infer(run_spec, problem)
        #
        summary = summaries[-1]
        summary['problem'] = problem
        summary['timing'] = {
            'start_time':datetime.datetime.now(),
            }
        iter_num = 0
        #
        pickle_str = os.path.join(
            settings.data_dir,
            create_pickle_file(self.num_nodes, run_key, iter_num)
            )
        #
        last_valid_zs = summary['last_valid_zs']
        master_alpha = summary['alpha']
        betas = summary['betas']
        master_inf_seed = summary['inf_seed']
        yielded_tuple = [last_valid_zs, master_alpha, betas,
                         master_inf_seed, iter_num]
        yield run_key, yielded_tuple

    def distribute_data(self, run_key, yielded_tuple):
        (init_z, master_alpha, betas, master_inf_seed, iter_num) = yielded_tuple
        node_data_indices, node_zs, child_gen_seed_list, \
            child_inf_seed_list, master_inf_seed = \
            rf.distribute_data(inf_seed=master_inf_seed,
                               num_nodes=self.num_nodes,
                               init_z=init_z)
        # does distribute_data permute random_state?
        for x_indices, zs, child_gen_seed, child_inf_seed in zip(
            node_data_indices, node_zs, child_gen_seed_list, child_inf_seed_list):
            if len(zs) == 0:
                continue
            yielded_tuple = (x_indices, zs, child_gen_seed, child_inf_seed,
                             master_alpha, betas, master_inf_seed, iter_num)
            yield run_key, yielded_tuple

    def infer(self, run_key, model_specs):
        x_indices, zs, child_gen_seed, child_inf_seed, \
                   master_alpha, betas, master_inf_seed, iter_num = model_specs
        run_spec = rf.run_spec_from_model_specs(model_specs, self)
        # FIXME : Perhaps problem can be generated in run_spec_from_model_specs?
        cifar = rf.unpickle(os.path.join(settings.data_dir, problem_file))
        cifar_xs = numpy.array(cifar['xs'],dtype=numpy.int32)
        init_x = [
            cifar['xs'][x_index]
            for x_index in x_indices
            ]
        problem = {'xs':init_x,'zs':None,'test_xs':None}
        # actually infer
        child_summaries = rf.infer(run_spec, problem)
        last_valid_zs = child_summaries[-1]['last_valid_zs']
        new_iter_num = iter_num + 1
        # FIXME : to robustify, should be checking for failure conditions
        # FIXME : here is where you pass timing if desired
        yielded_tuple = (last_valid_zs, x_indices,
                         master_alpha, betas,
                         master_inf_seed, new_iter_num)
        yield run_key, yielded_tuple

    def consolidate_data(self, run_key, yielded_val_generator):
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
        cifar = rf.unpickle(os.path.join(settings.data_dir, problem_file))
        init_x = numpy.array(cifar['xs'],dtype=numpy.int32)
        true_zs,cluster_idx = hf.canonicalize_list(cifar['zs'])
        test_xs = numpy.array(cifar['test_xs'],dtype=numpy.int32)
        consolidated_state = ds.DPMB_State(
            gen_seed=0, # FIXME : generate a random variate from master_inf_seed?
            num_cols=len(betas),
            num_rows=len(zs),
            init_alpha=master_alpha,
            init_betas=betas,
            init_z=zs,
            init_x=init_x
            )
        transitioner = dm.DPMB(master_inf_seed, consolidated_state,
                               infer_alpha=True, infer_beta=True)
        for transition_type in numpy.random.permutation([
                transitioner.transition_beta, transitioner.transition_alpha]):
            transition_type()
        summary = transitioner.extract_state_summary(
            true_zs=true_zs,
            test_xs=test_xs)
        summary['last_valid_zs'] = transitioner.state.getZIndices()
        summary['timing'] = {
            'timestamp':datetime.datetime.now()
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
