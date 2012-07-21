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

class MRSeedInferer(MRJob):

    INPUT_PROTOCOL = RawValueProtocol
    INTERNAL_PROTOCOL = PickleProtocol
    OUTPUT_PROTOCOL = PickleProtocol # RawValueProtocol # 

    def configure_options(self):
        super(MRSeedInferer, self).configure_options()
        self.add_passthrough_option(
            '--num-steps',type='int',default=2)
        self.add_passthrough_option(
            '--num-iters',type='int',default=8)
        self.add_passthrough_option(
            '--num-nodes',type='int',default=4)

    def load_options(self, args):
        super(MRSeedInferer, self).load_options(args=args)
        self.num_steps = self.options.num_steps
        self.num_iters = self.options.num_iters
        self.num_nodes = self.options.num_nodes
        #
        self.num_iters_per_step = self.num_iters/self.num_steps

    def init(self, key, infer_seed_str):
        infer_seed = int(infer_seed_str)
        run_spec = rf.gen_default_cifar_run_spec(
            problem_file=problem_file,
            infer_seed=infer_seed,
            num_iters=0 # just gibbs init 
           )
        problem = rf.gen_problem(run_spec['dataset_spec'])
        summaries = rf.infer(run_spec,problem)
        # problem contains the true zs,xs and test_xs
        summaries[0]['problem'] = problem
        summaries[0]['timing'] = {
            'start_time':datetime.datetime.now(),
            'run_sum':0
            }
        yield infer_seed_str,summaries

    def distribute_data(self,infer_seed_str,summaries):
        inf_seed = summaries[-1]['inf_seed']
        init_z = summaries[-1]['last_valid_zs']
        node_data_indices,node_zs,gen_seed_list,inf_seed_list,random_state = \
            rf.distribute_data(inf_seed=inf_seed, 
                               num_nodes=self.num_nodes,
                               init_z=init_z)
        for x_indices,zs,gen_seed,inf_seed in zip(
            node_data_indices,node_zs,gen_seed_list,inf_seed_list):
            if len(zs) == 0:
                continue
            yield infer_seed_str,(x_indices,zs,gen_seed,inf_seed,summaries)

    def infer(self,infer_seed_str,model_specs):
        (x_indices,zs,gen_seed,inf_seed,summaries) = model_specs
        run_spec = rf.run_spec_from_model_specs(model_specs,self)
        init_x = [
            summaries[0]['problem']['xs'][x_index]
            for x_index in x_indices
            ]
        problem = {'xs':init_x,'zs':None,'test_xs':None}
        # actually infer
        child_summaries = rf.infer(run_spec,problem)
        # FIXME : use modify_jobspec_to_results(jobspec,job_value) ? 
        val = (summaries,child_summaries,x_indices)
        # must assemble before yield, else reducer gets nothing ?
        yield infer_seed_str, val

    def consolidate_data(self,infer_seed_str,summaries_triplet_generator):
        zs_list = []
        x_indices_list = []
        for parent_summaries, child_summaries, child_x_indices in \
                summaries_triplet_generator:
            # let parent_summaries fall through
            zs_list.append(child_summaries[-1]['last_valid_zs'])
            x_indices_list.append(child_x_indices)
        zs = rf.consolidate_zs(zs_list)
        x_indices = [y for x in x_indices_list for y in x]
        # reorder zs according to original data
        z_reorder_indices = numpy.argsort(x_indices)
        prior_summary = parent_summaries[-1]
        inf_seed = prior_summary['inf_seed']
        consolidated_state = ds.DPMB_State(
            gen_seed=0,
            num_cols=len(prior_summary['betas']),
            num_rows=len(zs),
            init_alpha=prior_summary['alpha'],
            init_betas=prior_summary['betas'],
            init_z=[zs[reorder_index] for reorder_index in z_reorder_indices],
            init_x=parent_summaries[0]['problem']['xs']
            )
        transitioner = dm.DPMB(inf_seed,consolidated_state,
                               infer_alpha=True,infer_beta=True)
        for transition_type in numpy.random.permutation([
                transitioner.transition_beta,transitioner.transition_alpha]):
            transition_type()
        consolidated_summary = transitioner.extract_state_summary(
            true_zs=parent_summaries[0]['problem']['zs'],
            test_xs=parent_summaries[0]['problem']['test_xs'])
        #
        consolidated_summary['last_valid_zs'] = transitioner.state.getZIndices()
        parent_summaries.append(consolidated_summary)
        parent_summaries[-1]['timing'] = {
            'timestamp':datetime.datetime.now(),
            'run_sum':hf.delta_since(
                parent_summaries[0]['timing']['start_time'])
            }
        yield infer_seed_str, parent_summaries

    def steps(self):
        ret_list = [self.mr(self.init)]
        infer_step = [
            self.mr(self.distribute_data),
            self.mr(self.infer,self.consolidate_data)
            ]
        ret_list.extend(infer_step * self.num_steps)
        return ret_list

if __name__ == '__main__':
    MRSeedInferer.run()
