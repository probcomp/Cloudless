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


# problem_file = 'small_problem.pkl.gz'
problem_file = settings.cifar_100_problem_file

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
            num_iters=0
           )
        problem = rf.gen_problem(run_spec['dataset_spec'])
        # this will be gibbs-init
        summaries = rf.infer(run_spec,problem)
        # problem contains the true zs,xs and test_xs
        summaries[0]['problem'] = problem
        summaries[0]['timing'] = {
            'start_time':datetime.datetime.now(),
            'run_sum':0}
        yield infer_seed_str,summaries

    def distribute_data(self,infer_seed_str,summaries):
        init_alpha = summaries[-1]['alpha']
        init_betas = summaries[-1]['betas']
        inf_seed = summaries[-1]['inf_seed']
        init_z = summaries[-1]['last_valid_zs']
        init_x = summaries[0]['problem']['xs']
        node_data_indices,node_zs,gen_seed_list,inf_seed_list,random_state = \
            rf.distribute_data(inf_seed=inf_seed,num_nodes=self.num_nodes,
            init_z=init_z)
        #
        for x_indices,zs,gen_seed,inf_seed in zip(
            node_data_indices,node_zs,gen_seed_list,inf_seed_list):
            if len(zs) == 0:
                continue
            yield infer_seed_str,(x_indices,zs,gen_seed,inf_seed,summaries)

    def infer(self,infer_seed_str,model_specs):
        (x_indices,zs,gen_seed,inf_seed,summaries) = model_specs
        dataset_spec = {}
        dataset_spec["gen_seed"] = 0
        dataset_spec["num_cols"] = 256
        dataset_spec["num_rows"] = len(zs)
        dataset_spec["gen_alpha"] = 3.0 # FIXME: are these actually needed
        dataset_spec["gen_betas"] = [3.0 for x in range(dataset_spec['num_cols'])]
        dataset_spec["gen_z"] = None # FIXME: are these actually needed?
        #
        run_spec = {}
        run_spec["dataset_spec"] = dataset_spec
        run_spec["num_iters"] = self.num_iters_per_step
        run_spec["num_nodes"] = 1
        run_spec["infer_seed"] = inf_seed
        # sub_alpha = alpha/num_nodes
        run_spec["infer_init_alpha"] = summaries[-1]['alpha']/self.num_nodes
        run_spec["infer_init_betas"] = summaries[-1]['betas']
        # no hypers in child state inference
        run_spec["infer_do_alpha_inference"] = False
        run_spec["infer_do_betas_inference"] = False
        run_spec["infer_init_z"] = zs
        run_spec["time_seatbelt"] = None
        run_spec["ari_seatbelt"] = None
        run_spec["verbose_state"] = False
        init_x = [summaries[0]['problem']['xs'][x_index] for x_index in x_indices]
        problem = {'xs':init_x,'zs':None,'test_xs':None}
        child_summaries = rf.infer(run_spec,problem)
        # FIXME : use modify_jobspec_to_results(jobspec,job_value) ? 
        val = (summaries,child_summaries,x_indices) # must assemble before yield, else reducer gets nothing
        yield infer_seed_str, val

    def consolidate_data(self,infer_seed_str,summaries_triplet_generator):
        parent_summaries_list = []
        zs_list = []
        x_indices_list = []
        for summaries_triplet in summaries_triplet_generator:
            parent_summaries_list.append(summaries_triplet[0])
            zs_list.append(summaries_triplet[1][-1]['last_valid_zs'])
            x_indices_list.append(summaries_triplet[2])
        zs = rf.consolidate_zs(zs_list)
        x_indices = [y for x in x_indices_list for y in x]
        # reorder zs according to original data
        z_reorder_indices = numpy.argsort(x_indices)
        parent_summaries = parent_summaries_list[0]
        prior_summary = parent_summaries[-1]
        alpha = prior_summary['alpha']
        betas = prior_summary['betas']
        inf_seed = prior_summary['inf_seed']
        consolidated_state = ds.DPMB_State(
            gen_seed=0,
            num_cols=len(betas),
            num_rows=len(zs),
            init_alpha=alpha,
            init_betas=betas,
            init_z=[zs[reorder_index] for reorder_index in z_reorder_indices],
            init_x=parent_summaries[0]['problem']['xs']
            )
        transitioner = dm.DPMB(inf_seed,consolidated_state,
                               infer_alpha=True,infer_beta=True)
        # FIXME : randomize order?
        transitioner.transition_alpha()
        transitioner.transition_beta()
        consolidated_summary = transitioner.extract_state_summary(
            true_zs=parent_summaries[0]['problem']['zs'],
            test_xs=parent_summaries[0]['problem']['test_xs'])
        consolidated_summary['last_valid_zs'] = transitioner.state.getZIndices()
        # FIXME : not tracking timing
        parent_summaries.append(consolidated_summary)
        parent_summaries[-1]['timing'] = {
            'timestamp':datetime.datetime.now(),
            'run_sum':hf.delta_since(parent_summaries[0]['timing']['start_time'])}
        yield infer_seed_str, parent_summaries

    def steps(self):
        ret_list = [self.mr(self.init)]
        infer_step = [self.mr(self.distribute_data),self.mr(self.infer,self.consolidate_data)]
        ret_list.extend(infer_step * self.num_steps)
        return ret_list

if __name__ == '__main__':
    MRSeedInferer.run()
