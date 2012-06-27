#!python
import os
#
import numpy
from mrjob.job import MRJob
from mrjob.protocol import PickleProtocol, RawValueProtocol
#
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless.examples.DPMB.settings as settings
reload(settings)


problem_file = settings.cifar_100_problem_file

class MRSeedInferer(MRJob):

    INPUT_PROTOCOL = RawValueProtocol
    INTERNAL_PROTOCOL = PickleProtocol
    OUTPUT_PROTOCOL = PickleProtocol # RawValueProtocol # 

    def configure_options(self):
        super(MRSeedInferer, self).configure_options()
        self.add_passthrough_option(
            '--num-steps',type='int',default=1)
        self.add_passthrough_option(
            '--num-iters',type='int',default=4)
        self.add_passthrough_option(
            '--num-nodes',type='int',default=2)

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
        # this will be gibbs-init
        summaries = rf.infer(run_spec)
        yield infer_seed_str,summaries

    def distribute_data(self,infer_seed_str,summaries):
        init_alpha = summaries[-1]['alpha']
        init_betas = summaries[-1]['betas']
        inf_seed = summaries[-1]['inf_seed']
        init_z = summaries[-1]['last_valid_zs']
        node_data,node_zs,gen_seed_list,inf_seed_list,random_state = \
            rf.distribute_data(
            gen_seed=0, # this shouldn't matter/shouldn't be used
            inf_seed=inf_seed,
            num_nodes=self.num_nodes,
            init_x=init_x,
            init_z=init_z,
            init_alpha=init_alpha,
            init_betas=init_betas)
        #
        for xs,zs,gen_seed,inf_seed in zip(
            node_data,node_zs,gen_seed_list,inf_seed_list):
            if len(zs) == 0:
                continue
            yield infer_seed_str,(xs,zs,gen_seed,inf_seed,summaries)

    def infer(self,infer_seed_str,model_specs):
        (xs,zs,gen_seed,inf_seed,summaries) = model_specs
        dataset_spec = {}
        dataset_spec["gen_seed"] = 0
        dataset_spec["num_cols"] = 256
        dataset_spec["num_rows"] = len(zs)
        dataset_spec["gen_alpha"] = 3.0 #FIXME: could make it MLE alpha later
        dataset_spec["gen_betas"] = [3.0 for x in range(dataset_spec['num_cols'])]
        dataset_spec["gen_z"] = true_zs
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
        problem = {'xs':init_x,'zs':true_zs,'test_xs':test_xs}
        child_summaries = rf.infer(run_spec,problem)
        # FIXME : use modify_jobspec_to_results(jobspec,job_value) ? 
        yield infer_seed_str, (summaries,child_summaries)

    def consolidate_data(self,infer_seed_str,summaries_pair_list):
        zs_list = [summaries_pair[1][-1]['last_valid_zs'] 
                   for summaries_pair in summaries_pair_list]
        zs = rf.consolidate_zs(zs_list)
        parent_summaries = summaries_pair_list[0][0]
        prior_summary = parent_summaries[-1]
        alpha = prior_summary['alpha']
        betas = prior_summary['betas']
        inf_seed = prior_summary['inf_seed']
        consolidated_state = ds.DPMB_State(
            gen_seed=0,
            num_cols=len(betas)
            num_rows=len(zs),
            init_alpha=alphas,
            init_betas=betas,
            init_z=zs,
            init_x=init_x
            )
        transitioner = dm.DPMB(inf_seed,consolidated_state,alpha,betas)
        # FIXME : randomize order?
        consolidated_state.transition_alpha()
        consolidated_state.transition_beta()
        consolidated_summary = consolidated_state.extract_state_summary()
        # FIXME : not tracking timing
        parent_summaries.append(consolidated_summary)
        yield infer_seed_str, parent_summaries

    def steps(self):
        num_resume_steps = self.num_steps-1
        ret_list = [self.mr(self.init)]
        infer_step = [self.mr(self.distribute_data),self.mr(self.infer,self.consolidate_data)]
        ret_list.extend(infer_step * num_resume_steps)
        return ret_list

if __name__ == '__main__':
    MRSeedInferer.run()
