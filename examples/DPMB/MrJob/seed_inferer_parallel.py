e#!python
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
pkl_data = rf.unpickle(problem_file)
init_x = pkl_data["subset_xs"]
true_zs,ids = hf.canonicalize_list(pkl_data["subset_zs"])

class MRSeedInferer(MRJob):

    INPUT_PROTOCOL = RawValueProtocol
    INTERNAL_PROTOCOL = PickleProtocol
    OUTPUT_PROTOCOL = PickleProtocol # RawValueProtocol # 

    self.summaries = {}
    self.master_states = {}
    self.random_states = {}

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
        summaries = None
        infer_seed = int(infer_seed_str)
        run_spec = rf.gen_default_cifar_run_spec(
            problem_file,infer_seed,0)
        summaries = rf.infer(run_spec)
        self.summaries.sefdefault(infer_seed,[]).extend(summaries)
        yield infer_seed_str,summaries[-1]

    def distribute_data(self,infer_seed_str,last_summary):
        gen_seed = ?
        init_alpha = last_summary['alpha']
        init_betas = last_summary['betas']
        inf_seed = last_summary['infer_seed']
        init_z = last_summary['last_valid_zs']
        node_data,node_zs,gen_seed_list,inf_seed_list,random_state = distribute_data(
            gen_seed,inf_seed,self.num_nodes,init_x,init_z,init_alpha,init_betas)
        self.random_states[infer_seed_str] = random_state
        #
        for xs,zs,gen_seed,inf_seed in zip(node_data,node_zs,gen_seed_list,inf_seed_list):
            yield infer_seed_str,(xs,zs,gen_seed,inf_seed)

    def infer(self,infer_seed_str,(xs,zs,gen_seed,inf_seed)):
        # generate a run_spec
        # no alpha,beta inference should be going on in child states
        run_spec = None
        new_summaries = rf.infer(run_spec)
        yield infer_seed_str, new_summaries

    def consolidate_data(self,infer_seed_str,summaries_list):
        zs_list = [summaries[-1]['last_valid_zs'] for summaries in summaries_list]
        zs = rf.consolidate_zs(zs_list)
        new_state = state(zs)
        new_summaries = new_state.extract_state_summary()
        self.summaries[int(infer_seed)].extend(new_summaries[1:]) # don't include init summary
        yield infer_seed_str, new_summaries[-1]

    def steps(self):
        # one inference step is 
        # [ self.mr(mapper=pre-mapper), self.mr(mapper=DPMB_mapper,reducer=coalesce) ] * PDPMB_iters
        #                                              ^^^^^^^^^^^ runs N steps from hypers every N
        num_resume_steps = self.num_steps-1
        ret_list = [self.mr(self.init)]
        infer_steps = [self.mr(self.resume_infer, self.my_reduce)] * num_resume_steps
        ret_list.extend(infer_steps)
        return ret_list

if __name__ == '__main__':
    MRSeedInferer.run()
