#!python
from mrjob.job import MRJob
from mrjob.protocol import PickleProtocol, RawValueProtocol
#
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
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

    def load_options(self, args):
        super(MRSeedInferer, self).load_options(args=args)
        self.num_steps = self.options.num_steps
        self.num_iters = self.options.num_iters
        self.num_iters_per_step = self.num_iters/self.num_steps

    def infer(self, key, infer_seed_str):
        summaries = None
        try:
            infer_seed = int(infer_seed_str)
            run_spec = rf.gen_default_cifar_run_spec(
                problem_file,infer_seed,self.num_iters_per_step)
            summaries = rf.infer(run_spec)
        except Exception, e:
            print e
            summaries = [infer_seed_str]
        yield infer_seed_str,summaries

    def resume_infer(self,infer_seed_str,prior_summaries):
        summaries = None
        try:
            infer_seed = int(infer_seed_str)
            run_spec = rf.gen_default_cifar_run_spec(
                problem_file,infer_seed,self.num_iters_per_step)
            rf.modify_jobspec_to_results(run_spec,prior_summaries)
            summaries = rf.infer(run_spec)
        except Exception, e:
            print e
            summaries = [str(e)]
        prior_summaries.extend(summaries[1:]) # don't include init state
        yield infer_seed_str,prior_summaries

    def my_reduce(self,infer_seed_str,summaries_generator):
        for summaries in summaries_generator:
            yield infer_seed_str,summaries

    def steps(self):
        ret_list = [self.mr(self.infer, self.my_reduce)]
        num_resume_steps = self.num_steps-1
        ret_list.extend([self.mr(self.resume_infer, self.my_reduce)]*num_resume_steps)
        return ret_list

if __name__ == '__main__':
    MRSeedInferer.run()
