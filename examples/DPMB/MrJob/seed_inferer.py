#!python
from mrjob.job import MRJob
from mrjob.protocol import PickleProtocol, RawValueProtocol
#
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.settings as settings
reload(settings)


problem_file = settings.cifar_100_problem_file
num_steps = 1
num_iters = 6
num_iters_per_step = num_iters/num_steps

class MRSeedInferer(MRJob):

    INPUT_PROTOCOL = RawValueProtocol
    INTERNAL_PROTOCOL = PickleProtocol
    OUTPUT_PROTOCOL = PickleProtocol # RawValueProtocol # 
    
    def infer(self, key, infer_seed_str):
        summaries = None
        try:
            infer_seed = int(infer_seed_str)
            run_spec = rf.gen_default_cifar_run_spec(
                problem_file,infer_seed,num_iters_per_step)
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
                problem_file,infer_seed,num_iters_per_step)
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
        resume_steps = num_steps-1
        ret_list.extend([self.mr(self.resume_infer, self.my_reduce)]*resume_steps)
        return ret_list

if __name__ == '__main__':
    MRSeedInferer.run()
