#!python
from mrjob.job import MRJob
from mrjob.protocol import PickleProtocol, RawValueProtocol
#
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.settings as settings
reload(settings)


problem_file = settings.cifar_100_problem_file
num_iters = 1

import pdb

class MRSeedInferer(MRJob):

    INPUT_PROTOCOL = RawValueProtocol
    INTERNAL_PROTOCOL = PickleProtocol
    OUTPUT_PROTOCOL = PickleProtocol
    
    def infer(self, key, infer_seed_str):
        summaries = None
        try:
            infer_seed = int(infer_seed_str)
            run_spec = rf.gen_default_cifar_run_spec(
                problem_file,infer_seed,num_iters)
            # summaries = rf.infer(run_spec)
        except Exception, e:
            print e
        summaries = run_spec
        yield infer_seed_str,summaries

    def my_reduce(self, infer_seed,summaries):
        # list(summaries)
        yield infer_seed,[str(summary) for summary in summaries]

    def steps(self):
        return [self.mr(self.infer, self.my_reduce),]

if __name__ == '__main__':
    MRSeedInferer.run()
