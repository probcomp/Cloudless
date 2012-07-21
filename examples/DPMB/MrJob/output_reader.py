#!python
import os
#
from mrjob.job import MRJob
from mrjob.protocol import PickleProtocol, RawValueProtocol
from numpy import *
#
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.settings as settings
reload(settings)

which_protocol = PickleProtocol

summaries_dict = {}
filenames = ['num_nodes_1.txt','num_nodes_2.txt','num_nodes_4.txt']
for filename in filenames:
    this_summary = {}
    if os.path.isfile(filename):
        with open(filename) as fh:
            for line in fh:
                infer_seed,summaries = which_protocol.read(line)
                this_summary[infer_seed] = summaries
    summaries_dict[filename] = this_summary

extract_funcs = [
    lambda x : x['score'],
    lambda x : x['test_lls'],
    lambda x : x['timing']['run_sum']
    ]

for extract_func in extract_funcs:
    for filename in sorted(summaries_dict.keys()):
        summaries = summaries_dict[filename]
        print filename
        for key in summaries:
            print [extract_func(summary) for summary in summaries[key]]
