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

filename = "output_sequential.txt"
summaries_sequential = {}
if os.path.isfile(filename):
    with open(filename) as fh:
        for line in fh:
            infer_seed,summaries = which_protocol.read(line)
            summaries_sequential[infer_seed] = summaries

filename = "output_single.txt"
summaries_single = {}
if os.path.isfile(filename):
    with open(filename) as fh:
        for line in fh:
            infer_seed,summaries = which_protocol.read(line)
            summaries_single[infer_seed] = summaries

print 'sequential'
for key in summaries_sequential:
    print [summary['score'] for summary in summaries_sequential[key]]
print 'single'
for key in summaries_single:
    print [summary['score'] for summary in summaries_single[key]]
print
print 'sequential'
for key in summaries_sequential:
    print [mean(summary['test_lls']) for summary in summaries_sequential[key]]
print 'single'
for key in summaries_single:
    print [mean(summary['test_lls']) for summary in summaries_single[key]]
