#!python
import os
import re
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

is_summary = lambda x : x[:8] == 'summary_'
split_summary_re = re.compile('^(.*iternum)(\d*).pkl.gz')
split_summary = lambda x : split_summary_re.match(x).groups()
create_summary_str = lambda summary_name, iter_num : \
    summary_name + str(iter_num) + '.pkl.gz'

data_files = os.listdir(settings.data_dir)
summary_files = filter(is_summary,data_files)
summary_names = {}
for summary_file in sorted(summary_files):
    summary_name,iter_num_str = split_summary(summary_file)
    summary_names.setdefault(summary_name,[]).append(int(iter_num_str))
    
summaries_dict = {}
for summary_name, iter_list in summary_names.iteritems():
    iter_list = sorted(iter_list)
    for iter_num in iter_list:
        filename = create_summary_str(summary_name,iter_num)
        filename = os.path.join(settings.data_dir,filename)
        new_summary = rf.unpickle(filename)
        summaries_dict.setdefault(summary_name,[]).append(new_summary)

extract_funcs = [
    lambda x : x['score'],
    lambda x : x['test_lls'],
    # lambda x : x['timing']['run_sum']
    ]

for extract_func in extract_funcs:
    for filename in sorted(summaries_dict.keys()):
        summaries = summaries_dict[filename]
        print filename
        print [extract_func(summary) for summary in summaries]
