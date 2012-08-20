#!python
import os
#
import matplotlib
matplotlib.use('Agg')
import pylab
import numpy
#
import Cloudless.examples.DPMB.MrJob.consolidate_summaries as cs
reload(cs)
import Cloudless.examples.DPMB.settings as settings
reload(settings)

shorten_name = lambda x: x[8:-8]

# define the desired paths
data_dirs = [
    os.path.join(settings.data_dir,data_dir)
    for data_dir in ['one_node','two_node','four_node']
    ]

# read the summaries
summaries_dict = {}
for data_dir in data_dirs:
    summary_names = cs.get_summary_names(data_dir)
    temp_summaries_dict = cs.get_summaries_dict(summary_names,data_dir)
    cs.print_info(summaries_dict)
    summaries_dict.update(temp_summaries_dict)

# create test_lls plot
pylab.figure()
for summaries_name, summaries in summaries_dict.iteritems():
    timing = numpy.array(cs.extract_delta_t(summaries),dtype=float)
    test_lls = numpy.array(cs.extract_test_lls(summaries),dtype=float)
    pylab.plot(timing,test_lls)
legend_list = map(shorten_name,summaries_dict.keys())
pylab.legend(legend_list,loc='lower right')
pylab.title('test_lls')
pylab.xlabel('time (seconds)')
pylab.ylabel('test set mean log likelihood')
pylab.savefig('test_lls')

# create score plot
pylab.figure()
for summaries_name, summaries in summaries_dict.iteritems():
    timing = numpy.array(cs.extract_delta_t(summaries),dtype=float)
    scores = numpy.array(cs.extract_score(summaries),dtype=float)
    pylab.plot(timing,scores)
legend_list = map(shorten_name,summaries_dict.keys())
pylab.legend(legend_list,loc='lower right')
pylab.title('scores')
pylab.xlabel('time (seconds)')
pylab.ylabel('model score')
pylab.savefig('scores')
