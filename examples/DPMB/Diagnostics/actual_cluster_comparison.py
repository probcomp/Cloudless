# generate synthetic data
import os
import re
from collections import defaultdict
#
import numpy
import pylab
import pandas
#
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)


# some settings
base_dir = '/usr/local/Cloudless/examples/DPMB/Data'
data_dir_prefix = 'programmatic_mrjob_'
parameters_filename = 'run_parameters.txt'
reduced_summaries_name = 'reduced_summaries.pkl.gz'

# helper functions
def log_log_plot(xs, ys, ax=None, **kwargs):
    if ax is None:
        pylab.figure()
        ax = pylab.subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.scatter(xs, ys, **kwargs)
    return ax
#
programmatic_filter = lambda filename: filename.startswith(data_dir_prefix)
numnodes_re = re.compile('^summary_numnodes(\d+)_')
summary_name_to_numnodes = lambda name: int(numnodes_re.match(name).groups()[0])
defaultdict_result = lambda: dict(true_clusters=[], end_clusters=[])

# determine which files
all_dirs = os.listdir(base_dir)
programmatic_dirs = filter(programmatic_filter, all_dirs)
#
results_by_numnodes = defaultdict(defaultdict_result)
for programmatic_dir in programmatic_dirs:
    data_dir = os.path.join(base_dir, programmatic_dir)
    parameters_full_filename = os.path.join(data_dir, parameters_filename)
    if not os.path.isfile(parameters_full_filename): continue
    parameters = dict()
    exec open(parameters_full_filename) in parameters
    reduced_summaries = rf.unpickle(reduced_summaries_name, dir=data_dir)
    #
    true_clusters = parameters['num_clusters']
    for summary_name, summary in reduced_summaries.iteritems():
        numnodes = summary_name_to_numnodes(summary_name)
        end_clusters = summary['num_clusters'][-1]
        results_by_numnodes[numnodes]['end_clusters'].append(end_clusters)
        results_by_numnodes[numnodes]['true_clusters'].append(true_clusters)

series_dict = dict()
for key in results_by_numnodes:
    frame = pandas.DataFrame(results_by_numnodes[key]).set_index('true_clusters')
    series = frame['end_clusters']
    series_dict[key] = series

color_lookup = dict(zip([1, 2, 4], ['red', 'blue', 'green']))
marker_lookup = dict(zip([1, 2, 4], ['+', 'x', 'v']))
jitter_lookup = dict(zip([1, 2, 4], [.98, 1., 1.02]))
ax = None
random_state = numpy.random.RandomState(0)
v_jitter_delta = .02
v_low = 1 - v_jitter_delta
v_high = 1 + v_jitter_delta
for numnodes, series in series_dict.iteritems():
    color = color_lookup[numnodes]
    marker = marker_lookup[numnodes]
    h_jitter = jitter_lookup[numnodes]
    v_jitter = random_state.uniform(low=v_low, high=v_high, size=len(series))
    xs = series.index * h_jitter
    ys = series.values * v_jitter
    ax = log_log_plot(xs, ys, ax=ax, color=color, marker=marker)

legend_list = ['numnodes=' + str(key) for key in series_dict.keys()]
pylab.legend(legend_list)
pylab.xlabel('true num clusters')
pylab.ylabel('last sample num clusters')
pylab.title('Final num clusters comparison for comparable # iters')
pylab.savefig('true_vs_sampled_num_clusters')
