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
# base_dir = '/usr/local/Cloudless/examples/DPMB/Data'
base_dir = '/tmp'
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
    if parameters['beta_d'] > 1.0: continue
    #
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

num_nodes_list = [1, 2, 4]
colors_list = ['red', 'blue', 'green']
markers_list = ['+', 'x', 'v']
h_jitter_fixed_list = [.9, 1., 1.1]
v_jitter_delta = .1
v_low = 1 - v_jitter_delta
v_high = 1 + v_jitter_delta
h_jitter_delta = .1
h_low = 1 - h_jitter_delta
h_high = 1 + h_jitter_delta
#
color_lookup = dict(zip(num_nodes_list, colors_list))
marker_lookup = dict(zip(num_nodes_list, markers_list))
h_jitter_fixed_lookup = dict(zip(num_nodes_list, h_jitter_fixed_list))
#
ax = None
random_state = numpy.random.RandomState(0)
for numnodes, series in series_dict.iteritems():
    n_series = len(series)
    color = color_lookup[numnodes]
    marker = marker_lookup[numnodes]
    h_jitter_fixed = h_jitter_fixed_lookup[numnodes]
    h_jitter_rand = random_state.uniform(low=h_low, high=h_high, size=n_series)
    v_jitter = random_state.uniform(low=v_low, high=v_high, size=n_series)
    xs = series.index * h_jitter_fixed * h_jitter_rand
    ys = series.values * v_jitter
    label = 'numnodes=' + str(numnodes)
    ax = log_log_plot(xs, ys, ax=ax, color=color, marker=marker, label=label)

# show the actual number of clusters
uniq_true_num_clusters = numpy.unique([el for el in series.index])
# line_colors = ['orange', 'yellow', 'brown']
line_colors = ['cyan', 'magenta', 'yellow']
for uniq_idx, true_num_clusters in enumerate(uniq_true_num_clusters):
    color_idx = uniq_idx % len(line_colors)
    color = line_colors[color_idx]
    ax.axvline(true_num_clusters, color=color)
    ax.axhline(true_num_clusters, color=color)

# # add a diagonal line to show where true is
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# pylab.plot(xlim, ylim, color='red')
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)

# add notations
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='upper center', ncol=3,
                bbox_to_anchor=(0.5,-0.1))
pylab.xlabel('true num clusters')
pylab.ylabel('last sample num clusters')
pylab.title('Final num clusters comparison for comparable problems, #iters\n'
            'Vertical and horizontal jitter added\n'
            'True num clusters denoted by proximate vertical line')
pylab.savefig('true_vs_sampled_num_clusters',
              bbox_extra_artists=(lgd,), bbox_inches='tight')
