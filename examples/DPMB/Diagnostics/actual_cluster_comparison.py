# generate synthetic data
import Cloudless.examples.DPMB.optionally_use_agg as oua
reload(oua)
oua.optionally_use_agg()
#
import os
import re
import argparse
from collections import defaultdict
#
import numpy
import pylab
import pandas
#
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)


default_base_dir = '/usr/local/Cloudless/examples/DPMB/Data/20120928_programmatic_mrjob'
parser = argparse.ArgumentParser()
parser.add_argument('--dir', default=default_base_dir, type=str)
args = parser.parse_args()

# some settings
base_dir = args.dir
# base_dir = '/tmp'
data_dir_prefix = 'programmatic_mrjob_'
parameters_filename = 'run_parameters.txt'
problem_filename = 'problem.pkl.gz'
reduced_summaries_name = 'reduced_summaries.pkl.gz'

# helper functions
def my_plot(xs, ys, ax=None, do_log_log=True, **kwargs):
    if ax is None:
        pylab.figure()
        ax = pylab.subplot(111)
        if do_log_log:
            ax.set_xscale('log')
            ax.set_yscale('log')
    ax.scatter(xs, ys, **kwargs)
    return ax
#
programmatic_filter = lambda filename: filename.startswith(data_dir_prefix)
numnodes_re = re.compile('^summary_numnodes(\d+)_')
summary_name_to_numnodes = lambda name: int(numnodes_re.match(name).groups()[0])
defaultdict_result = lambda: dict(true=[], end=[])
strify = lambda num_list: ','.join([str(el) for el in num_list])
intify = lambda num_list: [int(el) for el in num_list]
#
def update_parameters_set_dict(data_dir, parameters_set_dict):
    parameters_full_filename = os.path.join(data_dir, parameters_filename)
    if not os.path.isfile(parameters_full_filename): return None
    parameters = dict()
    exec open(parameters_full_filename) in parameters
    if parameters['beta_d'] > 1.0: return None
    for field in parameters:
        if field == '__builtins__': continue
        parameter_value = parameters[field]
        try:
            parameters_set_dict[field].add(parameter_value)
        except Exception, e:
            pass
    return parameters
#
def extract_series_dict(end_true_dict):
    series_dict = dict()
    for key, value in end_true_dict.iteritems():
        frame = pandas.DataFrame(value).set_index('true')
        series = frame['end']
        series_dict[key] = series
    return series_dict

# determine which files
all_dirs = os.listdir(base_dir)
programmatic_dirs = filter(programmatic_filter, all_dirs)
#
fieldnames = ['num_rows', 'num_clusters', 'beta_d']
parameters_set_dict = defaultdict(set)
clusters_by_numnodes = defaultdict(defaultdict_result)
testlls_by_numnodes = defaultdict(defaultdict_result)
for programmatic_dir in programmatic_dirs:
    data_dir = os.path.join(base_dir, programmatic_dir)
    #
    parameters = update_parameters_set_dict(data_dir, parameters_set_dict)
    if parameters is None: continue
    reduced_summaries = rf.unpickle(reduced_summaries_name, dir=data_dir)
    problem = rf.unpickle(problem_filename, dir=data_dir)
    #
    true_clusters = parameters['num_clusters']
    true_testlls = numpy.mean(problem['test_lls'])
    for summary_name, summary in reduced_summaries.iteritems():
        numnodes = summary_name_to_numnodes(summary_name)
        #
        end_clusters = summary['num_clusters'][-1]
        clusters_by_numnodes[numnodes]['end'].append(end_clusters)
        clusters_by_numnodes[numnodes]['true'].append(true_clusters)
        #
        end_testlls = numpy.mean(summary['test_lls'][-1])
        testlls_by_numnodes[numnodes]['end'].append(end_testlls)
        testlls_by_numnodes[numnodes]['true'].append(true_testlls)

uniq_rows = sorted(list(parameters_set_dict['num_rows']))
uniq_clusters = sorted(list(parameters_set_dict['num_clusters']))
uniq_betas = sorted(list(parameters_set_dict['beta_d']))
problem_size_list = [
    'rows=2**{' + strify(intify(numpy.log2(uniq_rows))) + '}',
    'clusters=2**{' + strify(intify(numpy.log2(uniq_clusters))) + '}',
    'beta_d={' + strify(uniq_betas) + '}',
    ]
problem_size = ' x '.join(problem_size_list)

cluster_series_dict = extract_series_dict(clusters_by_numnodes)
testlls_series_dict = extract_series_dict(testlls_by_numnodes)

num_nodes_list = [1, 2, 4]
colors_list = ['red', 'blue', 'green']
markers_list = ['+', 'x', 'v']
h_index_list = [-1, 0, 1]
color_lookup = dict(zip(num_nodes_list, colors_list))
marker_lookup = dict(zip(num_nodes_list, markers_list))
h_index_lookup = dict(zip(num_nodes_list, h_index_list))

import operator
# operator.mul, operator.add
def jitterify(xs, ys, jitter_range, h_index, random_state,
              jitter_op=operator.mul):
    low = 1 - jitter_range
    high = 1 + jitter_range
    h_jitter_fixed = jitter_range * h_index
    if jitter_op == operator.mul:
        h_jitter_fixed = 1.0 + jitter_range * h_index
    size = len(xs)
    h_jitter_rand = random_state.uniform(low=low, high=high, size=size)
    v_jitter_rand = random_state.uniform(low=low, high=high, size=size)
    #
    xs = jitter_op(xs, h_jitter_fixed)
    xs = jitter_op(xs, h_jitter_rand)
    yx = jitter_op(ys, v_jitter_rand)
    #
    return xs, ys

def plot_series_dict(series_dict, series_name, do_log_log,
                     jitter_range, jitter_op, do_lines=True,
                     xlabel=None, ylabel=None):
    ax = None
    random_state = numpy.random.RandomState(0)
    for numnodes, series in series_dict.iteritems():
        color = color_lookup[numnodes]
        marker = marker_lookup[numnodes]
        h_index = h_index_lookup[numnodes]
        xs, ys = series.index, series.values
        xs, ys = jitterify(xs, ys, jitter_range, h_index, random_state,
                           jitter_op)
        label = 'numnodes=' + str(numnodes)
        ax = my_plot(xs, ys, ax=ax, color=color, marker=marker, label=label,
                     do_log_log=do_log_log, alpha=0.5)

    if do_lines:
        # show the actual number of clusters
        uniq_true_values = numpy.unique([el for el in series.index])
        # line_colors = ['orange', 'yellow', 'brown']
        line_colors = ['cyan', 'magenta', 'yellow']
        for uniq_idx, true_values in enumerate(uniq_true_values):
            color_idx = uniq_idx % len(line_colors)
            color = line_colors[color_idx]
            ax.axvline(true_values, color=color)
            ax.axhline(true_values, color=color)

    # add notations
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', ncol=3,
                    bbox_to_anchor=(0.5,-0.1))
    if xlabel is None:
        xlabel = 'true ' + series_name
    if ylabel is None:
        ylabel = 'last sample ' + series_name
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    title_list = [
        # 'Vertical and horizontal jitter added to datapoints',
        ]
    if do_lines:
        title_list.append('True values denoted by proximate vertical line')
    else:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot(xlim, ylim)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    title = '\n'.join(title_list)
    pylab.title(title)
    pylab.savefig('true_vs_sampled_' + series_name + '.pdf',
                  bbox_extra_artists=(lgd,), bbox_inches='tight',
                  )

testlls_xlabel = 'Ground truth test log-likelihoods assigned from hard-wired models'
testlls_ylabel = 'Average predictive log-likelihoods of learned models'

series_tuples = [
    (cluster_series_dict, 'num_clusters', True, .1, operator.mul, True, testlls_xlabel, testlls_xlabel),
    (testlls_series_dict, 'test_lls', False, 2, operator.add, False),
    ]
#
for series_tuple in series_tuples:
    plot_series_dict(*series_tuple)
