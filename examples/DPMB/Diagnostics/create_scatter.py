import argparse
import os
import re
from collections import defaultdict
#
import numpy
#
import Cloudless.examples.DPMB.remote_functions as rf



default_fig_suffix = 'pdf'
parser = argparse.ArgumentParser()
parser.add_argument('--fig_suffix',default=default_fig_suffix,type=str)
args = parser.parse_args()
fig_suffix = args.fig_suffix

# settings
base_dir = '/mnt'
dir_list = filter(lambda x: x.startswith('new_'), os.listdir(base_dir))
dir_list = filter(lambda x: not x.endswith('.png'), os.listdir(base_dir))
dir_list = [os.path.join(base_dir, dir) for dir in dir_list]
field_of_interst = "NONE"

# helper functions
get_filename = lambda tuple: tuple[0]
get_seed = lambda tuple: tuple[1]
get_iternum = lambda tuple: tuple[2]
#
summary_filename_tuple_re = re.compile('^score.*seed(\d+).*iternum(\d+).pkl.gz')
def get_summary_filename_tuple(filename):
    match = summary_filename_tuple_re.match(filename)
    ret_list = None
    if match:
        ret_list = [filename]
        ret_list.extend(map(int, match.groups()))
    return ret_list
#
def get_summary_filename_tuples(dir):
    filenames = os.listdir(dir)
    summary_filename_tuples = map(get_summary_filename_tuple, filenames)
    summary_filename_tuples = filter(None, summary_filename_tuples)
    return summary_filename_tuples
#
def get_dict_of_lists(in_list, dict_by):
    dict_of_lists = defaultdict(list)
    for el in in_list:
        key = dict_by(el)
        dict_of_lists[key].append(el)
    return dict_of_lists
#
def get_problem_and_final_summary(seed_summary_filename_tuples, dir=''):
    max_iternum = max(map(get_iternum, seed_summary_filename_tuples))
    is_max = lambda tuple: get_iternum(tuple)==max_iternum
    max_iternum_filename_tuple = \
        filter(is_max, seed_summary_filename_tuples)[0]
    print max_iternum_filename_tuple
    max_iternum_filename = get_filename(max_iternum_filename_tuple)
    problem = rf.unpickle('problem.pkl.gz', dir=dir, check_hdf5=False)
    summary = rf.unpickle(max_iternum_filename, dir=dir)
    return problem, summary

fieldname = 'test_lls'
# fieldname = 'num_clusters'
gen_and_final_tuples = []
for dir in dir_list:
    summary_filename_tuples = get_summary_filename_tuples(dir)
    dict_by_seed = get_dict_of_lists(summary_filename_tuples, get_seed)
    for seed_summary_filename_tuples in dict_by_seed.itervalues():
        problem, summary = \
            get_problem_and_final_summary(seed_summary_filename_tuples, dir)
        gen_and_final_tuple = (
            numpy.mean(problem[fieldname]), numpy.mean(summary[fieldname])
            )
        gen_and_final_tuples.append(gen_and_final_tuple)

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
#
def plot_series_dict(series_dict, series_name, do_log_log,
                     jitter_range, jitter_op, do_lines=True,
                     xlabel=None, ylabel=None,
                     fig_suffix=fig_suffix):
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
        min_extent = min(xlim[0], ylim[0])
        max_extent = max(xlim[1], ylim[1])
        new_lim = (min_extent, max_extent)
        ax.plot(new_lim, new_lim)
        ax.set_xlim(*new_lim)
        ax.set_ylim(*new_lim)

    title = '\n'.join(title_list)
    pylab.title(title)
    pylab.savefig('true_vs_sampled_' + series_name + '.' + fig_suffix,
                  bbox_extra_artists=(lgd,), bbox_inches='tight',
                  )

testlls_xlabel = 'GROUND TRUTH TEST LOG-LIKELIHOODS ASSIGNED FROM HARD-WIRED MODELS'
testlls_ylabel = 'AVERAGE PREDICTIVE LOG-LIKELIHOODS OF LEARNED MODELS'

num_nodes_list = [4,8,16]
colors_list = ['red', 'blue', 'green']
markers_list = ['+', 'x', 'v']
# h_index_list = [-1, 0, 1]
h_index_list = [0, 0, 0]
color_lookup = dict(zip(num_nodes_list, colors_list))
marker_lookup = dict(zip(num_nodes_list, markers_list))
h_index_lookup = dict(zip(num_nodes_list, h_index_list))


import Cloudless.examples.DPMB.optionally_use_agg as oua
oua.optionally_use_agg()
import pylab
import pandas
gen_and_final_tuples = numpy.array(gen_and_final_tuples)
tuples_S = pandas.Series(gen_and_final_tuples[:,1], gen_and_final_tuples[:,0])
series_dict = {8:tuples_S}
plot_series_dict(series_dict,'test',do_log_log=False,jitter_range=.01,jitter_op=operator.mul,do_lines=False,
                 fig_suffix=fig_suffix)
