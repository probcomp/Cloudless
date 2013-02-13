import argparse
import os
import re
from collections import defaultdict
#
import numpy
import pandas
#
import Cloudless.examples.DPMB.remote_functions as rf
import Cloudless.examples.DPMB.optionally_use_agg as oua
oua.optionally_use_agg()
# import pylab must come after optionally_use_agg
import pylab



# settings
default_fig_suffix = 'pdf'
default_base_dir = '/mnt/'
default_field_of_interest = 'test_lls'
default_min_iternum = 5

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
    return problem, summary, max_iternum

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
                     fig_suffix=default_fig_suffix):
    num_row_key = {1000000:'1MM', 200000:'200K', 500000:'500K'}
    ax = None
    random_state = numpy.random.RandomState(0)
    for (num_clusters, num_rows), series in series_dict.iteritems():
        color = color_lookup[num_clusters]
        marker = marker_lookup[num_rows]
        h_index = h_index_lookup[num_rows]
        xs, ys = series.index, series.values
        xs, ys = jitterify(xs, ys, jitter_range, h_index, random_state,
                           jitter_op)
        # label = 'numnodes=' + str(numnodes)
        num_row_str = num_row_key.get(num_rows, num_rows)
        label = '%s rows x %s clusters' % (num_row_str, int(num_clusters))
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

    num_datapoints = sum(map(len, series_dict.values()))
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    title_list = [
        'Scatter diagram demonstrating correctness',
        'Horizontal jitter added',
        '%s datapoints' % num_datapoints,
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

# parse some args
parser = argparse.ArgumentParser()
parser.add_argument('--fig_suffix', default=default_fig_suffix, type=str)
parser.add_argument('--base_dir', default=default_base_dir, type=str)
parser.add_argument('--field_of_interest', default=default_field_of_interest, type=str)
parser.add_argument('--min_iternum', default=default_min_iternum, type=int)
args = parser.parse_args()
fig_suffix = args.fig_suffix
base_dir = args.base_dir
field_of_interest = args.field_of_interest
min_iternum = args.min_iternum

# proces dir contents
dir_list = os.listdir(base_dir)
dir_list = filter(lambda x: x.startswith('new_'), dir_list)
dir_list = [os.path.join(base_dir, dir) for dir in dir_list]
dir_list = filter(os.path.isdir, dir_list)

gen_and_final_tuples = []
for dir in dir_list:
    summary_filename_tuples = get_summary_filename_tuples(dir)
    dict_by_seed = get_dict_of_lists(summary_filename_tuples, get_seed)
    for seed_summary_filename_tuples in dict_by_seed.itervalues():
        problem, summary, max_iternum = \
            get_problem_and_final_summary(seed_summary_filename_tuples, dir)
        if max_iternum < min_iternum: continue
        if field_of_interest not in summary: continue
        if field_of_interest not in problem: continue
        if 'num_clusters' not in problem: continue
        ground_truth_num_clusters = problem['num_clusters']
        num_vectors = len(problem['true_zs'])
        ground_truth_value = numpy.mean(problem[field_of_interest])
        last_sample_value = numpy.mean(summary[field_of_interest])
        if field_of_interest == 'test_lls' and abs(last_sample_value - ground_truth_value) > 3: continue
        gen_and_final_tuple = (
            ground_truth_value, last_sample_value,
            ground_truth_num_clusters, num_vectors,
            )
        gen_and_final_tuples.append(gen_and_final_tuple)

testlls_xlabel = 'GROUND TRUTH TEST LOG-LIKELIHOODS ASSIGNED FROM HARD-WIRED MODELS'
testlls_ylabel = 'AVERAGE PREDICTIVE LOG-LIKELIHOODS OF LEARNED MODELS'

color_lookup_helper = {128:'red', 512:'blue', 2048:'green'}
marker_lookup_helper = {200000:'+', 500000:'x', 1000000:'v'}
h_index_lookup_helper = {200000:0, 500000:0, 1000000:0}
#
color_lookup = defaultdict(lambda: 'orange')
color_lookup.update(color_lookup_helper)
#
marker_lookup = defaultdict(lambda: 'o')
marker_lookup.update(marker_lookup_helper)
#
h_index_lookup = defaultdict(lambda: 0)
h_index_lookup.update(h_index_lookup_helper)


gen_and_final_tuples = numpy.array(gen_and_final_tuples)
unique_configurations = numpy.unique([(el[0], el[1]) for el in gen_and_final_tuples[:, 2:].tolist()])
series_dict = dict()
for unique_configuration in unique_configurations.tolist():
    is_current_configuration = (gen_and_final_tuples[:, 2] == unique_configuration[0]) \
        & (gen_and_final_tuples[:, 3] == unique_configuration[1])
    tuples_S = pandas.Series(gen_and_final_tuples[is_current_configuration, 1],
                             gen_and_final_tuples[is_current_configuration, 0])
    series_dict[tuple(unique_configuration)] = tuples_S

plot_series_dict(series_dict, field_of_interest, do_log_log=False, jitter_range=.01, jitter_op=operator.mul, 
                 do_lines=False, fig_suffix=fig_suffix)
