#!python
import argparse
import os
import re
from multiprocessing import Pool, cpu_count
from collections import Counter
#
import matplotlib
matplotlib.use('Agg')
import pylab
import numpy
import scipy
#
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.settings as S
reload(S)
import Cloudless.examples.DPMB.plot_utils as pu
reload(pu)

# helper functions
is_summary = lambda x : x[:8] == 'summary_'
split_summary_re = re.compile('^(.*iternum)([-\d]*).pkl')
split_summary = lambda x : split_summary_re.match(x).groups()

def get_summary_tuples(data_dir):
    data_files = os.listdir(data_dir)
    summary_files = filter(is_summary,data_files)
    summary_names = {}
    for summary_file in sorted(summary_files):
        summary_name, iter_num_str = split_summary(summary_file)
        iter_num = int(iter_num_str)
        full_filename = os.path.join(data_dir, summary_file)
        summary_tuple = (full_filename, iter_num)
        summary_names.setdefault(summary_name,[]).append(summary_tuple)
    return summary_names

num_workers = cpu_count()
#
def read_tuple(in_tuple):
    full_filename, iter_num = in_tuple
    return rf.unpickle(full_filename), iter_num
#
def get_summaries_dict(summary_names,data_dir):
    summaries_dict = {}
    p = Pool(num_workers)
    for summary_name, tuple_list in summary_names.iteritems():
        result = p.map_async(read_tuple, tuple_list)
        result.wait(60)
        if not result.successful():
            raise Exception('pool not successful')
        sort_by_second_el = lambda x, y: cmp(x[1], y[1])
        sorted_results = sorted(result.get(), cmp=sort_by_second_el)
        sorted_results = map(lambda x: x[0], sorted_results)
        summaries_dict[summary_name] = sorted_results
    p.close()
    p.join()
    return summaries_dict

def process_timing(summaries):
    delta_ts = [0]
    if 'start_time' in summaries[0].get('timing',{}):
        start_time = summaries[0]['timing']['start_time']
        get_total_seconds = lambda summary : \
            (summary['timing']['timestamp'] - start_time).total_seconds()
        for summary in summaries[1:]:
            delta_ts.append('%.2f' % get_total_seconds(summary))
    else:
        for summary in summaries[1:]:
            delta_ts.append('%.2f' % summary['timing']['run_sum'])
    return delta_ts

extract_score = lambda summaries : [
    ('%.2f' % summary['score'])
    for summary in summaries
    ]
extract_log10_alpha = lambda summaries : [
    ('%.2f' % numpy.log10(summary['alpha']))
    for summary in summaries
    ]
extract_beta = lambda summaries : [
    summary['betas']
    for summary in summaries
    ]
extract_num_clusters = lambda summaries : [
    summary['num_clusters']
    for summary in summaries
    ]
extract_test_lls = lambda summaries : [
    ('%.6f' % numpy.mean(summary['test_lls']))
    for summary in summaries
    ]
extract_delta_t = process_timing

def print_info(summaries_dict):
    extract_funcs = [
        ('score', extract_score),
        ('test_lls', extract_test_lls),
        ('delta_t', extract_delta_t),
        ]
    #
    for extract_label, extract_func in extract_funcs:
        print extract_label
        for filename in sorted(summaries_dict.keys()):
            summaries = summaries_dict[filename]
            print filename
            print extract_func(summaries)
        print

shorten_re = re.compile('.*numnodes(\d+)_')
def shorten_name(instr):
    match = shorten_re.match(instr)
    shortened_name = instr
    if match is not None:
        shortened_name = 'nodes=' + match.groups()[0]
    return shortened_name

numnodes_to_color = {'1':'red', '2':'yellow', '4':'green', 'other':'black'}
def get_color(summaries_key):
    summaries_re = re.compile('.*numnodes(\d+)_.*')
    summaries_match = summaries_re.match(summaries_key)
    numnodes_str = None
    if summaries_match is not None:
        numnodes_str = summaries_match.groups()[0]
    else:
        numnodes_str = 'other'
    color = numnodes_to_color[numnodes_str]
    return color
    
def plot_vs_time(summaries_dict, extract_func, new_fig=False, label_func=None, 
                 do_legend=False):
    if new_fig:
        pylab.figure()
    if label_func is None:
        label_func = lambda x: x
    for summaries_name, summaries in summaries_dict.iteritems():
        timing = numpy.array(extract_delta_t(summaries), dtype=float)
        extract_vals = numpy.array(extract_func(summaries), dtype=float)
        color = get_color(summaries_name)
        label = label_func(summaries_name)
        pylab.plot(timing,extract_vals, label=label, color=color)
    if do_legend:
        legend_list = map(label_func, summaries_dict.keys())
        pylab.legend(legend_list, prop={"size":"medium"}) # ,loc='lower right')

def plot_cluster_counts(summary, new_fig=True, log_x=False):
    cluster_counts = summary['cluster_counts']
    cluster_counter = Counter(cluster_counts)
    runsum = 0
    xs = []
    ys = []
    for key in sorted(cluster_counter.keys()):
        value = cluster_counter[key]
        runsum += key * value
        xs.append(key)
        ys.append(runsum)
    #
    ys = numpy.array(ys, dtype=float)
    ys = ys / ys[-1]
    if new_fig:
        pylab.figure()
    if log_x:
        ax = pylab.subplot(111)
        ax.set_xscale('log')
    pylab.plot(xs, ys, linestyle='steps-post')
    pylab.xlabel('cluster_size')
    pylab.ylabel('fraction of data at or below cluster_size')
    pylab.title('data/cluster size distribution')

def read_summaries(data_dirs, do_print=False):
    # read the summaries
    summaries_dict = {}
    for data_dir in data_dirs:
        summary_tuples = get_summary_tuples(data_dir)
        working_summaries_dict = get_summaries_dict(summary_tuples,data_dir)
        if do_print:
            print_info(working_summaries_dict)
        summaries_dict.update(working_summaries_dict)
    # pop off the one_node parent keys
    numnodes1_prefix = 'summary_numnodes1_'
    is_numnode1_parent = lambda filename: \
        filename.startswith(numnodes1_prefix) and filename.find('child') == -1
    all_keys = summaries_dict.keys()
    parent_keys = filter(is_numnode1_parent, all_keys)
    numnodes1_parent_list = []
    for parent_key in parent_keys:
        parent_summaries = summaries_dict.pop(parent_key)
        numnodes1_parent_list.append(parent_summaries)
    #
    return summaries_dict, numnodes1_parent_list

def multiplot(data, plot_tuples, title='', xlabel='', save_str=None,
              subplots_hspace=.25):
    num_tuples = len(plot_tuples)
    gs = pu.get_gridspec(num_tuples)
    fh = pylab.figure()
    #
    for gs_i, extract_tuple in enumerate(plot_tuples):
        plot_func, ylabel = extract_tuple
        last_axes = pylab.subplot(gs[gs_i])
        plot_func(data)
        pylab.ylabel(ylabel)
    pylab.subplots_adjust(hspace=subplots_hspace)
    pu.legend_outside(last_axes)
    first_axes = fh.get_axes()[0]
    first_axes.set_title(title)
    last_axes.set_xlabel(xlabel)
    if save_str is not None:
        pu.savefig_legend_outside(save_str, last_axes)
    return fh

def plot_summaries(summaries_dict, title='', xlabel='', plot_dir=''):
    get_time_plotter = lambda extract_func: \
        (lambda summaries_dict: 
         plot_vs_time(summaries_dict, extract_func, label_func=shorten_name))
    def boxplotter(summaries_dict):
        for values in summaries_dict.values():
            betas = extract_beta(values)
            betas = numpy.array(betas).T
            pylab.boxplot(numpy.log10(betas))
    fh_list = []

    plot_tuples = [
        (get_time_plotter(extract_test_lls), 'test set\nmean log likelihood'),
        (get_time_plotter(extract_score), 'model score'),
        (get_time_plotter(extract_num_clusters), 'num clusters'),
        ]
    figname = 'test_lls_score_num_clusters'
    fig_full_filename = os.path.join(plot_dir, figname)
    #
    fh = multiplot(summaries_dict, plot_tuples,
                   title=title, xlabel=xlabel,
                   save_str=fig_full_filename)
    fh_list.append(fh)
    
    plot_tuples = [
        (get_time_plotter(extract_log10_alpha), 'log10 alpha'),
        (boxplotter, 'log10 beta boxplot'),
        (get_time_plotter(extract_num_clusters), 'num clusters'),
        ]
    figname = 'alpha_beta_num_clusters'
    fig_full_filename = os.path.join(plot_dir, figname)
    #
    fh = multiplot(summaries_dict, plot_tuples,
                   title=title, xlabel=xlabel,
                   save_str=fig_full_filename)
    fh_list.append(fh)

    return fh_list

reduced_summary_extract_func_tuples = [
    ('score', extract_score),
    ('test_lls', extract_test_lls),
    ('delta_t', extract_delta_t),
    ('log10_alpha', extract_log10_alpha),
    ('num_clusters', extract_num_clusters),
    ]
def extract_reduced_summaries(summaries_dict, extract_func_tuples):
    reduced_summaries_dict = dict()
    for summaries_name, summaries in summaries_dict.iteritems():
        reduced_summaries_dict.setdefault(summaries_name, dict())
        for extract_name, extract_func in extract_func_tuples:
            extract_vals = numpy.array(extract_func(summaries),dtype=float)
            reduced_summaries_dict[summaries_name][extract_name] = extract_vals
    return reduced_summaries_dict

def main():
    # parse some args
    parser = argparse.ArgumentParser('consolidate summaries')
    parser.add_argument('data_dirs',nargs='+',type=str)
    args = parser.parse_args()
    data_dirs = args.data_dirs
    #
    summaries_dict, numnodes1_parent_list = read_summaries(data_dirs)
    plot_summaries(summaries_dict)
    return summaries_dict, numnodes1_parent_list

if __name__ == '__main__':
    summaries_dict, numnodes1_parent_list = main()
