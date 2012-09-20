#!python
import argparse
import os
import re
from multiprocessing import Pool, cpu_count
#
import matplotlib
matplotlib.use('Agg')
import pylab
import numpy
import scipy
#
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.settings as settings
reload(settings)
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
    return summaries_dict

def process_timing(summaries):
    delta_ts = [0]
    if 'run_sum' in summaries[0]['timing']:
        for summary in summaries[1:]:
            delta_ts.append('%.2f' % summary['timing']['run_sum'])
    else:
        start_time = summaries[0]['timing']['start_time']
        get_total_seconds = lambda summary : \
            (summary['timing']['timestamp'] - start_time).total_seconds()
        for summary in summaries[1:]:
            delta_ts.append('%.2f' % get_total_seconds(summary))
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
#
extract_funcs = [
    ('score', extract_score),
    ('test_lls', extract_test_lls),
    ('delta_t', extract_delta_t),
    ]

def print_info(summaries_dict):
    for extract_label, extract_func in extract_funcs:
        print extract_label
        for filename in sorted(summaries_dict.keys()):
            summaries = summaries_dict[filename]
            print filename
            print extract_func(summaries)
        print

shorten_name = lambda x: x[8:-8]
def plot_vs_time(summaries_dict, extract_func, new_fig=True, do_legend=True):
    if new_fig:
        pylab.figure()
    for summaries_name, summaries in summaries_dict.iteritems():
        timing = numpy.array(extract_delta_t(summaries),dtype=float)
        extract_vals = numpy.array(extract_func(summaries),dtype=float)
        pylab.plot(timing,extract_vals)
    legend_list = map(shorten_name,summaries_dict.keys())
    if do_legend:
        pylab.legend(legend_list,prop={"size":"medium"}) # ,loc='lower right')
    
def main():

    # parse some args
    parser = argparse.ArgumentParser('consolidate summaries')
    parser.add_argument('data_dirs',nargs='+',type=str)
    args = parser.parse_args()
    data_dirs = args.data_dirs

    # read the summaries
    summaries_dict = {}
    for data_dir in data_dirs:
        summary_tuples = get_summary_tuples(data_dir)
        working_summaries_dict = get_summaries_dict(summary_tuples,data_dir)
        print_info(working_summaries_dict)
        summaries_dict.update(working_summaries_dict)

    # pop off the one_node parent key
    one_node_children_key = 'summary_numnodes1_seed1_iternum'
    numnodes1_seed1 = None
    if one_node_children_key in summaries_dict:
        numnodes1_seed1 = summaries_dict.pop(one_node_children_key)
        
    gs = pu.get_gridspec(3)
    subplots_hspace = .25

    gs_i = 0
    pylab.figure()
    #
    # create test_lls plot
    pylab.subplot(gs[gs_i])
    gs_i += 1
    plot_vs_time(
        summaries_dict, extract_test_lls, new_fig=False, do_legend=False)
    pylab.ylabel('test set\nmean log likelihood')
    #
    # create score plot
    pylab.subplot(gs[gs_i])
    gs_i += 1
    plot_vs_time(summaries_dict, extract_score, new_fig=False, do_legend=False)
    pylab.ylabel('model score')
    #
    # create num_clusters plot
    pylab.subplot(gs[gs_i])
    gs_i += 1
    plot_vs_time(summaries_dict, extract_num_clusters, new_fig=False)
    pylab.ylabel('num clusters')
    #
    pylab.xlabel('time (seconds)')
    pylab.subplots_adjust(hspace=subplots_hspace)
    pylab.savefig('test_lls_score_num_clusters')

    gs_i = 0
    pylab.figure()
    # create alpha plot
    pylab.subplot(gs[gs_i])
    gs_i += 1
    plot_vs_time(
        summaries_dict, extract_log10_alpha, new_fig=False, do_legend=False)
    pylab.ylabel('log10 alpha')
    #
    # betas distribution
    pylab.subplot(gs[gs_i])
    gs_i += 1
    for values in summaries_dict.values():
        betas = extract_beta(values)
        betas = numpy.array(betas).T
        pylab.boxplot(numpy.log10(betas))
    pylab.title('beta boxplot')
    pylab.ylabel('log10 beta')
    #
    # create num_clusters plot
    pylab.subplot(gs[gs_i])
    gs_i += 1
    plot_vs_time(summaries_dict, extract_num_clusters, new_fig=False)
    pylab.ylabel('num clusters')
    #
    pylab.xlabel('time (seconds)')
    pylab.subplots_adjust(hspace=subplots_hspace)
    pylab.savefig('alpha_beta_num_clusters')

    return summaries_dict, numnodes1_seed1

if __name__ == '__main__':
    summaries_dict, numnodes1_seed1 = main()
