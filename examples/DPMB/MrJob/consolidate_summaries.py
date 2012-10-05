#!python
import argparse
import os
import re
from multiprocessing import Pool, cpu_count
from collections import Counter, defaultdict
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
    #
    gibbs_init_filename = S.files.gibbs_init_filename
    gibbs_init_full_filename = os.path.join(data_dir, gibbs_init_filename)
    defaultfactory = list
    # if os.path.isfile(gibbs_init_full_filename):
    #     defaultfactory = lambda: list(((gibbs_init_full_filename, -1),))
    summary_names_dict = defaultdict(defaultfactory)
    for summary_file in sorted(summary_files):
        summary_name, iter_num_str = split_summary(summary_file)
        iter_num = int(iter_num_str)
        full_filename = os.path.join(data_dir, summary_file)
        summary_tuple = (full_filename, iter_num)
        summary_names_dict[summary_name].append(summary_tuple)
    return summary_names_dict

num_workers = cpu_count()
#
def read_tuple(in_tuple):
    full_filename, iter_num = in_tuple
    return rf.unpickle(full_filename), iter_num
#
def get_summaries_dict(summary_names, data_dir):
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
    delta_ts = []
    if 'start_time' in summaries[0].get('timing',{}):
        start_time = summaries[0]['timing']['start_time']
        delta_ts.append(summaries[0]['timing']['run_sum'])
        get_total_seconds = lambda summary : \
            (summary['timing']['timestamp'] - start_time).total_seconds()
        for summary in summaries[1:]:
            delta_ts.append('%.2f' % get_total_seconds(summary))
    else:
        for summary in summaries:
            delta_ts.append('%.2f' % summary['timing']['run_sum'])
    return delta_ts

get_format_extract = lambda field, formatter: \
    (lambda summaries: map(formatter, [summary[field] for summary in summaries]))
extract_score = get_format_extract('score', None)
extract_ari = get_format_extract('ari', None)
extract_log10_alpha = get_format_extract('alpha', numpy.log10)
extract_beta = get_format_extract('betas', None)
extract_log10_beta = get_format_extract('betas', numpy.log10)
extract_num_clusters = get_format_extract('num_clusters', None)
extract_test_lls = get_format_extract('test_lls', numpy.mean)
extract_delta_t = process_timing

def print_info(summaries_dict):
    extract_funcs = [
        ('score', extract_score),
        ('ari', extract_ari),
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
                 hline=None, do_legend=False, alpha=0.8):
    if new_fig:
        pylab.figure()
    if label_func is None:
        label_func = lambda x: x
    for summaries_name, summaries in summaries_dict.iteritems():
        timing = numpy.array(extract_delta_t(summaries), dtype=float)
        extract_vals = numpy.array(extract_func(summaries), dtype=float)
        color = get_color(summaries_name)
        label = label_func(summaries_name)
        pylab.plot(timing,extract_vals, label=label, color=color, alpha=alpha)
    if hline is not None:
        pylab.axhline(hline, color='magenta', label='gen')
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
        working_summaries_dict = get_summaries_dict(summary_tuples, data_dir)
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

parameters_filename = 'run_parameters.txt'
def read_run_parameters(data_dir, parameters_filename=parameters_filename):
    parameters_full_filename = os.path.join(data_dir, parameters_filename) 
    parameters = dict()
    if not os.path.isfile(parameters_full_filename):
        print parameters_full_filename, ' doesn\' exist!'
        return None
    with open(parameters_full_filename) as fh:
        exec fh in parameters
    return parameters

parameters_of_interest = ['beta_d', 'num_clusters', 'num_rows']
def title_from_parameters(parameters,
                          parameters_of_interest=parameters_of_interest):
    title_els = []
    for parameter_name in parameters_of_interest:
        parameter_value = str(parameters[parameter_name])
        title_el = parameter_name + '=' + str(parameter_value)
        title_els.append(title_el)
    title = '; '.join(title_els)
    return title

def get_time_plotter(extract_func, **kwargs):
    plot_func = lambda summaries_dict: \
        (plot_vs_time(summaries_dict, extract_func, label_func=shorten_name,
                      **kwargs))
    return plot_func
def plot_summaries(summaries_dict, problem=None,
                   title='', xlabel='', plot_dir=''):
    fh_list = []
    gen_test_lls, gen_score, gen_beta, true_num_clusters = None, None, None, None
    if problem is not None:
        gen_test_lls = numpy.mean(problem['test_lls'])
        gen_score = problem['gen_score']
        gen_beta = problem['beta_d']
        true_num_clusters = problem['num_clusters']

    figname = 'test_lls_score'
    plot_tuples = [
        (get_time_plotter(extract_test_lls, hline=gen_test_lls),
         'test set\nmean log likelihood'),
        (get_time_plotter(extract_score, hline=gen_score),
         'model score'),
        ]
    fig_full_filename = os.path.join(plot_dir, figname)
    fh = pu.multiplot(summaries_dict, plot_tuples,
                   title=title, xlabel=xlabel,
                   save_str=fig_full_filename)
    fh_list.append(fh)

    figname = 'ari_score'
    plot_tuples = [
        (get_time_plotter(extract_ari, hline=1.0),
         'ari'),
        (get_time_plotter(extract_score, hline=gen_score),
         'model score'),
        ]
    fig_full_filename = os.path.join(plot_dir, figname)
    fh = pu.multiplot(summaries_dict, plot_tuples,
                   title=title, xlabel=xlabel,
                   save_str=fig_full_filename)
    fh_list.append(fh)
    
    figname = 'alpha_num_clusters'
    plot_tuples = [
        (get_time_plotter(extract_log10_alpha),
         'log10 alpha'),
        (get_time_plotter(extract_num_clusters, hline=true_num_clusters),
         'num clusters'),
        ]
    fig_full_filename = os.path.join(plot_dir, figname)
    fh = pu.multiplot(summaries_dict, plot_tuples,
                   title=title, xlabel=xlabel,
                   save_str=fig_full_filename)
    fh_list.append(fh)

    figname = 'beta_num_clusters'
    plot_tuples = [
        (get_time_plotter(extract_log10_beta, hline=gen_beta, alpha=0.2),
         'log10 beta'),
        (get_time_plotter(extract_num_clusters, hline=true_num_clusters),
         'num clusters'),
        ]
    fig_full_filename = os.path.join(plot_dir, figname)
    fh = pu.multiplot(summaries_dict, plot_tuples,
                   title=title, xlabel=xlabel,
                   save_str=fig_full_filename)
    fh_list.append(fh)

    return fh_list

reduced_summary_extract_func_tuples = [
    ('score', extract_score),
    ('ari', extract_ari),
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
    problem_file = 'problem.pkl.gz'
    parameters_file = 'run_parameters.txt'
    for data_dir in data_dirs:
        problem_full_file = os.path.join(data_dir, problem_file)
        problem = None
        if os.path.isfile(problem_full_file):
            problem = rf.unpickle(problem_file, dir=data_dir)
        parameters_full_file = os.path.join(data_dir, parameters_file)
        title = ''
        if os.path.isfile(parameters_full_file):
            parameters = dict()
            with open(parameters_full_file) as fh:
                exec fh in parameters
            title = title_from_parameters(parameters)
        summaries_dict, numnodes1_parent_list = read_summaries([data_dir])
        plot_summaries(summaries_dict, problem=problem, title=title)
    return summaries_dict, numnodes1_parent_list

if __name__ == '__main__':
    summaries_dict, numnodes1_parent_list = main()
