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
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB.DPMB as dm
reload(dm)
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless.examples.DPMB.settings as S
reload(S)
import Cloudless.examples.DPMB.plot_utils as pu
reload(pu)


# helper functions
is_summary = lambda x : x[:8] == 'summary_'
split_summary_re = re.compile('^(.*iternum)([-\d]*).pkl')
split_summary = lambda x : split_summary_re.match(x).groups()

default_init_filename = S.files.gibbs_init_filename
def get_summary_tuples(data_dir, init_filename=default_init_filename):
    data_files = os.listdir(data_dir)
    summary_files = filter(is_summary,data_files)
    #
    init_full_filename = os.path.join(data_dir, init_filename)
    defaultfactory = list
    if os.path.isfile(init_full_filename):
        defaultfactory = lambda: list(((init_full_filename, -1),))
    summary_names_dict = defaultdict(defaultfactory)
    for summary_file in sorted(summary_files):
        summary_name, iter_num_str = split_summary(summary_file)
        iter_num = int(iter_num_str)
        if iter_num < 0: continue
        full_filename = os.path.join(data_dir, summary_file)
        summary_tuple = (full_filename, iter_num)
        summary_names_dict[summary_name].append(summary_tuple)
    return summary_names_dict

num_workers = cpu_count() / 2
#
def read_tuple(in_tuple):
    full_filename, iter_num = in_tuple
    summary = rf.unpickle(full_filename)
    return summary, iter_num
#
def score_tuple(in_tuple):
    problem = in_tuple[-1]
    summary, iter_num = read_tuple(in_tuple[:-1])
    if problem is not None:
        scored_summary = score_summary(summary, problem)
        field_list = ['ari', 'test_lls', 'score']
        for field in field_list:
            summary[field] = scored_summary[field]
    return summary, iter_num
def get_summaries_dict(summary_names, data_dir, problem_filename=None):
    problem = None
    if problem_filename is not None:
        problem = rf.unpickle(problem_filename, dir=data_dir)
    summaries_dict = {}
    p = Pool(num_workers)
    for summary_name, tuple_list in summary_names.iteritems():
        result = None
        if problem is not None:
            append_problem = lambda in_tuple: \
                tuple(numpy.append(in_tuple, problem))
            new_tuple_list = map(append_problem, tuple_list)
            result = p.map_async(score_tuple, new_tuple_list)
        else:
            result = p.map_async(read_tuple, tuple_list)
        result.wait(180)
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
    # perhaps I can always sum alpha, beta, zs to get delta?
    sum_parts = lambda summary: \
        sum([summary['timing'][field] for field in ['alpha', 'betas', 'zs']])
    delta_ts = []
    if 'iter_start_dt' in summaries[0].get('timing', {}):
        start_dts = [
            summary['timing']['iter_start_dt']
            for summary in summaries
            ]
        end_dts = [
            summary['timing']['iter_end_dt']
            for summary in summaries
            ]
        delta_ts = [
            (end - start).total_seconds()
            for end, start in zip(end_dts, start_dts)
            ]
        delta_ts = numpy.cumsum(delta_ts)
    elif 'start_time' in summaries[0].get('timing',{}):
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

shorten_re = re.compile('.*numnodes(\d+)_.*he(\d+)')
def shorten_name(instr):
    match = shorten_re.match(instr)
    shortened_name = instr
    if match is not None:
        num_nodes_str = match.groups()[0]
        he_str = match.groups()[1]
        shortened_name = 'nodes=' + num_nodes_str + '_' + 'he=' + he_str
    return shortened_name

numnodes_to_color = {'1':'blue', '2':'orange', '4':'green', '8':'red', '16':'brown', 'other':'black'}
def get_color(summaries_key):
    summaries_re = re.compile('.*numnodes(\d+)_.*')
    summaries_match = summaries_re.match(summaries_key)
    numnodes_str = 'other'
    if summaries_match is not None:
        numnodes_str = summaries_match.groups()[0]
    color = numnodes_to_color.get(numnodes_str, 'black')
    return color

he_to_style = {
    '1' : '-',
    '2' : ':',
    '4' : '-.',
    '8' : '--',
    'other' : '-'}
def get_style(summaries_key):
    summaries_re = re.compile('.*he(\d+)_.*')
    summaries_match = summaries_re.match(summaries_key)
    he_str = 'other'
    if summaries_match is not None:
        he_str = summaries_match.groups()[0]
    style = he_to_style.get(he_str, '--')
    return style
    
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
        style = get_style(summaries_name)
        label = label_func(summaries_name)
        pylab.plot(timing, extract_vals, label=label, color=color,
                   linestyle=style, alpha=alpha)
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

def score_summary(summary, problem):
    # pull out parameters
    true_zs = problem['true_zs']
    test_xs = problem['test_xs']
    init_x = problem['xs']
    num_rows = len(init_x)
    num_cols = len(init_x[0])
    #
    init_alpha = summary['alpha']
    init_betas = summary['betas']
    init_z = summary.get('zs')
    if init_z is None:
        list_of_x_indices = summary.get('list_of_x_indices',
                                        summary.get('last_valid_list_of_x_indices'))
        #
        # post process raw summary data
        uncannonicalized_zs = numpy.ndarray((num_rows,), dtype=int)
        for cluster_idx, cluster_x_indices in enumerate(list_of_x_indices):
            for x_index in cluster_x_indices:
                uncannonicalized_zs[x_index] = cluster_idx
        init_z, other = hf.canonicalize_list(uncannonicalized_zs)
    #
    # create a state
    state = ds.DPMB_State(gen_seed=0,
                          num_cols=num_cols,
                          num_rows=num_rows,
                          init_alpha=init_alpha,
                          init_betas=init_betas,
                          init_z=init_z,
                          init_x=init_x,
                          )
    transitioner = dm.DPMB(0, state, False, False)
    scored_summary = transitioner.extract_state_summary(
        true_zs=true_zs, test_xs=test_xs)
    return scored_summary

def read_summaries(data_dirs, init_filename=None, problem_filename=None, 
                   do_print=False):
    # read the summaries
    summaries_dict = {}
    for data_dir in data_dirs:
        summary_tuples = get_summary_tuples(
            data_dir,
            init_filename=init_filename,
            )
        working_summaries_dict = get_summaries_dict(
            summary_tuples, data_dir, problem_filename=problem_filename)
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
    parameters.pop('__builtins__')
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
                   title='', xlabel='TIME (SECONDS)', plot_dir='',
                   subset_betas=True):
    fh_list = []
    gen_test_lls, gen_score, gen_beta, true_num_clusters = None, None, None, None
    if problem is not None:
        gen_test_lls = numpy.mean(problem['test_lls'])
        gen_score = problem['gen_score']
        gen_beta = problem['beta_d']
        true_num_clusters = problem['num_clusters']

    figname = 'test_lls.pdf'
    plot_tuples = [
        (get_time_plotter(extract_test_lls, hline=gen_test_lls),
         'TEST SET\nMEAN LOG LIKELIHOOD'),
        ]
    fig_full_filename = os.path.join(plot_dir, figname)
    fh = pu.multiplot(summaries_dict, plot_tuples,
                   title=title, xlabel=xlabel,
                   save_str=fig_full_filename)
    fh_list.append(fh)

    figname = 'score.pdf'
    plot_tuples = [
        (get_time_plotter(extract_score, hline=gen_score),
         'MODEL SCORE'),
        ]
    fig_full_filename = os.path.join(plot_dir, figname)
    fh = pu.multiplot(summaries_dict, plot_tuples,
                   title=title, xlabel=xlabel,
                   save_str=fig_full_filename)
    fh_list.append(fh)

    figname = 'ari.pdf'
    plot_tuples = [
        (get_time_plotter(extract_ari, hline=1.0),
         'ARI'),
        ]
    fig_full_filename = os.path.join(plot_dir, figname)
    fh = pu.multiplot(summaries_dict, plot_tuples,
                   title=title, xlabel=xlabel,
                   save_str=fig_full_filename)
    fh_list.append(fh)

    figname = 'alpha.pdf'
    plot_tuples = [
        (get_time_plotter(extract_log10_alpha),
         'LOG10 ALPHA'),
        ]
    fig_full_filename = os.path.join(plot_dir, figname)
    fh = pu.multiplot(summaries_dict, plot_tuples,
                   title=title, xlabel=xlabel,
                   save_str=fig_full_filename)
    fh_list.append(fh)

    figname = 'num_clusters.pdf'
    plot_tuples = [
        (get_time_plotter(extract_num_clusters, hline=true_num_clusters),
         'NUM CLUSTERS'),
        ]
    fig_full_filename = os.path.join(plot_dir, figname)
    fh = pu.multiplot(summaries_dict, plot_tuples,
                   title=title, xlabel=xlabel,
                   save_str=fig_full_filename)
    fh_list.append(fh)

    figname = 'test_lls_score.pdf'
    plot_tuples = [
        (get_time_plotter(extract_test_lls, hline=gen_test_lls),
         'TEST SET\nMEAN LOG LIKELIHOOD'),
        (get_time_plotter(extract_score, hline=gen_score),
         'MODEL SCORE'),
        ]
    fig_full_filename = os.path.join(plot_dir, figname)
    fh = pu.multiplot(summaries_dict, plot_tuples,
                   title=title, xlabel=xlabel,
                   save_str=fig_full_filename)
    fh_list.append(fh)

    figname = 'ari_score.pdf'
    plot_tuples = [
        (get_time_plotter(extract_ari, hline=1.0),
         'ARI'),
        (get_time_plotter(extract_score, hline=gen_score),
         'MODEL SCORE'),
        ]
    fig_full_filename = os.path.join(plot_dir, figname)
    fh = pu.multiplot(summaries_dict, plot_tuples,
                   title=title, xlabel=xlabel,
                   save_str=fig_full_filename)
    fh_list.append(fh)
    
    figname = 'alpha_num_clusters.pdf'
    plot_tuples = [
        (get_time_plotter(extract_log10_alpha),
         'LOG10 ALPHA'),
        (get_time_plotter(extract_num_clusters, hline=true_num_clusters),
         'NUM CLUSTERS'),
        ]
    fig_full_filename = os.path.join(plot_dir, figname)
    fh = pu.multiplot(summaries_dict, plot_tuples,
                   title=title, xlabel=xlabel,
                   save_str=fig_full_filename)
    fh_list.append(fh)

    if subset_betas:
        def get_subset(betas, num_betas=4, seed=0):
            rs = numpy.random.RandomState(seed)
            which_indices = rs.permutation(range(len(betas)))[:num_betas]
            return map(lambda x: x[which_indices], betas)
        new_extract_log10_beta = lambda summaries: get_subset(extract_log10_beta(summaries))
    else:
        new_extract_log10_beta = extract_log10_beta

    figname = 'beta.pdf'
    plot_tuples = [
        (get_time_plotter(new_extract_log10_beta, hline=gen_beta, alpha=0.2),
         'LOG10 BETA'),
        ]
    fig_full_filename = os.path.join(plot_dir, figname)
    fh = pu.multiplot(summaries_dict, plot_tuples,
                   title=title, xlabel=xlabel,
                   save_str=fig_full_filename)
    fh_list.append(fh)

    figname = 'beta_num_clusters.pdf'
    plot_tuples = [
        (get_time_plotter(new_extract_log10_beta, hline=gen_beta, alpha=0.2),
         'LOG10 BETA'),
        (get_time_plotter(extract_num_clusters, hline=true_num_clusters),
         'NUM CLUSTERS'),
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

if __name__ == '__main__':
    # parse some args
    parser = argparse.ArgumentParser('consolidate summaries')
    parser.add_argument('data_dirs',nargs='+',type=str)
    parser.add_argument('--init_filename',default=default_init_filename,type=str)
    args = parser.parse_args()
    data_dirs = args.data_dirs
    init_filename = args.init_filename
    #
    problem_filename = 'problem.pkl.gz'
    parameters_file = 'run_parameters.txt'

    for data_dir in data_dirs:
        problem_full_filename = os.path.join(data_dir, problem_filename)
        problem = None
        if os.path.isfile(problem_full_filename):
            problem = rf.unpickle(problem_filename, dir=data_dir)
        parameters_full_file = os.path.join(data_dir, parameters_file)
        title = ''
        if os.path.isfile(parameters_full_file):
            parameters = dict()
            with open(parameters_full_file) as fh:
                exec fh in parameters
            title = title_from_parameters(parameters)
        summaries_dict, numnodes1_parent_list = read_summaries(
            [data_dir],
            init_filename=init_filename,
            # problem_filename=problem_filename, # uncomment to rescore
            )
        plot_summaries(summaries_dict, problem=problem, title='')

    # reduced_summaries_name = S.files.reduced_summaries_name
    # reduced_summaries_dict = extract_reduced_summaries(
    #     summaries_dict, reduced_summary_extract_func_tuples)
    # rf.pickle(reduced_summaries_dict, reduced_summaries_name, dir=data_dir)
