#!python
import argparse
import os
import re
#
import matplotlib
matplotlib.use('Agg')
import pylab
import numpy
#
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.settings as settings
reload(settings)

# helper functions
is_summary = lambda x : x[:8] == 'summary_'
split_summary_re = re.compile('^(.*iternum)([-\d]*).pkl.gz')
split_summary = lambda x : split_summary_re.match(x).groups()
create_summary_str = lambda summary_name, iter_num : \
    summary_name + str(iter_num) + '.pkl.gz'

def get_summary_names(data_dir):
    data_files = os.listdir(data_dir)
    summary_files = filter(is_summary,data_files)
    summary_names = {}
    for summary_file in sorted(summary_files):
        summary_name,iter_num_str = split_summary(summary_file)
        summary_names.setdefault(summary_name,[]).append(int(iter_num_str))
    return summary_names

def get_summaries_dict(summary_names,data_dir):
    summaries_dict = {}
    for summary_name, iter_list in summary_names.iteritems():
        iter_list = sorted(iter_list)
        for iter_num in iter_list:
            filename = create_summary_str(summary_name,iter_num)
            filename = os.path.join(data_dir,filename)
            new_summary = rf.unpickle(filename)
            summaries_dict.setdefault(summary_name,[]).append(new_summary)
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
extract_test_lls = lambda summaries : [
    ('%.2f' % numpy.mean(summary['test_lls']))
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

def main():

    # parse some args
    parser = argparse.ArgumentParser('consolidate summaries')
    parser.add_argument('data_dirs',nargs='+',type=str)
    args = parser.parse_args()
    data_dirs = args.data_dirs

    shorten_name = lambda x: x[8:-8]

    # read the summaries
    summaries_dict = {}
    for data_dir in data_dirs:
        summary_names = get_summary_names(data_dir)
        temp_summaries_dict = get_summaries_dict(summary_names,data_dir)
        print_info(summaries_dict)
        summaries_dict.update(temp_summaries_dict)

    # create test_lls plot
    pylab.figure()
    for summaries_name, summaries in summaries_dict.iteritems():
        timing = numpy.array(extract_delta_t(summaries),dtype=float)
        test_lls = numpy.array(extract_test_lls(summaries),dtype=float)
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
        timing = numpy.array(extract_delta_t(summaries),dtype=float)
        scores = numpy.array(extract_score(summaries),dtype=float)
        pylab.plot(timing,scores)
    legend_list = map(shorten_name,summaries_dict.keys())
    pylab.legend(legend_list,loc='lower right')
    pylab.title('scores')
    pylab.xlabel('time (seconds)')
    pylab.ylabel('model score')
    pylab.savefig('scores')

if __name__ == '__main__':
    main()
