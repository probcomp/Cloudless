#!python
import argparse
import os
import re
import time
from multiprocessing import Pool, cpu_count
from functools import partial
#
import boto
import simplejson
#
import Cloudless.examples.DPMB.settings as settings
import Cloudless.examples.DPMB.s3_helper as s3h
import Cloudless.examples.DPMB.remote_functions as rf
import Cloudless.examples.DPMB.helper_functions as hf
import Cloudless.examples.DPMB.MrJob.consolidate_summaries as cs


data_dir = settings.path.data_dir

get_filename_from_el = lambda el: os.path.split(el.name)[-1]
def filter_bucket_filenames(bucket, bucket_dir_suffix, filter_func):
    bucket_full_dir = 'tiny_image_summaries/' + bucket_dir_suffix + '/'
    bucket_dir_contents_generator = bucket.list(
        prefix=bucket_full_dir, delimiter='/')
    all_filenames = map(get_filename_from_el, bucket_dir_contents_generator)
    filter_filenames = filter(filter_func, all_filenames)
    return filter_filenames

get_score_name = lambda summary_name: re.sub('^summary_', 'score_', summary_name)
get_summary_name = lambda score_name: re.sub('^score_', 'summary_', score_name)

def clear_queue(queue_or_queuename):
    queue_iterator = get_queue_iterator(queue_or_queuename)
    for msg_data, message_deleter in queue_iterator:
        message_deleter()

def create_queue_from_list(in_list, queue_name):
    sqs = boto.connect_sqs()
    queue = sqs.get_queue(queue_name)
    if queue is None:
        queue = sqs.create_queue(queue_name)
    for list_el in in_list: 
        body = simplejson.dumps(list_el)
        message = queue.new_message(body=body)
        queue.write(message)
    return queue

def deleter_factory(queue, message):
    def deleter():
        hf.echo_date('deleter_factory: ' + str(message))
        queue.delete_message(message)
    return deleter
def get_queue_iterator(queue_or_queuename, visibility_timeout=600):
    queue, queuename = get_queue_helper(queue_or_queuename)
    message = queue.read(visibility_timeout=visibility_timeout)
    while message is not None:
        body = message.get_body()
        msg_data = simplejson.loads(body)
        message_deleter = deleter_factory(queue, message)
        yield msg_data, message_deleter
        message = queue.read(visibility_timeout=visibility_timeout)
    raise StopIteration()

setdiff = lambda target, actual: \
    list(set(target).difference(actual))
is_summary = lambda filename: \
    filename.startswith('summary_') and filename.endswith('.pkl.gz')
is_score = lambda filename: \
    filename.startswith('score_') and filename.endswith('.pkl.gz')
def create_file_queue(bucket_dir_suffix, bucket_str=None):
    if bucket_str is None:
        bucket_str = settings.s3.bucket_str
    #
    bucket = boto.connect_s3().get_bucket(bucket_str)
    summary_filenames = filter_bucket_filenames(bucket, bucket_dir_suffix,
                                                is_summary)
    score_filenames = filter_bucket_filenames(bucket, bucket_dir_suffix,
                                              is_score)
    target_score_filenames = map(get_score_name, summary_filenames)
    missing_score_filenames = setdiff(target_score_filenames, score_filenames)
    corresponding_summary_filenames = map(
        get_summary_name, missing_score_filenames)
    queue = create_queue_from_list(
        corresponding_summary_filenames, bucket_dir_suffix)
    return queue

def get_queue_helper(queue_or_queuename):
    queue, queuename = None, None
    if isinstance(queue_or_queuename, str):
        sqs = boto.connect_sqs()
        queuename = queue_or_queuename
        queue = sqs.get_queue(queuename)
        if queue is None:
            raise Exception('cant get queuename: ' + queuename)
    else:
        queue = queue_or_queuename
        queuename = queue.name
    return queue, queuename

def process_file_queue(queue_or_queuename):
    queue, queuename = get_queue_helper(queue_or_queuename)
    problem = rf.verify_problem_helper(queuename, do_unpickle=True)
    filename_tuple_generator = get_queue_iterator(queuename)
    process_summary_helper = partial(
        process_summary, problem=problem, bucket_dir_suffix=queuename, write_s3=True)
    result = map(process_summary_helper, filename_tuple_generator)
    #
    # can't seem to get this to work
    # would only really be useful if they could share the memory that stores problem data
    # num_workers = cpu_count()
    # p = Pool(num_workers)
    # result = p.map_async(process_summary_helper, filename_tuple_generator)
    # p.close()
    # p.join()
    return result

def process_all_local(bucket_dir_suffix):
    local_dir = os.path.join(data_dir, bucket_dir_suffix)
    problem = rf.unpickle('problem.pkl.gz', dir=local_dir)
    filename_tuple_generator = filter(is_summary, os.listdir(local_dir))
    num_files = len(filename_tuple_generator)
    dummy_deleter = lambda : 1
    filename_tuple_generator = zip(filename_tuple_generator, [dummy_deleter] * num_files)
    process_summary_helper = partial(
        process_summary, problem=problem, bucket_dir_suffix=bucket_dir_suffix, write_s3=False)
    result = map(process_summary_helper, filename_tuple_generator)
    return result

def process_summary(summary_tuple, problem, bucket_dir_suffix, write_s3):
    summary_filename, message_deleter = summary_tuple
    hf.echo_date(summary_filename)
    summary = rf.verify_file_helper(summary_filename, bucket_dir_suffix,
                                 do_unpickle=True)
    scored_summary = cs.score_summary(summary, problem)
    # must read in problem : will be building a state, may need m2.2xlarge
    score_dict = dict(
        ari=scored_summary['ari'],
        score=scored_summary['score'],
        test_lls=scored_summary['test_lls'],
        alpha=summary['alpha'],
        betas=summary['betas'],
        num_clusters=summary['num_clusters'],
        timing=summary['timing'],
        )
    score_filename = get_score_name(summary_filename)
    local_dir = os.path.join(data_dir, bucket_dir_suffix)
    rf.pickle(score_dict, score_filename, dir=local_dir)
    rf.verify_file_helper(score_filename, bucket_dir_suffix, write_s3=write_s3)
    message_deleter()
    return summary_filename

bucket_dir_suffix = 'programmatic_mrjob_a36e808195'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bucket_dir_suffix', type=str)
    parser.add_argument('--is_controller', action='store_true')
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    parser.add_argument('--do_create_queue', action='store_true')
    args = parser.parse_args()
    #
    bucket_dir_suffix = args.bucket_dir_suffix
    is_controller = args.is_controller
    num_workers = args.num_workers
    do_create_queue = args.do_create_queue
    #
    if is_controller:
        hf.echo_date('is_controller')
        rf.verify_problem_helper(bucket_dir_suffix)
        sqs = boto.connect_sqs()
        if do_create_queue or not sqs.get_queue(bucket_dir_suffix):
            create_file_queue(bucket_dir_suffix)
            time.sleep(10)
        for worker_idx in range(num_workers):
            os.system('python generate_scoring.py ' + bucket_dir_suffix + ' &')
    else:
        hf.echo_date('is_NOT_controller')
        processed_files = process_file_queue(bucket_dir_suffix)
        print_str = ', '.join(map(str, processed_files))
        hf.echo_date(print_str)
