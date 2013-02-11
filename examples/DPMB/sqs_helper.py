from collections import Counter
import argparse
#
import boto
import simplejson
#
import Cloudless.examples.DPMB.helper_functions as hf


def get_sqs():
    return boto.connect_sqs(**hf.get_boto_credentials())

def get_queue(queue_or_queuename):
    queue, queuename = None, None
    if isinstance(queue_or_queuename, str):
        sqs = get_sqs()
        queuename = queue_or_queuename
        queue = sqs.get_queue(queuename)
        # may return none
    else:
        queue = queue_or_queuename
        queuename = queue.name
    return queue, queuename

def get_or_create_queue(queue_or_queue_name):
    queue, queuename = get_queue(queue_or_queue_name)
    if queue is None:
        sqs = get_sqs()
        queue = sqs.create_queue(queuename)
    return queue, queuename

def deleter_factory(queue, message):
    def deleter():
        hf.echo_date('deleter_factory: ' + str(message))
        queue.delete_message(message)
    return deleter
def get_queue_iterator(queue_or_queuename, visibility_timeout=600):
    queue, queuename = get_queue(queue_or_queuename)
    message = queue.read(visibility_timeout=visibility_timeout)
    while message is not None:
        body = message.get_body()
        msg_data = simplejson.loads(body)
        message_deleter = deleter_factory(queue, message)
        yield msg_data, message_deleter
        message = queue.read(visibility_timeout=visibility_timeout)
    raise StopIteration()

def push_str_to_queue(in_str, queue_name):
    queue, queue_name = get_or_create_queue(queue_name)
    body = simplejson.dumps(in_str)
    message = queue.new_message(body=body)
    queue.write(message)
    return queue

def push_str_list_to_queue(in_list, queue_name):
    for el in in_list:
        push_str_to_queue(el, queue_name)

def count_and_delete(queue_or_queuename, do_print=False):
    q_iter = get_queue_iterator(queue_or_queuename)
    q_els = [el for el in q_iter]
    message_list = [el[0] for el in q_els]
    if do_print:
        for el in q_els:
            print el[0]
    [el[1]() for el in q_els]
    return Counter(message_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('queuename', type=str)
    args = parser.parse_args()
    queuename = args.queuename
    #
    count_and_delete(queuename, True)
