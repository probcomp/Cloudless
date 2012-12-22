import argparse
#
import numpy
#
import Cloudless.examples.DPMB.remote_functions as rf


parser = argparse.ArgumentParser()
parser.add_argument('summary_filename', type=str)
args = parser.parse_args()
summary_filename = args.summary_filename

summary = rf.unpickle(summary_filename)
clusters_per_node = map(len, summary['lolo_x_indices'])
datapoints_per_node = map(lambda x: sum(map(len, x)), summary['lolo_x_indices'])

argsort_indices = numpy.argsort(clusters_per_node)[::-1]
print 'clusters_per_node'
print numpy.array(clusters_per_node)[argsort_indices]
print 'datapoints_per_node'
print numpy.array(datapoints_per_node)[argsort_indices]
