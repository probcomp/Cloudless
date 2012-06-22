#!python
import cPickle
import os
#
import matplotlib
matplotlib.use('Agg')
import pylab
import numpy
from matplotlib.mlab import find
#
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.settings as settings
reload(settings)


# settings
label_type = "fine_labels"
subset_count_per_label = 100
sort_seed = 0

# read in labels from alex's data
path = os.path.join(settings.data_dir,"cifar-100-python")
# read the training data
filename = "train"
full_filename = os.path.join(path,filename)
with open(full_filename) as fh:
    cifar = cPickle.load(fh)
train_labels = numpy.array(cifar[label_type])
# read the test data
filename = "test"
full_filename = os.path.join(path,filename)
with open(full_filename) as fh:
    cifar = cPickle.load(fh)
test_labels = numpy.array(cifar[label_type])

# read in rpa datavectors
filename = os.path.join(settings.data_dir,"CIFAR100-codes.pkl")
with open(filename) as fh:
  cifar = cPickle.load(fh)
train_data = hf.convert_rpa_representation(cifar["codes"][:len(train_labels)])
test_data = hf.convert_rpa_representation(cifar["codes"][-len(test_labels):])

# create a reduced subset of the data
numpy.random.seed(sort_seed)
chosen_indices = []
for cluster_num in xrange(max(train_labels)):
    cluster_indices = find(cluster_num==numpy.array(train_labels))
    cluster_subset_indices = numpy.random.permutation(cluster_indices)[:subset_count_per_label]
    chosen_indices.extend(cluster_subset_indices)
subset_zs = train_labels[chosen_indices]
subset_xs = train_data[chosen_indices]


# write out to a pickle file
cifar = {
    "zs":train_labels,
    "xs":train_data,
    "subset_zs":subset_zs,
    "subset_xs":subset_xs,
    "test_zs":test_labels,
    "test_xs":test_data,
    "name":"cifar-100",
    "type":label_type,
    "chosen_indices":chosen_indices
}
filename = os.path.join(settings.data_dir,settings.cifar_100_problem_file)
rf.pickle(cifar,filename)
