#!python
import cPickle
import os
#
import matplotlib
matplotlib.use('Agg')
import pylab
import numpy
#
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless.examples.DPMB.settings as settings
reload(settings)


# read in labels from alex's data
path = os.path.join(settings.data_dir,"cifar-10-batches-py")
# read the train labels
filenames = ["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5"]
train_labels_list = []
for filename in filenames:
    full_filename = os.path.join(path,filename)
    with open(full_filename) as fh:
        cifar = cPickle.load(fh)
    train_labels_list.append(cifar["labels"])
train_labels = numpy.array(train_labels_list).flatten()
# read the test labels
filename = "test_batch"
full_filename = os.path.join(path,filename)
with open(full_filename) as fh:
    cifar = cPickle.load(fh)
test_labels= numpy.array(cifar["labels"])


# read in rpa datavectors
filename = os.path.join(settings.data_dir,"CIFAR10-codes.pkl")
with open(filename) as fh:
  cifar = cPickle.load(fh)
train_data = hf.convert_rpa_representation(cifar["codes"][:len(train_labels)])
test_data = hf.convert_rpa_representation(cifar["codes"][-len(test_labels):])

# write out to a pickle file
filename = "cifar_10_problem.pkl"
cifar = {
    "zs":train_labels,
    "xs":train_data,
    "test_zs":test_labels,
    "test_xs":test_data,
    "name":"cifar-10",
}
with open(filename,"wb") as fh:
    cPickle.dump(cifar,fh)
