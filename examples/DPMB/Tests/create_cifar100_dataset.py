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
path = os.path.join(settings.data_dir,"cifar-100-python")
# read the training data
filename = "train"
full_filename = os.path.join(path,filename)
with open(full_filename) as fh:
    cifar = cPickle.load(fh)
train_labels = numpy.array(cifar["coarse_labels"])
# read the test data
filename = "test"
full_filename = os.path.join(path,filename)
with open(full_filename) as fh:
    cifar = cPickle.load(fh)
test_labels = numpy.array(cifar["coarse_labels"])

# read in rpa datavectors
filename = os.path.join(settings.data_dir,"CIFAR100-codes.pkl")
with open(filename) as fh:
  cifar = cPickle.load(fh)
train_data = hf.convert_rpa_representation(cifar["codes"][:len(train_labels)])
test_data = hf.convert_rpa_representation(cifar["codes"][-len(test_labels):])

# write out to a pickle file
cifar = {
    "zs":train_labels,
    "xs":train_data,
    "test_zs":test_labels,
    "test_xs":test_data,
    "name":"cifar-100",
    "type":"coarse_labels"
}
filename = os.path.join(settings.data_dir,settings.cifar_100_problem_file)
with open(filename,"wb") as fh:
    cPickle.dump(cifar,fh)
