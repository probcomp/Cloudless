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

# read in labels from alex's data
path = os.path.join("Data","cifar-100-python")
filename = "train"
full_filename = os.path.join(path,filename)
with open(full_filename) as fh:
    cifar = cPickle.load(fh)
labels = numpy.array(cifar["coarse_labels"])

# read in rpa datavectors
filename = os.path.join("Data","CIFAR100-codes.pkl")
with open(filename) as fh:
  cifar = cPickle.load(fh)
data = hf.convert_rpa_representation(cifar["codes"][:len(labels)])

# write out to a pickle file
filename = "cifar_100_problem.pkl"
cifar = {
    "zs":labels,
    "xs":data,
    "name":"cifar-100",
}
with open(filename,"wb") as fh:
    cPickle.dump(cifar,fh)
