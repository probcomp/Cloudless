#!python
import cPickle
import os
#
import matplotlib
matplotlib.use('Agg')
import pylab
import numpy

# read in labels from alex's data
path = os.path.join("Data","cifar-100-python")
filename = "train"
full_filename = os.path.join(path,filename)
with open(full_filename) as fh:
    cifar = cPickle.load(fh)
labels = numpy.array(cifar["coarse_labels"])

# read in rpa datavectors
filename = os.path.join("Data","CIFAR10-codes.pkl")
with open(filename) as fh:
  cifar = cPickle.load(fh)
data = hf.convert_rpa_representation(cifar["codes"][:len(labels)])
