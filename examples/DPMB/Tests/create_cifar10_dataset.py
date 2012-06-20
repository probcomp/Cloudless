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
path = os.path.join("Data","cifar-10-batches-py")
# filename = "batches.meta"
filenames = ["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5"]
labels_list = []
for filename in filenames:
    full_filename = os.path.join(path,filename)
    with open(full_filename) as fh:
        cifar = cPickle.load(fh)
    labels_list.append(cifar["labels"])
#
labels = numpy.array(labels_list).flatten()

# read in rpa datavectors
filename = os.path.join("Data","CIFAR10-codes.pkl")
with open(filename) as fh:
  cifar = cPickle.load(fh)
data = hf.convert_rpa_representation(cifar["codes"][:len(labels)])
