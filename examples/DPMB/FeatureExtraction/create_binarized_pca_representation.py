#!python
import datetime
import os
import cPickle
#
import numpy as np
from scikits.learn.decomposition import RandomizedPCA
#
import Cloudless.examples.DPMB.settings as settings
reload(settings)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)


data_len = None

def read_cifar_10():
    path = os.path.join(settings.data_dir,'cifar-10-batches-py')
    filenames = ['data_batch_1','data_batch_2','data_batch_3',
                 'data_batch_4','data_batch_5']
    #
    data_list = []
    label_list = []
    for filename in filenames:
        full_filename = os.path.join(path,filename)
        with open(full_filename) as fh:
            cifar = cPickle.load(fh)
        data_list.extend(cifar['data'])
        label_list.extend(cifar['labels'])
    return np.array(data_list), np.array(label_list)

def read_cifar_100():
    path = os.path.join(settings.data_dir,"cifar-100-python")
    filename = "train"
    full_filename = os.path.join(path,filename)
    #
    with open(full_filename) as fh:
        cifar = cPickle.load(fh)
    labels = np.array(cifar['fine_labels'])
    data = np.array(cifar['data'])
    return data, labels

data,labels = read_cifar_100()
np.random.seed(0)
train_data = data[:data_len]
X = np.array(train_data)
pca = RandomizedPCA(n_components=256)
pca.fit(X)
# print pca.explained_variance_ratio_ 
#
weights = pca.components_.dot(train_data.T).T
medians = np.median(weights,axis=0)
bit_vectors = weights > medians
#
pickle_var = {
    'bit_vectors':bit_vectors,
    'weights':weights,
    'labels':labels
    }
rf.pickle(pickle_var,'pca_data.pkl.gz')
