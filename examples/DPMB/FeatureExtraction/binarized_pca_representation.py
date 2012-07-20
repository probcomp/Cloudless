#!python
import os
import argparse
#
import numpy as np
from scikits.learn.decomposition import RandomizedPCA
#
import Cloudless.examples.DPMB.settings as settings
reload(settings)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)


def read_cifar_10():
    path = os.path.join(settings.data_dir,'cifar-10-batches-py')
    filenames = ['data_batch_1','data_batch_2','data_batch_3',
                 'data_batch_4','data_batch_5']
    #
    data_list = []
    label_list = []
    for filename in filenames:
        full_filename = os.path.join(path,filename)
        cifar = rf.unpickle(full_filename)
        data_list.extend(cifar['data'])
        label_list.extend(cifar['labels'])
    return np.array(data_list), np.array(label_list)

def read_cifar_100():
    path = os.path.join(settings.data_dir,"cifar-100-python")
    filename = "train"
    full_filename = os.path.join(path,filename)
    #
    cifar = rf.unpickle(full_filename)
    labels = np.array(cifar['fine_labels'])
    data = np.array(cifar['data'])
    return data, labels

def generate_binarized_pca_model(train_data,n_components=256,seed=0):
    np.random.seed(seed)
    X = np.array(train_data)
    #
    pca = RandomizedPCA(n_components=n_components)
    pca.fit(X)
    weights = pca.components_.dot(train_data.T).T
    medians = np.median(weights,axis=0)
    #
    return pca.components_, medians, pca

def generate_binarized_pca_data(data,components,medians):
    weights = components.dot(data.T).T
    bit_vectors = weights > medians
    return bit_vectors

def main():
    parser = argparse.ArgumentParser(
        'create a binarized pca representation of cifar data')
    parser.add_argument('--n_components',default=256,type=int)
    parser.add_argument('--n_train',default=50000,type=int)
    parser.add_argument('--cifar_10',action='store_true')
    parser.add_argument('--pkl_name',default='pca_parameters.pkl.gz',type=str)
    args = parser.parse_args()
    n_components = args.n_components
    n_train = args.n_train
    cifar_10 = args.cifar_10
    pkl_name = args.pkl_name
    #
    read_func = None
    if cifar_10:
        read_func = read_cifar_10
    else:
        read_func = read_cifar_100
    #
    data, labels = read_func()
    train_data = data[:n_train]
    components, medians, pca = generate_binarized_pca_model(
        train_data, n_components)
    # bit_vectors = generate_binarized_pca_data(data,components,medians)
    #
    pickle_var = {
        'components':components,
        'medians':medians,
        'pca':pca
        }
    rf.pickle(pickle_var,pkl_name)

if __name__ == '__main__':
    main()
