#!python
import os
import argparse
#
import numpy as np
from matplotlib.mlab import find
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

def create_cifar_bpr_problem(parameter_file,read_func,outfile,
                             num_labels=20,train_rows_per_label=400,
                             test_rows_per_label=100,seed=0):
    # FIXME : may need to pick a subset of the labels to simplify problem
    assert train_rows_per_label+test_rows_per_label<=500, \
        'cifar-100 only has 500 rows per labels'
    np.random.seed(seed)
    parameter_hash = rf.unpickle(parameter_file)
    components = parameter_hash['components']
    medians = parameter_hash['medians']
    raw_data, labels = read_func()
    #
    chosen_labels = np.random.permutation(
        xrange(max(labels)+1))[:num_labels]
    train_indices = []
    test_indices = []
    for cluster_num in chosen_labels:
        cluster_indices = find(cluster_num==labels)
        cluster_subset_indices = np.random.permutation(
            cluster_indices)[:train_rows_per_label+test_rows_per_label]
        train_indices.extend(
            cluster_subset_indices[:train_rows_per_label])
        test_indices.extend(
            cluster_subset_indices[train_rows_per_label:])
    # one last permutation so gibbs init is in randomized order
    train_indices = np.random.permutation(train_indices)
    test_indices = np.random.permutation(test_indices)
    #
    train_raw_data = raw_data[train_indices]
    train_labels = labels[train_indices]
    train_bit_vectors = generate_binarized_pca_data(
        train_raw_data,components,medians)    
    test_raw_data = raw_data[test_indices]
    test_labels = labels[test_indices]
    test_bit_vectors = generate_binarized_pca_data(
        test_raw_data,components,medians)    
    #
    cifar = {
        'zs':train_labels,
        'xs':train_bit_vectors,
        'test_zs':test_labels,
        'test_xs':test_bit_vectors,
        'name':'cifar-100-bpr',
        'train_indices':train_indices,
        'test_indices':test_indices,
        }
    rf.pickle(cifar,outfile)
    return cifar

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
    # create a binarized representation of the data
    data, labels = read_func()
    train_data = data[:n_train]
    components, medians, pca = generate_binarized_pca_model(
        train_data, n_components)
    # save the binarized reprsentation
    pickle_var = {
        'components':components,
        'medians':medians,
        'pca':pca
        }
    rf.pickle(pickle_var,pkl_name)
    # create the problem file and pickle it
    cifar = create_cifar_bpr_problem(
        parameter_file=pkl_name,
        read_func=read_func,
        outfile='out.pkl.gz'
        )

if __name__ == '__main__':
    main()
