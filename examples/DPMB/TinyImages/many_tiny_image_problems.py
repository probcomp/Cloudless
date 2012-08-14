#!python
import os
import datetime
import argparse
#
import numpy
#
import Cloudless.examples.DPMB.TinyImages.loadTinyImages as lti
reload(lti)
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.s3_helper as s3_helper
reload(s3_helper)
import Cloudless.examples.DPMB.settings as settings
reload(settings)
import Cloudless.examples.DPMB.FeatureExtraction.binarized_pca_representation \
    as bpr
reload(bpr)


# set some read args
num_pieces = 4
num_pca_train_pieces = 1
num_pieces_list = [8,16,32]
max_num_pieces = max(num_pieces_list)

# set some parameters
images_per_piece = 10000
pixels_per_image = 3072
n_components = 256
#
n_pca_train = num_pca_train_pieces * images_per_piece
n_test = int(.01*num_pieces*images_per_piece)
base_dir = '/mnt/' if settings.is_aws else '/media/VonNeumann/'
pkl_dir = os.path.join(settings.base_dir,'Data')
bucket_dir = 'TinyImages'
local_dir = os.path.join(base_dir, bucket_dir)
problem_file = 'tiny_image_problem_nImages_' + str(num_pieces*images_per_piece) \
    + '_nPcaTrain_' + str(num_pca_train_pieces*images_per_piece)+ '.pkl.gz'
full_problem_file = os.path.join(pkl_dir, problem_file)
#
data_piece_filter = lambda x : x.find('_data')!=-1

# pull down base problem file
s3.local_dir=pkl_dir
if not s3.is_local(problem_file):
    s3.get_s3(problem_file)
base_problem = rf.unpickle(full_problem_file)
pca_components = base_problem['pca_components']
medians = base_problem['medians']
print datetime.datetime.now()
print 'Done unpickling base problem'

# make sure data files are in place
if not os.path.isdir(local_dir): os.makedirs(local_dir)
s3 = s3_helper.S3_helper(
    bucket_str=settings.bucket_str, bucket_dir=bucket_dir, local_dir=local_dir)
#
all_files = [key.name for key in s3.bucket.list(prefix=s3.bucket_dir)]
# all_files = sorted([key.name for key in s3.bucket.list(prefix=s3.bucket_dir)])
data_files = filter(data_piece_filter, all_files)
data_files = [os.path.split(data_file)[-1] for data_file in data_files]
#
for data_file in data_files[:max_num_pieces]:
    s3.verify_file(data_file)
print datetime.datetime.now()
print 'Done copying down files'

# read in all the data
image_data = numpy.ndarray(
    (max_num_pieces*images_per_piece,pixels_per_image)
    ,dtype=numpy.int32
    )
image_indices = []
for piece_idx, data_file in enumerate(data_files[:max_num_pieces]):
    full_filename = os.path.join(local_dir, data_file)
    unpickled = rf.unpickle(full_filename)
    start_idx = piece_idx*images_per_piece
    end_idx = (1+piece_idx)*images_per_piece
    image_data[start_idx:end_idx] = unpickled['image_list']
    image_indices.extend(unpickled['image_indices'])
    print datetime.datetime.now()
    print 'Done reading ' + data_file
#
print datetime.datetime.now()
print 'Done reading files'

for num_pieces in num_pieces_list:
    # read in the data
    binarized_data = bpr.generate_binarized_pca_data(
        image_data[:num_pieces*images_per_piece], pca_components, medians)
    #
    # FIXME : is it appropriate for test data to be part of features extraction?
    problem = {
        'xs':binarized_data[:-n_test],
        'test_xs':binarized_data[-n_test:],
        'name':'tiny-images-test-bpr',
        'train_indices':image_indices[:-n_test],
        'test_indices':image_indices[-n_test:],
        'pca_components':pca_components,
        'medians':medians,
        'n_pca_train':n_pca_train,
        }
    #
    problem_file = 'tiny_image_problem_nImages_' \
                   + str(num_pieces*images_per_piece) \
                   + '_nPcaTrain_' \
                   + str(num_pca_train_pieces*images_per_piece)+ '.pkl.gz'
    full_problem_file = os.path.join(pkl_dir, problem_file)

    rf.pickle(problem, full_problem_file)
    print datetime.datetime.now()
    print 'Done pickling problem: ' + problem_file
    s3.local_dir = pkl_dir
    s3.put_s3(problem_file)
    print datetime.datetime.now()
    print 'Done pushing problem up to s3'
