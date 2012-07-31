#!python
import os
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


# set some parameters
base_dir = '/media/VonNeumann/' if settings.is_aws else '/mnt/'
dest_dir = os.path.join(base_dir,'TinyImages')
problem_file = os.path.join(dest_dir,'tiny_image_problem.pkl.gz')
bucket_dir = 'TinyImages'
num_pieces = 100
n_test = int(.01*num_pieces*10000)
data_piece_filter = lambda x : x.find('_data')!=-1

# make sure files are in place
if not os.path.isdir(dest_dir): os.makedirs(dest_dir)
s3 = s3_helper.S3_helper(
    bucket_str=settings.bucket_str,bucket_dir=bucket_dir,local_dir=dest_dir)
all_files = [key.name for key in s3.bucket.list(prefix=s3.bucket_dir)]
data_files = filter(data_piece_filter,all_files)
data_files = [os.path.split(data_file)[-1] for data_file in data_files]
#
for data_file in data_files[:num_pieces]:
    s3.verify_file(data_file)

# read in the data
image_list = []
image_indices = []
for data_file in data_files[:num_pieces]:
    full_filename = os.path.join(dest_dir,data_file)
    unpickled = rf.unpickle(full_filename)
    image_list.extend(unpickled['image_list'])
    image_indices.extend(unpickled['image_indices'])

# run pca
image_data = numpy.array(image_list)
pca_components, medians, pca = bpr.generate_binarized_pca_model(image_data)
binarized_data = bpr.generate_binarized_pca_data(image_data,pca_components,medians)

# FIXME : is it appropriate for test data to be part of features extraction?
tiny_images = {
    'xs':binarized_data[:-n_test],
    'test_xs':binarized_data[-n_test:],
    'name':'tiny-images-test-bpr',
    'train_indices':image_list[:-n_test],
    'test_indices':image_list[-n_test:],
    }
rf.pickle(tiny_images,problem_file)
