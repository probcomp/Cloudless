#!python
import os


# resolve directories to use
root_dir = None
is_windows = None
is_aws = False
if os.sys.platform in {'cygwin':None,"win32":None}:
    root_dir = "c:" + os.path.sep
    is_windows = True
else:
    root_dir = os.path.join(os.path.sep,"usr","local")
    is_window = False
base_dir = os.path.join(root_dir,"Cloudless","examples","DPMB")
#
if not os.path.isdir(base_dir): # assume AWS
    root_dir = os.path.join(
        os.path.sep,"usr","local","lib","python2.7","dist-packages")
    base_dir = os.path.join(root_dir,"Cloudless","examples","DPMB")
    is_aws = True
#
data_dir = os.path.join(base_dir,"Data")
output_dir = os.path.join(base_dir,"Output")

try:
    os.makedirs(data_dir)
except OSError, e:
    pass

# compile pyx_functions.pyx
try:
    import Cloudless.examples.DPMB.pyx_functions
except ImportError:
    os.system(
        " ".join([
        'bash',
        os.path.join(base_dir,"compile_pyx_functions.sh"),
        base_dir,
        ]))
#
try:
    import pandas
except ImportError:
    os.system(
        'easy_install http://pypi.python.org/packages/source/p/' \
            'pandas/pandas-0.7.0rc1.tar.gz'
        )
#
try:
    import mrjob
except ImportError:
    os.system('easy_install mrjob')

# gdocs
auth_file = os.path.expanduser("~/mh_gdocs_auth")
gdocs_folder_default = "MH"

# cifar
cifar_10_problem_file = "cifar_10_problem.pkl.gz"
cifar_100_problem_file = "cifar_100_problem.pkl.gz"
cifar_100_bpr_problem_file = "cifar_100_bpr_problem.pkl.gz"
cifar_10_image_dir = "CIFAR10"
cifar_100_image_dir = "CIFAR100"
clustering_dir = "Current_clusterings"

# tiny images
# tiny_image_dir only necessary for pulling actual image data
tiny_image_dir = '/mnt/TinyImages' \
                 if is_aws else '/media/VonNeumann/TinyImages'
tiny_image_problem_file = 'tiny_image_problem_nImages_' \
    '1000000_nPcaTrain_400000.pkl.gz'


# s3
home_dir = os.path.expanduser("~/")
bucket_str = "dpmb.mitprobabilisticcomputingproj"
bucket_dir = ""
ec2_auth_key = os.path.expanduser("~/awsKeyPair.pem")
ec2_credentials_file = os.path.expanduser("~/.boto")
