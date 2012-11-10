#!python
import os

class env():
    is_windows = None
    is_aws = False
    if os.sys.platform in {'cygwin':None,"win32":None}:
        is_windows = True
    else:
        is_windows = False

class path():
    root_dir = None
    if env.is_windows:
        root_dir = "c:" + os.path.sep
    else:
        root_dir = os.path.join(os.path.sep,"usr","local")
    #
    base_dir = os.path.join(root_dir,"Cloudless","examples","DPMB")
    if not os.path.isdir(base_dir): # assume AWS
        root_dir = os.path.join(
            os.path.sep,"usr","local","lib","python2.7","dist-packages")
        base_dir = os.path.join(root_dir,"Cloudless","examples","DPMB")
        env.is_aws = True
    #
    # data_dir = os.path.join(base_dir,"Data")
    data_dir = '/tmp/'
    output_dir = os.path.join(base_dir,"Output")

class files():
    seed_filename = 'seed_list.txt'
    gen_state_image_save_str = 'mrjob_problem_gen_state'
    gibbs_init_filename = 'gibbs_init.pkl.gz'
    run_dir_prefix = 'programmatic_mrjob_'
    parameters_filename = 'run_parameters.txt'
    reduced_summaries_name = 'reduced_summaries.pkl.gz'
    
data_dir = path.data_dir # FIXME: remove this when dependent code finally changed

# ensure data_dir exists
try:
    os.makedirs(data_dir)
except OSError, e:
    pass

# compile pyx_functions.pyx
try:
    import Cloudless.examples.DPMB.pyx_functions
except ImportError:
    filestr = __file__
    rel_path = os.path.split(filestr)[0]
    pyx_filename = 'compile_pyx_functions.sh'
    pyx_full_filename = os.path.join(rel_path, pyx_filename)
    system_str = ' '.join(['bash', pyx_full_filename, rel_path])
    os.system(system_str)

try:
    import mrjob
except ImportError:
    os.system('easy_install mrjob')

# gdocs
class gdocs():
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
                 if env.is_aws else '/media/VonNeumann/TinyImages'
tiny_image_problem_file = 'tiny_image_problem_nImages_' \
    '1000000_nPcaTrain_400000.pkl.gz'


class s3():
    # bucket_str = 'dpmb.mitprobabilisticcomputingproj'
    bucket_str = 'mitpcp-dpmb'
    bucket_dir = ''
    problem_bucket_dir = 'tiny_image_problems'
    summary_bucket_dir = 'tiny_image_summaries'
    #
    # ec2_auth_key = os.path.expanduser('~/awsKeyPair.pem')
    ec2_credentials_file = os.path.expanduser('~/.boto')
