#!python
import os

# resolve directories to use
root_dir = None
is_windows = None
if os.sys.platform in {'cygwin':None,"win32":None}:
    root_dir = "c:" + os.path.sep
    is_windows = True
else:
    root_dir = os.path.join(os.path.sep,"usr","local")
    is_window = False
base_dir = os.path.join(root_dir,"Cloudless","examples","DPMB")
data_dir = os.path.join(base_dir,"Data")

# compile pyx_functions.pyx
os.system('bash ' + os.path.join(base_dir,"compile_pyx_functions.sh"))

# gdocs settings
auth_file = os.path.expanduser("~/mh_gdocs_auth")
gdocs_folder_default = "MH"

# cifar
cifar_10_problem_file = "cifar_10_problem.pkl.gz"
cifar_100_problem_file = "cifar_100_problem.pkl.gz"
