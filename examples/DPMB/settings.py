#!python
import os

# resolve directories to use
if os.sys.platform in {'cygwin':None,"win32":None}:
    root_dir = "c:" + os.path.sep
else:
    root_dir = os.path.join(os.path.sep,"usr","local")
dpmb_dir = os.path.join(root_dir,"Cloudless","examples","DPMB")
data_dir = os.path.join(dpmb_dir,"Data")

# compile pyx_functions.pyx
os.system('bash ' + os.path.join(dpmb_dir,"compile_pyx_functions.sh"))

# gdocs settings
auth_file = os.path.expanduser("~/mh_gdocs_auth")
gdocs_folder_default = "MH"

# cifar
cifar_10_problem_file = "cifar_10_problem.pkl.gz"
cifar_100_problem_file = "cifar_100_problem.pkl.gz"
