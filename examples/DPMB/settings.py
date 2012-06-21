#!python
import os


if os.sys.platform in {'cygwin':None,"win32":None}:
    root_dir = "c:" + os.path.sep
else:
    root_dir = os.path.join(os.path.sep,"usr","local")
data_dir = os.path.join(root_dir,"Cloudless","examples","DPMB","Data")

##data dimensions
numColumns = 50
numVectors = 2000

##priors
gamma_k = 1 
gamma_theta = 1

##computation
nGibbsSteps = 3

##control
clipBeta = [1E-2,1E10]
infer_alpha = None
infer_beta = None
verbose = True

##os
auth_file = os.path.expanduser("~/mh_gdocs_auth")
gdocs_folder_default = "MH"
