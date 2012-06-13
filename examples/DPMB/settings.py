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
linuxBaseDir = "/usr/local/"
windowsBaseDir = "c:/"
dataDirSuffix = "dpmb.data/"

auth_file = os.path.expanduser("~/mh_gdocs_auth")
gdocs_folder_default = "MH"
