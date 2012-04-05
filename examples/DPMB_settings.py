##data dimensions
numColumns = 50
numVectors = 20000

##priors
gamma_k = 1 
gamma_theta = 1

##computation
nGibbsSteps = 3

##other
clipBeta = [1E-2,1E10]
inferAlpha = True
inferBetas = True
verbose = True

##os
linuxBaseDir = "/usr/local/"
windowsBaseDir = "c:/"
dataDirSuffix = "dpmb.data/"
