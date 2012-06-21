import numpy as np
import scipy.special as ss
#
import DPMB as dm
reload(dm)
import DPMB_State as ds
reload(ds)
    
def visualize_mle_alpha(cluster_list=None,points_per_cluster_list=None,max_alpha=None):
    import pylab
    cluster_list = cluster_list if cluster_list is not None else 10**np.arange(0,4,.5)
    points_per_cluster_list = points_per_cluster_list if points_per_cluster_list is not None else 10**np.arange(0,4,.5)
    max_alpha = max_alpha if max_alpha is not None else int(1E4)
    ##
    mle_vals = []
    for clusters in cluster_list:
        for points_per_cluster in points_per_cluster_list:
            mle_vals.append([clusters,points_per_cluster,dm.mle_alpha(clusters,points_per_cluster,max_alpha=max_alpha)])
    ##
    mle_vals = np.array(mle_vals)
    pylab.figure()
    pylab.loglog(mle_vals[:,0],mle_vals[:,1],color='white') ##just create the axes
    pylab.xlabel("clusters")
    pylab.ylabel("points_per_cluster")
    pylab.title("MLE alpha for a given data configuration\nmax alpha: "+str(max_alpha-1))
    for clusters,points_per_cluster,mle in mle_vals:
        pylab.text(clusters,points_per_cluster,str(int(mle)),color='red')
