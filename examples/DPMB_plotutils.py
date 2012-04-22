import DPMB as dm
reload(dm)
import DPMB_State as ds
reload(ds)
import numpy as np
import scipy.special as ss
    
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

def debug_conditionals():
    temp_num_cols = 10
    temp_num_clusters = 10
    temp_num_rows_per_cluster = 10
    tempState = ds.DPMB_State(None,paramDict={"numColumns":temp_num_cols,"betas":np.repeat(.5,temp_num_cols),"alpha":1})
    tempState.reset_data()
    for temp_cluster_idx in range(temp_num_clusters):
         temp_cluster = ds.Cluster(tempState)
         for x in range(temp_num_rows_per_cluster):
              tempState.zs.append(temp_cluster)  ## why doesn't create vector do this?
              temp_cluster.create_vector()
    ##
    print
    for temp_vector in tempState.xs:
         print  temp_vector.vectorIdx,tempVector.data
         temp_cluster = temp_vector.cluster
         temp_cluster.remove_vector(temp_vector)
         dm.cluster_predictive(temp_vector,temp_cluster,tempState),temp_cluster.cluster_idx
         temp_cluster.add_vector(temp_vector)
    ##
    tempState.zs.append(temp_cluster) ## must increase the vector count so alpha_term is correct
    print
    for temp_cluster in tempState.cluster_list:
         tempVector = ds.Vector(None,data=[np.random.binomial(1,theta) for theta in temp_cluster.thetas])
         print "Creating from ",temp_cluster.cluster_idx
         print temp_vector.data
         for temp_cluster in tempState.cluster_list:
              dm.cluster_predictive(tempVector,temp_cluster,tempState),temp_cluster.cluster_idx
    ##
    print dm.cluster_predictive(tempVector,None,tempState),-1
    print tempState.getThetas().round(1)
    plot_state(tempState,{"phis":range(temp_num_clusters),"zs":np.repeat(range(temp_num_clusters),temp_num_rows_per_cluster)},aspect="auto")

def run_jobs(num_clusters_list=None,num_points_per_cluster_list=None,path=None):
    import os
    num_clusters_list = num_clusters_list if num_clusters_list is not None else [10,20,30]
    num_points_per_cluster_list = num_points_per_cluster_list if num_points_per_cluster_list is not None else [10,50,100]
    path = path if path is not None else ""
    ##
    ##perhaps I should make DPMB_basic.py a function?
    ##will each run get its own namespace?
    ##then can pass in dict for named arguments
    for num_clusters in num_clusters_list:
        for num_points_per_cluster in num_points_per_cluster_list:
            system_str = " ".join(["python DPMB_basic.py",str(num_clusters),str(num_points_per_cluster),path
                                ,"_".join([">LogFiles/job",str(num_clusters),str(num_points_per_cluster)])])
            os.system(system_str)
