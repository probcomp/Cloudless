import DPMB as dm
reload(dm)
import DPMB_State as ds
reload(ds)
import numpy as np
import scipy.special as ss

def log_conditional_to_norm_prob(logp_list):
    maxv = max(logp_list)
    scaled = [logpstar - maxv for logpstar in logp_list]
    logZ = reduce(np.logaddexp, scaled)
    logp_vec = [s - logZ for s in scaled]
    return np.exp(logp_vec)

def plot_state(state,gen_state=None,interpolation="nearest",**kwargs):
    ##sort by attributed state and then gen_state if available
    if gen_state is not None:
        mult_factor = np.round(np.log10(len(gen_state["phis"])))
        sort_by = np.array(mult_factor * state.getZIndices() + gen_state["zs"],dtype=int)
    else:
        sort_by = state.getZIndices()
    import pylab
    pylab.ion()
    ##plot the data
    fh1 = pylab.figure()
    pylab.imshow(state.getXValues()[np.argsort(sort_by)],interpolation=interpolation,**kwargs)
    ##label
    xlim = fh1.get_axes()[0].get_xlim()
    h_lines = np.array([cluster.count() for cluster in state.cluster_list]).cumsum()
    pylab.hlines(h_lines-.5,*xlim)
    ##
    ##plot the conditional posteriors
    ##alpha
    lnProdGammas = sum([ss.gammaln(cluster.count()) for cluster in state.cluster_list])
    lnPdf = lambda alpha: (ss.gammaln(alpha) + state.numClustersDyn()*np.log(alpha)
                           - ss.gammaln(alpha+state.numVectorsDyn()) + lnProdGammas)
    ##
    grid = 10.0**np.linspace(np.log10(state.alpha_min),np.log10(state.alpha_max),state.grid_N) ##endpoint should be set by MLE of all data in its own cluster?
    ##
    logp_list = []
    original_alpha = state.alpha
    for test_alpha in grid:
        state.removeAlpha(lnPdf)
        state.setAlpha(lnPdf,test_alpha)
        logp_list.append(state.score)
    ##put everything back how you found it
    state.removeAlpha(lnPdf)
    state.setAlpha(lnPdf,original_alpha)
    fh2 = pylab.figure()
    pylab.bar(np.log(grid),log_conditional_to_norm_prob(logp_list))
    pylab.title("Alpha conditional posterior")
    ##
    ##beta_i
    grid = 10.0**np.linspace(np.log10(state.beta_min),np.log10(state.beta_max),state.grid_N) ##endpoint should be set by MLE of all data in its own cluster?
    logp_list = []
    colIdx = 0
    S_list = [cluster.column_sums[colIdx] for cluster in state.cluster_list]
    R_list = [len(cluster.vectorIdxList) - cluster.column_sums[colIdx] for cluster in state.cluster_list]
    beta_d = state.betas[colIdx]
    lnPdf = lambda beta_d: sum([ss.gammaln(2*beta_d) - 2*ss.gammaln(beta_d)
                                + ss.gammaln(S+beta_d) + ss.gammaln(R+beta_d)
                                - ss.gammaln(S+R+2*beta_d) for S,R in zip(S_list,R_list)])
    logp_list = []
    ##
    original_beta = state.betas[colIdx]
    for test_beta in grid:
        state.removeBetaD(lnPdf,colIdx)
        state.setBetaD(lnPdf,colIdx,test_beta)
        logp_list.append(state.score)
    ##put everything back how you found it
    state.removeBetaD(lnPdf,colIdx)
    state.setBetaD(lnPdf,colIdx,original_beta)
    fh3 = pylab.figure()
    pylab.bar(np.log(grid),log_conditional_to_norm_prob(logp_list))
    pylab.title("Beta conditional posterior")
    ##
    return fh1,fh2,fh3

    
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

