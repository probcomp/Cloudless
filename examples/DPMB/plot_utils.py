import numpy
import pylab
import scipy.special as ss
import matplotlib.gridspec as gridspec
#
import DPMB as dm
reload(dm)
import DPMB_State as ds
reload(ds)



def get_gridspec(height_ratios_or_int):
    height_ratios = height_ratios_or_int
    if isinstance(height_ratios, int):
        height_ratios = numpy.repeat(1./height_ratios, height_ratios)
    return gridspec.GridSpec(len(height_ratios), 1, height_ratios=height_ratios)

def legend_outside(ax=None, bbox_to_anchor=(0.5, -.25), loc='upper center'):
    # labels must be set in original plot call: plot(..., label=label)
    if ax is None:
        ax = pylab.gca()
    handles, labels = ax.get_legend_handles_labels()
    ncol = len(labels)
    lgd = ax.legend(handles, labels, loc=loc, ncol=ncol,
    	            bbox_to_anchor=bbox_to_anchor)

def savefig_legend_outside(namestr, ax=None, bbox_inches='tight'):
    # issue with subplot: which axis to take?  this assumes use the last axis
    if ax is None:
        ax = pylab.gca()
    lgd = ax.get_legend()
    pylab.savefig(namestr, bbox_extra_artists=(lgd,), bbox_inches=bbox_inches)

def multiplot(data, plot_tuples, title='', xlabel='', save_str=None,
              subplots_hspace=.25):
    num_tuples = len(plot_tuples)
    gs = get_gridspec(num_tuples)
    fh = pylab.figure()
    #
    for gs_i, extract_tuple in enumerate(plot_tuples):
        plot_func, ylabel = extract_tuple
        last_axes = pylab.subplot(gs[gs_i])
        plot_func(data)
        pylab.ylabel(ylabel)
    pylab.subplots_adjust(hspace=subplots_hspace)
    legend_outside(last_axes)
    first_axes = fh.get_axes()[0]
    first_axes.set_title(title)
    last_axes.set_xlabel(xlabel)
    if save_str is not None:
        savefig_legend_outside(save_str, last_axes)
    return fh
    
def visualize_mle_alpha(cluster_list=None,points_per_cluster_list=None,max_alpha=None):
    import pylab
    cluster_list = cluster_list if cluster_list is not None else 10**numpy.arange(0,4,.5)
    points_per_cluster_list = points_per_cluster_list if points_per_cluster_list is not None else 10**numpy.arange(0,4,.5)
    max_alpha = max_alpha if max_alpha is not None else int(1E4)
    ##
    mle_vals = []
    for clusters in cluster_list:
        for points_per_cluster in points_per_cluster_list:
            mle_vals.append([clusters,points_per_cluster,dm.mle_alpha(clusters,points_per_cluster,max_alpha=max_alpha)])
    ##
    mle_vals = numpy.array(mle_vals)
    pylab.figure()
    pylab.loglog(mle_vals[:,0],mle_vals[:,1],color='white') ##just create the axes
    pylab.xlabel("clusters")
    pylab.ylabel("points_per_cluster")
    pylab.title("MLE alpha for a given data configuration\nmax alpha: "+str(max_alpha-1))
    for clusters,points_per_cluster,mle in mle_vals:
        pylab.text(clusters,points_per_cluster,str(int(mle)),color='red')
