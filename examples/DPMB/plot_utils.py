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

def get_unique_handles_labels(handles, labels):
    handles_lookup = dict()
    for handle, label in zip(handles, labels):
        if label not in handles_lookup:
            handles_lookup[label] = handle
    unique_labels = handles_lookup.keys()
    unique_handles = handles_lookup.values()
    return unique_handles, unique_labels

def legend_outside(ax=None, bbox_to_anchor=(0.5, -.25), loc='upper center',
                   ncol=None, unique=False):
    # labels must be set in original plot call: plot(..., label=label)
    if ax is None:
        ax = pylab.gca()
    handles, labels = ax.get_legend_handles_labels()
    if unique:
        handles, labels = get_unique_handles_labels(handles, labels)
    if ncol is None:
        ncol = len(labels)
    lgd = ax.legend(handles, labels, loc=loc, ncol=ncol,
    	            bbox_to_anchor=bbox_to_anchor)

def savefig_legend_outside(namestr, ax=None, bbox_inches='tight'):
    if ax is None:
        ax = pylab.gca()
    lgd = ax.get_legend()
    try:
        pylab.savefig(
            namestr, bbox_extra_artists=(lgd,), bbox_inches=bbox_inches)
    except Exception, e:
        print e
        pylab.savefig(namestr)

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
        last_axes.set_xlim((0, last_axes.get_xlim()[-1]))
    pylab.subplots_adjust(hspace=subplots_hspace)
    legend_outside(last_axes)
    first_axes = fh.get_axes()[0]
    first_axes.set_title(title)
    last_axes.set_xlabel(xlabel)
    if save_str is not None:
        savefig_legend_outside(save_str, last_axes)
    return fh
