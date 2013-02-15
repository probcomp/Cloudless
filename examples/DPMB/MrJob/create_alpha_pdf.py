import itertools
#
import pylab
#
import Cloudless.examples.DPMB.helper_functions as hf
import Cloudless.examples.DPMB.plot_utils as pu


fig_suffix = 'pdf'
num_clusters_list = [128, 512, 2048]
num_rows_per_cluster_list = [1024, 2048, 4096]
log10_alphas = pylab.arange(.5, 2.5, .001)
alphas = 10 ** log10_alphas

colors = ['red', 'green', 'blue', 'black', 'orange']
linestyles = ['-', '--', ':', '-.']
#
num_clusters_to_color = dict(zip(num_clusters_list, colors))
num_rows_per_cluster_to_linestyle = dict(zip(num_rows_per_cluster_list, linestyles))

config_to_str = lambda num_clusters, num_rows_per_cluster: \
    ' x '.join(map(str, [num_clusters, num_rows_per_cluster]))
def normalize(xs, log_probs):
    log_widths = pylab.log(pylab.diff(xs))
    log_area = reduce(pylab.logaddexp, log_widths + log_probs[:-1])
    norm_log_prob = log_probs - log_area
    return norm_log_prob

pylab.figure()
mle_alpha_list = []
for num_clusters, num_rows_per_cluster in \
        itertools.product(num_clusters_list, num_rows_per_cluster_list):
    mle_alpha, alpha_ps, alphas = hf.calc_mle_alpha(
        num_clusters, num_rows_per_cluster, alphas=alphas)
    mle_alpha_list.append(mle_alpha)
    norm_alpha_ps = pylab.exp(normalize(alphas, alpha_ps))
    label = config_to_str(num_clusters, num_rows_per_cluster)
    color = num_clusters_to_color[num_clusters]
    linestyle = num_rows_per_cluster_to_linestyle[num_rows_per_cluster]
    # pylab.plot(pylab.log10(alphas), norm_alpha_ps, label=label, color=color, linestyle=linestyle)
    pylab.plot(alphas, norm_alpha_ps, label=label, color=color, linestyle=linestyle)

pylab.title(
    'ALPHA PDF FOR VARIOUS BALANCED CLUSTER CONFIGURATIONS\n'
    'legend is num_clusters x num_rows_per_cluster')
pylab.xlabel('ALPHA', size=20)
pylab.ylabel('WITHIN GRAPH NORMALIZED PROBABILITY', size=20)
# pu.legend_outside(ncol=len(num_clusters_list), cmp_func=cmp)
pylab.legend()
pylab.ion()
pylab.show()
pu.savefig_legend_outside('alpha_pdf_over_configs.' + fig_suffix)
