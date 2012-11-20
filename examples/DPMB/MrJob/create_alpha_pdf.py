import itertools
#
import pylab
#
import Cloudless.examples.DPMB.helper_functions as hf


num_clusters_list = [64, 128, 256, 512]
num_rows_per_cluster_list = [8192, 4096, 2048]

config_to_str = lambda num_clusters, num_rows_per_cluster: \
    ' x '.join(map(str, [num_clusters, num_rows_per_cluster]))
normalize = lambda log_probs: log_probs - reduce(pylab.logaddexp, log_probs)
pylab.figure()
for num_clusters, num_rows_per_cluster in \
        itertools.product(num_clusters_list, num_rows_per_cluster_list):
    mle_alpha, alpha_ps, alphas = hf.mle_alpha(
        num_clusters, num_rows_per_cluster)
    norm_alpha_ps = pylab.exp(normalize(alpha_ps))
    label = config_to_str(num_clusters, num_rows_per_cluster)
    pylab.plot(alphas, norm_alpha_ps, label=label)

pylab.title(
    'ALPHA PDF FOR VARIOUS BALANCED CLUSTER CONFIGURATIONS\n'
    'legend is num_clusters x num_rows_per_cluster')
pylab.xlabel('ALPHA')
pylab.ylabel('WITHIN GRAPH NORMALIZED PROBABILITY')
pylab.legend()
pylab.ion()
pylab.show()
pylab.savefig('alpha_pdf_over_configs.pdf')
