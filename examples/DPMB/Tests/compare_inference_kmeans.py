#!python
import os
from collections import Counter
import datetime
#
import pylab
import pandas
import numpy
from scipy.cluster.vq import kmeans,kmeans2,whiten
#
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless.examples.DPMB.settings as settings
reload(settings)
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB.DPMB as dm
reload(dm)
import Cloudless.examples.DPMB.PDPMB_State as pds
reload(pds)
import Cloudless.examples.DPMB.PDPMB as pdm
reload(pdm)


# settings
problem_file = os.path.join(settings.data_dir,settings.cifar_100_problem_file)
image_dir = os.path.join(settings.data_dir,settings.cifar_100_image_dir)
clustering_dir = os.path.join(settings.data_dir,settings.clustering_dir)
#
pkl_data = rf.unpickle(problem_file)
init_x = pkl_data["subset_xs"]
true_zs,ids = hf.canonicalize_list(pkl_data["subset_zs"])
test_xs = pkl_data['test_xs']
image_indices = pkl_data['chosen_indices']

# learn via KMeans
delta_ts_list = []
kmeans_testlls_list = []
kmeans_summaries_list = []
numpy.random.seed(0)
for idx in range(10):
    start_ts = datetime.datetime.now()
    whitened_obs = whiten(init_x)
    centroids,labels = kmeans2(whitened_obs,k=50,iter=100)
    end_ts = datetime.datetime.now()
    delta_ts = end_ts-start_ts
    print "time to run k-means: " + str(end_ts-start_ts)
    Counter(labels)

    # determine predictive
    run_spec = rf.gen_default_cifar_run_spec(problem_file,0,0)
    run_spec['dataset_spec']['gen_seed'] = 0
    run_spec['dataset_spec']['num_cols'] = 256
    run_spec['dataset_spec']['num_rows'] = len(labels)
    run_spec['infer_init_z'] = labels
    kmeans_summaries = rf.infer(run_spec)

    # now stash
    delta_ts_list.append(delta_ts.total_seconds())
    kmeans_summaries_list.append(kmeans_summaries)
    kmeans_testlls_list.append(numpy.mean(kmeans_summaries[-1]['test_lls']))

# # show clusterings
# filename = os.path.join(settings.output_dir,'kmeans_results.csv')
# image_dir = os.path.join(settings.data_dir,settings.cifar_100_image_dir)
# clustering_dir = os.path.join(settings.data_dir,settings.clustering_dir)
# pandas.Series(labels,image_indices).to_csv(filename)
# hf.create_links(filename,image_dir,clustering_dir)

# read data from inference
if 'actual_inference_summaries' not in locals():
    actual_inference_summaries = rf.unpickle(
        '/usr/local/Cloudless/examples/DPMB/Tests/cifar_10_really_100/cifar_100_summaries_iter300.pkl.gz')

# comparative plots
pylab.plot(
    [summary['timing']['run_sum'] for summary in actual_inference_summaries],
    [numpy.mean(summary['test_lls']) for summary in actual_inference_summaries]
    )
#
pylab.plot(delta_ts_list,kmeans_testlls_list,'rx')
x_lim = pylab.gca().get_xlim()
y_lim = pylab.gca().get_ylim()
pylab.hlines(min(kmeans_testlls_list),*x_lim,color='r')
pylab.hlines(max(kmeans_testlls_list),*x_lim,color='r')
pylab.vlines(min(delta_ts_list),*y_lim,color='r')
pylab.vlines(max(delta_ts_list),*y_lim,color='r')
pylab.ion()
pylab.show()
#
pylab.xlabel('time (seconds)')
pylab.ylabel('predictive log likelohood')
pylab.ylabel('held out log likelihood')
pylab.legend(['inference','k-means'])
pylab.title('comparison of predictive between inference and kmeans')
