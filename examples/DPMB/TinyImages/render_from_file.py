import os
#
import numpy
import pylab
#
import Cloudless.examples.DPMB.TinyImages.render_images as ri
import Cloudless.examples.DPMB.remote_functions as rf


data_dir = '/mnt/new_programmatic_mrjob_tiny_images_1MM/'
problem_filename = 'problem.pkl.gz'
problem_full_filename = os.path.join(data_dir, problem_filename)
problem = rf.unpickle(problem_full_filename)
dest_dir = 'RenderedImages'
dest_full_dir = os.path.join(data_dir, dest_dir)
ri.makedirs(dest_full_dir)

summary_filename = 'summary_numnodes128_seed1_he1_iternum3.pkl.gz'
summary_full_filename = os.path.join(data_dir, summary_filename)
summary = rf.unpickle(summary_full_filename)
list_of_x_indices = summary['list_of_x_indices']

cluster_lens = map(len, list_of_x_indices)
cluster_len_tuples = zip(cluster_lens, range(len(cluster_lens)))
sorted_cluster_len_tuples = sorted(cluster_len_tuples, cmp=lambda x,y: cmp(x[0],y[0]))
num_clusters = len(list_of_x_indices)


how_many_bitvectors = 100
cluster_filter = lambda x: x[0] >= how_many_bitvectors
possible_cluster_tuples = filter(cluster_filter, cluster_len_tuples)
possible_cluster_indices = map(lambda x: x[1], possible_cluster_tuples)
#
numpy.random.seed(0)
square_len = 3
how_many_images = square_len ** 2
which_clusters = numpy.random.randint(len(possible_cluster_indices), size=3)
which_clusters = [possible_cluster_indices[which_cluster] for which_cluster in which_clusters]
for which_cluster in which_clusters:
    basename = 'cluster_' + str(which_cluster)
    #
    image_list, image_indices, image_for_rendering_list = \
        ri.render_problem_image_indices(list_of_x_indices[which_cluster], problem)
    montage = ri.make_contact_sheet(
        image_for_rendering_list[:how_many_images], (square_len,square_len), dir=dest_full_dir)
    montage.save(os.path.join(dest_full_dir, basename + '_images.png'), 'PNG')
    #
    pylab.imshow(problem['xs'][list_of_x_indices[which_cluster]][:how_many_bitvectors])
    full_figname = os.path.join(dest_full_dir, basename + '_binary.png')
    pylab.savefig(full_figname)
