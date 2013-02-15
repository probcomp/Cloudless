import os
import argparse
#
import numpy
import pylab
#
import Cloudless.examples.DPMB.TinyImages.render_images as ri
import Cloudless.examples.DPMB.remote_functions as rf


parser = argparse.ArgumentParser()
parser.add_argument('summary_filename', type=str)
parser.add_argument('--problem_filename', default='problem.pkl.gz',
                    type=str)
parser.add_argument('--dest_dir', default='RenderedImages', type=str)
parser.add_argument('--how_many_bitvectors', default=100, type=int)
parser.add_argument('--image_square_len', default=3, type=int)
#
args = parser.parse_args()
summary_filename = args.summary_filename
problem_filename = args.problem_filename
dest_dir = args.dest_dir
how_many_bitvectors = args.how_many_bitvectors
image_square_len = args.image_square_len


numpy.random.seed(0)

ri.makedirs(dest_dir)
problem = rf.unpickle(problem_filename)
summary = rf.unpickle(summary_filename)
list_of_x_indices = summary['list_of_x_indices']

cluster_lens = map(len, list_of_x_indices)
cluster_len_tuples = zip(cluster_lens, range(len(cluster_lens)))
sorted_cluster_len_tuples = sorted(cluster_len_tuples, cmp=lambda x,y: cmp(x[0],y[0]))
num_clusters = len(list_of_x_indices)

cluster_filter = lambda x: x[0] >= how_many_bitvectors
possible_cluster_tuples = filter(cluster_filter, cluster_len_tuples)
possible_cluster_indices = map(lambda x: x[1], possible_cluster_tuples)
#
how_many_images = image_square_len ** 2
montage_shape = (image_square_len, image_square_len)
which_clusters = numpy.random.randint(len(possible_cluster_indices), size=3)
which_clusters = [possible_cluster_indices[which_cluster] for which_cluster in which_clusters]
for which_cluster in which_clusters:
    basename = 'cluster_' + str(which_cluster)
    #
    image_list, image_indices, image_for_rendering_list = \
        ri.render_problem_image_indices(list_of_x_indices[which_cluster], problem)
    montage = ri.make_contact_sheet(
        image_for_rendering_list[:how_many_images], montage_shape,
        dir=dest_dir)
    montage.save(os.path.join(dest_dir, basename + '_images.png'), 'PNG')
    #
    pylab.imshow(problem['xs'][list_of_x_indices[which_cluster]][:how_many_bitvectors])
    full_figname = os.path.join(dest_dir, basename + '_binary.png')
    pylab.savefig(full_figname)

num_random = max(how_many_bitvectors, how_many_images)
random_indices_list = numpy.random.randint(len(problem['xs']), size=(3,num_random))
for random_idx, random_indices in enumerate(random_indices_list):
    basename = 'random_' + str(random_idx)
    image_list, image_indices, image_for_rendering_list = \
        ri.render_problem_image_indices(random_indices, problem)
    montage = ri.make_contact_sheet(
        image_for_rendering_list[:how_many_images], montage_shape,
        dir=dest_dir)
    montage.save(os.path.join(dest_dir, basename + '_images.png'), 'PNG')
    pylab.imshow(problem['xs'][random_indices][:how_many_bitvectors])
    full_figname = os.path.join(dest_dir, basename + '_binary.png')
    pylab.savefig(full_figname)
