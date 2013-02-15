import os
#
import Cloudless.examples.DPMB.remote_functions as rf
import Cloudless.examples.DPMB.TinyImages.loadTinyImages as lti

def makedirs(dir):
    try:
        os.makedirs(dir)
    except Exception,e:
        pass

def render_problem_image_indices(problem_image_indices, problem, dir=None):
    original_image_indices = [
        problem['train_indices'][problem_image_idx]
        for problem_image_idx in problem_image_indices
        ]
    image_list, image_indices, image_for_rendering_list = lti.read_images(
        0,image_indices=original_image_indices,return_images_for_rendering=True)
    if dir:
        for image_idx_idx, image_for_rendering in enumerate(image_for_rendering_list):
            original_image_index = original_image_indices[image_idx_idx]
            lti.save_image(image_for_rendering, original_image_index, dir=dir)
    return image_list, image_indices, image_for_rendering_list

def gen_cluster_images(which_cluster):
    cluster_dir = os.path.join(data_dir, dest_dir, str(which_cluster))
    makedirs(cluster_dir)
    image_list, image_indices, image_for_rendering_list = \
        render_problem_image_indices(list_of_x_indices[which_cluster], problem, dir=cluster_dir)
    return image_list, image_indices, image_for_rendering_list

def image_from_string(image_str_data):
    sx = 32
    return lti.Image.fromstring('RGB', (sx,sx), image_str_data)

from PIL import Image
def make_contact_sheet(image_strings,(ncols,nrows),(photow,photoh)=(32,32),
                       (marl,mart,marr,marb)=(1,1,1,1),
                       padding=1, dir=''):
    """\
    Make a contact sheet from a group of filenames:
    #
    ncols        Number of columns in the contact sheet
    nrows        Number of rows in the contact sheet
    photow       The width of the photo thumbs in pixels
    photoh       The height of the photo thumbs in pixels
    #
    marl         The left margin in pixels
    mart         The top margin in pixels
    marr         The right margin in pixels
    marl         The left margin in pixels
    #
    padding      The padding between images in pixels
    #
    returns a PIL image object.
    """
    images = map(image_from_string, image_strings)
    #
    # Calculate the size of the output image, based on the
    #  photo thumb sizes, margins, and padding
    marw = marl+marr
    marh = mart+ marb
    #
    padw = (ncols-1)*padding
    padh = (nrows-1)*padding
    isize = (ncols*photow+marw+padw,nrows*photoh+marh+padh)
    #
    # Create the new image. The background doesn't have to be white
    white = (255,255,255)
    inew = Image.new('RGB',isize,white)
    #
    # Insert each thumb:
    for irow in range(nrows):
        for icol in range(ncols):
            left = marl + icol*(photow+padding)
            right = left + photow
            upper = mart + irow*(photoh+padding)
            lower = upper + photoh
            bbox = (left,upper,right,lower)
            try:
                image = images.pop(0)
                image = image.transpose(Image.ROTATE_270)
            except:
                break
            inew.paste(image,bbox)
    return inew

if __name__ == '__main__':
    data_dir = '/tmp/tiny_images_1MM_rerun'
    problem_filename = 'problem.pkl.gz'
    problem_full_filename = os.path.join(data_dir, problem_filename)
    problem = rf.unpickle(problem_full_filename)
    dest_dir = 'RenderedImages'

    summary_filename = 'summary_numnodes32_seed0_he1_iternum99.pkl.gz'
    summary_full_filename = os.path.join(data_dir, summary_filename)
    summary = rf.unpickle(summary_full_filename)
    list_of_x_indices = summary['list_of_x_indices']

    cluster_lens = map(len, list_of_x_indices)
    cluster_len_tuples = zip(cluster_lens, range(len(cluster_lens)))
    sorted_cluster_len_tuples = sorted(cluster_len_tuples, cmp=lambda x,y: cmp(x[0],y[0]))

    # which_cluster = 1526 # large
    which_cluster = 2755 # small
    square_len = 3
    how_many_images = square_len ** 2
    cluster_dir = os.path.join(data_dir, dest_dir, str(which_cluster))
    image_list, image_indices, image_for_rendering_list = gen_cluster_images(which_cluster)
    montage = make_contact_sheet(image_for_rendering_list[:how_many_images], (square_len,square_len), dir=cluster_dir)
    montage.save(os.path.join(cluster_dir, 'montage.png'), 'PNG')
    pylab.imshow(problem['xs'][list_of_x_indices[which_cluster]][:how_many_images])
    figname = 'montage_in_binary'
    full_figname = os.path.join(cluster_dir, figname)
    pylab.savefig(full_figname)
