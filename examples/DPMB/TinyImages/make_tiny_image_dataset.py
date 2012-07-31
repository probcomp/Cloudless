#!python
import os
import csv
import random
#
import Cloudless.examples.DPMB.TinyImages.loadTinyImages as lti
import Cloudless.examples.DPMB.remote_functions as rf


dest_dir = '/media/VonNeumann/TinyImagePieces'
n_images = 1000*1000
n_pieces = 100
images_per_piece = n_images/n_pieces
seed = 0

random.seed(seed)
image_indices = random.sample(xrange(lti.total_num_images),n_images)
indices_list_list = []
for piece_idx in xrange(n_pieces):
    indices_list_list.append(
        image_indices[
            (piece_idx*images_per_piece):((piece_idx+1)*images_per_piece)
            ])

if not os.path.isdir(dest_dir): os.makedirs(dest_dir)
with open(os.path.join(dest_dir,'all_pieces_indices'),'w') as fh:
    csv_writer = csv.writer(fh)
    for index in image_indices:
        csv_writer.writerow([index])

for piece_idx in xrange(n_pieces):
    piece_path = os.path.join(dest_dir,'piece_'+str(piece_idx)+'_indices')
    with open(piece_path,'w') as fh:
        csv_writer = csv.writer(fh)
        for index in indices_list_list[piece_idx]:
            csv_writer.writerow([index])

for piece_idx, piece_indices in enumerate(indices_list_list):
    pkl_file_name = os.path.join(dest_dir,'piece_'+str(piece_idx)+'_data')
    if os.path.isfile(pkl_file_name): continue
    image_list, image_indices = lti.read_images(
        n_images=None,image_indices=piece_indices)
    rf.pickle({'image_list':image_list,'image_indices':image_indices},
              pkl_file_name)
    
