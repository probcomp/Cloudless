import argparse
import os
#
import Cloudless.examples.DPMB.helper_functions as hf
import Cloudless.examples.DPMB.settings as settings

parser = argparse.ArgumentParser(
    description='Create simlinks given a csv file specifying image index,cluster mappings')
parser.add_argument('filename',type=str)
args,unkown_args = parser.parse_known_args()

image_dir = os.path.join(settings.data_dir,settings.cifar_100_image_dir)
clustering_dir = os.path.join(settings.data_dir,settings.clustering_dir)

hf.create_links(args.filename,image_dir,clustering_dir)
