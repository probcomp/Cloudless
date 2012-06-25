import argparse
import os
#
import Cloudless.examples.DPMB.settings as settings

parser = argparse.ArgumentParser(
    description='Create simlinks given a csv file specifying image index,cluster mappings')
parser.add_argument('filename',type=str)
args,unkown_args = parser.parse_known_args()

image_dir = os.path.join(settings.data_dir,settings.cifar_100_image_dir)
clustering_dir = os.path.join(settings.data_dir,settings.clustering_dir)

def create_links(filename_or_series,source_dir,dest_dir):
    series = None
    if isinstance(filename_or_series,str):
        series = pandas.Series.from_csv(filename_or_series)
    elif isinstance(filename_or_series,pandas.Series):
        series = filename_or_series
    else:
        print "unknown type for filename_or_series!"
        return
    #
    if len(os.listdir(dest_dir)) != 0:
        print dest_dir + " not empty, empty and rerun"
        return
    #
    for vector_idx,cluster_idx in series.iteritems():
        cluster_dir = os.path.join(dest_dir,str(cluster_idx))
        if not os.path.isdir(cluster_dir):
            os.mkdir(cluster_dir)
        filename = ("%05d" % vector_idx) + ".png"
        from_file = os.path.join(source_dir,filename)
        to_file = os.path.join(cluster_dir,filename)
        #
        os.symlink(from_file,to_file)

hf.create_links(args.filename,image_dir,clustering_dir)
