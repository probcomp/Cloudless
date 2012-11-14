#!python
import os
import argparse
#
import Cloudless.examples.DPMB.s3_helper as s3h

# settings
bucket_dir_base = 'tiny_image_summaries'
local_dir_base = '/tmp'

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--bucket_dir_suffix', default='', type=str)
parser.add_argument('--filename_prefix', default='', type=str)
parser.add_argument('--filename_suffix', default='', type=str)
args = parser.parse_args()
bucket_dir_suffix = args.bucket_dir_suffix
filename_prefix = args.filename_prefix
filename_suffix = args.filename_suffix

# set up path variables
bucket_dir = os.path.join(bucket_dir_base, bucket_dir_suffix)
local_dir = os.path.join(local_dir_base, bucket_dir_suffix)

# set up puller
s3 = s3h.S3_helper(bucket_dir=bucket_dir, local_dir=local_dir)
bucket_dir_list = s3.bucket.list(prefix=bucket_dir)
bucket_els = [el for el in bucket_dir_list if el.name != bucket_dir]

# now pull
for bucket_el in bucket_els:
    filename = os.path.split(bucket_el.name)[-1]
    if not filename.startswith(filename_prefix): continue
    if not filename.endswith(filename_suffix): continue
    s3.verify_file(filename)
