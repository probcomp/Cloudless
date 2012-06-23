import sys
import os
#
import boto
#
import Cloudless.examples.DPMB.settings as settings
reload(settings)


class S3_helper():

    def __init__(self,bucket_str=None,bucket_dir=None,local_dir=None):
        if bucket_str is None:
            bucket_str = settings.bucket_str
        if bucket_dir is None:
            bucket_dir = settings.bucket_dir
        if local_dir is None:
            local_dir = settings.data_dir
        #
        self.bucket_str = bucket_str
        self.bucket_dir = bucket_dir
        self.local_dir = local_dir
        #
        self.bucket = boto.connect_s3().get_bucket(self.bucket_str)

    def is_local(self,filename):
        full_filename = os.path.join(self.local_dir,filename)
        return os.path.isfile(full_filename)

    def on_s3(self,filename):
        key_str = os.path.join(self.bucket_dir,filename)
        key = self.bucket.get_key(key_str)
        return key is not None

    def put_s3(self,filename):
        print "putS3('"+filename+"')"
        key_str = os.path.join(self.bucket_dir,filename)
        full_filename = os.path.join(self.local_dir,filename)
        #
        key = self.bucket.new_key(key_str)
        key.set_contents_from_filename(full_filename)
        return True
        
    def get_s3(self,filename):
        print "getS3('"+filename+"')"
        key_str = os.path.join(self.bucket_dir,filename)
        full_filename = os.path.join(self.local_dir,filename)
        #
        key = self.bucket.get_key(key_str)
        success = False
        if key is not None:
            key.get_contents_to_filename(full_filename)
            success = True
        return success

    def verify_file(self,filename,overwrite_s3=False):
        success = None
        if not self.is_local(filename):
            success = self.get_s3(filename)
        elif not self.on_s3(filename) and overwrite_s3:
            success = self.put_s3(filename)
        else:
            success = True # we have it and s3 has it
        return success

    def verify_files(self,file_list,overwrite_s3=False):
        results = {}
        for filename in file_list:
            results[filename] = self.verify_file(filename,overwrite_s3)
        return results
