import sys
import os
import argparse
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

    def verify_file(self,filename,write_s3=False):
        success = None
        if not self.is_local(filename):
            success = self.get_s3(filename)
        elif not self.on_s3(filename) and write_s3:
            success = self.put_s3(filename)
        else:
            success = True # we have it and s3 has it
        return success

    def verify_files(self,file_list,write_s3=False):
        results = {}
        for filename in file_list:
            results[filename] = self.verify_file(filename,write_s3)
        return results

def main():
    parser = argparse.ArgumentParser('s3_helper')
    parser.add_argument('filenames',nargs='+',type=str)
    args = parser.parse_args()
    filenames = args.filenames
    
    s3 = S3_helper()
    for filename in filenames:
        try:
            s3.put_s3(filename)
        except Exception, e:
            except_str = 'failed to put_s3(' + filename + ')'
            print except_str
            print str(e)
    return s3, filenames

if __name__ == '__main__':
    s3, filenames = main()
