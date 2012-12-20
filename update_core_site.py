import sys
import os
import argparse
import fileinput
import ConfigParser
#
import Cloudless.examples.DPMB.helper_functions as hf


def surround(tagstr, middle):
    front = '<' + tagstr + '>'
    back = '</' + tagstr + '>'
    return front + middle + back

def create_property(name_str, value_str):
    name_el = surround('name', name_str)
    value_el = surround('value', value_str)
    return surround('property', name_el + value_el)

if __name__ == '__main__':
    default_boto_file = os.path.expanduser('~/.boto')
    default_core_site_file = '/etc/hadoop-0.20/conf/core-site.xml'
    parser = argparse.ArgumentParser()
    parser.add_argument('--boto_file', type=str, default=default_boto_file)
    parser.add_argument('--core-site-file', type=str, default=default_core_site_file)
    args = parser.parse_args()
    boto_file = args.boto_file
    core_site_file = args.core_site_file
    #
    boto_credentials = hf.get_boto_credentials(boto_file)
    id_key_property = create_property(
        'fs.s3n.awsAccessKeyId',
        boto_credentials['aws_access_key_id'])
    secret_key_property = create_property(
        'fs.s3n.awsSecretAccessKey',
        boto_credentials['aws_secret_access_key'])
    #
    to_replace = '</configuration>'
    replace_with = '\n'.join([id_key_property, secret_key_property, to_replace])
    #
    for i, line in enumerate(fileinput.input(core_site_file, inplace = 1)):
        sys.stdout.write(line.replace(to_replace, replace_with))
