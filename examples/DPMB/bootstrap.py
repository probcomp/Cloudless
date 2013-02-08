#!/usr/bin/python

"""
Python version of bootstrap.sh. Untested
"""

import subprocess
from subprocess import check_output
import os

def call(*args):
    subprocess.call(*args, shell=True)

bucketname = "mitpcp-dpmb"
install_dir = '/home/hadoop/'
site_packages_dir = '/usr/local/lib/python2.7/site-packages'
cloudless_dir = '%s/Cloudless/' % site_packages_dir
python_binary = "Python-2.7.3"
python_dir = "%s/%s" % (install_dir, python_binary)
python_include = "%s/Include/" % python_dir
PYTHONPATH = install_dir
aws_access_key_id = 'AKIAIAGKVNQFYGBPHBDQ'
aws_secret_access_key = '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'

os.putenv('LD_LIBRARY_PATH', '/usr/local/lib:%s' % os.getenv('LD_LIBRARY_PATH'))
os.putenv('LD_RUN_PATH', '/usr/local/lib:%s' % os.getenv('LD_RUN_PATH'))
os.putenv('JAVA_HOME', '/usr/lib/jvm/java-6-sun-1.6.0.26/')


def append(f, s):
    with open(s, 'a') as f:
        f.write(s)


append('/home/hadoop/conf/hadoop-env.sh', "export JAVA_HOME=/usr/lib/jvm/java-6-sun-1.6.0.26/")

arch = '64'
arch_infix = '_x64'
if subprocess.check_output('uname', '-m') == 'i686':
    arch = '32'
    arch_infix = ''


def write_log(t):
    date = check_output('date')
    append('/home/hadoop/bootstrap_progress', "%s: %r" % (date, t))


write_log('starting swapfile creation')
swapfile = '/mnt/swapfile'
call('dd if=/dev/zero of=%s bs=1G count=10' % swapfile)
call('sudo mkswap %s' % swapfile)
call('sudo swapon %s' % swapfile)
write_log('done creating swapfile')
append('/home/hadoop/.bashrc', """
aws_access_key_id=$aws_access_key_id
aws_secret_access_key=$aws_secret_access_key""")
call('chown hadoop /home/hadoop/.bashrc')
append('/home/hadoop/.boto', """
aws_access_key_id = $aws_access_key_id
aws_secret_access_key = $aws_secret_access_key""")
call('chown hadoop /home/hadoop/.boto')

if os.getenv('arch') != 64:
    write_log("arch is not 64")
else:
    write_log("arch is 64")

    ubuntu_package_names = [
        'htop', # not a dependency, just for monitoring
        'libssl-dev', # for httplib.HTTPSConnection?
        'libatlas-base-dev',
        'git=1:1.7.2.5-3', # must upgrade, else can't get branch
        'python-matplotlib', # to get ALL its dependencies
        'libfreetype6-dev'
    ]

    for package_name in ubuntu_package_names:
        call('sudo apt-get install -y %s' % package_name)

    write_log('done ubuntu packages')
    os.chdir(install_dir)
    hadoop_full_path = "s3://%s/emr_resources/python_binaries" % bucketname
    filename = "%s.installed%s.tar.gz" % (python_binary, arch_infix)
    call('hadoop fs -get "%s%s" "%s"' % (hadoop_full_path, filename, filename))
    call('tar', 'xvfz', filename)
    os.chdir(python_binary)
    call('sudo make clean')
    call('sudo ./configure --enable-pydebug')
    call('sudo make install')
    os.chdir('..')
    write_log('done python install')

    def install_python_package(package_name):
        hadoop_full_path = "s3://%s/emr_resources/python_packages" % bucketname
        filename = "%s/installed.tar.gz" % package_name
        call('hadoop fs -get "%s%s" "%s"' % (hadoop_full_path, filename, filename))
        call('tar zvfz', filename)
        os.chdir(package_name)
        write_log('starting %s' % package_name)
        call('sudo python setup.py install')
        write_log('done %s' % package_name)
        os.chdir('..')

    python_pkgs = ['Cython-0.17 &', 'mrjob-0.3.5 &', 'numpy-1.6.2',
                   'matplotlib-1.1.1 &', 'scipy-0.11.0rc2 &']

    for pkg in python_pkgs:
        install_python_package(pkg)

    call('wait %4 %5') #not sure about this

    write_log('done wait on python packages')
    call('wget', 'http://peak.telecommunity.com/dist/ez_setup.py')
    call('sudo', 'python', 'ez_setup.py')

    easy_install_package_names=[
        'cython',
        'mrjob',
        'numpy',
        'matplotlib==1.1.1',
        'scipy',
        'boto'
        ]

    for pkg in easy_install_package_names:
        call('sudo','easy_install', pkg)

    write_log('done easy install packages')
    call('python','-c', "'import pylab")

    call('sudo apt-get update')
    for pkg in ['gcc-4.7-base', 'libquadmath0', 'libhdf5-7', 'libhdf5-dev']:
        call('sudo apt-get', '-f', 'install', '-y', pkg)
    call('sudo easy_install h5py')
    write_log('done h5py')

    os.chdir(site_packages_dir)
    call('sudo git clone -b mrjobify git://github.com/mit-probabilistic-computing-project/Cloudless.git')
    call('sudo chown -R hadoop Cloudless')
    extra_include='/home/hadoop/numpy-1.6.2/build/src.linux-i686-2.7/numpy/core/include/numpy/'
    os.chdir("%s/examples/DPMB" % cloudless_dir)
    call('cython -a -I "%s" pyx_functions.pyx' % python_include)
    call("gcc -fPIC -o pyx_functions.so -shared -pthread -I%s -I%s -I%s pyx_functions.c" % (python_dir, python_include, extra_include))

call('sudo chown hadoop /mnt')
write_log('done bootstrap.py')

