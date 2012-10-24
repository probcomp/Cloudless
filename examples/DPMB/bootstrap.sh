#!/bin/bash
###################
#configuration here
####################
bucketname="mitpcp-dpmb"
install_dir=/home/hadoop/
site_packages_dir=/usr/local/lib/python2.7/site-packages
cloudless_dir=${site_packages_dir}/Cloudless/
python_binary="Python-2.7.3"
python_dir="${install_dir}/${python_binary}/"
python_include="${python_dir}/Include/"
PYTHONPATH=$install_dir
aws_access_key_id=AWS_ACCESS_KEY_ID
aws_secret_access_key=AWS_SECRET_ACCESS_KEY
##########################
#first we set two vars...I had errors without this
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LD_RUN_PATH=/usr/local/lib:$LD_RUN_PATH

# install a few things to satisfy dependencies
ubuntu_package_names=(
    libssl-dev        # for httplib.HTTPSConnection?
    libatlas-base-dev
    python-matplotlib # to get ALL its dependencies
    libfreetype6-dev  # for matplotlib when compiling from source?
    git-core
    git=1:1.7.2.5-3   # must upgrade, else can't get branch
)
for package_name in ${ubuntu_package_names[*]} ; do
    sudo apt-get install -y $package_name
done

# install a recent python 
cd $install_dir
#
hadoop_full_path="s3://${bucketname}/emr_resources/python_binaries/"
filename="${python_binary}.tar.gz"
hadoop fs -get "${hadoop_full_path}$filename" "$filename"
tar xvfz $filename
cd $python_binary
sudo ./configure
sudo make install
cd ..
# special case for python binary
sudo rm /usr/bin/python
sudo ln -s /usr/bin/python2.7 /usr/bin/python

# some things must be installed post python install
ubuntu_package_names=(
    python-dateutil   # for pandas
    python-boto
    python-setuptools
)
for package_name in ${ubuntu_package_names[*]} ; do
    sudo apt-get install -y $package_name
done

# install from source to get particular versions
hadoop_full_path="s3://${bucketname}/emr_resources/python_packages/"
python_package_names=(
    Cython-0.17
    mrjob-0.3.5
    numpy-1.6.2
    matplotlib-1.1.1
    scipy-0.11.0rc2
    # pandas-0.7.0rc1
)
for package_name in ${python_package_names[*]} ; do
    filename="${package_name}.tar.gz"
    hadoop fs -get "${hadoop_full_path}$filename" "$filename"
    tar xvfz $filename
    cd $package_name
    sudo python setup.py install
    cd ..
done

# easy install to register packages properly?
wget http://peak.telecommunity.com/dist/ez_setup.py
sudo python ez_setup.py
#
reinstall_package_names=(
    cython
    mrjob
    numpy
    matplotlib
    scipy
    # pandas
)
for package_name in ${reinstall_package_names[*]} ; do
    sudo easy_install $package_name
done

# get Cloudless code
# MUST install to site_packages_dir, HADOOP doesn't pay attention to PYTHONPATH
cd $site_packages_dir
sudo git clone -b mrjobify git://github.com/mit-probabilistic-computing-project/Cloudless.git
sudo chown -R hadoop Cloudless

# compile cython code
cd "${cloudless_dir}/examples/DPMB/"
cython -a -I "${python_include}" pyx_functions.pyx
gcc -fPIC -o pyx_functions.so -shared -pthread -I${python_dir} -I${python_include} pyx_functions.c
