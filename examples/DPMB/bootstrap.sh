#!/bin/bash
###################
#configuration here
####################
bucketname="mitpcp-dpmb"
##########################
install_dir=/home/hadoop/
cloudless_dir=/home/hadoop/Cloudless/
cd install_dir
git clone git@github.com:mit-probabilistic-computing-project/Cloudless.git
PYTHONPATH=$install_dir
#first we set two vars...I had errors without this
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LD_RUN_PATH=/usr/local/lib:$LD_RUN_PATH

# install a few things to satisfy dependencies
sudo apt-get install -y libssl-dev # for httplib.HTTPSConnection?
sudo apt-get install -y libatlas-base-dev
# to get ALL its dependencies
sudo apt-get install -y python-matplotlib
# for matplotlib when compiling from source?
sudo apt-get install -y libfreetype6-dev

hadoop_full_path="s3://${bucketname}/emr_resources/python_binaries/"
python_binary="Python-2.7.3"
filename="${python_binary}.tar.gz"
hadoop fs -get "${hadoop_full_path}$filename" "$filename"
tar xvfz $filename
cd $python_binary
echo "sudo make install"
cd ..
# special case for python binary
sudo rm /usr/bin/python
sudo ln -s /usr/bin/python2.7 /usr/bin/python

hadoop_full_path="s3://${bucketname}/emr_resources/python_packages/"
package_names=(Cython-0.17.tar.gz mrjob-0.3.5.tar.gz matplotlib-1.1.1.tar.gz scipy-0.11.0rc2.tar.gz numpy-1.6.2.tar.gz pandas-0.7.0rc1.tar.gz)
for package_name in ${package_names[*]} ; do
    filename="${package_name}.tar.gz"
    hadoop fs -get "${hadoop_full_path}$filename" "$filename"
    tar xvfz $filename
    cd $package_name
    sudo make install
    cd ..
done

# install setup tools
wget http://peak.telecommunity.com/dist/ez_setup.py
sudo python ez_setup.py
# boto
sudo easy_install -y boto

# compile cython code
python_dir="${install_dir}/${python_binary}/"
python_include="${python_dir}/Include/"
cd "${cloudless_dir}/examples/DPMB/"
cython -a -I "${python_include}" pyx_functions.pyx
gcc -fPIC -o pyx_functions.so -shared -pthread -I/home/hadoop/Python-2.7.2/ -I/home/hadoop/Python-2.7.2/Include pyx_functions.c

# import gdata
# import h5py

exit
