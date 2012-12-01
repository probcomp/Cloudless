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
aws_access_key_id=AKIAIAGKVNQFYGBPHBDQ
aws_secret_access_key=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
##########################
#first we set two vars...I had errors without this
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LD_RUN_PATH=/usr/local/lib:$LD_RUN_PATH
#
export JAVA_HOME=/usr/lib/jvm/java-6-sun-1.6.0.26/
echo export JAVA_HOME=/usr/lib/jvm/java-6-sun-1.6.0.26/ >> \
    /home/hadoop/conf/hadoop-env.sh
# detect architecture for python install
arch=64
arch_infix=_x64
if [ $(uname -m) = i686 ] ; then
    arch=32
    arch_infix=
fi

function echo_time {
    echo "`date`: $1"
}

# make swapfile
(
    echo_time "starting swapfile creation" >> /home/hadoop/bootstrap_progress ;
    swapfile=/mnt/swapfile ;
    dd if=/dev/zero of=$swapfile bs=1G count=10 ;
    sudo mkswap $swapfile ;
    sudo swapon $swapfile ;
    echo_time "done creating swapfile" >> /home/hadoop/bootstrap_progress ;
) &

# set up credentials
cat > /home/hadoop/.bashrc <<EOF
aws_access_key_id=$aws_access_key_id
aws_secret_access_key=$aws_secret_access_key
EOF
chown hadoop /home/hadoop/.bashrc
#
cat > /home/hadoop/.boto <<EOF
[Credentials]
aws_access_key_id = $aws_access_key_id
aws_secret_access_key = $aws_secret_access_key
EOF
chown hadoop /home/hadoop/.boto

# detect architecture
if [ $arch != 64 ] ; then
    sudo apt-get install -y htop
    echo_time "arch not 64" >> /home/hadoop/bootstrap_progress
else
    echo_time "arch is 64" >> /home/hadoop/bootstrap_progress

    # install a few things to satisfy dependencies
    ubuntu_package_names=(
	htop              # not a dependency, just for monitoring
	libssl-dev        # for httplib.HTTPSConnection?
	libatlas-base-dev
	git=1:1.7.2.5-3   # must upgrade, else can't get branch
	python-matplotlib # to get ALL its dependencies
	libfreetype6-dev
    )
    for package_name in ${ubuntu_package_names[*]} ; do
	sudo apt-get install -y $package_name
    done
    echo_time "done ubuntu packages" >> /home/hadoop/bootstrap_progress

    # install a recent python 
    cd $install_dir
    #
    hadoop_full_path="s3://${bucketname}/emr_resources/python_binaries/"
    filename="${python_binary}.installed${arch_infix}.tar.gz"
    hadoop fs -get "${hadoop_full_path}$filename" "$filename"
    tar xvfz $filename
    cd $python_binary
    sudo make clean
    sudo ./configure --enable-pydebug
    sudo make install
    cd ..
    echo_time "done python install" >> /home/hadoop/bootstrap_progress

    function install_python_package {
	package_name=$1
	hadoop_full_path="s3://${bucketname}/emr_resources/python_packages/"
	filename="${package_name}.installed.tar.gz"
	hadoop fs -get "${hadoop_full_path}$filename" "$filename"
	tar xvfz $filename
	cd $package_name
	echo_time "starting $package_name" >> /home/hadoop/bootstrap_progress
	sudo python setup.py install ;
	echo_time "done $package_name" >> /home/hadoop/bootstrap_progress
	cd ..
    }
    # install from source to get particular versions
    install_python_package Cython-0.17 &
    install_python_package mrjob-0.3.5 &
    install_python_package numpy-1.6.2
    install_python_package matplotlib-1.1.1 &
    install_python_package scipy-0.11.0rc2 &
    wait %4 %5
    echo_time "done wait on python packages" >> /home/hadoop/bootstrap_progress

    # easy install to register packages properly?
    wget http://peak.telecommunity.com/dist/ez_setup.py
    sudo python ez_setup.py
    #
    easy_install_package_names=(
	cython
	mrjob
	numpy
	matplotlib==1.1.1
	scipy
	boto
    )
    for package_name in ${easy_install_package_names[*]} ; do
	sudo easy_install $package_name
    done
    echo_time "done easy install packages" >> /home/hadoop/bootstrap_progress

    python -c 'import pylab' # to run initializatoin

    # must update for hdf5?  MUST do after everything else, else scipy fails? 
    sudo apt-get update
    sudo apt-get -f install -y gcc-4.7-base
    sudo apt-get -f install -y libquadmath0
    sudo apt-get -f install -y libhdf5-7
    sudo apt-get -f install -y libhdf5-dev
    sudo easy_install h5py
    echo_time "done h5py" >> /home/hadoop/bootstrap_progress

    # get Cloudless code
    # MUST install to site_packages_dir, HADOOP doesn't pay attention to PYTHONPATH
    cd $site_packages_dir
    sudo git clone -b mrjobify git://github.com/mit-probabilistic-computing-project/Cloudless.git
    sudo chown -R hadoop Cloudless

    # compile cython code
    extra_include=/home/hadoop/numpy-1.6.2/build/src.linux-i686-2.7/numpy/core/include/numpy/
    cd "${cloudless_dir}/examples/DPMB/"
    cython -a -I "${python_include}" pyx_functions.pyx
    gcc -fPIC -o pyx_functions.so -shared -pthread -I${python_dir} -I${python_include} -I${extra_include} pyx_functions.c
fi

sudo chown hadoop /mnt
echo_time "done bootstrap.sh" >> /home/hadoop/bootstrap_progress

# hadoop_full_path="s3://${bucketname}/tiny_image_summaries/tiny_images_1MM/"
# hadoop fs -get "${hadoop_full_path}" "/tmp/"
