#!python
import os
#
from starcluster.clustersetup import ClusterSetup
from starcluster.logger import log
import Cloudless.examples.DPMB.settings as settings

cloudless_dir = '/usr/local/lib/python2.7/dist-packages/Cloudless'
git_repo = 'git://github.com/mit-probabilistic-computing-project/Cloudless.git'
hadoop_dir = '/etc/hadoop-0.20/conf/'
remote_home_dir = '/home/sgeadmin/'

class CloudlessSetup(ClusterSetup):
     def __init__(self):
         # TODO: Could be generalized to "install a python package plugin"
         pass

     def run(self, nodes, master, user, user_shell, volumes):
          # TODO: Shouldn't depend on manually refrencing python2.7; will break
          #       in later versions
          # NOTE: node includes master
          # master.apt_install("python-gdata")
          for node in nodes:
               log.info("Copying up boto file on %s" % node.alias)
               boto_full_file = os.path.join(remote_home_dir,'.boto')
               node.ssh.put(settings.s3.ec2_credentials_file,remote_home_dir)
               node.ssh.execute('chmod -R ugo+rwx ' + boto_full_file)
          for node in nodes:
               log.info("Installing Cloudless on %s" % node.alias)
               #
               node.ssh.execute('git clone ' + git_repo)
               node.ssh.execute('rm -rf ' + cloudless_dir)
               node.ssh.execute(
                    'mv Cloudless /usr/local/lib/python2.7/dist-packages')
               ##
               node.ssh.execute(
                    'cd ' + cloudless_dir + ' && git checkout mrjobify')
               #
               # start swap creation ASAP
               log.info("Starting swap creation on %s" % node.alias)
               node.ssh.execute('chown sgeadmin /mnt')
               node.ssh.execute_async(
                    'bash ' + os.path.join(cloudless_dir,'make_swap.sh'))
               #
          for node in nodes:
               node.ssh.execute('easy_install scikits.learn')
               node.ssh.execute('apt-get install -y python-h5py')
               node.ssh.execute(
                    'python -c \'import Cloudless.examples.DPMB.settings\'')
               node.ssh.execute('python -c \'import matplotlib\'')
               node.ssh.execute('chmod -R ugo+rwx ' + cloudless_dir)
               node.ssh.execute('if [ ! -d /mnt/TinyImages ] ; '
                                'then mkdir /mnt/TinyImages ; fi')
               node.ssh.execute('chown sgeadmin /mnt/TinyImages')
               #
               filename_tuples = [(hadoop_dir, 'core-site.xml'),
                                  (hadoop_dir, 'mapred-site.xml'),
                                  (remote_home_dir, '.mrjob.conf')]
               for dest_dir, filename in filename_tuples:
                    source_filename = os.path.join(cloudless_dir,
                                                   filename + '_for_hadoop')
                    dest_filename = os.path.join(dest_dir, filename)
                    cmd_str = ' '.join(['cp', source_filename, dest_filename])
                    node.ssh.execute(cmd_str)
               #
               core_site = os.path.join(cloudless_dir, 'update_core_site.py')
               cmd_str = ' '.join(['python', core_site, '--boto_file', boto_full_file])
               node.ssh.execute(cmd_str)
