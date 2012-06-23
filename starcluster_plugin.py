from starcluster.clustersetup import ClusterSetup
from starcluster.logger import log
import Cloudless.examples.DPMB.settings as settings

class CloudlessSetup(ClusterSetup):
     def __init__(self):
         # TODO: Could be generalized to "install a python package plugin"
         pass

     def run(self, nodes, master, user, user_shell, volumes):
         # TODO: Shouldn't depend on manually refrencing python2.7; will break
         #       in later versions
         # NOTE: node includes master
          # master.apt_install("python-gdata")
          # import os
          # os.system("starcluster put mycluster --user sgeadmin ~/google_docs_auth /home/sgeadmin/")

          master.ssh.execute('sudo swapoff /dev/xvda3')
          master.ssh.execute('sudo umount /dev/xvdm')
          master.ssh.execute('sudo mkswap /dev/xvdz')
          master.ssh.execute('sudo swapon /dev/xvdz')

          for node in nodes:
               log.info("Installing Cloudless on %s" % node.alias)
               node.ssh.execute('git clone git://github.com/mit-probabilistic-computing-project/Cloudless.git')
               node.ssh.execute('rm -rf /usr/local/lib/python2.7/dist-packages/Cloudless')
               node.ssh.execute('mv Cloudless /usr/local/lib/python2.7/dist-packages')
               ##
               node.ssh.execute('cd /usr/local/lib/python2.7/dist-packages/Cloudless/ && git checkout mrjobify')
               node.ssh.execute('chmod -R ugo+rwx /usr/local/lib/python2.7/dist-packages/Cloudless/')
               node.ssh.put(settings.ec2_credentials_file,"/home/sgeadmin/")
