#!python
import os
#
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

          for node in nodes:
               log.info("Installing Cloudless on %s" % node.alias)
               node.ssh.execute('git clone git://github.com/mit-probabilistic-computing-project/Cloudless.git')
               node.ssh.execute('rm -rf /usr/local/lib/python2.7/dist-packages/Cloudless')
               node.ssh.execute('mv Cloudless /usr/local/lib/python2.7/dist-packages')
               ##
               node.ssh.execute('cd /usr/local/lib/python2.7/dist-packages/Cloudless/ && git checkout mrjobify')
               node.ssh.execute('python -c \'import Cloudless.examples.DPMB.settings\'')
               node.ssh.execute('python -c \'import matplotlib\'')
               node.ssh.execute('chmod -R ugo+rwx /usr/local/lib/python2.7/dist-packages/Cloudless/')
               #
               remote_home_dir = '/home/sgeadmin/'
               node.ssh.put(settings.ec2_credentials_file,remote_home_dir)
               node.ssh.execute('chmod -R ugo+rwx ' + os.path.join(remote_home_dir,'.boto'))

          master.ssh.execute('bash /usr/local/lib/python2.7/dist-packages/Cloudless/use_ebs_swap.sh')

