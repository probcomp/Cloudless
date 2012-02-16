from starcluster.clustersetup import ClusterSetup
from starcluster.logger import log

class CloudlessSetup(ClusterSetup):
     def __init__(self):
         # TODO: Could be generalized to "install a python package plugin"
         pass

     def run(self, nodes, master, user, user_shell, volumes):
         # TODO: Shouldn't depend on manually refrencing python2.7; will break
         #       in later versions
         # NOTE: node includes master
          for node in nodes:
               log.info("Installing Cloudless on %s" % node.alias)
               node.ssh.execute('git clone git://github.com/mit-probabilistic-computing-project/Cloudless.git')
               node.ssh.execute('rm -rf /usr/local/lib/python2.7/dist-packages/Cloudless')
               node.ssh.execute('mv Cloudless /usr/local/lib/python2.7/dist-packages')

