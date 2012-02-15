from starcluster.clustersetup import ClusterSetup
from starcluster.logger import log

class PackageInstaller(ClusterSetup):
     def __init__(self):
         # TODO: Could be generalized to "install a python package plugin"
         pass

     def run(self, nodes, master, user, user_shell, volumes):
         # FIXME: Does this also install on master? I hope so...
         # TODO: Shouldn't depend on manually refrencing python2.7; will break
         #       in later versions
          for node in nodes:
               log.info("Installing Cloudless on %s" % node.alias)
               node.ssh.execute('git clone git://github.com/mit-probabilistic-computing-project/Cloudless.git')
               node.ssh.execute('mv Cloudless /usr/local/lib/python2.7/dist-packages')

