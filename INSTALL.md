Installation
============

1. Be sure [StarCluster](http://web.mit.edu/star/cluster/) is installed (e.g. `easy_install starcluster`).
1. Be sure the directory containing [Cloudless](https://github.com/mit-probabilistic-computing-project/Cloudless) is on your `PYTHONPATH`.
1. Add to StarCluster config file (typically `~/.starcluster/config`):

        [plugin Cloudless]
        SETUP_CLASS = Cloudless.starcluster_plugin.CloudlessSetup

1. Add Cloudless to your cluster config, for example:

        plugins = ipcluster, cloudless

1. Start a cluster; all the nodes and the master should have access to Cloudless.
