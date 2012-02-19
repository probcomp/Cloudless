Installation
============

1. Be sure StarCluster is installed (e.g. `easy_install starcluster`)
1. Be sure the directory containing Cloudless is on your `PYTHONPATH`
1. Add to StarCluster config file (typically `~/.starcluster/config`):

    [plugin Cloudless]
    SETUP_CLASS = Cloudless.starcluster_plugin.CloudlessSetup

Add Cloudless to your cluster config, for example:

    plugins = ipcluster, cloudless

1. Start a cluster; all the nodes and the master should have access to Cloudless.
