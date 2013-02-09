#!/usr/bin/bash
if [ -z $1 ]; then
    echo "Usage: open_terms_to_cluster.sh cluster_name"
    exit
fi

cluster_name=$1

terms_per_col=6
XPIX=60
YPIX=10
nodenames=($(starcluster listclusters $cluster_name | grep ec2 | awk '{print $1}'))
echo "nodenames: $nodenames"

seq_end=$(expr ${#nodenames[*]} - 1)
for nodeidx in $(seq 0 $seq_end); do
    nodename=${nodenames[nodeidx]}
    echo "$nodename, $nodeidx"
    XOFF=$(expr \( $nodeidx / $terms_per_col \) \* $XPIX \* 7)
    YOFF=$(expr \( $nodeidx % $terms_per_col \) \* $YPIX \* 20)
    xterm -geometry 60x10+$XOFF+$YOFF -e starcluster sshnode $cluster_name $nodename \
      -u sgeadmin &
done
xterm -geometry 75x15 -e starcluster sshnode $cluster_name master -u sgeadmin &
