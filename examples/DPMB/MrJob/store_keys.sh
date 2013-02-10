#!/usr/bin/bash
if [ -z $1 ]; then
    echo "store_keys.sh cluster_name"
    exit
fi

cluster_name=$1

#store keys so automation works
starcluster listclusters $cluster_name | grep ' ec2-' \
    | perl -pe 's/^.*(ec2.*com).*$/$1/' \
    | xargs -I{} ssh -o StrictHostKeyChecking=no \
    -i ~/.ssh/dlovell_mitpcp.pem sgeadmin@{} 'hostname'
# can have issue if key already exists
# ssh-keygen -f "/home/dlovell/.ssh/known_hosts" -R ec2-23-22-73-192.compute-1.amazonaws.com
# sed -i [offending_line_number]d ~/.ssh/known_hosts
