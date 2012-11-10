#!/bin/bash
# usage: bash create_job_flow.sh num_ec2_core_instances


# parse args
if [ -z $1 ] ; then
    echo "usage: bash create_job_flow.sh num_ec2_core_instances";
    exit;
fi

# settings
ec2_instance_type=c1.medium
num_ec2_instances=$(( $1 + 1 ))

# spin up a job flow
cd /home/dlovell/mrjob/ # this is where I pulled mrjob.git down to
python -m /home/dlovell/mrjob/mrjob/tools/emr/create_job_flow \
    --num-ec2-instances $num_ec2_instances \
    --bootstrap-action $DPMB/bootstrap.sh \
    > create_job_flow.out
emr_job_flow_id=$(tail -n 1 create_job_flow.out)
echo $emr_job_flow_id
# --ec2-instance-type $ec2_instance_type \

# this is supposed to run a command on all nodes but doesn't 
# python -m mrjob.tools.emr.mrboss $emr_job_flow_id -v 'cd /usr/local/lib/python2.7/site-packages/Cloudless/ && git pull'

# this pulls down logs, but they are often unuseful in debugging
# python -m mrjob.tools.emr.fetch_logs -a $emr_job_flow_id | less

# master_ip=ecXX-XX-XX-XX-XX.compute-1.amazonaws.com
# scp -i ~/.ssh/dlovell.pem ~/.ssh/dlovell.pem hadoop@$master_ip:/home/hadoop/.ssh/
# ssh -i ~/.ssh/dlovell.pem hadoop@$master_ip

# slave_ips=(
#     ecXX-XX-XX-XX-XX.compute-1.amazonaws.com
#     ecYY-YY-YY-YY-YY.compute-1.amazonaws.com
#     ...
#     ecZZ-ZZ-ZZ-ZZ-ZZ.compute-1.amazonaws.com
# )

# manually pull changes down to master, workers
# ssh -i ~/.ssh/dlovell.pem $master_ip 'cd /usr/local/lib/python2.7/site-packages/Cloudless && git pull'
# for slave_ip in ${slave_ips[*]} ; do
#     ssh -i ~/.ssh/dlovell.pem $slave_ip 'cd /usr/local/lib/python2.7/site-packages/Cloudless && git pull'
# done

# ssh -i ~/.ssh/dlovell.pem hadoop@${slave_ips[0]}
