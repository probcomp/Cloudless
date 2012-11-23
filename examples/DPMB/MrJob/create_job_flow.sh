#!/bin/bash
# usage: bash create_job_flow.sh num_ec2_core_instances
if [ -z $1 ] ; then
    echo "usage: bash create_job_flow.sh num_ec2_core_instances";
    exit;
fi

# settings
num_ec2_core_instances=$1
if [ $num_ec2_core_instances == 1 ] ; then
    num_ec2_instances=1
else
    num_ec2_instances=$(( $1 + 1 ))
fi
#
ec2_instance_type=c1.xlarge
ec2_master_instance_type=c1.xlarge
if [ ! -z $2 ] ; then
    ec2_instance_type=$2
fi
if [ ! -z $3 ] ; then
    ec2_master_instance_type=$3
fi

# spin up a job flow
cd /home/dlovell/mrjob/ # this is where I pulled mrjob.git down to
python -m /home/dlovell/mrjob/mrjob/tools/emr/create_job_flow \
    --num-ec2-instances $num_ec2_instances \
    --ec2-instance-type $ec2_instance_type \
    --bootstrap-action $DPMB/bootstrap.sh \
    --bootstrap-action $DPMB/core_dump_setup.sh \
    --ec2-master-instance-type $ec2_master_instance_type >> create_job_flow.out

emr_job_flow_id=$(tail -n 1 create_job_flow.out)
echo "ec2_instance_type: $ec2_instance_type"
echo "ec2_master_instance_type: $ec2_master_instance_type"
echo "num_ec2_instances: $num_ec2_instances"
echo "emr_job_flow_id: $emr_job_flow_id"
