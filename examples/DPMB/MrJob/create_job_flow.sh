#!/bin/bash


ec2_instance_type=c1.medium
# settings
num_ec2_instances=5
num_iters=10
num_iters_per_step=4
#
num_workers=$(( $num_ec2_instances - 1 ))
num_nodes=$(( $num_workers * 2 ))
num_map_tasks=$(( $num_nodes + 1 ))

# spin up a job flow
cd /home/dlovell/mrjob/ # this is where I pulled mrjob.git down to
python -m /home/dlovell/mrjob/mrjob/tools/emr/create_job_flow \
    --num-ec2-instances $num_ec2_instances \
    --bootstrap-action $DPMB/bootstrap.sh \
    > create_job_flow.out
emr_job_flow_id=$(tail -n 1 create_job_flow.out)
# --ec2-instance-type $ec2_instance_type \

# emr_job_flow_id="j-1HLTP8JEUD5BD" # 4 workers = 8 nodes

cd $DPMB/MrJob
run_dir=programmatic_mrjob_437b056c5a
seed_list=seed_list.txt
python $DPMB/MrJob/seed_inferer.py -r emr --emr-job-flow-id=$emr_job_flow_id \
    --pool-wait-minutes 30 \
    --jobconf mapred.map.tasks=$num_map_tasks --num-nodes $num_nodes \
    --resume-file gibbs_init.pkl.gz --problem-file problem.pkl.gz \
    --run_dir $run_dir \
    --num-iters $num_iters --num-iters-per-step $num_iters_per_step \
    --push_to_s3 \
    $seed_list > out

# python -m mrjob.tools.emr.mrboss $emr_job_flow_id -v 'cd /usr/local/lib/python2.7/site-packages/Cloudless/ && git pull'
# python -m mrjob.tools.emr.fetch_logs -a $emr_job_flow_id | less

# master_ip=ec2-23-21-28-54.compute-1.amazonaws.com
# scp -i ~/.ssh/dlovell.pem ~/.ssh/dlovell.pem hadoop@$master_ip:/home/hadoop/.ssh/
# ssh -i ~/.ssh/dlovell.pem hadoop@$master_ip

# slave_ips=(
#     ec2-50-19-204-231.compute-1.amazonaws.com
#     ec2-23-22-85-58.compute-1.amazonaws.com
#     ec2-54-242-66-133.compute-1.amazonaws.com
#     ec2-72-44-54-14.compute-1.amazonaws.com
# )
# ssh -i ~/.ssh/dlovell.pem hadoop@${slave_ips[0]}

# for slave_ip in ${slave_ips[*]} ; do
#     ssh -i ~/.ssh/dlovell.pem $slave_ip 'cd /usr/local/lib/python2.7/site-packages/Cloudless && git pull'
# done