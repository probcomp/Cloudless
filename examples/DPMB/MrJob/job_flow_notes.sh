emr_job_flow_id=SOMESTRING

# this is supposed to run a command on all nodes but doesn't 
python -m mrjob.tools.emr.mrboss $emr_job_flow_id -v \
    'cd /usr/local/lib/python2.7/site-packages/Cloudless/ && git pull'

# tried adding ec2-key-pair-file, but don't think it changed anything 
python -m mrjob.tools.emr.mrboss  j-2DLB05ODZ9ASD -v \
    --ec2-key-pair-file=/home/hadoop/.ssh/dlovell.pem \
    '/home/hadoop/bin/hadoop job  -Dmapred.job.tracker=10.114.17.221:9001 -kill job_201211102241_0004'

# this pulls down logs, but they are often unuseful in debugging
python -m mrjob.tools.emr.fetch_logs -a $emr_job_flow_id | less


master_ip=ecXX-XX-XX-XX-XX.compute-1.amazonaws.com
# to connect to workers, must go through master
# so push up key pair file to master 
scp -i ~/.ssh/dlovell.pem ~/.ssh/dlovell.pem hadoop@$master_ip:/home/hadoop/.ssh/
ssh -i ~/.ssh/dlovell.pem hadoop@$master_ip

# on master, specify slave ips
slave_ips=(
    ecXX-XX-XX-XX-XX.compute-1.amazonaws.com
    ecYY-YY-YY-YY-YY.compute-1.amazonaws.com
    ...
    ecZZ-ZZ-ZZ-ZZ-ZZ.compute-1.amazonaws.com
)
# and then ssh in to desired worker
ssh -i ~/.ssh/dlovell.pem ${slave_ips[0]}


# manually pull changes down to master, workers
# do this from local machine
scp -i ~/.ssh/dlovell.pem ~/.ssh/dlovell.pem hadoop@$master_ip:/home/hadoop/.ssh/
ssh -i ~/.ssh/dlovell.pem hadoop@$master_ip 'cd /usr/local/lib/python2.7/site-packages/Cloudless && git pull'
# then ssh into master to run on slaves
ssh -i ~/.ssh/dlovell.pem hadoop@$master_ip
for slave_ip in ${slave_ips[*]} ; do
    ssh -i ~/.ssh/dlovell.pem $slave_ip 'cd /usr/local/lib/python2.7/site-packages/Cloudless && git pull'
done
