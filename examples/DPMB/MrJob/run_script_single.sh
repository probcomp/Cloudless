#!/usr/bin/bash
if [ -z $4 ]; then
    echo "run_script.sh hexdigest seed cluster_name num_tasks"
    exit
fi

# most changed settings
hexdigest=$1
seed=$2
cluster_name=$3
num_tasks=$4
#
num_iters=50
code_dir=/usr/local/lib/python2.7/dist-packages/Cloudless/examples/DPMB/MrJob/
seed_file=seed_${seed}.txt

# create seed file
starcluster sshmaster $cluster_name -u sgeadmin "echo $seed > ${seed_file}"

# BE SURE TO REMOVE PREVIOUS PROBLEM
starcluster sshmaster $cluster_name -u sgeadmin "rm problem.{pkl.gz,h5}"

# kick off first job, give it 120 seconds to download the problem file
echo "`date`:: initiating seed $seed"
nohup starcluster sshmaster $cluster_name -u sgeadmin "nohup python ${code_dir}seed_inferer.py $seed_file -v -r hadoop --num-iters $num_iters --push_to_s3 --run_dir new_programmatic_mrjob_${hexdigest} --file problem.pkl.gz --file problem.h5 --num-nodes $num_tasks --jobconf mapred.tasktracker.expiry.interval=60000000 --jobconf mapred.task.timeout=60000000 --jobconf mapred.task.limit.maxvmem=-1 --jobconf mapred.child.java.opts=-Xmx1536M --jobconf cleanup=NONE --cleanup=NONE >seed_inferer_${hexdigest}_seed${seed}.out 2>seed_inferer_${hexdigest}_seed${seed}.err &" &

#  --resume-file summary_numnodes8_seed0_he1_iternum-1.pkl.gz

# to kill jobs
# > ls seed_*err | xargs -n 1 bash -c 'tail -n 1000 $1 | grep kill | tail -n 1' --
# can automate killing with something like
# for job_id in $(seq 1747 1776); do /usr/lib/hadoop-0.20/bin/../bin/hadoop job  -Dmapred.job.tracker=master:54311 -kill job_201212280012_$job_id; done
