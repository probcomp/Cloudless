# most changed settings
num_c1m=4
hexdigest=4708ce2021
seed=10
num_iters=200

# calcalate some other things
num_nodes=$(expr 2 \* $num_c1m)
cluster_name=hadoop_on_${num_c1m}_${hexdigest}_${seed}
seq_end=$(expr $num_c1m - 1)
task_count=$(expr $num_nodes \* 2)
code_dir=/usr/local/lib/python2.7/dist-packages/Cloudless/examples/DPMB/MrJob/

# SET BLANK IF NOT RESUME
# resume_cmd="--resume-file summary_numnodes${num_nodes}_seed${seed}_he1_iternum20.pkl.gz"
resume_cmd=

# start the cluster
starcluster start -c c1m -s ${num_c1m} $cluster_name

#store keys so automation works
starcluster listclusters $cluster_name | grep ' ec2-' | perl -pe 's/^.*(ec2.*com).*$/$1/' | xargs -I{} ssh -o StrictHostKeyChecking=no -i ~/.ssh/dlovell.pem sgeadmin@{} 'hostname'
# can have issue if key already exists
# ssh-keygen -f "/home/dlovell/.ssh/known_hosts" -R ec2-23-22-73-192.compute-1.amazonaws.com

# open up windows to monitor progress.  Use sshnode so window title is nodename
for idx in $(seq 1 $seq_end); do
  xterm -geometry 75x15 -e starcluster sshnode $cluster_name node00${idx} \
    -u sgeadmin &
done
xterm -geometry 75x15 -e starcluster sshnode $cluster_name master -u sgeadmin &
xterm -geometry 75x15 -e starcluster sshnode $cluster_name master -u sgeadmin &

seed_file=seed_${seed}.txt
starcluster sshmaster $cluster_name -u sgeadmin "echo $seed > $seed_file"

nohup starcluster sshmaster $cluster_name -u sgeadmin "nohup python ${code_dir}seed_inferer.py $seed_file -v -r hadoop --num-iters $num_iters --push_to_s3 --run_dir new_programmatic_mrjob_${hexdigest} --file problem.pkl.gz --file problem.h5 $resume_cmd --num-nodes $num_nodes --jobconf mapred.map.tasks=$task_count --jobconf mapred.tasktracker.map.tasks.maximum=$task_count --jobconf mapred.task.timeout=60000000 >seed_inferer_${hexdigest}_seed${seed}.out 2>seed_inferer_${hexdigest}_seed${seed}.err &" &
