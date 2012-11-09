#!/bin/bash


# settings
num_cols=256
gen_seed=0
beta_d=2.0
num_iters=10
num_clusters=64
num_iters_per_step_list="1 4"
num_nodes_list="2 4"
#
num_rows=16384
infer_seed_list="0 1 2 3"

function get_logbase {
    infer_seed=$1
    num_iters_per_step=$2
    if [ -z $num_iters_per_step ] ; then
	num_iters_per_step=1
	fi
    echo "seed${infer_seed}_he${num_iters_per_step}"
}

for infer_seed in ${infer_seed_list[*]}; do
    for num_iters_per_step in ${num_iters_per_step_list[*]}; do
	for num_nodes in ${num_nodes_list[*]}; do
	    logbase=$(get_logbase $infer_seed $num_iters_per_step)
	    python programmatic_mrjob.py --infer_seed $infer_seed --num_iters_per_step $num_iters_per_step \
		$gen_seed $num_rows $num_cols $num_clusters $beta_d $num_iters $num_nodes \
		> "${logbase}.out" 2> "${logbase}.err"
	done
    done
done

num_nodes=1
for infer_seed in ${infer_seed_list[*]}; do
    logbase=$(get_logbase $infer_seed $num_iters_per_step)
    echo "Kicking off infer_seed=$infer_seed"
    python programmatic_mrjob.py --infer_seed $infer_seed \
	$gen_seed $num_rows $num_cols $num_clusters $beta_d $num_iters $num_nodes \
	> "${logbase}.out" 2> "${logbase}.err" &
    sleep 30 # must sleep since mjrob names folders by time?
done
wait
