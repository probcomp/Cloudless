#!bash
# USAGE: generate_scoring.sh SLEEP_DUR NUM_WORKERS
if [ -z $2 ] ; then
    echo "USAGE: generate_scoring.sh SLEEP_DUR NUM_WORKERS"
    exit
fi


sleep_dur=$1
num_workers=$2
single_dir=$3

function process_single_run_dir() {
    sleep_dur=$1
    num_workers=$2
    run_dir=$3
    echo "`date` :: starting :: process_single_run_dir $sleep_dur $num_workers $run_dir"
    python generate_scoring.py $run_dir --is_controller --num_workers 0 --do_create_queue --do_clear_queue
    sleep $sleep_dur
    for worker_idx in $(seq $num_workers); do
	python generate_scoring.py $run_dir &
	sleep 1
    done
    wait
    echo "`date` :: finished :: process_single_run_dir $sleep_dur $num_workers $run_dir"
}

if [ ! -z $single_dir ]; then
    process_single_run_dir $sleep_dur $num_workers $single_dir
    exit
fi

run_dir_list=$(python generate_scoring.py --do_print_all_queues .)
while true; do
    for run_dir in ${run_dir_list[*]}; do
	process_single_run_dir $sleep_dur $num_workers $run_dir
    done
done
