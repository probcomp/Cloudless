#!bash
# USAGE: generate_scoring.sh SLEEP_DUR NUM_WORKERS BASE_DIR
if [ -z $3 ] ; then
    echo "USAGE: generate_scoring.sh SLEEP_DUR NUM_WORKERS BASE_DIR"
    exit
fi


sleep_dur=$1
num_workers=$2
base_dir=$3
run_dir_list=($(ls $base_dir | grep ^new_prog))


function process_single_run_dir() {
    sleep_dur=$1
    num_workers=$2
    run_dir=$3
    echo "`date` :: starting :: process_single_run_dir $sleep_dur $num_workers $run_dir"
    python generate_scoring.py $run_dir --is_controller --num_workers 0 --do_create_queue --do_clear_queue
    sleep $sleep_dur
    for worker_idx in $(seq $num_workers); do
	python generate_scoring.py $run_dir &
    done
    wait
    echo "`date` :: finished :: process_single_run_dir $sleep_dur $num_workers $run_dir"
}

while true; do
    for run_dir in ${run_dir_list[*]}; do
	process_single_run_dir $sleep_dur $num_workers $run_dir
    done
done
