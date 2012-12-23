run_dir_list=($(ls /mnt/ | grep ^new_prog))

sleep_dur=60
num_workers=3

function process_run_dir() {
    sleep_dur=$1
    num_workers=$2
    run_dir=$3
    echo "`date` :: starting :: process_run_dir $sleep_dur $num_workers $run_dir"
    python generate_scoring.py $run_dir --is_controller --num_workers 0 --do_create_queue
    sleep $sleep_dur
    for worker_idx in $(seq $num_workers); do
	python generate_scoring.py $run_dir &
    done
    wait
    echo "`date` :: finished :: process_run_dir $sleep_dur $num_workers $run_dir"
}

while true; do
    for run_dir in ${run_dir_list[*]}; do
	process_run_dir $sleep_dur $num_workers $run_dir
    done
done
