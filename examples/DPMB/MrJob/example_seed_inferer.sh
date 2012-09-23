#! bash

num_rows="4096"
num_iters="3"
num_clusters="8"
beta="10.0"

mrjob_dir="$DPMB/MrJob/"
data_dir="/tmp/"
inferer_script="${mrjob_dir}/seed_inferer.py"
consolidate_script="${mrjob_dir}/consolidate_summaries.py"
problem_file="clean_balanced_data_rows_${num_rows}_cols_256_pkl.gz"

cd "${mrjob_dir}"
#
python create_synthetic_mrjob_problem.py 0 \
    "${num_rows}" 256 "${num_clusters}" "${beta}"

cd $data_dir
#
printf "0\n" > seed_list.txt
printf "using seed_list.txt:\n`cat seed_list.txt`\n"
# python "${inferer_script}" --jobconf mapred.map.tasks=4 \
#     --num-iters "${num_iters}" --num-nodes 4 \
#     --problem-file "${problem_file}" \
#     < seed_list.txt >out 2>err
python "${inferer_script}" --jobconf mapred.map.tasks=2 \
    --num-iters "${num_iters}" --num-nodes 2 \
    --problem-file "${problem_file}" \
    < seed_list.txt >out 2>err
python "${inferer_script}" --jobconf mapred.map.tasks=1 \
    --num-iters "${num_iters}" --num-nodes 1 \
    --problem-file "${problem_file}" \
    < seed_list.txt >out 2>err
#
python -i "${consolidate_script}" "$data_dir"
