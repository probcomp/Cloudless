#! bash
mrjob_dir="$DPMB/MrJob/"
data_dir="/tmp/"
inferer_script="${mrjob_dir}/seed_inferer.py"
consolidate_script="${mrjob_dir}/consolidate_summaries.py"

cd $data_dir
printf "1\n" > seed_list.txt
printf "using seed_list.txt:\n`cat seed_list.txt`\n"
# python "${inferer_script}" --jobconf mapred.map.tasks=4 --num-iters 8 --num-nodes 4 \
#     < seed_list.txt >/dev/null
# python "${inferer_script}" --jobconf mapred.map.tasks=2 --num-iters 4 --num-nodes 2 \
#     < seed_list.txt >/dev/null
python "${inferer_script}" --jobconf mapred.map.tasks=1 --num-iters 2 --num-nodes 1 \
    < seed_list.txt >/dev/null

python -i "${consolidate_script}" "$data_dir"
