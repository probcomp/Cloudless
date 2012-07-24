#! bash
printf "1\n2\n" > seed_list.txt
printf "using seed_list.txt:\n`cat seed_list.txt`\n"
python seed_inferer.py --jobconf mapred.map.tasks=4 --num-steps 2 --num-iters 8 --num-nodes 4 < seed_list.txt > num_nodes_4.txt &
python seed_inferer.py --jobconf mapred.map.tasks=2 --num-steps 4 --num-iters 8 --num-nodes 2 < seed_list.txt > num_nodes_2.txt &
python seed_inferer.py --jobconf mapred.map.tasks=1 --num-steps 8 --num-iters 8 --num-nodes 1 < seed_list.txt > num_nodes_1.txt &

wait
python -i consolidate_summaries.py
