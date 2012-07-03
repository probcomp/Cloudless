#! bash
printf "1\n2\n" > seed_list.txt
printf "using seed_list.txt:\n`cat seed_list.txt`\n"
python seed_inferer.py --jobconf mapred.map.tasks=2 --num-steps 2 --num-iters 4 --num-nodes 1 < seed_list.txt > output_sequential.txt &
python seed_inferer.py --jobconf mapred.map.tasks=2 --num-steps 1 --num-iters 4 --num-nodes 1 < seed_list.txt > output_single.txt &
wait
python -i output_reader.py
