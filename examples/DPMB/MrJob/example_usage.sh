#! bash
printf "1\n2\n" > seed_list.txt
python seed_inferer_parallel.py --num-steps 2 --num-iters 4 --num-nodes 1 < seed_list.txt > output_sequential.txt &
python seed_inferer_parallel.py --num-steps 1 --num-iters 4 --num-nodes 1 < seed_list.txt > output_single.txt &
wait
python -i output_reader.py
