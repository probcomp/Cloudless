import os
import sys
import pstats
import argparse
##http://docs.python.org/library/profile.html#module-cProfile

parser = argparse.ArgumentParser(description='Profile a script')
parser.add_argument('script_to_profile',type=str)
parser.add_argument('--num_lines',default=40,type=int)
args,unknown_args = parser.parse_known_args()

script_to_profile = args.script_to_profile
num_lines = args.num_lines

script_base = os.path.splitext(script_to_profile)[0]
prof_file = script_base + ".prof"
out_file = script_base+ ".out"
if not os.path.isfile(prof_file):
    cmd_str = ("python -m cProfile -o " + prof_file 
               + " " + script_to_profile  
               + " " + " ".join(unknown_args)
               + " >" + out_file)
    print cmd_str
    sys_out = os.system(cmd_str)
else:
    print "using cached file : " + str(prof_file)
p = pstats.Stats(prof_file)
p.sort_stats('t').print_stats(num_lines)
