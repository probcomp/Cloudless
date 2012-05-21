import os,sys
import pstats
##http://docs.python.org/library/profile.html#module-cProfile

if len(sys.argv) < 2:
    raise Exception("must pass a script argument")

script_str = sys.argv[1]
num_lines = 40 if len(sys.argv) < 3 else int(sys.argv[2])

prof_file = os.path.splitext(script_str)[0] + ".prof"
if not os.path.isfile(prof_file):
    cmd_str = ("python -m cProfile -o " + prof_file 
               + " " + script_str )
    sys_out = os.system(cmd_str)
p = pstats.Stats(prof_file)
p.sort_stats('t').print_stats(num_lines)
