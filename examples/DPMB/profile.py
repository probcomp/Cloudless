import os,sys
import pstats
##http://docs.python.org/library/profile.html#module-cProfile

num_lines = 40 if len(sys.argv) < 2 else int(sys.argv[1])

script_str = "DPMB_Diagnostics/plot_test_states_local.py"
prof_file = os.path.splitext(script_str)[0] + ".prof"

if not os.path.isfile(prof_file):
    cmd_str = ("python -m cProfile -o " + prof_file 
               + " " + script_str )
    sys_out = os.system(cmd_str)

p = pstats.Stats(prof_file)
p.sort_stats('total').print_stats(num_lines)