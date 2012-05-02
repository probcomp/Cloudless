import os
import pstats
##http://docs.python.org/library/profile.html#module-cProfile

script_str = "DPMB_Diagnostics/plot_test_states_local.py"
prof_file = os.path.splitext(script_str)[0] + ".prof"

if not os.path.isfile(prof_file):
    cmd_str = ("python -m cProfile -o " + prof_file 
               + " " + script_str )
    sys_out = os.system(cmd_str)

p = pstats.Stats(prof_file)
p.sort_stats('cumulative').print_stats(10)
