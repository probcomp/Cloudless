import os
import pstats
##http://docs.python.org/library/profile.html#module-cProfile

script_str = "DPMB_Diagnostics/plot_test_states_local.py"
output_file = "profile.out"
cmd_str = "python -m cProfile -o " + output_file + " " + script_str 
os.system(cmd_str)
##
p = pstats.Stats(output_file)
p.sort_stats('cumulative').print_stats(10)
