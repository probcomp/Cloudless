import os
#
import h5py
import numpy
#
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless.examples.DPMB.settings as settings
reload(settings)


data_dir = settings.data_dir
get_h5_name_from_pkl_name = lambda filename: \
    os.path.splitext(os.path.splitext(filename)[0])[0] + '.h5'
is_problem_file = lambda filename: filename[:18] == 'tiny_image_problem'
dir_contents = os.listdir(data_dir)
problem_files = filter(is_problem_file, dir_contents)

for problem_file in problem_files:
  h5_file = get_h5_name_from_pkl_name(problem_file)

  with hf.Timer('unpickle', verbose=True) as unpickle_timer:
    problem = rf.unpickle(problem_file, dir=data_dir)

  with hf.Timer('create_store', verbose=True) as create_store_timer:
    with rf.h5_context(h5_file, dir=data_dir) as my_h5:
      for key, value in problem.iteritems():
        rf.h5ify(key, value, my_h5)

# # # how to read
# store_contents = dict()
# with hf.Timer('read_store', verbose=True) as create_store_timer:
#   with rf.h5_context(h5_file, dir=data_dir) as my_h5:
#     for key, value in my_h5.iteritems():
#       store_contents[key] = rf.unh5ify(key, my_h5)

# # how to operate on summaries
# summary_file = 'summary_numnodes2_seed1_iternum2.pkl.gz'
# with hf.Timer('unpickle', verbose=True) as unpickle_timer:
#   summary = rf.unpickle(summary_file, dir=data_dir)

# h5_file = get_h5_name_from_pkl_name(summary_file)
# with rf.h5_context(h5_file, dir=data_dir) as my_h5:
#   rf.h5ify_dict(summary, my_h5)

# with rf.h5_context(h5_file, dir=data_dir) as my_h5:
#   store_dict = rf.unh5ify_dict(my_h5)



# unpickle took:	    657 ms
# create_store took:	     16 ms
# pickle took:	   1946 ms
# pickle_gz took:	  37908 ms
# read_store took:	     10 ms
# dlovell@jaynes:/usr/local/Cloudless/examples/DPMB/Data$ du -sh store.h5 test.*
# 8.6M    store.h5
# 27M     test.pkl
# 8.1M    test.pkl.gz
