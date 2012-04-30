import DPMB_plotutils as dp
reload(dp)
import DPMB_helper_functions as hf
reload(hf)
import DPMB_State as ds
reload(ds)
import numpy as np
import matplotlib.pylab as pylab

import sys
sys.path.append("c://")
import Cloudless
reload(Cloudless)
import Cloudless.memo
reload(Cloudless.memo)

import os
dir_str = "c:/Cloudless/examples/PklFiles/"
pkl_file_list = reversed(filter(lambda x:x[-4:]==".pkl",os.listdir(dir_str)))
##pkl_file_list = filter(lambda x:x.find("16")!=-1,pkl_file_list)
for pkl_file in pkl_file_list:
    print "Trying ",pkl_file
    memoized_infer = Cloudless.memo.AsyncMemoize("infer", ["run_spec"], hf.infer, override=True) #FIXME once we've debugged, we can eliminate this override
    hf.unpickle_asyncmemoize(memoized_infer,os.path.join(dir_str,pkl_file))
    ##
    hf.try_plots(memoized_infer,which_measurements=["ari"],do_legend=False)
