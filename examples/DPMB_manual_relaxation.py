import DPMB_plotutils as dp
reload(dp)
import DPMB_helper_functions as hf
reload(hf)
import DPMB_State as ds
reload(ds)
import DPMB as dm
reload(dm)
import numpy as np
import matplotlib.pylab as pylab

#generate with a=1,beta=.1

global counts
counts = {}

def plot_helper(name, state):
    global counts
    if state not in counts:
        counts[state] = 0
    count = counts[state]
    state.plot(show=False,save_str = name + "-" + "%3d" % count + ".png")
    counts[state] += 1
    
NUM_COLS = 8
NUM_ROWS = 16

gen_state = ds.DPMB_State(gen_seed=0,
                          num_cols=NUM_COLS,
                          num_rows=NUM_ROWS,
                          init_alpha=1.0,
                          init_betas=np.repeat(0.01, NUM_COLS),
                          init_z=("balanced",4),
                          init_x=None,
                          alpha_min=1)

gen_state.plot(show=False, save_str = "ground_truth.png")

infer_state = ds.DPMB_State(gen_seed=0,
                            num_cols=NUM_COLS,
                            num_rows=NUM_ROWS,
                            init_alpha=None,
                            init_betas=None,
                            init_z=1,
                            init_x=gen_state.getXValues(),
                            alpha_min=1)

kernel = dm.DPMB(inf_seed=0, state=infer_state, infer_alpha=True, infer_beta=True)
    
#inspect p_z_0

    
plot_helper("infer", infer_state)
kernel.transition()

plot_helper("infer", infer_state)
kernel.transition()

plot_helper("infer", infer_state)
kernel.transition()

plot_helper("infer", infer_state)
kernel.transition()

plot_helper("infer", infer_state)
kernel.transition()


