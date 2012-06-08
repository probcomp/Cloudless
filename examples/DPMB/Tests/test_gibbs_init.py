import matplotlib
matplotlib.use('Agg')
import numpy as np
#
import Cloudless.examples.DPMB.DPMB as dm
reload(dm)
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)

num_cols = 16
num_rows = 16

state_0 = ds.DPMB_State(
    gen_seed = 0
    , num_cols = num_cols
    , num_rows = num_rows
)
state_0.plot(save_str="random_state")

state_1 = ds.DPMB_State(
    gen_seed = 0
    , num_cols = num_cols
    , num_rows = num_rows
    , init_alpha = 1
    , init_betas = np.repeat(.1,num_cols)
    , init_z = ("balanced",4)
)
state_1.plot(save_str="balanced_state")

init_x = state_1.getXValues()
state_2 = ds.DPMB_State(
    gen_seed = 0
    , num_cols = num_cols
    , num_rows = num_rows
    , init_alpha = 1
    , init_betas = np.repeat(.1,num_cols)
    , init_z = None
    , init_x = init_x
)
state_2.plot(save_str="gibbs_init")

init_x_permuted = np.random.permutation(init_x)
state_3 = ds.DPMB_State(
    gen_seed = 0
    , num_cols = num_cols
    , num_rows = num_rows
    , init_alpha = 1
    , init_betas = np.repeat(.1,num_cols)
    , init_z = None
    , init_x = init_x_permuted
)
state_3.plot(save_str="gibbs_init_permuted")
