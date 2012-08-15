import seed_inferer as si
reload(si)

problem_file = 'tiny_image_problem_nImages_' \
    '20000_nPcaTrain_10000.pkl.gz'

msi = si.MRSeedInferer()
msi.num_steps = 1
msi.num_iters = 0
msi.num_nodes = 1
msi.problem_file = problem_file
#
initer = msi.init('0','0')
run_key, yielded_tuple = initer.next()
