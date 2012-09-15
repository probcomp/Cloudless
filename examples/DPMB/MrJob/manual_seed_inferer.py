import numpy
#
import Cloudless.examples.DPMB.MrJob.seed_inferer as si
reload(si)

non_internal = lambda obj: filter(lambda x:x.find('__')==-1,dir(obj))
flatten = lambda data: [y for x in data for y in x]

msi = si.MRSeedInferer()
msi.num_steps = 2
msi.num_iters = 4
msi.num_nodes = 2

init_yielder = msi.init('0','0')
run_key, init_tuple = init_yielder.next()

distribute_yielder = msi.distribute_data(run_key, init_tuple)
run_key_0, distribute_state_out_0 = distribute_yielder.next()
run_key_1, distribute_state_out_1 = distribute_yielder.next()

run_key_0, infer_out_0 = msi.infer(run_key_0, distribute_state_out_0).next()
run_key_1, infer_out_1 = msi.infer(run_key_1, distribute_state_out_1).next()
# are all x_indices represented in output of infer?
x_indices_0 = infer_out_0.x_indices
x_indices_1 = infer_out_1.x_indices
len(numpy.union1d(x_indices_0, x_indices_1))
# are all x_indices represented in output of infer via list_of_x_indices?
list_of_x_indices_0 = flatten(infer_out_0.list_of_x_indices)
list_of_x_indices_1 = flatten(infer_out_1.list_of_x_indices)
len(numpy.unique(numpy.append(list_of_x_indices_0, list_of_x_indices_1)))

run_key_0, consolidate_data_out_0 = msi.consolidate_data(
    run_key_0, [infer_out_0, infer_out_1]).next()
