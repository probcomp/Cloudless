#!python
import os
#
import h5py


class h5_context(object):
    def __init__(self, filename, mode=None, dir=''):
        self.filename = filename
        self.mode = mode
        self.dir = dir
    def __enter__(self):
        full_filename = os.path.join(self.dir, self.filename)
        self.f = h5py.File(full_filename, self.mode)
        return self.f
    def __exit__(self, *args):
        self.f.close()

failed_to_h5ify_groupname = 'failed_to_h5ify'
def h5ify(key, value, my_h5):
    try:
        dset = my_h5.create_dataset(key, data=value)
    except Exception, e:
        if failed_to_h5ify_groupname not in my_h5:
            failed_group = my_h5.create_group(failed_to_h5ify_groupname)
        failed_group = my_h5[failed_to_h5ify_groupname]
        failed_group[key] = str(value)

def unh5ify(key, my_h5):
    value = None
    if key in my_h5:
        value = my_h5[key].value
    else:
        from numpy import uint32
        failed_group = my_h5[failed_to_h5ify_groupname]
        value = eval(failed_group[key].value)
    return value

def h5ify_dict(in_dict, my_h5):
    for key, value in in_dict.iteritems():
        h5ify(key, value, my_h5)

def unh5ify_dict(my_h5):
    store_dict = dict()
    for key, value in my_h5.iteritems():
        if key == failed_to_h5ify_groupname: continue
        store_dict[key] = unh5ify(key, my_h5)
    if failed_to_h5ify_groupname in my_h5:
        failed_group = my_h5[failed_to_h5ify_groupname]
        for key in failed_group.keys():
            store_dict[key] = unh5ify(key, my_h5)
    return store_dict
