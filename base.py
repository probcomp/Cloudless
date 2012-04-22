##from IPython.parallel import *

global remote_client
remote_client = None

global remote
remote = False

global memoizers
memoizers = {}

global scratch
scratch = {}

# TODO: Add a mode that creates a local collection of engines, but still uses
#       iPython.parallel

def remote_mode():
    global remote
    remote = True
    initialize_client()

def local_mode():
    global remote
    global remote_client
    remote = False
    remote_client = False

def clear_all():
    for m in memoizers.values():
        m.clear()

def save(name):
    if remote:
        raise Exception("saving and loading not yet implemented for remote workspaces")
    else:
        raise Exception("saving and loading not yet implemented for local")

    #import pickle
    # save the memoizers
    # save the scratch

def load(name):
    if remote:
        raise Exception("saving and loading not yet implemented for remote workspaces")
    else:
        raise Exception("Saving and loading not yet implemented for local")

def initialize_client():
    if remote:
        global remote_client
        if remote_client is None:
            # TODO: investigate other packers
            import os
            if os.uname()[1] == 'vagabond' and False:  ##this may fail for using starcluster shell -p CLUSTER
                remote_client = Client(packer="json")
            else: ##alternatively, could test for 'master' for packer="pickle"
                remote_client = Client(packer="pickle")
    
def get_view():
    if remote:
        initialize_client()
        return remote_client.load_balanced_view()
    else:
        return None

def get_direct_view():
    if remote:
        initialize_client()
        return remote_client[:]
    else:
        return None

# FIXME: this breaks symmetry between local and remote, and if you name
#        badly (e.g. do remote_procedure('somename', 'not_somename'))
#        then your code could fail on switching
def remote_exec(pystr):
    if remote:
        get_direct_view().execute(pystr)

def remote_procedure(name, proc):
    if remote:
        get_direct_view()[name] = proc
    
