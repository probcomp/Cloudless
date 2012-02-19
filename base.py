from IPython.parallel import *

global remote_client
remote_client = None

def initialize_client():
    global remote_client
    if remote_client is None:
        # TODO: investigate other packers
        remote_client = Client(packer="pickle")
    
def get_view():
    initialize_client()
    return remote_client.load_balanced_view()

def get_direct_view():
    initialize_client()
    return remote_client[:]

def remote_exec(pystr):
    get_direct_view().execute(pystr)

def remote_procedure(name, proc):
    get_direct_view()[name] = proc
