from IPython.parallel import *

remote_client = None

def get_view():
    if remote_client is None:
        remote_client = Client(packer="pickle")
    return remote_client[:]
