#!python
import datetime as dt,os, os.path as op,sys
import gdata,gdata.docs.client as gdc
##http://gdata-python-client.googlecode.com/hg/pydocs/gdata.docs.html
import docs_helper as dh
reload(dh)

auth_file_str = "/home/dlovell/google_docs_auth"
auth_dict = {} ##for keyword args email,password
with open(auth_file_str) as fh:
    exec fh in auth_dict
    temp = auth_dict.pop("__builtins__") ##pop prints otherwise
    client = dh.Docs_helper(**auth_dict)

mh_folder = client.get_collection("MH")

##try looking at this for how to upload?
##https://github.com/greggoryhz/Google-Docs-Sync/

##push files up to a given collection
base_dir = "/usr/local/Cloudless/examples/Plots/"
file_names = filter(lambda x:x[-3:]=="png",os.listdir(base_dir))
for file_name in file_names[:2]:
    full_path = os.path.join(base_dir,file_name)
    if not op.isfile(full_path):
        continue
    client.push_file(full_path,"image/png",collection=mh_folder,replace=True)

client.get_file(title_str='A=DiscG,B=0.1,init=P,CL=2,PPC=16_ari_by_time.png'
                ,dest_path="/usr/local/Cloudless/examples/Plots/temp.png")
