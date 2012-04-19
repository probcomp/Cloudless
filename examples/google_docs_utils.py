#!python
import datetime as dt,os, os.path as op
import gdata
##http://gdata-python-client.googlecode.com/hg/pydocs/gdata.docs.html

auth_file_str = "/home/dlovell/google_docs_auth"
dateStr = dt.datetime.strftime(dt.datetime.now(),"%Y%m%d")
rootDir = "/usr/local/" if os.uname()[0].lower()=="linux" or os.uname()[0].lower()=="freebsd" else "/cygdrive/c/"
client = gdata.docs.client.DocsClient()
auth_dict = {} ##for keyword args email,password
with open(auth_file_str) as fh:
    exec fh in auth_dict
    auth_dict.pop("__builtins__")
    client.ClientLogin(source="writely",**auth_dict)

##find a collection and create a new one in it
##http://stackoverflow.com/questions/10054604/google-docs-api-with-python
q = gdata.docs.client.DocsQuery(
        title='MetropolisHastings',
            title_exact='true',
            show_collections='true'
        )
mh_folder = client.GetResources(q=q).entry[0]
##create the new collection
newCol = gdata.docs.data.Resource(type='folder', title=dateStr)
newCol = client.CreateResource(newCol,collection=mh_folder)


##try looking at this for how to upload?
##https://github.com/greggoryhz/Google-Docs-Sync/

##push files up to a given collection
baseDir = "/usr/local/Cloudless/examples/Plots/"
fileNames = filter(lambda x:x[-3:]=="png",os.listdir(baseDir))
for fileName in fileNames[:2]:
    fullPath = os.path.join(baseDir,fileName)
    if not op.isfile(fullPath):
        continue
    mediaResource = gdata.data.MediaSource(content_type="image/png",file_path=fullPath)
    myResource = gdata.docs.data.Resource(type="file",title=fileName,convert=False)
    client.create_resource(myResource,media=mediaResource,collection=newCol)


##delete a file if it exists
try:
    q = gdata.docs.client.DocsQuery(
            title='hello_world.txt',
                title_exact='true',
                show_collections='false'
            )
    hello_world = client.GetResources(q=q).entry[0]
    client.delete_resource(hello_world,force=True)
except Exception,e:
    print e    


##copy up a text file
fullPath = os.path.join(baseDir,"hello_world.txt")
if op.isfile(fullPath):
    fileName = op.split(fullPath)[-1]
    mediaResource = gdata.data.MediaSource(content_type="text/plain",file_path=fullPath,file_name=fileName)
    myResource = gdata.docs.data.Resource(title=fileName)
    client.create_resource(myResource,media=mediaResource,collection=mh_folder)


# import gdata.docs as gd
# resources = client.get_all_resources()
# resources[0].title.to_string()  ## can get a resources
# client.download_resource(resources[0],"/usr/local/Cloudless/examples/Plots")

q = gdata.docs.client.DocsQuery(
    ##title='mle_alpha.png',
    title='A=DiscG,B=0.1,init=P,CL=2,PPC=16_ari_by_time.png',
    title_exact='true',
    show_collections='true'
    )
mle_alpha = client.GetResources(q=q).entry[0]
client.download_resource(mle_alpha,"/usr/local/Cloudless/examples/Plots/temp.png")