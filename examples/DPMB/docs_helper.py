#!python
import gdata
import gdata.docs.client as gdc
import os
import sys
##general documentation
##http://gdata-python-client.googlecode.com/hg/pydocs/gdata.docs.html
##how to find a collection
##http://stackoverflow.com/questions/10054604/google-docs-api-with-python

class Docs_helper():
    def __init__(self,email,password):
        self.client = gdc.DocsClient()
        self.client.ClientLogin(email,password,"writely")

    def get_collection(self,collection_str):
        try:
            q = gdata.docs.client.DocsQuery(
                    title=collection_str,title_exact='true',
                    show_collections='true')
            collection = self.client.GetResources(q=q).entry[0]
        except Exception, e:
            collection = None
            
        return collection

    def create_collection(self,title,in_collection=None):
        new_col = gdata.docs.data.Resource(type='folder', title=title)
        new_col = self.client.CreateResource(new_col,collection=in_collection)
        return new_col
    
    def push_file(self,file_path,content_type,collection=None,docs_name=None,replace=False):
        if not os.path.isfile(file_path):
            print file_path + " doesn't exist!"
            return None

        file_name = os.path.split(file_path)[-1]
        docs_name = docs_name if docs_name is not None else file_name
        if replace:
            try:
                q = gdata.docs.client.DocsQuery(
                    title=docs_name,title_exact='true',
                    show_collections='false')
                target_resource = self.client.GetResources(q=q).entry[0]
                self.client.delete_resource(target_resource,force=True)
            except Exception,e:
                pass ##most likely, there was no file

        mediaResource = gdata.data.MediaSource(
            content_type=content_type
            , file_path=file_path
            , file_name=file_name)
        myResource = gdata.docs.data.Resource(title=docs_name)
        self.client.create_resource(myResource,media=mediaResource
                               ,collection=collection)

    def get_file(self,title_str,dest_path):
        q = gdata.docs.client.DocsQuery(
            title=title_str,
            title_exact='true',
            show_collections='true'
            )
        my_resource = self.client.GetResources(q=q).entry[0]
        self.client.download_resource(my_resource,dest_path)

def main():
    import argparse
    #
    parser = argparse.ArgumentParser(
        description='A script that can programatically interact with google docs')
    parser.add_argument('file_strs',nargs="+",type=str)
    parser.add_argument('--auth_file'
                        , default=os.path.expanduser("~/google_docs_auth")
                        , type=str)
    parser.add_argument('--email',default=None,type=str)
    parser.add_argument('--password',default=None,type=str)
    parser.add_argument('--folder',default="MH",type=str)
    parser.add_argument('--replace',action='store_true')
    parser.add_argument('--mime_type',default="text/plain",type=str)
    args = parser.parse_args()
    if not os.path.isfile(args.file_strs[0]):
        print "Files doesn't exists: file_str: " + str(args.file_strs) 
        exit()
    #
    auth_dict = {}
    if os.path.isfile(args.auth_file):
        exec open(args.auth_file) in auth_dict
    email = args.email if args.email is not None else auth_dict["email"]
    password = args.password if args.password is not None else auth_dict["password"]
    #
    client = Docs_helper(email=email,password=password)
    collection = client.get_collection(args.folder)
    for file_str in args.file_strs:
        client.push_file(file_str
                         , args.mime_type
                         , collection=collection
                         , replace=args.replace)

if __name__ == "__main__":
    main()
