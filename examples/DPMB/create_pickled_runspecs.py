import argparse
import cPickle

parser = argparse.ArgumentParser(description='Generate ALL_RUN_SPECS and store in a pkl file')
parser.add_argument('in_file_str',type=str)
parser.add_argument('out_file_str',type=str)
args = parser.parse_args()
in_file_str = args.in_file_str
out_file_str = args.out_file_str

exec_in = dict()
with open(in_file_str) as fh:
    exec fh in exec_in

try:
    with open(out_file_str,"wb") as fh:
        cPickle.dump(exec_in["ALL_RUN_SPECS"],fh)
except Exception, e:
    print "Couldn't write ALL_RUN_SPECS to " + out_file_str
    print str(e)
