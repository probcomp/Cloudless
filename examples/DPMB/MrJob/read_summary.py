import argparse
#
import Cloudless.examples.DPMB.remote_functions as rf


parser = argparse.ArgumentParser()
parser.add_argument('summary_filename', type=str)
args = parser.parse_args()
summary_filename = args.summary_filename

summary = rf.unpickle(summary_filename)
print map(len, summary['lol_of_x_indices'])
print map(lambda x: sum(map(len, x)), summary['lol_of_x_indices'])
