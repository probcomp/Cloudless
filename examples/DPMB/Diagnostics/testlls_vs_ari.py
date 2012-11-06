import os
from collections import Counter
#
import pylab
#
import Cloudless.examples.DPMB.remote_functions as rf
reload(rf)
import Cloudless.examples.DPMB.MrJob.consolidate_summaries as cs
reload(cs)
import Cloudless.examples.DPMB.plot_utils as pu
reload(pu)
import Cloudless.examples.DPMB.settings as S
reload(S)


reduced_summaries_name = S.files.reduced_summaries_name
init_filename = 'summary_numnodes4_seed0_iternum-1.pkl.gz'
top_dir = '/usr/local/Cloudless/examples/DPMB/Data/BigLearnData/programmatic_rerun'
os.chdir(top_dir)
all_dirs = os.listdir('.')

is_programmatic = lambda dir: dir.startswith('programmatic_mrjob_')
has_reduced_summary = lambda dir: os.path.isfile(os.path.join(dir, reduced_summaries_name))
programmatic_dirs = filter(is_programmatic, all_dirs)
programmatic_dirs = filter(has_reduced_summary, programmatic_dirs)

color_lookup = {
    'summary_numnodes4_seed0_iternum':'red',
    'summary_numnodes1_seed0_child0_iternum':'green',
    }
marker_lookup = {
    'summary_numnodes4_seed0_iternum':'x',
    'summary_numnodes1_seed0_child0_iternum':'o',
    }

parameters_counter = Counter()
pylab.figure()
for data_dir in programmatic_dirs:
    parameters = cs.read_run_parameters(data_dir)
    parameters.pop('gen_seed')
    reduced_summaries = rf.unpickle(reduced_summaries_name, data_dir)
    for summaries_name, summaries in reduced_summaries.iteritems():
        aris = summaries['ari']
        # aris[0] = 0 # if aris[0] is None else aris[0]
        test_lls = summaries['test_lls']
        if (pylab.array(test_lls) > -120).any():
            break
        parameters_counter.update([str(parameters)])
        color = color_lookup[summaries_name]
        marker = marker_lookup[summaries_name]
        pylab.plot(aris, test_lls, label=summaries_name,
                   color=color, alpha=0.5)
        pylab.plot(aris[-1], test_lls[-1], marker=marker,
                   color=color, alpha=0.5)

pylab.xlabel('ari')
pylab.ylabel('test_lls')
ax = pylab.gca()
handles, labels = ax.get_legend_handles_labels()
handles = handles[:2]
labels = labels[:2]
bbox_to_anchor=(0.5, -.1)
loc='upper center'
ncol = len(labels)
lgd = ax.legend(handles, labels, loc=loc, ncol=ncol,
                bbox_to_anchor=bbox_to_anchor)
pu.savefig_legend_outside('test_lls_vs_ari')
