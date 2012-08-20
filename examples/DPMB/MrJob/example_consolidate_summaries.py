#!python
import os
#
import Cloudless.examples.DPMB.MrJob.consolidate_summaries as cs
reload(cs)
import Cloudless.examples.DPMB.settings as settings
reload(settings)

data_dirs = [
    os.path.join(settings.data_dir,data_dir)
    for data_dir in ['one_node','two_node','four_node']
    ]
summaries_dict_dict = {}
for data_dir in data_dirs:
    summary_names = cs.get_summary_names(data_dir)
    summaries_dict = cs.get_summaries_dict(summary_names,data_dir)
    cs.print_info(summaries_dict)
    summaries_dict_dict[data_dir] = summaries_dict
