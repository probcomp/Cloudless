#!python
import Cloudless.examples.DPMB_basic as b

if "job_list" not in locals():
    job_list = []
    done_list = []
    for CLUSTERS in [2,16,64]:
        for POINTS_PER_CLUSTER in [2,16,64]:
            packed_params = b.create_dict()
            packed_params["CLUSTERS"] = CLUSTERS
            packed_params["POINTS_PER_CLUSTER"] = POINTS_PER_CLUSTER
            job_list.append((b.queue_jobs(packed_params=packed_params,**packed_params),packed_params))

job_list,done_list = b.filter_plottable(job_list,done_list)
##to rerun, must 'del job_list'
