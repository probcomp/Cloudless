from numpy import array

for run_spec_str,summaries in memoized_infer.memo.iteritems():
    run_spec = eval(run_spec_str)
    num_cols = run_spec["dataset_spec"]["num_cols"]
    num_rows = run_spec["dataset_spec"]["num_rows"]

    cluster_counts = []
    z_diff_times = []
    for summary in summaries[1:]: # first summary is initialization state
        micro_z_timing = summary["timing"]["micro_z_timing"]
        cluster_counts.extend(micro_z_timing["cluster_counts"][1:])
        z_diff_times.extend(np.diff(micro_z_timing["z_cumulative_time"]))

    cluster_counts = np.array(cluster_counts)
    z_diff_times = np.array(z_diff_times)

    box_input = {}
    for cluster_count,diff_time in zip(cluster_counts,z_diff_times):
        box_input.setdefault(cluster_count,[]).append(diff_time)

    median_times = []
    for cluster_count in np.sort(box_input.keys()):
        median_times.append(np.median(box_input[cluster_count]))

    slope,intercept,r_value,p_value,stderr = linregress(
        np.sort(box_input.keys())
        ,median_times)
    title_str = "slope = " + ("%.3g" % slope) \
        + "; intercept = " + ("%.3g" % intercept) \
        + "; R^2 = " + ("%.5g" % r_value**2)

    cutoff = cluster_counts.max()/3
    box_every_n = max(1,len(box_input)/10)

    pylab.figure()
    pylab.plot(cluster_counts,z_diff_times,'x')
    pylab.title(title_str)
    pylab.xlabel("num_clusters")
    pylab.ylabel("single-z scan time (seconds)")
    pylab.savefig("scatter_scan_times_num_cols_"+str(num_cols)+"_num_rows_"+str(num_rows))
    #
    pylab.figure()
    pylab.boxplot(box_input.values()[::box_every_n]
                  ,positions=box_input.keys()[::box_every_n]
                  ,sym="")
    pylab.title(title_str)
    pylab.xlabel("num_clusters")
    pylab.ylabel("single-z scan time (seconds)")
    pylab.savefig("boxplot_scan_times_num_cols_"+str(num_cols)+"_num_rows_"+str(num_rows))
    pylab.close()
    #
    pylab.figure()
    pylab.hexbin(cluster_counts[cluster_counts<cutoff],z_diff_times[cluster_counts<cutoff])
    pylab.title(title_str)
    pylab.xlabel("num_clusters")
    pylab.ylabel("single-z scan time (seconds)")
    pylab.colorbar()
    pylab.savefig("hexbin_scan_times_num_cols_"+str(num_cols)+"_num_rows_"+str(num_rows)+"_lt_"+str(cutoff))
    #
    pylab.figure()
    pylab.hexbin(cluster_counts[cluster_counts>cutoff],z_diff_times[cluster_counts>cutoff])
    pylab.title(title_str)
    pylab.xlabel("num_clusters")
    pylab.ylabel("single-z scan time (seconds)")
    pylab.colorbar()
    pylab.savefig("hexbin_scan_times_num_cols_"+str(num_cols)+"_num_rows_"+str(num_rows)+"_gt_"+str(cutoff))

