import cPickle
import copy
import time
from threading import Thread
##
import numpy as np
import pylab
##
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB.DPMB as dm
reload(dm)

# takes in a dataset spec
# returns a dictionary describing the problem, containing:
# out["dataset_spec"] --- input dataset spec
# out["zs"] --- generating zs for training data
# out["xs"] --- list of raw vectors for the training data
# out["test_xs"] --- list of raw vectors for the test data
# out["test_lls_under_gen"] --- list of the log predictive probabilities of the test vectors under the generating model

def arr_copy(arr_or_other):
    if arr_or_other is None or type(arr_or_other) is int:
        return arr_or_other
    elif type(arr_or_other) == tuple: # only inf_seed
        seed_copy = (arr_or_other[0],arr_or_other[1].copy(),arr_or_other[2],arr_or_other[3],arr_or_other[4])
        return seed_copy
    else:
        return np.array(arr_or_other).copy()

def copy_jobspec(jobspec): #my own deepcopy
    copyspec = copy.deepcopy(jobspec)
    copyspec["dataset_spec"]["gen_betas"] = arr_copy(jobspec["dataset_spec"]["gen_betas"])
    copyspec["infer_seed"] = arr_copy(jobspec["infer_seed"])
    copyspec["infer_init_betas"] = arr_copy(jobspec["infer_init_betas"])
    copyspec["infer_init_z"] = arr_copy(jobspec["infer_init_z"])
    #
    return copyspec

def modify_jobspec_to_results(jobspec,job_value):
    last_summary = job_value[-1]
    jobspec["infer_seed"] = arr_copy(last_summary["inf_seed"])
    jobspec["infer_init_alpha"] = last_summary["alpha"]
    jobspec["infer_init_betas"] = arr_copy(last_summary["betas"])
    jobspec["infer_init_z"] = arr_copy(last_summary["zs"])
    jobspec["decanon_indices"] = arr_copy(last_summary["decanon_indices"])
                                        
class Chunked_Job(Thread):
    def __init__(self,parent_jobspec,asyncmemo,chunk_iter=100,sleep_duration=5,lock=None):
        Thread.__init__(self)
        self.parent_jobspec = parent_jobspec
        self.asyncmemo = asyncmemo
        self.chunk_iter = chunk_iter
        #
        self.child_jobspec_list = []
        self.child_job_list = []
        self.consolidated_data = []
        self.done = False
        self.failed = False
        self.pause = False
        self.sleep_duration = sleep_duration
        self.lock = lock
        #
        self.problem = gen_problem(self.parent_jobspec["dataset_spec"])
        #
        # submit a job immediately

    def next_chunk_size(self):
        self.consolidate_jobs()
        remaining_iters = self.parent_jobspec["num_iters"] - (len(self.consolidated_data)-1) # subtract 1 for initial state
        return min(self.chunk_iter,remaining_iters)
    
    def get_current_jobspec(self):
        if len(self.child_jobspec_list) == len(self.child_job_list):
            return None
        else:
            return self.child_jobspec_list[-1]

    def get_next_jobspec(self):
        if self.check_done():
            return None
        #
        child_jobspec = copy_jobspec(self.parent_jobspec)
        if len(self.child_job_list) > 0:
            modify_jobspec_to_results(child_jobspec,self.child_job_list[-1])
        child_jobspec["num_iters"] = self.next_chunk_size()
        #
        return child_jobspec

    def acquire_lock(self):
        if self.lock is not None:
            self.lock.acquire()

    def release_lock(self):
        if self.lock is not None:
            self.lock.release()
        
    def evolve_chain(self):
        self.check_done()
        if self.done:
            return # nothing to submit
        current_jobspec = self.get_current_jobspec()
        if current_jobspec is not None:
            return # still working
        next_jobspec = self.get_next_jobspec()
        print "Submitting a jobspec"
        self.child_jobspec_list.append(next_jobspec)
        self.asyncmemo(next_jobspec)

    def run(self):
        while not self.done and not self.pause:
            self.evolve_chain()
            time.sleep(self.sleep_duration)
            
    def pull_down_jobs(self):
        list_len_delta = len(self.child_jobspec_list) > len(self.child_job_list)
        assert list_len_delta <= 1,"Chunked_Job.pull_down_jobs: child_jobspec_list got too far ahead of child_job_list"
        if list_len_delta == 1:
            self.acquire_lock()
            self.asyncmemo.advance()
            self.release_lock()
            current_jobspec = self.get_current_jobspec()
            job_value = self.asyncmemo(current_jobspec)
            if job_value is None: # still working
                return False
            else:
                self.child_job_list.append(job_value)
    
    def check_done(self):
        if self.done:
            return self.done
        self.pull_down_jobs()
        self.consolidate_jobs()
        self.check_failure(self.get_current_jobspec())
        if self.next_chunk_size() == 0:
            self.done = True
        return self.done
    
    def consolidate_jobs(self):
        if len(self.child_job_list) == 0:
            self.consolidated_data = []
            return
        ret_list = self.child_job_list[0][:]
        # FIXME : this isn't correct.  Need to detrmine num_iters contained in all jobs
        for child_job in self.child_job_list[1:]:
            ret_list.extend(child_job[1:])
        self.consolidated_data = ret_list
        
    def check_failure(self,jobspec):
        if jobspec is None:
            return False
        # this is dependent on arguments to infer
        arg_str = str((jobspec,))
        self.acquire_lock()
        job = self.asyncmemo.jobs[arg_str]
        if not job["remote"]:
            async_res = job['async_res']
            if async_res.ready() and not async_res.successful():
                self.done = True
                self.failed = True
        self.release_lock()
        return self.failed
        
def gen_problem(dataset_spec):
    # generate a state, initialized according to the generation parameters from dataset spec,
    # containing all the training data only
    state = ds.DPMB_State(dataset_spec["gen_seed"],
                          dataset_spec["num_cols"],
                          dataset_spec["num_rows"],
                          init_alpha=dataset_spec["gen_alpha"],
                          init_betas=dataset_spec["gen_betas"],
                          init_z=dataset_spec["gen_z"],
                          init_x = None)
    problem = {}
    problem["dataset_spec"] = dataset_spec
    problem["zs"] = state.getZIndices()
    problem["xs"] = state.getXValues()
    test_xs, test_lls = state.generate_and_score_test_set(dataset_spec["N_test"])
    problem["test_xs"] = test_xs
    problem["test_lls_under_gen"] = test_lls
    return problem

global counts
counts = {}
def plot_helper(name, state):
    global counts
    if state not in counts:
        counts[state] = 0
    count = counts[state]
    state.plot(show=False,save_str = name + "-" + "%3d" % count + ".png")
    counts[state] += 1

def infer(run_spec):
    ##problem = run_spec["problem"]
    dataset_spec = run_spec["dataset_spec"] ## problem["dataset_spec"]
    verbose_state = "verbose_state" in run_spec and run_spec["verbose_state"]
    decanon_indices = run_spec.get("decanon_indices",None)
    #
    if verbose_state:
        print "doing run: "
        for (k, v) in run_spec.items():
            if k.find("seed") != -1:
                print "   " + "hash(" + str(k) + ")" + " ---- " + str(hash(str(v)))
            else:
                print "   " + str(k) + " ---- " + str(v)
    #
    print "initializing"
    initial_state = ds.DPMB_State(dataset_spec["gen_seed"],
                                  dataset_spec["num_cols"],
                                  dataset_spec["num_rows"],
                                  init_alpha=dataset_spec["gen_alpha"],
                                  init_betas=dataset_spec["gen_betas"],
                                  init_z=dataset_spec["gen_z"],
                                  init_x = None)
    # all you need are the xs
    gen_xs = initial_state.getXValues()
    gen_zs = initial_state.getZIndices()
    inference_state = ds.DPMB_State(dataset_spec["gen_seed"],
                              dataset_spec["num_cols"],
                              dataset_spec["num_rows"],
                              # these could be the state at the end of an inference run
                              init_alpha=run_spec["infer_init_alpha"],
                              init_betas=run_spec["infer_init_betas"],
                              init_z=run_spec["infer_init_z"],
                              # 
                              init_x = gen_xs,decanon_indices=decanon_indices)
    #
    print "...initialized"
    #
    transitioner = dm.DPMB(inf_seed = run_spec["infer_seed"],
                           state = inference_state,
                           infer_alpha = run_spec["infer_do_alpha_inference"],
                           infer_beta = run_spec["infer_do_betas_inference"])
    #
    summaries = []
    summaries.append(
        transitioner.extract_state_summary(
            true_zs=gen_zs,verbose_state=verbose_state))
    #
    print "saved initialization"
    #
    time_seatbelt = None
    ari_seatbelt = None
    if "time_seatbelt" in run_spec:
        time_seatbelt = run_spec["time_seatbelt"]
    if "ari_seatbelt" in run_spec:
        ari_seatbelt = run_spec["ari_seatbelt"]
    #
    last_valid_zs = None
    decanon_indices = None
    for i in range(run_spec["num_iters"]):
        transition_return = transitioner.transition(time_seatbelt=time_seatbelt,ari_seatbelt=ari_seatbelt,true_zs=gen_zs) # true_zs necessary for seatbelt 
        print "finished doing iteration" + str(i)
        summaries.append(
            transitioner.extract_state_summary(
            true_zs=gen_zs,verbose_state=verbose_state))
        print "finished saving iteration" + str(i)
        if transition_return is not None:
            summaries[-1]["break"] = transition_return
            break
        last_valid_zs = transitioner.state.getZIndices()
        decanon_indices = transitioner.state.get_decanonicalizing_indices()
    summaries[-1]["last_valid_zs"] = last_valid_zs
    summaries[-1]["decanon_indices"] = decanon_indices
    return summaries

def extract_problems_from_memo(asyncmemo):
    from numpy import array
    ALL_RUN_SPECS = [eval(key)[0] for key in asyncmemo.memo.keys()]
    ALL_PROBLEM_STRS = dict(zip([str(run_spec["problem"]) for run_spec in ALL_RUN_SPECS],np.repeat(None,len(ALL_RUN_SPECS)))).keys()
    ALL_PROBLEMS = [eval(problem_str) for problem_str in ALL_PROBLEM_STRS]
    return ALL_PROBLEMS

def pickle_asyncmemoize(asyncmemo,file_str):
    with open(file_str,"wb") as fh:
        cPickle.dump(asyncmemo.memo,fh)

def unpickle_asyncmemoize(asyncmemo,file_str):
    from numpy import array
    with open(file_str,"rb") as fh:
        pickled_memo = cPickle.load(fh)
    #
    ALL_RUN_SPECS = [eval(run_spec)[0] for run_spec in pickled_memo.keys()]
    new_memo = dict(zip([str((run_spec,)) for run_spec in ALL_RUN_SPECS],pickled_memo.values()))
    new_args = dict(zip([str((run_spec,)) for run_spec in ALL_RUN_SPECS],[(run_spec,) for run_spec in ALL_RUN_SPECS]))
    # FIXME: action through setting asyncmemo elements, perhaps should return these for caller to set?
    asyncmemo.memo = new_memo
    asyncmemo.args = new_args
    return ALL_RUN_SPECS

def pickle_if_done(memoized_infer,file_str="pickled_jobs.pkl"):
    status = memoized_infer.report_status()
    if status["waiting"] != 0:
        print "Not done, not pickling"
        return False
    else:
        temp = [(k,v) for k,v in memoized_infer.iter()] ##FIXME : how better to ensure memo pullis it down?
        with open(file_str,"wb") as fh:
            cPickle.dump(memoized_infer.memo,fh)
        print "Done all jobs, memo pickled"
        return True

def plot_measurement(memoized_infer, which_measurement, target_problem, by_time = True
                     ,run_spec_filter=None,save_str=None,title_str=None,ylabel_str=None,legend_args=None
                     ,do_legend=True):
    matching_runs = []
    matching_summaries = []
    #
    for (args, summaries) in memoized_infer.iter():
        run_spec = args[0]
        if run_spec_filter is not None and not run_spec_filter(run_spec):
            continue
        if str(run_spec["problem"]) == str(target_problem): ##FIXME: This is a hack, it should work without making str
            matching_runs.append(run_spec)
            matching_summaries.append(summaries)
    #
    if len(matching_summaries) == 0:
        res = memoized_infer.report_status()
        if len(res["failures"]) > 0:
            print "**********************************************************"
            print "FIRST EXCEPTION: "
            print "**********************************************************"
            print res["failures"][0][1]
        raise Exception("No data to plot with these characteristics!")
    #
    matching_measurements = []
    matching_linespecs = []
    matching_legendstrs = []
    for (run, summary) in zip(matching_runs, matching_summaries):
        try:
            matching_measurements.append(extract_measurement(which_measurement, summary))
        except Exception, e:
            print e
        #
        linespec = {}
        legendstr = ""
        # for now, red if both hyper inference, black otherwise FIXME expand out all 4 bit options
        if run["infer_do_alpha_inference"] and run["infer_do_betas_inference"]:
            linespec["color"] = "red"
            legendstr += "inf_a=T,inf_b=T"
        elif run["infer_do_alpha_inference"] and not run["infer_do_betas_inference"]:
            linespec["color"] = "green"
            legendstr += "inf_a=T,inf_b=F"
        elif not run["infer_do_alpha_inference"] and run["infer_do_betas_inference"]:
            linespec["color"] = "magenta"
            legendstr += "inf_a=F,inf_b=T"
        else:
            linespec["color"] = "blue"
            legendstr += "inf_a=F,inf_b=F"
        # linestyle for initialization
        init_z = run["infer_init_z"]
        if init_z == 1:
            linespec["linestyle"] = "-"
            legendstr += ";init=P"
        elif init_z == "N":
            linespec["linestyle"] = "--"
            legendstr += ";init=N"
        elif init_z == None:
            linespec["linestyle"] = "-."
            legendstr += ";init=1"
        else:
            raise Exception("invalid init_z" + str(init_z))
        #
        matching_linespecs.append(linespec)
        matching_legendstrs.append(legendstr)
    # FIXME: enable plots. still need to debug timing ***urgent***

    # unique_legendstrs = np.unique(matching_legendstrs)
    # unique_linespecs = []
    # for unique_legendstr in unique_legendstrs:
    #     unique_linespecs.append(matching_linespecs.index(

    pylab.figure()
    if do_legend:
        pylab.subplot(211)
    line_list = []
    if by_time:
        for measurement, summary, linespec in zip(matching_measurements, matching_summaries, matching_linespecs):
            xs = extract_time_elapsed_vs_iterations(summary)
            fh = pylab.plot(xs, measurement, color = linespec["color"], linestyle = linespec["linestyle"])
            pylab.xlabel("time (seconds)")
            line_list.append(fh[0])
    else:
        for measurement,linespec in zip(matching_measurements,matching_linespecs):
            fh = pylab.plot(measurement,color=linespec["color"], linestyle=linespec["linestyle"])
            pylab.xlabel("iter")
            line_list.append(fh[0])
    #
    if title_str is not None:
        if type(title_str) is str:
            pylab.title(title_str)
        else:
            pylab.title(title_str[0])
    if ylabel_str is not None:
        pylab.ylabel(ylabel_str)
    if do_legend:
        pylab.subplot(212)
        if legend_args is None:
            legend_args = {"ncol":2,"prop":{"size":"medium"}}
        pylab.legend(line_list,matching_legendstrs,**legend_args)
    ##pylab.subplots_adjust(hspace=.4)
    if save_str is not None:
        pylab.savefig(save_str)

def try_plots(memoized_infer,which_measurements=None,run_spec_filter=None,do_legend=True):
    temp = [(k,v) for k,v in memoized_infer.iter()] ##FIXME : how better to ensure memo pullis it down?
    which_measurements = ["ari"] if which_measurements is None else which_measurements
    #
    for problem_idx,target_problem in enumerate(extract_problems_from_memo(memoized_infer)):
        cluster_str = "clusters" + str(target_problem["dataset_spec"]["gen_z"][1]) ##
        col_str = "cols" + str(target_problem["dataset_spec"]["num_cols"])
        row_str = "rows" + str(target_problem["dataset_spec"]["num_rows"])
        config_str = "_".join([col_str,row_str,cluster_str])    
        #
        for which_measurement in which_measurements:
            try:
                plot_measurement(memoized_infer, which_measurement, target_problem, run_spec_filter=run_spec_filter
                                    ,save_str="_".join([which_measurement,config_str,"time.png"]),title_str=[config_str,which_measurement],ylabel_str=which_measurement
                                    ,legend_args={"ncol":2,"markerscale":2},do_legend=do_legend)
                plot_measurement(memoized_infer, which_measurement, target_problem, run_spec_filter=run_spec_filter, by_time=False
                                    ,save_str="_".join([which_measurement,config_str,"iter.png"]),title_str=[config_str,which_measurement],ylabel_str=which_measurement
                                    ,legend_args={"ncol":2,"markerscale":2},do_legend=do_legend)
            except Exception, e:
                print e
    #plot_measurement(memoized_infer, "predictive", target_problem)

def extract_measurement(which_measurement, one_runs_data):
    # measurement can be:
    # "predictive" FIXME
    if np.in1d(which_measurement,["num_clusters","ari","alpha","score"])[0]:
        return [summary[which_measurement] for summary in one_runs_data]
    elif which_measurement == "mean_beta":
        return [np.mean(summary["betas"]) for summary in one_runs_data]
    else:
        raise Exception("not implemented yet: " + str(which_measurement))

def extract_time_elapsed_vs_iterations(summary_seq):
    out = []
    cumsum = 0
    for summary in summary_seq:
        timing = summary["timing"]
        iter_sum = sum([timing[key] for key in ["alpha","betas","zs"]])
        cumsum += iter_sum
        out.append(cumsum)    
    return out
