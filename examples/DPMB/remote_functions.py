import cPickle
import copy
import time
from threading import Thread
import datetime
from numpy import array
import sets
import os
import gzip
##
import numpy as np
import pylab
##
import Cloudless.examples.DPMB.DPMB_State as ds
reload(ds)
import Cloudless.examples.DPMB.DPMB as dm
reload(dm)
import Cloudless.examples.DPMB.PDPMB_State as pds
reload(pds)
import Cloudless.examples.DPMB.PDPMB as pdm
reload(pdm)
import Cloudless.examples.DPMB.helper_functions as hf
reload(hf)
import Cloudless
reload(Cloudless)
import Cloudless.memo
reload(Cloudless.memo)

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
    def __init__(self,parent_jobspec,asyncmemo=None,chunk_iter=100,sleep_duration=20,lock=None):
        Thread.__init__(self)
        self.parent_jobspec = parent_jobspec
        if asyncmemo is None:
            register_str = str(self.parent_jobspec)
            self.asyncmemo = Cloudless.memo.AsyncMemoize(
                register_str, ["run_spec"], infer, override=False)
        else:
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
        self.verbose = False
        #
        self.problem = gen_problem(self.parent_jobspec["dataset_spec"])
        #
        # submit a job immediately

    def get_current_jobspec(self):
        if len(self.child_jobspec_list) == len(self.child_job_list):
            return None
        else:
            return self.child_jobspec_list[-1]

    def get_next_jobspec(self):
        # check_done() should have been called prior
        child_jobspec = copy_jobspec(self.parent_jobspec)
        if len(self.child_job_list) > 0:
            modify_jobspec_to_results(child_jobspec,self.child_job_list[-1])
        child_jobspec["num_iters"] = self.next_chunk_size
        #
        return child_jobspec

    def acquire_lock(self):
        if self.lock is not None:
            print str(datetime.datetime.now()) + " :: acquiring lock :: " + str(self)
            self.lock.acquire()

    def release_lock(self):
        if self.lock is not None:
            print str(datetime.datetime.now()) + " :: releasing lock :: " + str(self)
            self.lock.release()
        
    def evolve_chain(self):
        if self.pause:
            return
        self.check_done()
        if self.done:
            return # nothing to submit
        current_jobspec = self.get_current_jobspec()
        if current_jobspec is not None:
            return # still working
        next_jobspec = self.get_next_jobspec()
        print "Submitting a jobspec from: " + str(self)
        self.child_jobspec_list.append(next_jobspec)
        self.acquire_lock()
        self.asyncmemo(next_jobspec)
        self.release_lock()

    def run(self):
        while not self.done:
            self.evolve_chain()
            time.sleep(self.sleep_duration)
            
    def pull_down_jobs(self):
        list_len_delta = len(self.child_jobspec_list) - len(self.child_job_list)
        assert list_len_delta <= 1,"Chunked_Job.pull_down_jobs: child_jobspec_list got too far ahead of child_job_list"
        if list_len_delta == 1:
            print "pull_down_jobs: list_len_delta==1 : " + str(self)
            self.acquire_lock()
            current_jobspec = self.get_current_jobspec()
            # current_jobstr = str((current_jobspec,))
            # job_value = None
            # if self.asyncmemo.jobs.get(current_jobstr,False):
            #     job = self.asyncmemo.jobs[current_jobstr]
            #     if job["remote"]:
            #         value = job['async_res']
            #         if value.ready() and value.successful():
            #             self.asyncmemo.advance()
            #             job_value = self.asyncmemo(current_jobspec)
            # self.asyncmemo.advance()
            job_value = self.asyncmemo(current_jobspec)
            self.release_lock()
            print "pull_down_jobs: job_value is None : "  + str(job_value is None) + str(self)
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
        #
        remaining_iters = (self.parent_jobspec["num_iters"]
                           - (len(self.consolidated_data)-1)) # subtract 1 for initial state
        self.next_chunk_size = min(self.chunk_iter,remaining_iters)
        #
        if self.next_chunk_size == 0:
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
        return False # FIXME : this short circuits, should be removed after debugging
        if jobspec is None:
            return False
        # this is dependent on arguments to infer
        arg_str = str((jobspec,))
        self.acquire_lock()
        job = self.asyncmemo.jobs[arg_str]
        if not job["remote"] and "async_res" in job:
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
    dataset_spec = run_spec["dataset_spec"]
    problem = gen_problem(dataset_spec)
    verbose_state = run_spec.get("verbose_state",False)
    decanon_indices = run_spec.get("decanon_indices",None)
    num_nodes = run_spec.get("num_nodes",1)
    hypers_every_N = run_spec.get("hypers_every_N",1)
    time_seatbelt = run_spec.get("time_seatbelt",None)
    ari_seatbelt = run_spec.get("ari_seatbelt",None)
    #
    if verbose_state:
        print "doing run: "
        for (k, v) in run_spec.items():
            if k.find("seed") != -1:
                print "   " + "hash(" + str(k) + ")" + " ---- " + str(hash(str(v)))
            else:
                print "   " + str(k) + " ---- " + str(v)
    #
    state_kwargs = {}
    model_kwargs = {}
    print "initializing"
    if num_nodes == 1:
        state_type = ds.DPMB_State
        model_type = dm.DPMB
        state_kwargs = {"decanon_indices":decanon_indices}
    else:
        state_type = pds.PDPMB_State
        model_type = pdm.PDPMB
        state_kwargs = {"num_nodes":num_nodes}
        model_kwargs = {"hypers_every_N":hypers_every_N}

    inference_state = state_type(dataset_spec["gen_seed"],
                                 dataset_spec["num_cols"],
                                 dataset_spec["num_rows"],
                                 init_alpha=run_spec["infer_init_alpha"],
                                 init_betas=run_spec["infer_init_betas"],
                                 init_z=run_spec["infer_init_z"],
                                 init_x = problem["xs"],
                                 **state_kwargs
                                 )

    print "...initialized"
    transitioner = model_type(
        inf_seed = run_spec["infer_seed"],
        state = inference_state,
        infer_alpha = run_spec["infer_do_alpha_inference"],
        infer_beta = run_spec["infer_do_betas_inference"],
        **model_kwargs
        )
    #
    summaries = []
    summaries.append(
        transitioner.extract_state_summary(
            true_zs=problem["zs"]
            ,verbose_state=verbose_state
            ,test_xs=problem["test_xs"]))
    #
    print "saved initialization"
    #
    last_valid_zs = None
    decanon_indices = None
    for i in range(run_spec["num_iters"]):
        transition_return = transitioner.transition(
            time_seatbelt=time_seatbelt
            ,ari_seatbelt=ari_seatbelt
            ,true_zs=problem["zs"]) # true_zs necessary for seatbelt 
        hf.printTS("finished doing iteration" + str(i))
        next_summary = transitioner.extract_state_summary(
            true_zs=problem["zs"]
            ,verbose_state=verbose_state
            ,test_xs=problem["test_xs"])
        if transition_return is not None:
            summaries[-1]["break"] = transition_return
            summaries[-1]["failed_info"] = next_summary
            break
        summaries.append(next_summary)
        hf.printTS("finished saving iteration" + str(i))
        if hasattr(transitioner.state,"getZIndices"):
            last_valid_zs = transitioner.state.getZIndices()
            decanon_indices = transitioner.state.get_decanonicalizing_indices()
    summaries[-1]["last_valid_zs"] = last_valid_zs
    summaries[-1]["decanon_indices"] = decanon_indices
    return summaries

def extract_dataset_specs_from_memo(asyncmemo):
    ALL_RUN_SPECS = [eval(key)[0] for key in asyncmemo.memo.keys()]
    ALL_DATASET_SPEC_STRS = [str(runspec["dataset_spec"]) for runspec in ALL_RUN_SPECS]
    import sets
    ALL_DATASET_SPECS = [eval(spec_str) for spec_str in list(sets.Set(ALL_DATASET_SPEC_STRS))]
    return ALL_DATASET_SPECS

def pickle_asyncmemoize(asyncmemo,file_str):
    if file_str[-3:] == ".gz":
        my_open = gzip.open
    else:
        my_open = open
    with my_open(file_str,"wb") as fh:
        cPickle.dump(asyncmemo.memo,fh)

def unpickle_asyncmemoize(asyncmemo,file_str):
    from numpy import array
    if file_str[-3:] == ".gz":
        my_open = gzip.open
    else:
        my_open = open
    with my_open(file_str,"rb") as fh:
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

def runspec_to_plotspec_bak(runspec):
    linespec = {}
    legendstr = ""
    # for now, red if both hyper inference, black otherwise FIXME expand out all 4 bit options
    if runspec["infer_do_alpha_inference"] and runspec["infer_do_betas_inference"]:
        linespec["color"] = "red"
        legendstr += "inf_a=T,inf_b=T"
    elif runspec["infer_do_alpha_inference"] and not runspec["infer_do_betas_inference"]:
        linespec["color"] = "green"
        legendstr += "inf_a=T,inf_b=F"
    elif not runspec["infer_do_alpha_inference"] and runspec["infer_do_betas_inference"]:
        linespec["color"] = "magenta"
        legendstr += "inf_a=F,inf_b=T"
    else:
        linespec["color"] = "blue"
        legendstr += "inf_a=F,inf_b=F"
    # linestyle for initialization
    init_z = runspec["infer_init_z"]
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
    return linespec,legendstr

def speclist_to_plotspecs(speclist,keylist):
    # take a list of keys that you want know unique combinations of
    colors = ["black","brown","green","red","blue","orange"]
    linestyles = ["-","--","-."]
    #
    legendstrs = []
    for spec in speclist:
        curr_legendstr_list = []
        for key in keylist:
            value = spec.get(key,"ERR")
            curr_legendstr_list.append(key+"="+str(value))
        legendstrs.append(";".join(curr_legendstr_list))
    #
    unique_legendstrs = np.sort(list(sets.Set(legendstrs)))
    legendstrs_lookup = dict(zip(unique_legendstrs,range(len(unique_legendstrs))))
    #
    linespecs = []
    for legendstr in legendstrs:
        legendstr_idx = legendstrs_lookup[legendstr]
        linespec = {}
        linespec["color"] = colors[legendstr_idx % len(colors)]
        linespec["linestyle"] = linestyles[(legendstr_idx / len(colors)) % len(linestyles)]
        linespecs.append(linespec)
    #
    return linespecs,legendstrs

def runspec_to_plotspec(runspec):
    linespec = {}
    legendstr = ""
    num_nodes = runspec.get("num_nodes",1)
    every_N = runspec.get("hypers_every_N",1)
    legendstr = "K="+str(num_nodes)
    if num_nodes == 1:
        linespec["color"] = "black"
    elif every_N == 1:
        linespec["color"] = "green"
        legendstr += ";every_N=1"
    elif every_N < num_nodes:
        linespec["color"] = "red"
        legendstr += ";1<every_N<K"
    elif every_N == num_nodes:
        linespec["color"] = "blue"
        legendstr += ";every_N==K"
    elif every_N > num_nodes:
        linespec["color"] = "orange"
        legendstr += ";every_N>K"
    else:
        linespec["color"] = "yellow"
        legendstr += ";every_N???K"
    linespec["linestyle"] = "-"
    return linespec,legendstr

def plot_measurement(memoized_infer, which_measurement, target_dataset_spec
                     ,by_time=True,run_spec_filter=None,save_str=None
                     ,title_str=None,ylabel_str=None,legend_args=None
                     ,do_legend=True):

    h_line = None
    if which_measurement == "predictive":
        problem = gen_problem(target_dataset_spec)
        h_line = np.mean(problem["test_lls_under_gen"])
    run_spec_filter = run_spec_filter if run_spec_filter is not None else (lambda x: True)
    title_str = title_str if title_str is not None else ""
    ylabel_str = ylabel_str if ylabel_str is not None else ""
    legend_args = legend_args if legend_args is not None else {"ncol":2,"prop":{"size":"medium"}}

    matching_runs = []
    matching_summaries = []
    for (args, summaries) in memoized_infer.iter():
        run_spec = args[0]
        if not run_spec_filter(run_spec):
            continue
        if str(run_spec["dataset_spec"]) == str(target_dataset_spec):
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

    matching_measurements = [extract_measurement(which_measurement,summary) 
                             for summary in matching_summaries]
    matching_linespecs,matching_legendstrs = speclist_to_plotspecs(matching_runs,["num_nodes","hypers_every_N"])

    # matching_measurements = []
    # matching_linespecs = []
    # matching_legendstrs = []
    # for (run, summary) in zip(matching_runs, matching_summaries):
    #     matching_measurements.append(extract_measurement(which_measurement, summary))
    #     linespec,legendstr = runspec_to_plotspec(run)
    #     matching_linespecs.append(linespec)
    #     matching_legendstrs.append(legendstr)

    pylab.figure()
    if do_legend:
        pylab.subplot(211)
    line_list = []
    for measurement, summary, linespec in zip(matching_measurements, matching_summaries, matching_linespecs):
        if by_time:
            xs = extract_time_elapsed_vs_iterations(summary)
            xlabel = "time (seconds)"
        else:
            xs = range(len(summary))
            xlabel = "iter"
        fh = pylab.plot(xs, measurement, color = linespec["color"], linestyle = linespec["linestyle"])
        line_list.append(fh[0])
        if h_line is not None:
            xlim = fh[0].get_axes().get_xlim()
            pylab.hlines(h_line,*xlim,color="red",linewidth=3)

    unique_legendstrs = []
    unique_lines = []
    legendstr_set = sets.Set()
    for legendstr,line in zip(matching_legendstrs,line_list):
        if legendstr not in legendstr_set:
            legendstr_set.add(legendstr)
            unique_legendstrs.append(legendstr)
            unique_lines.append(line)
    unique_lines = np.array(unique_lines)[np.argsort(unique_legendstrs)]
    unique_legendstrs = np.sort(unique_legendstrs)

    pylab.xlabel(xlabel)
    pylab.title(title_str)
    pylab.ylabel(ylabel_str)
    if do_legend:
        pylab.subplot(212)
        pylab.legend(unique_lines,unique_legendstrs,**legend_args)
    ##pylab.subplots_adjust(hspace=.4)
    if save_str is not None:
        pylab.savefig(save_str)
    pylab.close()

def try_plots(memoized_infer,which_measurements=None,run_spec_filter=None,do_legend=True,save_dir=None):
    temp = [(k,v) for k,v in memoized_infer.iter()] ##FIXME : how better to ensure memo pullis it down?
    which_measurements = ["ari"] if which_measurements is None else which_measurements
    save_dir = save_dir if save_dir is not None else os.path.expanduser("~/")
    legend_args = {"ncol":1,"markerscale":2}
    #
    for target_dataset_spec in extract_dataset_specs_from_memo(memoized_infer):
        cluster_str = "clusters" + str(target_dataset_spec["gen_z"][1]) ## FIXME : presumes dataset_spec is always balanced
        col_str = "cols" + str(target_dataset_spec["num_cols"])
        row_str = "rows" + str(target_dataset_spec["num_rows"])
        config_str = "_".join([col_str,row_str,cluster_str])    
        #
        for which_measurement in which_measurements:
            try:
                # by time
                plot_measurement(memoized_infer
                                 , which_measurement
                                 , target_dataset_spec
                                 , run_spec_filter=run_spec_filter
                                 , by_time=True
                                 , save_str=os.path.join(save_dir
                                                         ,"_".join([which_measurement,config_str,"time.png"]))
                                 , title_str=config_str
                                 , ylabel_str=which_measurement
                                 , legend_args=legend_args
                                 , do_legend=do_legend)
                # by iter
                plot_measurement(memoized_infer
                                 , which_measurement
                                 , target_dataset_spec
                                 , run_spec_filter=run_spec_filter
                                 , by_time=False
                                 , save_str=os.path.join(save_dir,
                                                         "_".join([which_measurement,config_str,"iter.png"]))
                                 , title_str=config_str
                                 , ylabel_str=which_measurement
                                 , legend_args=legend_args
                                 , do_legend=do_legend)
            except Exception, e:
                print e

def extract_measurement(which_measurement, one_runs_data):
    # measurement can be:
    if np.in1d(which_measurement,["num_clusters","ari","alpha","score"])[0]:
        return [summary[which_measurement] for summary in one_runs_data]
    elif which_measurement == "predictive":
        return [np.mean(summary["test_lls"]) for summary in one_runs_data]
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


####

def gen_default_run_spec(num_cols=16):
    dataset_spec = {}
    dataset_spec["gen_seed"] = 0
    dataset_spec["num_cols"] = num_cols
    dataset_spec["num_rows"] = 32*32
    dataset_spec["gen_alpha"] = 3.0 #FIXME: could make it MLE alpha later
    dataset_spec["gen_betas"] = np.repeat(0.1, dataset_spec["num_cols"])
    dataset_spec["gen_z"] = ("balanced", 32)
    dataset_spec["N_test"] = 128
    ##
    run_spec = {}
    run_spec["dataset_spec"] = dataset_spec
    run_spec["num_iters"] = 1000
    run_spec["num_nodes"] = 1
    run_spec["infer_seed"] = 0
    run_spec["infer_init_alpha"] = 3.0
    run_spec["infer_init_betas"] = np.repeat(0.1, dataset_spec["num_cols"])
    run_spec["infer_do_alpha_inference"] = True
    run_spec["infer_do_betas_inference"] = True
    run_spec["infer_init_z"] = None
    run_spec["time_seatbelt"] = 1200
    run_spec["ari_seatbelt"] = None
    run_spec["verbose_state"] = False
    #
    return run_spec
