import argparse
import copy
import cPickle
import datetime
import gzip
import os
import sets
import time
from threading import Thread
##
import numpy as np
import pylab
from numpy import array
from scipy.stats import linregress
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
import Cloudless.examples.DPMB.s3_helper as s3h
reload(s3h)
import Cloudless.examples.DPMB.settings as settings
reload(settings)
import pyx_functions as pf
reload(pf)
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
        seed_copy = (arr_or_other[0]
                     , arr_or_other[1].copy()
                     , arr_or_other[2]
                     , arr_or_other[3]
                     , arr_or_other[4])
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
    jobspec["infer_seed"] = arr_copy(last_summary["last_valid_seed"])
    jobspec["infer_init_alpha"] = last_summary["alpha"]
    jobspec["infer_init_betas"] = arr_copy(last_summary["betas"])
    jobspec["infer_init_z"] = arr_copy(last_summary["last_valid_zs"])
    jobspec["decanon_indices"] = arr_copy(last_summary["decanon_indices"])

def run_spec_from_model_specs(model_specs,seed_inferer):
    num_cols = 256 # FIXME : hardcoded
    gen_alpha_beta = 3.0
    #
    (x_indices, zs, gen_seed, inf_seed, master_alpha, betas,
     master_inf_seed, iter_num, child_counter) = model_specs
    # gen dataset_spec
    dataset_spec = {}
    dataset_spec["gen_seed"] = 0
    dataset_spec["num_cols"] = num_cols
    dataset_spec["num_rows"] = len(zs)
    # FIXME: are the following dataset_spec entries actually needed?
    dataset_spec["gen_alpha"] = gen_alpha_beta
    dataset_spec["gen_betas"] = np.repeat(gen_alpha_beta,num_cols)
    dataset_spec["gen_z"] = None
    # gen run_spec
    run_spec = {}
    run_spec["dataset_spec"] = dataset_spec
    run_spec["num_iters"] = seed_inferer.num_iters_per_step
    run_spec["num_nodes"] = 1
    run_spec["infer_seed"] = inf_seed
    # sub_alpha = alpha/num_nodes
    run_spec["infer_init_alpha"] = master_alpha/seed_inferer.num_nodes
    run_spec["infer_init_betas"] = betas
    # no hypers in child state inference
    run_spec["infer_do_alpha_inference"] = False
    run_spec["infer_do_betas_inference"] = False
    run_spec["infer_init_z"] = zs
    run_spec["time_seatbelt"] = None
    run_spec["ari_seatbelt"] = None
    run_spec["verbose_state"] = False
    return run_spec
                                        
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
            print str(datetime.datetime.now()) \
                + " :: acquiring lock :: " + str(self)
            self.lock.acquire()

    def release_lock(self):
        if self.lock is not None:
            print str(datetime.datetime.now()) \
                + " :: releasing lock :: " + str(self)
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
        assert list_len_delta <= 1 \
            , "Chunked_Job.pull_down_jobs: " \
            "child_jobspec_list got too far ahead of child_job_list"
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
            print "pull_down_jobs: job_value is None : " \
                + str(job_value is None) + str(self)
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
        # subtract 1 for initial state
        remaining_iters = (self.parent_jobspec["num_iters"]
                           - (len(self.consolidated_data)-1))
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
        
def gen_problem(dataset_spec,permute=True,save_str=None):
    # generate a state
    # initialize according to the generation parameters from dataset spec
    # containing all the training data only

    xs, zs, test_xs, test_lls, state = None, None, None, None, None
    if "pkl_file" in dataset_spec:
        # make sure the file is local
        pkl_file = dataset_spec['pkl_file']
        pkl_full_file = os.path.join(settings.data_dir,pkl_file)
        permute = False # presume data is already permuted
                        # repermuting now makes tracking ground truth hard
        if not os.path.isfile(pkl_full_file):
            have_file = s3h.S3_helper().verify_file(pkl_file)
            if not have_file:
                exception_str = 'gen_problem couldn\'t get pkl_file: ' + pkl_file
                raise Exception(exception_str)
        pkl_data = unpickle(pkl_full_file)
        # set problem variables
        xs = np.array(pkl_data["xs"],dtype=np.int32)
        if 'zs' in pkl_data:
            zs,ids = hf.canonicalize_list(pkl_data["zs"])
        else:
            zs = None
        test_xs = pkl_data['test_xs']
        if 'N_test' in dataset_spec:
            test_xs = np.array(test_xs[:dataset_spec['N_test']],dtype=np.int32)
        # verify dataset spec variables correct
        dataset_spec['num_cols'] = len(xs[0])
        dataset_spec['num_rows'] = len(xs)
        # FIXME : convenience operations, should do elsewhere
        dataset_spec.setdefault('gen_seed',0)
        dataset_spec.setdefault('gen_alpha',1.0)
        dataset_spec.setdefault('gen_betas',np.repeat(2.0,256))
    elif 'last_valid_zs' in dataset_spec:
        zs = dataset_spec['last_valid_zs']
        xs = dataset_spec['xs']
    else:
        zs = dataset_spec.get('gen_z', None)
        state = ds.DPMB_State(dataset_spec["gen_seed"],
                              dataset_spec["num_cols"],
                              dataset_spec["num_rows"],
                              init_alpha=dataset_spec["gen_alpha"],
                              init_betas=dataset_spec["gen_betas"],
                              init_z=zs,
                              init_x=xs)
        xs = state.getXValues()
        if save_str is not None:
            state.plot(save_str=save_str)
        
    if permute:
        # permute the data before passing out
        permutation_sequence = state.random_state.permutation(
            range(dataset_spec["num_rows"]))
        xs = [xs[perm_idx] for perm_idx in permutation_sequence]
        zs = [zs[perm_idx] for perm_idx in permutation_sequence]
        # canonicalize zs
        zs, cluster_idx = hf.canonicalize_list(temp_zs)

    if state is not None and 'N_test' in dataset_spec:
        test_xs, test_lls = \
            state.generate_and_score_test_set(dataset_spec["N_test"])
    problem = {}
    problem['xs'] = xs
    problem['zs'] = zs
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

def infer(run_spec, problem=None, send_zs=False):
    dataset_spec = run_spec["dataset_spec"]
    if problem is None:
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

    init_start_ts = datetime.datetime.now()
    inference_state = state_type(dataset_spec["gen_seed"],
                                 dataset_spec["num_cols"],
                                 dataset_spec["num_rows"],
                                 init_alpha=run_spec["infer_init_alpha"],
                                 init_betas=run_spec["infer_init_betas"],
                                 # init_z=run_spec["infer_init_z"],
                                 init_z=problem['zs'],
                                 init_x=np.array(problem["xs"],dtype=np.int32),
                                 **state_kwargs
                                 )
    init_delta_seconds = hf.delta_since(init_start_ts)

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
            # true_zs=problem["zs"],
            verbose_state=verbose_state,
            test_xs=problem["test_xs"]))
    summaries[-1]["timing"]["init"] = init_delta_seconds
    #
    print "saved initialization"
    #
    last_valid_zs = transitioner.state.getZIndices()
    last_valid_seed = transitioner.random_state.get_state()
    decanon_indices = transitioner.state.get_decanonicalizing_indices()
    for i in range(run_spec["num_iters"]):
        transition_return = transitioner.transition(
            time_seatbelt=time_seatbelt,
            ari_seatbelt=ari_seatbelt,
            # true_zs=problem["zs"]) # true_zs necessary for seatbelt 
            )
        hf.printTS("finished doing iteration " + str(i))
        next_summary = transitioner.extract_state_summary(
            # true_zs=problem["zs"],
            verbose_state=verbose_state,
            test_xs=problem["test_xs"],
            send_zs=send_zs,
            )
        time_elapsed_str = "%.1f" % next_summary["timing"].get("run_sum",0)
        hf.printTS("time elapsed: " + time_elapsed_str)
        if type(transition_return) == dict:
            summaries[-1]["break"] = transition_return
            summaries[-1]["failed_info"] = next_summary
            break
        summaries.append(next_summary)
        hf.printTS("finished saving iteration " + str(i))
        if hasattr(transitioner.state,"getZIndices"):
            last_valid_zs = transitioner.state.getZIndices()
            last_valid_seed = transitioner.random_state.get_state()
            decanon_indices = transitioner.state.get_decanonicalizing_indices()
    summaries[-1]["last_valid_zs"] = last_valid_zs
    summaries[-1]["last_valid_seed"] = last_valid_seed
    summaries[-1]["decanon_indices"] = decanon_indices
    return summaries

def infer_separate(run_spec):
    dataset_spec = run_spec["dataset_spec"]
    problem = gen_problem(dataset_spec)
    verbose_state = run_spec.get("verbose_state",False)
    decanon_indices = run_spec.get("decanon_indices",None)
    num_nodes = run_spec.get("num_nodes",2)
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

    init_start_ts = datetime.datetime.now()
    inference_state = state_type(dataset_spec["gen_seed"],
                                 dataset_spec["num_cols"],
                                 dataset_spec["num_rows"],
                                 init_alpha=run_spec["infer_init_alpha"],
                                 init_betas=run_spec["infer_init_betas"],
                                 init_z=run_spec["infer_init_z"],
                                 init_x = problem["xs"],
                                 **state_kwargs
                                 )
    init_delta_seconds = hf.delta_since(init_start_ts)

    print "...initialized"
    transitioner = model_type(
        inf_seed = run_spec["infer_seed"],
        state = inference_state,
        infer_alpha = run_spec["infer_do_alpha_inference"],
        infer_beta = run_spec["infer_do_betas_inference"],
        **model_kwargs
        )
# evolve individual models, at each step 
#   record individual state's progress (num_clusters,score)
#          to verify its doing something meaningful
#   pull data back to PDPMB_State to determine predictive, score 
    #
    master_summaries = []
    master_summaries.append(
        transitioner.extract_state_summary(
            true_zs=problem["zs"]
            ,verbose_state=verbose_state
            ,test_xs=problem["test_xs"]))
    master_summaries[-1]["timing"]["init"] = init_delta_seconds
    #
    node_summaries_element = []
    for node in inference_state.model_list:
        node_summaries_element.append(node.extract_state_summary())
    master_summaries[-1]["node_summaries"] = node_summaries_element
    print "saved initialization"
    #
    last_valid_zs = None
    decanon_indices = None
    for iter_idx in range(run_spec["num_iters"]):

        # never actually transition transitioner
        # only inference_state.model_list elements
        transitioner_return_list = []
        for node in inference_state.model_list:
            transitioner_return_list.append(
                node.transition(time_seatbelt=time_seatbelt))
            if transitioner_return_list[-1] is not None:
                break
        # check for seatbelt violations in any of the individual states
        if len(filter(None,transitioner_return_list)) > 0:
            break
        hf.printTS("finished doing iteration " + str(iter_idx))

        # get node summaries
        node_summaries_element = []
        for node in inference_state.model_list:
            node_summaries_element.append(node.extract_state_summary())
        time_elapsed = max([
                summary["timing"].get("run_sum",0)
                for summary in node_summaries_element
                ])

        # get master summary 
        next_summary = transitioner.extract_state_summary(
            verbose_state=verbose_state
            ,test_xs=problem["test_xs"])
        time_elapsed_str = "%.1f" % time_elapsed
        hf.printTS("time elapsed: " + time_elapsed_str)
        next_summary["node_summaries"] = node_summaries_element
        master_summaries.append(next_summary)
        hf.printTS("finished saving iteration " + str(iter_idx))
        if hasattr(transitioner.state,"getZIndices"):
            last_valid_zs = transitioner.state.getZIndices()
            decanon_indices = transitioner.state.get_decanonicalizing_indices()
    master_summaries[-1]["last_valid_zs"] = last_valid_zs
    master_summaries[-1]["decanon_indices"] = decanon_indices
    return master_summaries    

def extract_dataset_specs_from_memo(asyncmemo):
    ALL_RUN_SPECS = [eval(key)[0] for key in asyncmemo.memo.keys()]
    ALL_DATASET_SPEC_STRS = [str(runspec["dataset_spec"]) for runspec in ALL_RUN_SPECS]
    import sets
    ALL_DATASET_SPECS = [eval(spec_str)
                         for spec_str in list(sets.Set(ALL_DATASET_SPEC_STRS))]
    return ALL_DATASET_SPECS

def get_open(file_str):
    if file_str[-3:] == ".gz":
        my_open = gzip.open
    else:
        my_open = open
    return my_open

def pickle(var_to_pkl, file_str, dir=None):
    my_open = get_open(file_str)
    if dir:
        file_str = os.path.join(dir, file_str)
    with my_open(file_str, 'wb') as fh:
        cPickle.dump(var_to_pkl, fh)

def unpickle(file_str, dir=None):
    from numpy import array
    my_open = get_open(file_str)
    if dir:
        file_str = os.path.join(dir, file_str)
    with my_open(file_str, 'rb') as fh:
        var_from_pkl = cPickle.load(fh)
    return var_from_pkl

def pickle_asyncmemoize(asyncmemo,file_str):
    pickle(asyncmemo.memo,file_str)

def unpickle_asyncmemoize(asyncmemo,file_str):
    pickled_memo = unpickle(file_str)
    #
    ALL_RUN_SPECS = [eval(run_spec)[0] for run_spec in pickled_memo.keys()]
    new_memo = dict(zip(
            [str((run_spec,)) for run_spec in ALL_RUN_SPECS]
            , pickled_memo.values()
            ))
    new_args = dict(zip(
            [str((run_spec,)) for run_spec in ALL_RUN_SPECS]
            , [(run_spec,) for run_spec in ALL_RUN_SPECS]
            ))
    # FIXME: action through setting asyncmemo elements
    #        perhaps should return these for caller to set?
    asyncmemo.memo = new_memo
    asyncmemo.args = new_args
    return ALL_RUN_SPECS

def pickle_if_done(memoized_infer,file_str="pickled_jobs.pkl"):
    status = memoized_infer.report_status()
    if status["waiting"] != 0:
        print "Not done, not pickling"
        return False
    else:
        memoized_infer.advance()
        pickle_asyncmemoize(memoized_infer,file_str)
        print "Done all jobs, memo pickled"
        return True

def runspec_to_plotspec_bak(runspec):
    linespec = {}
    legendstr = ""
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
        linespec["linestyle"] = linestyles[
            (legendstr_idx / len(colors)) % len(linestyles)
            ]
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
                     ,do_legend=True,h_line=None):

    run_spec_filter = run_spec_filter \
        if run_spec_filter is not None \
        else (lambda x: True)
    title_str = title_str if title_str is not None else ""
    ylabel_str = ylabel_str if ylabel_str is not None else ""
    legend_args = legend_args \
        if legend_args is not None \
        else {"ncol":2,"prop":{"size":"medium"}}

    matching_runs = []
    matching_summaries = []
    target_dataset_spec_copy = target_dataset_spec.copy()
    target_dataset_spec_copy.pop("gen_seed")
    for (args, summaries) in memoized_infer.iter():
        run_spec = args[0]
        if not run_spec_filter(run_spec):
            continue
        run_spec_copy = run_spec["dataset_spec"].copy()
        run_spec_copy.pop("gen_seed")
        if str(run_spec_copy) == str(target_dataset_spec_copy):
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
    matching_linespecs,matching_legendstrs = speclist_to_plotspecs(
        matching_runs
        , ["num_nodes","hypers_every_N"]
        )

    pylab.figure()
    if do_legend:
        pylab.subplot(211)
    line_list = []
    for measurement, summary, linespec \
            in zip(matching_measurements, matching_summaries, matching_linespecs):
        if by_time:
            xs = extract_time_elapsed_vs_iterations(summary)
            xlabel = "time (seconds)"
        else:
            xs = range(len(summary))
            xlabel = "iter"
        fh = pylab.plot(
            xs
            , measurement
            , color = linespec["color"]
            , linestyle = linespec["linestyle"])
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
    memoized_infer.advance()
    which_measurements = ["ari"] if which_measurements is None else which_measurements
    save_dir = save_dir if save_dir is not None else os.path.expanduser("~/")
    legend_args = {"ncol":1,"markerscale":2}
    #
    h_line=None
    if len(filter(lambda x:x=="predictive",which_measurements)) > 0:
        target_dataset_spec =  extract_dataset_specs_from_memo(memoized_infer)[0]
        # FIXME : only process 1?
        problem = gen_problem(target_dataset_spec)
        h_line = np.mean(problem["test_lls_under_gen"])

    for target_dataset_spec in extract_dataset_specs_from_memo(memoized_infer):
        # FIXME : presumes dataset_spec is always balanced
        cluster_str = "clusters" + str(target_dataset_spec["gen_z"][1])
        col_str = "cols" + str(target_dataset_spec["num_cols"])
        row_str = "rows" + str(target_dataset_spec["num_rows"])
        config_str = "_".join([col_str,row_str,cluster_str])    
        #
        for which_measurement in which_measurements:

            try:
                # by time
                plot_measurement(
                    memoized_infer
                    , which_measurement
                    , target_dataset_spec
                    , run_spec_filter=run_spec_filter
                    , by_time=True
                    , save_str=os.path.join(
                        save_dir
                        ,"_".join([which_measurement,config_str,"time.png"]))
                    , title_str=config_str
                    , ylabel_str=which_measurement
                    , legend_args=legend_args
                    , do_legend=do_legend
                    , h_line=h_line)
                # by iter
                plot_measurement(
                    memoized_infer
                    , which_measurement
                    , target_dataset_spec
                    , run_spec_filter=run_spec_filter
                    , by_time=False
                    , save_str=os.path.join(
                        save_dir,
                        "_".join([which_measurement,config_str,"iter.png"]))
                    , title_str=config_str
                    , ylabel_str=which_measurement
                    , legend_args=legend_args
                    , do_legend=do_legend
                    , h_line=h_line)
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

def timing_plots(cluster_counts,z_diff_times,args,save_dir=None):
    save_dir = os.path.expanduser("~/") if save_dir is None else save_dir

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

    num_cols = args.num_cols
    num_rows = args.num_rows
    cutoff = cluster_counts.max()/3
    box_every_n = max(1,len(box_input)/10)

    pylab.figure()
    pylab.plot(cluster_counts,z_diff_times,'x')
    pylab.title(title_str)
    pylab.xlabel("num_clusters")
    pylab.ylabel("single-z scan time (seconds)")
    fig_str = "scatter_scan_times_num_cols_"+str(num_cols)+"_num_rows_"+str(num_rows)
    pylab.savefig(os.path.join(save_dir,fig_str))
    #
    pylab.figure()
    pylab.boxplot(box_input.values()[::box_every_n]
                  ,positions=box_input.keys()[::box_every_n]
                  ,sym="")
    pylab.title(title_str)
    pylab.xlabel("num_clusters")
    pylab.ylabel("single-z scan time (seconds)")
    fig_str = "boxplot_scan_times_num_cols_"+str(num_cols)+"_num_rows_"+str(num_rows)
    pylab.savefig(os.path.join(save_dir,fig_str))
    pylab.close()
    #
    try:
        pylab.figure()
        pylab.hexbin(cluster_counts[cluster_counts<cutoff],z_diff_times[cluster_counts<cutoff])
        pylab.title(title_str)
        pylab.xlabel("num_clusters")
        pylab.ylabel("single-z scan time (seconds)")
        pylab.colorbar()
        fig_str = "hexbin_scan_times_num_cols_"+str(num_cols)+"_num_rows_"+str(num_rows)+"_lt_"+str(cutoff)
        pylab.savefig(os.path.join(save_dir,fig_str))
    except Exception, e:
        print e
    #
    try:
        pylab.figure()
        pylab.hexbin(cluster_counts[cluster_counts>cutoff],z_diff_times[cluster_counts>cutoff])
        pylab.title(title_str)
        pylab.xlabel("num_clusters")
        pylab.ylabel("single-z scan time (seconds)")
        pylab.colorbar()
        fig_str = "hexbin_scan_times_num_cols_"+str(num_cols)+"_num_rows_"+str(num_rows)+"_gt_"+str(cutoff)
        pylab.savefig(os.path.join(save_dir,fig_str))
    except Exception, e:
        print e


####

def gen_default_cifar_run_spec(problem_file,infer_seed,num_iters):
    dataset_spec = {}
    dataset_spec['pkl_file'] = problem_file
    dataset_spec['gen_seed'] = 0
    dataset_spec['gen_alpha'] = 1.0
    dataset_spec['gen_betas'] = np.repeat(2.0,256)
    dataset_spec['N_test'] = 300
    run_spec = {}
    run_spec['infer_seed'] = infer_seed
    run_spec['dataset_spec'] = dataset_spec ## settings.cifar_100_problem_file}
    run_spec["num_iters"] = num_iters
    run_spec["time_seatbelt"] = None
    run_spec["infer_init_z"] = None
    run_spec['infer_init_alpha'] = dataset_spec['gen_alpha']
    run_spec['infer_init_betas'] = dataset_spec['gen_betas']
    run_spec["num_nodes"] = 1
    run_spec["infer_do_alpha_inference"] = True
    run_spec["infer_do_betas_inference"] = True
    #
    return run_spec

def gen_default_run_spec(num_clusters, vectors_per_cluster,
                         num_cols=256, beta_d=3.0
                         ):
    dataset_spec = {}
    dataset_spec["gen_seed"] = 0
    dataset_spec["num_cols"] = num_cols
    dataset_spec["num_rows"] = num_clusters*vectors_per_cluster
    dataset_spec["gen_alpha"] = 3.0 #FIXME: could make it MLE alpha later
    dataset_spec["gen_betas"] = np.repeat(beta_d, dataset_spec["num_cols"])
    dataset_spec["gen_z"] = ("balanced", num_clusters)
    dataset_spec["N_test"] =  max(64,dataset_spec["num_rows"]/16)
    #
    run_spec = {}
    run_spec["dataset_spec"] = dataset_spec
    run_spec["num_iters"] = 10
    run_spec["num_nodes"] = 1
    run_spec["infer_seed"] = 0
    run_spec["infer_init_alpha"] = 3.0
    run_spec["infer_init_betas"] = np.repeat(beta_d, dataset_spec["num_cols"])
    run_spec["infer_do_alpha_inference"] = True
    run_spec["infer_do_betas_inference"] = True
    run_spec["infer_init_z"] = None
    run_spec["hypers_every_N"] = 1
    run_spec["time_seatbelt"] = 60
    run_spec["ari_seatbelt"] = None
    run_spec["verbose_state"] = False
    #
    return run_spec

def gen_runspec_from_argparse(parser):

    run_spec = gen_default_run_spec(
        num_clusters = parser.num_clusters,
        vectors_per_cluster = parser.num_rows/parser.num_clusters,
        num_cols = parser.num_cols,
        beta_d = parser.beta_d
        )
    run_spec["dataset_spec"]["num_rows"] = parser.num_rows
    run_spec["dataset_spec"]["gen_z"] = ("balanced",parser.num_clusters)
    run_spec["num_iters"] = parser.num_iters
    run_spec["num_nodes"] = parser.num_nodes
    run_spec["hypers_every_N"] = parser.num_nodes
    run_spec["time_seatbelt"] = parser.time_seatbelt
    run_spec["infer_init_z"] = None \
        if parser.balanced == -1 \
        else ("balanced",parser.balanced)
    run_spec["N_test"] = max(64,run_spec["dataset_spec"]["num_rows"]/16)

    return run_spec

def gen_default_arg_parser(description=None):

    description = "" if description is None else description

    default_save_dir = os.path.expanduser("~/Run/")
    default_pkl_file_str = os.join(default_save_dir,"pickled_jobs.pkl.gz")
    # load up some arguments
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--num_cols',default=256,type=int)
    parser.add_argument('--num_rows',default=32*32,type=int)
    parser.add_argument('--num_clusters',default=32,type=int)
    parser.add_argument('--beta_d',default=3.0,type=float)
    parser.add_argument('--balanced',default=-1,type=int)
    parser.add_argument('--num_iters',default=1000,type=int)
    parser.add_argument('--num_nodes',default=1,type=int)
    parser.add_argument('--time_seatbelt',default=60,type=int)
    parser.add_argument('--save_dir',default=default_save_dir,type=str)
    parser.add_argument('--pkl_file_str',default=default_pkl_file_str,type=str)

    return parser

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

def distribute_data(inf_seed, num_nodes, zs):
    random_state = hf.generate_random_state(inf_seed)
    mus = np.repeat(1.0/num_nodes, num_nodes)

    # group x_indices
    # zs must be canonicalized!!!
    cluster_data_indices = {}
    for x_index, z in enumerate(zs):
        cluster_data_indices.setdefault(z,[]).append(x_index)

    # deal out data to states
    node_data_indices = [[] for node_idx in xrange(num_nodes)]
    node_zs = [[] for node_idx in xrange(num_nodes)]
    num_clusters = len(cluster_data_indices)
    # determine node choices in bulk
    bulk_counts = random_state.multinomial(num_clusters,mus)
    node_choices = []
    for cluster_idx,cluster_count in enumerate(bulk_counts):
        node_choices.extend(np.repeat(cluster_idx,cluster_count))
    node_choices = random_state.permutation(node_choices)
    #
    for cluster_idx,dest_node in enumerate(node_choices):
        vector_index_list = cluster_data_indices[cluster_idx]
        node_data_indices[dest_node].extend(vector_index_list)
        new_zs_value = node_zs[dest_node][-1] + 1 \
            if len(node_zs[dest_node]) > 0 else 0
        node_zs[dest_node].extend(np.repeat(new_zs_value,len(vector_index_list)))
    gen_seed_list = [int(x) for x in random_state.tomaxint(num_nodes)]
    inf_seed_list = [int(x) for x in random_state.tomaxint(num_nodes)]

    return node_data_indices, node_zs, gen_seed_list, inf_seed_list, random_state

def consolidate_zs(zs_list):
    single_state = None
    cluster_idx = 0
    zs = []
    for temp_zs in zs_list:
        if len(temp_zs) == 0:
            continue
        zs.extend(np.array(temp_zs) + cluster_idx)
        max_zs = max(temp_zs)
        cluster_idx += max_zs + 1
    return zs
