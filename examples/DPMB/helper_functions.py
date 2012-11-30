import datetime
import os
import re 
import sys
import pdb
from timeit import default_timer
from collections import defaultdict
#
import pylab
# import pandas # imported below only in case actually used
import matplotlib
import numpy as np
import numpy
import scipy.special as ss
from numpy.random import RandomState
#
import DPMB_State as ds
import pyx_functions as pf
import Cloudless.examples.DPMB.s3_helper as s3h

def transition_single_z(vector,random_state):
    cluster = vector.cluster
    state = cluster.state
    #
    vector.cluster.deassign_vector(vector)

    score_vec,draw = pf.calculate_cluster_conditional(
        state,vector,random_state.uniform())

    cluster = None
    if draw == len(state.cluster_list):
        cluster = state.generate_cluster_assignment(force_new = True)
    else:
        cluster = state.cluster_list[draw]
    cluster.assign_vector(vector)
    #
    return len(score_vec)-1

####################
# PROBABILITY FUNCTIONS

# deprecated : use pyx_functions optimized version instead
# def renormalize_and_sample(random_state,logpstar_vec):
#     p_vec = log_conditional_to_norm_prob(logpstar_vec)
#     randv = random_state.uniform()
#     for (i, p) in enumerate(p_vec):
#         if randv < p:
#             return i
#         else:
#             randv = randv - p

def log_conditional_to_norm_prob(logp_list):
    maxv = max(logp_list)
    scaled = [logpstar - maxv for logpstar in logp_list]
    logZ = reduce(np.logaddexp, scaled)
    logp_vec = [s - logZ for s in scaled]
    return np.exp(logp_vec)

def cluster_vector_joint(vector,cluster,state):
    alpha = state.alpha
    numVectors = len(state.get_all_vectors())
    if cluster is None or len(cluster.vector_list) == 0:
        alpha_term = np.log(alpha) - np.log(numVectors-1+alpha)
        data_term = state.num_cols*np.log(.5)
    else:
        boolIdx = np.array(vector.data,dtype=type(True))
        alpha_term = np.log(len(cluster.vector_list)) - np.log(numVectors-1+alpha)
        numerator1 = boolIdx * np.log(cluster.column_sums + state.betas)
        numerator2 = (~boolIdx) * np.log(len(cluster.vector_list) \
                                             - cluster.column_sums + state.betas)
        denominator = np.log(len(cluster.vector_list) + 2*state.betas)
        data_term = (numerator1 + numerator2 - denominator).sum()
    retVal = alpha_term + data_term
    return retVal,alpha_term,data_term

def create_alpha_lnPdf(state):
    # Note : this is extraneous work for relative probabilities
    #      : but necessary to determine true distribution probabilities
    lnProdGammas = sum([ss.gammaln(len(cluster.vector_list)) 
                        for cluster in state.cluster_list])

    lnPdf = lambda alpha: ss.gammaln(alpha) \
        + len(state.cluster_list)*np.log(alpha) \
        - ss.gammaln(alpha+len(state.vector_list)) \
        + lnProdGammas

    return lnPdf

def create_beta_lnPdf(state,col_idx):
    S_list = [cluster.column_sums[col_idx] for cluster in state.cluster_list]
    R_list = [len(cluster.vector_list) - cluster.column_sums[col_idx] \
                  for cluster in state.cluster_list]
    lnPdf = lambda beta_d: sum([ss.gammaln(2*beta_d) - 2*ss.gammaln(beta_d)
                                + ss.gammaln(S+beta_d) + ss.gammaln(R+beta_d)
                                - ss.gammaln(S+R+2*beta_d) 
                                for S,R in zip(S_list,R_list)])
    return lnPdf

def slice_sample_alpha(state,init=None):
    logprob = create_alpha_lnPdf(state)
    lower = state.alpha_min
    upper = state.alpha_max
    init = state.alpha if init is None else init
    slice = np.log(state.random_state.uniform()) + logprob(init)
    while True:
        a = state.random_state.uniform()*(upper-lower) + lower
        if slice < logprob(a):
            break;
        elif a < init:
            lower = a
        elif a > init:
            upper = a
        else:
            raise Exception('Slice sampler for alpha shrank to zero.')
    return a

def slice_sample_beta(state,col_idx,init=None):
    logprob = create_beta_lnPdf(state,col_idx)
    lower = state.beta_min
    upper = state.beta_max
    init = state.betas[col_idx] if init is None else init
    slice = np.log(state.random_state.uniform()) + logprob(init)
    while True:
        a = state.random_state.uniform()*(upper-lower) + lower
        if slice < logprob(a):
            break;
        elif a < init:
            lower = a
        elif a > init:
            upper = a
        else:
            raise Exception('Slice sampler for beta shrank to zero.')
    return a

def calc_alpha_conditional(state):
    original_alpha = state.alpha
    ##
    grid = state.get_alpha_grid()
    lnPdf = create_alpha_lnPdf(state)
    logp_list = []
    state.removeAlpha(lnPdf)
    base_score = state.score
    for test_alpha in grid:
        state.setAlpha(lnPdf,test_alpha)
        logp_list.append(state.score)
        state.removeAlpha(lnPdf)
    # Note: log gridding introduces (implicit) -log(x) prior
    #     : to get uniform prior, need to add back np.log(x)
    # logp_list += np.log(grid)
    
    state.setAlpha(lnPdf,original_alpha)
    return np.array(logp_list)-base_score,lnPdf,grid

def calc_beta_conditional(state,col_idx):
    original_beta = state.betas[col_idx]
    ##
    grid = state.get_beta_grid()
    lnPdf = create_beta_lnPdf(state,col_idx)
    logp_list = []
    state.removeBetaD(lnPdf,col_idx)
    base_score = state.score
    # Note: log gridding introduces (implicit) -log(x) prior
    #     : to get uniform prior, need to add back np.log(x)
    # prior_func = lambda x : +np.log(x) # uniform
    # prior_func = lambda x: -x          # unormalized gamma_func(k=1, theta=1)
    prior_func = None                    # retain implicit -log prior
    logp_arr = pf.calc_beta_conditional_helper(
        state,grid,col_idx,prior_func)
    logp_list = logp_arr.tolist()[0]
    ##
    state.setBetaD(lnPdf,col_idx,original_beta)
    return np.array(logp_list)-base_score,lnPdf,grid

# deprecated, use pyx_functions version
# def calculate_cluster_conditional(state,vector):
#     ##vector should be unassigned
#     conditionals = []
#     for cluster in state.cluster_list + [None]:
#         scoreDelta,alpha_term,data_term = cluster_vector_joint(
#             vector,cluster,state)
#         conditionals.append(scoreDelta + state.score)
#     return conditionals

def calculate_node_conditional(pstate,cluster):
    conditionals = pstate.mus
    return conditionals

def mle_alpha(clusters,points_per_cluster,max_alpha=100,alphas=None):
    if alphas is None:
        alphas = range(1,max_alpha)
    alpha_ps = [ss.gammaln(alpha) + clusters*np.log(alpha) 
                       - ss.gammaln(clusters*points_per_cluster+alpha) 
                       for alpha in alphas]
    mle = alphas[np.argmax(alpha_ps)]
    return mle, alpha_ps, alphas

def mhSample(initVal,nSamples,lnPdf,sampler,random_state):
    samples = [initVal]
    priorSample = initVal
    for counter in range(nSamples):
        unif = random_state.uniform()
        proposal = sampler(priorSample)
        thresh = np.exp(lnPdf(proposal) - lnPdf(priorSample)) ## presume symmetric
        if np.isfinite(thresh) and unif < min(1,thresh):
            samples.append(proposal)
        else:
            samples.append(priorSample)
        priorSample = samples[-1]
    return samples

####################
# UTILITY FUNCTIONS
def plot_data(data, fh=None, h_lines=None, title_str='',
              interpolation='nearest', linewidth=1, h_line_alpha=0.7, **kwargs):
    if fh is None:
        fh = pylab.figure()
    pylab.imshow(data, interpolation=interpolation,
                 cmap=matplotlib.cm.binary, **kwargs)
    if h_lines is not None:
        for h_line in h_lines:
            pylab.axhline(h_line-.5, color='red', linewidth=linewidth,
                          alpha=h_line_alpha)
    pylab.title(title_str)
    return fh

def bar_helper(x, y, fh=None, v_line=None, title_str='', which_id=0):
    if fh is None:
        fh = pylab.figure()
    min_delta = min(np.diff(x))
    pylab.bar(x, y, width=min_delta)
    if v_line is not None:
        pylab.vlines(v_line,*fh.get_axes()[which_id].get_ylim()
                     ,color="red",linewidth=3)
    pylab.ylabel(title_str)
    return fh

def zs_to_hlines(zs):
    is_different = pylab.diff(zs) != 0
    which_different = matplotlib.mlab.find(is_different) + 1
    return which_different

def printTS(printStr):
    print datetime.datetime.now().strftime("%H:%M:%S") + " :: " + printStr
    sys.stdout.flush()

def listCount(listIn):
    return dict([(currValue,sum(np.array(listIn)==currValue)) 
                 for currValue in np.unique(listIn)])

def delta_since(start_dt):
    try: ##older datetime modules don't have .total_seconds()
        delta = (datetime.datetime.now()-start_dt).total_seconds()
    except Exception, e:
        delta = (datetime.datetime.now()-start_dt).seconds()
    return delta

def convert_rpa_representation(intarray):
    num_cols = 32*len(intarray[0])
    num_rows = len(intarray)
    data = np.ndarray((num_rows,num_cols),dtype=np.int32)
    for row_idx,row in enumerate(intarray):
        binary_rep = []
        for number in row:
            string_rep = bin(number)[2:].zfill(32)
            binary_rep.extend([int(value) for value in string_rep])
        data[row_idx] = binary_rep
    return data

def cifar_data_to_image(raw_data,filename=None):
    image_data = raw_data.reshape((3,32,32)).T
    fh = pylab.figure(figsize=(.5,.5))
    pylab.imshow(image_data,interpolation='nearest')
    if filename is not None:
        pylab.savefig(filename)
        pylab.close()
    return fh

def canonicalize_list(in_list):
    z_indices = []
    next_id = 0
    cluster_ids = {}
    for el in in_list:
        if el not in cluster_ids:
            cluster_ids[el] = next_id
            next_id += 1
        z_indices.append(cluster_ids[el])
    return z_indices,cluster_ids

def list_of_x_indices_to_zs(list_of_x_indices):
    cluster_counts = [len(sub_list) for sub_list in list_of_x_indices]
    zs = []
    for cluster_idx, cluster_count in enumerate(cluster_counts):
        zs.extend(numpy.repeat(cluster_idx, cluster_count))
    return zs

def ensure_pandas():
    try:
        import pandas
    except ImportError:
        pandas_uri = 'http://pypi.python.org/packages/source/p/pandas/' + \
            'pandas-0.7.0rc1.tar.gz'
        system_str = ' '.join(['easy_install', pandas_uri])
        os.system(system_str)

def create_links(filename_or_series,source_dir,dest_dir):
    ensure_pandas()
    import pandas
    series = None
    if isinstance(filename_or_series,str):
        series = pandas.Series.from_csv(filename_or_series)
    elif isinstance(filename_or_series,pandas.Series):
        series = filename_or_series
    else:
        print "unknown type for filename_or_series!"
        return
    #
    if len(os.listdir(dest_dir)) != 0:
        print dest_dir + " not empty, empty and rerun"
        return
    #
    for vector_idx,cluster_idx in series.iteritems():
        cluster_dir = os.path.join(dest_dir,str(cluster_idx))
        if not os.path.isdir(cluster_dir):
            os.mkdir(cluster_dir)
        filename = ("%05d" % vector_idx) + ".png"
        from_file = os.path.join(source_dir,filename)
        to_file = os.path.join(cluster_dir,filename)
        #
        os.symlink(from_file,to_file)
    
####################
# SEED FUNCTIONS
def generate_random_state(seed):
    random_state = RandomState()
    if type(seed) == tuple:
        random_state.set_state(seed)
    elif type(seed) == int:
        random_state.seed(seed)
    elif type(seed) == RandomState:
        random_state = seed
    else:
        raise Exception("Bad argument to generate_random_state: " + str(seed)) 
    return random_state

####################
# ARI FUNCTIONS
def get_matched_subset(list_1, list_2, seed, count):
    random_state = generate_random_state(seed)
    num_vectors = len(list_1)
    subset_indices = random_state.permutation(xrange(num_vectors))[:count]
    # subset_indices = random_state.randint(low=0, high=num_vectors, count)
    return numpy.array(list_1)[subset_indices], numpy.array(list_2)[subset_indices]

def calc_ari(group_idx_list_1, group_idx_list_2, seed=0, count=10000):
    ##https://en.wikipedia.org/wiki/Rand_index#The_contingency_table
    group_idx_list_1, group_idx_list_2 = get_matched_subset(
        group_idx_list_1, group_idx_list_2, seed, count)
    Ns, As, Bs = gen_contingency_data(group_idx_list_1, group_idx_list_2)
    n_choose_2 = choose_2_sum(np.array([len(group_idx_list_1)]))
    cross_sums = choose_2_sum(Ns[Ns>1])
    a_sums = choose_2_sum(As)
    b_sums = choose_2_sum(Bs)
    numerator = n_choose_2 * cross_sums - a_sums * b_sums
    denominator = .5 * n_choose_2 * (a_sums + b_sums) - a_sums * b_sums
    return numerator / denominator

def choose_2_sum(x):
    return sum(x * (x - 1) / 2.0)

def gen_contingency_data(group_idx_list_1, group_idx_list_2):
    group_idx_dict_1 = defaultdict(set)
    for list_idx, group_idx in enumerate(group_idx_list_1):
        group_idx_dict_1[group_idx].add(list_idx)
    group_idx_dict_2 = defaultdict(set)
    for list_idx, group_idx in enumerate(group_idx_list_2):
        group_idx_dict_2[group_idx].add(list_idx)
    ##
    array_dim = (len(group_idx_dict_1), len(group_idx_dict_2))
    Ns = np.ndarray(array_dim)
    for idx_1, value1 in enumerate(group_idx_dict_1.values()):
        for idx_2, value2 in enumerate(group_idx_dict_2.values()):
            Ns[idx_1, idx_2] = len(value1.intersection(value2))
    As = Ns.sum(axis=1)
    Bs = Ns.sum(axis=0)
    return Ns, As, Bs

class Timer(object):
    def __init__(self, task='action', verbose=False):
        self.task = task
        self.verbose = verbose
        self.timer = default_timer
    def __enter__(self):
        self.start = self.timer()
        return self
    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs * 1000 # millisecs
        if self.verbose:
            print '%s took:\t% 7d ms' % (self.task, self.elapsed)

def get_max_iternum_filenames(dir_str):
    def get_max_iternum(file_tuples, basename):
        is_same_basename = lambda in_tuple: in_tuple[0] == basename
        basename_file_tuples = filter(is_same_basename, file_tuples)
        iternums = map(lambda x: int(x[1]), basename_file_tuples)
        return str(max(iternums))
    base_re = re.compile('^(summary_.*iternum)(\d+).pkl.gz$')
    base_re_func = lambda filename: base_re.match(filename)
    get_base_match = lambda in_list: filter(None, map(base_re_func, in_list))
    get_base_names = lambda in_list: \
        list(set(map(lambda x: x.groups()[0], get_base_match(in_list))))
    get_base_tuples = lambda in_list: \
        list(set(map(lambda x: x.groups(), get_base_match(in_list))))
    #
    all_files = os.listdir(dir_str)
    base_names = get_base_names(all_files)
    base_tuples = get_base_tuples(all_files)
    max_tuples = [
        (base_name, get_max_iternum(base_tuples, base_name))
        for base_name in base_names
        ]
    create_filename = lambda in_tuple: ''.join(in_tuple) + '.pkl.gz'
    filenames = map(create_filename, max_tuples)
    return filenames

def visualize_mle_alpha(cluster_list=None,points_per_cluster_list=None,max_alpha=None):
    import pylab
    cluster_list = cluster_list if cluster_list is not None else 10**numpy.arange(0,4,.5)
    points_per_cluster_list = points_per_cluster_list if points_per_cluster_list is not None else 10**numpy.arange(0,4,.5)
    max_alpha = max_alpha if max_alpha is not None else int(1E4)
    ##
    mle_vals = []
    for clusters in cluster_list:
        for points_per_cluster in points_per_cluster_list:
            mle_vals.append([clusters,points_per_cluster,dm.mle_alpha(clusters,points_per_cluster,max_alpha=max_alpha)])
    ##
    mle_vals = numpy.array(mle_vals)
    pylab.figure()
    pylab.loglog(mle_vals[:,0],mle_vals[:,1],color='white') ##just create the axes
    pylab.xlabel("clusters")
    pylab.ylabel("points_per_cluster")
    pylab.title("MLE alpha for a given data configuration\nmax alpha: "+str(max_alpha-1))
    for clusters,points_per_cluster,mle in mle_vals:
        pylab.text(clusters,points_per_cluster,str(int(mle)),color='red')

def echo_date(in_str, outfile='/tmp/steps'):
    cmd_str = 'echo "`date` :: ' + in_str + '" >> ' + outfile
    os.system(cmd_str)

def verify_file_helper(filename, bucket_dir_suffix,
                       unpickle=False, write_s3=False):
    local_dir = os.path.join(data_dir, bucket_dir_suffix)
    bucket_dir = os.path.join('tiny_image_summaries', bucket_dir_suffix)
    s3 = s3h.S3_helper(bucket_dir=bucket_dir, local_dir=local_dir)
    s3.verify_file(filename, write_s3=write_s3)
    pkl_contents = None
    if unpickle:
        pkl_contents = rf.unpickle(filename, dir=local_dir)
    return pkl_contents

def verify_problem_local(bucket_dir_suffix):
    verify_file_helper('problem.h5', bucket_dir_suffix, unpickle=False)
    problem = verify_file_helper('problem.pkl.gz', bucket_dir_suffix,
                                 unpickle=True)
    return problem 
