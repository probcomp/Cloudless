import Cloudless.base

# FIXME: handle named arguments correctly, to simplify building up
# experiments and remove the need for all this extra name-tracking
# and handle code with default arguments specified nicely

class AsyncMemoize:
    def __init__(self, name, argnames, f):
        self.f = f
        self.memo = {}
        self.args = {}
        self.jobs = {}
 
        self.name = name
        self.argnames = argnames
        
        self.recording_only = False
        self.to_call = []

        if name in Cloudless.base.memoizers:
            raise Exception("Already have a procedure under " + name)
        else:
            Cloudless.base.memoizers[name] = self

    def clear(self):
        # FIXME: is this bad-leaky?
        self.memo = {}
        self.args = {}
        self.jobs = {}

    # FIXME: make deleting this object deregister itself from Cloudless

    def start_recording(self):
        self.recording_only = True
        
    def submit_jobs(self):
        self.recording_only = False
        for (i, args) in enumerate(self.to_call):
            self.__call__(args)
        self.to_call = []

    def __call__(self, *args):
        if self.recording_only:
            self.to_call.append(args)
            return None

        key = str(args)
        self.args[key] = args

        if not key in self.memo:
            # try to apply it
            if Cloudless.base.remote:
                view = Cloudless.base.get_view()
                async_res = view.apply_async(self.f, *args)
                
                # if it's done, store and return
                if async_res.ready() and async_res.successful():
                    val = async_res.get()
                    self.memo[key] = val
                    return val
                else:
                    # else store it in jobs and return None
                    # NOTE: this keeps all failed jobs around all the time
                    self.jobs[key] = {'remote':True, 'async_res':async_res}
                    return None
            else:
                # we are running locally
                try:
                    res = apply(self.f, *args)
                    self.memo[key] = res
                    return res
                except Exception as e:
                    # FIXME: support really eager mode, for debugging?
                    #        with ipython? so the whole thing stops?
                    #        or do we want to discourage this style?
                    self.jobs[key] = {'remote':False, 'exception':e}
                    return None

        # we already knew the value
        return self.memo[key]

    def advance(self):
        new_jobs = {}
        for (k, j) in self.jobs.items():
            if j['remote']:
                v = j['async_res']
                if v.ready() and v.successful():
                    self.memo[k] = v.get()
                else:
                    new_jobs[k] = j
            else:
                # copy things over for a local, failed job
                new_jobs[k] = j
        self.jobs = new_jobs

    def report_status(self, verbose=False, silent=False):
        success = 0
        failed = 0
        waiting = 0
        failures = []
        for (args, job) in self.jobs_iter():
            if not job['remote']:
                failed += 1

                failures.append((args, async_result.metadata))
                if verbose:
                    print "FAILED ON ARGS: " + str(args)
                    print str(async_result.metadata['pyerr'])
            else:
                async_result = job['async_res']

                if async_result.ready():
                    if async_result.successful():
                        success += 1
                    else:
                        failed += 1
                        failures.append((args, async_result.metadata))
                        if verbose:
                            print "FAILED ON ARGS: " + str(args)
                            print str(async_result.metadata['pyerr'])
                else:
                    waiting += 1
        
        out = {'unclaimed_success' : success, 'failed' : failed, 'waiting' : waiting, 'failures' : failures}

        if not silent:
            print "STATS: " + str(success) + " unclaimed successful, " + str(failed) + " failed, " + str(waiting) + " waiting."

        return out

    def jobs_iter(self):
        for (k, j) in self.jobs.items():
            yield (self.args[k], self.jobs[k])

    def iter(self, local_only=False):
        if not local_only:
            self.advance()

        for (k, v) in self.memo.items():
            yield (self.args[k], self.memo[k])
