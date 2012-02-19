# FIXME: Lay this out better in the future

from IPython.parallel import *

# memoizer

# Handles None return values w/o storing, so that asynchronous things
# that aren't ready yet can be nicely memoized
#
# FIXME: handle keyword arguments too
class Memoize:

    def __init__(self, f):
        self.f = f
        self.memo = {}
        self.args = {}

    def __call__(self, *args):
        key = str(args)
        self.args[key] = args
        if not key in self.memo:
            val = self.f(*args)
            if val is not None:
                self.memo[key] = val
            return val
        return self.memo[key]

    def iter(self):
        for (k, v) in self.memo.items():
            yield (self.args[k], self.memo[k])

class AsyncMemoize:
    def __init__(self, f, view):
        self.f = f
        self.view = view
        self.memo = {}
        self.args = {}
        self.jobs = {}

    def __call__(self, *args):
        key = str(args)
        self.args[key] = args

        if not key in self.memo:
            # try to apply it
            async_res = self.view.apply_async(self.f, *args)

            # if it's done, store and return
            if async_res.ready() and async_res.successful():
                val = async_res.get()
                self.memo[key] = val
                return val
            else:
                # else store it in jobs and return None
                # NOTE: this keeps all failed jobs around all the time
                self.jobs[key] = async_res
                return None

        # we already knew the value
        return self.memo[key]

    def advance(self):
        new_jobs = {}
        for (k, v) in self.jobs.items():
            if v.ready() and v.successful():
                # NOTE: this keeps all failed jobs around all the time
                self.memo[k] = v.get()
            else:
                new_jobs[k] = v
        self.jobs = new_jobs

    def report_status(self, verbose=False, silent=False):
        success = 0
        failed = 0
        waiting = 0
        failures = []
        for (args, async_result) in self.jobs_iter():
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
        out = {'success' : success, 'failed' : failed, 'waiting' : waiting, 'failures' : failures}
        if not silent:
            print "STATS: " + str(success) + " successful, " + str(failed) + " failed, " + str(waiting) + " waiting."
        return out

    def jobs_iter(self):
        for (k, j) in self.jobs.items():
            yield (self.args[k], self.jobs[k])

    def iter(self, local_only=False):
        if not local_only:
            self.advance()

        for (k, v) in self.memo.items():
            yield (self.args[k], self.memo[k])
