import copy

class IterativeJob:
    """
    An iterative job skeleton for use with dispatchers like
    multiprocessing and picloud.

    Designed to make it easy to run and monitor iterative jobs
    (e.g. Markov chain simulations) that go for minutes to hours ---
    or exceptionally stable jobs that go for days.
    """

    """
    Initialize the state of the job
    """
    def __init__(self, summarize_every_N = 1, max_seconds = 100, max_iter = 1000, render_every_N = 0)
        self.state = None

        self.summarize_every_N = summarize_every_N
        self.render_every_N = render_every_N

        self.max_seconds = max_seconds
        self.max_iter = max_iter

        self.started = False
        self.finished = False

        self.cur_iter = None

        self.final_state = None

    """
    Construct a state dictionary from a parameter dictionary

    Overridden by children.
    """
    def do_initialize(self, params_dict):
        return {}

    """
    Return a new state from the current state

    Overriden by children.
    """
    def do_state_update(self, state):
        pass

    """
    Generate a visualization of the current state and return it as a pickled image.

    Overridden by children.
    """
    def do_render(self, state):
        return None

    """
    Check if the job needs to be terminated, in some state-dependent way.

    Overridden by children.
    """
    def do_termination_check(self):
        return False

    # run an iteration, or stop
    def iterate(self):
        if self.finished:
            return

        if not self.started:
            # start the job FIXME
            pass

        
    # check if done
    def is_completed(self):
        if self.do_termination_check():
            return True

        if self.cur_iter is not None and self.cur_iter >= self.max_iter:
            return True

        
        
    # reports summaries that can be used to monitor progress
    def get_summary_json(self):
        return None

    # used for checkpointing
    def get_state_copy(self):
        return copy.deepcopy(self.state)

    # used for checkpointing
    def set_state(self, state):
        self.state = copy.deepcopy(state)

class CloudIterativeJobRunner:
    """
    Makes it easy to run an iterative job on picloud, as a function
    from initial parameters to a {"final_state": ..., "summaries":
    ...} dictionary.

    Memoized via a picloud cache.

    Currently requires iterative deepening, so you can get a sense of
    what's going on for each job, stress reliability, etc.

    Deferred: inspecting results while jobs are running; checkpointing
    and recovery
    """
    
    def __init__(self, iterative_job):
        self.ijob = iterative_job

    def run(self, params):
        # runs the job and returns the result object
        # FIXME
        reurn None
