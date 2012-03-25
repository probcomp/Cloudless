class Longjob:
    # initializes the state of the job
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs

    # run an iteration, either returning None or a value
    # once the job is complete
    def iterate(self):
        pass

    # reports summaries that can be used to monitor progress
    def get_summary(self):
        pass

    # used for checkpointing
    def get_state(self):
        pass

    # used for checkpointing
    def set_state(self, obj):
        pass
