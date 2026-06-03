
class ComponentBase:
    script_name = None

    def __init__(self, object_name : str = None):
        self.is_initialized = False

        self.object_name = object_name
        if self.object_name is None:
            self.object_name = type(self).__name__

    def __enter__(self):
        self.is_initialized = True

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.is_initialized = False

        return False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def default_post_init(self):
            super(cls, self).__init__()

        cls.__post_init__ = default_post_init
