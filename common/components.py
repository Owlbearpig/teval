
class ComponentBase:
    _settings_file = None

    def __init__(self):
        self.is_initialized = False

    def __enter__(self):
        self.is_initialized = True

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.is_initialized = False

        return False

