from traitlets import HasTraits, Unicode, Bool, Int, Instance
import traitlets

def is_component_trait(x):
    return isinstance(x, Instance) and issubclass(x.klass, ComponentBase)

def _dumb_list_of_actions(inst):
    for name in dir(inst):
        try:
            attr = getattr(inst, name, None)  # same as inst.name
            if not attr._isAction:
                continue

            yield name, attr
        except AttributeError:
            pass
        except traitlets.TraitError:
            pass

def action(name=None, help=None, **kwargs):
    if name is None:
        name = ''
    if help is None:
        help = ''

    kwargs['name'] = name
    kwargs['help'] = help

    def action_impl(method):
        method._isAction = True
        method.metadata = kwargs
        method.help = help
        return method

    return action_impl


class ComponentBase(HasTraits):

    def __init__(self, object_name : str = None):
        self.is_initialized = False

        self.object_name = object_name
        if self.object_name is None:
            self.object_name = type(self).__name__

        self.__actions = []
        for name, memb in _dumb_list_of_actions(self):
            self.__actions.append((name, memb))

    def __enter__(self):
        self.is_initialized = True

        return self

    def __exit__(self, *args):
        self.is_initialized = False

        for name, trait in self.traits().items():
            if is_component_trait(trait):
                trait.get(self).__exit__(*args)

        return False

    @property
    def actions(self):
        return self.__actions

    @property
    def attributes(self):
        return self.traits()