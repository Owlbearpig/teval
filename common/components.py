import logging
from traitlets import HasTraits, Unicode, Bool, Int, Instance
import traitlets
from PySide6 import QtWidgets
from functools import wraps
import matplotlib as mpl

def is_component_trait(x):
    return isinstance(x, Instance) and issubclass(x.klass, ComponentBase)

def _dumb_list_of_actions(inst):
    for name in dir(inst):
        try:
            attr = getattr(inst, name, None)
            if not attr._isAction:
                continue

            yield name, attr
        except AttributeError:
            pass
        except traitlets.TraitError:
            pass

def action(name=None, help=None, check_init=False, **kwargs):
    if name is None:
        name = ''
    if help is None:
        help = ''

    kwargs['name'] = name
    kwargs['help'] = help
    kwargs['check_init'] = check_init

    def action_impl(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs_fn):
            if check_init:
                dataset = getattr(self, "dataset", None)
                if dataset is not None and not getattr(dataset, "is_initialized", False):
                    logging.warning(f"Action '{name or method.__name__}' blocked: Dataset is not initialized.")
                    return None

            rc_params = kwargs.get("rc_params", None)
            rc_params_dict = rc_params() if callable(rc_params) else rc_params

            if rc_params_dict:
                with mpl.rc_context(rc_params_dict):
                    return method(self, *args, **kwargs_fn)
            else:
                return method(self, *args, **kwargs_fn)

        wrapper._isAction = True
        wrapper.metadata = kwargs
        wrapper.help = help
        return wrapper

    return action_impl


class ComponentBase(HasTraits):

    script_name = None

    def __init__(self, object_name : str = None):

        self.object_name = object_name
        if self.object_name is None:
            self.object_name = type(self).__name__

        self.__actions = []
        for name, memb in _dumb_list_of_actions(self):
            self.__actions.append((name, memb))

    def __enter__(self, *args):
        for name, trait in self.traits().items():
            if is_component_trait(trait):
                trait.get(self).__enter__(*args)

        return self

    def __exit__(self, *args):
        for name, trait in self.traits().items():
            if is_component_trait(trait):
                trait.get(self).__exit__(*args)

        return False

    def toggle_traits(self, active_traits, component_inst, group_filter="", endswith_filter=""):
        ui_widget = getattr(component_inst, "_ui_control_widget", None)
        if ui_widget and hasattr(ui_widget, "param_widgets"):
            for trait_name, widgets in ui_widget.param_widgets.items():
                if trait_name.endswith(endswith_filter) and trait_name in component_inst.traits(group=group_filter):
                    is_visible = (trait_name in active_traits)
                    widgets[0].setVisible(is_visible)

                    if isinstance(widgets[1], QtWidgets.QLayout):
                        for i in range(widgets[1].count()):
                            w = widgets[1].itemAt(i).widget()
                            if w: w.setVisible(is_visible)
                    else:
                        widgets[1].setVisible(is_visible)

    @property
    def actions(self):
        return self.__actions

    @property
    def attributes(self):
        return self.traits()

    def toggle_traits(self, active_traits, group_filter="", endswith_filter=""):
        ui_widget = getattr(self, "_ui_control_widget", None)
        if ui_widget and hasattr(ui_widget, "param_widgets"):
            for trait_name, widgets in ui_widget.param_widgets.items():
                if trait_name.endswith(endswith_filter) and trait_name in self.traits(group=group_filter):
                    is_visible = (trait_name in active_traits)
                    widgets[0].setVisible(is_visible)

                    if isinstance(widgets[1], QtWidgets.QLayout):
                        for i in range(widgets[1].count()):
                            w = widgets[1].itemAt(i).widget()
                            if w: w.setVisible(is_visible)
                    else:
                        widgets[1].setVisible(is_visible)
