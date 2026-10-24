from enum import Enum
from pathlib import Path
import json
from common.eval_component.eval_result import EvalResult
from common.traits import (ValueRange, Path as TPath, Quantity, MultiPathClass, MultiPathSelection, FitParameter,
                           FitParameterTable)
from dataclasses import asdict
from common.units import Q_
from traitlets import Instance, Bool, Integer, Float, Enum as TEnum, Unicode, List
from common.components import ComponentBase, is_component_trait
from common.default_appsettings import AppSettings, EvalOpt, PpOpt, ImageEvalOpt
from traitlets.traitlets import TraitError

_simple_types = (float, str, list, bool, int)


def _dump_settings(dump_dict, instance):
    if instance is None:
        return
    for k, trait in instance.attributes.items():
        en_save = trait.metadata.get("en_save", True)
        if not en_save:
            continue
        val = trait.get(instance)
        if not is_component_trait(trait):
            if isinstance(val, Enum):
                val = val.name
            elif isinstance(val, Path):
                val = str(val)
            elif isinstance(val, Q_):
                val = val.magnitude
            elif issubclass(trait.__class__, ValueRange):
                if isinstance(val[0], Q_):
                    val = [val[0].magnitude, val[1].magnitude]
                else:
                    val = [val[0], val[1]]
            elif isinstance(val, MultiPathClass):
                dump_dict[k] = [str(p) for p in val.selected_paths]
            elif isinstance(trait, FitParameterTable):
                val = [{**asdict(p), "value": None} for p in val]
            if isinstance(val, _simple_types):
                dump_dict[k] = val
        else:
            if issubclass(trait.klass, EvalResult):
                continue
            dump_dict[k] = {}

            _dump_settings(dump_dict[k], val)


def component_settings(instance):
    # the saved (en_save) trait values of the component and its sub-components, as in the config file
    dump_dict = {}
    _dump_settings(dump_dict, instance)
    return dump_dict


# option groups of the settings whose traits moved into sub-components (sub-tabs)
_split_option_groups = {"eval_opt": EvalOpt, "pp_opt": PpOpt}


def migrate_config(settings_dict, json_dict=None):
    # entries saved at the old place (directly in the option group) are moved into the sub-component, unless the new
    # place already has a value. json_dict: the whole config file, for settings that moved from another component
    if json_dict is not None:
        selector_dict = json_dict.get("DataSet", {}).get("measurement_selector", {})
        if isinstance(selector_dict, dict) and "interpolate_refs" in selector_dict:
            drift_dict = settings_dict.setdefault("pp_opt", {}).setdefault("ref_drift", {})
            drift_dict.setdefault("interpolate_refs", selector_dict["interpolate_refs"])
        image_dict = json_dict.get("QSpaceEval", {}).get("image_eval")
        if isinstance(image_dict, dict):
            moved_names = set(ImageEvalOpt.class_trait_names()) - set(ComponentBase.class_trait_names())
            target = settings_dict.setdefault("eval_opt", {}).setdefault("image_eval", {})
            for name in moved_names.intersection(image_dict):
                target.setdefault(name, image_dict[name])
    for group_name, group_cls in _split_option_groups.items():
        group_dict = settings_dict.get(group_name)
        if not isinstance(group_dict, dict):
            continue
        own_names = set(group_cls.class_trait_names())
        for sub_name, trait in group_cls.class_traits().items():
            if not is_component_trait(trait):
                continue
            moved_names = set(trait.klass.class_trait_names()) - own_names
            moved = {name: group_dict.pop(name) for name in list(group_dict) if name in moved_names}
            if moved:
                sub_dict = group_dict.setdefault(sub_name, {})
                for name, val in moved.items():
                    sub_dict.setdefault(name, val)


class Settings(AppSettings):

    def __init__(self, settings_file: Path | str = "", **kwargs):
        super().__init__(**kwargs)
        self.script_name = self.script_name if self.script_name is not None else ""
        self._settings_file = Path(settings_file).stem if Path(settings_file).is_file() else Path(self.script_name).stem

        self.config_path = Path(f"config/{self._settings_file}.json")

        self.load_configuration(self)

        self._set_component_names()

    def __exit__(self, *args):
        self.save_configuration(self)

        return False

    def _set_component_names(self):
        custom_names = {
            "pp_opt": "Preprocessing",
            "eval_opt": "Evaluation",
            "save_settings": "Save Plot Settings",
            "plot_opt": "Plotting",
            "shown_plots": "Visible Plots"
        }

        for k, trait in self.attributes.items():
            if is_component_trait(trait):
                instance = getattr(self, k)
                if instance is not None:
                    instance.object_name = custom_names.get(k, k.replace('_', ' ').title())


    def save_configuration(self, component_instance):
        if not self.config_path.parent.exists():
            self.config_path.parent.mkdir()

        comp_class_name = component_instance.__class__.__name__

        if self.config_path.exists():
            with open(self.config_path, "r") as fp:
                settings_dict = json.load(fp)
        else:
            settings_dict = {}

        if not self.save_settings_on_exit:
            # only the option itself, otherwise the next start would load the stored "on" and save again
            if component_instance is not self:
                return
            settings_dict.setdefault(comp_class_name, {})["save_settings_on_exit"] = False
            print(f"Saving settings is disabled, only stored that in {self.config_path}")
        else:
            settings_dict[comp_class_name] = component_settings(component_instance)

        with open(self.config_path, 'w') as fp:
            json.dump(settings_dict, fp, indent=4)

    def load_configuration(self, component_instance):
        component_class_name = component_instance.__class__.__name__
        config_path = self.config_path
        print(f"Loading settings from {config_path} for component {component_class_name}")

        if not config_path.exists():
            print(f"No custom config found at {config_path}. Using default values.")
            return

        def set_trait_values(instance, dict_):
            def set_trait(name, val, default=None):
                if instance is None:
                    return
                try:
                    instance.set_trait(name, val)
                except TraitError:
                    if default is None:
                        print(f"Could not load trait {name}. Resetting to default on next load.")
                    else:
                        instance.set_trait(name, default)

            for trait_name, dict_val in dict_.items():
                if instance is None or getattr(instance.__class__, trait_name, None) is None:
                    continue
                actual_type = type(getattr(instance.__class__, trait_name))
                if actual_type == Instance:
                    instance_class = getattr(instance, trait_name)
                    set_trait_values(instance_class, dict_val)
                elif issubclass(actual_type, (Bool, Integer, Float, Unicode)):
                    set_trait(trait_name, dict_val)
                elif issubclass(actual_type, Quantity):
                    value = getattr(instance, trait_name)
                    set_trait(trait_name, Q_(dict_val, value.units))
                elif issubclass(actual_type, TEnum):
                    trait_definition = instance.traits()[trait_name]
                    target_enum_member = None
                    for member in trait_definition.values:
                        if hasattr(member, "name") and member.name == dict_val:
                            target_enum_member = member
                            break
                        elif hasattr(member, "value") and member.value == dict_val:
                            target_enum_member = member
                            break
                        elif member == dict_val:
                            target_enum_member = member
                            break
                    if target_enum_member is not None:
                        set_trait(trait_name, target_enum_member)
                    else:
                        set_trait(trait_name, dict_val)
                elif issubclass(actual_type, TPath):
                    set_trait(trait_name, Path(dict_val), default=Path("."))
                elif issubclass(actual_type, ValueRange):
                    value = getattr(instance, trait_name)
                    if isinstance(value[0], Q_):
                        unit = value[0].units
                        value = [Q_(dict_val[0], unit), Q_(dict_val[1], unit)]
                    else:
                        value = dict_val
                    set_trait(trait_name, value)
                elif issubclass(actual_type, List):
                    set_trait(trait_name, dict_val)
                elif issubclass(actual_type, MultiPathSelection):
                    value = getattr(instance, trait_name)
                    value.selected_paths = [Path(p) for p in dict_val]
                elif issubclass(actual_type, FitParameterTable):
                    try:
                        set_trait(trait_name, tuple(FitParameter(**p) for p in dict_val))
                    except TypeError:
                        print(f"Could not load trait {trait_name}. Resetting to default on next load.")

        with open(config_path, "r") as f:
            json_dict = json.load(f)
            if component_class_name in json_dict:
                if isinstance(component_instance, AppSettings):
                    migrate_config(json_dict[component_class_name], json_dict)
                set_trait_values(component_instance, json_dict[component_class_name])
            else:
                print(f"No custom config found for component {component_class_name}. Using default values.")