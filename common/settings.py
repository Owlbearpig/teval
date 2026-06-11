from enum import Enum
from pathlib import Path
import json

from common.traits import ValueRange, Path as TPath, Quantity
from common.units import Q_
from traitlets import Instance, Tuple, List, Bool, Integer, Float, Enum as TEnum
from common.components import is_component_trait
from common.default_appsettings import AppSettings, PpOpt
from common.default_appsettings import WindowTypes, PixelInterpolation, Dist, LogLevel

class Settings(AppSettings):

    script_name = None

    def __init__(self, settings_file: Path = None):
        super().__init__()

        if self.script_name is None:
            self._settings_file = Path(settings_file).stem
        else:
            self._settings_file = Path(self.script_name).stem
        self.config_path = Path(f"config/{self._settings_file}.json")

        self.load_configuration()

        self._set_component_names()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_configuration()

        return False

    def _set_component_names(self):
        custom_names = {
            "pp_opt": "Preprocessing",
            "eval_opt": "Evaluation",
            "sample_properties": "Sample Properties",
            "save_plots_settings": "Save Plot Settings",
            "plot_opt": "Plotting",
            "shown_plots": "Visible Plots"
        }

        for k, trait in self.attributes.items():
            if is_component_trait(trait):
                instance = getattr(self, k)
                if instance is not None:
                    friendly_name = custom_names.get(k, k.replace('_', ' ').title())
                    instance.object_name = friendly_name


    def save_configuration(self):
        if not self.config_path.parent.exists():
            self.config_path.parent.mkdir()

        settings_dict = {}
        def make_dump_dict(dump_dict, instance):
            for k, trait in instance.attributes.items():
                val = trait.get(instance)
                if not is_component_trait(trait):
                    if isinstance(val, Enum):
                        val = val.name
                    elif isinstance(val, Path):
                        val = str(val)
                    elif issubclass(trait.__class__, ValueRange):
                        if isinstance(val[0], Q_):
                            val = [val[0].magnitude, val[1].magnitude]
                    dump_dict[k] = val
                else:
                    dump_dict[k] = {}

                    make_dump_dict(dump_dict[k], val)

        make_dump_dict(settings_dict, self)

        with open(self.config_path, 'w') as fp:
            json.dump(settings_dict, fp, indent=4)

    def load_configuration(self):
        config_path = self.config_path
        print(f"Loading settings from {config_path}")

        if self._settings_file is None:
            print(f"No config file path set. Loading global defaults.")
            return

        if not config_path.exists():
            print(f"No custom config found at {config_path}. Loading global defaults.")
            return

        def parse_dict(instance, dict_):
            for trait_name, dict_val in dict_.items():
                actual_type = type(getattr(instance.__class__, trait_name))
                if actual_type == Instance:
                    instance_class = getattr(instance, trait_name)
                    parse_dict(instance_class, dict_val)
                elif issubclass(actual_type, (Bool, Integer, Float)):
                    instance.set_trait(trait_name, dict_val)
                elif issubclass(actual_type, TEnum):
                    enum_attr = getattr(instance, trait_name)
                    instance.set_trait(trait_name, enum_attr)
                elif issubclass(actual_type, TPath):
                    instance.set_trait(trait_name, Path(dict_val))
                elif issubclass(actual_type, ValueRange):
                    value = getattr(instance, trait_name)
                    if isinstance(value[0], Q_):
                        unit = value[0].units
                        value = [dict_val[0] * unit, dict_val[1] * unit]
                    instance.set_trait(trait_name, value)

        with open(config_path, "r") as f:
            json_dict = json.load(f)
            parse_dict(self, json_dict)
