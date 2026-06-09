from enum import Enum
from pathlib import Path
import json

from traitlets import Instance, Tuple, List, Bool
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
        self.config_path = Path(f"configs/{self._settings_file}.json")

        self.load_configuration()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_configuration()

        return False

    def __enter__(self):
        print("enter")
        return self

    def save_configuration(self, ):
        settings_dict = {}
        def make_dump_dict(dump_dict, instance):
            for k, trait in instance.attributes.items():
                val = trait.get(instance)
                if not is_component_trait(trait):
                    if isinstance(val, Enum):
                        val = val.name
                    if isinstance(val, Path):
                        val = str(val)
                    dump_dict[k] = val
                else:
                    dump_dict[k] = {}

                    make_dump_dict(dump_dict[k], val)

        make_dump_dict(settings_dict, self)

        with open(self.config_path, 'w') as fp:
            json.dump(settings_dict, fp, indent=4)

        # json.dumps(settings_dict, self._settings_file, indent=4)
        # print(settings_dict)

        for k, trait in self.attributes.items():
            if not type(trait) is Instance:
                settings_dict[k] = trait.get(self)
            else:
                pass
            # trait.get(self)
            #print(k, trait.get(self), trait, type(trait) is Bool)
            #print(trait, isinstance(trait, Instance), is_component_trait(trait), isinstance(trait, List))
        #print(self.traits())
        #print("saving settings")

    # TODO set traits?
    def load_configuration(self) -> AppSettings:
        config_path = self.config_path
        print(f"loading {config_path}")

        getattr(self, "pp_opt").set_trait("filter_enabled", True)

        if self._settings_file is None:
            print(f"No config file path set. Loading global defaults.")
            return AppSettings()

        if not config_path.exists():
            print(f"No custom config found for {self._settings_file}. Loading global defaults.")
            return AppSettings()

        with open(config_path, "r") as f:
            settings_dict = json.load(f)

        win_data = settings_dict["pp_opt"]["window_opt"]
        window_obj = WindowOpt(**{**win_data, "type": WindowTypes(win_data["type"])})

        pp_data = settings_dict["pp_opt"]
        pp_obj = PpOpt(**{**pp_data, "window_opt": window_obj})

        return AppSettings(
            log_level=LogLevel(settings_dict["log_level"]),
            pixel_interpolation=PixelInterpolation(settings_dict["pixel_interpolation"]),
            dist_func=Dist(settings_dict["dist_func"]),
            pp_opt=pp_obj,
            **{k: v for k, v in settings_dict.items() if
               k not in ["log_level", "pixel_interpolation", "dist_func", "pp_opt"]}
        )

