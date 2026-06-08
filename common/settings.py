from pathlib import Path
import json
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
        self.app_settings = self.load_configuration()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_configuration()

        return False

    def __enter__(self):
        print("enter")
        return self

    def save_configuration(self, ):
        print("saving settings")

    def load_configuration(self) -> AppSettings:
        config_path = Path(f"configs/{self._settings_file}.json")
        print(f"loading {config_path}")

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

