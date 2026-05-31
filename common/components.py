from config import *
import json

class ComponentBase:
    def __init__(self):
        self.is_initialized = False
        self._settings_file = None
        self.settings = None

    def __enter__(self):
        self.is_initialized = True
        self.settings = self.loadConfiguration()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.saveConfiguration()
        self.is_initialized = False

        return False

    def saveConfiguration(self, ):
        pass

    def loadConfiguration(self) -> AppSettings:
        config_path = Path(f"configs/{self._settings_file}.json")

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
