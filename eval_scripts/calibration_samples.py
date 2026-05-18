from dataset import Dist, QuantityEnum, PixelInterpolation, Direction, DataSet, ClimateQuantity
import os
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from functions import WindowTypes
import gc


figure_dir = Path(r"/home/alex/MEGA/AG/Projects/Conductivity/Calibration test samples - Andreone")

options = {
"ref_pos": (10, None),

# "ref_threshold": 0.90,
"pixel_interpolation": PixelInterpolation.none,
"dist_func": Dist.Position,
"img_title": "",
"save_plots": False,
"save_plots_settings": {"path": figure_dir, "filetype": "png", "suffix": "", "dpi": 300, "bbox_inches": "tight",
                        "pad_inches": 0, "set_size_inches": (19, 9)},
"sample_properties": {"d": 534, "fp_spacing": 12.0},
"enable_q_eval": True,
"pp_opt": {"window_opt": {"enabled": False,
                          # "slope": 0.01, # 0.999, # 0.99
                          #"win_start": 27, # 11,
                          "win_width": 11, # 18,#2*32,# 38*2, # 5*15 # 36
                          "type": WindowTypes.tukey,
                          },
           "filter_opt": {"enabled": False, "f_range": (0.3, 3.0), },
           "remove_dc": True,
           },
"eval_opt": {"shift_sub": 0, # ref <-> sam pulse shift in fs
             "shift_film": 0,
             "sub_pnt": (32, 5), #(35, 5) # (30, 5) # (32, 5)
             "fit_range_film": (0.65, 3.2),
             "fit_range_sub": (0.5, 3.5), # (0.10, 3.0)
             "nfp": 2, # number of fp pulses contained in window ("inf" or 0=main pulse only, 1, ..., N),
             "area_fit": False,
             "sub_bounds": [(3.05, 3.14), (0.000, 0.0165)],
             "film_bounds": [(1, 25), (0, 25)],
             },
"sim_opt": {"enabled": True,
            "n_sub": 3.05 + 0.005j,
            "shift_sim": 0,
            "nfp_sim": 0,
},
"plot_opt": {"shift_sam2ref": False, "stability_plot_rel_change": True, "disable_legend": [],
             "temp_sensor_idx": None, "subtract_mean": True, "plot_range": (0.25, 3.00),
             },
"shown_plots": {
    "Window": True,
    "Time domain": True,
    "Spectrum": True,
    "Phase": True,
    "Phase slope": False,
    "Amplitude transmission": False,
    "Absorbance": False,
    "Refractive index": False,
    "Absorption coefficient": False,
    "Conductivity": True,
    "Absorption coefficient optimum": True,
},
}

# pulse monitoring mod
if "nt" in os.name:
    dataset_path = r"C:\Users\alexj\Data\CalibrationSamples\Graphene"
else:
    dataset_path = r"/home/ftpuser/ftp/Data/CalibrationSamples/Graphene"

dataset = DataSet(dataset_path, options)

dataset.select_freq(2.0)
dataset.select_quantity(QuantityEnum.TransmissionAmp)
dataset.plot_line(line_coords=-8)
# dataset.select_quantity(QuantityEnum.TransmissionPhase)
# dataset.select_quantity(QuantityEnum.P2P)

# dataset.select_quantity(QuantityEnum.Phase)
dataset.plot_meas((105, -8))
# dataset.plot_meas(timestamp="2026-04-22T20-53-54.638573")
# dataset.plot_image()

# dataset.system_stability_diff_plot()
# dataset.plot_system_stability(climate_log_file=r"2026-04-16 14-12-08_log_pitaya_subset_0start.txt") # set5
# dataset.plot_climate(log_file="2026-04-17 00-00-00_log_subset.txt", quantity=ClimateQuantity.Humidity)

# dataset.plot_system_stability(climate_log_file="2026-04-22 11-58-34_log_pitaya.txt")
# dataset.options["pp_opt"]["window_opt"]["enabled"] = False
# dataset.plot_ref()
# dataset.plot_ref(ref_idx=1250)

dataset.plt_show()

