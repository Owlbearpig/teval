from dataset import Dist, QuantityEnum, PixelInterpolation, Direction, DataSet
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from functions import WindowTypes
import gc

options = {
"plot_range": slice(0, 750),

"ref_pos": (10, None),

# "ref_threshold": 0.90,
"pixel_interpolation": PixelInterpolation.none,
"dist_func": Dist.Position,
"img_title": "",
"sample_properties": {"d_1": 640,#650,
                      "d_2": 650,
                      "d_film": 0.300, },
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
"plot_opt": {"shift_sam2ref": False, "stability_plot_rel_change": False, "disable_legend": ["Reference delay"],
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
},
# "cbar_lim": (-1.5, -1.0), # phase 0.5 THz
# "cbar_lim": (-2.40, -2.0), # phase 2.0 THz
# "cbar_lim": (-2.00, -1.7), # phase 2.0 THz
# "cbar_lim": (6.8, 7.1),
# "cbar_lim": (5.75, 5.95), # img0
# "cbar_lim": (4.95, 5.15),
# "cbar_lim": (0.10, 0.12),
# "cbar_lim": (-2.0, -1.5),
# "cbar_lim": (-2.10, -1.50),
# "cbar_lim": (0.090, 0.120),
# "cbar_lim": (-1.90, -1.45),
#"cbar_lim": (4.3, 4.7),
# "cbar_lim": (0.70, 0.730),
# "cbar_lim": (2.85, 3.05),
# "cbar_lim": (7.4),
}
# dataset_path = r"C:\Users\alexj\Data\SemiconductorSamples\MarielenaData\2022_02_14\GaAs_Te 19073"

# paper + graphene
# dataset_path = r"C:\Users\alexj\Data\IPHT2\Filter_coated/img0"
# dataset_path = r"C:\Users\alexj\Data\IPHT2\Filter_coated/img3"
# dataset_path = r"/home/ftpuser/ftp/Data/IPHT2/Filter_coated/img3"
# dataset_path = r"C:\Users\alexj\Data\IPHT2\Filter_uncoated\img0"

# quartz + Ag
# dataset_path = r"C:\Users\alexj\Data\HHI_Aachen\remeasure_02_09_2024\sample3\img2"

# smartT + ITO
# dataset_path = r"C:\Users\alexj\Data\IPHT\uncoated\s4"
# dataset_path = r"C:\Users\alexj\Data\IPHT\coated\s4"

# Sapphire + Vanadium
# dataset_path = r"C:\Users\alexj\Data\Furtwangen\Vanadium Oxide\img8"
# dataset_path = r"/media/storage/ArchivedData/Conductivity/Furtwangen/Vanadium Oxide/img5"

# pulse monitoring mod
# dataset_path = r"C:\Users\alexj\Data\monitoring_pulse_mod\set1"
# dataset_path = r"/home/ftpuser/ftp/Data/Stability/monitoring_pulse_mod/set4_subset"
dataset_path = r"/home/ftpuser/ftp/Data/Stability/monitoring_pulse_mod/set4"

dataset = DataSet(dataset_path, options)

dataset.select_freq(0.75)
# dataset.select_quantity(QuantityEnum.TransmissionPhase)
# dataset.select_quantity(QuantityEnum.P2P)
# dataset.select_quantity(QuantityEnum.TransmissionAmp)
# dataset.select_quantity(QuantityEnum.Phase)
# dataset.plot_meas((50, 20))
# dataset.plot_image()

# dataset.system_stability_diff_plot()
# dataset.plot_system_stability(climate_log_file=r"2026-04-07 14-08-28_log_pitaya_subset")
dataset.plot_system_stability(climate_log_file=r"2026-04-07 14-08-28_log_pitaya")
# dataset.options["pp_opt"]["window_opt"]["enabled"] = False
# dataset.plot_ref()
# dataset.plot_ref(ref_idx=1250)

dataset.plt_show()

