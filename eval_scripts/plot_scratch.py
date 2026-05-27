from dataset import Dist, QuantityEnum, PixelInterpolation, Direction, DataSet
import os
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from functions import WindowTypes
import gc

from teval import ClimateQuantity

figure_dir = Path(r"/home/alex/MEGA/AG/Projects/Conductivity/ErrorAnalysis/ManuscriptFigures")

options = {
"plot_range": slice(0, 750),

"ref_pos": (10, None),

# "ref_threshold": 0.90,
"pixel_interpolation": PixelInterpolation.none,
"dist_func": Dist.Position,
"img_title": "",
"save_plots": False,
"save_plots_settings": {"path": figure_dir, "filetype": "png", "suffix": "", "dpi": 300, "bbox_inches": "tight",
                        "pad_inches": 0, "set_size_inches": (19, 9)},
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
"plot_opt": {"shift_sam2ref": False, "stability_plot_rel_change": True, "disable_legend": [],
             "temp_sensor_idx": None, "subtract_mean": True,
             #"redp_sensor_labels": {"Redp idx 0": r"$\theta_{system\,fiber}$", "Redp idx 1": r"$\theta_{delay\,line}$",
             #                       "Redp idx 2": r"$\theta_{table\,fiber}$", "Redp idx 3": r"$\theta_{box}$"},
             },
"enable_q_eval": False,
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
if "nt" in os.name:
    # dataset_path = r"C:\Users\alexj\Data\monitoring_pulse_mod\set1"
    # dataset_path = r"C:\Users\alexj\Data\monitoring_pulse_mod\test\set4_test"
    # dataset_path = r"C:\Users\alexj\Data\monitoring_pulse_mod\set4_subset"
    dataset_path = r"C:\Users\alexj\Data\monitoring_pulse_mod\set5_subset"
else:
    # dataset_path = r"/home/ftpuser/ftp/Data/Stability/monitoring_pulse_mod/set4_subset"
    # dataset_path = r"/home/ftpuser/ftp/Data/Stability/monitoring_pulse_mod/set4"
    # dataset_path = r"/home/ftpuser/ftp/Data/Stability/monitoring_pulse_mod/set4_subset"
    # dataset_path = r"/home/ftpuser/ftp/Data/Stability/TeraK15_comparison/set1"
    # dataset_path = r"/home/ftpuser/ftp/Data/Stability/TeraK15_comparison/set2_subset"
    # dataset_path = r"/home/ftpuser/ftp/Data/Stability/monitoring_pulse_mod/set5_subset"
    # dataset_path = r"/home/ftpuser/ftp/Data/Stability/18052026_systemcover"
    # dataset_path = r"/home/ftpuser/ftp/Data/Stability/18052026_terasaat"
    # dataset_path = r"/home/ftpuser/ftp/Data/Stability/19052026_terasaat_largerwindow"
    # dataset_path = r"/home/ftpuser/ftp/Data/Stability/20052026_terasaat_dlineheating"
    # dataset_path = r"/home/ftpuser/ftp/Data/Stability/20052026_terasaat_dlineheating_subset"
    # dataset_path = r"/home/ftpuser/ftp/Data/Stability/21052026_terasaat_dlineheating_nosystemlid"
    # dataset_path = r"/home/ftpuser/ftp/Data/Stability/26052026_terasaat_dlineheating_sensor1_at_ring"
    # dataset_path = r"/media/storage/ArchivedData/Stability/31-10-2024_L1/air"
    dataset_path = r"/home/ftpuser/ftp/Data/Stability/27052026_lab1_no_tlog"


dataset = DataSet(dataset_path, options)

dataset.select_freq(0.5)
# dataset.select_quantity(QuantityEnum.TransmissionPhase)
# dataset.select_quantity(QuantityEnum.P2P)
# dataset.select_quantity(QuantityEnum.TransmissionAmp)
# dataset.select_quantity(QuantityEnum.Phase)
# dataset.plot_meas((50, 20))
# dataset.plot_meas(timestamp="2026-04-22T20-53-54.638573")
# dataset.plot_image()

# dataset.system_stability_diff_plot()
# dataset.plot_system_stability(climate_log_file=r"2026-04-16 14-12-08_log_pitaya_subset_0start.txt") # set5
# dataset.plot_system_stability(climate_log_file=r"2026-05-18 11-13-04_log_pitaya_start0.txt") # systemcover
# dataset.plot_system_stability(climate_log_file=r"2026-05-18 11-13-04_log_pitaya_terasaat_start0.txt") # terasaat
# dataset.plot_system_stability(climate_log_file=r"2026-05-18 11-13-04_log_pitaya_terasaat_widerrange_start0.txt") # terasaat
# dataset.plot_system_stability(climate_log_file=r"2026-05-20 11-32-11_log_pitaya_start0.txt") # dline heating
# dataset.plot_system_stability(climate_log_file=r"2026-05-21 11-41-56_log_pitaya.txt") # dline heating no lid
# dataset.plot_system_stability(climate_log_file=r"2026-05-22 10-10-38_log_pitaya.txt") # dline heating no lid sensor1 at ring
dataset.plot_system_stability() # lab1 data


# dataset.plot_climate(log_file="2026-04-17 00-00-00_log_subset.txt", quantity=ClimateQuantity.Humidity)

# dataset.plot_system_stability(climate_log_file="2026-04-22 11-58-34_log_pitaya.txt")
# dataset.options["pp_opt"]["window_opt"]["enabled"] = False
# dataset.plot_ref()
# dataset.plot_ref(ref_idx=1250)

dataset.plt_show()

