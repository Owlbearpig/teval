from dataset import DataSet, Dist, QuantityEnum, PixelInterpolation, WindowTypes
from dataset_eval import DatasetEval, DataSetType
import os

options = {

"plot_range": slice(13, 230),

"ref_pos": (20, None), # img9

"ref_threshold": 0.90,
"pixel_interpolation": PixelInterpolation.none,
"dist_func": Dist.Time,
"img_title": "",

"pp_opt": {"window_opt": {"enabled": True,
                          "slope": 0.05, # 0.999, # 0.99
                          # "win_start": 0,
                          "win_width": 70, # 18,#2*32,# 38*2, # 5*15 # 36
                          "type": WindowTypes.tukey,
                          },
           "filter_opt": {"enabled": False, "f_range": (0.3, 3.0), },
           "remove_dc": True,
           },

"shown_plots": {
    "Window": True,
    "Time domain": True,
    "Spectrum": True,
    "Phase": False,
    "Phase slope": False,
    "Amplitude transmission": False,
    "Absorbance": True,
    "Refractive index": False,
    "Absorption coefficient": False,
    "Conductivity": False,
},
}
"""
HE_A1: 100, 0
HE_A2: 74, 0
HE_C: 90, 0
LE_A: 80, 0
LE_AP: 90, 0
LE_C: 90, 0
"""

meas_dict = {# "HE_A1": (100, 0), #"HE_A2": (74, 0), "HE_C": (90, 0),
             "LE_A": (80, 0), "LE_A-90": (90, 0), "LE_A-95": (95, 0),
            #"LE_AP": (90, 0), "LE_C": (90, 0),
}

for meas_set in meas_dict:
    if "-" in meas_set:
        dir_name = meas_set.split("-")[0]
    else:
        dir_name = meas_set

    if 'nt' in os.name:
        sam_dataset_path = fr"C:\Users\alexj\Data\Pectin_wAg_Nanoparticles\{dir_name}"
    else:
        sam_dataset_path = fr"/home/ftpuser/ftp/Data/Pectin_wAg_Nanoparticles/{dir_name}"

    dataset = DataSet(sam_dataset_path, options)
    pos = meas_dict[meas_set]
    dataset.plot_point(pos, label=meas_set, err_bar_limits=(70, 95))

dataset.plt_show()
