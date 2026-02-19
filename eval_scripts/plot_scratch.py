from dataset import Dist, QuantityEnum, PixelInterpolation, Direction, DataSet
import logging
import numpy as np
from functions import WindowTypes
import gc

options = {
"plot_range": slice(0, 750),

# "ref_pos": (30, None), # img8
"ref_pos": (30, None), # img9

"ref_threshold": 0.90,
"pixel_interpolation": PixelInterpolation.none,
"dist_func": Dist.Time,
"img_title": "",
"sample_properties": {"d_1": 640,#650,
                      "d_2": 650,
                      "d_film": 0.300, },
"pp_opt": {"window_opt": {"enabled": True,
                          "slope": 0.01, # 0.999, # 0.99
                          # "win_start": 0,
                          "win_width": 70, # 18,#2*32,# 38*2, # 5*15 # 36
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
"plot_opt": {"shift_sam2ref": False,},
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
"cbar_lim": (6.40, 6.80),
}
dataset_path = r"C:\Users\alexj\Data\SemiconductorSamples\MarielenaData\2022_02_14\GaAs_Te 19073"

dataset = DataSet(dataset_path, options)

dataset.select_freq(0.5)
dataset.select_quantity(QuantityEnum.P2P)

dataset.plot_image()

dataset.plt_show()

