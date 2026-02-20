from dataset import DataSet, Dist, QuantityEnum, PixelInterpolation, WindowTypes
from dataset_eval import DatasetEval, DataSetType
import os
import numpy as np

options = {

"plot_range": slice(10, 230),

"ref_pos": (0, None),
"fix_ref": 0,
"ref_threshold": 0.90,
"pixel_interpolation": PixelInterpolation.none,
"dist_func": Dist.Time,
"img_title": "",
"save_plots": True,
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
    "Window": False,
    "Time domain": True,
    "Spectrum": False,
    "Phase": False,
    "Phase slope": False,
    "Amplitude transmission": False,
    "Absorbance": True,
    "Refractive index": False,
    "Absorption coefficient": False,
    "Conductivity": False,
},
    # "cbar_lim": (-1.9, -1.4), # graphene phase at 0.5 THz
    "cbar_lim": (1.7, 2.05),
    # "cbar_lim": (0, 30),
}

sam_dataset_path = fr"/home/ftpuser/ftp/Data/THzConductivity/MarielenaData/2022_02_14/GaAs_Te 19073"
dataset = DataSet(sam_dataset_path, options)
dataset.select_freq(0.5)
dataset.select_quantity(QuantityEnum.Phase)
# dataset.select_quantity(QuantityEnum.P2P)

dataset.plot_image()

dataset.plt_show()
