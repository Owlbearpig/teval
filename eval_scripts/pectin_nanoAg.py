from dataset import DataSet, Dist, QuantityEnum, PixelInterpolation, WindowTypes
from dataset_eval import DatasetEval, DataSetType
import os

options = {

"plot_range": slice(0, 1000),

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
    "Amplitude transmission": True,
    "Absorbance": False,
    "Refractive index": False,
    "Absorption coefficient": False,
    "Conductivity": True,
},
}

if 'nt' in os.name:
    sam_dataset_path = r"C:\Users\alexj\Data\Pectin_wAg_Nanoparticles\HE_A1"
else:
    sam_dataset_path = r"/home/ftpuser/ftp/Data/Pectin_wAg_Nanoparticles/HE_A1"

dataset = DataSet(sam_dataset_path, options)

dataset.select_freq(0.5)
dataset.select_quantity(QuantityEnum.P2P)

dataset.plot_line()

dataset.plt_show()
