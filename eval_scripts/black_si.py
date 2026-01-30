from dataset import DataSet, Dist, QuantityEnum, PixelInterpolation, WindowTypes
from dataset_eval import DatasetEval, DataSetType
import os

options = {

"plot_range": slice(13, 230),

"ref_pos": (0, None),
"fix_ref": 0,
"ref_threshold": 0.90,
"pixel_interpolation": PixelInterpolation.none,
"dist_func": Dist.Time,
"img_title": "",
"sample_properties": {"d": 534, "layers": 1, "default_values": True},
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
    "Absorbance": False,
    "Refractive index": False,
    "Absorption coefficient": False,
    "Conductivity": False,
},
}

if 'nt' in os.name:
    sam_dataset_path = fr"C:\Users\alexj\Data\Black_Si\NoPattern_Linescan2_NoNitrogen"
else:
    sam_dataset_path = fr"/home/ftpuser/ftp/Data/Black_Si/NoPattern_Linescan2_NoNitrogen"

dataset = DataSet(sam_dataset_path, options)

dataset.select_quantity(QuantityEnum.AbsorptionCoe)
dataset.plot_line(line_coords=16, label="1.0 THz", fig_num_="x-slice_alpha", y_label=r"Refractive index")
dataset.select_freq(0.50)
dataset.plot_line(line_coords=16, label="0.5 THz", fig_num_="x-slice_alpha", y_label=r"Refractive index")
# dataset.plot_point((10, 16))
# dataset.plot_point((40, 16))
# dataset.plot_point((50, 16))
# dataset.plot_point((70, 16))

dataset.plt_show()
