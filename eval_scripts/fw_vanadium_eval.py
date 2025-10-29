from dataset import Dist, QuantityEnum, PixelInterpolation, Direction
import logging
import numpy as np
from dataset_eval import DatasetEval
from functions import WindowTypes
import gc

options = {

# "cbar_lim": (0.975, 0.982),

# 1.5 THz
# "cbar_lim": (0.65, 0.70), # img8 1.5 THz
# "cbar_lim": (0.60, 0.68), # img9 1.5 THz
#phase
# "cbar_lim": (0.65, 0.70), # img8 1.5 THz
# "cbar_lim": (0.330, 0.390), # img12

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
"cbar_lim": (0.530, 0.570),
}
sam_dataset_path = r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img15"
sub_dataset_path = r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img15"

#sam_dataset_path = r"C:\Users\alexj\Data\Furtwangen\Vanadium Oxide\img15"
#sub_dataset_path = r"C:\Users\alexj\Data\Furtwangen\Vanadium Oxide\img15"

dataset_eval = DatasetEval(sam_dataset_path, sub_dataset_path, options)

dataset_eval.select_freq((0.50, 1.00))
# dataset_eval.select_freq(0.5)
dataset_eval.select_quantity(QuantityEnum.P2P)

sub_pnt = options["eval_opt"]["sub_pnt"]
sam_pnt = (75, 5)
# dataset_eval.plot_point(sub_pnt, label="Substrate")
# dataset_eval.plot_meas_phi_diff(sub_pnt, sam_pnt, label="Substrate - sample")

# dataset_eval.plot_system_stability()

dataset_eval.plot_image()
# dataset_eval.plot_refs()
# dataset_eval.plot_line(line_coords=10.0, direction=Direction.Horizontal)

dataset_eval.plot_point(sub_pnt)
dataset_eval.eval_point_n_fit(sam_pnt)
# dataset_eval.ref_difference_plot()

"""
x_coords = np.arange(23.5, 50.0, 0.5)
y_coords = np.arange(-4.5, 12.5, 0.5)
for x in x_coords:
    for y in y_coords:
        dataset_eval = DatasetEval(sam_dataset_path, sub_dataset_path, options)
        pnt = (x, y)
        pnt = (30, 10)
        dataset_eval.options["eval_opt"]["sub_pnt"] = pnt
        dataset_eval.eval_point(pnt)
        dataset_eval.plt_show()
        del dataset_eval
        gc.collect()
"""
dataset_eval.plt_show()

