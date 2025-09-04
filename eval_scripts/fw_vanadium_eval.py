from dataset import Dist, QuantityEnum, PixelInterpolation, Direction
import logging
from dataset_eval import DatasetEval

options = {
# TODO setting this sucks. Fix runtime / use cache for t calc and set lims based on area min max
# "cbar_lim": (0.975, 0.982),

# 1.5 THz
# "cbar_lim": (0.65, 0.70), # img8 1.5 THz
# "cbar_lim": (0.60, 0.68), # img9 1.5 THz
#phase
# "cbar_lim": (0.65, 0.70), # img8 1.5 THz

"plot_range": slice(30, 1000),

# "ref_pos": (30, None), # img8
"ref_pos": (30, None), # img9

"ref_threshold": 0.90,
"pixel_interpolation": PixelInterpolation.none,
"dist_func": Dist.Time,
"img_title": "",
"sample_properties": {"d": 650,
                      "d_film": 0.350,
                      },
"pp_opt": {"window_opt": {"enabled": True,
                          "slope": 0.99,
                          # "win_start": 0, # TODO allow negative values (wrap around)
                          "win_width": 15,
                          },
           },
"eval_opt": {"dt": -0, # dt in fs
             "sub_pnt": (35, 0),
             },
"shown_plots": {
    "Window": True,
    "Time domain": True,
    "Spectrum": True,
    "Phase": True,
    "Phase slope": True,
    "Amplitude transmission": False,
    "Absorbance": False,
    "Refractive index": False,
    "Absorption coefficient": False,
    "Conductivity": True,
},
}
sam_dataset_path = r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img12"
sub_dataset_path = r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img12"

dataset_eval = DatasetEval(sam_dataset_path, sub_dataset_path, options)

dataset_eval.select_freq((1.40, 1.5))
# dataset_eval.select_quantity(QuantityEnum.ZeroCrossing)
dataset_eval.select_quantity(QuantityEnum.Power)
# dataset_eval.select_quantity(QuantityEnum.Phase)

# dataset_eval.plot_system_stability()

dataset_eval.plot_image()
#dataset_eval.plot_refs()
# dataset_eval.plot_line(line_coords=10.0, direction=Direction.Horizontal)

# img 8
#dataset_eval.average_area((45, 5), (57, 19), label="Sub. 1")
#dataset_eval.average_area((73, 5), (80, 19), label="2")
#dataset_eval.plot_point((50, 10), apply_window=False)

# img 9
#dataset_eval.average_area((45, 5), (57, 19), label="Sub. 2")
#dataset_eval.average_area((75, 5), (80, 19), label="9")
# dataset_eval.plot_point((72, 10))
# dataset_eval.plot_point((82, 10))
dataset_eval.plot_point((80, 10))

# dataset_eval.ref_difference_plot()

dataset_eval.plt_show()

