from dataset import DataSet, Dist, QuantityEnum, PixelInterpolation
import logging

options = {
# TODO setting this sucks. Fix runtime / use cache for t calc and set lims based on area min max
# "cbar_lim": (0.975, 0.982),

# 1.5 THz
# "cbar_lim": (0.65, 0.70), # img8 1.5 THz
# "cbar_lim": (0.60, 0.68), # img9 1.5 THz

"plot_range": slice(30, 1000),

# "ref_pos": (30, None), # img8
"ref_pos": (30, None), # img9

"ref_threshold": 0.95,
"pixel_interpolation": PixelInterpolation.none,
"dist_func": Dist.Position,
"img_title": "",
"sample_properties": {"d": 650,
                      "d_film": 0.350,
                      },
"pp_opt": {"window_opt": {"enabled": True,
                          "slope": 0.99},
           },
"eval_opt": {"dt": -5, # dt in fs
             "sub_pnt": (50, 10),
             },
"shown_plots": {
    "Time domain": False,
    "Spectrum": False,
    "Phase": False,
    "Phase slope": True,
    "Amplitude transmission": False,
    "Absorbance": False,
    "Refractive index": False,
    "Absorption coefficient": False,
    "Conductivity": True,
},
}

dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img8", options)
sub_dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img8", options)
# dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img9", options)

dataset.link_sub_dataset(sub_dataset)

dataset.select_freq(1.5)
dataset.select_quantity(QuantityEnum.P2P)
# dataset.select_quantity(QuantityEnum.TransmissionAmp)

dataset.plot_system_stability()

dataset.plot_image()
dataset.plot_refs()

# img 8
#dataset.average_area((45, 5), (57, 19), label="Sub. 1")
#dataset.average_area((73, 5), (80, 19), label="2")
#dataset.plot_point((50, 10), apply_window=False)

# img 9
dataset.average_area((45, 5), (57, 19), label="Sub. 2")
dataset.average_area((75, 5), (80, 19), label="9")
# dataset.plot_point((72, 10))
# dataset.plot_point((82, 10))
dataset.plot_point((82, 9.5))

dataset.plt_show()

