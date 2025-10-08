from dataset import DataSet, Dist, QuantityEnum, PixelInterpolation, WindowTypes
from dataset_eval import DatasetEval, DataSetType

options = {
# filter paper
"cbar_lim": (6.5, 7.6), # filter paper uncoated p2p img0
# "cbar_lim": (1.15, 1.45), # filter paper uncoated p2p img0
# "cbar_lim": (4.9, 5.9), # filter paper uncoated p2p img1
# "cbar_lim": (0.8, 1.9), # filter paper coated p2p
# "cbar_lim": (0.75, 0.90), # 1.5 THz
# "cbar_lim": (0.86, 0.96), # 1.0 THz
# "cbar_lim": (0.80, 0.88), # 1.5 THz
# "cbar_lim": (0.69, 0.79), # 1.45-1.55 THz
# "cbar_lim": (0.79, 0.86), # 1.0-1.2 THz

# filter paper coated
# "cbar_lim": (1.00, 1.7), # p2p
#"cbar_lim": (0.05, 0.13), # 1.5 THz
#"cbar_lim": (0.0045, 0.0100), # 1.45 - 1.55 THz
#"cbar_lim": (0.005, 0.02), # 1.00 - 1.20 THz
#"cbar_lim": (0.003, 0.013), # 1.00 - 1.20 THz img2

"plot_range": slice(10, 1000),

"ref_pos": (20, None), # img9

"ref_threshold": 0.90,
"pixel_interpolation": PixelInterpolation.none,
"dist_func": Dist.Time,
"img_title": "",
"sample_properties": {"d_1": 640,#650,
                      "d_2": 650,
                      "d_film": 0.300, },
"pp_opt": {"window_opt": {"enabled": True,
                          "slope": 0.05, # 0.999, # 0.99
                          # "win_start": 0,
                          "win_width": 70, # 18,#2*32,# 38*2, # 5*15 # 36
                          "type": WindowTypes.tukey,
                          },
           "filter_opt": {"enabled": False, "f_range": (0.3, 3.0), },
           "remove_dc": True,
           },
"eval_opt": {"shift_sub": 0, # ref <-> sam pulse shift in fs
             "shift_film": 0,
             "sub_pnt": (30, 5),#(32, 5),
             "fit_range_film": (0.65, 3.2),
             "fit_range_sub": (0.5, 1.5), # (0.10, 3.0)
             "nfp": 2, # number of fp pulses contained in window ("inf" or 0, 1, ..., N),
             "area_fit": False,
             },
"sim_opt": {"enabled": True,
            "n_sub": 3.05 + 0.005j,
            "shift_sim": 35,
            "nfp_sim": 2,
},
"plot_opt": {"shift_sam2ref": False,},
"shown_plots": {
    "Window": True,
    "Time domain": True,
    "Spectrum": True,
    "Phase": False,
    "Phase slope": False,
    "Amplitude transmission": False,
    "Absorbance": False,
    "Refractive index": False,
    "Absorption coefficient": False,
    "Conductivity": True,
},
}

#sub_dataset = DataSet(r"C:\Users\alexj\Data\IPHT2\Filter_uncoated\img0", options)
#dataset = DataSet(r"C:\Users\alexj\Data\IPHT2\Filter_coated\img0", options)

sub_dataset_path = r"/home/ftpuser/ftp/Data/IPHT2/Filter_uncoated/img0"
sam_dataset_path = r"/home/ftpuser/ftp/Data/IPHT2/Filter_coated/img0"


dataset_eval = DatasetEval(sam_dataset_path, sub_dataset_path, options)

# dataset_eval.select_freq((2.00, 2.10))
dataset_eval.select_freq(0.5)
dataset_eval.select_quantity(QuantityEnum.P2P)

dataset_eval.sub_dataset.plot_image()
dataset_eval.sub_dataset.plot_point((60, 10))
# dataset_eval.plot_point((60, 10))
# dataset_eval.plot_point((70, 10))
# dataset_eval.plot_point((61, 10))
# dataset_eval.plot_point((65, 10))
# dataset_eval.plot_point((64, 12), apply_window=False)

# dataset_eval.plot_image()
# dataset_eval.plot_refs()

# dataset_eval.average_area((19, -2), (32, 5), label="2")

dataset_eval.plt_show()
