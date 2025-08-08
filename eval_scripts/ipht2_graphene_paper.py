from dataset import DataSet, Dist, QuantityEnum, plt_show, PixelInterpolation

options = {
# filter paper
# "cbar_lim": (6.5, 7.6), # filter paper uncoated p2p img0
"cbar_lim": (1.15, 1.45), # filter paper uncoated p2p img0
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

"pixel_interpolation": PixelInterpolation.none,
"plot_range": slice(30, 650),
"ref_pos": (4.0, None),
"dist_func": Dist.Position,
"img_title": "",
    "sample_properties": {"d": 150,
                          "d_film": 0.010,
                          },
    "pp_opt": {"window_opt": {"enabled": True, "slope": 0.99},
               "dt": 55, # dt in fs
               },
}

#sub_dataset = DataSet(r"C:\Users\alexj\Data\IPHT2\Filter_uncoated\img0", options)
#dataset = DataSet(r"C:\Users\alexj\Data\IPHT2\Filter_coated\img0", options)

sub_dataset = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Filter_uncoated/img0", options)
dataset = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Filter_coated/img0", options)

dataset.link_sub_dataset(sub_dataset)

dataset.select_freq(0.55)
dataset.select_quantity(QuantityEnum.TransmissionAmp)

sub_dataset.plot_point((60, 10))
# sub_dataset.plot_point((70, 10))
# dataset.plot_point((61, 10))
# dataset.plot_point((65, 10))
# dataset.plot_point((64, 12), apply_window=False)

# dataset.plot_image()
# dataset.plot_refs()

# dataset.average_area((19, -2), (32, 5), label="2")

plt_show(en_save=False)
