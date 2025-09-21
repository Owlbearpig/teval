from dataset import DataSet, Dist, QuantityEnum, PixelInterpolation

options = {
# "cbar_lim": (6.90, 7.30), # img0
# "cbar_lim": (5.2, 5.60), # img1

# leaf uncoated
# "cbar_lim": (4.3, 4.6), # img2 p2p drifts clearly visible
# "cbar_lim": (5.6, 6.00), # img0
# "cbar_lim": (4.3, 4.50), # img2
# "cbar_lim": (0.945, 0.980), # img2 # looks alright > 1.0 THz
# "cbar_lim": (0.89, 1.0), # leaf img2 1.7 THz

# leaf coated
# "cbar_lim": (0.5, 0.9), # leaf img0 1.7 THz

# filter paper
# "cbar_lim": (6.5, 7.6), # filter paper uncoated p2p
# "cbar_lim": (0.8, 1.9), # filter paper coated p2p
# "cbar_lim": (0.75, 0.90), # 1.5 THz
# "cbar_lim": (0.03, 0.12), # filter paper coated 1.5 THz
"cbar_lim": (0.007, 0.029), # filter paper coated 1.7 THz # humid
# "cbar_lim": (0.86, 0.96), # 1.0 THz
# filter paper coated
# "cbar_lim": (1.00, 1.7), # p2p
# "cbar_lim": (0.04, 0.10), # 2.0 THz
"pixel_interpolation": PixelInterpolation.none,
"plot_range": slice(30, 650),
"ref_pos": (2.0, None),
"dist_func": Dist.Position,
"img_title": "(coated)",
}

# dataset = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Leaf_coated/img3", options)

# dataset = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Filter_uncoated/img0", options)
dataset = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Filter_coated/img3", options)

# dataset = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Holzfurniere/Uncoated_birke_s1/img0", options) # birke s1/s2

dataset.select_freq(1.7)
# dataset.select_quantity(QuantityEnum.P2P)
dataset.select_quantity(QuantityEnum.TransmissionAmp)
# dataset.plot_point((54, 17.5), apply_window=False)

dataset.plot_image()
dataset.plot_refs()

# dataset.average_area((19, -2), (32, 5), label="2") # img3

dataset.plt_show()
