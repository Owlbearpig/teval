from dataset import DataSet, Dist, QuantityEnum, plt_show, PixelInterpolation

options = {

# leaf uncoated
# "cbar_lim": (4.3, 4.6), # img2 p2p drifts clearly visible
# "cbar_lim": (5.6, 6.00), # img0
# "cbar_lim": (4.3, 4.50), # img2
# "cbar_lim": (0.945, 0.980), # img2 # looks alright > 1.0 THz
# "cbar_lim": (0.89, 1.0), # leaf img2 1.7 THz

# leaf coated
# "cbar_lim": (0.5, 0.9), # leaf img0 1.7 THz

# filter paper coated
# "cbar_lim": (1.00, 1.7), # p2p
#"cbar_lim": (0.05, 0.13), # 1.5 THz
#"cbar_lim": (0.0045, 0.0100), # 1.45 - 1.55 THz
"cbar_lim": (0.005, 0.02), # 1.00 - 1.20 THz
"cbar_lim": (0.003, 0.013), # 1.00 - 1.20 THz img2

"pixel_interpolation": PixelInterpolation.none,
"plot_range": slice(30, 650),
"ref_pos": (4.0, None),
"dist_func": Dist.Position,
"img_title": "(coated)",
}

# dataset = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Leaf/img2", options) # uncoated
dataset = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Leaf_coated/img0", options) # coated


dataset.select_freq((1.45, 1.55))
# dataset.select_quantity(QuantityEnum.P2P)
dataset.select_quantity(QuantityEnum.Power)
# dataset.plot_point((60, 10), apply_window=False)
# dataset.plot_point((64, 12), apply_window=False)

dataset.plot_point((69, 32), apply_window=False)
dataset.plot_point((68, 17), apply_window=False)
dataset.plot_point((57, 38), apply_window=False)
dataset.plot_point((57, 13), apply_window=False)

dataset.plot_image()
# dataset.plot_refs()

# dataset.average_area((19, -2), (32, 5), label="2") # img3

plt_show(en_save=False)
