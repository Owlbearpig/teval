from dataset import DataSet, Dist, QuantityEnum, plt_show, PixelInterpolation

options = {

# "cbar_lim": (6.90, 7.30), # img0
# "cbar_lim": (5.2, 5.60), # img1
# "cbar_lim": (5.6, 6.00), # leaf img0

# "cbar_lim": (5.6, 6.00), # leaf img0
# "cbar_lim": (4.3, 4.50), # leaf img2
# "cbar_lim": (0.945, 0.980), # leaf img2 # looks alright > 1.0 THz
"cbar_lim": (0.80, 0.99), # leaf img2

# filter paper
# "cbar_lim": (0.75, 0.90), # 1.5 THz
# "cbar_lim": (0.86, 0.96), # 1.0 THz
"pixel_interpolation": PixelInterpolation.none,
"plot_range": slice(30, 650),
"ref_pos": (4.0, None),
"dist_func": Dist.Position,
}

dataset = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Leaf/img2", options)
# dataset = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Filter_uncoated/img1", options)

dataset.select_freq(2.0)
# dataset.select_quantity(QuantityEnum.P2P)
dataset.select_quantity(QuantityEnum.TransmissionAmp)
# dataset.plot_point((50, 0), apply_window=False)

dataset.plot_image()
dataset.plot_refs()

# dataset.average_area((19, -2), (32, 5), label="2") # img3

plt_show(en_save=False)
