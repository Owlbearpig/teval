from dataset import DataSet

from dataset import DataSet, Dist, QuantityEnum, plt_show, PixelInterpolation

options = {
"cbar_lim": (0.5, 0.9), # leaf img0 1.7 THz

"pixel_interpolation": PixelInterpolation.none,
"plot_range": slice(30, 650),
"ref_pos": (4.0, None),
"dist_func": Dist.Position,
"img_title": "(coated)",
}

dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/sub_inc_angle_var/10deg", options)

dataset.select_freq(1.5)
# dataset.select_quantity(QuantityEnum.P2P)
dataset.select_quantity(QuantityEnum.TransmissionAmp)
dataset.plot_point((40, 0), apply_window=False)

dataset.plot_image()
# dataset.plot_refs()

dataset.average_area((39, -1), (41, 1))

plt_show(en_save=False)

