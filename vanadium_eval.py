from dataset import DataSet, Dist, QuantityEnum, plt_show
import logging

options = {
# "cbar_lim": (0.52, 0.60), # img3 2.5 THz
# "cbar_lim": (0.64, 0.66), # img4 2.5 THz
# "cbar_lim": (0.60, 0.66), # img5 1.5 THz
# "cbar_lim": (0.53, 0.56), # img5 2.5 THz
# "cbar_lim": (0.54, 0.60), # img6 2.5 THz
# "cbar_lim": (0.54, 0.58), # img7 2.5 THz

# TODO this
"cbar_lim": (0.52, 0.60), # img3 2.5 THz
# "cbar_lim": (0.64, 0.66), # img4 2.5 THz
# "cbar_lim": (0.60, 0.66), # img5 1.5 THz
# "cbar_lim": (0.53, 0.56), # img5 2.5 THz
# "cbar_lim": (0.54, 0.60), # img6 2.5 THz
# "cbar_lim": (0.54, 0.58), # img7 2.5 THz

# "cbar_lim": (0, 12),

"plot_range": slice(30, 650),
# "ref_pos": (7.0, 0.0),
# "ref_pos": (50, 0.0), # img3
"ref_pos": (10, 0), # img4
# "ref_pos": (10, -4), # img5
# "ref_pos": (10, -4), # img6
# "ref_pos": (50, 4), # img7
"dist_func": Dist.Position,
}

# dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img2", options)
# dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img3", options)
dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img4", options)
# dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img5", options)
# dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img6", options)
# dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img7", options)

dataset.select_freq(1.5)
# dataset.select_quantity(QuantityEnum.P2P)
dataset.select_quantity(QuantityEnum.TransmissionAmp)
# dataset.plot_point((35, 0), apply_window=False)

dataset.plot_image()
dataset.plot_refs()

# dataset.average_area((19, -2), (32, 5), label="2") # img3
# dataset.average_area((72, -1), (85, 2), label="9") # img3
#dataset.average_area((25, -10), (48, 3), label="7") # img4
#dataset.average_area((62, -10), (83, 3), label="8") # img4
dataset.average_area((33, -10), (40, 0), label="12.1")  # img5
dataset.average_area((23.5, 2), (28.5, 3), label="12.2") # img5
dataset.average_area((64.5, -10), (80.0, 2), label="Sub") # img5
#dataset.average_area((27, -10), (47, -2), label="VR01") # img6
#dataset.average_area((63.5, -10), (83, -2), label="VR04") # img6
# dataset.average_area((66, 1), (83, 6), label="4") # img7

plt_show(en_save=False)
