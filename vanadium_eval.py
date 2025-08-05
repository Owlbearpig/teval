from dataset import DataSet, Dist, QuantityEnum, plt_show
import logging

options = {
# TODO setting this sucks. Fix runtime / use cache for t calc and set lims based on area min max
# "cbar_lim": (0.975, 0.982),
# 2.5 THz
# "cbar_lim": (0.52, 0.60), # img3
# "cbar_lim": (0.64, 0.66), # img4
# "cbar_lim": (0.53, 0.56), # img5
# "cbar_lim": (0.54, 0.60), # img6
# "cbar_lim": (0.54, 0.58), # img7

# 1.5 THz
# "cbar_lim": (0.67, 0.68), # img3 1.5 THz
# "cbar_lim": (0.60, 0.66), # img4 1.5 THz
# "cbar_lim": (0.60, 0.66), # img5 1.5 THz
# "cbar_lim": (0.64, 0.67), # img6 1.5 THz
# "cbar_lim": (0.66, 0.67), # img7 1.5 THz
# "cbar_lim": (0.65, 0.70), # img8 1.5 THz
# "cbar_lim": (0.60, 0.68), # img9 1.5 THz


"plot_range": slice(30, 1000),
# "ref_pos": (7.0, 0.0),
# "ref_pos": (50, None), # img3
# "ref_pos": (10, None), # img4
# "ref_pos": (5, None), # img5
# "ref_pos": (10, None), # img6
# "ref_pos": (50, None), # img7
# "ref_pos": (30, None), # img8
"ref_pos": (30, None), # img9
"ref_threshold": 0.95,
"dist_func": Dist.Time,
}

# dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img2", options)
# dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img3", options)
# dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img4", options) # S8 has 2 local defects
# dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img5", options)
# dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img6", options)
# dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img7", options)
dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img8", options)
# dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img9", options)

dataset.select_freq(1.5)
dataset.select_quantity(QuantityEnum.P2P)
# dataset.select_quantity(QuantityEnum.TransmissionAmp)

# test
#dataset.plot_frequency_noise()
#plt_show()
#exit()
# dataset.plot_system_stability()
#plt_show()
#exit()
dataset.plot_image()
dataset.plot_refs()

#dataset.average_area((19, -2), (32, 5), label="2") # img3
#dataset.average_area((72, -1), (85, 2), label="9") # img3
#dataset.plot_point((25, 0), apply_window=False)
#dataset.plot_point((80, 0), apply_window=False)

#dataset.average_area((25, -10), (48, 3), label="7") # img4
#dataset.average_area((62, -7.5), (83, -1.5), label="8") # img4

#dataset.average_area((33, -10), (40, 2), label="12.1")  # img5
#dataset.average_area((23.5, 2), (28.5, 3), label="12.2") # img5
#dataset.average_area((64.5, -10), (80.0, -4.5), label="Sub1") # img5
#dataset.average_area((70, -10), (80.0, 3.5), label="Sub2") # img5
#dataset.plot_point((75, 2.5), apply_window=False, label="Sub")
#dataset.plot_point((35.0, 2.5), apply_window=False, label="12.1")
#dataset.plot_point((25.5, 2.5), apply_window=False, label="12.2")
# dataset.plot_point((25.5, 3), apply_window=False)

#dataset.average_area((33, 1), (40, 2), label="12.3")  # img5
#dataset.average_area((64.5, 1), (80.0, 2), label="Sub1") # img5

# dataset.average_area((30, -10), (40, -2), label="VR01") # img6
# dataset.average_area((70, -10), (80, -1.5), label="VR04") # img6
#dataset.average_area((26, -5.5), (47.0, -5.0), label="VR01") # img6
#dataset.average_area((26, 7.5), (47.0, 8.0), label="Luft") # img6
# dataset.average_area((60, 8.0), (79.0, 8.0), label="Luft2") # img6
# dataset.average_area((70, -2), (80, -1.5), label="VR04") # img6
#dataset.plot_point((40, 0), apply_window=False)
#dataset.plot_point((70, 0), apply_window=False)

# dataset.average_area((65, 3.5), (84, 6.5), label="4") # img7
#dataset.plot_point((70, 5), apply_window=False)

# img 8
#dataset.average_area((45, 5), (57, 19), label="Sub. 1")
#dataset.average_area((73, 5), (80, 19), label="2")
#dataset.plot_point((50, 10), apply_window=False)

# img 9
dataset.average_area((45, 5), (57, 19), label="Sub. 2")
dataset.average_area((75, 5), (80, 19), label="9")
dataset.plot_point((50, 10), apply_window=False)

plt_show(en_save=False)
