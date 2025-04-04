from dataset import DataSet, Dist, QuantityEnum, plt_show
import logging

options = {
"cbar_lim": (0.52, 0.60),  # img3 2.5 THz
"plot_range": slice(30, 650),
"ref_pos": (45, None),
"dist_func": Dist.Position,
"result_dir": r"E:\Mega\AG\Projects\Conductivity\IPHT_2\Results\Holzfurniere",
}

dataset = DataSet(r"E:\measurementdata\IPHT2\Holzfurniere\Tulpe", options)
# dataset = DataSet(r"E:\measurementdata\IPHT2\Wood\S1", options)

dataset.select_freq(1.25)
dataset.select_quantity(QuantityEnum.TransmissionAmp)
dataset.plot_point((15, 0), apply_window=False)
dataset.plot_point((80, 0), apply_window=False)

dataset.plot_line(line_coords=[-5.0, 0.0, 5.0])
dataset.plot_refs()

# dataset.average_area((33, -10), (40, 0), label="12.1")  # img5

plt_show(en_save=False)
