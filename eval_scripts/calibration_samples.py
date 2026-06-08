from common.components import ComponentBase
from common.dataset import DataSet
import os
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from functions import WindowTypes, f_axis_idx_map
from common.settings import Settings
from traitlets import Instance
from common.datasetplotter import DataSetPlotter
from common.consts import c_thz, eps0_thz
from common.components import action

if "nt" in os.name:
    figure_dir = r"C:\Users\alexj\Mega\AG\Projects\Conductivity\Calibration test samples - Andreone\Results"
else:
    figure_dir = Path(r"/home/alex/MEGA/AG/Projects/Conductivity/Calibration test samples - Andreone")

options = {
"save_plots_settings": {"path": figure_dir, "filetype": "png", "suffix": "", "dpi": 300, "bbox_inches": "tight",
                        "pad_inches": 0, "set_size_inches": (19, 9)},
"sample_properties": {"d": 520, "fp_spacing": 12.0},
"enable_q_eval": True,
"pp_opt": {"window_opt": {"enabled": True,
                          "slope": 0.05, # 0.999, # 0.99
                          #"win_start": 27, # 11,
                          "win_width": 61, # 18,#2*32,# 38*2, # 5*15 # 36
                          # "win_width": 11,
                          "type": WindowTypes.tukey,
                          },
           "filter_opt": {"enabled": False, "f_range": (0.3, 3.0), },
           "remove_dc": True,
           },
"eval_opt": {#"fit_range": (0.35, 2.50),
             #"q-space_range": (1.00, 2.50), # "q-space_range": (0.75, 2.00),
             "phi_fit_range": (0.25, 0.55), # "phi_fit_range": (0.47, 1.05),
             "average": False,
             "fit_range_film": (0.65, 3.2),
             "fit_range_sub": (0.5, 3.5), # (0.10, 3.0)
             "nfp": 2, # number of fp pulses contained in window ("inf" or 0=main pulse only, 1, ..., N),
             "area_fit": False,
             "sub_bounds": [(3.05, 3.14), (0.000, 0.0165)],
             "film_bounds": [(1, 25), (0, 25)],
             # "d_opt_axis": [523.2, 523.25, 523.30, 523.35, 523.40, 523.45, 523.50],
             },
"sim_opt": {"enabled": True,
            "n_sub": 3.05 + 0.005j,
            "shift_sim": 0,
            "nfp_sim": 0,
},
"plot_opt": {"shift_sam2ref": False,
             "stability_plot_rel_change": True,
             "disable_legend": [],
             "temp_sensor_idx": None,
             "subtract_mean": True,
             "plot_range": (0.65, 3.00),
             },
"shown_plots": {
    "Window": True,
    "Time domain": True,
    "Spectrum": True,
    "Phase": True,
    "Phase slope": False,
    "Amplitude transmission": False,
    "Absorbance": False,
    "Refractive index": True,
    "Absorption coefficient": False,
    "Conductivity": True,
    "Absorption coefficient optimum": True,
},
}

# pulse monitoring mod
if "nt" in os.name:
    dataset_path = r"C:\Users\alexj\Data\CalibrationSamples\Graphene"
else:
    dataset_path = r"/home/ftpuser/ftp/Data/CalibrationSamples/Graphene"

from common.traits import Q_

class AppRoot(ComponentBase):

    settings = Instance(Settings)
    dataset = Instance(DataSet)
    dataset_plotter = Instance(DataSetPlotter)

    def __init__(self):
        super().__init__()
        self.settings = Settings()
        self.dataset = DataSet(dataset_path, self.settings)
        self.dataset_plotter = DataSetPlotter(self.dataset)
        self.dataset_plotter.test = Q_(1000000.0, "THz")

    @action("Take new measurement")
    def takeMeasurement(self):
        res = self.dataset_plotter.plot_system_stability()
        self.dataset_plotter.plt_show()
        # self.set_trait('someDataSet', dataSet)

"""
dataset = DataSet(dataset_path)
dataset.select_freq(2.0)
dataset.select_quantity(QuantityEnum.P2P)
#dataset.plot_line(line_coords=-8)
#dataset.plot_line(line_coords=-12)
#dataset.plot_line(line_coords=-14)
# dataset.select_quantity(QuantityEnum.TransmissionPhase)
# dataset.select_quantity(QuantityEnum.P2P)
dataset.plot_system_stability()
dataset.plt_show()
"""

"""
# dataset.select_quantity(QuantityEnum.Phase)
res_sub, mq_sub = dataset.plot_meas((105, -12))
dataset.plt_show()

res_sam, mq_sam = dataset.plot_meas((99, -12))

freq_axis = mq_sub["freq_axis"]
f_idx_range = f_axis_idx_map(freq_axis, freq_range=options["plot_opt"]["plot_range"])
freq_axis = freq_axis[f_idx_range]
t_sub = mq_sub["t_exp"][f_idx_range, 1]
t_sam = mq_sam["t_exp"][f_idx_range, 1] # * np.exp(-1j * 2*np.pi * freq_axis * 0.025)

try:
    n_sub = np.load("n_sub_graphene.npy")
except FileNotFoundError:
    n_sub = res_sub["n"]
    np.save("n_sub_graphene", n_sub)


d_film = 0.010 # µm

sig_tink = (t_sub/t_sam - 1) * eps0_thz * c_thz * (1 + n_sub) / (d_film * 1e-4)

fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, num="Conductivity", sharex=True)
ax0.set_title("Real part")
ax0.plot(freq_axis, sig_tink.real, label="x=97, y=-12")

ax1.set_title("Imaginary part")
ax1.plot(freq_axis, sig_tink.imag, label="x=97, y=-12")

ax0.set_ylabel("Conductivity (S/cm)")
ax1.set_ylabel("Conductivity (S/cm)")
ax1.set_xlabel("Frequency (THz)")

ax0.legend()
ax1.legend()

fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, num="Refractive index sub", sharex=True)
ax0.set_title("Real part")
ax0.plot(freq_axis, n_sub.real, label="substrate")

ax1.set_title("Imaginary part")
ax1.plot(freq_axis, n_sub.imag, label="substrate")

ax0.set_ylabel("Refractive index")
ax1.set_ylabel("Refractive index")
ax1.set_xlabel("Frequency (THz)")

ax0.legend()
ax1.legend()

# dataset.plot_meas(timestamp="2026-04-22T20-53-54.638573")
# dataset.plot_image()

# dataset.system_stability_diff_plot()
# dataset.plot_system_stability(climate_log_file=r"2026-04-16 14-12-08_log_pitaya_subset_0start.txt") # set5
# dataset.plot_climate(log_file="2026-04-17 00-00-00_log_subset.txt", quantity=ClimateQuantity.Humidity)

# dataset.plot_system_stability(climate_log_file="2026-04-22 11-58-34_log_pitaya.txt")
# dataset.options["pp_opt"]["window_opt"]["enabled"] = False
# dataset.plot_ref()
# dataset.plot_ref(ref_idx=1250)

dataset.plt_show()
"""
