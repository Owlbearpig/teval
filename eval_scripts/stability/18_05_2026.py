from common.components import ComponentBase
from common.dataset import DataSet
import os
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from functions import WindowTypes, f_axis_idx_map
from common.settings import Settings
from traitlets import Instance, Float
from common.datasetplotter import DataSetPlotter
from common.consts import c_thz, eps0_thz
from common.components import action

if "nt" in os.name:
    dataset_path = r"C:\Users\alexj\Data/Stability/18052026_systemcover"
else:
    dataset_path = r"/home/ftpuser/ftp/Data/Stability/18052026_systemcover"

redp_labels = {
    "Redp idx 0": "0",
    "Redp idx 1": "1",
    "Redp idx 2": "2",
    "Redp idx 3": "3"
}

class AppRoot(ComponentBase):

    settings = Instance(Settings)
    dataset = Instance(DataSet)
    dataset_plotter = Instance(DataSetPlotter)

    def __init__(self):
        super().__init__()
        self.settings = Settings()
        self.settings.plot_opt.redp_sensor_labels = redp_labels
        self.dataset = DataSet(dataset_path, self.settings, object_name="Dataset")
        self.dataset_plotter = DataSetPlotter(self.dataset, object_name="Dataset Plotter")

    @action("Take new measurement")
    def takeMeasurement(self):
        res = self.dataset_plotter.plot_system_stability()
        self.dataset_plotter.plt_show()
        # self.set_trait('someDataSet', dataSet)

