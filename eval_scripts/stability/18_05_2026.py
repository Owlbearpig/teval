from common.components import ComponentBase
from common.dataset import DataSet
import os
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from common.settings import Settings
from traitlets import Instance, Float
from common.datasetplotter import DataSetPlotter
from common.eval_component.dataset_eval import DatasetEval
from common.consts import c_thz, eps0_thz
from common.components import action

if "nt" in os.name:
    dataset_path = r"C:\Users\alexj\Data/Stability/18052026_systemcover"
    dataset_path_sub = r"C:\Users\alexj\Data\Stability\01062026_systemcover_subset"
else:
    dataset_path_sub = r"/home/ftpuser/ftp/Data/Stability/01062026_systemcover_subset"
    dataset_path = r"/home/ftpuser/ftp/Data/Stability/18052026_systemcover"


redp_labels = {
    "Redp idx 0": "0",
    "Redp idx 1": "1",
    "Redp idx 2": "2",
    "Redp idx 3": "3"
}

settings_file = "18_05_2026"
sub_settings_file = "18_05_2026_sub"

class AppRoot(ComponentBase):

    settings = Instance(Settings)
    settings_sub = Instance(Settings)
    dataset = Instance(DataSet)
    dataset_sub = Instance(DataSet)
    dataset_plotter = Instance(DataSetPlotter)
    dataset_eval = Instance(DatasetEval)

    def __init__(self):
        super().__init__()
        self.settings = Settings(settings_file, object_name="Settings")
        self.settings_sub = Settings(sub_settings_file, object_name="Sub settings")
        self.settings.plot_opt.redp_sensor_labels = redp_labels
        self.dataset = DataSet(dataset_path, self.settings, object_name="Dataset")

        self.dataset_sub = DataSet(dataset_path_sub, self.settings_sub, object_name="Dataset substrate")

        self.dataset_plotter = DataSetPlotter(self.dataset, object_name="Dataset Plotter")
        self.dataset_eval = DatasetEval(self.dataset, self.dataset_sub, object_name="Dataset Evaluation")
        
    @action("Take new measurement")
    def takeMeasurement(self):
        res = self.dataset_plotter.plot_system_stability()
        self.dataset_plotter.plt_show()
        # self.set_trait('someDataSet', dataSet)

