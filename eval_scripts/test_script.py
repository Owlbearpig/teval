from common.components import ComponentBase
from common.dataset import DataSet
from common.eval_component.q_space_eval import QSpaceEval
import os
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from common.settings import Settings
from traitlets import Instance, Float
from common.dataset_plotter import QuantityPlots, ImagePlot
from common.consts import c_thz, eps0_thz
from common.components import action

redp_labels = {
    "Redp idx 0": "0",
    "Redp idx 1": "1",
    "Redp idx 2": "2",
    "Redp idx 3": "3"
}

settings_file = "18_05_2026"
settings_file = ""
sub_settings_file = "18_05_2026_sub"


class AppRoot(ComponentBase):
    settings = Instance(Settings, allow_none=True)
    # settings_sub = Instance(Settings)
    dataset = Instance(DataSet, allow_none=True)
    # dataset_sub = Instance(DataSet)
    quantity_plotter = Instance(QuantityPlots, allow_none=True)
    image_plot = Instance(ImagePlot, allow_none=True)
    q_space_eval = Instance(QSpaceEval, allow_none=True)

    def __init__(self):
        super().__init__()
        self.settings = Settings(settings_file, object_name="Settings")
        # self.settings_sub = Settings(sub_settings_file, object_name="Sub settings")
        self.settings.plot_opt.redp_sensor_labels = redp_labels
        self.dataset = DataSet(self.settings, object_name="Dataset")

        # self.dataset_sub = DataSet(self.settings_sub, object_name="Dataset substrate")
        self.dataset_sub = None
        self.quantity_plotter = QuantityPlots(self.dataset, object_name="Quantity plotter")
        self.image_plot = ImagePlot(self.dataset, en_avg_rect_component=False,
                                    en_slice_plots=False, object_name="Image plot")
        self.q_space_eval = QSpaceEval(self.dataset, object_name="Q-space evaluation")

    @action("Take new measurement")
    def takeMeasurement(self):
        self.quantity_plotter.plot_system_stability()
        self.quantity_plotter.plt_show()
        # self.set_trait('someDataSet', dataSet)

