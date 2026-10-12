from common.components import ComponentBase, action
from common.eval_component.eval_result import EvalResult, EvalResultData
from common.save import ResultSaver
from common.functions import f_axis_idx_map
import numpy as np
import threading
import logging
from PySide6.QtCore import QObject, Signal
from common.eval_component.transfer_functions import TransmissionModels
from common.eval_component.cost_functions import CostFunctions
from common.traits import Q_, ValueRange, Path as TPath
from traitlets import Enum as TEnum, observe, Integer, Float, Bool, Instance
from pathlib import Path

class ResultSignal(QObject):
    result_ready = Signal(EvalResultData)
    save_selected = Signal(EvalResultData, list)

class ProgressSignalCarrier(QObject):
    progress_changed = Signal(float)

class EvalComponentBase(ComponentBase):
    selected_result_path = TPath(Path("")).tag(name="Load result", en_save=False, priority=-100)
    number_of_workers = Integer(8).tag(name="Number of cpu cores to assign", priority=2)
    transmission_model = TEnum(TransmissionModels, default_value=TransmissionModels.tmm_1layer).tag(
        name="Selected transmission model", priority=-1)
    selected_cost_fun = TEnum(CostFunctions, default_value=CostFunctions.abs_cost,
                              help="Model to experimental data metric").tag(name="Selected cost function", priority=3)

    only_eval_avg = Bool(True).tag(name="Only evaluate average", priority=1003)
    fit_range = ValueRange([Q_(0.50, "THz"), Q_(2.20, "THz")]).tag(name="Fit range", priority=1004)
    optimization_progress = Float(0, min=0, max=1, read_only=True).tag(name="Progress", priority=1005, en_save=False)

    current_result = Instance(EvalResult)
    result_saver = Instance(ResultSaver)

    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.result_carrier = ResultSignal()

        self.current_result = EvalResult(object_name="Current result", result_carrier=self.result_carrier)

        self.result_saver = ResultSaver()
        self.result_saver.registerObjectAttribute(EvalResultData(), "result_type")

        self.add_result_ready_callback(self.current_result.parse_eval_result_data)
        self.add_result_ready_callback(self.result_saver.save_result_data)
        self.result_carrier.save_selected.connect(self.result_saver.save_result_data)

        self.progress_carrier = ProgressSignalCarrier()
        self.progress_carrier.progress_changed.connect(self.update_progress)

        self._cancel_event = threading.Event()
        self._thread_executor = None
        self._active_eval = None

    def __exit__(self, *args):
        if self.settings is not None:
            self.settings.save_configuration(self)

    def __enter__(self, *args):
        if self.settings is not None:
            self.settings.load_configuration(self)
        return self

    def add_result_ready_callback(self, callback):
        self.result_carrier.result_ready.connect(callback)

    def remove_result_ready_callback(self, callback):
        self.result_carrier.result_ready.disconnect(callback)

    def _result_ready(self, eval_res_data: EvalResultData):
        self.result_carrier.result_ready.emit(eval_res_data)

    def _request_cancel(self):
        self._cancel_event.set()

    @action("Cancel evaluation")
    def cancel_evaluation(self):
        if self._active_eval is None:
            logging.info("Evaluation is not running")
            return
        self._request_cancel()
        logging.info("Cancelling evaluation...")

    def update_progress(self, progress_value):
        self.set_trait("optimization_progress", progress_value)

    @property
    def freq_axis(self):
        f_axis = self.dataset.freq_axis[self.freq_range_idx]
        return f_axis

    @property
    def freq_range_idx(self):
        return f_axis_idx_map(self.dataset.freq_axis, self.fit_range)

    @property
    def settings(self):
        return self.dataset.settings

    @observe("selected_result_path")
    def set_selected_result(self, change):
        result_path = change["new"]
        self.current_result.load_result(result_path)

    def is_two_layer_t_model(self):
        layer_cnt = getattr(self.transmission_model.value, "layer_cnt", 1)
        return layer_cnt == 2

    def get_t_model_kwargs(self, substrate_result=None):
        model_kwargs = {"nfp": self.settings.eval_opt.fp_count,
                        "n1": 1,
                        "n4": 1,
                        "shift": 0,
                        }

        if self.is_two_layer_t_model():
            if substrate_result is None:
                raise Exception("Substrate result required for two layer model optimization")
            model_kwargs["h"] = self.settings.eval_opt.d_film.magnitude
            substrate_result = substrate_result.dataset_dict
            try:
                n_sub_real_dataset = substrate_result["n"]
                n_sug_imag_dataset = substrate_result["k"]
                n_sub = n_sub_real_dataset.data.magnitude + 1j * n_sug_imag_dataset.data.magnitude
            except KeyError:
                raise Exception("Substrate result required for two layer model optimization")
            model_kwargs["n_sub"] = n_sub
            if not np.array_equal(n_sub_real_dataset.axes[0], self.freq_axis):
                raise Exception("Frequency axis must equal substrate result frequency axis")

        return model_kwargs

