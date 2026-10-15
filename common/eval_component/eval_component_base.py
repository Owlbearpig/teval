from common.components import ComponentBase, action
from common.eval_component.eval_result import EvalResult, EvalResultData
from common.save import ResultSaver
from common.measurements import DataSetBase
import numpy as np
import threading
import logging
from PySide6.QtCore import QObject, Signal, Slot, QTimer
from common.eval_component.transfer_functions import TransmissionModels
from common.eval_component.cost_functions import CostFunctions
from common.traits import Q_, ValueRange, Path as TPath
from traitlets import Enum as TEnum, observe, Integer, Float, Bool, Instance, Unicode
from pathlib import Path

class BusyIndicator(QObject):
    running_changed = Signal(bool)
    frame_changed = Signal(str)

    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._idx = 0
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        self.running_changed.connect(self._set_running)

    def _set_running(self, running):
        if running:
            self._idx = 0
            self._timer.start()
        else:
            self._timer.stop()
            self.frame_changed.emit("Idle")

    def _tick(self):
        self._idx = (self._idx + 1) % len(self.frames)
        self.frame_changed.emit(f"{self.frames[self._idx]} Running")

class ResultSignal(QObject):
    result_ready = Signal(EvalResultData)

class RemainingTaskCntSignal(QObject):
    cnt_changed = Signal(int)

class EvalComponentBase(DataSetBase):
    selected_result_path = TPath(Path("")).tag(name="Load result", en_save=False, priority=-100)
    number_of_workers = Integer(8).tag(name="Number of cpu cores to assign", priority=2)
    transmission_model = TEnum(TransmissionModels, default_value=TransmissionModels.tmm_1layer).tag(
        name="Selected transmission model", priority=-1)
    selected_cost_fun = TEnum(CostFunctions, default_value=CostFunctions.combined_cost,
                              help="Model to experimental data metric").tag(name="Selected cost function", priority=3)
    only_use_avg = Bool(True).tag(name="Only evaluate average", priority=1003)
    freq_range = ValueRange([Q_(0.50, "THz"), Q_(2.20, "THz")]).tag(name="Fit range", priority=1004)

    process_status = "Processing"
    activity = Unicode("Idle", read_only=True).tag(name="Status", group=process_status,
                                                   priority=1004, en_save=False)
    optimization_progress = Float(0, min=0, max=1, read_only=True).tag(name="Progress", group=process_status,
                                                                       priority=1005, en_save=False)
    completed_tasks = Integer(0, read_only=True).tag(name="Completed tasks",
                                                     priority=1006, group=process_status, en_save=False)
    total_tasks = Integer(0, read_only=True).tag(name="Total tasks",
                                                 priority=1007, group=process_status, en_save=False)

    current_result = Instance(EvalResult)
    result_saver = Instance(ResultSaver)

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.result_carrier = ResultSignal()

        self.current_result = EvalResult(object_name="Current result")

        self.result_saver = ResultSaver(object_name="Result saver")
        self.result_saver.registerObjectAttribute(EvalResultData, "result_type")

        self.add_result_ready_callback(self.current_result.parse_eval_result_data)
        self.add_result_ready_callback(self.result_saver.save_result_data)
        self.current_result.signals.save_requested.connect(self.result_saver.save_result_data)

        self.task_cnt_carrier = RemainingTaskCntSignal()
        self.task_cnt_carrier.cnt_changed.connect(self.update_progress)

        self._cancel_event = threading.Event()
        self._thread_executor = None
        self.busy_indicator = BusyIndicator()
        self.busy_indicator.frame_changed.connect(self._set_activity)
        self._active_eval_ = None

    def _set_activity(self, text):
        self.set_trait("activity", text)

    @property
    def _active_eval(self):
        return self._active_eval_

    @_active_eval.setter
    def _active_eval(self, value):
        self._active_eval_ = value
        self.busy_indicator.running_changed.emit(value is not None)

    def add_result_ready_callback(self, callback):
        self.result_carrier.result_ready.connect(callback)

    def remove_result_ready_callback(self, callback):
        self.result_carrier.result_ready.disconnect(callback)

    def _result_ready(self, eval_res_data: EvalResultData):
        self.result_carrier.result_ready.emit(eval_res_data)

    def _request_cancel(self):
        self._cancel_event.set()

    @action("Cancel evaluation", priority=1008, group=process_status)
    def cancel_evaluation(self):
        if self._active_eval is None:
            logging.info("Evaluation is not running")
            return
        self._request_cancel()
        logging.info("Cancelling evaluation...")

    def update_progress(self, task_cnt_change: int = 0):
        if task_cnt_change == 0:
            completed, total = 0, 0
            progress = 0
        elif task_cnt_change == 1:
            completed, total = self.completed_tasks + task_cnt_change, self.total_tasks
        else:
            completed, total = 0, task_cnt_change
            progress = 0

        self.set_trait("completed_tasks", completed)
        self.set_trait("total_tasks", total)

        progress = 0 if not self.total_tasks else self.completed_tasks / self.total_tasks
        self.set_trait("optimization_progress", progress)

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

