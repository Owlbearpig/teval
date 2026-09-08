from pathlib import Path
import matplotlib.pyplot as plt
import h5py
import numpy as np
from PySide6.QtCore import QObject, Signal
from common.components import ComponentBase, action
from common.eval_component.conductivity_models import model_params
from common.eval_component.quantity_set import QuantityDataSetDict, QuantityDataSet
from common.traits import QuantityDict, Path as TPath, Quantity, Q_, StrListSelection, StrList
from traitlets import Bool, Float, Unicode, Integer, observe, Dict
from dataclasses import dataclass, field, asdict
from typing import Any
from functools import partial
from mpl_settings import mpl_style_params

action = partial(action, rc_params=mpl_style_params)

@dataclass
class SingleResultData:
    # optimization arguments
    measurement: str = None
    d : Q_ = Q_(0, "µm")
    shift : Q_ = Q_(0, "fs")
    freq_axis : Q_ = Q_(np.array(0), "THz")

    # result
    regression_params: dict[str, Any] = field(default_factory=dict)  # regression params,
    optimization_info:  dict[str, Any] = field(default_factory=dict) # q_val, gof, converged, timestamp, ...
    datasets: dict[str, "QuantityDataSet"] = field(default_factory=dict) # n0, alpha, t_exp, ...

@dataclass
class EvalResultData:
    result_type: str = ""
    dataset_path: Path = Path(".")
    model_name: str = ""
    measurement_quantity: str = ""
    measurement_names: list[str] = field(default_factory=list)
    results: list[SingleResultData] = field(default_factory=list)

class ResultSignal(QObject):
    received_result = Signal(EvalResultData)
    result_ready = Signal(object)

class EvalResult(ComponentBase):
    quantity_dict = QuantityDict().tag(name="Quantity plot")

    measurement = Unicode("", read_only=True).tag(priority=0, name="Measurement")
    result_type = Unicode("None", read_only=True).tag(priority=1, name="Result type")
    model_name = Unicode("", read_only=True).tag(priority=2, name="Model")
    timestamp = Unicode("", read_only=True).tag(priority=3, name="Timestamp")
    dataset_path = TPath(Path("."), read_only=True).tag(priority=4, name="Dataset path")
    sub_dataset_path = TPath(Path("."), read_only=True).tag(priority=5, name="Sub. dataset path")
    converged = Bool(False, read_only=True).tag(priority=6, name="Converged")

    q_val = Quantity(Q_(0.0, ""), read_only=True)
    gof = Quantity(Q_(0.0, ""), read_only=True)

    reg_result_grp_name = "Regression result values"
    fun = Float(0.0, read_only=True, group=reg_result_grp_name).tag(priority=-1)
    nit = Integer(0, read_only=True, group=reg_result_grp_name).tag(priority=0)
    sig0 = Quantity(Q_(0, "S/cm"), read_only=True, group=reg_result_grp_name).tag(name="σ₀")
    tau = Quantity(Q_(0, "fs"), read_only=True, group=reg_result_grp_name).tag(name="τ")
    wp = Quantity(Q_(0, "THz"), read_only=True, group=reg_result_grp_name).tag(name="ωₚ")
    eps_inf = Float(0, read_only=True, group=reg_result_grp_name).tag(name="ε_inf")
    eps_s = Float(0, read_only=True, group=reg_result_grp_name).tag(name="ε_s")
    c1 = Float(0, read_only=True, group=reg_result_grp_name).tag(name="c₁")

    measurement_list = StrListSelection(group="Evaluated measurements", read_only=True, combine=True, priority=1)
    thicknesses = StrListSelection(group="Thicknesses", read_only=True).tag(max_width=70, combine=True, priority=2)
    shifts = StrListSelection(group="Pulse shifts", read_only=True).tag(max_width=70, combine=True, priority=3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eval_result_data: EvalResultData = None
        self.result_carrier = ResultSignal()
        self.result_carrier.received_result.connect(self.parse_eval_result_data)

        self.set_trait("measurement_list", StrList())
        self.set_trait("thicknesses", StrList())
        self.set_trait("shifts", StrList())

        self.set_observers()

    def select_results(self, meas=None, thickness=None, shift=None):
        if (thickness == "") or (shift == ""):
            return None

        d_cond = (lambda res: True) if thickness is None else (
            lambda res: np.isclose(res.d.magnitude, float(thickness)))
        shift_cond = (lambda res: True) if shift is None else (
            lambda res: np.isclose(res.shift.magnitude, float(shift)))
        meas_cond = (lambda res: True) if meas is None else (lambda res: res.measurement == meas)

        selected_results = (
            res for res in self.eval_result_data.results
            if d_cond(res) and shift_cond(res) and meas_cond(res)
        )

        if meas is not None and thickness is not None and shift is not None:
            return next(selected_results, None)

        return selected_results

    @action(name="Show Q-space plot")
    def plot_q_space(self):
        thicknesses = self.thicknesses.items
        shifts = self.shifts.items
        sel_meas = self.measurement_list.selected_item
        if not thicknesses or not shifts or not sel_meas:
            return

        plt.figure("Q-space plot")
        for shift in shifts:
            results = list(self.select_results(meas=sel_meas, shift=shift))
            if not results:
                continue

            x_vals = [res.d.magnitude if hasattr(res.d, "magnitude") else res.d for res in results]
            y_vals = []
            for res in results:
                q_val = res.optimization_info["q_val"]
                y_vals.append(q_val.magnitude if hasattr(q_val, "magnitude") else q_val)

            plt.plot(x_vals, y_vals, label=f"shift={shift}")

        plt.xlabel("Thickness (µm)")
        plt.ylabel("Q-value")
        plt.legend()
        plt.grid(True)
        plt.show()

    def set_simple_traits(self, trait_values):
        trait_names = self.trait_names()
        for k, v in trait_values.items():
            if isinstance(v, (int, str, float, Q_, Path)) and k in trait_names:
                self.set_trait(k, v)

    def set_observers(self):
        def on_measurement_selection(change):
            selected_meas = change["new"]
            optimization_results = [res for res in self.eval_result_data.results if res.measurement == selected_meas]
            self.thicknesses.items = list(set([str(res.d.magnitude) for res in optimization_results]))
            self.shifts.items = list(set([str(res.shift.magnitude) for res in optimization_results]))

            if self.thicknesses.items and self.shifts.items:
                self.thicknesses.selected_item = self.thicknesses.items[0]
                self.shifts.selected_item = self.shifts.items[0]

            select_quantity_dict(None)

        self.measurement_list.observe(on_measurement_selection, "selected_item")

        def select_quantity_dict(change):
            selected_result = self.select_results(meas=self.measurement_list.selected_item,
                                                  thickness=self.thicknesses.selected_item,
                                                  shift=self.shifts.selected_item)

            if isinstance(selected_result, SingleResultData):
                scalar_values = {
                    "d": selected_result.d,
                    "shift": selected_result.shift,
                    "measurement": selected_result.measurement,
                    **selected_result.optimization_info
                }
                self.set_simple_traits(scalar_values)
                self.quantity_dict = QuantityDataSetDict(selected_result.datasets)

        self.thicknesses.observe(select_quantity_dict, "selected_item")
        self.shifts.observe(select_quantity_dict, "selected_item")

    def load_result(self, res_path):
        res_dict = self.parse_hdf5(res_path)

        self.parse_eval_result_data(res_dict, is_loading=True)

    def parse_hdf5(self, res_path):
        with h5py.File(res_path, "r") as f:
            parsed_result_dict = {}

            if "scalars" in f:
                for k in f["scalars"].keys():
                    dset = f["scalars"][k]
                    val = dset[()]

                    if isinstance(val, bytes):
                        val = val.decode("utf-8")

                    if "unit" in dset.attrs:
                        unit_str = dset.attrs["unit"]
                        if isinstance(unit_str, bytes):
                            unit_str = unit_str.decode("utf-8")

                        val = Q_(val, unit_str)
                    if "path_type" in dset.attrs:
                        val = Path(val)

                    parsed_result_dict[k] = val

            if "quantity_dict" in f:
                qd_group = f["quantity_dict"]

                for k in qd_group.keys():
                    dataset_group = qd_group[k]

                    data_dset = dataset_group["data"]

                    d_unit = data_dset.attrs["unit"]
                    data_label = data_dset.attrs["data_label"]
                    data_q = Q_(data_dset[()], d_unit)

                    axes_q, axes_labels = [], []
                    axes_group = dataset_group["axes"]

                    i = 0
                    while f"axis_{i}" in axes_group:
                        ax_subgroup = axes_group[f"axis_{i}"]
                        axis_dset = ax_subgroup["axis_dset"]

                        ax_unit = axis_dset.attrs["unit"]
                        axes_labels.append(axis_dset.attrs["axis_label"])
                        axes_q.append(Q_(axis_dset[()], ax_unit))
                        i += 1

                    parsed_result_dict[k] = QuantityDataSet(data=data_q, axes=axes_q,
                                                            data_label=data_label, axes_labels=axes_labels)

        return parsed_result_dict

    def parse_eval_result_data(self, eval_result_data: EvalResultData, is_loading=False):
        if not eval_result_data:
            return
        self.eval_result_data = eval_result_data
        self.set_simple_traits(asdict(eval_result_data))
        self.measurement_list.items = eval_result_data.measurement_names

        if eval_result_data.result_type == "Regression":
            active_parameters = model_params(eval_result_data.model_name)
            self.toggle_traits(active_parameters, group_filter=self.reg_result_grp_name)
        elif eval_result_data.result_type == "Transmission fit":
            self.toggle_traits([], group_filter=self.reg_result_grp_name)

        if not is_loading:
            self.result_carrier.result_ready.emit(self)
