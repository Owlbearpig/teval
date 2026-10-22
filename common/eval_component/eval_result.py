from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6 import QtWidgets
from datetime import datetime
from common.components import ComponentBase, action
from common.eval_component.conductivity_models import parameter_label as model_parameter_label
from common.eval_component.model_latex import identifier_label
from common.eval_component.quantity_set import QuantityDataSetDict, QuantityDataSet
from common.traits import (QuantityDict, Path as TPath, Quantity, Q_, StrListSelection, StrList, FitParameter,
                           FitParameterTable)
from traitlets import Bool, Float, Unicode, Integer
from dataclasses import dataclass, field, asdict
from typing import Any
from functools import partial
from mpl_settings import mpl_style_params
import logging

action = partial(action, rc_params=mpl_style_params)


def parameter_label(name):
    return model_parameter_label(name) or identifier_label(name)

@dataclass
class SingleResultData:
    measurement: str = ""
    d : Q_ = Q_(0, "µm")
    shift : Q_ = Q_(0, "fs")
    freq_axis : Q_ = Q_(np.array(0), "THz")

    # result
    regression_params: dict[str, Any] = field(default_factory=dict)  # regression params,
    optimization_info:  dict[str, Any] = field(default_factory=dict) # q_val, gof, converged, timestamp, ...
    datasets: dict[str, QuantityDataSet] = field(default_factory=dict) # n0, alpha, t_exp, ...

@dataclass
class EvalResultData:
    result_type: str = ""
    dataset_path: Path = Path(".")
    model_name: str = ""
    measurement_quantity: str = ""
    measurement_names: list[str] = field(default_factory=list)
    reference_names: list[str] = field(default_factory=list)
    results: dict[str, SingleResultData] = field(default_factory=dict)
    optimum_results: dict[str, str] = field(default_factory=dict)

class EvalResultSignals(QObject):
    save_requested = Signal(EvalResultData, list)

class EvalResult(ComponentBase):
    quantity_dict = QuantityDict().tag(name="Quantity plot")

    transmission_res_grp_name = "Transmission fit result"
    measurement = Unicode("", read_only=True,
                          group=transmission_res_grp_name).tag(priority=0, name="Measurement")
    result_type = Unicode("", read_only=True,
                          group=transmission_res_grp_name).tag(priority=1, name="Result type")
    model_name = Unicode("", read_only=True,
                         group=transmission_res_grp_name).tag(priority=2, name="Model")
    timestamp = Unicode("", read_only=True,
                        group=transmission_res_grp_name).tag(priority=3, name="Timestamp")
    dataset_path = TPath(Path("."), must_exist=False, read_only=True,
                         group=transmission_res_grp_name).tag(priority=4, name="Dataset path")
    measurement_info = Unicode("", read_only=True, group=transmission_res_grp_name).tag(
        priority=5, name="Measurements")
    converged = Bool(False, read_only=True,
                     group=transmission_res_grp_name).tag(priority=6, name="Converged")
    q_val = Quantity(Q_(0.0, ""), read_only=True, group=transmission_res_grp_name,
                     significant_figures=5).tag(name="Q-val")
    gof = Quantity(Q_(0.0, ""), read_only=True, group=transmission_res_grp_name,
                   significant_figures=5).tag(name="GoF")
    fp_time = Quantity(Q_(0.0, "ps"), read_only=True, group=transmission_res_grp_name,
                       significant_figures=4).tag(name="FP round trip time")
    q_space_f_max = Quantity(Q_(0.0, "THz"), read_only=True, group=transmission_res_grp_name,
                             significant_figures=3).tag(name="Q-value range upper limit")
    nfev = Integer(0, read_only=True,
                   group=transmission_res_grp_name).tag(name="Number of function evaluations")

    reg_result_grp_name = "Regression result values"
    fun = Float(0.0, read_only=True, group=reg_result_grp_name).tag(priority=-1)
    nit = Integer(0, read_only=True, group=reg_result_grp_name).tag(priority=0)
    fit_parameters = FitParameterTable(read_only=True, group=reg_result_grp_name).tag(name="Fitted parameters",
                                                                                    disable_label=True, priority=1)

    measurement_list = StrListSelection(group="Evaluated measurements", read_only=True, combine=True, priority=1)
    thicknesses = StrListSelection(group="Thicknesses (µm)", read_only=True).tag(max_width=70, combine=True, priority=2)
    shifts = StrListSelection(group="Pulse shifts (fs)", read_only=True).tag(max_width=70, combine=True, priority=3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eval_result_data: EvalResultData = EvalResultData()
        self.signals = EvalResultSignals()

        self.set_trait("measurement_list", StrList())
        self.set_trait("thicknesses", StrList())
        self.set_trait("shifts", StrList())

        self.set_observers()

    @property
    def _ui_control_widget(self):
        return getattr(self, "_ui_widget_internal", None)

    @_ui_control_widget.setter
    def _ui_control_widget(self, value):
        self._ui_widget_internal = value
        if value:
            self.toggle_traits([], group_filter=self.reg_result_grp_name)

    def select_results(self, meas=None, thickness=None, shift=None):
        if (thickness == "") or (shift == ""):
            return None

        d_cond = (lambda res: True) if thickness is None else (
            lambda res: np.isclose(res.d.magnitude, float(thickness)))
        shift_cond = (lambda res: True) if shift is None else (
            lambda res: np.isclose(res.shift.magnitude, float(shift)))
        meas_cond = (lambda res: True) if meas is None else (lambda res: res.measurement == meas)

        opt_results = self.eval_result_data.results.values()
        selected_results = (
            res for res in opt_results
            if d_cond(res) and shift_cond(res) and meas_cond(res)
        )

        if meas is not None and thickness is not None and shift is not None:
            return next(selected_results, None)

        return selected_results

    @staticmethod
    def _timestamp(name):
        try:
            return datetime.strptime(str(name)[:26], "%Y-%m-%dT%H-%M-%S.%f")
        except ValueError:
            return None

    def _measurement_summary(self):
        sams = self.eval_result_data.measurement_names
        refs = self.eval_result_data.reference_names
        if not sams:
            return ""
        parts = [f"{len(sams)} sample{'s' * (len(sams) != 1)}"]
        if refs:
            unique = len(set(refs))
            parts.append(f"{len(refs)} reference{'s' * (len(refs) != 1)}"
                         + (f" ({unique} unique)" if unique != len(refs) else ""))
        times = [t for t in map(self._timestamp, sams) if t is not None]
        if times:
            t0, t1 = min(times), max(times)
            span = t0.strftime("%Y-%m-%d %H:%M:%S")
            if t1 != t0:
                span += " – " + t1.strftime("%H:%M:%S" if t1.date() == t0.date() else "%Y-%m-%d %H:%M:%S")
            parts.append(span)
        return ", ".join(parts)

    def measurement_details(self):
        data = self.eval_result_data
        lines = [f"Result type: {data.result_type}", f"Model: {data.model_name}", f"Dataset: {data.dataset_path}", ""]

        selected = self.measurement_list.selected_item
        if selected == "Average":
            lines.append(f"Selected result: average of the {len(data.measurement_names)} sample measurements below")
        elif selected:
            lines.append(f"Selected result: {selected}")
        key = data.optimum_results.get(selected) if selected else None
        layer_stack = data.results[key].optimization_info.get("layer_stack") if key in data.results else None
        if layer_stack:
            lines.append(f"Layer stack (propagation order): {layer_stack}")
        lines.append("")

        lines.append(f"Sample measurements ({len(data.measurement_names)}):")
        lines += [f"  {name}" for name in data.measurement_names]
        lines.append("")
        if data.reference_names:
            lines.append(f"Reference measurements ({len(data.reference_names)}, "
                         f"{len(set(data.reference_names))} unique), in the order of the samples:")
            lines += [f"  {name}" for name in data.reference_names]
        else:
            lines.append("Reference measurements: not stored in this result")

        return "\n".join(lines)

    @action(name="Show measurement details", group=transmission_res_grp_name)
    def show_measurement_details(self):
        if not self.eval_result_data.measurement_names:
            logging.info("No result loaded")
            return

        dialog = getattr(self, "_details_dialog", None)
        if dialog is None:
            dialog = QtWidgets.QDialog(self._ui_control_widget)
            dialog.setWindowTitle("Measurement details")
            layout = QtWidgets.QVBoxLayout(dialog)
            dialog.text_edit = QtWidgets.QPlainTextEdit(dialog)
            dialog.text_edit.setReadOnly(True)
            dialog.text_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
            layout.addWidget(dialog.text_edit)
            dialog.resize(900, 500)
            self._details_dialog = dialog

        dialog.text_edit.setPlainText(self.measurement_details())
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    @action(name="Show Q-space plot", group=transmission_res_grp_name)
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

            x_vals = np.array([res.d.magnitude for res in results], dtype=float)
            y_vals = np.array([res.optimization_info["q_val"].magnitude for res in results], dtype=float)
            sort_idx = np.argsort(x_vals)
            plt.plot(x_vals[sort_idx], y_vals[sort_idx], label=f"shift={shift}", marker="o", zorder=1)

        opt_res = self.eval_result_data.results.get(self.eval_result_data.optimum_results.get(sel_meas))
        if opt_res is not None:
            d_opt, shift_opt = np.round(opt_res.d, 3), np.round(opt_res.shift, 3)
            plt.scatter(opt_res.d.magnitude, opt_res.optimization_info["q_val"].magnitude, c="red",
                        label=f"Optimum (d={d_opt})", zorder=2)
            logging.info(f"Optimum thickness: {d_opt} (shift: {shift_opt})")

        plt.xlabel("Thickness (µm)")
        plt.ylabel("Q-value")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.show()

    def selected_result_key(self):
        selected = self.select_results(meas=self.measurement_list.selected_item,
                                       thickness=self.thicknesses.selected_item,
                                       shift=self.shifts.selected_item)
        if not isinstance(selected, SingleResultData):
            return None
        return next((key for key, res in self.eval_result_data.results.items()
                     if res is selected), None)

    @action(name="Save selected result", group=transmission_res_grp_name)
    def save_selected_result(self):
        if self.eval_result_data is None:
            return
        key = self.selected_result_key()
        if key is None:
            logging.warning("No result selected")
            return
        self.signals.save_requested.emit(self.eval_result_data, [key])

    def set_simple_traits(self, trait_values):
        trait_names = self.trait_names()
        for k, v in trait_values.items():
            if isinstance(v, (int, str, float, Q_, Path)) and k in trait_names:
                self.set_trait(k, v)

    def set_observers(self):
        def on_measurement_selection(change):
            selected_meas = change["new"]
            optimization_results = [res for res in self.eval_result_data.results.values()
                                    if res.measurement == selected_meas]
            thicknesses = sorted(list(set([res.d.magnitude for res in optimization_results])))
            shifts = sorted(list(set([res.shift.magnitude for res in optimization_results])))
            self.thicknesses.items = [str(d) for d in thicknesses]
            self.shifts.items = [str(shift) for shift in shifts]

            opt_res = self.eval_result_data.results.get(
                self.eval_result_data.optimum_results.get(selected_meas))
            if opt_res is not None:
                self.thicknesses.selected_item = str(opt_res.d.magnitude)
                self.shifts.selected_item = str(opt_res.shift.magnitude)
            elif self.thicknesses.items and self.shifts.items:
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
                fitted = []
                for name, val in selected_result.regression_params.items():
                    if isinstance(val, Q_):
                        fitted.append(FitParameter(name, unit=f"{val.units:~P}", label=parameter_label(name),
                                                   value=float(val.magnitude)))
                    elif isinstance(val, (int, float)):
                        fitted.append(FitParameter(name, label=parameter_label(name), value=float(val)))
                self.set_trait("fit_parameters", tuple(fitted))
                self.quantity_dict = QuantityDataSetDict(selected_result.datasets)

        self.thicknesses.observe(select_quantity_dict, "selected_item")
        self.shifts.observe(select_quantity_dict, "selected_item")

    def load_result(self, res_path):
        eval_result_data = self.parse_hdf5(res_path)
        self.parse_eval_result_data(eval_result_data)

        logging.info(f"Loaded {res_path}")

    def parse_hdf5(self, res_path) -> EvalResultData:
        def read_quantity(group, name):
            if name in group:
                dset = group[name]
                unit = dset.attrs.get("unit", "")
                return Q_(dset[()], unit)
            return None

        def hdf5group_to_quantity_dataset(hdf5_group):
            data_dset = hdf5_group["data"]
            data_q = Q_(data_dset[()], data_dset.attrs.get("unit", ""))
            data_label = data_dset.attrs.get("data_label", "")
            uncert_q = Q_(hdf5_group["uncert"][()], data_dset.attrs.get("unit", ""))

            axes_q, axes_labels = [], []
            axes_group = hdf5_group["axes"]
            i = 0
            while f"axis_{i}" in axes_group:
                ax_dset = axes_group[f"axis_{i}"]["axis_dset"]
                axes_q.append(Q_(ax_dset[()], ax_dset.attrs.get("unit", "")))
                axes_labels.append(ax_dset.attrs.get("axis_label", ""))
                i += 1

            return QuantityDataSet(
                data=data_q,
                uncert=uncert_q,
                axes=axes_q,
                data_label=data_label,
                axes_labels=axes_labels
            )

        with h5py.File(res_path, "r") as f:
            eval_result_data = EvalResultData(
                result_type=f.attrs.get("result_type", ""),
                dataset_path=Path(f.attrs.get("dataset_path", ".")),
                model_name=f.attrs.get("model_name", ""),
                measurement_quantity=f.attrs.get("measurement_quantity", ""),
                measurement_names=list(f.attrs.get("measurement_names", [])),
                reference_names=list(f.attrs.get("reference_names", [])),
                optimum_results=dict(f["optimum_results"].attrs) if "optimum_results" in f else {}
            )

            results_grp = f.get("results", {})
            for res_key in results_grp.keys():
                res_subgroup = results_grp[res_key]
                single_res = SingleResultData(
                    measurement=res_subgroup.attrs.get("measurement", ""),
                    d=read_quantity(res_subgroup, "d"),
                    shift=read_quantity(res_subgroup, "shift"),
                    freq_axis=read_quantity(res_subgroup, "freq_axis")
                )

                reg_grp = res_subgroup.get("regression_params", {})
                for k in reg_grp.keys():
                    single_res.regression_params[k] = read_quantity(reg_grp, k)
                for k, v in reg_grp.attrs.items():
                    single_res.regression_params[k] = v

                opt_grp = res_subgroup.get("optimization_info", {})
                for k in opt_grp.keys():
                    single_res.optimization_info[k] = read_quantity(opt_grp, k)
                for k, v in opt_grp.attrs.items():
                    single_res.optimization_info[k] = v

                ds_grp = res_subgroup.get("datasets", {})
                for ds_name in ds_grp.keys():
                    single_res.datasets[ds_name] = hdf5group_to_quantity_dataset(ds_grp[ds_name])

                eval_result_data.results[res_key] = single_res

        return eval_result_data

    def parse_eval_result_data(self, eval_result_data: EvalResultData):
        if not eval_result_data:
            return
        self.eval_result_data = eval_result_data
        self.set_simple_traits(asdict(eval_result_data))
        self.set_trait("measurement_info", self._measurement_summary())
        self.measurement_list.items = list(set([res.measurement for res in eval_result_data.results.values()]))
        self.measurement_list.selected_item = None
        if self.measurement_list.items:
            self.measurement_list.selected_item = self.measurement_list.items[0]

        transmission_traits = list(self.traits(group=self.transmission_res_grp_name))
        if eval_result_data.result_type == "Regression":
            transmission_only = ["q_val", "fp_time", "q_space_f_max"]
            self.toggle_traits([t for t in transmission_traits if t not in transmission_only],
                               group_filter=self.transmission_res_grp_name)
            self.toggle_traits(list(self.traits(group=self.reg_result_grp_name)),
                               group_filter=self.reg_result_grp_name)
        elif eval_result_data.result_type == "Transmission fit":
            self.toggle_traits(transmission_traits, group_filter=self.transmission_res_grp_name)
            self.toggle_traits([], group_filter=self.reg_result_grp_name)
        elif eval_result_data.result_type == "Conductivity":
            hidden = ["q_val", "fp_time", "q_space_f_max", "gof", "nfev"]
            self.toggle_traits([t for t in transmission_traits if t not in hidden],
                               group_filter=self.transmission_res_grp_name)
            self.toggle_traits([], group_filter=self.reg_result_grp_name)
        elif eval_result_data.result_type == "Sparse deconvolution":
            hidden = ["q_val", "fp_time", "q_space_f_max", "gof", "nfev"]
            self.toggle_traits([t for t in transmission_traits if t not in hidden],
                               group_filter=self.transmission_res_grp_name)
            self.toggle_traits(list(self.traits(group=self.reg_result_grp_name)),
                               group_filter=self.reg_result_grp_name)
