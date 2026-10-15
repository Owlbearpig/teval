import traceback
from datetime import datetime
from common.dataset import DataSet, Domain, format_meas_dict
from common.components import ComponentBase, action
from common.functions import f_axis_idx_map
from common.eval_component.shgo import shgo
from scipy.optimize import shgo
from functools import partial
import numpy as np
import logging
from common.eval_component.quantity_set import QuantityDataSet
from enum import Enum, member
from common.eval_component.conductivity_models import RegressionModels, model_params
from common.traits import Quantity, Q_, ValueRange, Path as TPath
from pathlib import Path
from traitlets import Enum as TEnum, observe, Integer, Float, Bool, Instance
from common.default_appsettings import QuantityEnum
from common.eval_component.shgo_settings import MinimizerOptions
from concurrent.futures import ThreadPoolExecutor
import threading
from common.eval_component.eval_component_base import EvalComponentBase



class Regression(EvalComponentBase):
    reg_grp_name = "Regression"
    sel_meas_quantity_enum = TEnum(QuantityEnum, default_value=QuantityEnum.TransmissionAmp,
                                   group=reg_grp_name).tag(name="Measurement quantity")
    selected_reg_model = TEnum(RegressionModels, default_value=RegressionModels.drude,
                               group=reg_grp_name).tag(name="Selected regression model")
    convert_sigma_to_t = Bool(default_value=False, group=reg_grp_name,
                              help="Calculate sigma -> 2 layer model to calculate t").tag(name="Fit to transmission")
    sig0_bounds = ValueRange([Q_(10, "S/cm"), Q_(20, "S/cm")], group=reg_grp_name).tag(name="σ₀ Bounds")
    tau_bounds = ValueRange([Q_(10, "fs"), Q_(1000, "fs")], group=reg_grp_name).tag(name="τ Bounds")
    wp_bounds = ValueRange([Q_(-10, "THz"), Q_(100, "THz")], group=reg_grp_name).tag(name="ωₚ Bounds")
    eps_inf_bounds = ValueRange([1.0, 100.0], group=reg_grp_name).tag(name="ε_inf Bounds")
    eps_s_bounds = ValueRange([1.0, 100.0], group=reg_grp_name).tag(name="ε_s Bounds")
    c1_bounds = ValueRange([-1.0, 1.0], group=reg_grp_name).tag(name="c₁ Bounds")

    def __init__(self, dataset: "DataSet", **kwargs):
        super().__init__(dataset, **kwargs)

    @property
    def _ui_control_widget(self):
        return getattr(self, '_ui_widget_internal', None)

    @_ui_control_widget.setter
    def _ui_control_widget(self, value):
        self._ui_widget_internal = value
        if value:
            self.select_regression_model()

    @property
    def y_meas_dict(self):
        selection = self.dataset.measurement_selector.get_selection()
        if selection is None:
            return None
        meas_list = selection.sams
        meas_quant_func = self.dataset.func_map[self.sel_meas_quantity_enum]
        y_meas = meas_quant_func(meas_list)

        y_sub = y_meas[:, self.freq_range_idx]

        freq_grid = np.tile(self.freq_axis, (y_sub.shape[0], 1))
        y_meas_stacked = np.stack((freq_grid, y_sub, np.zeros_like(y_sub)), axis=2)
        return format_meas_dict(meas_list, y_meas_stacked, self.only_use_avg)

    @property
    def _opt_conf(self):
        bounds, bounds_units = self.get_bounds()
        selection = self.dataset.measurement_selector.get_selection()
        if selection is None:
            return None
        conf_dict = {"measurements": {meas.identifier: meas for meas in selection.sams},
                     "freq_axis": self.freq_axis,
                     "meas_quantity": self.sel_meas_quantity_enum,
                     "y_meas_dict": self.y_meas_dict,
                     "model": self.set_reg_model_freq,
                     "model_name": self.selected_reg_model.name,
                     "bounds": bounds,
                     "bounds_units": bounds_units,
                     "opt_func_dict": self.opt_func_dict,
                     }

        if self.dataset.sub_dataset is not None:
            sub_dataset_path = self.dataset.sub_dataset.data_path
        else:
            sub_dataset_path = Path(".")

        conf_dict["sub_dataset_path"] = sub_dataset_path
        conf_dict["dataset_path"] = self.dataset.data_path

        return conf_dict

    @property
    def opt_func_dict(self):
        reg_mod = self.set_reg_model_freq
        cost_func = self.selected_cost_fun.value

        t_mod_kwargs = self.get_t_model_kwargs()
        t_mod_kwargs["d"] = self.settings.eval_opt.d.magnitude

        def t_mod_func(*args, **kwargs):
            sigma = reg_mod(*args, **kwargs)
            n = self.dataset.sigma_to_n(self.freq_axis, sigma)

            t = self.transmission_model.value(n, self.freq_axis, **t_mod_kwargs)

            return t

        mod_func = t_mod_func if self.convert_sigma_to_t else reg_mod

        y_dict = self.y_meas_dict
        return {
            meas_id: (lambda params, m_id=meas_id: np.sum(cost_func(y_dict[m_id], mod_func(*params))))
            for meas_id in y_dict
        }


    @property
    def set_reg_model_freq(self):
        model_func = self.selected_reg_model.value
        return partial(model_func, self.freq_axis)

    @observe("selected_reg_model")
    def select_regression_model(self, change=None):
        reg_model = self.selected_reg_model if change is None else change["new"]

        param_keys = model_params(reg_model.name)
        active_bounds = [f"{p}_bounds" for p in param_keys if p not in ['freq', 'freq_']]
        self.toggle_traits(active_bounds, group_filter=self.reg_grp_name, endswith_filter="_bounds")


    def get_bounds(self):
        params = model_params(self.selected_reg_model.name)
        bounds, units = [], []
        for p in params:
            bound = getattr(self, p + "_bounds", None)
            if bound is None:
                min_value, max_value = -np.inf, np.inf
                units.append("")
            else:
                min_value = bound[0].magnitude if isinstance(bound[0], Q_) else bound[0]
                max_value = bound[1].magnitude if isinstance(bound[1], Q_) else bound[1]
                units.append(bound[0].units if isinstance(bound[0], Q_) else "")

            bounds.append([min_value, max_value])

        return bounds, units

    @action("Fit regression model", group=reg_grp_name)
    def perform_regression(self):
        if self.dataset.measurement_selector.selected_sam_cnt == "0":
            logging.warning("No measurements selected")
            return
        opt_conf = self._opt_conf
        shgo_options = self.settings.shgo_options

        def bg_worker(meas_id):
            try:
                min_kwargs = shgo_options.minimizer_kwargs.traits(group=MinimizerOptions.minimizer_opt_grp)
                min_kwargs["method"] = str(shgo_options.minimizer_kwargs.method.value)

                shgo_opt_res = shgo(func=opt_conf["opt_func_dict"][meas_id],
                                    bounds=opt_conf["bounds"],
                                    n=shgo_options.n,
                                    iters=shgo_options.iters,
                                    minimizer_kwargs=min_kwargs,
                                    options=shgo_options.get_shgo_options(),
                                    )
                logging.info(f"Fit result {meas_id}: {shgo_opt_res}")

                reg_res = self.prepare_regression_result(shgo_opt_res, opt_conf, meas_id)

                self.current_result.result_carrier.received_result.emit(reg_res)
            except Exception:
                traceback.print_exc()

        executor = ThreadPoolExecutor(max_workers=self.number_of_workers)
        for meas_key in opt_conf["y_meas_dict"].keys():
            executor.submit(bg_worker, meas_key)
        executor.shutdown(wait=False)

    def prepare_regression_result(self, shgo_opt_res, opt_conf, meas_id):
        model_name = opt_conf["model_name"]
        x = shgo_opt_res.x

        opt_res_dict = {
            "measurement": str(opt_conf["measurements"].get(meas_id, "Average")),
            "model_name": model_name,
            "result_type": "Regression",
            "sub_dataset_path": opt_conf["sub_dataset_path"],
            "dataset_path": opt_conf["dataset_path"],
        }
        for param_idx, p in enumerate(model_params(model_name)):
            unit = opt_conf["bounds_units"][param_idx]
            if not unit:
                opt_res_dict[p] = x[param_idx]
            else:
                opt_res_dict[p] = Q_(x[param_idx], unit)

        opt_res_dict["nit"] = shgo_opt_res.nit
        opt_res_dict["fun"] = shgo_opt_res.fun
        opt_res_dict["converged"] = shgo_opt_res.success
        opt_res_dict["timestamp"] = str(datetime.now().isoformat())

        freq_axis = Q_(opt_conf["freq_axis"], "THz")
        unit = opt_conf["meas_quantity"].value.unit
        y_meas_data = opt_conf["y_meas_dict"][meas_id][:, 1]

        opt_res_dict["y_meas"] = QuantityDataSet(axes=[freq_axis],
                                                 data=Q_(y_meas_data, unit),
                                                 axes_labels=["Frequency"],
                                                 data_label=opt_conf["meas_quantity"].name)
        opt_res_dict["y_mod"] = QuantityDataSet(axes=[freq_axis],
                                                data=Q_(opt_conf["model"](*x), unit),
                                                axes_labels=["Frequency"],
                                                data_label=opt_conf["meas_quantity"].name)

        return opt_res_dict


if __name__ == "__main__":

    print(RegressionModels.drude.name)
