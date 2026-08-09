import traceback
from datetime import datetime
from common.dataset import DataSet, Domain
from common.components import ComponentBase, action
from common.datasetplotter import DataSetPlotter
from common.functions import f_axis_idx_map
from common.eval_component.shgo import shgo
from scipy.optimize import shgo
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import logging
from common.consts import eps0_thz
from common.eval_component.q_space_eval import QSpaceEval
from common.eval_component.quantity_set import DataSet as SingleQuantityDataSet
from enum import Enum, member
from common.eval_component.conductivity_models import RegressionModels, model_params
from common.traits import Quantity, Q_, ValueRange, Path as TPath
from pathlib import Path
from traitlets import Enum as TEnum, observe, Integer, Float, Bool, Instance
from common.default_appsettings import QuantityEnum
from common.eval_component.transfer_functions import (t_tmm_model_1layer, model_1layer, t_tmm_model_2layer,
                                                      model_2layer, _t_model_2layer)
from common.eval_component.shgo_settings import SHGOOptions, MinimizerOptions
from common.save import ResultSaver
from common.eval_component.eval_result import EvalResult
from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import QObject, Signal

action = partial(action, check_init=True)

class ProgressSignalCarrier(QObject):
    progress_changed = Signal(float)

def abs_cost_fun(y_meas, y_mod):

    abs_diff = (np.abs(y_meas) - np.abs(y_mod)) ** 2

    return np.sum(abs_diff)


def phi_cost_fun(y_meas, y_mod):
    phi_diff = (np.angle(y_meas) - np.angle(y_mod)) ** 2

    return np.sum(phi_diff)


def combined_cost_fun(y_meas, y_mod):
    return abs_cost_fun(y_meas, y_mod) + phi_cost_fun(y_meas, y_mod)

class TransmissionModels(Enum):
    tmm_1layer = member(t_tmm_model_1layer)
    tmm_2layer = member(t_tmm_model_2layer)
    model_1layer = member(model_1layer)
    model_2layer = member(model_2layer)
    t_model_2layer = member(_t_model_2layer)

class CostFunctions(Enum):
    abs_cost = member(abs_cost_fun)
    phi_cost = member(phi_cost_fun)
    combined_cost = member(combined_cost_fun)

class DataSetType(Enum):
    Main = "main"
    Sub = "sub"
    Other = "other"

class DatasetEval(ComponentBase):

    shgo_options = Instance(SHGOOptions)

    sel_point = ValueRange(default_value=[Q_(0.0, "mm"), Q_(0.0, "mm")]).tag(name="Selected point (x, y)")
    selected_cost_fun = TEnum(CostFunctions, default_value=CostFunctions.abs_cost,
                              help="Model to experimental data metric").tag(name="Selected cost function")
    selected_result_path = TPath(Path("")).tag(name="Load result")
    selected_substrate_result_path = TPath(Path("")).tag(name="Substrate result")
    optimization_progress = Float(0, min=0, max=1,
                                  group="General", read_only=True).tag(name="Progress")

    reg_grp_name = "Regression"
    selected_meas_quantity = TEnum(QuantityEnum, default_value=QuantityEnum.TransmissionAmp,
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

    t_fit_grp_name = "Transmission q-space fit"
    transmission_model = TEnum(TransmissionModels, default_value=TransmissionModels.tmm_1layer,
                               group=t_fit_grp_name).tag(name="Selected transmission model")
    d_opt_axis_bounds = ValueRange([Q_(500, "µm"), Q_(580, "µm", )],
                                   group=t_fit_grp_name).tag(name="Custom thickness axis bounds")
    d_opt_axis_step = Quantity(Q_(10, "µm"), group=t_fit_grp_name).tag(name="Custom thickness axis step")
    use_custom_d_opt_axis = Bool(True, group=t_fit_grp_name).tag(name="Use custom thickness axis")
    number_of_workers = Integer(8, group=t_fit_grp_name).tag(name="Number of cpu cores to assign")
    add_sim_to_res = Bool(False, group=t_fit_grp_name).tag(name="Add simulated t to result")

    current_result = Instance(EvalResult)
    selected_substrate_result = Instance(EvalResult)
    result_saver = Instance(ResultSaver)

    def __init__(self, dataset: DataSet, dataset_sub: DataSet=None, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.dataset.link_sub_dataset(dataset_sub)

        self.shgo_options = SHGOOptions()

        self.current_result = EvalResult(object_name="Current result")
        self.selected_substrate_result = EvalResult(object_name="Substrate result")

        self.result_saver = self.setup_saver()

        self.current_result.result_carrier.result_ready.connect(self.result_saver.process)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.settings is not None:
            self.settings.save_configuration(self)

    def __enter__(self):
        if self.settings is not None:
            self.settings.load_configuration(self)
        return self
    
    @property
    def settings(self):
        return self.dataset.settings
    
    @property
    def freq_axis(self):
        f_axis = self.dataset.freq_axis[self.f_idx]
        return f_axis

    @property
    def f_idx(self):
        return f_axis_idx_map(self.dataset.freq_axis, self.settings.eval_opt.fit_range)

    @property
    def meas_quantity(self):
        meas_quantity = self.selected_meas_quantity
        meas_quantity.value.func = self.dataset.func_map[meas_quantity]

        return meas_quantity

    @property
    def y_meas(self):
        meas = self.dataset.get_measurement_from_point(*self.sel_point)
        y_meas = self.meas_quantity.value.func(meas)

        y_meas = y_meas[self.f_idx]

        return y_meas

    @property
    def reg_model_freq_set(self):
        model_func = self.selected_reg_model.value
        return partial(model_func, self.freq_axis)

    @property
    def opt_func(self):
        reg_mod = self.reg_model_freq_set
        cost_func = self.selected_cost_fun.value

        t_mod_kwargs = self.get_t_model_kwargs()
        t_mod_kwargs["d"] = self.settings.eval_opt.d.magnitude

        def t_mod_func(*args, **kwargs):
            sigma = reg_mod(*args, **kwargs)
            n = self.sigma_to_n(self.freq_axis, sigma)

            t = self.transmission_model.value(n, self.freq_axis, **t_mod_kwargs)

            return t

        mod_func = t_mod_func if self.convert_sigma_to_t else reg_mod

        return lambda p: cost_func(self.y_meas, mod_func(*p))

    @property
    def _opt_conf(self):
        bounds, bounds_units = self.get_bounds()
        conf_dict = {
            "freq_axis": self.freq_axis,
            "meas_quantity": self.meas_quantity,
            "y_meas": self.y_meas,
            "model": self.reg_model_freq_set,
            "model_name": self.selected_reg_model.name,
            "bounds": bounds,
            "bounds_units": bounds_units,
            "opt_func": self.opt_func,
        }

        conf_dict["h"] = self.settings.eval_opt.d_film.magnitude

        if self.dataset.sub_dataset is not None:
            sub_dataset_path = self.dataset.sub_dataset.data_path
        else:
            sub_dataset_path = Path(".")

        conf_dict["sub_dataset_path"] = sub_dataset_path
        conf_dict["dataset_path"] = self.dataset.data_path

        return conf_dict

    @property
    def _ui_control_widget(self):
        return getattr(self, '_ui_widget_internal', None)

    @_ui_control_widget.setter
    def _ui_control_widget(self, value):
        self._ui_widget_internal = value
        if value:
            self.select_regression_model()

    @observe("selected_reg_model")
    def select_regression_model(self, change=None):
        reg_model = self.selected_reg_model if change is None else change["new"]

        param_keys = model_params(reg_model.name)
        active_bounds = [f"{p}_bounds" for p in param_keys if p not in ['freq', 'freq_']]
        self.toggle_traits(active_bounds, group_filter=self.reg_grp_name, endswith_filter="_bounds")

    @observe("selected_result_path")
    def set_selected_result(self, change):
        result_path = change["new"]
        self.current_result.load_result(result_path)

    @observe("selected_substrate_result_path")
    def set_substrate_result(self, change):
        result_path = change["new"]
        self.selected_substrate_result.load_result(result_path)

    def setup_saver(self):
        res_saver = ResultSaver()
        res_saver.registerObjectAttribute(self, "sel_point")
        res_saver.registerObjectAttribute(self.current_result, "result_type")

        return res_saver

    def update_progress(self, progress_value):
        self.set_trait("optimization_progress", progress_value)

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

    def is_two_layer_t_model(self, model_kwargs):
        try:
            # check if transmission model expects n_sub
            test_kwargs = model_kwargs.copy()
            test_kwargs["d"] = 1
            test_kwargs["shift"] = 0
            test_kwargs["n_sub"] = 1
            if "h" in test_kwargs:
                test_kwargs.pop("h")
            self.transmission_model.value(n=1, freq=1, **test_kwargs)
            return False
        except KeyError:
            return True

    def get_t_model_kwargs(self):
        single_layer_properties = self.dataset.get_single_layer_properties()
        meas = single_layer_properties["meas"]
        ref_meas = self.dataset.get_nearest_ref(meas)

        meas_quants = self.dataset.calc_meas_quantities(ref_meas, meas)
        model_kwargs = {}
        model_kwargs["meas_quants"] = meas_quants
        model_kwargs["single_layer_approx"] = single_layer_properties["single_layer_approx"]
        model_kwargs["nfp"] = self.settings.eval_opt.fp_count
        model_kwargs["n1"] = 1
        model_kwargs["n4"] = 1
        model_kwargs["shift"] = 0

        if self.is_two_layer_t_model(model_kwargs):
            model_kwargs["h"] = self.settings.eval_opt.d_film.magnitude
            substrate_result = self.selected_substrate_result.quantity_dict
            try:
                n_sub_real_dataset = substrate_result["n"]
                n_sug_imag_dataset = substrate_result["k"]
                n_sub = n_sub_real_dataset.data.magnitude + 1j * n_sug_imag_dataset.data.magnitude
            except KeyError:
                raise Exception("Substrate result required for two layer model optimization")
            model_kwargs["n_sub"] = n_sub
            if np.isclose(np.sum(n_sub_real_dataset.data.axis[0]-self.freq_axis), 0):
                raise Exception("Frequency axis must equal substrate result frequency axis")

        return model_kwargs

    def prepare_regression_result(self, opt_res, opt_conf):
        model_name = opt_conf["model_name"]
        x = opt_res.x

        opt_res_dict = {"model_name": model_name,
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

        opt_res_dict["nit"] = opt_res.nit
        opt_res_dict["fun"] = opt_res.fun
        opt_res_dict["converged"] = opt_res.success
        opt_res_dict["timestamp"] = str(datetime.now().isoformat())

        freq_axis = Q_(opt_conf["freq_axis"], "THz")
        unit = opt_conf["meas_quantity"].value.unit
        opt_res_dict["y_meas"] = SingleQuantityDataSet(axes=[freq_axis], data=Q_(opt_conf["y_meas"], unit),
                                                       axes_labels=["Frequency"],
                                                       data_label=opt_conf["meas_quantity"].name)
        opt_res_dict["y_mod"] = SingleQuantityDataSet(axes=[freq_axis], data=Q_(opt_conf["model"](*x), unit),
                                                      axes_labels=["Frequency"],
                                                      data_label=opt_conf["meas_quantity"].name)

        return opt_res_dict

    @action("Fit regression model", group=reg_grp_name)
    def perform_regression(self):
        opt_conf = self._opt_conf
        shgo_options = self.shgo_options
        def bg_worker():
            try:
                min_kwargs = shgo_options.minimizer_kwargs.traits(group=MinimizerOptions.minimizer_opt_grp)
                min_kwargs["method"] = str(shgo_options.minimizer_kwargs.method.value)

                opt_res_ = shgo(func=opt_conf["opt_func"],
                                bounds=opt_conf["bounds"],
                                n=shgo_options.n,
                                iters=shgo_options.iters,
                                minimizer_kwargs=min_kwargs,
                                options=shgo_options.get_shgo_options(),
                                )
                logging.info("Fit result: {}".format(opt_res_))

                reg_res = self.prepare_regression_result(opt_res_, opt_conf)

                self.current_result.result_carrier.received_result.emit(reg_res)
            except Exception:
                traceback.print_exc()

        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(bg_worker)

    @action("Fit transmission model", group=t_fit_grp_name)
    def fit_unknown_layer(self):
        try:
            progress_carrier = ProgressSignalCarrier()
            progress_carrier.progress_changed.connect(self.update_progress)

            def bg_worker():
                qs_eval = QSpaceEval(self)
                qs_res = qs_eval.q_space_eval_mp(progress_carrier=progress_carrier)

                self.current_result.result_carrier.received_result.emit(qs_res)
            executor = ThreadPoolExecutor(max_workers=1)
            executor.submit(bg_worker)
        except Exception as e:
            traceback.print_exc()

    def sigma_to_n(self, freq, sigma):
        w = 2 * np.pi * freq

        sigma *= 1e-4 # S/cm -> S/µm
        n_ = (1 + 1j) * np.sqrt(sigma/(2*w*eps0_thz))

        return n_

if __name__ == "__main__":

    print(RegressionModels.drude.name)


