import traceback
import threading
from dataclasses import replace
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import shgo
import numpy as np
import logging
from common.components import action
from common.measurements import Measurement
from common.eval_component.quantity_set import QuantityDataSet
from common.eval_component.conductivity_models import (RegressionModels, make_expression_model, parameter_label,
                                                       sigma_to_n)
from common.eval_component.model_latex import expression_to_latex, identifier_label
from common.eval_component.eval_result import EvalResultData, SingleResultData
from common.traits import Q_, FitParameter, FitParameterTable
from traitlets import Enum as TEnum, Unicode, observe, Bool
from common.default_appsettings import QuantityEnum
from common.eval_component.eval_component_base import EvalComponentBase


class Regression(EvalComponentBase):
    activity = Unicode("Idle", read_only=True).tag(name="Status", group=EvalComponentBase.process_status,
                                                   priority=-1, en_save=False)

    reg_grp_name = "Regression"
    sel_meas_quantity_enum = TEnum(QuantityEnum, default_value=QuantityEnum.TransmissionAmp,
                                   group=reg_grp_name).tag(name="Measurement quantity", priority=0)
    selected_reg_model = TEnum(RegressionModels, default_value=RegressionModels.drude,
                               group=reg_grp_name).tag(name="Selected regression model", priority=1)
    model_grp_name = "Model"
    custom_expression = Unicode("sig0 / (1 - 1j * 2 * pi * freq * tau * 1e-3)", group=model_grp_name,
                                help="Python expression of the model quantity (Ctrl+Enter or ✓ to apply, line "
                                     "breaks and # comments are allowed). freq is the frequency axis in THz, "
                                     "every other name is a free parameter, its value is passed in the unit set in "
                                     "the parameter table. Names can also be written as latex symbols, e.g. "
                                     r"\sigma_0, \omega_{p}, \varepsilon_\infty (stored as sigma_0, omega_p, "
                                     r"varepsilon_infty), \pi is the constant. Available: np, pi, c_thz (µm/ps), eps0_thz (S ps/µm), "
                                     "sqrt, exp, log, log10, sin, cos, tan, arctan, sinh, cosh, tanh, abs, real, imag, "
                                     "conj, angle and the built-in models (drude, drude2, drude_smith, "
                                     "lattice_contrib, total_response, n_to_sigma)"
                                ).tag(name="Model expression", priority=0, multiline=False, fullwidth=True,
                                      disable_label=True)
    model_latex = Unicode("", read_only=True, group=model_grp_name).tag(name="Model formula", priority=1, latex=True,
                                                                        disable_label=True, en_save=False)
    convert_sigma_to_t = Bool(default_value=False, group=reg_grp_name,
                              help="Calculate sigma -> 2 layer model to calculate t").tag(name="Fit to transmission",
                                                                                         priority=3)

    param_grp_name = "Fit parameters"
    fit_parameters = FitParameterTable(group=param_grp_name,
                                       help="Free parameters of the selected model. Bounds are given in the unit "
                                            "of the parameter, the fit result is reported in the same unit"
                                       ).tag(name="Fit parameters", disable_label=True)

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self._param_memory: dict[str, FitParameter] = {}
        self.update_fit_parameters()
        self.update_model_latex()

    @property
    def _ui_control_widget(self):
        return getattr(self, '_ui_widget_internal', None)

    @_ui_control_widget.setter
    def _ui_control_widget(self, value):
        self._ui_widget_internal = value
        if value:
            self._toggle_expression()

    def _toggle_expression(self):
        active = [name for name in self.traits(group=self.model_grp_name)
                  if name != "custom_expression" or self.is_custom_model]
        self.toggle_traits(active, group_filter=self.model_grp_name)

    @observe("selected_reg_model", "custom_expression", "convert_sigma_to_t")
    def update_model_latex(self, change=None):
        if self.is_custom_model:
            try:
                lines = [expression_to_latex(self.custom_expression,
                                             lhs=r"\sigma(f)" if self.convert_sigma_to_t else "y(f)")]
            except SyntaxError:
                lines = []
            except ValueError as e:
                logging.warning(f"No formula preview: {e}")
                lines = []
        else:
            lines = list(self.selected_reg_model.value.latex)

        if self.convert_sigma_to_t and lines:
            lines.insert(0, r"t(f) = T\left(n(\sigma(f)), d\right)")

        self.set_trait("model_latex", "\n".join(lines))

    @property
    def is_custom_model(self):
        return self.selected_reg_model == RegressionModels.custom

    @observe("fit_parameters")
    def _remember_parameters(self, change):
        self._param_memory.update({p.name: p for p in change["new"]})

    @observe("selected_reg_model", "custom_expression")
    def update_fit_parameters(self, change=None):
        if self.is_custom_model:
            try:
                _, names = make_expression_model(self.custom_expression)
            except SyntaxError as e:
                logging.error(f"Invalid model expression: {e}")
                return
            defaults = [FitParameter(name, label=parameter_label(name) or identifier_label(name)) for name in names]
        else:
            defaults = self.selected_reg_model.value.parameters

        parameters = []
        for default in defaults:
            remembered = self._param_memory.get(default.name)
            if remembered is None or not self._units_compatible(remembered.unit, default.unit, default):
                parameters.append(default)
            else:
                parameters.append(replace(remembered, label=default.label, value=None))

        self.fit_parameters = tuple(parameters)
        self._toggle_expression()

    def _units_compatible(self, unit, default_unit, default):
        if self.is_custom_model:
            return True
        try:
            Q_(1, unit).to(default_unit)
            return True
        except Exception:
            return False

    def get_model(self):
        if self.is_custom_model:
            func, names = make_expression_model(self.custom_expression)
            factors = {p.name: 1.0 for p in self.fit_parameters}
            if set(names) != set(factors):
                raise ValueError("Parameter table does not match the model expression")
        else:
            func = self.selected_reg_model.value.func
            model_units = {p.name: p.unit for p in self.selected_reg_model.value.parameters}
            factors = {p.name: Q_(1, p.unit).to(model_units[p.name]).magnitude for p in self.fit_parameters}

        names = [p.name for p in self.fit_parameters]

        def reg_model(freq, p):
            return func(freq, **{name: val * factors[name] for name, val in zip(names, p)})

        if not self.convert_sigma_to_t:
            return reg_model

        t_mod_kwargs = self.get_t_model_kwargs()
        t_mod_kwargs["d"] = self.settings.eval_opt.d.magnitude
        transmission_model = self.transmission_model.value

        def t_model(freq, p):
            n = sigma_to_n(freq, reg_model(freq, p))
            return transmission_model(n, freq, **t_mod_kwargs)

        return t_model

    def get_bounds(self):
        bounds = [(p.lower, p.upper) for p in self.fit_parameters]
        for p, (lower, upper) in zip(self.fit_parameters, bounds):
            if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
                raise ValueError(f"Invalid bounds for parameter {p.display_name}: [{lower}, {upper}]")

        return bounds

    @property
    def y_meas_dict(self):
        selection = self.dataset.measurement_selector.get_selection()
        if selection is None:
            return None
        meas_list = list(selection.sams)
        meas_quant_func = self.dataset.quantity_func(self.sel_meas_quantity_enum)
        y_meas = meas_quant_func(meas_list)

        return self.format_meas_dict(meas_list, y_meas)

    @action("Fit regression model", group=reg_grp_name, priority=10)
    def perform_regression(self):
        if self._active_eval is not None:
            logging.warning("Regression is already running")
            return

        y_meas_dict = self.y_meas_dict
        if not y_meas_dict:
            logging.warning("No measurements selected")
            return

        try:
            model = self.get_model()
            bounds = self.get_bounds()
            if not bounds:
                raise ValueError("Model has no free parameters")
        except Exception as e:
            logging.error(f"Invalid regression model: {e}")
            return

        freq_axis = self.freq_axis
        cost_func = self.selected_cost_fun.value
        fit_parameters = self.fit_parameters
        quantity = self.sel_meas_quantity_enum
        shgo_options = self.settings.shgo_options
        minimizer_kwargs = {**shgo_options.get_minimizer_kwargs(),
                            "method": str(shgo_options.minimizer_kwargs.method.value)}

        eval_result_data = EvalResultData()
        eval_result_data.result_type = "Regression"
        eval_result_data.dataset_path = self.dataset.data_path
        eval_result_data.model_name = self.selected_reg_model.name
        eval_result_data.measurement_quantity = quantity.name
        eval_result_data.measurement_names = [str(meas) if not isinstance(meas, Measurement) else meas.filepath.name
                                              for meas in y_meas_dict]

        common_info = {"model_expression": self.custom_expression if self.is_custom_model else "",
                       "fit_to_transmission": self.convert_sigma_to_t}

        self._cancel_event = threading.Event()
        evaluator = self
        self._active_eval = evaluator
        self.task_cnt_carrier.cnt_changed.emit(len(y_meas_dict))

        def bg_worker():
            try:
                for meas_idx, (meas, y_meas) in enumerate(y_meas_dict.items()):
                    if self._cancel_event.is_set():
                        logging.info("Regression cancelled")
                        return
                    y_vals = y_meas[:, 1]

                    def opt_fun(p):
                        return np.sum(cost_func(y_vals, model(freq_axis, p)))

                    opt_res = shgo(opt_fun, bounds=bounds, n=shgo_options.n, iters=shgo_options.iters,
                                   minimizer_kwargs=minimizer_kwargs, options=shgo_options.get_shgo_options())

                    single_res = self.prepare_result(meas, y_meas, model(freq_axis, opt_res.x), opt_res,
                                                     fit_parameters, quantity)
                    single_res.optimization_info.update(common_info)

                    res_key = f"result_{meas_idx:03d}"
                    eval_result_data.results[res_key] = single_res
                    eval_result_data.optimum_results[single_res.measurement] = res_key

                    params_str = ", ".join(f"{p.display_name}={v:.4g} {p.unit}".rstrip()
                                           for p, v in zip(fit_parameters, opt_res.x))
                    logging.info(f"Fit result {single_res.measurement}: {params_str} (cost: {opt_res.fun:.4g})")
                    self.task_cnt_carrier.cnt_changed.emit(1)

                self._result_ready(eval_result_data)
            except Exception:
                traceback.print_exc()
            finally:
                if self._active_eval is evaluator:
                    self._active_eval = None

        if self._thread_executor is None:
            self._thread_executor = ThreadPoolExecutor(max_workers=1)
        self._thread_executor.submit(bg_worker)

    def prepare_result(self, meas, y_meas, y_mod, opt_res, fit_parameters, quantity) -> SingleResultData:
        freq_axis = Q_(y_meas[:, 0].real, "THz")
        unit = quantity.value.unit

        res = SingleResultData()
        res.measurement = str(meas) if not isinstance(meas, Measurement) else meas.filepath.name
        res.freq_axis = freq_axis
        res.regression_params = {p.name: Q_(val, p.unit) for p, val in zip(fit_parameters, opt_res.x)}
        res.optimization_info = {
            "fun": float(opt_res.fun),
            "gof": Q_(opt_res.fun / len(freq_axis), ""),
            "nit": int(opt_res.nit),
            "nfev": int(opt_res.nfev),
            "converged": bool(opt_res.success),
            "timestamp": str(datetime.now().isoformat()),
        }

        datasets = {"y_meas": (y_meas[:, 1], y_meas[:, 2], "Measured"),
                    "y_mod": (y_mod, np.zeros_like(y_mod), "Model")}
        for key, (data, uncert, label) in datasets.items():
            label = f"{quantity.value.label} ({label})"
            if np.iscomplexobj(data) and np.any(np.imag(data)):
                parts = {"real": np.real, "imag": np.imag}
            else:
                parts = {"": np.real}
            for part, part_func in parts.items():
                res.datasets[f"{key}_{part}" if part else key] = QuantityDataSet(
                    axes=[freq_axis],
                    data=Q_(part_func(data), unit),
                    uncert=Q_(part_func(uncert), unit),
                    axes_labels=["Frequency"],
                    data_label=f"{label} ({part.capitalize()})" if part else label)

        return res
