import traceback
from datetime import datetime
from common.dataset import DataSet, Domain
from common.components import ComponentBase, action
from common.datasetplotter import DataSetPlotter
from common.functions import window, do_ifft, phase_correction, f_axis_idx_map
from common.eval_component.shgo import shgo
from scipy.optimize import shgo
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import logging
import inspect
from scipy.signal import iirnotch, filtfilt
from common.eval_component.q_space_eval import QSpaceEval
from common.eval_component.quantity_set import DataSet as SingleQuantityDataSet
from common.eval_component.conductivity_models import RegressionModels, model_params
from common.traits import Quantity, Q_, ValueRange, Path as TPath
from pathlib import Path
from traitlets import Enum as TEnum, observe, Integer, Float, Bool, Instance
from common.default_appsettings import QuantityEnum
from common.eval_component.transfer_functions import (t_tmm_model_1layer, model_1layer, t_tmm_model_2layer,
                                                      model_2layer, _t_model_2layer)
from common.consts import eps0_thz
from common.eval_component.shgo_settings import SHGOOptions, MinimizerOptions
from common.save import ResultSaver
from common.eval_component.eval_result import EvalResult
from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import QObject, Signal
from enum import Enum, member

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
    cost_fun = TEnum(CostFunctions, default_value=CostFunctions.abs_cost)
    selected_result_path = TPath(Path("")).tag(name="Load result")
    selected_substrate_result_path = TPath(Path("")).tag(name="Substrate result")
    optimization_progress = Float(0, min=0, max=1,
                                  group="General", read_only=True).tag(name="Progress")

    regression_grp_name = "Regression"
    meas_quantity = TEnum(QuantityEnum, default_value=QuantityEnum.TransmissionAmp,
                          group=regression_grp_name).tag(name="Measurement quantity")
    regression_model = TEnum(RegressionModels, default_value=RegressionModels.drude, group=regression_grp_name)
    convert_sigma_to_t = Bool(default_value=False, group=regression_grp_name)

    sig0_bounds = ValueRange([Q_(10, "S/cm"), Q_(20, "S/cm")], group=regression_grp_name).tag(name="σ₀ Bounds")
    tau_bounds = ValueRange([Q_(10, "fs"), Q_(1000, "fs")], group=regression_grp_name).tag(name="τ Bounds")
    wp_bounds = ValueRange([Q_(-10, "THz"), Q_(100, "THz")], group=regression_grp_name).tag(name="ωₚ Bounds")
    eps_inf_bounds = ValueRange([1.0, 100.0], group=regression_grp_name).tag(name="ε_inf Bounds")
    eps_s_bounds = ValueRange([1.0, 100.0], group=regression_grp_name).tag(name="ε_s Bounds")
    c1_bounds = ValueRange([-1.0, 1.0], group=regression_grp_name).tag(name="c₁ Bounds")

    transmission_model = TEnum(TransmissionModels, default_value=TransmissionModels.tmm_1layer,
                               group="Transmission fit")
    d_opt_axis_bounds = ValueRange([Q_(500, "µm"), Q_(580, "µm", )], group="Transmission fit")
    d_opt_axis_step = Quantity(Q_(10, "µm"), group="Transmission fit")
    use_custom_d_opt_axis = Bool(True, group="Transmission fit")
    number_of_workers = Integer(8, group="Transmission fit").tag(name="Number of workers")

    current_result = Instance(EvalResult)
    selected_substrate_result = Instance(EvalResult)
    result_saver = Instance(ResultSaver)

    def __init__(self, dataset: DataSet, dataset_sub: DataSet=None,
                 plotter: DataSetPlotter=None, object_name: str = None):
        super().__init__(object_name=object_name)
        self.dataset = dataset
        self.settings = dataset.settings
        self.sub_dataset = self._link_sub_dataset(dataset_sub)
        self.plotter = plotter

        self.shgo_options = SHGOOptions()

        self.result_saver = self.setup_saver()

        self.freq_axis = self.dataset.freq_axis

        self._opt_conf = {}

        self.current_result = EvalResult(object_name="Current result")
        self.selected_substrate_result = EvalResult(object_name="Substrate result")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.settings.save_configuration(self)

    def __enter__(self):
        self.settings.load_configuration(self)

    def setup_saver(self):
        res_saver = ResultSaver()
        res_saver.registerObjectAttribute(self, "sel_point")

        return res_saver

    def update_progress(self, progress_value: float):
        try:
            self.set_trait("optimization_progress", progress_value)
        except Exception as exc:
            logging.error(f"Failed to update progress traitlet: {exc}")

    def _link_sub_dataset(self, dataset: DataSet = None):
        if dataset is None:
            return None

        self.dataset.link_sub_dataset(dataset)

        return dataset

    def _get_dataset(self, which=DataSetType.Main):
        if which == DataSetType.Main:
            return self
        elif which == DataSetType.Sub:
            if self.sub_dataset is None:
                raise ValueError("No sub-dataset linked.")
            return self.sub_dataset
        else:
            return self, self.sub_dataset

    def set_y_meas(self):
        meas_quantity = self.meas_quantity
        func = self.dataset.func_map[meas_quantity]

        meas = self.dataset.get_measurement(*self.sel_point)
        y_meas = func(meas)

        f_idx = f_axis_idx_map(self.freq_axis, self.settings.eval_opt.fit_range)
        self._opt_conf["y_meas"] = y_meas[f_idx]

        return self._opt_conf["y_meas"]

    @property
    def _ui_control_widget(self):
        return getattr(self, '_ui_widget_internal', None)

    @_ui_control_widget.setter
    def _ui_control_widget(self, value):
        self._ui_widget_internal = value
        if value:
            self.select_regression_model()

    @observe("regression_model")
    def select_regression_model(self, change=None):
        self.update_freq_axis()
        model = self.regression_model.value if change is None else change["new"].value
        self._opt_conf["model"] = partial(model, self._opt_conf["freq_axis"])

        param_keys = model_params(self.regression_model.name)
        active_bounds = [f"{p}_bounds" for p in param_keys]

        self.toggle_traits(active_bounds, self, group_filter=self.regression_grp_name, endswith_filter="_bounds")

        return self._opt_conf["model"]

    def update_freq_axis(self):
        f_idx = f_axis_idx_map(self.freq_axis, self.settings.eval_opt.fit_range)
        self._opt_conf["freq_axis"] = self.freq_axis[f_idx]

    def setup_bounds(self):
        signature = inspect.signature(self._opt_conf["model"])
        bounds = []
        for arg in signature.parameters.values():
            bound = getattr(self, arg.name + "_bounds", None)
            if bound is None:
                min_value, max_value = -np.inf, np.inf
            else:
                min_value = bound[0].magnitude if isinstance(bound[0], Q_) else bound[0]
                max_value = bound[1].magnitude if isinstance(bound[1], Q_) else bound[1]

            bounds.append([min_value, max_value])

        self._opt_conf["bounds"] = bounds

    @observe("selected_result_path")
    def set_selected_result(self, change):
        result_path = change["new"]
        self.current_result.load_result(result_path)

    @observe("selected_substrate_result_path")
    def set_substrate_result(self, change):
        result_path = change["new"]
        self.selected_substrate_result.load_result(result_path)

    @observe("cost_fun")
    def setup_cost(self, change=None):
        cost_func = self.cost_fun.value if change is None else change["new"]
        mod_func = self._opt_conf.get("model", self.select_regression_model())
        y_meas = self._opt_conf.get("y_meas", self.set_y_meas())

        if self.convert_sigma_to_t:
            def t_mod_func(*args, **kwargs):
                freq_axis = self._opt_conf["freq_axis"]
                sigma = mod_func(*args, **kwargs)
                n = self.sigma_to_n(freq_axis, sigma)

                t = self.transmission_model(n, freq_axis, **self._opt_conf)

                return t

            mod_func = t_mod_func

        self._opt_conf["func"] = lambda p: cost_func(y_meas, mod_func(*p))

    def update_opt_config(self):
        self.update_freq_axis()
        self._opt_conf["h"] = self.settings.sample_properties.d_film.magnitude

        self.set_y_meas()
        self.setup_cost()
        self.setup_bounds()

        return self._opt_conf

    def prepare_regression_result(self, opt_res):
        model_name = self.regression_model.name
        param_keys = model_params(model_name)
        x = opt_res.x

        opt_res_dict = {"model_name": model_name}
        for param_idx, p in enumerate(param_keys):
            bound = getattr(self, p + "_bounds", None)
            if bound is None or not isinstance(bound[0], Q_):
                opt_res_dict[p] = x[param_idx]
            else:
                opt_res_dict[p] = Q_(x[param_idx], bound[0].units)

        opt_res_dict["nit"] = opt_res.nit
        opt_res_dict["fun"] = opt_res.fun
        opt_res_dict["converged"] = opt_res.success
        opt_res_dict["timestamp"] = str(datetime.now().isoformat())

        freq_axis = Q_(self._opt_conf["freq_axis"], "THz")
        unit = self.meas_quantity.value.unit
        opt_res_dict["y_meas"] = SingleQuantityDataSet(axes=[freq_axis], data=Q_(self._opt_conf["y_meas"], unit),
                                                       axes_labels=["Frequency"], data_label=self.meas_quantity.name)
        opt_res_dict["y_mod"] = SingleQuantityDataSet(axes=[freq_axis], data=Q_(self._opt_conf["model"](*x), unit),
                                                      axes_labels=["Frequency"], data_label=self.meas_quantity.name)

        opt_res_dict["result_type"] = "Regression"
        return opt_res_dict

    @action("Fit regression model", group=regression_grp_name)
    def perform_regression(self):
        def bg_worker():
            try:
                opt_config = self.update_opt_config()

                min_kwargs = self.shgo_options.minimizer_kwargs.traits(group=MinimizerOptions.minimizer_opt_grp)
                min_kwargs["method"] = str(self.shgo_options.minimizer_kwargs.method.value)

                opt_res_ = shgo(func=opt_config["func"],
                                bounds=opt_config["bounds"],
                                n=self.shgo_options.n,
                                iters=self.shgo_options.iters,
                                minimizer_kwargs=min_kwargs,
                                options=self.shgo_options.get_shgo_options(),
                                )
                logging.info("Fit result: {}".format(opt_res_))

                reg_res = self.prepare_regression_result(opt_res_)
                reg_res["measurement_quantity"] = self.meas_quantity.name

                self.current_result.set_traits_from_dict(reg_res)
                self.result_saver.process(self.current_result)
            except Exception:
                traceback.print_exc()

        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(bg_worker)

    @action("Fit transmission model", group="Transmission fit")
    def fit_unknown_layer(self):
        progress_carrier = ProgressSignalCarrier()
        progress_carrier.progress_changed.connect(self.update_progress)

        def bg_worker():
            try:
                qs_eval = QSpaceEval(self)
                qs_res = qs_eval.q_space_eval_mp(progress_carrier=progress_carrier)

                self.current_result.set_traits_from_dict(qs_res)
                self.result_saver.process(self.current_result)
            except Exception:
                traceback.print_exc()

        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(bg_worker)

    def sigma_to_n(self, freq, sigma):
        w = 2 * np.pi * freq

        sigma *= 1e-4 # S/cm -> S/µm
        n_ = (1 + 1j) * np.sqrt(sigma/(2*w*eps0_thz))

        return n_


    def conductivity_model(self, sigma_exp):
        self._opt_conf["sigma_exp"] = sigma_exp
        opt_res = self._fit_freq_model()
        p = opt_res.x # x = [tau, sig0, wp, eps_inf, eps_s]
        # p = [1, 100, 4*np.pi, 10, 20]
        # p = [1, 100, 2, 16.8, 20] # ulatowski plot
        # p = [-1.588e-02,  5.044e+01,  920.8,  40.55, -43.13] # fit result
        # p = [-0.01588,  50.44,  920.8,  40.55, -43.13]
        # p = [40, 3, 50, 9]
        sigma_model_ = self._total_response(self.freq_axis, *p) # TODO _total_response returns n

        return sigma_model_

    def t_sim_1layer(self):
        if not self.options["sim_opt"]["enabled"]:
            exit("BB")

        self.options["eval_opt"]["shift_sub"] = self.options["sim_opt"]["shift_sim"]
        n_sub = self.options["sim_opt"]["n_sub"]
        nfp_og = self.options["eval_opt"]["nfp"]

        self.options["eval_opt"]["nfp"] = self.options["sim_opt"]["nfp_sim"]
        t_sim = np.zeros_like(self.freq_axis, dtype=complex)
        for f_idx, freq in enumerate(self.freq_axis):
            n_sub_ = n_sub#0.03*freq + n_sub.real + 1j * freq * 0.001
            t_sim[f_idx] = self._t_model_1layer(freq, n_sub_)
        t_sim = np.abs(t_sim) * np.exp(-1j * np.angle(t_sim))

        self.options["eval_opt"]["nfp"] = nfp_og

        return t_sim

    # should be included in the transmission fit result (also calculates t spectrum)
    def sub_meas_sim(self):
        t_sim = self.t_sim_1layer()

        sub_pnt = self.options["eval_opt"]["sub_pnt"]

        # ref1_fd, ref1_meas = self.get_ref_data(Domain.Frequency, ref_idx=10, ret_meas=True)
        ref1_fd, ref1_meas = self.sub_dataset.get_ref_data(Domain.Frequency, point=sub_pnt, ret_meas=True)
        ref2_fd, ref2_meas = self.sub_dataset.get_ref_data(Domain.Frequency, ref_idx=10, ret_meas=True)

        meas_time_diff = (ref1_meas.meas_time - ref2_meas.meas_time).total_seconds()
        print("ref1 - ref2 measurement time difference (seconds): ", np.round(meas_time_diff, 2))

        ref_amp = np.abs(ref2_fd[:, 1])
        ref_phi = np.angle(ref2_fd[:, 1])
        ref2_fd[:, 1] = ref_amp * np.exp(1j * ref_phi)

        t_sim_meas = t_sim * ref1_fd[:, 1] / ref2_fd[:, 1]

        sam_sim = t_sim * ref1_fd[:, 1]
        sam_sim_fd = np.array([self.freq_axis, sam_sim], dtype=complex).T

        t_sim_meas = sam_sim_fd[:, 1] / ref2_fd[:, 1]

        sam_sim_td = do_ifft(sam_sim_fd, conj=False)

        plt.figure("Time domain")
        plt.plot(sam_sim_td[:, 0], sam_sim_td[:, 1], label="Model")

        return t_sim_meas

    # should be covered by fit single layer model to point selected on substrate
    # -> 2 layer fit to film point /w substrate result selected
    def eval_point_n_fit(self, film_pnt=None):
        """
        Fit refractive index to the substrate measurement (n_sub)
        then use n_sub in the fit of the refractive index to the film measurement (n_film)
        calculate sigma from n_film
        """
        sub_pnt = self.options["eval_opt"]["sub_pnt"]
        if film_pnt is None:
            film_pnt = sub_pnt
        res = {}

        meas_sub = self.sub_dataset.get_measurement(*sub_pnt)
        meas_film = self.get_measurement(*film_pnt)

        single_layer_eval_res = self.sub_dataset.windowing_eval(meas_sub, (0, 10))
        res["alpha"] = single_layer_eval_res["alpha"]

        res["sigma_exp"] = self.conductivity(meas_film)
        res["sigma_mod"] = self.regression_model(res["sigma_exp"])

        #self.sub_dataset.options["pp_opt"]["window_opt"]["enabled"] = True
        self.sub_dataset.options["pp_opt"]["window_opt"]["en_plot"] = False
        self.sub_dataset.options["pp_opt"]["window_opt"]["fig_label"] = "sub"
        t_exp_1layer = self.sub_dataset.transmission(meas_sub, 1, phase_sign=-1)
        # t_exp_1layer = self.transmission_sim()
        # t_exp_1layer = self.sub_meas_sim()

        #self.sub_dataset.options["pp_opt"]["window_opt"]["enabled"] = False

        # phi = np.unwrap(np.angle(t_exp_1layer))
        # phi = phase_correction(self.freq_axis, phi, en_plot=True, fit_range=(0.3, 0.6))
        # phi -= 0.03
        # t_exp_1layer = np.abs(t_exp_1layer) * np.exp(1j * phi)

        res["t_exp_1layer"] = t_exp_1layer

        n_sub = self._fit_1layer(t_exp_1layer)
        res["n_sub"] = n_sub
        self._opt_conf["n_sub"] = n_sub

        if self.options["eval_opt"]["area_fit"]:
            return res

        self.options["pp_opt"]["window_opt"]["fig_label"] = "film"
        t_exp_2layer = self.transmission(meas_film, 1, phase_sign=-1)

        res["t_exp_2layer"] = t_exp_2layer

        n_film = self._fit_2layer(t_exp_2layer, n_sub)

        res["t_mod_film"] = self._t_model_2layer(self.freq_axis, n_sub=n_sub, n_film=n_film)

        w = 2 * np.pi * np.array(self.freq_axis)
        res["sigma_n_film"] = 1e4 * 2 * eps0_thz * n_film ** 2 * w / (1 + 1j)  # S/cm

        return res

    # check if regression can handle this. Result saving? plotting?
    def eval_point_model_fit(self, film_pnt=None):
        """
        Fit model with frequency independent parameters to the full spectrum
        """
        sub_pnt = self.options["eval_opt"]["sub_pnt"]
        if film_pnt is None:
            film_pnt = sub_pnt
        res = {}

        meas_sub = self.sub_dataset.get_measurement(*sub_pnt)
        meas_film = self.get_measurement(*film_pnt)

        res["t_exp_1layer"] = self.sub_dataset.transmission(meas_sub, 1, phase_sign=-1)
        n_sub = self._fit_1layer(res["t_exp_1layer"])
        res["n_sub"] = n_sub

        res["t_exp_2layer"] = self.transmission(meas_film, 1, phase_sign=-1)

        # n_sub.imag = 0.023*self.freq_axis

        self._opt_conf["y_meas"] = res["t_exp_2layer"]
        self._opt_conf["n_sub"] = n_sub
        self._opt_conf["eps_s"] = 5
        self._opt_conf["eps_inf"] = 50
        # self._opt_consts["tau"] = 100 * 10

        freq_fit_res = self._fit_freq_model()
        p_opt = freq_fit_res.x

        res["t_mod_film"] = self._t_cond_model(self.freq_axis, p_opt)
        # best res: [ 2.720e+04 -1.130e+03  9.979e-01 -2.997e+03  4.591e+04] or [ 3.714e+04  4.980e+02  1.801e+00  5.587e+04  1.945e+00]
        # p_opt = [100, 1000, 2, 20, 0.025*16.8] # sig0, tau, wp, eps_s, eps_inf = 16.8 # 0.025*16.8
        # p_opt = [ 3.714e+04,  4.980e+02,  1.801e+00, 5.587e+04,  1.945e+00]
        # p_opt = [*p_opt, self._opt_consts["eps_s"], self._opt_consts["eps_inf"]]

        sig_cc = self._drude(self.freq_axis, p_opt[0], p_opt[1])
        eps_l = self._lattice_contrib(self.freq_axis, *p_opt[1:])
        n_film = self._total_response(self.freq_axis, *p_opt)
        sig_tot = self._n_to_sigma(self.freq_axis, n_film) + sig_cc

        # res["t_mod_film"] = self._t_cond_model(self.freq_axis, p_opt)

        # n_film = self._sigma_to_n(self.freq_axis, sig_tot)

        if self.plotter is not None:
            self.plotter.plot_freq_fit(res)

        plt.figure("_drude_cc_part")
        plt.title("Charge carrier part")
        plt.plot(self.freq_axis, sig_cc.real, label="real part")
        plt.plot(self.freq_axis, sig_cc.imag, label="imag part")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Sigma_cc (S/cm)")

        plt.figure("_drude_l_part")
        plt.title("Lattice part")
        plt.plot(self.freq_axis, eps_l.real, label="real part")
        plt.plot(self.freq_axis, eps_l.imag, label="imag part")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("eps_l")

        plt.figure("_total_response")
        plt.title("Total response")
        plt.plot(self.freq_axis, sig_tot.real, label="real part")
        plt.plot(self.freq_axis, sig_tot.imag, label="imag part")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("sig_tot (S/cm)")

        """
        plt.figure("Transmission fit abs film")
        for sig0_ in [1e4, 1e5]:
            f0, f1 = self.options["eval_opt"]["fit_range_sub"]
            f_mask = (f0 < self.freq_axis) * (self.freq_axis < f1)
            t_mod = self.selected_model(self.freq_axis, sig0_, tau=100)
            plt.plot(self.freq_axis[f_mask], np.abs(t_mod[f_mask]), label=f"Model sigma0={sig0_}")
            print(self._opt_fun_freq_model([sig0_, 100]), sig0_, 100)
        """

        res["t_mod_sub"] = t_tmm_model_1layer(self.freq_axis, n_sub, self.settings.sample_properties.d)
        if self.plotter is not None:
            self.plotter.plot_eval_res(res)

        return res

if __name__ == "__main__":

    print(RegressionModels.drude.name)


