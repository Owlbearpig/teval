from common.dataset import DataSet, Domain
from common.components import ComponentBase, action
from common.datasetplotter import DataSetPlotter
from common.functions import window, do_ifft, phase_correction, f_axis_idx_map
from common.eval_component.shgo import shgo
from scipy.optimize import shgo
from functools import partial
import numpy as np
from common.consts import eps0_thz, c_thz
import matplotlib.pyplot as plt
import logging
import inspect
from scipy.signal import iirnotch, filtfilt
from q_space_eval import QSpaceEval
from numpy import polyfit
from enum import Enum, member
from common.eval_component.conductivity_models import *
from common.traits import Quantity, Q_, ValueRange
from traitlets import Enum as TEnum, observe, Integer, Float, Bool, Instance
from common.default_appsettings import QuantityEnum
from common.eval_component.transfer_functions import (t_tmm_model_1layer, model_1layer, t_tmm_model_2layer,
                                                      model_2layer, _t_model_2layer)
from common.eval_component.shgo_settings import SHGOOptions, MinimizerOptions


def abs_cost_fun(y_meas, y_mod):

    abs_diff = (np.abs(y_meas) - np.abs(y_mod)) ** 2

    return np.sum(abs_diff)


def phi_cost_fun(y_meas, y_mod):
    phi_diff = (np.angle(y_meas) - np.angle(y_mod)) ** 2

    return np.sum(phi_diff)


def combined_cost_fun(y_meas, y_mod):
    return abs_cost_fun(y_meas, y_mod) + phi_cost_fun(y_meas, y_mod)

# logging.basicConfig(level=logging.WARNING)

class OptRes:
    pass

class RegressionModels(Enum):
    drude = member(drude)
    drude2 = member(drude2)
    lattice_drude = member(total_response)
    drude_smith = member(drude_smith)
    # TODO add transmission models derived from conductivity (from freq. indep. parameters)

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

    meas_quantity = TEnum(QuantityEnum, default_value=QuantityEnum.TransmissionAmp).tag(name="Measurement quantity")
    regression_model = TEnum(RegressionModels, default_value=RegressionModels.drude)
    transmission_model = TEnum(TransmissionModels, default_value=TransmissionModels.tmm_1layer)
    cost_fun = TEnum(CostFunctions, default_value=CostFunctions.abs_cost)


    sig0 = Quantity(Q_(10, "S/cm"), group="Initial optimization values")
    sig0_bounds = ValueRange([Q_(10, "S/cm"), Q_(20, "S/cm")], group="Optimization bounds")

    wp = Quantity(Q_(10, "THz"), group="Initial optimization values")
    wp_bounds = ValueRange([Q_(-10, "THz"), Q_(100, "THz")], group="Optimization bounds")


    def __init__(self, dataset: DataSet, dataset_sub: DataSet=None,
                 plotter: DataSetPlotter=None, object_name: str = None):
        super().__init__(object_name=object_name)
        self.dataset = dataset
        self.settings = dataset.settings
        self.sub_dataset = self._link_sub_dataset(dataset_sub)
        self.plotter = plotter

        self.shgo_options = SHGOOptions()


        self.freq_axis = self.dataset.freq_axis


        self._opt_args = {}
        self._fixed_params = {}

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.settings.save_configuration(self)

    def __enter__(self):
        self.settings.load_configuration(self)

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
        meas_quantity = self.settings.eval_opt.meas_quantity
        func = self.dataset.func_map(meas_quantity)

        meas = self.dataset.get_measurement(*self.sel_point)
        y_meas = func(meas)

        f_idx = f_axis_idx_map(self.freq_axis, self.settings.eval_opt.fit_range)
        self._opt_args["y_meas"] = y_meas[f_idx]

    @observe("regression_model")
    def select_regression_model(self, change):
        fun = getattr(self.regression_model, change["new"])
        f_idx = f_axis_idx_map(self.freq_axis, self.settings.eval_opt.fit_range)

        self._opt_args["model"] = partial(fun, self.freq_axis[f_idx])

        self.setup_bounds()

    def setup_bounds(self):
        signature = inspect.signature(self._opt_args["model"])

        bounds = []
        for arg in signature.parameters.values():
            bound = getattr(self, arg.name + "_bounds", None)
            if bound:
                bounds.append(bound)
            else:
                bounds.append([-np.inf, np.inf])

        self._opt_args["bounds"] = bounds

    @observe("cost_fun")
    def setup_cost(self, c=None):
        if c is None:
            cost_func = getattr(self, "cost_fun")
        else:
            cost_func = c["new"]
        mod_func = self._opt_args["model"]
        y_meas = self._opt_args["y_meas"]

        self._opt_args["func"] = lambda p: cost_func(y_meas, mod_func(*p))

    @action("Fit regression model")
    def perform_regression(self):
        self.set_y_meas()
        self.setup_cost()

        min_kwargs = self.shgo_options.minimizer_kwargs.traits(group=MinimizerOptions.minimizer_opt_grp)
        min_kwargs["method"] = self.shgo_options.minimizer_kwargs.method

        opt_res_ = shgo(func=self._opt_args["func"],
                        bounds=self._opt_args["bounds"],
                        n=self.shgo_options.n,
                        iters=self.shgo_options.iters,
                        minimizer_kwargs=min_kwargs,
                        options=self.shgo_options.traits(group=SHGOOptions.shgo_options_grp),
                        )
        return opt_res_

    """
    - 
    
    """

    @observe("transmission_model")
    def select_model(self, change):

        self.transmission_model = getattr(self.transmission_model, change["new"])



    def fit_1layer(self):



        qs_eval = QSpaceEval(self)
        qs_res = qs_eval.q_space_eval()



        bounds = self.options["eval_opt"]["sub_bounds"]


        # t_exp_phi_ = phase_correction(self.freq_axis, np.unwrap(np.angle(t_exp_)), en_plot=True)
        # t_exp_ = np.abs(t_exp_) * np.exp(1j * t_exp_phi_)

        f0, f1 = self.options["eval_opt"]["fit_range_sub"]
        freq_mask = (f0 <= self.freq_axis) * (self.freq_axis <= f1)

        shift = 15.3  # 15.3  # 30
        best_ = ((None, None), np.inf)
        n_opt_best = None
        d_sub0 = self.options["sample_properties"]["d_1"]
        for d_sub in [
            *np.arange(d_sub0, d_sub0 + 1, 1.0)]:  # [639.5]:#[*np.arange(639, 640, 0.5)]: # 639 / 640 (Teralyzer)
            for shift in [*np.arange(-0, 1, 1.0)]:  # 28, 22, -3
                self.options["sample_properties"]["d_1"] = d_sub
                self.options["eval_opt"]["shift_sub"] = shift
                gof = 0
                n_opt = np.zeros_like(self.freq_axis, dtype=complex)
                for f_idx in np.arange(len(self.freq_axis))[freq_mask]:
                    cost = partial(self._opt_fun_1layer,
                                   freq_=self.freq_axis[f_idx],
                                   # t_exp_=t_exp_[f_idx], TODO
                                   bounds_=bounds)

                    opt_res_ = shgo(cost,
                                    bounds,
                                    #minimizer_kwargs=min_kwargs, TODO
                                    #iters=shgo_iters TODO
                                    )
                    n_opt[f_idx] = opt_res_.x[0] + opt_res_.x[1] * 1j
                    gof += opt_res_.fun

                n_imag = n_opt.imag
                peak_val, sum_val, n_imag_filt = self._q_val(n_imag)

                gof = gof / np.sum(freq_mask)

                print("Sub:", sum_val, peak_val, shift, d_sub, gof)

                res = ((shift, d_sub), peak_val, sum_val)
                if peak_val < best_[1]:
                    best_ = res
                    n_opt_best = n_opt

                with open("debug_out", "a") as f:
                    res_line = f"{pnt} {str(res)}\n"
                    f.write(res_line)

        print(f"Best q-val sub: {best_} @{pnt}")
        self.options["sample_properties"]["d_1"] = best_[0][1]
        self.options["eval_opt"]["shift_sub"] = best_[0][0]

        en_debug = False
        if en_debug:
            with open("result_out", "a") as f:
                res_line = f"{pnt} {str(best_)}\n"
                f.write(res_line)

            with open("debug_out", "a") as f:
                f.write("\n")

        return n_opt_best

    def fit_2layer(self):
        pass

    @action("Fit transmission model")
    def fit_transmission_coefficient(self):
        pass

    def _sigma_dc(self, freq, sig0):
        w = 2 * np.pi * freq

        sig0 *= 1e-4 # S/cm -> S/µm
        n_ = (1 + 1j) * np.sqrt(sig0/(2*w*eps0_thz))

        return n_

    def _t_cond_model(self, freq_, p_):
        n_sub_ = self._opt_args["n_sub"]
        # tau = self._opt_consts["tau"]
        n_film_ = self.selected_n_model(freq_, *p_)

        # n_film_ = self._sigma_to_n(freq_, sig_model_)
        t_mod_ = self._t_model_2layer(freq_, n_sub_, n_film_)

        return t_mod_

    def _opt_fun_1layer(self, x_, freq_, t_exp_, bounds_=None):
        if bounds_ is not None:
            for i, p_ in enumerate(x_):
                if not (bounds_[i][0] <= p_) * (p_ <= bounds_[i][1]):
                    return np.inf

        n = x_[0] + 1j * x_[1]
        t_mod = self._t_model_1layer(freq_, n)
        # t_mod = self._tmm_model_1layer(freq_, n)

        abs_loss = (np.abs(t_mod) - np.abs(t_exp_)) ** 2
        ang_loss = (np.angle(t_mod) - np.angle(t_exp_)) ** 2

        return abs_loss + ang_loss

    def _opt_fun_2layer(self, x_, freq, n_sub_, t_exp_):
        #if x_[0] < 1 or x_[1] < 0:
        #    return np.inf

        n_film_ = x_[0] + 1j * x_[1]

        t_mod = self._t_model_2layer(freq, n_sub=n_sub_, n_film=n_film_)

        mag = (np.abs(t_mod) - np.abs(t_exp_))**2
        ang = (np.angle(t_mod) - np.angle(t_exp_))**2

        # real_part = (t_mod.real - t_exp_.real) ** 2
        # imag_part = (t_mod.imag - t_exp_.imag) ** 2

        return mag + ang

    def _gof(self, n_film_, n_sub_, t_exp_):
        f0, f1 = self.options["eval_opt"]["fit_range_film"]
        freq_axis = self.dataset.freq_axis
        freq_mask = (f0 <= freq_axis) * (freq_axis <= f1)

        s = 0
        for f_idx in np.arange(len(freq_axis))[freq_mask]:
            n_f = (n_film_[f_idx].real, n_film_[f_idx].imag)
            s += self._opt_fun_2layer(n_f, freq_axis[f_idx], n_sub_[f_idx], t_exp_[f_idx])

        return s / len(freq_axis[freq_mask])

    def _q_val(self, n_imag_):
        f0, f1 = self.options["eval_opt"]["fit_range_film"]
        freq_axis = self.dataset.freq_axis
        freq_mask = (f0 <= freq_axis) * (freq_axis <= f1)

        n_imag_mean = np.mean(n_imag_[freq_mask])

        win_settings = {"en_plot": False, "win_start": f0, "win_width": f1 - f0, "fig_label": "FFT"}
        n_imag_ = window(np.array([freq_axis, n_imag_]).T, **win_settings)
        n_imag_ = n_imag_[:, 1]

        fft_ = np.fft.rfft(n_imag_[freq_mask] - n_imag_mean)
        fft_freq_axis = np.fft.rfftfreq(len(n_imag_[freq_mask]),
                                        d=np.mean(np.diff(freq_axis)))

        # mask_ = (10 < fft_freq_axis) * (fft_freq_axis < 20)
        mask_ = (11 <= fft_freq_axis)
        peak_val, peak_idx = np.max(np.abs(fft_[mask_])), np.argmax(np.abs(fft_[mask_]))
        sum_val = np.sum(np.abs(fft_[7 <= fft_freq_axis]))

        # plt.figure("TESTFFT")
        # plt.plot(fft_freq_axis, np.abs(fft_), label=f"shift {shift}")

        fs = 1/np.mean(np.diff(freq_axis))
        Q = 0.5  # quality factor: higher = narrower

        peak_freq = fft_freq_axis[mask_][peak_idx]
        # print(peak_freq, fs)
        b, a = iirnotch(peak_freq / (fs / 2), Q)

        n_imag_filt_ = filtfilt(b, a, n_imag_)

        return peak_val, sum_val, n_imag_filt_

    def _fit_2layer(self, t_exp_, n_sub_):
        pnt = self.options["eval_opt"]["sub_pnt"]
        bounds_ = self.options["eval_opt"]["film_bounds"]
        # bounds_ = [(1, 15), (1, 15)]

        min_kwargs = {"method": "Nelder-Mead",
                      "options": {
                          "maxfev": 1500,
                          "maxiter": 2000,
                          "tol": 1e-10,
                          "fatol": 1e-10,
                          "xatol": 1e-10,
                      }
                      }
        shgo_options = {
            "maxfev": 1650,
            "maxiter": 2000,
            # "f_tol": 1e-12,
            # "maxiter": 50,
            # "ftol": 1e-12,
            # "xtol": 1e-12,
            # "maxev": 4000,
            # "minimize_every_iter": True,
            "disp": False,
        }
        shgo_iters = 2
        shgo_n = 200

        f0, f1 = self.options["eval_opt"]["fit_range_film"]
        freq_mask = (f0 <= self.freq_axis) * (self.freq_axis <= f1)

        best_ = (None, np.inf)
        n_opt_best = None
        d_sub0 = self.options["sample_properties"]["d_1"]
        for d_sub in [*np.arange(d_sub0, d_sub0 + 1, 1.0)]:#[*np.arange(640.00, 655, 5.0)]: # [656.8]:# 650
            for shift in [0]:  # [61.1]: # 30
                # self.options["sample_properties"]["d_film"] = d_film
                self.options["sample_properties"]["d_2"] = d_sub
                self.options["eval_opt"]["shift_film"] = shift

                gof = 0
                n_opt = np.zeros((len(self.freq_axis), len(bounds_)), dtype=complex)
                for f_idx in np.arange(len(self.freq_axis))[freq_mask]:
                    args_ = (self.freq_axis[f_idx], n_sub_[f_idx], t_exp_[f_idx])

                    opt_res_ = shgo(self._opt_fun_2layer,
                                    bounds=bounds_,
                                    minimizer_kwargs=min_kwargs,
                                    options=shgo_options,
                                    args=args_,
                                    iters=shgo_iters,
                                    n=shgo_n
                                    )
                    n_opt[f_idx] = opt_res_.x
                    # print(self.freq_axis[f_idx], n_opt[f_idx])
                    gof += opt_res_.fun

                n_opt = n_opt[:, 0] + 1j * n_opt[:, 1]
                n_imag = n_opt.imag

                peak_val, sum_val, n_imag_filt_ = self._q_val(n_imag)

                if peak_val < best_[1]:
                    best_ = ((shift, d_sub), peak_val, gof)
                    n_opt_best = n_opt

        print("Best q-val film: ", best_)
        return n_opt_best

    def _fit_1layer(self, t_exp_):
        pnt = self.options["eval_opt"]["sub_pnt"]
        bounds = self.options["eval_opt"]["sub_bounds"]
        #bounds = [(3.05, 3.13), (0.0025, 0.0050)]
        min_kwargs = {"method": "Nelder-Mead",
                      "options": {
                          "maxfev": 120,
                          # "maxiter": 20,
                          "tol": 1e-12,
                          "fatol": 1e-12,
                          "xatol": 1e-12,
                      }
                      }
        shgo_iters = 3

        #t_exp_phi_ = phase_correction(self.freq_axis, np.unwrap(np.angle(t_exp_)), en_plot=True)
        #t_exp_ = np.abs(t_exp_) * np.exp(1j * t_exp_phi_)

        f0, f1 = self.options["eval_opt"]["fit_range_sub"]
        freq_mask = (f0 <= self.freq_axis) * (self.freq_axis <= f1)

        shift = 15.3 # 15.3  # 30
        best_ = ((None, None), np.inf)
        n_opt_best = None
        d_sub0 = self.options["sample_properties"]["d_1"]
        for d_sub in [*np.arange(d_sub0, d_sub0+1, 1.0)]:#[639.5]:#[*np.arange(639, 640, 0.5)]: # 639 / 640 (Teralyzer)
            for shift in [*np.arange(-0, 1, 1.0)]:# 28, 22, -3
                self.options["sample_properties"]["d_1"] = d_sub
                self.options["eval_opt"]["shift_sub"] = shift
                gof = 0
                n_opt = np.zeros_like(self.freq_axis, dtype=complex)
                for f_idx in np.arange(len(self.freq_axis))[freq_mask]:

                    cost = partial(self._opt_fun_1layer,
                                   freq_=self.freq_axis[f_idx],
                                   t_exp_=t_exp_[f_idx],
                                   bounds_=bounds)

                    opt_res_ = shgo(cost,
                                    bounds,
                                    minimizer_kwargs=min_kwargs,
                                    iters=shgo_iters)
                    n_opt[f_idx] = opt_res_.x[0] + opt_res_.x[1] * 1j
                    gof += opt_res_.fun

                n_imag = n_opt.imag
                peak_val, sum_val, n_imag_filt = self._q_val(n_imag)

                gof = gof / np.sum(freq_mask)

                print("Sub:", sum_val, peak_val, shift, d_sub, gof)

                res = ((shift, d_sub), peak_val, sum_val)
                if peak_val < best_[1]:
                    best_ = res
                    n_opt_best = n_opt

                with open("debug_out", "a") as f:
                    res_line = f"{pnt} {str(res)}\n"
                    f.write(res_line)

        print(f"Best q-val sub: {best_} @{pnt}")
        self.options["sample_properties"]["d_1"] = best_[0][1]
        self.options["eval_opt"]["shift_sub"] = best_[0][0]

        with open("result_out", "a") as f:
            res_line = f"{pnt} {str(best_)}\n"
            f.write(res_line)

        with open("debug_out", "a") as f:
            f.write("\n")

        return n_opt_best

    def conductivity_model(self, sigma_exp):
        self._opt_args["sigma_exp"] = sigma_exp
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

    def ref_std(self, en_plot=False):
        all_refs = self.measurements["refs"]
        ref_data = np.zeros((len(all_refs), len(self.freq_axis)), dtype=complex)
        for ref_idx, ref_meas in enumerate(all_refs):
            ref_fd = self._get_data(ref_meas, domain=Domain.Frequency)
            ref_data[ref_idx] = ref_fd[:, 1]

        freq_range = (0.35 < self.freq_axis)*(self.freq_axis < 4.0)
        amp_argmin = np.argmin(np.abs(ref_data[:, freq_range]))
        amp_argmin = np.unravel_index(amp_argmin, ref_data[:, freq_range].shape)[0]
        amp_min = ref_data[amp_argmin]

        amp_argmax = np.argmax(np.abs(ref_data[:, freq_range]))
        amp_argmax = np.unravel_index(amp_argmax, ref_data[:, freq_range].shape)[0]
        amp_max = ref_data[amp_argmax]

        amp_mean, amp_std = np.mean(np.abs(ref_data), axis=0), np.std(np.abs(ref_data), axis=0)
        phi = np.unwrap(np.angle(ref_data))
        phi_mean, phi_std = np.mean(phi, axis=0), np.std(phi, axis=0)

        def dB(y):
            return 20*np.log10(y)

        if en_plot:
            plt.figure("Ref Standard deviation")
            #plt.plot(self.freq_axis, dB(np.abs(amp_min)), label="Min")
            #plt.plot(self.freq_axis, dB(amp_mean), label="Mean")
            #plt.plot(self.freq_axis, dB(np.abs(amp_max)), label="Max")
            plt.plot(self.freq_axis, amp_std, label="Std")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Amplitude (dB)")

        return amp_std

    def sub_meas_sim(self):
        t_sim = self.t_sim_1layer()

        sub_pnt = self.options["eval_opt"]["sub_pnt"]

        # ref1_fd, ref1_meas = self.get_ref_data(Domain.Frequency, ref_idx=10, ret_meas=True)
        ref1_fd, ref1_meas = self.sub_dataset.get_ref_data(Domain.Frequency, point=sub_pnt, ret_meas=True)
        ref2_fd, ref2_meas = self.sub_dataset.get_ref_data(Domain.Frequency, ref_idx=10, ret_meas=True)

        meas_time_diff = (ref1_meas.meas_time - ref2_meas.meas_time).total_seconds()
        print("ref1 - ref2 measurement time difference (seconds): ", np.round(meas_time_diff, 2))

        ref_amp_std = self.ref_std(en_plot=False)

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

        single_layer_eval_res = self.sub_dataset.single_layer_eval(meas_sub, (0, 10))
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
        self._opt_args["n_sub"] = n_sub

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

        self._opt_args["y_meas"] = res["t_exp_2layer"]
        self._opt_args["n_sub"] = n_sub
        self._opt_args["eps_s"] = 5
        self._opt_args["eps_inf"] = 50
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
    pass


