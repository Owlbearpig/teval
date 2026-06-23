import numpy as np
import matplotlib.pyplot as plt
import scipy
from common.functions import f_axis_idx_map, moving_average
from common.eval_component.transfer_functions import model_1layer, transferfunction_error, dtdn, dtdd
from common.consts import c_thz, GREEN, RESET
from tqdm import tqdm
from scipy.optimize import shgo
from common.eval_component.dataset_eval import DatasetEval
from common.eval_component.shgo_settings import SHGOOptions, MinimizerOptions

class QSpaceEval:

    def __init__(self, dataset_eval: DatasetEval):
        self.dataset_eval = dataset_eval
        self.dataset = dataset_eval.dataset
        self.settings = dataset_eval.dataset.settings
        self.freq_axis = self.dataset.freq_axis

        self.opt_consts = {}
        self.cost_fun = self.dataset_eval.cost_fun
        self.transmission_model = self.dataset_eval.transmission_model

        self.opt_state = {
            "d": self.settings.sample_properties.d,
            "q_min": np.inf
        }

    def set_opt_config(self):
        single_layer_properties = self.dataset.get_single_layer_properties()
        meas = single_layer_properties["meas"]
        ref_meas = self.dataset.get_nearest_ref(meas)

        meas_quants = self.dataset.calc_meas_quantities(ref_meas, meas)

        self.opt_consts["meas_quants"] = meas_quants
        self.opt_consts["single_layer_approx"] = single_layer_properties["single_layer_approx"]

    def calc_uncertainties(self, result):
        uncertainties = {**result}

        meas_quants = self.opt_consts["meas_quants"]

        f_idx_plot_range = f_axis_idx_map(self.freq_axis, self.settings.plot_opt.plot_range)

        sam_fd_, sam_fd_std = meas_quants["sam_fd"], meas_quants["sam_fd_std"]
        ref_fd_, ref_fd_std = meas_quants["ref_fd"], meas_quants["ref_fd_std"]
        delta_amp = meas_quants["t_exp_amp_std"][f_idx_plot_range, 1]
        delta_phi = meas_quants["t_exp_phi_std"][f_idx_plot_range, 1]
        amp = meas_quants["t_exp_amp"][f_idx_plot_range, 1]
        phi = meas_quants["t_exp_phi"][f_idx_plot_range, 1]

        f_axis = self.freq_axis[f_idx_plot_range]
        w = 2 * np.pi * f_axis

        delta_t = transferfunction_error(sam_fd_, ref_fd_, ref_fd_std, sam_fd_std, noise_freq=5.0)
        delta_t = delta_t[f_idx_plot_range]
        n, d = result["n"] + 1j * result["k"], result["d"]

        dtdn_ = dtdn(n, d, f_axis)
        dtdd_ = dtdd(n, d, f_axis)

        delta_d = self.settings.eval_opt.delta_d

        uncertainties["delta_n"] = np.sqrt(((1 / dtdn_) * delta_t) ** 2 + ((1 / dtdn_) * dtdd_ * delta_d) ** 2)
        uncertainties["delta_alpha"] = (4 * np.pi * f_axis / (1e-4 * c_thz)) * uncertainties["delta_n"].imag

        delta_k_term1 = delta_phi * -(c_thz / (w * d)) ** 2 * (n.real - 1) / (n.real * (n.real + 1))
        delta_k_term2 = delta_amp * (c_thz / (w * d)) * (1 / amp)
        delta_k_term3 = delta_d * (c_thz / (w * d ** 2)) * np.log(amp * (1 + n.real) ** 2 / (4 * n))
        delta_k = np.sqrt(np.abs(delta_k_term1) ** 2 + np.abs(delta_k_term2) ** 2 + np.abs(delta_k_term3) ** 2)

        delta_n_term1 = delta_phi * c_thz / (w * d)
        delta_n_term2 = delta_d * (-phi * c_thz) / (w * d ** 2)

        uncertainties["delta_alpha"] = np.abs(4 * np.pi * f_axis * delta_k / (1e-4 * c_thz))
        uncertainties["delta_n"] = np.sqrt(np.abs(delta_n_term1) ** 2 + np.abs(delta_n_term2) ** 2) + 1j * delta_k

        return uncertainties

    def calc_q_val(self, opt_res_, en_plot=False):
        q_space_range = self.settings.eval_opt.q_space_range
        q_space_idx_range = f_axis_idx_map(opt_res_["freq_axis"], q_space_range)

        dt = np.mean(np.diff(opt_res_["freq_axis"][q_space_idx_range]))
        # y = opt_res_["n"][q_space_idx_range]
        y = opt_res_["k"][q_space_idx_range]
        y = y - np.mean(y)

        y = scipy.signal.detrend(y, type="linear")

        y = np.array([opt_res_["freq_axis"][q_space_idx_range], y]).T
        # y = window(y, win_width=len(y), win_start=0, shift=40, en_plot=True, type=WindowTypes.hann)

        y = y[:, 1]

        y = np.concatenate([np.zeros(3 * len(y)), y, np.zeros(3 * len(y))])

        y_ft = np.fft.rfft(y)
        t_axis = np.fft.rfftfreq(len(y), d=dt)

        q_val = np.abs(y_ft)[0:]
        t_axis = t_axis[0:]

        fp_spacing = self.settings.sample_properties.fp_spacing
        t0 = np.argmin(np.abs(t_axis - (fp_spacing - 2)))
        t1 = np.argmin(np.abs(t_axis - (fp_spacing + 2)))
        if en_plot:
            plt.figure("qval")
            plt.plot(t_axis, q_val, label=str(opt_res_["d"]) + " µm")
            plt.axvline(x=t_axis[t0], color='r', linestyle='--', linewidth=2)
            plt.axvline(x=t_axis[t1], color='r', linestyle='--', linewidth=2)
            # plt.plot(q_val, label=opt_res_["d"])
            plt.xlabel("Time (ps)")
            plt.ylabel("Oscillation amplitude")
            plt.legend()
            plt.show()

        # t0, t1 = 0.85*3*t_diff, 1.15*3*t_diff
        # t0_idx, t1_idx = np.argmin(np.abs(t0-t_axis)), np.argmin(np.abs(t1-t_axis))
        # print(t_axis[t0_idx], t_axis[t1_idx], t_diff)
        # t_diff = np.abs(self._delay_from_phaseslope(meas_, ref_meas_))
        # exit()

        return np.max(q_val[t0:t1])

    def model_opt(self, d_, f_idx_range_):
        min_kwargs_comp = self.dataset_eval.shgo_options.minimizer_kwargs
        minimizer_kwargs = min_kwargs_comp.traits(group=MinimizerOptions.minimizer_opt_grp)
        minimizer_kwargs["method"] = min_kwargs_comp.method

        shgo_options = self.dataset_eval.shgo_options.traits(group=SHGOOptions.shgo_options_grp)

        freq_axis = self.freq_axis[f_idx_range_]

        n0 = self.opt_consts["single_layer_approx"]["refr_idx"]
        n0_ = n0[f_idx_range_]

        t_exp = self.opt_consts["meas_quants"]["t_exp"]
        t_exp_ = t_exp[f_idx_range_, 1]  # * np.exp(1j * 2*np.pi*freq_axis*0.150)

        # phi_corrected = self.ana_eval_res["phi_corrected"][f_idx_range_]
        # t_exp_ = np.abs(t_exp_) * np.exp(1j * phi_corrected)

        n_opt_res_ = np.zeros_like(freq_axis, dtype=complex)
        for f_idx, f_ in enumerate(freq_axis):
            args = [f_, d_]
            def opt_fun(p):
                n = p[0] + 1j * p[1]
                t_mod_ = self.transmission_model(n, *args)

                return self.cost_fun(t_exp_[f_idx], t_mod_)


            n0_f_idx = n0_[f_idx]
            n_min, n_max = 0.90 * n0_f_idx.real, 1.10 * n0_f_idx.real
            k_min, k_max = 0.10 * n0_f_idx.imag, 1.10 * n0_f_idx.imag
            bounds = [(n_min, n_max), (k_min, k_max)]

            conv, i_ = False, 0
            while not conv:
                i_ += 1
                shgo_opt_res_ = shgo(opt_fun,
                                     bounds=bounds,
                                     minimizer_kwargs=minimizer_kwargs,
                                     options=shgo_options,
                                     # n=1, iters=200,
                                     )

                x = shgo_opt_res_.x
                n_opt_res_[f_idx] = x[0] + 1j * x[1]
                if f_idx == 0:
                    break

                diff = (n_opt_res_[f_idx].real - n_opt_res_[f_idx - 1].real)
                if np.abs(diff) < 0.10:
                    conv = True
                else:
                    c0, c1 = 0.90 + i_ * 0.02, 1.10 - i_ * 0.02
                    n_bounds = (n0_f_idx.real * c0, n0_f_idx.real * c1)
                    k_bounds = (n0_f_idx.imag * c0, n0_f_idx.imag * c1)

                    bounds = [(min(n_bounds), max(n_bounds)), (min(k_bounds), max(k_bounds))]
                if i_ > 4:
                    n_prev = n_opt_res_[f_idx - 1]
                    c0, c1 = 0.90 + i_ * 0.01, 1.10 - i_ * 0.01
                    n_bounds = (n_prev.real * c0, n_prev.real * c1)
                    k_bounds = (n_prev.imag * c0, n_prev.imag * c1)

                    bounds = [(min(n_bounds), max(n_bounds)), (min(k_bounds), max(k_bounds))]
                if i_ > 5:
                    break

        alpha_ = freq_axis * 4 * np.pi * n_opt_res_.imag / (1e-4 * c_thz)

        opt_res_ = {"freq_axis": freq_axis,
                    "d": d_, "n": n_opt_res_.real,
                    "k": n_opt_res_.imag, "alpha": alpha_, "n0": n0_}
        # fp_spacing_estimate = ...
        opt_res_["q_val"] = self.calc_q_val(opt_res_, en_plot=False)

        t_mod_ = model_1layer(n_opt_res_, d=d_, freq=freq_axis, n1=1, shift_=0)

        opt_res_["t_mod"] = t_mod_
        opt_res_["sam_mod"] = self.opt_consts["meas_quants"]["ref_fd"][f_idx_range_, 1] * t_mod_

        return opt_res_

    def q_space_eval(self, fit_range=None, q_space_range=None, **kwargs):
        if fit_range is None:
            fit_range = self.settings.eval_opt.fit_range

        """ # TODO add to final result dict and plot
        for k in simple_eval_res:
            plt.figure()
            plt.plot(self.freq_axis, simple_eval_res[k], label=k)
            plt.legend()
        plt.show()
        """

        f_idx_fit_range = f_axis_idx_map(self.freq_axis, fit_range)
        f_idx_plot_range = f_axis_idx_map(self.freq_axis, self.settings.plot_opt.plot_range)

        opt_results = []
        def opt_d_axis(d_axis_, it_prog=None):
            custom_format = f"{GREEN}{{l_bar}}{{bar}}{GREEN}{{r_bar}}{RESET}"
            pbar_ = tqdm(d_axis_, total=len(d_axis_), colour="green", bar_format=custom_format)
            for d in pbar_:
                desc = ""
                if it_prog:
                    desc += f"Step {it_prog[0] + 1}/{it_prog[1]}: "

                desc += f"Optimizing thickness {np.round(d, 2)} µm"
                desc += f" of {d_axis_}"
                pbar_.set_description(desc)
                opt_res = self.model_opt(d, f_idx_fit_range)
                opt_results.append(opt_res)

                q_val = opt_res["q_val"]
                if q_val < self.opt_state["q_min"]:
                    self.opt_state["d"] = d
                    self.opt_state["q_min"] = q_val

        d_axis = self.settings["eval_opt"]["d_opt_axis"]
        if d_axis is not None:
            opt_d_axis(d_axis)
        else:
            iterations = 3
            step_size = [20, 5, 1]
            for i in range(iterations):
                d0 = self.opt_state["d"]
                d_min, d_max = d0 - step_size[i], d0 + step_size[i]
                d_axis = np.linspace(d_min, d_max, 5)

                opt_d_axis(d_axis, it_prog=(i, iterations))

        opt_results = sorted(opt_results, key=lambda res: res["d"])

        d_vals = np.array([res["d"] for res in opt_results])
        q_vals = np.array([res["q_val"] for res in opt_results])
        q_vals = q_vals / np.max(q_vals)

        opt_d_res = opt_results[np.argmin(q_vals)]
        d_opt = np.round(opt_d_res["d"], 2)

        best_res = self.model_opt(d_opt, f_idx_plot_range)

        best_res = self.calc_uncertainties(best_res)

        delta_n, delta_alpha = best_res["delta_n"], best_res["delta_alpha"]

        sas = (5, 20)
        smoothed_quantities = ["n", "k", "alpha"]
        for q in smoothed_quantities:
            best_res[q] = moving_average(best_res[q], iterations=sas[0], n=sas[1])

        fig_num = "Optimal result"
        if not plt.fignum_exists(fig_num):
            fig, (ax0, ax1) = plt.subplots(2, 1, num=fig_num, sharex=True, gridspec_kw={"hspace": 0})
            ax1.set_xlabel("Frequency (THz)")
            ax0.set_ylabel("Refractive index")
            ax1.set_ylabel("Extinction coefficient")
        else:
            fig = plt.figure(fig_num)
            ax0, ax1 = fig.get_axes()
        if "label" in kwargs:
            label = kwargs["label"] + f" ({d_opt} µm)"
        else:
            label = f"{d_opt} µm"

        ax0.plot(best_res["freq_axis"], best_res["n"], label=label)
        ax0.fill_between(best_res["freq_axis"],
                         best_res["n"] + delta_n.real,
                         best_res["n"] - delta_n.real, alpha=0.3)

        ax1.plot(best_res["freq_axis"], best_res["k"], label=label)
        ax1.fill_between(best_res["freq_axis"],
                         best_res["k"] + delta_n.imag,
                         best_res["k"] - delta_n.imag, alpha=0.3)

        plt.figure("Absorption coefficient optimum")
        plt.plot(best_res["freq_axis"], best_res["alpha"], label=label)
        plt.fill_between(best_res["freq_axis"],
                         best_res["alpha"] + delta_alpha,
                         best_res["alpha"] - delta_alpha, alpha=0.3)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Absorption coefficient (1/cm)")

        plt.figure("Q-Space maxima")
        plt.plot(d_vals, q_vals, marker="o")
        plt.xlabel("Thickness (µm)")
        plt.ylabel("Normalized maximum of q-space")
        # plt.legend()
        # plt.show()

        plt.figure("Spectrum")
        plt.plot(best_res["freq_axis"], 20 * np.log10(np.abs(best_res["sam_mod"])), label="Model")

        return best_res

