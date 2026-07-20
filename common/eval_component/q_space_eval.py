import logging
from cmath import inf

import numpy as np
import scipy
from common.default_appsettings import AppSettings
from common.functions import f_axis_idx_map, moving_average
from common.eval_component.transfer_functions import model_1layer, transferfunction_error, dtdn, dtdd
from common.eval_component.quantity_set import DataSet
from common.units import Q_
from common.consts import c_thz, GREEN, RESET
from tqdm import tqdm
from scipy.optimize import shgo
from scipy.signal import iirnotch, filtfilt
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from common.eval_component.single_opt import optimize_transmission
from datetime import datetime

class QSpaceEval:

    def __init__(self, dataset_eval):
        self.dataset_eval = dataset_eval
        self.dataset = dataset_eval.dataset
        self.settings = dataset_eval.dataset.settings
        self.freq_axis = self.dataset.freq_axis

        self.model_kwargs = {}
        self.cost_fun = self.dataset_eval.cost_fun.value
        self.transmission_model = self.dataset_eval.transmission_model

        self.opt_state = {
            "d": self.settings.sample_properties.d.magnitude,
            "shift": 0,
            "q_min": np.inf
        }

    def _is_two_layer_model(self):
        try:
            # check if transmission model expects n_sub
            test_kwargs = self.model_kwargs.copy()
            test_kwargs["d"] = 1
            self.transmission_model.value(n=1, freq=1, **test_kwargs)
            return False
        except KeyError as e:
            logging.info(e)
            return True

    def set_two_layer_kwargs(self):
        self.model_kwargs["h"] = self.settings.sample_properties.d_film.magnitude

        substrate_result = self.dataset_eval.selected_substrate_result.quantity_dict
        try:
            n_sub_real_dataset = substrate_result["n"]
            n_sug_imag_dataset = substrate_result["k"]
            n_sub = n_sub_real_dataset.data.magnitude + 1j * n_sug_imag_dataset.data.magnitude
        except KeyError:
            raise Exception("Substrate result required for two layer model optimization")
        self.model_kwargs["n_sub"] = n_sub

    def set_model_kwargs(self):
        single_layer_properties = self.dataset.get_single_layer_properties()
        meas = single_layer_properties["meas"]
        ref_meas = self.dataset.get_nearest_ref(meas)

        meas_quants = self.dataset.calc_meas_quantities(ref_meas, meas)

        self.model_kwargs["meas_quants"] = meas_quants
        self.model_kwargs["single_layer_approx"] = single_layer_properties["single_layer_approx"]
        self.model_kwargs["nfp"] = self.settings.eval_opt.fp_count
        self.model_kwargs["n1"] = 1
        self.model_kwargs["n4"] = 1

        if self._is_two_layer_model():
            self.set_two_layer_kwargs()

    def calc_uncertainties(self, result):
        uncertainties = {**result}

        meas_quants = self.model_kwargs["meas_quants"]

        f_idx_plot_range = f_axis_idx_map(self.freq_axis, self.settings.eval_opt.fit_range)

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

    def calc_q_val(self, opt_res_):
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

        q_val_axis = np.abs(y_ft)[0:]
        t_axis = t_axis[0:]

        fp_spacing = self.settings.sample_properties.fp_spacing.magnitude
        t0 = np.argmin(np.abs(t_axis - (fp_spacing - 2)))
        t1 = np.argmin(np.abs(t_axis - (fp_spacing + 2)))

        # t0, t1 = 0.85*3*t_diff, 1.15*3*t_diff
        # t0_idx, t1_idx = np.argmin(np.abs(t0-t_axis)), np.argmin(np.abs(t1-t_axis))
        # print(t_axis[t0_idx], t_axis[t1_idx], t_diff)
        # t_diff = np.abs(self._delay_from_phaseslope(meas_, ref_meas_))
        # exit()

        q_val, peak_idx = np.max(q_val_axis[t0:t1]), np.argmax(q_val_axis[t0:t1])
        q_sum = np.sum(q_val_axis[t0:t1])

        # plt.figure("TESTFFT")
        # plt.plot(fft_freq_axis, np.abs(fft_), label=f"shift {shift}")

        fs = 1 / np.mean(np.diff(self.freq_axis))
        Q = 0.5  # quality factor: higher = narrower

        peak_freq = t_axis[t0:t1][peak_idx]
        # print(peak_freq, fs)
        b, a = iirnotch(peak_freq / (fs / 2), Q)

        y_filtered = filtfilt(b, a, y)

        return {"q_val": q_val, "q_sum": q_sum, "q_y": y_filtered}

    def q_space_eval_mp(self, fit_range=None, progress_carrier=None):
        if fit_range is None:
            fit_range = self.settings.eval_opt.fit_range

        f_idx_fit_range = f_axis_idx_map(self.freq_axis, fit_range)

        self.set_model_kwargs()

        opt_config = {
            "f_idx_range_": f_idx_fit_range,
            "freq_axis": self.freq_axis,
            "single_layer_approx": self.model_kwargs["single_layer_approx"],
            "t_exp": self.model_kwargs["meas_quants"]["t_exp"],
            "transmission_model": self.transmission_model.value,
            "cost_fun": self.cost_fun,
            "minimizer_kwargs": self.dataset_eval.shgo_options.get_minimizer_kwargs(),
            "shgo_options": self.dataset_eval.shgo_options.get_shgo_options()
        }

        def process_tasks(tasks, iteration=None):
            if iteration is not None:
                it_idx, tot_it = iteration
                logging.info(f"Thickness refinement iteration {it_idx} / {tot_it}")
            if iteration is None or iteration[0] == 0:
                logging.info(f"Starting optimization")

            results = []
            with ProcessPoolExecutor(max_workers=self.dataset_eval.number_of_workers) as executor:
                worker_func = partial(optimize_transmission, config_dict=opt_config)
                futures = [executor.submit(worker_func, d, shift) for d, shift in tasks]
                total_tasks = len(futures)

                for fut_idx, future in enumerate(futures):
                    res = future.result()

                    completed_tasks = fut_idx + 1
                    percentage = (completed_tasks / total_tasks) * 100

                    progress_str = f"Processed task {completed_tasks}/{total_tasks} ({percentage:.1f}%)"
                    logging.info(progress_str)
                    info_str = f"Finished optimizing thickness {np.round(res['d'], 2)} µm "
                    info_str += f"with a shift of {res['shift']} fs"
                    logging.info(info_str)

                    progress = (fut_idx + 1) / len(futures)
                    if progress_carrier is not None:
                        progress_carrier.progress_changed.emit(progress)

                    q_val_calc_res = self.calc_q_val(res)
                    res.update(q_val_calc_res)

                    q_val = q_val_calc_res["q_val"]
                    if q_val < self.opt_state["q_min"]:
                        self.opt_state["d"] = res["d"]
                        self.opt_state["shift"] = res["shift"]
                        self.opt_state["q_min"] = q_val

                    results.append(res)

            results = sorted(results, key=lambda res: res["d"])

            return results

        shift_axis = [*np.arange(-0, 3, 1.0)]
        if self.dataset_eval.use_custom_d_opt_axis:
            bnds = self.dataset_eval.d_opt_axis_bounds
            step = self.dataset_eval.d_opt_axis_step
            d_axis = np.arange(bnds[0].magnitude, bnds[1].magnitude, step.magnitude)

            tasks = []
            for d in d_axis:
                for shift in shift_axis:
                    tasks.append((d, shift))

            opt_results = process_tasks(tasks)
        else:
            iterations = 3
            step_size = [20, 5, 1]
            for i in range(iterations):
                tasks = []
                d0 = self.opt_state["d"]
                d_min = np.max((d0 - step_size[i], 0))
                d_max = np.max((d0 + step_size[i], 0))
                d_axis = np.linspace(d_min, d_max, 5)
                for d in d_axis:
                    for shift in shift_axis:
                        tasks.append((d, shift))
                opt_results = process_tasks(tasks, iteration=(i, iterations))

        q_vals = np.array([res["q_val"] for res in opt_results])
        q_vals = q_vals / np.max(q_vals)

        best_res = opt_results[np.argmin(q_vals)]
        best_res["d_vals"] = np.array([res["d"] for res in opt_results])

        best_res = self.calc_uncertainties(best_res)

        n_opt_res_ = best_res["n"] + 1j * best_res["k"]
        self.model_kwargs["shift"] = best_res["shift"]
        self.model_kwargs["d"] = best_res["d"]

        t_mod_ = self.transmission_model(n_opt_res_, self.freq_axis[f_idx_fit_range], **self.model_kwargs)

        best_res["t_mod"] = t_mod_
        best_res["sam_mod"] = self.model_kwargs["meas_quants"]["ref_fd"][f_idx_fit_range, 1] * t_mod_

        sas = (5, 20)
        smoothed_quantities = ["n", "k", "alpha"]
        for q in smoothed_quantities:
            best_res[q] = moving_average(best_res[q], iterations=sas[0], n=sas[1])

        best_res["measurement_quantity"] = "Transmission"
        best_res["model_name"] = self.transmission_model.name

        return self.prepare_result(best_res)

    def prepare_result(self, res_dict):
        rd = res_dict

        freq_axis = Q_(rd["freq_axis"], "THz")
        parsed_dict = {
            # --- Scalars ---
            "d": Q_(rd["d"], "µm"),
            "q_val": Q_(rd["q_val"], ""),
            "gof": Q_(rd["gof"], ""),
            "shift": Q_(rd["shift"], "fs"),
            "converged": True,

            # --- Strings ---
            "timestamp": str(datetime.now().isoformat()),

            # --- Datasets ( Q_(x) ) ---
            "n0_real": DataSet(axes=[freq_axis], data=Q_(rd["n0_real"], "T"),
                               data_label="Simple n", axes_labels=["Frequency"]),
            "delta_n": DataSet(axes=[freq_axis], data=Q_(rd["delta_n"], "S"),
                               data_label="delta_n", axes_labels=["Frequency"]),
            "delta_alpha": DataSet(axes=[freq_axis], data=Q_(rd["delta_alpha"], "m"), axes_labels=["Frequency"]),
            "n": DataSet(axes=[freq_axis], data=Q_(rd["n"], "nm"), axes_labels=["Frequency"]),
            "k": DataSet(axes=[freq_axis], data=Q_(rd["k"], "W"), axes_labels=["Frequency"]),
            "alpha": DataSet(axes=[freq_axis], data=Q_(rd["alpha"], "1/cm"), axes_labels=["Frequency"]),
            "t_mod": DataSet(axes=[freq_axis], data=Q_(rd["t_mod"], "V"), axes_labels=["ABE"]),
            "sam_mod": DataSet(axes=[freq_axis], data=Q_(rd["sam_mod"], "J")),
        }

        parsed_dict["result_type"] = "Transmission fit"

        return parsed_dict
