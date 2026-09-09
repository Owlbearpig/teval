import logging
import traceback
import numpy as np
import scipy
from common.dataset import format_meas_dict, DataSet
from common.default_appsettings import SimRISelection, AppSettings, Domain
from common.functions import f_axis_idx_map, moving_average, do_ifft, to_db, avg_data_array
from common.eval_component.transfer_functions import model_1layer, transferfunction_error, dtdn, dtdd
from common.eval_component.quantity_set import QuantityDataSet
from common.eval_component.eval_result import EvalResultData, SingleResultData
from common.units import Q_
from common.measurements import Measurement
from common.consts import c_thz
from scipy.optimize import shgo
from scipy.signal import iirnotch, filtfilt, detrend
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from common.eval_component.single_opt import optimize_transmission
from datetime import datetime

class QSpaceEval:

    def __init__(self, dataset_eval):
        self.dataset_eval = dataset_eval
        self.settings = dataset_eval.dataset.settings

        self.cost_fun = self.dataset_eval.selected_cost_fun.value
        self.transmission_model = self.dataset_eval.transmission_model

        self.opt_state = {}

    def reset_opt_state(self):
        self.opt_state["d"] = self.settings.eval_opt.d
        self.opt_state["shift"] = Q_(0, "fs")
        self.opt_state["q_min"] = np.inf

    @property
    def selected_measurements(self):
        return self.dataset_eval.dataset.measurement_selector.selected_measurements

    @property
    def freq_axis(self):
        return self.dataset_eval.freq_axis

    @property
    def freq_idx(self):
        return self.dataset_eval.f_idx

    @property
    def ref_fd_dict(self):
        ref_list = self.dataset_eval.dataset.measurement_selector.get_matching_refs(self.selected_measurements)
        ref_fd = self.dataset_eval.dataset.get_multi_data(ref_list)

        ref_fd = ref_fd[:, self.freq_idx]

        return format_meas_dict(ref_list, ref_fd, self.dataset_eval.only_eval_avg)

    @property
    def t_exp_dict(self):
        meas_list = self.selected_measurements

        t_exp = self.dataset_eval.dataset.transmission(meas_list)

        f_axis_tile = np.tile(self.freq_axis, (len(meas_list), 1))
        arrays = (f_axis_tile, t_exp[:, self.freq_idx], np.zeros_like(f_axis_tile))
        t_exp_stacked = np.stack(arrays, axis=2)

        return format_meas_dict(meas_list, t_exp_stacked, self.dataset_eval.only_eval_avg)

    @property
    def n_guess(self):
        meas_list = self.selected_measurements
        ref_idx = self.dataset_eval.dataset.tof_refractive_index(meas_list)

        f_axis_tile = np.tile(self.freq_axis, (len(meas_list), 1))
        arrays = (f_axis_tile, ref_idx[:, self.freq_idx], np.zeros_like(f_axis_tile))
        ref_idx_stacked = np.stack(arrays, axis=2)

        return format_meas_dict(meas_list, ref_idx_stacked, self.dataset_eval.only_eval_avg)

    def calc_uncertainties(self, opt_res: SingleResultData, meas_list):
        meas_list = [meas for meas in meas_list if meas != "Average"]
        ref_list = self.dataset_eval.dataset.measurement_selector.get_matching_refs(meas_list)

        sam_fd = self.dataset_eval.dataset.get_multi_data(meas_list, Domain.Frequency)
        ref_fd = self.dataset_eval.dataset.get_multi_data(ref_list, Domain.Frequency)

        t_exp_amp = self.dataset_eval.dataset.amplitude_transmission(meas_list)
        t_exp_phi = self.dataset_eval.dataset.phase_difference(meas_list)

        freq_tile = np.tile(self.freq_axis, (len(meas_list), 1))
        t_exp_amp = np.stack((freq_tile, t_exp_amp[:, self.freq_idx], np.zeros_like(freq_tile)), axis=2)
        t_exp_phi = np.stack((freq_tile, t_exp_phi[:, self.freq_idx], np.zeros_like(freq_tile)), axis=2)

        sam_fd_avg = avg_data_array(sam_fd)
        ref_fd_avg = avg_data_array(ref_fd)
        t_exp_amp_avg = avg_data_array(t_exp_amp)
        phi_avg = avg_data_array(t_exp_phi)

        amp = t_exp_amp_avg[:, 1]
        phi = phi_avg[:, 1]
        delta_amp = t_exp_amp_avg[:, 2]
        delta_phi = phi_avg[:, 2]

        f_axis = self.freq_axis
        w = 2 * np.pi * f_axis

        n, d = opt_res.datasets["n"].data.magnitude, opt_res.d.magnitude
        """
        dtdn_ = dtdn(n, d, f_axis)
        dtdd_ = dtdd(n, d, f_axis)

        delta_t = transferfunction_error(sam_fd_avg, ref_fd_avg, noise_freq=5.0)
        delta_t = delta_t[self.freq_idx]
        
        delta_n = np.sqrt(((1 / dtdn_) * delta_t) ** 2 + ((1 / dtdn_) * dtdd_ * delta_d) ** 2)
        delta_alpha = (4 * np.pi * f_axis / (1e-4 * c_thz)) * delta_n.imag
        """

        delta_d = self.settings.eval_opt.delta_d.magnitude

        delta_k_term1 = delta_phi * -(c_thz / (w * d)) ** 2 * (n.real - 1) / (n.real * (n.real + 1))
        delta_k_term2 = delta_amp * (c_thz / (w * d)) * (1 / amp)
        delta_k_term3 = delta_d * (c_thz / (w * d ** 2)) * np.log(amp * (1 + n.real) ** 2 / (4 * n))
        delta_k = np.sqrt(np.abs(delta_k_term1) ** 2 + np.abs(delta_k_term2) ** 2 + np.abs(delta_k_term3) ** 2)

        delta_n_term1 = delta_phi * c_thz / (w * d)
        delta_n_term2 = delta_d * (-phi * c_thz) / (w * d ** 2)

        opt_res.datasets["alpha"].uncert = Q_(np.abs(4 * np.pi * f_axis * delta_k / (1e-4 * c_thz)), "1/cm")
        opt_res.datasets["n"].uncert = Q_(np.sqrt(np.abs(delta_n_term1) ** 2 + np.abs(delta_n_term2) ** 2)
                                          + 1j * delta_k, "")

    def calc_q_val(self, res_data: SingleResultData):
        q_space_range = self.settings.eval_opt.q_space_range
        freq_axis = res_data.freq_axis.magnitude
        q_space_idx_range = f_axis_idx_map(freq_axis, q_space_range)

        dt = np.mean(np.diff(freq_axis[q_space_idx_range]))
        # y = opt_res_["n"][q_space_idx_range]
        y = res_data.datasets["n"].data.magnitude[q_space_idx_range].imag
        y = y - np.mean(y)

        y = detrend(y, type="linear")

        # y = np.array([freq_axis[q_space_idx_range], y]).T
        # y = window(y, win_width=len(y), win_start=0, shift=40, en_plot=True, type=WindowTypes.hann)
        # y = y[:, 1]
        og_len = len(y)
        y = np.concatenate([np.zeros(3 * len(y)), y, np.zeros(3 * len(y))])

        y_ft = np.fft.rfft(y)
        t_axis = np.fft.rfftfreq(len(y), d=dt)

        q_val_axis = np.abs(y_ft)[0:]
        t_axis = t_axis[0:]

        fp_spacing = self.settings.eval_opt.fp_spacing.magnitude
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

        fs = 1 / np.mean(np.diff(freq_axis))
        qual_factor = 0.5  # quality factor: higher = narrower

        peak_freq = t_axis[t0:t1][peak_idx]
        # print(peak_freq, fs)
        b, a = iirnotch(peak_freq / (fs / 2), qual_factor)

        y_filtered = filtfilt(b, a, y)
        q_val = Q_(q_val, "")

        res_data.optimization_info["q_val"] = q_val
        res_data.optimization_info["q_sum"] = Q_(q_sum, "")

        return q_val

    def q_space_eval_mp(self, progress_carrier=None) -> EvalResultData:
        t_model_kwargs = self.dataset_eval.get_t_model_kwargs()

        shift_axis = [*np.arange(-0, 3, 1.0)]
        iterations = 3
        step_size = [20, 5, 1]
        sas = (5, 20) # smoothing avg settings
        is_iterative = not self.dataset_eval.use_custom_d_opt_axis
        ref_fd_dict = self.ref_fd_dict
        ref_sam_map = self.dataset_eval.dataset.measurement_selector.sam_ref_meas_map
        t_exp_dict = self.t_exp_dict
        n_guess = self.n_guess
        meas_list = list(t_exp_dict.keys())
        common_opt_params = {
            "freq_axis": self.freq_axis,
            "transmission_model": self.transmission_model.value,
            "cost_fun": self.cost_fun,
            "minimizer_kwargs": self.settings.shgo_options.get_minimizer_kwargs(),
            "shgo_options": self.settings.shgo_options.get_shgo_options(),
            **t_model_kwargs,
        }
        opt_configs = {meas: {**common_opt_params, "n_guess": n_guess[meas],
                              "t_exp": t_exp_dict[meas]} for meas in meas_list}

        def get_new_tasks():
            tasks = []
            if self.dataset_eval.use_custom_d_opt_axis:
                bnds = self.dataset_eval.d_opt_axis_bounds
                step = self.dataset_eval.d_opt_axis_step
                d_axis = np.arange(bnds[0].magnitude, bnds[1].magnitude+step.magnitude, step.magnitude)
            else:
                d0 = self.opt_state["d"]
                d_min = np.max((d0 - step_size[i], 0))
                d_max = np.max((d0 + step_size[i], 0))
                d_axis = np.linspace(d_min, d_max, 5)

            for d in d_axis:
                for shift in shift_axis:
                    tasks.append((d, shift))

            return tasks

        def process_tasks(tasks, opt_config, iteration_progress=None):
            if iteration_progress is not None:
                it_idx, tot_it = iteration_progress
                logging.info(f"Thickness refinement iteration {it_idx} / {tot_it}")
            if iteration_progress is None or iteration_progress[0] == 0:
                logging.info(f"Starting optimization")

            results = []
            with ProcessPoolExecutor(max_workers=self.dataset_eval.number_of_workers) as executor:
                worker_func = partial(optimize_transmission, config_dict=opt_config)
                futures = [executor.submit(worker_func, d, shift) for d, shift in tasks]
                total_tasks = len(futures)

                for fut_idx, future in enumerate(futures):
                    res: SingleResultData = future.result()

                    completed_tasks = fut_idx + 1
                    percentage = (completed_tasks / total_tasks) * 100

                    progress_str = f"Processed task {completed_tasks}/{total_tasks} ({percentage:.1f}%)"
                    logging.info(progress_str)
                    info_str = f"Finished optimizing thickness {np.round(res.d, 2)} "
                    info_str += f"with a shift of {res.shift}"
                    logging.info(info_str)

                    if progress_carrier is not None:
                        progress_carrier.progress_changed.emit(percentage/100)

                    q_val = self.calc_q_val(res)
                    if q_val < self.opt_state["q_min"]:
                        self.opt_state["d"] = res.d
                        self.opt_state["shift"] = res.shift
                        self.opt_state["q_min"] = q_val

                    results.append(res)

            results = sorted(results, key=lambda res_: res_.d)

            return results

        meas_to_str = lambda meas: meas.filepath.name if isinstance(meas, Measurement) else str(meas)
        meas_names = {meas: meas_to_str(meas) for meas in meas_list}

        eval_result_data = EvalResultData()
        eval_result_data.result_type = "Transmission fit"
        eval_result_data.dataset_path = self.dataset_eval.dataset.data_path
        eval_result_data.measurement_names = list(meas_names.values())
        eval_result_data.model_name = self.transmission_model.name
        eval_result_data.measurement_quantity = "Transmission"

        for meas in meas_list:
            ref_meas = ref_sam_map(meas)
            self.reset_opt_state()

            opt_results: list[SingleResultData] = []
            for i in range(max(1, iterations)):
                it_prog = (i, iterations) if is_iterative else None
                new_tasks = get_new_tasks()
                opt_results.extend(process_tasks(new_tasks, opt_configs[meas], iteration_progress=it_prog))
                if not is_iterative:
                    break

            for opt_res in opt_results:
                opt_res.measurement = meas_to_str(meas)
                if meas == "Average":
                    self.calc_uncertainties(opt_res, meas_list)

                t_model_kwargs["shift"] = opt_res.shift.magnitude
                t_model_kwargs["d"] = opt_res.d.magnitude

                t_mod_ = self.transmission_model.value(opt_res.freq_axis.magnitude,
                                                       opt_res.datasets["n"].data.magnitude,
                                                       **t_model_kwargs)
                sam_mod_db = to_db(ref_fd_dict[ref_meas][:, 1] * t_mod_)
                opt_res.datasets["t_mod"] = QuantityDataSet(axes=[opt_res.freq_axis],
                                                            data=Q_(t_mod_, ""),
                                                            axes_labels=["Frequency"],
                                                            data_label="Transmission coefficient model")
                opt_res.datasets["sam_mod"] = QuantityDataSet(axes=[opt_res.freq_axis],
                                                              data=Q_(sam_mod_db, "dB"),
                                                              axes_labels=["Frequency"],
                                                              data_label="Sample spectrum")

                if self.dataset_eval.add_sim_to_res:
                    opt_res.datasets.update(self.calc_sim(t_model_kwargs, ref_fd_dict[ref_meas]))

                smoothed_quantities = ["n", "alpha"]
                for q in smoothed_quantities:
                    if q in opt_res.datasets:
                        smoothed_data = moving_average(opt_res.datasets[q].data.magnitude, *sas)
                        opt_res.datasets[q].data = Q_(smoothed_data, opt_res.datasets[q].data.units)

            eval_result_data.results.extend(self.prepare_results(opt_results))

        return eval_result_data

    def calc_sim(self, model_kwargs, ref_fd):
        model = self.transmission_model.value

        freq_axis = self.freq_axis
        one = np.ones_like(freq_axis, dtype=float)

        if self.settings.eval_opt.sim_n_selection == SimRISelection.const:
            sim_n_sub = self.settings.eval_opt.sim_n_sub
            n_sub = (sim_n_sub[0] + 1j * sim_n_sub[1]) * one

            sim_n_film = self.settings.eval_opt.sim_n_film
            n_film = (sim_n_film[0] + 1j * sim_n_film[1]) * one
        else:
            n_sub = 1 * one
            n_film = 1 * one

        model_kwargs = {k: v for k, v in model_kwargs.items()}
        model_kwargs["d"] = self.settings.eval_opt.sim_d.magnitude
        model_kwargs["nfp"] = self.settings.eval_opt.sim_nfp
        model_kwargs["shift"] = self.settings.eval_opt.sim_shift.magnitude

        if self.dataset_eval.is_two_layer_t_model():
            model_kwargs["n_sub"] = n_sub
            model_kwargs["h"] = self.settings.eval_opt.sim_h.magnitude
            n = n_film
        else:
            n = n_sub

        t_sim = model(freq_axis, n, **model_kwargs)

        sam_sim_fd = np.array([freq_axis, t_sim * ref_fd[:, 1]], dtype=complex).T

        sam_sim_td = do_ifft(sam_sim_fd, conj=False)

        freq_axis_quant = Q_(freq_axis, "THz")
        time_axis_quant = Q_(sam_sim_td[:, 0], "ps")
        sim_res = {"t_sim": QuantityDataSet(axes=[freq_axis_quant], data=Q_(t_sim, "")),
                   "sam_sim_fd": QuantityDataSet(axes=[freq_axis_quant], data=Q_(to_db(sam_sim_fd[:, 1]), "dB")),
                   "sam_sim_td": QuantityDataSet(axes=[time_axis_quant], data=Q_(sam_sim_td[:, 1], ""))
                   }

        return sim_res

    def prepare_results(self, opt_results: list[SingleResultData]):
        norm_q_vals = self.dataset_eval.normalize_q_vals
        q_vals = [opt_res.optimization_info["q_val"] for opt_res in opt_results]
        for opt_res in opt_results:
            q_val = opt_res.optimization_info["q_val"]
            opt_res.optimization_info["q_val"] = q_val / np.max(q_vals) if norm_q_vals else q_val

        return opt_results
