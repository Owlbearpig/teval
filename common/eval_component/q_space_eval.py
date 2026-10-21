import logging
import traceback
from tokenize import group

import numpy as np
import matplotlib as mpl
from matplotlib.widgets import SpanSelector
from common.eval_component.transfer_functions import TransmissionModels
from common.eval_component.eval_component_base import EvalComponentBase
from common.measurement_selection import Selection, SelectionQueue
from common.default_appsettings import SimRISelection, Domain, QSpaceQuantity, ComponentToggle
from common.functions import f_axis_idx_map, moving_average, do_ifft, to_db, avg_data_array
from common.eval_component.quantity_set import QuantityDataSet
from common.eval_component.eval_result import EvalResultData, SingleResultData
from enum import member, Enum
from common.measurements import Measurement
from common.dataset import Averaging
from common.consts import c_thz
from scipy.signal import iirnotch, filtfilt, detrend
from concurrent.futures import ProcessPoolExecutor, wait as futures_wait
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from common.eval_component.single_opt import shgo_transmission_optimization
from common.eval_component.t_fit_global import spline_transmission_optimization
from dataclasses import dataclass
from mpl_settings import mpl_style_params
from traitlets import Enum as TEnum, observe, Integer, Float, Bool, Instance, link
from common.traits import Quantity, Q_, ValueRange, PlotFigure
from common.components import action
import threading
from concurrent.futures import ThreadPoolExecutor
from common.dataset_plotter import DataSetPlotter

action = partial(action, check_init=True, rc_params=mpl_style_params)


class OptimizationCancelled(Exception):
    """Raised inside the evaluation thread when the optimization is canceled."""


def kill_process_pool(executor):
    procs = list((getattr(executor, "_processes", None) or {}).values())
    executor.shutdown(wait=False, cancel_futures=True)
    for proc in procs:
        try:
            if proc.is_alive():
                proc.kill()
        except Exception:
            pass

@dataclass(frozen=True)
class SelectionContext:
    selection: Selection
    ref_fd_dict: dict
    t_exp_dict: dict
    n_guess: dict
    t_model_kwargs: dict
    smoothing: tuple

    def model_kwargs_for(self, opt_res):
        return {**self.t_model_kwargs,
                "shift": opt_res.shift.magnitude,
                "d": opt_res.d.magnitude}

class OptimizationMethods(Enum):
    Spline = member(spline_transmission_optimization)
    SHGO = member(shgo_transmission_optimization)

class Preprocessing(DataSetPlotter):
    fd_fig = PlotFigure().tag(name="Phase correction range")

    freq_range = ValueRange(default_value=[Q_(0.300, "THz"), Q_(4.500, "THz")]).tag(
        name="Plotted frequency range", priority=1000)
    averaging = TEnum(Averaging, Averaging.only_avg).tag(name="Plot average / individual", priority=1003)

    phase_correction_grp = "Phase correction"
    phi_fit_range = ValueRange([Q_(0.47, "THz"), Q_(1.05, "THz")]).tag(
        name="Phase extrapolation range", group=phase_correction_grp, priority=999)
    auto_adjust_fit_range = Bool(False).tag(name="Auto adjust spectral fit range",  group=phase_correction_grp,
                                            priority=1000)

    fit_range_grp = "Spectral fit range"
    fit_range = ValueRange([Q_(0.47, "THz"), Q_(1.05, "THz")]).tag(
        name="Spectral fit range", group=fit_range_grp, priority=2000)
    noise_starts_at = Quantity(Q_(6.0, "THz")).tag(name="Noise starts at", group=fit_range_grp, priority=2001)
    percent_of_spectral_max = Quantity(Q_(10, "%")).tag(name="Percent of spectral max", group=fit_range_grp,
                                                       priority=2002)

    save_open_figures = None
    close_figures = None
    plt_show = None

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self._fit_range_target = None
        self._span_fit = None
        self._span_phi = None
        self._phi_ax = None
        self._phi_data_lines = []
        self._phi_fit_lines = []

        self._spectral_amp_ax = None
        self._spectral_amp_data_lines = []
        self.spectral_noise_floor_lines = []

        self.observe(self.draw_noise_floor, "noise_starts_at")
        self.observe(self.draw_noise_floor, "percent_of_spectral_max")

        self.fd_fig.subplots(2, 1, sharex=True)

    @observe("fit_range")
    def _on_fit_range(self, change):
        self._set_extents(self._span_fit, change["new"])

    @observe("phi_fit_range")
    def _on_phi_fit_range(self, change):
        self._set_extents(self._span_phi, change["new"])
        self._update_phase_fit()

    @staticmethod
    def _set_extents(span, rng):
        if span is None:
            return
        f0, f1 = (q.magnitude for q in rng)
        if not np.allclose(span.extents, (f0, f1)):
            span.extents = (f0, f1)
            span.ax.figure.canvas.draw_idle()

    def draw_noise_floor(self, change=None):
        if not self._spectral_amp_data_lines:
            return None, None, None
        noise_freq = self.noise_starts_at if change is None else change["new"]

        if self.spectral_noise_floor_lines:
            for line in self.spectral_noise_floor_lines[::-1]:
                line.remove()
                self.spectral_noise_floor_lines.pop()

        line = self._spectral_amp_data_lines[1]
        y = np.asarray(line.get_ydata(), dtype=float)
        x = np.asarray(line.get_xdata(), dtype=float)
        x_noise_starts_idx = np.argmin(np.abs(x - noise_freq.magnitude))
        noise_floor = np.mean(y[x_noise_starts_idx:])

        ax1 = self._spectral_amp_ax
        l1 = ax1.axhline(noise_floor, color="red", linestyle="--", alpha=0.7, lw=1, label="Noise floor")
        self.spectral_noise_floor_lines.append(l1)

        floor = noise_floor + np.abs(np.max(y) - noise_floor) * (self.percent_of_spectral_max.magnitude / 100)
        l2 = ax1.axhline(floor, color="deepskyblue", linestyle="--", alpha=0.7, lw=1,
                         label=f"Choice floor {self.percent_of_spectral_max.magnitude} %")
        self.spectral_noise_floor_lines.append(l2)

        with mpl.rc_context(mpl.rcParamsDefault.copy()):
            ax1.legend()

        self.fd_fig.canvas.draw_idle()

        return x, y, floor

    @action("Auto set spectral fit range", group=fit_range_grp, priority=2003)
    def auto_set_fit_range(self):
        x_min = self.phi_fit_range[0]
        x_max = Q_(1.0, "THz")

        x, y, floor = self.draw_noise_floor()
        if x is not None:

            x_max = Q_(x[np.argmax(y < floor)], "THz")

        auto_range = [x_min, x_max]
        self.set_trait("fit_range", auto_range)


    @action("Plot spectrum and phase", group=phase_correction_grp,
            rc_params=mpl.rcParamsDefault.copy(), priority=1001)
    def plot_phase_fit_selector(self):
        fig = self.fd_fig
        fig.clear()
        fig.set_layout_engine("constrained")
        fig.get_layout_engine().set(h_pad=0, hspace=0)
        ax0, ax1 = fig.subplots(2, 1, sharex=False, gridspec_kw={'hspace': 0})
        ax0.set_ylabel("Phase difference (rad)")
        ax1.set_xlabel("Frequency (THz)")
        ax1.set_ylabel("Amplitude (dB)")
        meas_list = self.plot_waveform_impl(fd_fig=fig, plot_td=False)
        if meas_list is None:
            return
        self._redraw_legend(fig)

        self._phi_ax = ax0
        self._phi_data_lines = list(ax0.lines)
        self._phi_fit_lines = []

        self._spectral_amp_ax = ax1
        self._spectral_amp_data_lines = list(ax1.lines)
        self.spectral_noise_floor_lines = []

        ax0.axvline(0, color="red", linestyle="--", alpha=0.7, lw=1)
        ax0.axhline(0, color="red", linestyle="--", alpha=0.7, lw=1)

        def _make_selector(ax, callback, color):
            return SpanSelector(ax, callback, "horizontal", useblit=False,
                                interactive=True, drag_from_anywhere=True,
                                props=dict(facecolor=color, alpha=0.2))

        def _on_fit_select(xmin, xmax):
            if xmax - xmin <= 0:
                return
            self.fit_range = [Q_(float(xmin), "THz"), Q_(float(xmax), "THz")]

        def _on_phi_select(xmin, xmax):
            if xmax - xmin <= 0:
                return
            self.phi_fit_range = [Q_(float(xmin), "THz"), Q_(float(xmax), "THz")]

        self._span_phi = _make_selector(ax0, _on_phi_select, "tab:blue")
        self._span_fit = _make_selector(ax1, _on_fit_select, "tab:green")

        self._set_extents(self._span_fit, self.fit_range)
        self._set_extents(self._span_phi, self.phi_fit_range)
        ax0.set_xlim((-0.07, 1.5))

        ax1.set_xlim(*self.freq_range.magnitude)
        ax1.legend(loc="upper right")

        if self._phi_data_lines:
            line = self._phi_data_lines[0]
            y = np.asarray(line.get_ydata(), dtype=float)
            x = np.asarray(line.get_xdata(), dtype=float)
            bottom = y[np.argmin(np.abs(x - ax0.get_xlim()[1]))]
            top = np.max([0.1 , np.max(y[np.argmin(np.abs(x - ax0.get_xlim()[0])):
                           np.argmin(np.abs(x - ax0.get_xlim()[1]))])])
            ax0.set_ylim((bottom, top))

        self._update_phase_fit()

        self.draw_noise_floor()
        if self.auto_adjust_fit_range:
            self.auto_set_fit_range()

        fig.canvas.draw_idle()

    def _update_phase_fit(self):
        ax = self._phi_ax
        if ax is None:
            return
        for ln in self._phi_fit_lines:
            try:
                ln.remove()
            except ValueError:
                pass
        self._phi_fit_lines = []

        f0, f1 = (q.to("THz").magnitude for q in self.phi_fit_range)
        for line in self._phi_data_lines:
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            m = (x >= f0) & (x <= f1) & np.isfinite(y)
            if m.sum() < 2 or np.ptp(x[m]) == 0:
                continue
            slope, offset = np.polyfit(x[m], y[m], 1)
            xs = np.array([-0.10, x.max()])
            (fit_line,) = ax.plot(xs, slope * xs + offset, ls="--", lw=1.2,
                                  color=line.get_color(), label=f"offset: {np.round(offset, 2)} (rad)",
                                  scalex=False, scaley=False)
            self._phi_fit_lines.append(fit_line)
        ax.legend(loc="upper right")
        ax.figure.canvas.draw_idle()


class QSpaceEval(EvalComponentBase):
    t_fit_grp_name = "Transmission q-space fit"
    optimization_method = TEnum(OptimizationMethods, default_value=OptimizationMethods.Spline,
                                group=t_fit_grp_name).tag(name="Optimization method", priority=-2)
    transmission_model = TEnum(TransmissionModels, default_value=TransmissionModels.tmm_1layer,
                               group=t_fit_grp_name).tag(name="Selected transmission model", priority=-1)

    d_opt_axis_bounds = ValueRange([Q_(500, "µm"), Q_(580, "µm", )],
                                   group=t_fit_grp_name).tag(name="Thickness axis bounds", priority=0)
    d_opt_axis_step = Quantity(Q_(10, "µm"), group=t_fit_grp_name).tag(name="Custom thickness axis step", priority=1)
    iterative_thickness_optimization = Bool(True,
                                            group=t_fit_grp_name).tag(name="Optimize thickness iteratively", priority=2)
    iteration_count = Integer(3, group=t_fit_grp_name).tag(name="Iteration count", priority=3)
    bound_shrink_factor = Float(0.20, group=t_fit_grp_name).tag(name="Thickness bounds shrink factor",
                                                                priority=4)

    preprocessing = Instance(Preprocessing, allow_none=True).tag(priority=1, toggle=ComponentToggle.en_preprocessing)

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

        self.current_process_executor = None

        self.opt_state = {}

        self.preprocessing = Preprocessing(self.dataset)

        self._fit_range_link = link((self, "freq_range"), (self.preprocessing, "fit_range"))
        self._phi_fit_range_link = link((self.settings.eval_opt, "phi_fit_range"),
                                        (self.preprocessing, "phi_fit_range"))

    def reset_state(self):
        self.task_cnt_carrier.cnt_changed.emit(0)
        self.opt_state["d"] = None
        self.opt_state["shift"] = Q_(0, "fs")
        self.opt_state["q_min"] = np.inf

    def ref_fd_dict(self, selection):
        ref_list = list(selection.refs)
        ref_fd = self.dataset.get_multi_data(ref_list, domain=Domain.Frequency)
        return self.format_meas_dict(ref_list, ref_fd)

    def t_exp_dict(self, selection):
        meas_list = list(selection.sams)
        t_exp = self.dataset.transmission(meas_list, list(selection.refs))
        return self.format_meas_dict(meas_list, t_exp)

    def n_guess(self, selection):
        meas_list = list(selection.sams)
        d_init = np.mean(self.d_opt_axis_bounds.magnitude)
        ref_idx = self.dataset.refractive_index_estimate(meas_list,
                                                         ref_meas_=list(selection.refs),
                                                         d=d_init)
        return self.format_meas_dict(meas_list, ref_idx)

    @action("Fit transmission model", group=t_fit_grp_name)
    def fit_unknown_layer(self):
        selection = self.dataset.measurement_selector.get_selection()
        if selection is None:
            return
        self._evaluate_queue_impl(no_queue_selection=selection)

    @action("Evaluate queue", group=t_fit_grp_name,
            enable_checker=lambda self: self.dataset.measurement_selector.selection_queue is not None)
    def evaluate_queue(self):
        queue = self.dataset.measurement_selector.selection_queue
        if queue is None:
            logging.warning("Queue is not enabled")
            return
        if len(queue) == 0:
            logging.warning("Queue is empty")
            return
        self._evaluate_queue_impl()

    def _evaluate_queue_impl(self, no_queue_selection=None):
        if self._active_eval is not None:
            logging.warning("Optimization is already running")
            return

        if no_queue_selection is not None:
            queue = SelectionQueue()
            queue.append(no_queue_selection)
        else:
            queue = self.dataset.measurement_selector.selection_queue
        self._cancel_event = threading.Event()

        evaluator = self
        self._active_eval = evaluator

        def bg_worker():
            try:
                queue_id, next_selection = queue.pop_next()
                while next_selection is not None and not self._cancel_event.is_set():
                    logging.info(f"Evaluating selection {queue_id}")
                    qs_eval_data = evaluator.eval_selection(next_selection)
                    if not self._cancel_event.is_set():
                        self._result_ready(qs_eval_data)
                    queue_id, next_selection = queue.pop_next()
            except OptimizationCancelled:
                logging.info("Optimization cancelled")
            except Exception:
                traceback.print_exc()
            finally:
                if self._active_eval is evaluator:
                    self._active_eval = None

        if self._thread_executor is None:
            self._thread_executor = ThreadPoolExecutor(max_workers=1)
        self._thread_executor.submit(bg_worker)

    def calc_uncertainties(self, opt_res: SingleResultData, selection):
        meas_list, ref_list = list(selection.sams), list(selection.refs)

        sam_fd = self.dataset.get_multi_data(meas_list, Domain.Frequency)
        ref_fd = self.dataset.get_multi_data(ref_list, Domain.Frequency)

        t_exp_amp = self.dataset.amplitude_transmission(meas_list, ref_list)
        t_exp_phi = self.dataset.phase_difference(meas_list, ref_list)

        freq_tile = np.tile(self.freq_axis, (len(meas_list), 1))
        t_exp_amp = np.stack((freq_tile, t_exp_amp[:, self.freq_range_idx], np.zeros_like(freq_tile)), axis=2)
        t_exp_phi = np.stack((freq_tile, t_exp_phi[:, self.freq_range_idx], np.zeros_like(freq_tile)), axis=2)

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
        if self.settings.eval_opt.use_separate_range:
            q_space_range = self.settings.eval_opt.q_space_range
        else:
            q_space_range = self.freq_range

        freq_axis = res_data.freq_axis.magnitude
        freq_range_idx = f_axis_idx_map(freq_axis, q_space_range)

        dt = np.mean(np.diff(freq_axis[freq_range_idx]))
        q_space_q = self.settings.eval_opt.q_space_quantity

        y = res_data.datasets["n"].data.magnitude
        match q_space_q:
            case QSpaceQuantity.n:
                y = y.real
            case QSpaceQuantity.k:
                y = y.imag
            case QSpaceQuantity.alpha:
                y = res_data.datasets[QSpaceQuantity.alpha.name].data.magnitude

        y = y[freq_range_idx]
        y = y - np.mean(y)

        y = detrend(y, type="linear")

        og_len = len(y)
        y = np.concatenate([np.zeros(3 * len(y)), y, np.zeros(3 * len(y))])

        y_ft = np.fft.rfft(y)
        t_axis = np.fft.rfftfreq(len(y), d=dt)

        q_val_axis = np.abs(y_ft)[0:]
        t_axis = t_axis[0:]

        fp_spacing = self.settings.eval_opt.fp_spacing.magnitude
        t0 = np.argmin(np.abs(t_axis - (fp_spacing - 2)))
        t1 = np.argmin(np.abs(t_axis - (fp_spacing + 2)))

        q_val, peak_idx = np.max(q_val_axis[t0:t1]), np.argmax(q_val_axis[t0:t1])
        q_sum = np.sum(q_val_axis[t0:t1])

        # plt.figure("TESTFFT")
        # plt.plot(fft_freq_axis, np.abs(fft_), label=f"shift {shift}")

        fs = 1 / np.mean(np.diff(freq_axis))
        qual_factor = 0.5  # quality factor: higher = narrower

        peak_freq = t_axis[t0:t1][peak_idx]
        b, a = iirnotch(peak_freq / (fs / 2), qual_factor)

        y_filtered = filtfilt(b, a, y)
        q_val = Q_(q_val, "")

        res_data.optimization_info["q_val"] = q_val
        res_data.optimization_info["q_sum"] = Q_(q_sum, "")

        return q_val

    def build_context(self, selection) -> SelectionContext:
        return SelectionContext(
            selection=selection,
            ref_fd_dict=self.ref_fd_dict(selection),
            t_exp_dict=self.t_exp_dict(selection),
            n_guess=self.n_guess(selection),
            t_model_kwargs=self.get_t_model_kwargs(),
            smoothing=(self.settings.eval_opt.smoothing_avg_n,
                       self.settings.eval_opt.smoothing_avg_iters),
        )

    def eval_selection(self, selection) -> EvalResultData:
        ctx = self.build_context(selection)

        is_iterative = self.iterative_thickness_optimization
        iteration_count = self.iteration_count if is_iterative else 1

        sel_meas_list = list(selection.sams)

        common_opt_params = {
            "freq_axis": self.freq_axis,
            "transmission_model": self.transmission_model.value,
            "cost_fun": self.selected_cost_fun.value,
            "minimizer_kwargs": self.settings.shgo_options.get_minimizer_kwargs(),
            "shgo_options": self.settings.shgo_options.get_shgo_options(),
            "n": self.settings.shgo_options.n,
            "iters": self.settings.shgo_options.iters,
            **ctx.t_model_kwargs,
        }
        if self.optimization_method == OptimizationMethods.Spline:
            common_opt_params["knot_count"] = self.settings.eval_opt.knot_cnt
            common_opt_params["reg_n"] = self.settings.eval_opt.reg_n
            common_opt_params["reg_k"] = self.settings.eval_opt.reg_k
            common_opt_params["max_nfev"] = self.settings.eval_opt.max_nfev
        opt_configs = {meas: {**common_opt_params, "n_guess": ctx.n_guess[meas],
                              "t_exp": ctx.t_exp_dict[meas]} for meas in ctx.t_exp_dict}

        def get_new_tasks():
            shift_bnds = self.settings.eval_opt.shift_opt_axis_bounds
            shift_step = self.settings.eval_opt.shift_opt_axis_step
            shift_axis = np.arange(shift_bnds[0].magnitude,
                                   shift_bnds[1].magnitude + shift_step.magnitude,
                                   shift_step.magnitude)
            if not self.settings.eval_opt.enable_shift_opt:
                shift_axis = [0]

            opt_d = self.opt_state["d"]
            tasks = []
            if not self.iterative_thickness_optimization or opt_d is None:
                bnds = self.d_opt_axis_bounds
                step = self.d_opt_axis_step
                num_steps = int(round((bnds[1].magnitude - bnds[0].magnitude) / step.magnitude)) + 1
                d_axis = np.linspace(bnds[0].magnitude, bnds[1].magnitude, num_steps)
            else:
                shrink_factor = self.bound_shrink_factor
                d0 = self.opt_state["d"].magnitude
                d_min = np.round(d0 * (1 - shrink_factor), 3)
                d_max = np.round(d0 * (1 + shrink_factor), 3)
                d_axis = np.linspace(d_min, d_max, 5)
            for d in d_axis:
                for shift in shift_axis:
                    tasks.append(np.array([d, shift], dtype=float))

            return tasks

        def check_cancel():
            if self._cancel_event is not None and self._cancel_event.is_set():
                raise OptimizationCancelled()

        def get_result(future):
            while not future.done():
                check_cancel()
                futures_wait([future], timeout=0.2)
            try:
                return future.result()
            except BrokenProcessPool:
                check_cancel()
                raise

        def process_tasks(tasks, opt_config, iteration_progress):
            it_idx, tot_it = iteration_progress
            if it_idx == 0:
                total_tasks = len(tasks) + 5 * (tot_it - 1)
                self.task_cnt_carrier.cnt_changed.emit(total_tasks)
                logging.info(f"Processing a total of {total_tasks} task{'s' * (total_tasks > 1)}")
            if tot_it != 1:
                logging.info(f"Starting thickness refinement iteration {it_idx+1} / {tot_it}")

            check_cancel()
            task_results = []
            with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:
                self.current_process_executor = executor
                try:
                    worker_func = partial(self.optimization_method.value, config_dict=opt_config)
                    futures = [executor.submit(worker_func, d, shift) for d, shift in tasks]

                    for future in futures:
                        res: SingleResultData = get_result(future)
                        self.task_cnt_carrier.cnt_changed.emit(1)

                        info_str = f"Finished optimizing thickness {np.round(res.d, 2)} "
                        info_str += f"with a shift of {res.shift}"
                        logging.info(info_str)

                        q_val = self.calc_q_val(res)
                        if q_val < self.opt_state["q_min"]:
                            self.opt_state["d"] = res.d
                            self.opt_state["shift"] = res.shift
                            self.opt_state["q_min"] = q_val

                        task_results.append(res)
                except BaseException:
                    kill_process_pool(executor)
                    raise
                finally:
                    self.current_process_executor = None

            task_results = sorted(task_results, key=lambda res_: res_.d)

            return task_results

        eval_result_data = EvalResultData()
        eval_result_data.result_type = "Transmission fit"
        eval_result_data.dataset_path = self.dataset.data_path
        eval_result_data.measurement_names = [meas.filepath.name for meas in sel_meas_list]
        eval_result_data.model_name = self.transmission_model.name
        eval_result_data.measurement_quantity = "Transmission"

        for meas_idx, meas in enumerate(ctx.t_exp_dict):
            check_cancel()
            if meas != "Average":
                logging.info(f"Evaluating measurement {meas.filepath.name}")
            self.reset_state()

            opt_results: list[SingleResultData] = []
            for iter_idx in range(max(1, iteration_count)):
                check_cancel()
                new_tasks = get_new_tasks()
                new_results = process_tasks(new_tasks, opt_configs[meas],
                                            iteration_progress=(iter_idx, iteration_count))
                opt_results.extend(new_results)

            ready_results = self.prepare_results(meas, opt_results, ctx)
            meas_results = {f"result_{meas_idx}_{res_idx:03d}": res for res_idx, res in enumerate(ready_results)}
            eval_result_data.results.update(meas_results)

            best_key = min(meas_results, key=lambda k: meas_results[k].optimization_info["q_val"])
            eval_result_data.optimum_results[ready_results[0].measurement] = best_key

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

        if self.is_two_layer_t_model():
            model_kwargs["n_sub"] = n_sub
            model_kwargs["h"] = self.settings.eval_opt.sim_h.magnitude
            n = n_film
        else:
            n = n_sub

        t_sim = model(n, freq_axis, **model_kwargs)

        sam_sim_fd = np.array([freq_axis, t_sim * ref_fd[:, 1]], dtype=complex).T

        sam_sim_td = do_ifft(sam_sim_fd, conj=False)

        freq_axis_quant = Q_(freq_axis, "THz")
        time_axis_quant = Q_(sam_sim_td[:, 0], "ps")
        sim_res = {"t_sim": QuantityDataSet(axes=[freq_axis_quant], data=Q_(t_sim, "")),
                   "sam_sim_fd": QuantityDataSet(axes=[freq_axis_quant], data=Q_(to_db(sam_sim_fd[:, 1]), "dB")),
                   "sam_sim_td": QuantityDataSet(axes=[time_axis_quant], data=Q_(sam_sim_td[:, 1], ""))
                   }

        return sim_res

    def prepare_results(self, meas, opt_results: list[SingleResultData], ctx: SelectionContext):
        def split_datasets(datasets):
            returned_split_datasets = {}
            for dataset_label, dataset in datasets.items():
                if np.iscomplexobj(dataset.data.magnitude):
                    if "t_" in dataset_label:
                        q_abs = Q_(np.abs(dataset.data.magnitude), dataset.data.units)
                        q_abs_uncert = Q_(np.abs(dataset.uncert.magnitude), dataset.data.units)
                        q_phi = Q_(np.angle(dataset.data.magnitude), "rad")
                        q_phi_uncert = Q_(np.angle(dataset.uncert.magnitude), dataset.data.units)
                        returned_split_datasets[f"{dataset_label}_abs"] = (
                            QuantityDataSet(axes=dataset.axes, data=q_abs,
                                            uncert=q_abs_uncert,
                                            axes_labels=dataset.axes_labels,
                                            data_label=dataset.data_label + " (Magnitude)"))
                        returned_split_datasets[f"{dataset_label}_phi"] = (
                            QuantityDataSet(axes=dataset.axes, data=q_phi,
                                            uncert=q_phi_uncert,
                                            axes_labels=dataset.axes_labels,
                                            data_label=dataset.data_label + " (Angle)"))
                    else:
                        q_real = Q_(np.real(dataset.data.magnitude), dataset.data.units)
                        q_imag = Q_(np.imag(dataset.data.magnitude), dataset.data.units)
                        returned_split_datasets[f"{dataset_label}_real"] = (
                            QuantityDataSet(axes=dataset.axes, data=q_real,
                                            uncert=Q_(np.real(dataset.uncert.magnitude), dataset.uncert.units),
                                            axes_labels=dataset.axes_labels,
                                            data_label=dataset.data_label + " (Real)"))
                        returned_split_datasets[f"{dataset_label}_imag"] = (
                            QuantityDataSet(axes=dataset.axes, data=q_imag,
                                            uncert=Q_(np.imag(dataset.uncert.magnitude), dataset.uncert.units),
                                            axes_labels=dataset.axes_labels,
                                            data_label=dataset.data_label + " (Imag)"))
                else:
                    returned_split_datasets[dataset_label] = dataset

            return returned_split_datasets

        is_avg_meas = not isinstance(meas, Measurement)

        ref_meas = ctx.selection.ref_map(meas)
        ref_fd = ctx.ref_fd_dict[ref_meas]

        norm_q_vals = self.settings.eval_opt.normalize_q_vals
        q_vals = [opt_res.optimization_info["q_val"] for opt_res in opt_results]
        for opt_res in opt_results:
            opt_res.measurement = str(meas) if is_avg_meas else meas.filepath.name

            q_val = opt_res.optimization_info["q_val"]
            opt_res.optimization_info["q_val"] = q_val / np.max(q_vals) if norm_q_vals else q_val

            if is_avg_meas:
                self.calc_uncertainties(opt_res, ctx.selection)

            t_model_kwargs = ctx.model_kwargs_for(opt_res)

            t_mod_ = self.transmission_model.value(opt_res.datasets["n"].data.magnitude,
                                                   opt_res.freq_axis.magnitude,
                                                   **t_model_kwargs)
            sam_mod_db = to_db(ref_fd[:, 1] * t_mod_)

            residual = opt_res.datasets["t_exp"].data.magnitude - t_mod_
            opt_res.datasets["t_mod"] = QuantityDataSet(axes=[opt_res.freq_axis],
                                                        data=Q_(t_mod_, ""),
                                                        axes_labels=["Frequency"],
                                                        data_label="Transmission coefficient model")
            opt_res.datasets["sam_mod"] = QuantityDataSet(axes=[opt_res.freq_axis],
                                                          data=Q_(sam_mod_db, "dB"),
                                                          axes_labels=["Frequency"],
                                                          data_label="Sample spectrum")
            opt_res.datasets["residual"] = QuantityDataSet(axes=[opt_res.freq_axis],
                                                           data=Q_(residual, ""),
                                                           axes_labels=["Frequency"],
                                                           data_label="Residual")

            if self.settings.eval_opt.add_sim_to_res:
                opt_res.datasets.update(self.calc_sim(t_model_kwargs, ref_fd))

            smoothed_quantities = ["n", "alpha"]
            for q in smoothed_quantities:
                if not self.settings.eval_opt.smoothing_avg_en:
                    continue
                if q in opt_res.datasets:
                    smoothed_data = moving_average(opt_res.datasets[q].data.magnitude, *ctx.smoothing)
                    opt_res.datasets[q].data = Q_(smoothed_data, opt_res.datasets[q].data.units)

            opt_res.datasets = split_datasets(opt_res.datasets)

        return opt_results
