import logging
import time
import traceback
import numpy as np
import matplotlib as mpl
from matplotlib.widgets import SpanSelector
from common.eval_component.transfer_functions import TransmissionModels
from common.eval_component.eval_component_base import EvalComponentBase
from common.measurement_selection import Selection, SelectionQueue
from common.default_appsettings import SimRISelection, Domain, ComponentToggle, AppSettings
from common.functions import f_axis_idx_map, moving_average, do_ifft, to_db, avg_data_array
from common.eval_component.quantity_set import QuantityDataSet
from common.eval_component.eval_result import EvalResultData, SingleResultData
from enum import member, Enum
from common.measurements import Measurement
from common.dataset import Averaging
from common.consts import c_thz
from scipy.signal import find_peaks
from concurrent.futures import ProcessPoolExecutor, wait as futures_wait
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from common.eval_component.single_opt import shgo_transmission_optimization
from common.eval_component.t_fit_global import spline_transmission_optimization
from common.eval_component.n_guess import NGuessMethod
from common.eval_component.layer_stack import LayerStack
from common.eval_component.q_space_scan import ScanTarget, OptimizationCancelled
from common.eval_component.q_value import fp_round_trip_time, compute_q_value, optimize_and_rate
from common.eval_component import fast_kernels
from common.eval_component.q_space_image import QSpaceImageEval, alpha_of
from common.eval_component.dynamic_range import noise_floor, alpha_max_dataset, limit_frequency, alpha_max
from common.eval_component.uncertainty import (estimator_for, random_generator, perturbed_transmissions,
                                               replicate_optimum, ThicknessUncertainty, TransmissionUncertainty)
from dataclasses import dataclass
from contextlib import contextmanager
from mpl_settings import mpl_style_params
from traitlets import Enum as TEnum, observe, Integer, Float, Bool, Instance, Unicode, link
from common.traits import Quantity, Q_, ValueRange, PlotFigure
from common.components import action
from PySide6.QtCore import QObject, Signal
import threading
from concurrent.futures import ThreadPoolExecutor
from common.dataset_plotter import DataSetPlotter
from common.remote.client import active_connection, HybridExecutor

action = partial(action, check_init=True, rc_params=mpl_style_params)


def estimate_fp_echo_count(time_traces, t_rt):
    counts = []
    for trace in time_traces:
        t, y_abs = trace[:, 0].real, np.abs(trace[:, 1].real)
        recorded = np.flatnonzero(y_abs > 1e-6 * y_abs.max())
        counts.append((t[recorded[-1]] - t[np.argmax(y_abs)]) / t_rt)

    return max(int(np.floor(min(counts))), 0)


class OptimumCriterion(Enum):
    prominent_min = "Most prominent local minimum"
    global_min = "Global minimum"


def select_optimum(results, criterion=OptimumCriterion.prominent_min):
    def q_of(res):
        return float(np.asarray(res.optimization_info["q_val"].magnitude))

    global_best = min(results, key=q_of)
    if criterion == OptimumCriterion.global_min:
        return global_best

    best_per_d = {}
    for res in results:
        d = round(float(res.d.magnitude), 6)
        if d not in best_per_d or q_of(res) < q_of(best_per_d[d]):
            best_per_d[d] = res
    profile = [best_per_d[d] for d in sorted(best_per_d)]
    if len(profile) < 3:
        return global_best

    log_q = np.log(np.maximum([q_of(res) for res in profile], 1e-300))
    minima, props = find_peaks(-log_q, prominence=0)
    if not len(minima):
        return global_best

    return profile[minima[np.argmax(props["prominences"])]]


class TaskStatusSignal(QObject):
    # (trait name, text), emitted from the evaluation thread and applied in the GUI thread
    changed = Signal(str, str)


def kill_process_pool(executor):
    if isinstance(executor, HybridExecutor):
        procs = executor.local_processes
        executor.shutdown(wait=False, cancel_futures=True)
        for proc in procs:
            try:
                if proc.is_alive():
                    proc.kill()
            except Exception:
                pass
        return
    procs = list((getattr(executor, "_processes", None) or {}).values())
    executor.shutdown(wait=False, cancel_futures=True)
    for proc in procs:
        try:
            if proc.is_alive():
                proc.kill()
        except Exception:
            pass


def limit_worker_threads():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        logging.warning("threadpoolctl is not installed, worker processes are not limited to one BLAS thread. "
                        "Install it (pip install threadpoolctl) to avoid slow spline fits")
        return
    threadpool_limits(limits=1)

@dataclass(frozen=True)
class SelectionContext:
    selection: Selection
    ref_fd_dict: dict
    t_exp_dict: dict
    n_guess: dict | None
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

    fp_grp = "Fabry-Perot echoes"
    fp_period_est = Quantity(Q_(0.0, "GHz"), read_only=True, significant_figures=4).tag(
        name="Fabry-Perot oscillation period", group=fp_grp, priority=3000, en_save=False)
    fp_count_est = Integer(0, read_only=True).tag(name="Estimated number of echoes", group=fp_grp, priority=3001,
                                                  en_save=False)
    auto_fp_count = Bool(True).tag(name="Use estimated number of echoes", group=fp_grp, priority=3002, en_save=False)
    fp_count = Integer(25, min=-1).tag(name="Assumed number of echoes", group=fp_grp, priority=3003, en_save=False)

    freq = None
    selected_quantity_enum = None
    d_sample = None
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
        self._phi_fits = []

        self._spectral_amp_ax = None
        self._spectral_amp_data_lines = []
        self.spectral_noise_floor_lines = []
        self._plotted_meas_list = []
        self.fp_estimator = None

        self.observe(self.draw_noise_floor, "noise_starts_at")
        self.observe(self.draw_noise_floor, "percent_of_spectral_max")

        self.fd_fig.subplots(2, 1, sharex=True)

    @observe("fit_range")
    def _on_fit_range(self, change):
        self._set_extents(self._span_fit, change["new"])
        self.update_fp_estimate()

    def update_fp_estimate(self):
        if self.fp_estimator is None or not self._plotted_meas_list:
            return
        try:
            t_rt, count = self.fp_estimator(self._plotted_meas_list)
        except Exception:
            logging.warning("Fabry-Perot estimate failed")
            traceback.print_exc()
            return
        self.set_trait("fp_period_est", Q_(1e3 / t_rt, "GHz"))
        self.set_trait("fp_count_est", count)

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

    def _sample_spectrum(self):
        y_db = self.fd_db_data(self._plotted_meas_list)[:, :, 1]
        if self.averaging == Averaging.only_individual:
            return y_db[0]
        return np.mean(y_db, axis=0)

    def draw_noise_floor(self, change=None):
        if not self._spectral_amp_data_lines or not self._plotted_meas_list:
            return None, None, None
        noise_freq = self.noise_starts_at  # also called by the percent_of_spectral_max observer

        if self.spectral_noise_floor_lines:
            for line in self.spectral_noise_floor_lines[::-1]:
                line.remove()
                self.spectral_noise_floor_lines.pop()

        f_full = self.dataset.freq_axis
        y_full = self._sample_spectrum()
        noise_mask = f_full >= noise_freq.magnitude
        if not noise_mask.any():
            logging.warning(f"Noise start frequency {noise_freq} is above the measured frequency range")
            return None, None, None
        noise_floor = np.mean(y_full[noise_mask])

        x, y = self.freq_axis, y_full[self.freq_range_idx]

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
        x, y, floor = self.draw_noise_floor()
        if x is None:
            logging.warning("Plot the spectrum first to set the fit range automatically")
            return

        x_min = self.phi_fit_range[0]
        below_floor = (x > x_min.to("THz").magnitude) & (y < floor)
        if not below_floor.any():
            logging.warning("Spectrum does not drop below the choice floor in the plotted frequency range, "
                            "fit range not changed")
            return

        x_max = Q_(x[np.argmax(below_floor)], "THz")
        self.set_trait("fit_range", [x_min, x_max])


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
        self._plotted_meas_list = list(meas_list)
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

        self._update_phase_fit()

        self.draw_noise_floor()
        if self.auto_adjust_fit_range:
            self.auto_set_fit_range()
        self.update_fp_estimate()

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
        self._phi_fits = []

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
            self._phi_fits.append((slope, offset))
        self._scale_phase_axis()
        ax.legend(loc="upper right")
        ax.figure.canvas.draw_idle()

    def _scale_phase_axis(self):
        ax = self._phi_ax
        x0, x1 = ax.get_xlim()
        values = [0.0]
        for line in self._phi_data_lines:
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            values.extend(y[(x >= x0) & (x <= x1) & np.isfinite(y)])
        for slope, offset in self._phi_fits:
            values.extend(slope * np.array([x0, x1]) + offset)
            values.append(offset)

        lower, upper = np.min(values), np.max(values)
        pad = 0.05 * (upper - lower) if upper > lower else 0.1
        ax.set_ylim(lower - pad, upper + pad)


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
    bound_shrink_factor = Float(0.20, group=t_fit_grp_name,
                                help="Each refinement iteration evaluates the best thickness +- the previous "
                                     "thickness step, with the step multiplied by this factor (0.5: 2 new "
                                     "thicknesses per iteration, 0.2: 8)").tag(name="Thickness step shrink factor",
                                                                                 priority=4)
    optimum_criterion = TEnum(OptimumCriterion, default_value=OptimumCriterion.prominent_min, group=t_fit_grp_name,
                              help="Most prominent local minimum: the deepest notch of log(q) along the thickness "
                                   "axis, ignores the falling q-value trend towards large thicknesses (use a "
                                   "thickness step that resolves the notch, ~10 µm). Global minimum: smallest "
                                   "q-value").tag(name="Optimum selection", priority=5)

    n_guess_grp = "Initial refractive index guess"
    n_guess_method = TEnum(NGuessMethod, default_value=NGuessMethod.tof, group=n_guess_grp,
                           help="Time of flight: estimate from the mean of the thickness axis bounds. "
                                "Model cost scan: for each thickness and shift, scan a constant n with the "
                                "selected transmission model and use the minimum of the phase cost").tag(
        name="Initial guess method", priority=10)
    n_scan_range = ValueRange([1.0, 6.0], group=n_guess_grp).tag(name="Scanned n range", priority=11)
    n_scan_step = Float(0.02, group=n_guess_grp).tag(name="Scan step n", priority=12)
    k_scan_max = Float(0.5, group=n_guess_grp).tag(name="Maximum scanned k", priority=13)

    current_stage = Unicode("", read_only=True).tag(name="Current stage", group=EvalComponentBase.process_status,
                                                    priority=1010, en_save=False)
    running_tasks = Unicode("", read_only=True).tag(name="Running tasks", group=EvalComponentBase.process_status,
                                                    priority=1011, en_save=False)
    current_optimum = Unicode("", read_only=True).tag(name="Current optimum", group=EvalComponentBase.process_status,
                                                      priority=1012, en_save=False)

    preprocessing = Instance(Preprocessing, allow_none=True).tag(priority=1, toggle=ComponentToggle.en_preprocessing)
    layer_stack = Instance(LayerStack, allow_none=True).tag(priority=2, toggle=ComponentToggle.en_multilayer)
    image_eval = Instance(QSpaceImageEval, allow_none=True).tag(priority=3, toggle=ComponentToggle.en_q_space_image)

    def __init__(self, dataset, image_plot=None, **kwargs):
        super().__init__(dataset, **kwargs)

        self.current_process_executor = None

        self.preprocessing = Preprocessing(self.dataset)
        self.layer_stack = LayerStack(object_name="Multi-layer sample")

        self.image_eval = QSpaceImageEval(self, image_plot=image_plot, object_name="Image evaluation")

        self.task_status_carrier = TaskStatusSignal()
        self.task_status_carrier.changed.connect(self.set_trait)

        self._fit_range_link = link((self, "freq_range"), (self.preprocessing, "fit_range"))
        self._phi_fit_range_link = link((self.settings.eval_opt, "phi_fit_range"),
                                        (self.preprocessing, "phi_fit_range"))
        self._fp_links = [link((self.settings.eval_opt, name), (self.preprocessing, name))
                          for name in ("auto_fp_count", "fp_count")]
        self.preprocessing.fp_estimator = lambda meas_list: self.fp_echo_estimate(list(meas_list))

    def fp_echo_estimate(self, meas_list, ref_list=None, t_exp=None):
        if t_exp is None:
            t_exp = np.mean(self.dataset.transmission(meas_list, ref_list)[:, self.freq_range_idx], axis=0)
        d = np.mean(self.d_opt_axis_bounds.magnitude)
        t_rt = fp_round_trip_time(self.freq_axis, t_exp, d, self.known_layer_delay(self.freq_axis))
        count = estimate_fp_echo_count(self.dataset.get_multi_data(meas_list, Domain.Time), t_rt)

        return t_rt, count

    def is_multilayer(self):
        return self.transmission_model == TransmissionModels.tmm_multilayer

    def known_layer_delay(self, freq_axis):
        if not self.is_multilayer():
            return 0.0
        try:
            return self.layer_stack.known_delay(freq_axis)
        except ValueError:
            return 0.0

    @property
    def effective_n_guess_method(self):
        if self.is_multilayer() and self.n_guess_method == NGuessMethod.tof:
            return NGuessMethod.cost_scan
        return self.n_guess_method

    def _set_status(self, name, text):
        self.task_status_carrier.changed.emit(name, text)

    def _task_range_text(self, tasks):
        if not tasks:
            return ""
        d = sorted({float(task[0]) for task in tasks})
        text = f"d = {d[0]:.1f} µm" if len(d) == 1 else f"d = {d[0]:.1f} … {d[-1]:.1f} µm"
        if self.settings.eval_opt.enable_shift_opt:
            shifts = sorted({float(task[1]) for task in tasks})
            text += f", shift {shifts[0]:.1f} fs" if len(shifts) == 1 else f", shift {shifts[0]:.1f} … {shifts[-1]:.1f} fs"
        return text + f" ({len(tasks)} running)"

    def _optimum_text(self, best):
        text = f"{best.d.to('µm').magnitude:.2f} µm"
        if self.settings.eval_opt.enable_shift_opt:
            text += f", shift {best.shift.to('fs').magnitude:.1f} fs"
        return text + f" (q = {Q_(best.optimization_info['q_val']).magnitude:.4g})"

    def reset_state(self):
        self._set_status("running_tasks", "")
        self._set_status("current_optimum", "")

    def ref_fd_dict(self, selection):
        ref_list = list(selection.refs)
        ref_fd = self.dataset.get_multi_data(ref_list, domain=Domain.Frequency)
        return self.format_meas_dict(ref_list, ref_fd)

    def t_exp_dict(self, selection):
        meas_list = list(selection.sams)
        t_exp = self.dataset.transmission(meas_list, list(selection.refs))
        return self.format_meas_dict(meas_list, t_exp)

    def selection_task_count(self, selection):
        # tasks of eval_selection: one thickness / shift scan per evaluated measurement (individual and / or average)
        sam_cnt = len(list(selection.sams))
        if sam_cnt <= 1:
            target_cnt = sam_cnt
        else:
            target_cnt = {Averaging.only_avg: 1, Averaging.only_individual: sam_cnt}.get(self.averaging, sam_cnt + 1)
        d_axis, _ = self.thickness_axis()
        shift_axis = self.shift_axis()
        refine_d_cnt, refine_shift_cnt = self.refine_counts(shift_axis)

        return target_cnt * (len(d_axis) * len(shift_axis)
                             + refine_d_cnt * refine_shift_cnt * (max(1, self.iteration_cnt) - 1))

    def n_guess(self, selection):
        if self.effective_n_guess_method == NGuessMethod.cost_scan:
            return None
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
            state = "Failed"
            try:
                counted = {queue_id: sel for queue_id, sel in queue.pending()}
                self.progress_start(sum(self.selection_task_count(sel) for sel in counted.values()))
                queue_id, next_selection = queue.pop_next()
                while next_selection is not None and not self._cancel_event.is_set():
                    if counted.pop(queue_id, None) is None:  # queued while running
                        self.progress_add_total(self.selection_task_count(next_selection))
                    logging.info(f"Evaluating selection {queue_id}")
                    qs_eval_data = evaluator.eval_selection(next_selection)
                    if not self._cancel_event.is_set():
                        self._result_ready(qs_eval_data)
                    queue_id, next_selection = queue.pop_next()
                state = "Cancelled" if self._cancel_event.is_set() else "Finished"
            except OptimizationCancelled:
                state = "Cancelled"
                logging.info("Optimization cancelled")
            except Exception:
                traceback.print_exc()
            finally:
                self.progress_stop(state)
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

        n_r, k = n.real, n.imag
        a = c_thz / (w * d)

        dk_dn = -a * (n_r - 1) / (n_r * (n_r + 1))
        dn_dd = -(n_r - 1) / d
        dk_dd = -k / d + dk_dn * dn_dd

        delta_n = np.sqrt((a * delta_phi) ** 2 + (dn_dd * delta_d) ** 2)
        delta_k = np.sqrt((dk_dn * a * delta_phi) ** 2 + (a * delta_amp / amp) ** 2 + (dk_dd * delta_d) ** 2)

        opt_res.datasets["alpha"].uncert = Q_(np.abs(4 * np.pi * f_axis * delta_k / (1e-4 * c_thz)), "1/cm")
        opt_res.datasets["n"].uncert = Q_(delta_n + 1j * delta_k, "")

    def q_value_params(self):
        # everything compute_q_value needs from the settings, as a plain dict for the worker processes
        eval_opt = self.settings.eval_opt.q_value
        q_space_range = eval_opt.q_space_range if eval_opt.use_separate_range else self.freq_range
        q_idx = np.asarray(f_axis_idx_map(self.freq_axis, q_space_range))
        q_freq_axis = self.freq_axis[q_idx]
        n_prev, n_next = self.layer_stack.unknown_neighbor_n(q_freq_axis) if self.is_multilayer() else (1, 1)

        return {"q_idx": q_idx,
                "quantity": eval_opt.q_space_quantity.name,
                "fp_time": None if eval_opt.auto_fp_time else eval_opt.fp_spacing.to("ps").magnitude,
                "known_delay": self.known_layer_delay(q_freq_axis),
                "n_prev": n_prev,
                "n_next": n_next,
                "auto_f_max": eval_opt.auto_q_space_f_max,
                "amp_threshold": eval_opt.fp_amp_threshold.to("").magnitude,
                "remove_spikes": eval_opt.remove_q_spikes}

    def calc_q_val(self, res_data: SingleResultData):
        return compute_q_value(res_data, self.q_value_params())

    def base_t_model_kwargs(self):
        t_model_kwargs = self.get_t_model_kwargs()
        if self.is_multilayer():
            try:
                t_model_kwargs["layers"] = self.layer_stack.model_layers()
            except ValueError as e:
                logging.error(f"Multi-layer sample: {e}")
                raise
            self.layer_stack.check_frequency_range(self.freq_axis)
            logging.info(f"Multi-layer sample (propagation order): {self.layer_stack.describe_stack()}")
            if self.n_guess_method == NGuessMethod.tof:
                logging.info("Using the model cost scan as initial guess, the time of flight guess "
                             "doesn't account for the known layers")

        return t_model_kwargs

    def build_context(self, selection) -> SelectionContext:
        t_exp_dict = self.t_exp_dict(selection)
        t_model_kwargs = self.base_t_model_kwargs()
        if self.settings.eval_opt.auto_fp_count:
            t_exp_0 = np.asarray(next(iter(t_exp_dict.values())))[:, 1]
            t_rt, t_model_kwargs["nfp"] = self.fp_echo_estimate(list(selection.sams), list(selection.refs), t_exp_0)
            logging.info(f"Fabry-Perot oscillation period {1e3 / t_rt:.1f} GHz, "
                         f"estimated number of echoes {t_model_kwargs['nfp']}")

        return SelectionContext(
            selection=selection,
            ref_fd_dict=self.ref_fd_dict(selection),
            t_exp_dict=t_exp_dict,
            n_guess=self.n_guess(selection),
            t_model_kwargs=t_model_kwargs,
            smoothing=(self.settings.eval_opt.smoothing_avg_n,
                       self.settings.eval_opt.smoothing_avg_iters),
        )

    def spline_knot_count(self, t_exp):
        eval_opt = self.settings.eval_opt.spline_fit
        knot_cnt = eval_opt.knot_cnt
        if len(self.freq_axis) < 2:
            return knot_cnt

        fp_period = 1 / fp_round_trip_time(self.freq_axis, t_exp, self.d_opt_axis_bounds[1].magnitude,
                                           self.known_layer_delay(self.freq_axis))
        knots_per_period = eval_opt.knots_per_fp_period
        min_knots = int(np.ceil(knots_per_period * np.ptp(self.freq_axis) / fp_period)) + 1

        if eval_opt.auto_knot_cnt:
            knot_cnt = int(np.clip(min_knots, 4, len(self.freq_axis)))
            logging.info(f"Using {knot_cnt} splines ({knots_per_period} per Fabry-Perot period "
                         f"of ~{fp_period:.3f} THz)")
        elif knot_cnt < min_knots:
            knot_spacing = np.ptp(self.freq_axis) / max(knot_cnt - 1, 1)
            logging.warning(f"Spline knot spacing ({knot_spacing:.3f} THz) is too large to resolve the "
                            f"Fabry-Perot oscillations (period ~{fp_period:.3f} THz), q-values will not "
                            f"discriminate the thickness. Use at least {min_knots} splines")

        return knot_cnt

    def common_opt_params(self, t_model_kwargs, t_exp):
        # worker config shared by all targets, t_exp: representative transmission for the spline knot count
        params = {
            "freq_axis": self.freq_axis,
            "transmission_model": self.transmission_model.value,
            "cost_fun": self.selected_cost_fun.value,
            "minimizer_kwargs": self.settings.shgo_options.get_minimizer_kwargs(),
            "shgo_options": self.settings.shgo_options.get_shgo_options(),
            "n": self.settings.shgo_options.n,
            "iters": self.settings.shgo_options.iters,
            "n_guess_method": self.effective_n_guess_method,
            "n_scan_range": tuple(float(v) for v in self.n_scan_range),
            "n_scan_step": self.n_scan_step,
            "k_scan_max": self.k_scan_max,
            **t_model_kwargs,
        }
        if self.optimization_method == OptimizationMethods.Spline:
            params["knot_count"] = self.spline_knot_count(t_exp)
            params["reg_n"] = self.settings.eval_opt.spline_fit.reg_n
            params["reg_k"] = self.settings.eval_opt.spline_fit.reg_k
            params["max_nfev"] = self.settings.eval_opt.spline_fit.max_nfev
            params["tr_solver"] = self.settings.eval_opt.spline_fit.tr_solver.name
            params["per_freq_refine"] = self.settings.eval_opt.spline_fit.per_freq_refine
        params["use_compiled_kernels"] = self.settings.eval_opt.spline_fit.use_compiled_kernels
        if params["use_compiled_kernels"] and not fast_kernels.fast_path_available(params["transmission_model"]):
            reason = "numba is not installed" if not fast_kernels.NUMBA_AVAILABLE else "only for the model tmm_1layer"
            logging.info(f"Compiled kernels not used ({reason})")

        return params

    @property
    def iteration_cnt(self):
        return self.iteration_count if self.iterative_thickness_optimization else 1

    def shift_axis(self):
        if not self.settings.eval_opt.enable_shift_opt:
            return np.array([0.0])
        shift_bnds = self.settings.eval_opt.shift_opt_axis_bounds
        shift_step = self.settings.eval_opt.shift_opt_axis_step.magnitude

        return np.arange(shift_bnds[0].magnitude, shift_bnds[1].magnitude + shift_step, shift_step)

    def thickness_axis(self, d_min=None, d_max=None, d_step=None):
        # defaults: the thickness axis bounds and step
        d_min = self.d_opt_axis_bounds[0].magnitude if d_min is None else d_min
        d_max = self.d_opt_axis_bounds[1].magnitude if d_max is None else d_max
        d_step = self.d_opt_axis_step.magnitude if d_step is None else d_step
        num_steps = int(round((d_max - d_min) / d_step)) + 1
        d_axis = np.linspace(d_min, d_max, max(num_steps, 1))
        if num_steps > 1:
            d_step = d_axis[1] - d_axis[0]

        return d_axis, d_step

    def refine_counts(self, shift_axis):
        shrink_factor = min(max(self.bound_shrink_factor, 1e-3), 1.0)
        refine_d_cnt = 2 * int(round(1 / shrink_factor)) + 1
        refine_shift_cnt = min(3, len(shift_axis))

        return refine_d_cnt, refine_shift_cnt

    @staticmethod
    def _log_dynamic_range_limit(res):
        if "alpha_max" not in res.datasets or "alpha" not in res.datasets:
            return
        freq = res.datasets["alpha"].axes[0].to("THz").magnitude
        f_lim = limit_frequency(freq, res.datasets["alpha"].data.magnitude.real,
                                res.datasets["alpha_max"].data.magnitude)
        if f_lim is not None:
            logging.info(f"The absorption reaches 90 % of the maximum measurable absorption (dynamic range) at "
                         f"{f_lim:.2f} THz, above that it is set by the noise")

    def check_cancel(self):
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise OptimizationCancelled()

    def _get_result(self, future):
        while not future.done():
            self.check_cancel()
            futures_wait([future], timeout=0.2)
        try:
            return future.result()
        except BrokenProcessPool:
            self.check_cancel()
            raise

    @property
    def worker_count(self):
        # local + remote workers of the current process pool
        remote = active_connection()
        if remote is None:
            return self.number_of_workers
        return remote.workers + self.number_of_workers * self.settings.remote.use_local_workers

    def _executor(self):
        remote = active_connection()
        if remote is None:
            return ProcessPoolExecutor(max_workers=self.number_of_workers, initializer=limit_worker_threads)
        use_local = self.settings.remote.use_local_workers and self.number_of_workers > 0
        local = (ProcessPoolExecutor(max_workers=self.number_of_workers, initializer=limit_worker_threads)
                 if use_local else None)
        logging.info(f"Using {remote.workers} remote workers ({remote.info.get('host', remote.host)})"
                     + (f" and {self.number_of_workers} local workers" if use_local else ""))
        return HybridExecutor(remote, local, self.number_of_workers if use_local else 0)

    @contextmanager
    def process_pool(self):
        self.check_cancel()
        with self._executor() as executor:
            self.current_process_executor = executor
            try:
                yield executor
            except BaseException as e:
                kill_process_pool(executor)
                self._set_status("current_stage", "Cancelled" if isinstance(e, OptimizationCancelled) else "Failed")
                raise
            finally:
                self.current_process_executor = None
                self._set_status("running_tasks", "")

    def _new_tasks(self, target: ScanTarget, iter_idx, shift_axis, refine_d_cnt, refine_shift_cnt):
        if iter_idx == 0:
            d_axis, shifts = target.d_axis, shift_axis
        else:
            d0 = target.best.d.magnitude
            d_axis = np.linspace(d0 - target.d_step, d0 + target.d_step, refine_d_cnt)
            d_axis = d_axis[d_axis > 0]
            s0 = target.best.shift.magnitude
            shifts = shift_axis[np.argsort(np.abs(shift_axis - s0))[:refine_shift_cnt]]
            target.d_step = 2 * target.d_step / (refine_d_cnt - 1)

        tasks = []
        for d in d_axis:
            for shift in shifts:
                key = round(float(d), 6), round(float(shift), 6)
                if key not in target.done_keys:
                    target.done_keys.add(key)
                    tasks.append(np.array(key, dtype=float))
        return tasks

    def scan_task_count(self, targets, iteration_count, shift_axis):
        refine_d_cnt, refine_shift_cnt = self.refine_counts(shift_axis)
        coarse = sum(len(target.d_axis) for target in targets) * len(shift_axis)

        return coarse + len(targets) * refine_d_cnt * refine_shift_cnt * (max(1, iteration_count) - 1)

    def run_scan(self, executor, targets: list[ScanTarget], iteration_count, shift_axis, label="",
                 kept_datasets=None, on_iteration=None, show_stage=True):
        # scans all targets together (one task per target, thickness and shift), each iteration refines around
        # the current optimum of each target. kept_datasets: names of the result datasets to keep (default all).
        # Advances the overall progress by scan_task_count, the caller includes it in the progress total.
        # show_stage=False keeps the current stage text, for short follow-up scans that would only make it flicker
        refine_d_cnt, refine_shift_cnt = self.refine_counts(shift_axis)
        refine_task_cnt = len(targets) * refine_d_cnt * refine_shift_cnt
        iteration_count = max(1, iteration_count)
        q_params = self.q_value_params()

        for iter_idx in range(iteration_count):
            self.check_cancel()
            stage = "coarse scan" if iter_idx == 0 else f"refinement {iter_idx} / {iteration_count - 1}"
            if show_stage:
                self._set_status("current_stage", f"{label}, {stage}")

            tasks = [(target, task) for target in targets
                     for task in self._new_tasks(target, iter_idx, shift_axis, refine_d_cnt, refine_shift_cnt)]
            if iter_idx == 0:
                total_tasks = self.scan_task_count(targets, iteration_count, shift_axis)
                logging.info(f"Processing up to {total_tasks} task{'s' * (total_tasks > 1)}")
            else:
                logging.info(f"Starting thickness refinement iteration {iter_idx} / {iteration_count - 1}")
                # already scanned thicknesses aren't repeated
                self.progress_advance(refine_task_cnt - len(tasks))

            # the q-value is computed in the workers and unneeded datasets are dropped there, which keeps the main
            # process (unpickling, bookkeeping) from becoming the bottleneck
            futures = [executor.submit(partial(optimize_and_rate, config_dict=target.opt_config,
                                               opt_func=self.optimization_method.value, q_params=q_params,
                                               kept_datasets=kept_datasets),
                                       *task) for target, task in tasks]

            def update_running(start_idx):
                # submitted in order, so the running tasks follow the last finished one
                running = []
                for (_, task), fut in zip(tasks[start_idx:], futures[start_idx:]):
                    if len(running) == self.worker_count:
                        break
                    if not fut.done():
                        running.append(task)
                self._set_status("running_tasks", self._task_range_text(running))

            update_running(0)
            last_report = time.monotonic()
            for task_idx, ((target, _), future) in enumerate(zip(tasks, futures)):
                res: SingleResultData = self._get_result(future)
                self.progress_advance()
                update_running(task_idx + 1)

                info_str = f"Finished optimizing thickness {np.round(res.d, 3)} "
                if self.settings.eval_opt.enable_shift_opt:
                    info_str += f"with a shift of {res.shift}"
                logging.debug(info_str)

                is_last = task_idx == len(futures) - 1
                if is_last or time.monotonic() - last_report > 5:
                    last_report = time.monotonic()
                    logging.info(f"Finished {task_idx + 1} / {len(futures)} tasks (thickness {np.round(res.d, 3)})")

                if self.is_multilayer():
                    res.optimization_info["layer_stack"] = self.layer_stack.describe_stack()
                target.results.append(res)

            for target in targets:
                if target.results:
                    target.best = select_optimum(target.results, self.optimum_criterion)
            if on_iteration is not None:
                on_iteration(iter_idx, iteration_count)

    def eval_selection(self, selection) -> EvalResultData:
        ctx = self.build_context(selection)
        sel_meas_list = list(selection.sams)

        t_exp_0 = np.asarray(next(iter(ctx.t_exp_dict.values())))[:, 1]
        common_opt_params = self.common_opt_params(ctx.t_model_kwargs, t_exp_0)
        d_axis0, d_step0 = self.thickness_axis()
        shift_axis = self.shift_axis()

        eval_result_data = EvalResultData()
        eval_result_data.result_type = "Transmission fit"
        eval_result_data.dataset_path = self.dataset.data_path
        eval_result_data.measurement_names = [meas.filepath.name for meas in sel_meas_list]
        eval_result_data.reference_names = [meas.filepath.name for meas in selection.refs]
        eval_result_data.model_name = self.transmission_model.name
        eval_result_data.measurement_quantity = "Transmission"

        with self.process_pool() as executor:
            for meas_idx, meas in enumerate(ctx.t_exp_dict):
                self.check_cancel()
                if meas != "Average":
                    logging.info(f"Evaluating measurement {meas.filepath.name}")
                self.reset_state()
                meas_label = (f"Average of {len(sel_meas_list)} measurements" if meas == "Average"
                              else meas.filepath.name)

                target = ScanTarget(key=meas, d_axis=d_axis0, d_step=d_step0,
                                    opt_config={**common_opt_params,
                                                "n_guess": ctx.n_guess[meas] if ctx.n_guess else None,
                                                "t_exp": ctx.t_exp_dict[meas]})

                def on_iteration(iter_idx, iteration_count, target_=target):
                    self._set_status("current_optimum", self._optimum_text(target_.best))
                    if iter_idx < iteration_count - 1:
                        logging.info(f"Current optimum {np.round(target_.best.d, 3)}")

                self.run_scan(executor, [target], self.iteration_cnt, shift_axis, label=meas_label,
                              on_iteration=on_iteration)

                opt_results = sorted(target.results, key=lambda res_: (res_.d.magnitude, res_.shift.magnitude))
                ready_results = self.prepare_results(meas, opt_results, ctx)
                meas_results = {f"result_{meas_idx}_{res_idx:03d}": res
                                for res_idx, res in enumerate(ready_results)}
                eval_result_data.results.update(meas_results)

                best = select_optimum(ready_results, self.optimum_criterion)
                best_key = next(k for k, res in meas_results.items() if res is best)
                logging.info(f"Optimum thickness {np.round(best.d, 3)}"
                             + f" (shift {best.shift})" * self.settings.eval_opt.enable_shift_opt)
                self._log_dynamic_range_limit(best)
                if self.settings.eval_opt.uncertainty.qspace_enabled:
                    self._selection_thickness_uncertainty(executor, ctx, meas, target, best, meas_label)
                eval_result_data.optimum_results[ready_results[0].measurement] = best_key
                self._set_status("current_stage", f"{meas_label}, finished")
                self._set_status("current_optimum", self._optimum_text(best))

        return eval_result_data

    mc_window_moves = 4

    def mc_thickness_uncertainty(self, executor, specs, label, add_progress=True):
        # specs: (key, opt_config, t (F), TransmissionUncertainty, d_center (µm), shift (fs)) per target. Each perturbed
        # transmission gets a scan of the thickness window around d_center at the fixed shift (moved up to
        # mc_window_moves times if its minimum is at the end), returns {key: ThicknessUncertainty} from the spread of
        # their q-value minima
        opt = self.settings.eval_opt.uncertainty
        rng = random_generator(opt)
        freq = self.freq_axis
        half = opt.qspace_window.to("µm").magnitude
        step = 2 * half / (opt.qspace_points - 1)
        by_shift = {}
        for key, opt_config, t, unc, d_center, shift in specs:
            d_axis, d_step = self.thickness_axis(max(d_center - half, step), d_center + half, step)
            for rep_idx, t_rep in enumerate(perturbed_transmissions(freq, t, unc, opt.qspace_samples, rng)):
                config = {**opt_config, "t_exp": np.column_stack((freq, t_rep))}
                by_shift.setdefault(float(shift), []).append(
                    ScanTarget(key=(key, rep_idx), opt_config=config, d_axis=d_axis, d_step=d_step))
        targets = [target for group in by_shift.values() for target in group]
        if add_progress:
            self.progress_add_total(sum(len(target.d_axis) for target in targets))
        for shift, group in by_shift.items():
            self.run_scan(executor, group, 1, np.array([shift]), label=f"{label}, uncertainty",
                          kept_datasets=("n",))

        # a minimum at the end of the window: the window continues on that side, the same number of thicknesses
        for _ in range(self.mc_window_moves):
            moved = {}
            for shift, group in by_shift.items():
                for target in group:
                    d_done = np.array(sorted(res.d.to("µm").magnitude for res in target.results))
                    d_opt, _, at_edge = replicate_optimum(target.results)
                    if not at_edge:
                        continue
                    offsets = step * np.arange(1, opt.qspace_points)
                    new_d = d_done[0] - offsets if d_opt <= d_done[0] + 1e-9 else d_done[-1] + offsets
                    target.d_axis = new_d[new_d > 0]
                    if len(target.d_axis):
                        moved.setdefault(shift, []).append(target)
            if not moved:
                break
            self.progress_add_total(sum(len(target.d_axis) for group in moved.values() for target in group))
            for shift, group in moved.items():
                self.run_scan(executor, group, 1, np.array([shift]), label=f"{label}, uncertainty (moved windows)",
                              kept_datasets=("n",), show_stage=False)

        optima = {}
        for target in targets:
            optima.setdefault(target.key[0], []).append(replicate_optimum(target.results))

        return {key: ThicknessUncertainty.from_replicates(key_optima) for key, key_optima in optima.items()}

    def log_thickness_uncertainty(self, label, d, res: ThicknessUncertainty, unc: TransmissionUncertainty):
        logging.info(f"{label}: thickness {d:.2f} ± {res.d_std:.2f} µm (Monte Carlo, {res.samples} samples, their "
                     f"mean {res.d_mean:.2f} µm; {unc.describe()})")
        sem = res.d_std / np.sqrt(res.samples)
        if abs(res.d_mean - d) > 3 * sem + 0.05 * res.d_std + 1e-3:
            logging.warning(f"{label}: the mean of the Monte Carlo samples differs from the optimum by "
                            f"{res.d_mean - d:+.2f} µm, more than the noise explains. The thickness step (refinement "
                            f"iterations) limits the optimum, the samples interpolate between the scanned thicknesses")
        if res.at_edge:
            logging.warning(f"{label}: {res.at_edge} of {res.samples} Monte Carlo samples have their minimum at the "
                            f"end of the thickness window after moving it {self.mc_window_moves} times, the uncertainty is "
                            f"underestimated. Increase the window (Settings -> Evaluation -> Uncertainty)")

    def _selection_thickness_uncertainty(self, executor, ctx: SelectionContext, meas, target, best, label):
        selection = ctx.selection
        meas_list = list(selection.sams) if not isinstance(meas, Measurement) else [meas]
        ref_list = [selection.ref_map(m) for m in meas_list]
        unc = estimator_for(self.dataset).transmission_uncertainty(meas_list, ref_list, self.freq_range_idx,
                                                                   self.settings.eval_opt.uncertainty)
        t = np.asarray(ctx.t_exp_dict[meas])[:, 1]
        d, shift = best.d.to("µm").magnitude, best.shift.to("fs").magnitude
        res = self.mc_thickness_uncertainty(executor, [(meas, target.opt_config, t, unc, d, shift)], label)[meas]
        self.log_thickness_uncertainty(label, d, res, unc)

        best.optimization_info["d_uncertainty"] = Q_(res.d_std, "µm")
        best.optimization_info["d_mc_mean"] = Q_(res.d_mean, "µm")
        best.optimization_info["uncertainty_samples"] = res.samples
        freq = best.freq_axis.to("THz").magnitude
        datasets = best.datasets
        if "n_real" in datasets and "n_imag" in datasets:
            datasets["n_real"].uncert = Q_(res.n_std.real, "")
            datasets["n_imag"].uncert = Q_(res.n_std.imag, "")
        if "alpha" in datasets:
            datasets["alpha"].uncert = Q_(alpha_of(freq, 1j * res.n_std.imag), "1/cm")

    def calc_sim(self, model_kwargs, ref_fd):
        model = self.transmission_model.value

        freq_axis = self.freq_axis
        one = np.ones_like(freq_axis, dtype=float)

        if self.settings.eval_opt.transmission_sim.sim_n_selection == SimRISelection.const:
            sim_n_sub = self.settings.eval_opt.transmission_sim.sim_n_sub
            n_sub = (sim_n_sub[0] + 1j * sim_n_sub[1]) * one

            sim_n_film = self.settings.eval_opt.transmission_sim.sim_n_film
            n_film = (sim_n_film[0] + 1j * sim_n_film[1]) * one
        else:
            n_sub = 1 * one
            n_film = 1 * one

        model_kwargs = {k: v for k, v in model_kwargs.items()}
        model_kwargs["d"] = self.settings.eval_opt.transmission_sim.sim_d.magnitude
        model_kwargs["nfp"] = self.settings.eval_opt.transmission_sim.sim_nfp
        model_kwargs["shift"] = self.settings.eval_opt.transmission_sim.sim_shift.magnitude

        if self.is_two_layer_t_model():
            model_kwargs["n_sub"] = n_sub
            model_kwargs["h"] = self.settings.eval_opt.transmission_sim.sim_h.magnitude
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

        norm_q_vals = self.settings.eval_opt.q_value.normalize_q_vals
        q_vals = [opt_res.optimization_info["q_val"] for opt_res in opt_results]
        alpha_noise = None
        if self.settings.eval_opt.dynamic_range.enabled:
            try:
                alpha_noise = noise_floor(self.dataset, len(ctx.selection.sams) if is_avg_meas else 1)
            except Exception as e:
                logging.warning(f"No noise floor for the maximum measurable absorption: {e}")
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
            if alpha_noise is not None:
                opt_res.datasets["alpha_max"] = alpha_max_dataset(
                    opt_res.freq_axis, ref_fd[:, 1], alpha_noise, opt_res.datasets["n"].data.magnitude.real,
                    opt_res.d.to("µm").magnitude)

            if self.settings.eval_opt.transmission_sim.add_sim_to_res:
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
