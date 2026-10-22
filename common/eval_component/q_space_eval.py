import logging
import time
import traceback
import numpy as np
import matplotlib as mpl
from matplotlib.widgets import SpanSelector
from common.eval_component.transfer_functions import TransmissionModels
from common.eval_component.eval_component_base import EvalComponentBase
from common.measurement_selection import Selection, SelectionQueue
from common.default_appsettings import SimRISelection, Domain, QSpaceQuantity, ComponentToggle, AppSettings
from common.functions import f_axis_idx_map, moving_average, do_ifft, to_db, avg_data_array
from common.eval_component.quantity_set import QuantityDataSet
from common.eval_component.eval_result import EvalResultData, SingleResultData
from enum import member, Enum
from common.measurements import Measurement
from common.dataset import Averaging
from common.consts import c_thz
from scipy.signal import detrend, find_peaks
from concurrent.futures import ProcessPoolExecutor, wait as futures_wait
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from common.eval_component.single_opt import shgo_transmission_optimization
from common.eval_component.t_fit_global import spline_transmission_optimization
from common.eval_component.n_guess import NGuessMethod
from common.eval_component.layer_stack import LayerStack
from dataclasses import dataclass
from mpl_settings import mpl_style_params
from traitlets import Enum as TEnum, observe, Integer, Float, Bool, Instance, Unicode, link
from common.traits import Quantity, Q_, ValueRange, PlotFigure
from common.components import action
from PySide6.QtCore import QObject, Signal
import threading
from concurrent.futures import ThreadPoolExecutor
from common.dataset_plotter import DataSetPlotter

action = partial(action, check_init=True, rc_params=mpl_style_params)


def fp_round_trip_time(freq_axis, t_exp, d, known_delay=0.0):
    # known_delay: part of the measured group delay caused by other (known) layers of the sample
    phi = np.unwrap(np.angle(t_exp))
    group_delay = np.abs(np.polyfit(2 * np.pi * freq_axis, phi, 1)[0])

    return 2 * (max(group_delay - known_delay, 0.0) + d / c_thz)


def estimate_fp_echo_count(time_traces, t_rt):
    counts = []
    for trace in time_traces:
        t, y_abs = trace[:, 0].real, np.abs(trace[:, 1].real)
        recorded = np.flatnonzero(y_abs > 1e-6 * y_abs.max())
        counts.append((t[recorded[-1]] - t[np.argmax(y_abs)]) / t_rt)

    return max(int(np.floor(min(counts))), 0)


def fp_echo_amplitude(freq_axis, n, d, n_prev=1, n_next=1):
    r_prev = (n - n_prev) / (n + n_prev)
    r_next = (n - n_next) / (n + n_next)
    return np.abs(r_prev * r_next) * np.exp(-4 * np.pi * freq_axis * np.abs(n.imag) * 2 * d / c_thz)


def q_space_spectrum(freq_axis, y, pad_factor=16):
    y = detrend(y, type="linear")
    n_fft = pad_factor * len(y)
    q_spec = np.abs(np.fft.rfft(y, n=n_fft))
    q_axis = np.fft.rfftfreq(n_fft, d=np.mean(np.diff(freq_axis)))

    return q_axis, q_spec


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


class OptimizationCancelled(Exception):
    """Raised inside the evaluation thread when the optimization is canceled."""


class TaskStatusSignal(QObject):
    # (trait name, text), emitted from the evaluation thread and applied in the GUI thread
    changed = Signal(str, str)


def kill_process_pool(executor):
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

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

        self.current_process_executor = None

        self.opt_state = {}

        self.preprocessing = Preprocessing(self.dataset)
        self.layer_stack = LayerStack(object_name="Multi-layer sample")

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
        self.task_cnt_carrier.cnt_changed.emit(0)
        self._set_status("running_tasks", "")
        self._set_status("current_optimum", "")
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

    def calc_q_val(self, res_data: SingleResultData):
        eval_opt = self.settings.eval_opt
        q_space_range = eval_opt.q_space_range if eval_opt.use_separate_range else self.freq_range

        freq_axis = res_data.freq_axis.magnitude
        freq_range_idx = f_axis_idx_map(freq_axis, q_space_range)
        freq_axis = freq_axis[freq_range_idx]

        match eval_opt.q_space_quantity:
            case QSpaceQuantity.n:
                y = res_data.datasets["n"].data.magnitude.real
            case QSpaceQuantity.k:
                y = res_data.datasets["n"].data.magnitude.imag
            case QSpaceQuantity.alpha:
                y = res_data.datasets[QSpaceQuantity.alpha.name].data.magnitude
        y = y[freq_range_idx]

        d = res_data.d.magnitude
        if eval_opt.auto_fp_time:
            t_exp = res_data.datasets["t_exp"].data.magnitude[freq_range_idx]
            t_fp = fp_round_trip_time(freq_axis, t_exp, d, self.known_layer_delay(freq_axis))
        else:
            t_fp = eval_opt.fp_spacing.to("ps").magnitude

        n = res_data.datasets["n"].data.magnitude[freq_range_idx]
        n_prev, n_next = self.layer_stack.unknown_neighbor_n(freq_axis) if self.is_multilayer() else (1, 1)
        echo_amp = fp_echo_amplitude(freq_axis, n, d, n_prev, n_next)
        if eval_opt.auto_q_space_f_max:
            below = np.flatnonzero(echo_amp < eval_opt.fp_amp_threshold.to("").magnitude)
            f_max = freq_axis[below[0]] if len(below) else freq_axis[-1]
            # keep at least two FP periods
            f_max = min(max(f_max, freq_axis[0] + 2 / t_fp), freq_axis[-1])
            band = freq_axis <= f_max
            freq_axis, y = freq_axis[band], y[band]
        res_data.optimization_info["q_space_f_max"] = Q_(freq_axis[-1], "THz")

        q_axis, q_spec = q_space_spectrum(freq_axis, y)
        # half the main lobe width of the unwindowed transform, the peak position is only known to ~1 / bandwidth
        half_width = 0.5 / np.ptp(freq_axis)
        window = np.abs(q_axis - t_fp) <= half_width
        if not window.any():
            window[np.argmin(np.abs(q_axis - t_fp))] = True

        res_data.optimization_info["q_val"] = Q_(np.max(q_spec[window]), "")
        res_data.optimization_info["q_sum"] = Q_(np.sum(q_spec[window]), "")
        res_data.optimization_info["fp_time"] = Q_(t_fp, "ps")
        res_data.optimization_info["fp_period"] = Q_(1 / t_fp, "THz")

        q_name = eval_opt.q_space_quantity.name
        res_data.datasets["q_space"] = QuantityDataSet(axes=[Q_(q_axis, "ps")],
                                                       data=Q_(q_spec, ""),
                                                       axes_labels=["Round trip time"],
                                                       data_label=f"Q-space |FFT({q_name})|, "
                                                                  f"FP time {t_fp:.3f} ps, "
                                                                  f"{freq_axis[0]:.2f}-{freq_axis[-1]:.2f} THz")
        res_data.datasets["fp_echo_amp"] = QuantityDataSet(axes=[res_data.freq_axis[freq_range_idx]],
                                                           data=Q_(100 * echo_amp, "%"),
                                                           axes_labels=["Frequency"],
                                                           data_label="Estimated Fabry-Perot echo amplitude")

        return res_data.optimization_info["q_val"]

    def build_context(self, selection) -> SelectionContext:
        t_exp_dict = self.t_exp_dict(selection)
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

    def spline_knot_count(self, ctx: SelectionContext):
        eval_opt = self.settings.eval_opt
        knot_cnt = eval_opt.knot_cnt
        if len(self.freq_axis) < 2:
            return knot_cnt

        t_exp_0 = np.asarray(next(iter(ctx.t_exp_dict.values())))[:, 1]
        fp_period = 1 / fp_round_trip_time(self.freq_axis, t_exp_0, self.d_opt_axis_bounds[1].magnitude,
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
            "n_guess_method": self.effective_n_guess_method,
            "n_scan_range": tuple(float(v) for v in self.n_scan_range),
            "n_scan_step": self.n_scan_step,
            "k_scan_max": self.k_scan_max,
            **ctx.t_model_kwargs,
        }
        if self.optimization_method == OptimizationMethods.Spline:
            common_opt_params["knot_count"] = self.spline_knot_count(ctx)
            common_opt_params["reg_n"] = self.settings.eval_opt.reg_n
            common_opt_params["reg_k"] = self.settings.eval_opt.reg_k
            common_opt_params["max_nfev"] = self.settings.eval_opt.max_nfev
            common_opt_params["per_freq_refine"] = self.settings.eval_opt.per_freq_refine
        opt_configs = {meas: {**common_opt_params, "n_guess": ctx.n_guess[meas] if ctx.n_guess else None,
                              "t_exp": ctx.t_exp_dict[meas]} for meas in ctx.t_exp_dict}


        shift_bnds = self.settings.eval_opt.shift_opt_axis_bounds
        shift_step = self.settings.eval_opt.shift_opt_axis_step.magnitude
        if self.settings.eval_opt.enable_shift_opt:
            shift_axis = np.arange(shift_bnds[0].magnitude, shift_bnds[1].magnitude + shift_step, shift_step)
        else:
            shift_axis = np.array([0.0])

        d_bnds = self.d_opt_axis_bounds
        d_step0 = self.d_opt_axis_step.magnitude
        num_steps = int(round((d_bnds[1].magnitude - d_bnds[0].magnitude) / d_step0)) + 1
        d_axis0 = np.linspace(d_bnds[0].magnitude, d_bnds[1].magnitude, num_steps)
        if num_steps > 1:
            d_step0 = d_axis0[1] - d_axis0[0]

        shrink_factor = min(max(self.bound_shrink_factor, 1e-3), 1.0)
        refine_d_cnt = 2 * int(round(1 / shrink_factor)) + 1
        refine_shift_cnt = min(3, len(shift_axis))
        refine_task_cnt = refine_d_cnt * refine_shift_cnt

        def task_key(d, shift):
            return round(float(d), 6), round(float(shift), 6)

        def get_new_tasks(iter_idx, d_step, done_keys):
            if iter_idx == 0:
                d_axis, shifts = d_axis0, shift_axis
            else:
                d0 = self.opt_state["d"].magnitude
                d_axis = np.linspace(d0 - d_step, d0 + d_step, refine_d_cnt)
                d_axis = d_axis[d_axis > 0]
                s0 = self.opt_state["shift"].magnitude
                shifts = shift_axis[np.argsort(np.abs(shift_axis - s0))[:refine_shift_cnt]]

            tasks = []
            for d in d_axis:
                for shift in shifts:
                    key = task_key(d, shift)
                    if key not in done_keys:
                        done_keys.add(key)
                        tasks.append(np.array(key, dtype=float))
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

        def process_tasks(executor, tasks, opt_config, iteration_progress):
            it_idx, tot_it = iteration_progress
            if it_idx == 0:
                total_tasks = len(tasks) + refine_task_cnt * (tot_it - 1)
                self.task_cnt_carrier.cnt_changed.emit(total_tasks)
                logging.info(f"Processing up to {total_tasks} task{'s' * (total_tasks > 1)}")
            else:
                logging.info(f"Starting thickness refinement iteration {it_idx} / {tot_it - 1}")
                for _ in range(refine_task_cnt - len(tasks)):
                    self.task_cnt_carrier.cnt_changed.emit(1)

            check_cancel()
            task_results = []
            worker_func = partial(self.optimization_method.value, config_dict=opt_config)
            futures = [executor.submit(worker_func, d, shift) for d, shift in tasks]

            def update_running(start_idx):
                # submitted in order, so the running tasks follow the last finished one
                pending = [task for task, fut in zip(tasks[start_idx:], futures[start_idx:]) if not fut.done()]
                self._set_status("running_tasks", self._task_range_text(pending[:self.number_of_workers]))

            update_running(0)
            last_report = time.monotonic()
            for task_idx, future in enumerate(futures):
                res: SingleResultData = get_result(future)
                self.task_cnt_carrier.cnt_changed.emit(1)
                update_running(task_idx + 1)

                info_str = f"Finished optimizing thickness {np.round(res.d, 3)} "
                if self.settings.eval_opt.enable_shift_opt:
                    info_str += f"with a shift of {res.shift}"
                logging.debug(info_str)

                is_last = task_idx == len(futures) - 1
                if is_last or time.monotonic() - last_report > 5:
                    last_report = time.monotonic()
                    logging.info(f"Finished {task_idx + 1} / {len(futures)} tasks (thickness {np.round(res.d, 3)})")

                self.calc_q_val(res)
                if self.is_multilayer():
                    res.optimization_info["layer_stack"] = self.layer_stack.describe_stack()
                task_results.append(res)

            return task_results

        eval_result_data = EvalResultData()
        eval_result_data.result_type = "Transmission fit"
        eval_result_data.dataset_path = self.dataset.data_path
        eval_result_data.measurement_names = [meas.filepath.name for meas in sel_meas_list]
        eval_result_data.reference_names = [meas.filepath.name for meas in selection.refs]
        eval_result_data.model_name = self.transmission_model.name
        eval_result_data.measurement_quantity = "Transmission"

        check_cancel()
        with ProcessPoolExecutor(max_workers=self.number_of_workers,
                                 initializer=limit_worker_threads) as executor:
            self.current_process_executor = executor
            try:
                for meas_idx, meas in enumerate(ctx.t_exp_dict):
                    check_cancel()
                    if meas != "Average":
                        logging.info(f"Evaluating measurement {meas.filepath.name}")
                    self.reset_state()
                    meas_label = (f"Average of {len(sel_meas_list)} measurements" if meas == "Average"
                                  else meas.filepath.name)

                    opt_results: list[SingleResultData] = []
                    done_keys = set()
                    d_step = d_step0
                    for iter_idx in range(max(1, iteration_count)):
                        check_cancel()
                        stage = "coarse scan" if iter_idx == 0 else f"refinement {iter_idx} / {iteration_count - 1}"
                        self._set_status("current_stage", f"{meas_label}, {stage}")
                        new_tasks = get_new_tasks(iter_idx, d_step, done_keys)
                        if iter_idx > 0:
                            d_step = 2 * d_step / (refine_d_cnt - 1)
                        opt_results.extend(process_tasks(executor, new_tasks, opt_configs[meas],
                                                         iteration_progress=(iter_idx, iteration_count)))
                        if opt_results:
                            best = select_optimum(opt_results, self.optimum_criterion)
                            self.opt_state.update(d=best.d, shift=best.shift,
                                                  q_min=best.optimization_info["q_val"])
                            self._set_status("current_optimum", self._optimum_text(best))
                            if iter_idx < iteration_count - 1:
                                logging.info(f"Current optimum {np.round(best.d, 3)}")

                    opt_results.sort(key=lambda res_: (res_.d.magnitude, res_.shift.magnitude))
                    ready_results = self.prepare_results(meas, opt_results, ctx)
                    meas_results = {f"result_{meas_idx}_{res_idx:03d}": res
                                    for res_idx, res in enumerate(ready_results)}
                    eval_result_data.results.update(meas_results)

                    best = select_optimum(ready_results, self.optimum_criterion)
                    best_key = next(k for k, res in meas_results.items() if res is best)
                    logging.info(f"Optimum thickness {np.round(best.d, 3)}"
                                 + f" (shift {best.shift})" * self.settings.eval_opt.enable_shift_opt)
                    eval_result_data.optimum_results[ready_results[0].measurement] = best_key
                    self._set_status("current_stage", f"{meas_label}, finished")
                    self._set_status("current_optimum", self._optimum_text(best))
            except BaseException as e:
                kill_process_pool(executor)
                self._set_status("current_stage", "Cancelled" if isinstance(e, OptimizationCancelled) else "Failed")
                raise
            finally:
                self.current_process_executor = None
                self._set_status("running_tasks", "")

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
