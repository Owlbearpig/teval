import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
import numpy as np
from scipy.signal import find_peaks
from scipy.signal.windows import tukey
from traitlets import Bool, Float, Int, Enum as TEnum
from common.components import action
from common.consts import c_thz
from common.dataset import Averaging
from common.measurements import Measurement
from common.traits import Q_, Quantity
from common.eval_component.eval_component_base import EvalComponentBase
from common.eval_component.eval_result import EvalResultData, SingleResultData
from common.eval_component.quantity_set import QuantityDataSet

"""
Sparse deconvolution: the sample trace is modelled as the reference pulse convolved with a sparse impulse response,
y = h * x + noise, x = argmin 0.5 ||y - h * x||^2 + lambda ||x||_1 (FISTA). The peaks of x are the transmitted pulse /
echoes (transmission) or the interface reflections (reflection), their delays give optical thicknesses.
Units: [t] = ps, [d] = µm
"""


class Geometry(Enum):
    transmission = "Transmission"
    reflection = "Reflection"


def group_soft_threshold(x, threshold):
    # x: (..., K, T), shrinks the norm of the K coefficients at each delay (K = 1: ordinary soft threshold)
    norm = np.linalg.norm(x, axis=-2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(norm > 0, np.maximum(1 - threshold / norm, 0), 0)
    return x * scale


def fista_deconvolution(y, h, lambda_rel=0.01, max_iter=2000, tol=1e-6, hilbert=False, shapes=None, masks=None):
    # y, h: (N, T) sample and reference traces on the same uniform time axis. The convolution is done with zero
    # padding to 2T (no wrap around of delays within the trace). Index k of the result is the delay k dt (k < T) or
    # (k - 2T) dt (negative delays).
    # hilbert: each delay gets the reference and its Hilbert transform (group sparsity), a pulse with phase phi is
    # cos(phi) h + sin(phi) H{h}. Returns the complex amplitudes c (N, 2T), spectrum of each peak: c * H(f), i.e.
    # angle(c) is in the convention of the measured phase (angle(sample / reference)). Without hilbert c is real.
    # shapes, masks: delay dependent atoms, the reference is filtered with shapes[:, j] (N, J, T + 1) for the delays
    # in masks[:, j] (N, J, 2T, bool, disjoint), e.g. a pulse shape per echo order
    n_t = y.shape[-1]
    n_pad = 2 * n_t
    h_f = np.fft.rfft(h, n=n_pad, axis=-1)[..., None, :]
    if shapes is None:
        shapes = np.ones(h_f.shape)
        masks = np.ones(y.shape[:-1] + (1, n_pad), dtype=bool)
    atoms_f = h_f * shapes
    if hilbert:  # channels: (shape 0, hilbert 0, shape 1, hilbert 1, ...)
        hilbert_f = -1j * atoms_f
        hilbert_f[..., 0] = 0
        hilbert_f[..., -1] = 0
        atoms_f = np.stack([atoms_f, hilbert_f], axis=-2).reshape(atoms_f.shape[:-2] + (-1, atoms_f.shape[-1]))
        masks = np.repeat(masks, 2, axis=-2)
    masks = masks.astype(float)
    y_pad = np.zeros(y.shape[:-1] + (n_pad,))
    y_pad[..., :n_t] = y

    def forward(x, idx=slice(None)):  # (N, C, n_pad) -> (N, n_pad), x is zero outside of its masks
        return np.fft.irfft(np.sum(atoms_f[idx] * np.fft.rfft(x, axis=-1), axis=-2), n=n_pad, axis=-1)

    def adjoint(r, idx=slice(None)):  # (N, n_pad) -> (N, C, n_pad)
        return masks[idx] * np.fft.irfft(np.conj(atoms_f[idx]) * np.fft.rfft(r, axis=-1)[..., None, :], n=n_pad,
                                         axis=-1)

    lipschitz = np.max(np.sum(np.abs(atoms_f) ** 2, axis=-2), axis=-1)[..., None, None]
    # lambda >= max |A^T y| gives x = 0, the regularization is set relative to it
    lam = lambda_rel * np.max(np.linalg.norm(adjoint(y_pad), axis=-2), axis=-1)[..., None, None]

    x = np.zeros(y.shape[:-1] + (atoms_f.shape[-2], n_pad))
    z = x.copy()
    t_k = np.ones(y.shape[:-1] + (1, 1))
    converged = np.zeros(y.shape[:-1], dtype=bool)
    n_iter = np.full(y.shape[:-1], max_iter)
    for it in range(max_iter):
        # only the traces which did not converge yet are updated
        act = np.flatnonzero(~converged)
        z_a, x_a = z[act], x[act]
        step = lipschitz[act]
        x_new = group_soft_threshold(z_a - adjoint(forward(z_a, act) - y_pad[act], act) / step, lam[act] / step)

        # adaptive restart (O'Donoghue, Candes): reset the momentum when it points uphill, avoids the oscillations
        # of FISTA (strongly correlated atoms with the hilbert option)
        restart = np.sum((z_a - x_new) * (x_new - x_a), axis=(-2, -1), keepdims=True) > 0
        t_a = np.where(restart, 1.0, t_k[act])
        t_new = (1 + np.sqrt(1 + 4 * t_a ** 2)) / 2

        change = np.linalg.norm(x_new - x_a, axis=(-2, -1)) / np.maximum(np.linalg.norm(x_new, axis=(-2, -1)), 1e-30)
        done = change < tol
        z[act] = np.where(done[:, None, None], x_new, x_new + ((t_a - 1) / t_new) * (x_new - x_a))
        x[act], t_k[act] = x_new, t_new
        n_iter[act[done]] = it + 1
        converged[act[done]] = True
        if converged.all():
            break

    y_model = forward(x)[..., :n_t]
    lam = lam[..., 0, 0]
    cost = 0.5 * np.sum((y - y_model) ** 2, axis=-1) + lam * np.sum(np.linalg.norm(x, axis=-2), axis=-1)
    # the masks are disjoint: sum over the shape channels
    c = np.sum(x[..., 0::2, :] - 1j * x[..., 1::2, :], axis=-2) if hilbert else np.sum(x, axis=-2)

    return c, y_model, {"n_iter": n_iter, "converged": converged, "cost": cost, "lambda": lam}


def calibrate_pulse_shape(y, h, dt, shift, before=3.0, after=8.0, n_pad=None):
    # transfer function of the pulse shape between the reference and the sample pulse delayed by shift (ps):
    # both are windowed identically (tukey, before / after the pulse in ps), the delay is removed (only the shape
    # remains: phase offset, dispersion, frequency dependent amplitude) and it is normalized to an amplitude of 1.
    # Outside of the reference bandwidth (< -40 dB) it fades to 1. Returns the shape on the rfft grid of n_pad and
    # the (frequency, raw normalized shape, weight) for inspection
    n_t = y.shape[-1]
    n_pad = 2 * n_t if n_pad is None else n_pad
    i_ref = int(np.argmax(np.abs(h)))
    shift_idx = int(np.round(shift / dt))
    i0, i1 = max(i_ref - int(before / dt), 0), min(i_ref + int(after / dt), n_t - 1 - max(shift_idx, 0))
    if i1 - i0 < 10 or i0 + shift_idx < 0:
        raise ValueError("The calibration pulse is too close to the end of the trace")

    window = tukey(i1 - i0, 0.4)
    h_w = np.zeros(n_t)
    y_w = np.zeros(n_t)
    h_w[i0:i1] = h[i0:i1] * window
    y_w[i0:i1] = y[i0 + shift_idx:i1 + shift_idx] * window

    f = np.fft.rfftfreq(n_pad, dt)
    ref_f = np.fft.rfft(h_w, n_pad)
    with np.errstate(divide="ignore", invalid="ignore"):
        shape = np.nan_to_num(np.fft.rfft(y_w, n_pad) / ref_f)

    ref_db = 20 * np.log10(np.abs(ref_f) / np.abs(ref_f).max() + 1e-30)
    weight = np.clip((ref_db + 40) / 10, 0, 1)
    band = weight > 0.99

    # remaining (sub sample) delay: linear phase slope, the phase offset is kept
    phase = np.unwrap(np.angle(shape[band]))
    slope = np.polyfit(f[band], phase, 1, w=np.abs(ref_f[band]))[0]
    shape *= np.exp(-1j * slope * f)
    shape /= np.average(np.abs(shape[band]), weights=np.abs(ref_f[band]) ** 2)

    calibrated = weight * shape + (1 - weight)

    return calibrated, (f, shape, weight)


def detect_peaks(delays, x, threshold_rel=0.1, min_distance=0.2, max_peaks=10):
    # peaks of |x| above threshold_rel * max |x|, delays refined by parabolic interpolation, sorted by delay
    dt = np.mean(np.diff(delays))
    abs_x = np.abs(x)
    if not np.any(abs_x > 0):
        return np.array([]), np.array([])
    peak_idx, props = find_peaks(abs_x, height=threshold_rel * abs_x.max(), distance=max(1, int(min_distance / dt)))
    peak_idx = peak_idx[np.argsort(props["peak_heights"])[::-1][:max_peaks]]
    peak_idx = np.sort(peak_idx)

    tau, amp = [], []
    for i in peak_idx:
        offset = 0.0
        if 0 < i < len(x) - 1:
            y0, y1, y2 = abs_x[i - 1], abs_x[i], abs_x[i + 1]
            denom = y0 - 2 * y1 + y2
            offset = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        tau.append(delays[i] + offset * dt)
        # a delay between two samples is split over neighbouring samples, the amplitude is their sum (neighbours with
        # the same sign / a similar phase)
        neighbours = x[max(i - 1, 0):i + 2]
        amp.append(np.sum(neighbours[np.real(neighbours * np.conj(x[i])) > 0]))

    return np.array(tau), np.array(amp)


def transmission_n_d(tau, amp):
    # main pulse: largest peak, delay (n - 1) d / c relative to the reference; first echo: next peak, 2 n d / c later
    if len(tau) < 2:
        return None
    main = int(np.argmax(np.abs(amp)))
    if main + 1 >= len(tau):
        return None
    tau_main, delta_tau = tau[main], tau[main + 1] - tau[main]
    if delta_tau <= 2 * tau_main or tau_main <= 0:
        return None
    n = delta_tau / (delta_tau - 2 * tau_main)
    d = c_thz * (delta_tau - 2 * tau_main) / 2

    return n, d


class SparseDeconvolution(EvalComponentBase):
    # not used here
    transmission_model = None
    selected_cost_fun = None
    number_of_workers = None
    freq_range = None

    deconv_grp = "Sparse deconvolution"
    geometry = TEnum(Geometry, default_value=Geometry.transmission, group=deconv_grp,
                     help="Transmission: reference without sample, peaks are the transmitted pulse and its echoes. "
                          "Reflection: reference from a mirror, peaks are the interface reflections").tag(
        name="Geometry", priority=0)
    lambda_rel = Float(0.01, min=0.0, max=1.0, group=deconv_grp,
                       help="L1 regularization relative to the smallest value that gives an all zero impulse "
                            "response. Larger: fewer, cleaner peaks; smaller: better fit, more noise peaks. "
                            "Disable the time domain window in the preprocessing if it cuts off echoes"
                       ).tag(name="Regularization (relative)", priority=1, significant_figures=4)
    max_iter = Int(20000, min=1, group=deconv_grp).tag(name="Maximum iterations", priority=2)
    tol = Float(1e-6, group=deconv_grp,
                help="Stop when the relative change of the impulse response per iteration is below this value"
                ).tag(name="Tolerance", priority=3, significant_figures=3)
    use_hilbert = Bool(False, group=deconv_grp,
                       help="Each peak is a phase shifted reference pulse (reference and its Hilbert transform), "
                            "the peaks get an amplitude and a phase. Absorbs constant phase offsets (focusing, "
                            "phase shifts at interfaces) which otherwise show up as small side peaks. A frequency "
                            "dependent amplitude change (absorption, diffraction) still shows up as a small side "
                            "peak, a slightly larger regularization (~0.03) removes it but pulls the phases "
                            "towards 0").tag(
        name="Phase shifted pulses (Hilbert)", priority=4)
    calibrate_shape = Bool(False, group=deconv_grp,
                           help="The pulse shape change caused by the sample (phase offset, dispersion, frequency "
                                "dependent amplitude) is calibrated from the strongest peak of a first deconvolution "
                                "and included in the reference. Transmission: echo k (2k more passes through the "
                                "sample) uses the shape to the power 2k + 1. Reflection: the shape of the strongest "
                                "reflection is used for all delays. Usually removes the side peaks without the "
                                "Hilbert option (the calibrated shape contains the phase offset), with the Hilbert "
                                "option the phases are relative to the calibration pulse"
                           ).tag(name="Self-calibrated pulse shape",
                                                                         priority=5)

    peak_grp = "Peak detection"
    peak_threshold = Float(0.05, min=0.0, max=1.0, group=peak_grp).tag(name="Peak threshold (rel. to max)",
                                                                      priority=0)
    min_peak_distance = Quantity(Q_(0.3, "ps"), group=peak_grp).tag(name="Minimum peak distance", priority=1)
    max_peaks = Int(10, min=1, group=peak_grp).tag(name="Maximum number of peaks", priority=2)
    n_layer = Float(1.0, group=peak_grp,
                    help="Refractive index used to convert the delays between interface reflections into "
                         "thicknesses (1: optical thickness)").tag(name="Layer refractive index", priority=3)

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

    def traces(self, meas_list):
        # sample and matching reference traces for the result entries (average and / or individual measurements)
        refs = self.measurement_selector.get_matching_refs(meas_list)
        sam_td = self.dataset.get_multi_data(meas_list)
        ref_td = self.dataset.get_multi_data(refs)

        t_sam, t_ref = sam_td[:, :, 0], ref_td[:, :, 0]
        dt = np.mean(np.diff(t_sam[0]))
        if not np.allclose(np.diff(t_sam), dt, rtol=1e-3) or not np.allclose(np.diff(t_ref), dt, rtol=1e-3):
            raise ValueError("Sparse deconvolution needs a uniform time axis with equal sampling for sample and "
                             "reference")
        t0_offset = t_sam[:, 0] - t_ref[:, 0]

        entries = {}
        if len(meas_list) == 1 or self.averaging in (Averaging.only_individual, Averaging.both):
            entries.update({meas: (sam_td[i, :, 1], ref_td[i, :, 1], t0_offset[i]) for i, meas in enumerate(meas_list)})
        if len(meas_list) > 1 and self.averaging in (Averaging.only_avg, Averaging.both):
            entries = {"Average": (np.mean(sam_td[:, :, 1], axis=0), np.mean(ref_td[:, :, 1], axis=0),
                                   np.mean(t0_offset)), **entries}

        return entries, t_sam[0] - t_sam[0, 0], dt

    def calibrated_atoms(self, y, h, x, delay_idx, t0_offsets, dt):
        # per trace: pulse shape calibrated with the strongest peak of the first deconvolution x. Transmission: delays
        # around echo k (main pulse: k = 0) get the shape^(2k + 1), reflection: the same shape for all delays
        n_traces, n_pad = x.shape
        min_distance = self.min_peak_distance.to("ps").magnitude
        shapes, masks, calibrations = [], [], []
        for i in range(n_traces):
            delays = delay_idx + t0_offsets[i]
            order = np.argsort(delays)
            tau, amp = detect_peaks(delays[order], x[i, order], self.peak_threshold, min_distance, self.max_peaks)
            shape_i, mask_i, calibration = [np.ones(n_pad // 2 + 1)], [np.ones(n_pad, dtype=bool)], None
            if len(tau):
                main = int(np.argmax(np.abs(amp)))
                tau_main = tau[main]
                delta_tau = None
                if self.geometry == Geometry.transmission and main + 1 < len(tau):
                    delta_tau = tau[main + 1] - tau_main
                after = min(0.45 * delta_tau, 8.0) if delta_tau else 8.0
                try:
                    shape, calibration = calibrate_pulse_shape(y[i], h[i], dt, tau_main - t0_offsets[i], 3.0, after,
                                                               n_pad)
                    if delta_tau:
                        # one order per detected echo (each order costs one atom), the last one covers the rest
                        n_orders = max(1, min(len(tau) - main, 6))
                        region = np.clip(np.floor((delays - tau_main) / delta_tau + 0.5), 0, n_orders - 1)
                        shape_i = [shape ** (2 * k + 1) for k in range(n_orders)]
                        mask_i = [region == k for k in range(n_orders)]
                    else:
                        shape_i = [shape]
                except ValueError as e:
                    logging.warning(f"Pulse shape calibration failed, using the reference pulse: {e}")
            shapes.append(shape_i)
            masks.append(mask_i)
            calibrations.append(calibration)

        n_orders = max(len(s) for s in shapes)
        shapes_arr = np.ones((n_traces, n_orders, n_pad // 2 + 1), dtype=complex)
        masks_arr = np.zeros((n_traces, n_orders, n_pad), dtype=bool)
        for i in range(n_traces):
            shapes_arr[i, :len(shapes[i])] = shapes[i]
            masks_arr[i, :len(masks[i])] = masks[i]

        return shapes_arr, masks_arr, calibrations

    def evaluate(self, meas_list):
        entries, t_rel, dt = self.traces(meas_list)
        y = np.array([e[0] for e in entries.values()])
        h = np.array([e[1] for e in entries.values()])
        t0_offsets = np.array([e[2] for e in entries.values()])
        x, y_model, info = fista_deconvolution(y, h, self.lambda_rel, self.max_iter, self.tol, self.use_hilbert)

        n_t = y.shape[-1]
        k = np.arange(2 * n_t)
        delay_idx = np.where(k < n_t, k, k - 2 * n_t) * dt
        order = np.argsort(delay_idx)

        calibrations, echo_orders = [None] * len(entries), np.ones(len(entries), dtype=int)
        if self.calibrate_shape:
            shapes, masks, calibrations = self.calibrated_atoms(y, h, x, delay_idx, t0_offsets, dt)
            echo_orders = masks.any(axis=-1).sum(axis=-1)
            x, y_model, info = fista_deconvolution(y, h, self.lambda_rel, self.max_iter, self.tol, self.use_hilbert,
                                                   shapes, masks)

        results = []
        for i, (meas, (y_i, _, t0_offset)) in enumerate(entries.items()):
            delays = delay_idx[order] + t0_offset
            x_i = x[i, order]
            tau, amp = detect_peaks(delays, x_i, self.peak_threshold, self.min_peak_distance.to("ps").magnitude,
                                    self.max_peaks)
            results.append({"meas": meas, "delays": delays, "x": x_i, "t": t_rel, "y": y_i, "y_model": y_model[i],
                            "tau": tau, "amp": amp, "n_iter": int(info["n_iter"][i]),
                            "converged": bool(info["converged"][i]),
                            "cost": float(info["cost"][i]), "lambda": float(info["lambda"][i]),
                            "calibration": calibrations[i], "echo_orders": int(echo_orders[i])})

        return results

    @action("Run sparse deconvolution", group=deconv_grp, priority=10)
    def run_sparse_deconvolution(self):
        if self._active_eval is not None:
            logging.warning("Sparse deconvolution is already running")
            return
        selection = self.measurement_selector.get_selection()
        if selection is None or not selection.sams:
            logging.warning("No sample measurements selected")
            return
        meas_list = list(selection.sams)

        evaluator = self
        self._active_eval = evaluator

        def bg_worker():
            try:
                results = self.evaluate(meas_list)
                self._result_ready(self.prepare_result(results))
            except ValueError as e:
                logging.error(f"Sparse deconvolution: {e}")
            except Exception:
                traceback.print_exc()
            finally:
                if self._active_eval is evaluator:
                    self._active_eval = None

        if self._thread_executor is None:
            self._thread_executor = ThreadPoolExecutor(max_workers=1)
        self._thread_executor.submit(bg_worker)

    def derived_values(self, tau, amp):
        values = {}
        if self.geometry == Geometry.transmission:
            n_d = transmission_n_d(tau, amp)
            if n_d is not None:
                values["n"], values["d"] = Q_(n_d[0], ""), Q_(n_d[1], "µm")
        else:
            for i, delta_tau in enumerate(np.diff(tau)):
                values[f"d_{i + 1}"] = Q_(c_thz * delta_tau / (2 * self.n_layer), "µm")

        return values

    def prepare_result(self, results) -> EvalResultData:
        eval_result_data = EvalResultData()
        eval_result_data.result_type = "Sparse deconvolution"
        eval_result_data.dataset_path = self.dataset.data_path
        eval_result_data.model_name = self.geometry.name
        eval_result_data.measurement_quantity = "Time domain signal"

        for res_idx, res in enumerate(results):
            meas = res["meas"]
            label = meas.filepath.name if isinstance(meas, Measurement) else str(meas)
            single = SingleResultData(measurement=label)

            params = {}
            for i, (tau, amp) in enumerate(zip(res["tau"], res["amp"])):
                params[f"tau_{i + 1}"] = Q_(tau, "ps")
                if self.use_hilbert:
                    params[f"a_{i + 1}"] = Q_(np.abs(amp), "")
                    params[f"phi_{i + 1}"] = Q_(np.angle(amp), "rad")
                else:
                    params[f"a_{i + 1}"] = Q_(amp, "")
            params.update(self.derived_values(res["tau"], res["amp"]))
            single.regression_params = params

            single.optimization_info = {
                "fun": res["cost"], "nit": res["n_iter"], "converged": res["converged"],
                "lambda": res["lambda"], "lambda_rel": self.lambda_rel, "geometry": self.geometry.value,
                "hilbert": self.use_hilbert, "peak_count": len(res["tau"]),
                "calibrated_shape": res["calibration"] is not None, "echo_orders": res["echo_orders"],
                "timestamp": str(datetime.now().isoformat()),
            }

            t_axis, delay_axis = Q_(res["t"], "ps"), Q_(res["delays"], "ps")
            if self.use_hilbert:
                x = res["x"]
                impulse_response = {
                    "impulse_response_abs": QuantityDataSet(axes=[delay_axis], data=Q_(np.abs(x), ""),
                                                            axes_labels=["Delay"],
                                                            data_label="Impulse response (Magnitude)"),
                    "impulse_response_phi": QuantityDataSet(axes=[delay_axis],
                                                            data=Q_(np.where(np.abs(x) > 0, np.angle(x), 0), "rad"),
                                                            axes_labels=["Delay"],
                                                            data_label="Impulse response (Phase)"),
                }
            else:
                impulse_response = {
                    "impulse_response": QuantityDataSet(axes=[delay_axis], data=Q_(res["x"], ""),
                                                        axes_labels=["Delay"], data_label="Impulse response")}
            single.datasets = {
                **impulse_response,
                "signal_measured": QuantityDataSet(axes=[t_axis], data=Q_(res["y"], ""),
                                                   axes_labels=["Time"], data_label="Sample signal (measured)"),
                "signal_model": QuantityDataSet(axes=[t_axis], data=Q_(res["y_model"], ""),
                                                axes_labels=["Time"], data_label="Sample signal (reconstructed)"),
                "residual": QuantityDataSet(axes=[t_axis], data=Q_(res["y"] - res["y_model"], ""),
                                            axes_labels=["Time"], data_label="Residual"),
            }
            if res["calibration"] is not None:  # calibrated pulse shape within the reference bandwidth
                f, shape, weight = res["calibration"]
                band = weight > 0.99
                f_axis = Q_(f[band], "THz")
                single.datasets["pulse_shape_abs"] = QuantityDataSet(
                    axes=[f_axis], data=Q_(np.abs(shape[band]), ""), axes_labels=["Frequency"],
                    data_label="Calibrated pulse shape (Magnitude)")
                single.datasets["pulse_shape_phi"] = QuantityDataSet(
                    axes=[f_axis], data=Q_(np.unwrap(np.angle(shape[band])), "rad"), axes_labels=["Frequency"],
                    data_label="Calibrated pulse shape (Phase)")

            if self.use_hilbert:
                peaks_str = ", ".join(f"{t:.3f} ps ({np.abs(a):.3f}, {np.angle(a):+.2f} rad)"
                                      for t, a in zip(res["tau"], res["amp"]))
            else:
                peaks_str = ", ".join(f"{t:.3f} ps ({a:+.3f})" for t, a in zip(res["tau"], res["amp"]))
            derived_str = ", ".join(f"{k} = {v:.4g~P}" for k, v in self.derived_values(res["tau"], res["amp"]).items())
            logging.info(f"Sparse deconvolution {label}: peaks {peaks_str or 'none'}"
                         + (f" -> {derived_str}" if derived_str else "")
                         + ("" if res["converged"] else f" (not converged after {res['n_iter']} iterations)"))

            key = f"result_{res_idx:03d}"
            eval_result_data.results[key] = single
            eval_result_data.optimum_results[label] = key
            eval_result_data.measurement_names.append(label)

        return eval_result_data
