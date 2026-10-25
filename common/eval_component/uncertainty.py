import json
import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from common.default_appsettings import Domain

"""
Measurement uncertainty of the transmission and its Monte Carlo propagation to the evaluated quantities.
Noise of one measured spectrum (E|dS|^2 per frequency bin): additive (detector) noise plus amplitude and timing
jitter proportional to the spectrum, fitted to the differences of consecutive references. The spectra aren't zero
padded, so the noise of neighbouring frequency bins is (nearly) independent.
Reference drift: change of the reference pulse (arrival time, amplitude) between the time of the reference and the
time of the sample measurement, from the time series of all references. With the drift correction the residual error
of its interpolation, estimated by predicting each reference block from its neighbours.
Units: [freq] = THz, [time] = ps
"""


@dataclass
class NoiseModel:
    additive: float = 0.0  # rms of the additive noise per frequency bin (spectrum units)
    amplitude: float = 0.0  # relative amplitude jitter of one measurement
    jitter: float = 0.0  # ps, timing jitter of one measurement
    pair_cnt: int = 0  # reference pairs it is fitted to

    def variance(self, freq, spectrum):
        # E|dS|^2 of one measured spectrum
        rel = self.amplitude ** 2 + (2 * np.pi * freq * self.jitter) ** 2
        return self.additive ** 2 + rel * np.abs(spectrum) ** 2

    def describe(self):
        if not self.pair_cnt:
            return f"additive noise {self.additive:.3g} (from the end of the spectrum, less than 2 references)"
        return (f"amplitude jitter {100 * self.amplitude:.2f} %, timing jitter {1e3 * self.jitter:.2f} fs, additive "
                f"noise {self.additive:.3g} ({self.pair_cnt} consecutive reference pairs)")


@dataclass
class TransmissionUncertainty:
    variance: np.ndarray  # E|dT|^2 per frequency of the (mean) transmission
    timing: float = 0.0  # ps, standard uncertainty of the sample / reference timing (drift)
    amplitude: float = 0.0  # relative standard uncertainty of the reference amplitude (drift)
    meas_cnt: int = 1
    empirical: bool = False  # variance from the scatter of the measurements (larger than the noise model's)

    def describe(self):
        source = "scatter of the measurements" if self.empirical else "noise model"
        return (f"noise from the {source}, median |dT| {np.sqrt(np.median(self.variance)):.2g}, reference drift "
                f"{1e3 * self.timing:.2f} fs / {100 * self.amplitude:.2f} %")


def fit_noise_model(freq, spectra, times, max_gap_factor=3.0):
    # spectra: (N, F) of the references, times: their measurement times (s). Pairs further apart than max_gap_factor x
    # the median time between references (e.g. between blocks) aren't used, their difference is mostly drift
    order = np.argsort(times)
    spectra, times = np.asarray(spectra)[order], np.asarray(times, dtype=float)[order]
    if len(spectra) < 2:
        tail = spectra[0, -max(len(freq) // 10, 1):]
        return NoiseModel(additive=float(np.sqrt(np.mean(np.abs(tail) ** 2))))

    gaps = np.diff(times)
    pairs = gaps <= max_gap_factor * max(np.median(gaps), 1e-9)
    diff = spectra[1:][pairs] - spectra[:-1][pairs]
    # |diff|^2 of complex normal noise is exponential with the mean 2 var and the median 2 var ln 2. The median
    # ignores the pairs that differ by more than noise (references at the edge of a holder, beam blocked partly)
    var = np.median(np.abs(diff) ** 2, axis=0) / (2 * np.log(2))
    level = np.mean(np.abs(spectra) ** 2, axis=0)
    valid = (freq > 0) & (var > 0)
    w = 2 * np.pi * freq[valid]
    # relative least squares: every frequency counts the same, from the noise floor up to the spectrum maximum
    basis = np.column_stack((np.ones(valid.sum()), level[valid], w ** 2 * level[valid])) / var[valid, None]
    scale = np.linalg.norm(basis, axis=0)
    scale[scale == 0] = 1
    coef, _ = nnls(basis / scale, np.ones(valid.sum()))
    coef /= scale

    return NoiseModel(additive=float(np.sqrt(coef[0])), amplitude=float(np.sqrt(coef[1])),
                      jitter=float(np.sqrt(coef[2])), pair_cnt=int(pairs.sum()))


def pulse_parameters(traces, time_axis):
    # arrival time (ps, sub-sample peak of |y| by a parabola) and peak to peak value of each trace (N, T)
    traces = np.asarray(traces, dtype=float)
    idx = np.argmax(np.abs(traces), axis=1)
    idx = np.clip(idx, 1, traces.shape[1] - 2)
    rows = np.arange(len(traces))
    y0, y1, y2 = np.abs(traces[rows, idx - 1]), np.abs(traces[rows, idx]), np.abs(traces[rows, idx + 1])
    denom = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = np.where(denom != 0, 0.5 * (y0 - y2) / denom, 0.0)
    dt = float(np.mean(np.diff(time_axis)))

    return time_axis[idx] + np.clip(offset, -1, 1) * dt, np.ptp(traces, axis=1)


class UncertaintyEstimator:
    # noise model and reference pulse series of a dataset, cached for the current references and preprocessing
    # blocks used for the residual drift: within this many block gaps of the measurement times, at least the minimum
    residual_block_range = 3
    residual_min_blocks = 7
    # blocks with a median amplitude below this ratio of the 90th percentile of all blocks are clipped
    unclipped_ratio = 0.98

    def __init__(self, dataset):
        self.dataset = dataset
        self._key = None
        self.noise_model = None
        self._ref_times = self._ref_delays = self._ref_amps = None
        self._ref_index = {}
        self._residual = {}

    def _cache_key(self):
        from common.settings import component_settings
        refs = self.dataset.measurements["refs"]
        return (tuple(ref.identifier for ref in refs),
                json.dumps(component_settings(self.dataset.settings.pp_opt), sort_keys=True, default=str))

    def update(self):
        key = self._cache_key()
        if key == self._key:
            return
        refs = sorted(self.dataset.measurements["refs"], key=lambda m: m.meas_time)
        if not refs:
            raise ValueError("The uncertainty estimate needs reference measurements")
        times = np.array([ref.meas_time.timestamp() for ref in refs])
        fd = self.dataset.get_multi_data(refs, Domain.Frequency)
        td = self.dataset.get_multi_data(refs, Domain.Time)
        freq = fd[0, :, 0].real
        self.noise_model = fit_noise_model(freq, fd[:, :, 1], times)
        self._ref_times = times
        self._ref_delays, self._ref_amps = pulse_parameters(td[:, :, 1].real, td[0, :, 0].real)
        self._ref_index = {ref: i for i, ref in enumerate(refs)}
        self._residual = {}
        self._key = key
        logging.info(f"Measurement noise fitted to the references: {self.noise_model.describe()}")

    def drift_offsets(self, meas_list, ref_list):
        # change of the reference pulse from the reference's to the sample's measurement time, interpolated in the
        # reference series: timing (ps) and relative amplitude per measurement
        self.update()
        sam_t = np.array([meas.meas_time.timestamp() for meas in meas_list])
        ref_i = np.array([self._ref_index.get(ref, -1) for ref in ref_list])
        delay_at_sam = np.interp(sam_t, self._ref_times, self._ref_delays)
        amp_at_sam = np.interp(sam_t, self._ref_times, self._ref_amps)
        own_delay = np.where(ref_i >= 0, self._ref_delays[ref_i], delay_at_sam)
        own_amp = np.where(ref_i >= 0, self._ref_amps[ref_i], amp_at_sam)

        return delay_at_sam - own_delay, amp_at_sam / own_amp - 1

    def residual_drift(self, drift, times):
        # standard uncertainty (timing in ps, relative amplitude) left by the drift correction around the measurement
        # times (s), see _block_residual. The amplitude only from the unclipped blocks: references at the edge of a
        # holder are clipped by a fixed amount, that isn't drift and the same in every line (a higher reference
        # threshold avoids them)
        self.update()
        key = (drift.interpolate_refs, drift.block_median, drift.block_gap.to("s").magnitude)
        if key not in self._residual:
            blocks, centers, _ = self.dataset._reference_blocks(drift)
            members = [np.array([self._ref_index[ref] for ref in block]) for block in blocks]
            amps = self._ref_amps / np.median(self._ref_amps)
            self._residual[key] = (centers, np.array([1 / len(m) for m in members]),
                                   np.array([np.median(self._ref_delays[m]) for m in members]),
                                   np.array([np.median(amps[m]) for m in members]))
        centers, inv_size, delays, amps = self._residual[key]
        unclipped = amps >= self.unclipped_ratio * np.percentile(amps, 90) if len(amps) else amps.astype(bool)
        times = np.asarray(times, dtype=float)

        return (self._block_residual(centers, delays, inv_size, self.noise_model.jitter, drift.interpolate_refs, times),
                self._block_residual(centers[unclipped], amps[unclipped], inv_size[unclipped],
                                     self.noise_model.amplitude, drift.interpolate_refs, times))

    def _block_residual(self, centers, vals, inv_size, noise, interpolate, times):
        # each block value near the times is predicted from its neighbours like a sample (interpolation, or the
        # previous block without it). Leaving the block out doubles the distance, the error of a random walk like
        # drift grows with its square root: the (robust) rms prediction error / sqrt 2, without the noise of the block
        # medians. Only nearby blocks, the drift changes during a measurement
        if len(centers) < 3:
            return 0.0
        if interpolate:
            w = (centers[1:-1] - centers[:-2]) / (centers[2:] - centers[:-2])
            at, err = centers[1:-1], vals[1:-1] - ((1 - w) * vals[:-2] + w * vals[2:])
            noise_var = noise ** 2 * (inv_size[1:-1] + w ** 2 * inv_size[:-2] + (1 - w) ** 2 * inv_size[2:])
        else:
            at, err = centers[1:], np.diff(vals)
            noise_var = noise ** 2 * (inv_size[1:] + inv_size[:-1])
        margin = self.residual_block_range * float(np.median(np.diff(centers)))
        near = np.flatnonzero((at >= times.min() - margin) & (at <= times.max() + margin))
        if len(near) < self.residual_min_blocks:
            near = np.argsort(np.abs(at - np.median(times)))[:self.residual_min_blocks]
        rms_sq = (1.4826 * np.median(np.abs(err[near]))) ** 2

        return float(np.sqrt(max(rms_sq - float(np.mean(noise_var[near])), 0.0) / 2))

    def _reference_averaging(self, ref_list, drift):
        # factor of the reference noise variance with the drift correction: the block median averages the block's
        # references, the interpolation between two blocks ((1 - w)^2 + w^2 of the weight w) on average 2 / 3
        blocks, _, block_of = self.dataset._reference_blocks(drift)
        sizes = np.array([len(blocks[block_of[ref]]) if drift.block_median and ref in block_of else 1
                          for ref in ref_list], dtype=float)
        return (2 / 3 if drift.interpolate_refs else 1.0) / sizes

    def transmission_uncertainty(self, meas_list, ref_list, freq_idx, opt) -> TransmissionUncertainty:
        # of the mean transmission of the measurements (each against its reference) on the frequency indices. opt:
        # Settings -> Evaluation -> Uncertainty
        self.update()
        meas_list, ref_list = list(meas_list), list(ref_list)
        freq = self.dataset.freq_axis[freq_idx]
        sam = self.dataset.get_multi_data(meas_list, Domain.Frequency)[:, freq_idx, 1]
        ref = self.dataset.get_multi_data(ref_list, Domain.Frequency)[:, freq_idx, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_sq = np.abs(sam / ref) ** 2
            sam_part = np.nan_to_num(t_sq * self.noise_model.variance(freq, sam) / np.abs(sam) ** 2)
            ref_part = np.nan_to_num(t_sq * self.noise_model.variance(freq, ref) / np.abs(ref) ** 2)
        drift = self.dataset.settings.pp_opt.ref_drift
        if drift.interpolate_refs or drift.block_median:
            ref_part = ref_part * self._reference_averaging(ref_list, drift)[:, None]
        n_sam, n_ref = len(meas_list), len({ref.identifier for ref in ref_list})
        variance = np.mean(sam_part, axis=0) / n_sam + np.mean(ref_part, axis=0) / n_ref

        empirical = False
        if n_sam >= 3:
            t = self.dataset.transmission(meas_list, ref_list)[:, freq_idx]
            scatter = (np.var(t.real, axis=0, ddof=1) + np.var(t.imag, axis=0, ddof=1)) / n_sam
            empirical = bool(np.median(scatter) > np.median(variance))
            variance = np.maximum(variance, scatter)

        timing, amplitude = 0.0, 0.0
        if opt.reference_drift and (drift.interpolate_refs or drift.block_median):
            timing, amplitude = self.residual_drift(drift, [meas.meas_time.timestamp() for meas in meas_list])
        elif opt.reference_drift:
            # slow drift: the offsets of the measurements don't average out, the mean offset is the error of the mean
            d_delay, d_amp = self.drift_offsets(meas_list, ref_list)
            timing, amplitude = abs(float(np.mean(d_delay))), abs(float(np.mean(d_amp)))
        timing = float(np.hypot(timing, opt.extra_timing.to("ps").magnitude))
        amplitude = float(np.hypot(amplitude, opt.extra_amplitude / 100))

        return TransmissionUncertainty(variance=variance, timing=timing, amplitude=amplitude, meas_cnt=n_sam,
                                       empirical=empirical)


def estimator_for(dataset) -> UncertaintyEstimator:
    estimator = getattr(dataset, "_uncertainty_estimator", None)
    if estimator is None:
        estimator = dataset._uncertainty_estimator = UncertaintyEstimator(dataset)
    return estimator


def random_generator(opt):
    return np.random.default_rng(opt.random_seed if opt.random_seed > 0 else None)


def perturbed_transmissions(freq, t, unc: TransmissionUncertainty, count, rng):
    # (count, F): t (1 + a) exp(-i w tau) + complex normal noise with E|dT|^2 = unc.variance. The sign of the phase
    # factor doesn't matter, tau is symmetric around 0
    tau = rng.normal(0.0, unc.timing, (count, 1)) if unc.timing > 0 else np.zeros((count, 1))
    amp = rng.normal(0.0, unc.amplitude, (count, 1)) if unc.amplitude > 0 else np.zeros((count, 1))
    noise = np.sqrt(unc.variance / 2) * (rng.standard_normal((count, len(freq)))
                                         + 1j * rng.standard_normal((count, len(freq))))

    return np.asarray(t) * (1 + amp) * np.exp(-2j * np.pi * freq * tau) + noise


def replicate_optimum(results):
    # thickness of the q-value minimum of one Monte Carlo replicate (window scan at a fixed shift): vertex of the
    # parabola through the best scanned thickness and its neighbours, n(f) interpolated between the two closest fits.
    # at_edge: the minimum is at the end of the window (the thickness is the window end, the spread underestimated)
    results = sorted(results, key=lambda res: res.d.magnitude)
    d = np.array([res.d.to("µm").magnitude for res in results])
    q = np.array([float(np.asarray(res.optimization_info["q_val"].magnitude)) for res in results])
    i = int(np.argmin(q))
    at_edge = i == 0 or i == len(d) - 1
    d_opt = d[i]
    if not at_edge:
        a, b, _ = np.polyfit(d[i - 1:i + 2] - d[i], q[i - 1:i + 2], 2)
        if a > 0:
            d_opt = float(np.clip(d[i] - b / (2 * a), d[i - 1], d[i + 1]))
    j = int(np.clip(np.searchsorted(d, d_opt) - 1, 0, len(d) - 2))
    w = (d_opt - d[j]) / (d[j + 1] - d[j]) if d[j + 1] > d[j] else 0.0
    n_lo, n_hi = (np.asarray(results[k].datasets["n"].data.magnitude) for k in (j, j + 1))

    return float(d_opt), (1 - w) * n_lo + w * n_hi, at_edge


@dataclass
class ThicknessUncertainty:
    d_std: float  # µm
    d_mean: float  # µm, mean of the replicates
    n_std: np.ndarray  # std of the real part + 1j std of the imaginary part
    samples: int
    at_edge: int  # replicates with the minimum at the end of the scanned window

    @classmethod
    def from_replicates(cls, optima):
        d = np.array([opt[0] for opt in optima])
        n = np.array([opt[1] for opt in optima])
        return cls(d_std=float(np.std(d, ddof=1)), d_mean=float(np.mean(d)),
                   n_std=np.std(n.real, axis=0, ddof=1) + 1j * np.std(n.imag, axis=0, ddof=1),
                   samples=len(optima), at_edge=int(sum(opt[2] for opt in optima)))
