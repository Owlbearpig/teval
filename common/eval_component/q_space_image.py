import json
import logging
import os
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import QObject, Signal
from traitlets import Bool, Enum as TEnum, Integer, Unicode

from common.components import action
from common.consts import c_thz
from common.dataset import DataSetFormatter
from common.default_appsettings import QuantityEnum, Domain
from common.eval_component.n_guess import NGuessMethod
from common.eval_component.q_space_scan import ScanTarget, OptimizationCancelled
from common.eval_component.uncertainty import estimator_for
from common.eval_component.dynamic_range import noise_floor, alpha_max
from common.settings import component_settings
from common.traits import Q_, ValueRange, Path as PathTrait
from mpl_settings import mpl_style_params

action = partial(action, check_init=True, rc_params=mpl_style_params)

q_space_quantities = (QuantityEnum.QSpaceThickness, QuantityEnum.QSpaceShift, QuantityEnum.QSpaceQVal,
                      QuantityEnum.QSpaceRefractiveIdx, QuantityEnum.QSpaceAbsorptionCoe,
                      QuantityEnum.QSpaceReliable, QuantityEnum.QSpaceMinWidth, QuantityEnum.QSpaceQRatio,
                      QuantityEnum.QSpaceDDeviation, QuantityEnum.QSpaceFullScan,
                      QuantityEnum.QSpaceBandRefractiveIdx, QuantityEnum.QSpaceBandAbsorptionCoe,
                      QuantityEnum.QSpaceThicknessUncertainty)


@dataclass
class PixelResult:
    d: float  # µm
    shift: float  # fs
    q_val: float
    n: np.ndarray  # on the fit range of the evaluation
    full_scan: bool
    # q(d) of the last scan at the optimum shift, sorted by d
    profile_d: np.ndarray = field(default_factory=lambda: np.empty(0))
    profile_q: np.ndarray = field(default_factory=lambda: np.empty(0))
    min_width: float = np.nan  # µm, minimum_width of the profile
    d_uncert: float = np.nan  # µm, Monte Carlo thickness uncertainty (on demand)


@dataclass
class PixelConfidence:
    min_width: float  # µm, nan if the optimum is no interior minimum of the scanned q(d)
    width_ratio: float  # min_width / median min_width of all pixels
    q_ratio: float  # q-value / median of the neighbours, nan with less than 3 neighbours
    d_deviation: float  # µm, thickness - median of the neighbours, nan with less than 3 neighbours
    d_limit: float  # µm, allowed thickness deviation
    reliable: bool


@dataclass
class ImageContext:
    common_opt_params: dict
    t_exp: dict
    n_guess: dict
    d_axis: np.ndarray  # thickness axis of the full scans, narrowed by the probe pixels
    d_step: float
    shift_axis: np.ndarray
    time_of_flight: dict
    full_d_axis: np.ndarray  # thickness axis of the q-space evaluation settings
    full_d_step: float


class ImageEvalSignal(QObject):
    finished = Signal()


def q_of(res):
    return float(np.asarray(res.optimization_info["q_val"].magnitude))


def q_profile(results, shift):
    # q(d) of the results at the shift, sorted by d
    pts = {float(res.d.to("µm").magnitude): q_of(res) for res in results if abs(res.shift.magnitude - shift) < 1e-9}
    d = np.array(sorted(pts))
    return d, np.array([pts[v] for v in d])


def min_parabola(d, q, d_opt, side_cnt=1):
    # the scanned thicknesses from side_cnt below to side_cnt above the optimum, the optimum's thickness and the
    # coefficients of the (least squares) parabola through them as polynomial in d - optimum. None if the optimum has
    # no scanned neighbour on both sides
    if len(d) < 3:
        return None
    idx = int(np.argmin(np.abs(d - d_opt)))
    if idx == 0 or idx == len(d) - 1:
        return None
    sel = slice(max(idx - side_cnt, 0), idx + side_cnt + 1)
    return d[sel], d[idx], np.polyfit(d[sel] - d[idx], q[sel], 2)


def minimum_width(d, q, d_opt):
    # distance from the minimum at which a parabola through the optimum and up to two scanned thicknesses on each side
    # (least squares, three points are too noisy with a narrow window) reaches twice the minimum q-value: a sharp
    # minimum pins the thickness down, a shallow one doesn't. nan if the optimum has no scanned neighbour on both sides
    # or the points don't form a minimum
    fit = min_parabola(d, q, d_opt, side_cnt=2)
    if fit is None:
        return np.nan
    _, center, (a, b, c) = fit
    if a <= 0:
        return np.nan
    q_min = c - b ** 2 / (4 * a)
    if q_min <= 0:
        q_min = q[np.argmin(np.abs(d - center))]
    return float(np.sqrt(q_min / a))


def alpha_of(freq, n):
    return 4 * np.pi * freq * n.imag / (1e-4 * c_thz)


_code_version = None


def code_version():
    # git commit of the repository (with "-dirty" for uncommitted changes), "" if git isn't available
    global _code_version
    if _code_version is None:
        try:
            _code_version = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=12"],
                                           cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
                                           timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            _code_version = ""
    return _code_version


def scan_levels(keys, seed_stride):
    # coarse to fine pixel subsets: every seed_stride-th pixel in x and y first, then halving the spacing
    remaining, levels, stride = set(keys), [], max(1, seed_stride)
    while remaining:
        level = sorted(k for k in remaining if k[0] % stride == 0 and k[1] % stride == 0)
        if level:
            levels.append((stride, level))
            remaining.difference_update(level)
        stride = max(1, stride // 2)

    return levels


class QSpaceImageEval(DataSetFormatter):
    """
    Q-space evaluation of every pixel of the image grid. A few random probe pixels get the full thickness scan of the
    q-space evaluation and set the (narrower) thickness range of the later full scans. Seed pixels get the full scan,
    all other pixels only scan a narrow window around the thickness of their evaluated neighbours (coarse to fine, so
    the scans of one level run in parallel). Pixels without sample (time of flight, value mask) are skipped. The
    results are provided to the image plot and the regression as the QSpace* quantities.
    """
    area_grp = "Evaluated area"
    confine_to_area = Bool(False, group=area_grp).tag(name="Confine to area", priority=0)
    area_x_range = ValueRange([Q_(0.0, "mm"), Q_(0.0, "mm")], group=area_grp).tag(name="x-range", priority=1)
    area_y_range = ValueRange([Q_(0.0, "mm"), Q_(0.0, "mm")], group=area_grp).tag(name="y-range", priority=2)
    use_value_mask = Bool(False, group=area_grp,
                          help="Pixels whose mean value of the mask quantity lies outside the value range are not "
                               "evaluated, e.g. the opaque holder with the peak to peak value. The values are "
                               "calculated like the image plot (selected frequency, complex value representation), "
                               "so the range can be read off the image of the quantity. Select the quantity "
                               "\"Q-space sample mask\" in the image plot to see which pixels are evaluated. Pixels "
                               "below the minimum time of flight (Settings -> Evaluation -> Image evaluation) are "
                               "skipped as air").tag(name="Skip pixels outside a value range", priority=4)
    mask_quantity = TEnum(QuantityEnum, default_value=QuantityEnum.P2P, group=area_grp).tag(
        name="Mask quantity", priority=5)
    mask_range = ValueRange([Q_(0.0, ""), Q_(0.0, "")], group=area_grp).tag(name="Mask value range", priority=6)
    mask_erosion = Integer(0, min=0, group=area_grp,
                           help="Number of pixel rings removed from the sample area next to skipped pixels, e.g. "
                                "the pixels at the holder edge that are only partly covered by the sample").tag(
        name="Shrink the sample area (pixels)", priority=7)
    pixel_binning = Integer(1, min=1, group=area_grp,
                            help="Evaluates blocks of n x n pixels as one pixel (mean transmission), for a quick "
                                 "preview or noisy data. Re-evaluations and continued evaluations use the binning of "
                                 "the image evaluation").tag(name="Pixel binning", priority=8)

    run_grp = "Run"
    plot_when_done = Bool(True, group=run_grp).tag(name="Plot image when finished", priority=10)
    evaluated_pixels = Unicode("", read_only=True, group=run_grp).tag(name="Evaluated pixels", priority=11,
                                                                     en_save=False)

    reeval_grp = "Re-evaluate"
    inspected_pixel = Unicode("", read_only=True, group=reeval_grp,
                              help="Last pixel clicked in the image plot of a Q-space quantity").tag(
        name="Inspected pixel", priority=20, en_save=False)

    results_grp = "Results"
    band_range = ValueRange([Q_(0.0, "THz"), Q_(0.0, "THz")], group=results_grp,
                            help="Frequency band of the quantities \"Q-space refractive index (band mean)\" and "
                                 "\"Q-space absorption coe. (band mean)\" (within the fit range). Equal bounds: the "
                                 "whole fit range").tag(name="Band of the band mean quantities", priority=30)
    results_file = PathTrait(PathTrait.default_value, is_dir=False, must_exist=False, group=results_grp,
                             help="Image results (.npz) loaded by \"Load image results\", if no file is selected the "
                                  "newest q-space image result in the result directory").tag(
        name="Results file", priority=31)
    thickness_stats = Unicode("", read_only=True, group=results_grp).tag(name="Thickness statistics", priority=32,
                                                                        en_save=False)

    uncertainty_grp = "Uncertainty"
    uncertainty_stats = Unicode("", read_only=True, group=uncertainty_grp,
                                help="Monte Carlo thickness uncertainty of the pixels (settings: Settings -> "
                                     "Evaluation -> Uncertainty), shown as the quantity \"Q-space thickness "
                                     "uncertainty\"").tag(name="Thickness uncertainty", priority=40, en_save=False)

    @property
    def opt(self):
        # tuning settings, Settings -> Evaluation -> Image evaluation
        return self.settings.eval_opt.image_eval

    def __init__(self, q_space_eval, image_plot=None, **kwargs):
        # q_space_eval: the parent QSpaceEval (not imported, q_space_eval imports this module)
        super().__init__(q_space_eval.dataset, **kwargs)
        self.qs = q_space_eval
        self.image_plot = image_plot

        self.pixel_results: dict[tuple, PixelResult] = {}
        self._pixel_meas = {}
        self._meas_pixel = {}
        self._freq_idx = None
        self._freq_axis = None
        self._q_ref = {}  # median q-value of the neighbours of the window scan targets
        self._task_cnt = 0
        self._mask_vals = {}
        self._skipped_keys = set()
        self._masked_keys = set()  # skipped by time of flight / value mask, not by shrinking the area
        # untrusted results that a follow-up scan replaces, still to be done if the run was cancelled before it
        self._provisional = set()
        self._last_autosave = 0.0
        self._run_info = {}  # echo count, knot count and probe thickness axis of the image evaluation
        self._inspected_key = None
        self._replot_key = None
        self._reset_counters()

        self._signals = ImageEvalSignal()
        self._signals.finished.connect(self._on_finished)

        register = self.dataset.register_quantity_func
        register(QuantityEnum.QSpaceThickness, partial(self.scalar_quantity, "d"))
        register(QuantityEnum.QSpaceShift, partial(self.scalar_quantity, "shift"))
        register(QuantityEnum.QSpaceQVal, partial(self.scalar_quantity, "q_val"))
        register(QuantityEnum.QSpaceRefractiveIdx, self.refractive_idx_quantity)
        register(QuantityEnum.QSpaceAbsorptionCoe, self.absorption_quantity)
        register(QuantityEnum.QSpaceSampleMask, self.sample_mask_quantity)
        register(QuantityEnum.QSpaceReliable, partial(self.confidence_quantity, "reliable"))
        register(QuantityEnum.QSpaceMinWidth, partial(self.confidence_quantity, "min_width"))
        register(QuantityEnum.QSpaceQRatio, partial(self.confidence_quantity, "q_ratio"))
        register(QuantityEnum.QSpaceDDeviation, partial(self.confidence_quantity, "d_deviation"))
        register(QuantityEnum.QSpaceFullScan, partial(self.scalar_quantity, "full_scan"))
        register(QuantityEnum.QSpaceThicknessUncertainty, partial(self.scalar_quantity, "d_uncert"))
        register(QuantityEnum.QSpaceBandRefractiveIdx, partial(self.band_quantity, lambda res: res.n.real))
        register(QuantityEnum.QSpaceBandAbsorptionCoe,
                 partial(self.band_quantity, lambda res: alpha_of(self._freq_axis, res.n)))

        if image_plot is not None:
            image_plot.click_handlers.append(self._on_image_click)

    def _lookup(self, meas_list):
        return [self.pixel_results.get(self._meas_pixel.get(meas)) for meas in meas_list]

    def scalar_quantity(self, attr, meas_list):
        return np.array([np.nan if res is None else float(getattr(res, attr)) for res in self._lookup(meas_list)])

    def _deviation_limit(self, d_nbrs):
        # median of the neighbour thicknesses and the allowed deviation from it: the threshold or 3x the robust
        # scatter of the neighbours (1.4826 * MAD), so the normal scatter of the thickness isn't flagged
        d_med = float(np.median(d_nbrs))
        mad = float(np.median(np.abs(d_nbrs - d_med)))
        return d_med, max(self.opt.outlier_threshold.to("µm").magnitude, 3 * 1.4826 * mad)

    def median_width(self):
        widths = [res.min_width for res in self.pixel_results.values()]
        return float(np.nanmedian(widths)) if np.any(np.isfinite(widths)) else np.nan

    def pixel_confidence(self, key, median_width=None) -> PixelConfidence:
        # median_width: of all pixels, pass it when calling this for many pixels
        res = self.pixel_results[key]
        median_width = self.median_width() if median_width is None else median_width
        width_ratio = res.min_width / median_width
        nbrs = self._neighbors(key, 1)
        q_ratio = d_dev = d_limit = np.nan
        if len(nbrs) >= 3:
            d_med, d_limit = self._deviation_limit(np.array([nbr.d for nbr in nbrs]))
            d_dev = res.d - d_med
            q_ratio = res.q_val / max(float(np.median([nbr.q_val for nbr in nbrs])), 1e-300)
        # nan neighbour values (isolated pixel) don't count against the pixel
        reliable = (np.isfinite(res.min_width) and not width_ratio > self.opt.max_width_ratio
                    and not q_ratio > self.opt.max_q_ratio and not abs(d_dev) > d_limit)

        return PixelConfidence(min_width=res.min_width, width_ratio=width_ratio, q_ratio=q_ratio, d_deviation=d_dev,
                               d_limit=d_limit, reliable=bool(reliable))

    def confidence_quantity(self, attr, meas_list):
        cache, median_width = {}, self.median_width()
        out = np.full(len(meas_list), np.nan)
        for i, meas in enumerate(meas_list):
            key = self._meas_pixel.get(meas)
            if key not in self.pixel_results:
                continue
            if key not in cache:
                cache[key] = float(getattr(self.pixel_confidence(key, median_width), attr))
            out[i] = cache[key]
        return out

    def _spectral_quantity(self, meas_list, func):
        # (N_meas, F) on the full frequency axis, nan outside of the fit range and for pixels without a result
        res_list = self._lookup(meas_list)
        out = np.full((len(res_list), len(self.dataset.freq_axis)), np.nan, dtype=complex)
        for i, res in enumerate(res_list):
            if res is not None:
                out[i, self._freq_idx] = func(res)
        return out

    def refractive_idx_quantity(self, meas_list):
        return self._spectral_quantity(meas_list, lambda res: res.n)

    def absorption_quantity(self, meas_list):
        return self._spectral_quantity(meas_list, lambda res: alpha_of(self._freq_axis, res.n)).real

    def band_quantity(self, func, meas_list):
        out = np.full(len(meas_list), np.nan)
        if self._freq_axis is None:
            return out
        f0, f1 = sorted(q.to("THz").magnitude for q in self.band_range)
        band = (self._freq_axis >= f0) & (self._freq_axis <= f1) if f1 > f0 else np.ones(len(self._freq_axis), bool)
        if not band.any():
            return out
        for i, res in enumerate(self._lookup(meas_list)):
            if res is not None:
                out[i] = float(np.mean(func(res)[band]))
        return out

    def mask_values(self, meas_list):
        # mask quantity per measurement, reduced to a real number like the image plot does
        ip = self.image_plot
        freq_range = ip.grid_calc_freq_range if ip is not None else self.freq_range
        freq_range = freq_range.magnitude if hasattr(freq_range, "magnitude") else freq_range
        func = self.dataset.quantity_func(self.mask_quantity, freq_range=freq_range,
                                          d_sam=ip.d_sample if ip is not None else None)
        vals = np.asarray(func(meas_list))
        if vals.ndim == 2:
            freq_idx = ip.sel_freq_idx if ip is not None else self.freq_range_idx[len(self.freq_range_idx) // 2]
            vals = vals[:, freq_idx]

        return ip.to_real(vals) if ip is not None else np.real(vals)

    def _in_sample(self, tof, mask_val):
        if abs(tof) < self.opt.min_time_of_flight.to("ps").magnitude:
            return False
        if self.use_value_mask:
            lo, hi = sorted(v.magnitude for v in self.mask_range)
            return bool(lo <= mask_val <= hi)
        return True

    def _shrink_area(self, sample, skipped):
        # removes mask_erosion rings of sample pixels next to skipped pixels
        sample, skipped = set(sample), set(skipped)
        for _ in range(self.mask_erosion):
            edge = {k for k in sample
                    if any((k[0] + dx, k[1] + dy) in skipped for dx in (-1, 0, 1) for dy in (-1, 0, 1))}
            if not edge:
                break
            sample -= edge
            skipped |= edge
        return sample

    def sample_mask_quantity(self, meas_list):
        # 1: measurement would be evaluated, 0: skipped (time of flight / value mask / shrunk area), with the pixel
        # binning of the settings
        shape = self.dataset.shape_properties
        keys = [self._pixel_key(meas.position, shape, self.pixel_binning) for meas in meas_list]
        tof = np.asarray(self.dataset.time_of_flight(meas_list))
        mask_vals = np.asarray(self.mask_values(meas_list)) if self.use_value_mask else np.zeros(len(meas_list))
        groups = {}
        for i, key in enumerate(keys):
            groups.setdefault(key, []).append(i)
        in_sample = {key: self._in_sample(np.mean(tof[idx]), np.mean(mask_vals[idx])) for key, idx in groups.items()}
        sample = self._shrink_area([k for k, v in in_sample.items() if v], [k for k, v in in_sample.items() if not v])
        return np.array([float(key in sample) for key in keys])

    @staticmethod
    def _pixel_key(position, shape, binning):
        x, y = position
        ix, iy = int(np.argmin(np.abs(x - shape["x_coords"]))), int(np.argmin(np.abs(y - shape["y_coords"])))
        return ix // binning, iy // binning

    def pixel_measurements(self, confine=None, binning=None):
        confine = self.confine_to_area if confine is None else confine
        binning = self.pixel_binning if binning is None else binning
        shape = self.dataset.shape_properties
        x_rng = sorted(q.to("mm").magnitude for q in self.area_x_range)
        y_rng = sorted(q.to("mm").magnitude for q in self.area_y_range)
        pixels = {}
        for meas in self.measurements["sams"]:
            x, y = meas.position
            if confine and not (x_rng[0] <= x <= x_rng[1] and y_rng[0] <= y <= y_rng[1]):
                continue
            pixels.setdefault(self._pixel_key(meas.position, shape, binning), []).append(meas)

        return pixels

    def _pixel_pos(self, key):
        # mm, mean position of the pixel's measurements
        return np.mean([meas.position for meas in self._pixel_meas[key]], axis=0)

    def _run_binning(self):
        return self._run_info.get("binning", 1)

    def _pixel_data(self, pixels, estimate_fp=True, batch_size=500):
        # mean transmission, time of flight (and time of flight n guess) of the measurements of each pixel
        qs = self.qs
        freq_idx = qs.freq_range_idx
        use_tof = qs.effective_n_guess_method == NGuessMethod.tof
        d_init = np.mean(qs.d_opt_axis_bounds.magnitude)
        auto_fp = self.settings.eval_opt.auto_fp_count and estimate_fp

        keys = list(pixels)
        t_exp, n_guess, tof, mask_vals, fp_counts = {}, {}, {}, {}, []
        for i in range(0, len(keys), batch_size):
            qs.check_cancel()
            batch_keys = keys[i:i + batch_size]
            batch_meas = [meas for key in batch_keys for meas in pixels[key]]
            batch_refs = self.measurement_selector.get_matching_refs(batch_meas)
            if len(batch_refs) != len(batch_meas):
                raise ValueError("Could not resolve a reference for every sample measurement")

            t = self.dataset.transmission(batch_meas, batch_refs)[:, freq_idx]
            batch_tof = self.dataset.time_of_flight(batch_meas, batch_refs)
            batch_mask = self.mask_values(batch_meas) if self.use_value_mask else None
            n0 = self.dataset.refractive_index_estimate(batch_meas, batch_refs, d=d_init)[:, freq_idx] \
                if use_tof else None
            if auto_fp:
                fp_counts.append(qs.fp_echo_estimate(batch_meas, batch_refs, np.mean(t, axis=0))[1])

            pos = 0
            for key in batch_keys:
                cnt = len(pixels[key])
                t_exp[key] = np.mean(t[pos:pos + cnt], axis=0)
                tof[key] = float(np.mean(batch_tof[pos:pos + cnt]))
                mask_vals[key] = float(np.mean(batch_mask[pos:pos + cnt])) if batch_mask is not None else 0.0
                if use_tof:
                    n_guess[key] = np.mean(n0[pos:pos + cnt], axis=0)
                pos += cnt

        return t_exp, n_guess, tof, mask_vals, (min(fp_counts) if fp_counts else None)

    def _build_context(self, pixels, run_info=None) -> ImageContext:
        # run_info: re-evaluation of some pixels, the echo count and knot count of the image evaluation are kept so
        # the q-values stay comparable to the other pixels
        qs = self.qs
        t_model_kwargs = qs.base_t_model_kwargs()
        t_exp, n_guess, tof, mask_vals, nfp = self._pixel_data(pixels, estimate_fp=not run_info)
        self._mask_vals.update(mask_vals)
        if run_info:
            nfp = run_info.get("nfp")
        elif nfp is not None:
            logging.info(f"Estimated number of Fabry-Perot echoes {nfp} (minimum of all pixels)")
        if nfp is not None:
            t_model_kwargs["nfp"] = nfp
        t_mean = np.mean(np.array(list(t_exp.values())), axis=0)
        d_axis, d_step = qs.thickness_axis()
        opt_params = qs.common_opt_params(t_model_kwargs, t_mean)
        if run_info and "knot_count" in opt_params and run_info.get("knot_count"):
            opt_params["knot_count"] = run_info["knot_count"]

        return ImageContext(common_opt_params=opt_params,
                            t_exp=t_exp, n_guess=n_guess, d_axis=d_axis, d_step=d_step,
                            shift_axis=qs.shift_axis(), time_of_flight=tof, full_d_axis=d_axis, full_d_step=d_step)

    def _target(self, ctx: ImageContext, key, d_axis=None, d_step=None):
        freq_axis = self.qs.freq_axis
        config = {**ctx.common_opt_params,
                  "t_exp": np.column_stack((freq_axis, ctx.t_exp[key])),
                  "n_guess": np.column_stack((freq_axis, ctx.n_guess[key])) if key in ctx.n_guess else None}
        if d_axis is None:
            d_axis, d_step = ctx.d_axis, ctx.d_step
        return ScanTarget(key=key, opt_config=config, d_axis=d_axis, d_step=d_step)

    def _neighbors(self, key, radius):
        ix, iy = key
        return [self.pixel_results[(ix + dx, iy + dy)]
                for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
                if (dx or dy) and (ix + dx, iy + dy) in self.pixel_results]

    def _centered_target(self, ctx: ImageContext, key, d_center):
        step = self.opt.nbr_d_step.to("µm").magnitude
        w = max(self.opt.nbr_d_window.to("µm").magnitude, step)
        d_axis, d_step = self.qs.thickness_axis(max(d_center - w, step), d_center + w, step)

        return self._target(ctx, key, d_axis, d_step)

    def _window_target(self, ctx: ImageContext, key, nbrs):
        # None if scanning the window around the neighbours takes as many tasks as the full thickness scan
        target = self._centered_target(ctx, key, float(np.median([res.d for res in nbrs])))

        task_cnt = self.qs.scan_task_count([target], self.opt.nbr_iteration_count, ctx.shift_axis)
        full_task_cnt = self.qs.scan_task_count([self._target(ctx, key)], self._full_iterations(), ctx.shift_axis)
        if task_cnt >= full_task_cnt:
            return None
        self._q_ref[key] = float(np.median([res.q_val for res in nbrs]))

        return target

    def _window_check(self, target: ScanTarget):
        # None if the window optimum is trusted, otherwise the reason ("q": q-value far above the neighbours,
        # "edge": optimum at the edge of the window)
        q_val = q_of(target.best)
        if q_val > self.opt.max_q_ratio * self._q_ref[target.key]:
            return "q"
        d = target.best.d.to("µm").magnitude
        if not target.d_axis[0] + 1e-6 < d < target.d_axis[-1] - 1e-6:
            return "edge"
        return None

    @staticmethod
    def _full_scan_check(ctx: ImageContext, target: ScanTarget):
        # "range" if the optimum is at an edge of the narrowed thickness range that isn't a configured bound
        d = target.best.d.to("µm").magnitude
        lo, hi = target.d_axis[0], target.d_axis[-1]
        at_lo = d <= lo + 1e-6 and lo > ctx.full_d_axis[0] + 1e-6
        at_hi = d >= hi - 1e-6 and hi < ctx.full_d_axis[-1] - 1e-6
        return "range" if at_lo or at_hi else None

    def _full_iterations(self):
        return 1 if self.opt.interpolate_minimum else self.qs.iteration_cnt

    @staticmethod
    def _interpolated_minimum(target: ScanTarget):
        # vertex of the parabola through the best result and its scanned neighbour thicknesses (same shift),
        # None if there is no neighbour on both sides or the q-values don't form a minimum
        best = target.best
        fit = min_parabola(*q_profile(target.results, best.shift.magnitude), best.d.to("µm").magnitude)
        if fit is None:
            return None
        x, center, (a, b, _) = fit
        if a <= 0:
            return None
        vertex = center - b / (2 * a)
        if not x[0] < vertex < x[-1] or abs(vertex - center) < 1e-3:
            return None
        return float(vertex)

    def _fit_interpolated_minima(self, executor, targets, label):
        # one fit per target at its interpolated minimum, select_optimum then decides between it and the grid optimum
        by_shift = {}
        for target in targets:
            vertex = self._interpolated_minimum(target)
            if vertex is not None:
                target.d_axis = np.array([vertex])
                by_shift.setdefault(float(target.best.shift.magnitude), []).append(target)
        for shift, group in by_shift.items():
            self._progress_done += len(group)
            self.qs.run_scan(executor, group, 1, np.array([shift]), label=f"{label}, interpolated minimum",
                             kept_datasets=("n",), show_stage=False)

    def _store(self, target: ScanTarget, full_scan):
        best = target.best
        profile_d, profile_q = q_profile(target.results, best.shift.magnitude)
        self.pixel_results[target.key] = PixelResult(d=float(best.d.to("µm").magnitude),
                                                     shift=float(best.shift.to("fs").magnitude),
                                                     q_val=q_of(best),
                                                     n=np.asarray(best.datasets["n"].data.magnitude),
                                                     full_scan=full_scan, profile_d=profile_d, profile_q=profile_q,
                                                     min_width=minimum_width(profile_d, profile_q,
                                                                             best.d.to("µm").magnitude))

    def _pixel_costs(self, ctx: ImageContext):
        # nominal tasks per pixel of a full thickness scan and of a neighbour window scan, with the interpolated
        # minimum fit
        qs = self.qs
        refine_d_cnt, refine_shift_cnt = qs.refine_counts(ctx.shift_axis)

        def cost(d_cnt, iteration_count):
            return (d_cnt * len(ctx.shift_axis) + refine_d_cnt * refine_shift_cnt * (max(1, iteration_count) - 1)
                    + int(self.opt.interpolate_minimum))

        step = self.opt.nbr_d_step.to("µm").magnitude
        window_d_cnt = int(round(2 * max(self.opt.nbr_d_window.to("µm").magnitude, step) / step)) + 1
        full_cost = cost(len(ctx.d_axis), self._full_iterations())

        return full_cost, min(cost(window_d_cnt, self.opt.nbr_iteration_count), full_cost)

    # prior of the actual / nominal cost of a window pixel and its weight in pixels, so the first few window
    # pixels don't decide the estimate (smooth areas ~1.1, areas with thickness steps ~2.5)
    window_cost_prior, window_prior_weight = 1.3, 20
    # tasks per sample pixel of the outlier passes, until they start (~0.05 of the pixels get a window scan)
    outlier_cost_prior = 0.05
    # prior of the fraction of moved windows that are untrusted again, and its weight in moved windows
    move_fail_prior, move_fail_prior_weight = 0.4, 10

    def _move_fail_rate(self):
        fails, attempts = self._move_stats
        return (fails + self.move_fail_prior * self.move_fail_prior_weight) / (attempts + self.move_fail_prior_weight)

    def _window_cost_ratio(self, window_cost):
        # actual / nominal cost of the window pixels including window moves and full scan fallbacks: of the finished
        # levels and of the current level once its window scan is done (with the estimated cost of its known
        # follow-up scans)
        nominal, actual = self._win_nominal, self._win_actual
        if self._level_phase == "follow":
            nominal += self._level_window_cnt * window_cost
            actual += self._progress_done - self._level_window_start + self._level_follow_cost
        weight = self.window_prior_weight * window_cost
        return (actual + self.window_cost_prior * weight) / (nominal + weight) if nominal + weight else 1.0

    def _plan_progress(self, ctx: ImageContext, levels, sample_cnt, first_stage_cost=0):
        # levels: of the coming level scan, sample_cnt: pixels the outlier passes check
        self._plan_levels, self._next_level_idx, self._level_phase = levels, 0, None
        full_cost, window_cost = self._pixel_costs(ctx)
        self._outlier_estimate = self.outlier_cost_prior * sample_cnt * window_cost * (self.opt.outlier_passes > 0)
        self._update_progress_total(ctx, first_stage_cost)

    def _update_progress_total(self, ctx: ImageContext, pending=0):
        # total: the running scan (pending: tasks of the scan that is about to start), the estimated rest of the
        # current level, the following levels (window pixels at the measured cost ratio) and the outlier passes
        if pending:
            self._stage_end = self._progress_done + pending
        full_cost, window_cost = self._pixel_costs(ctx)
        ratio = self._window_cost_ratio(window_cost)
        level_rest = {"before": self._level_window_cnt * window_cost * ratio,
                      "window": self._level_window_cnt * window_cost * max(ratio - 1, 0.0),
                      "follow": self._level_follow_cost}.get(self._level_phase, 0.0)
        future = 0.0
        for idx in range(self._next_level_idx, len(self._plan_levels)):
            keys = [key for key in self._plan_levels[idx][1] if key not in self.pixel_results]
            future += len(keys) * (full_cost if idx == 0 else ratio * window_cost)
        total = max(self._stage_end, self._progress_done) + level_rest + future + self._outlier_estimate
        self.qs.progress_set_total(total)

    def _retry_cost(self, ctx: ImageContext, moves_left):
        # expected tasks after an untrusted window result at the edge: a moved window, which is untrusted again at the
        # rate measured so far, or the full scan when no moves are left
        full_cost, window_cost = self._pixel_costs(ctx)
        if moves_left <= 0:
            return full_cost
        return window_cost + self._move_fail_rate() * self._retry_cost(ctx, moves_left - 1)

    def _follow_up_cost(self, ctx: ImageContext, untrusted, moves_left, full_keys):
        # expected tasks of the scans that follow the untrusted window results (optimum at the edge: moved windows,
        # otherwise a full scan) and of the queued full scans
        full_cost, _ = self._pixel_costs(ctx)
        edge_cnt = sum(reason == "edge" for _, reason in untrusted)
        return ((len(full_keys) + len(untrusted) - edge_cnt) * full_cost
                + edge_cnt * self._retry_cost(ctx, moves_left))

    def _run_targets(self, executor, ctx: ImageContext, targets, iteration_count, label, full_scan,
                     store_untrusted=True):
        # returns (target, reason) of the targets whose optimum is not trusted (window scans) or at the edge of the
        # narrowed thickness range (full scans)
        qs = self.qs
        if not targets:
            return []
        task_cnt = qs.scan_task_count(targets, iteration_count, ctx.shift_axis)
        self._update_progress_total(ctx, task_cnt + len(targets) * self.opt.interpolate_minimum)
        logging.info(f"{label}: {len(targets)} pixel(s)")

        # bounds the number of results kept in memory (only n is kept, a few kB each); each chunk ends with a barrier
        # where workers idle until the slowest task finishes, so it shouldn't be too small either
        chunk_size = max(32 * qs.worker_count, 64)
        untrusted = []
        for i in range(0, len(targets), chunk_size):
            chunk = targets[i:i + chunk_size]
            chunk_label = f"{label}, pixels {i + 1}-{i + len(chunk)} / {len(targets)}"
            self._progress_done += qs.scan_task_count(chunk, iteration_count, ctx.shift_axis)
            qs.run_scan(executor, chunk, iteration_count, ctx.shift_axis, label=chunk_label, kept_datasets=("n",))
            reasons = [self._full_scan_check(ctx, t) if full_scan else self._window_check(t) for t in chunk]
            if self.opt.interpolate_minimum:
                self._fit_interpolated_minima(executor, [t for t, r in zip(chunk, reasons) if r is None], chunk_label)
            for target, reason in zip(chunk, reasons):
                self._task_cnt += len(target.results)
                if reason is not None:
                    untrusted.append((target, reason))
                if reason is None or store_untrusted:
                    self._store(target, full_scan)
                    if reason is None:
                        self._provisional.discard(target.key)
                    else:
                        self._provisional.add(target.key)
                target.results = []
            self._autosave()

        return untrusted

    def _run_full_scans(self, executor, ctx: ImageContext, keys, label):
        at_edge = self._run_targets(executor, ctx, [self._target(ctx, key) for key in keys], self._full_iterations(),
                                    label, full_scan=True)
        if at_edge:
            logging.info(f"{label}: optimum at the edge of the probe thickness range for {len(at_edge)} pixel(s), "
                         f"scanning the full range")
            self._run_targets(executor, ctx, [self._target(ctx, t.key, ctx.full_d_axis, ctx.full_d_step)
                                              for t, _ in at_edge],
                              self._full_iterations(), f"{label}, full range", full_scan=True)

    def _probe_thickness_range(self, executor, ctx: ImageContext, sample_keys):
        # full scans of random pixels, the later full scans only cover the thickness range of their results
        cnt = min(self.opt.probe_pixel_count, len(sample_keys))
        if cnt < 3:
            return
        rng = np.random.default_rng(0)
        keys = [sample_keys[i] for i in rng.choice(len(sample_keys), cnt, replace=False)]
        self._run_full_scans(executor, ctx, keys, "Thickness range probe")

        full_axis = ctx.full_d_axis
        d = np.array([self.pixel_results[key].d for key in keys])
        q_val = np.array([self.pixel_results[key].q_val for key in keys])
        # holder edges, partially covered pixels etc.: high q-value or optimum at a scan bound
        valid = (q_val <= 2 * np.median(q_val)) & (d > full_axis[0] + 1e-6) & (d < full_axis[-1] - 1e-6)
        if valid.sum() < 3:
            logging.info(f"Thickness range probe: only {valid.sum()} / {cnt} valid result(s), scanning the full range")
            return
        margin = self.opt.probe_range_margin.to("µm").magnitude
        lo, hi = d[valid].min() - margin, d[valid].max() + margin
        # on the grid of the configured axis, so the edge check can tell the configured bounds apart
        i0 = max(int(np.searchsorted(full_axis, lo, side="right")) - 1, 0)
        i1 = min(int(np.searchsorted(full_axis, hi, side="left")), len(full_axis) - 1)
        ctx.d_axis = full_axis[i0:i1 + 1]
        logging.info(f"Thickness range probe: {valid.sum()} / {cnt} valid pixel(s), d {d[valid].min():.1f} - "
                     f"{d[valid].max():.1f} µm, full scans now cover {ctx.d_axis[0]:g} - {ctx.d_axis[-1]:g} µm "
                     f"({len(ctx.d_axis)} instead of {len(full_axis)} thicknesses)")

    def _evaluate_image_impl(self):
        qs = self.qs
        pixels = self.pixel_measurements()
        if not pixels:
            raise ValueError("No sample measurements in the evaluated area")

        self.pixel_results = {}
        self._pixel_meas = pixels
        self._meas_pixel = {meas: key for key, meas_list in pixels.items() for meas in meas_list}
        self._freq_idx = qs.freq_range_idx
        self._freq_axis = qs.freq_axis
        self._mask_vals, self._skipped_keys, self._masked_keys, self._provisional = {}, set(), set(), set()
        self._reset_counters()

        qs.reset_state()
        ctx = self._build_context(pixels)
        self._run_info = {"nfp": ctx.common_opt_params.get("nfp"),
                          "knot_count": ctx.common_opt_params.get("knot_count"),
                          "d_axis": ctx.d_axis, "d_step": ctx.d_step, "binning": self.pixel_binning}
        sample_keys = self._sample_keys(ctx, pixels)
        levels = self._levels(sample_keys)

        probe_cnt = min(self.opt.probe_pixel_count, len(sample_keys))
        probe_cnt = probe_cnt if probe_cnt >= 3 else 0
        qs.progress_start(0)
        self._plan_progress(ctx, levels, len(sample_keys), probe_cnt * self._pixel_costs(ctx)[0])
        with qs.process_pool() as executor:
            self._probe_thickness_range(executor, ctx, sample_keys)
            self._run_info["d_axis"] = ctx.d_axis
            self._scan_levels(executor, ctx, levels)
            self._outlier_passes(executor, ctx)
        qs.progress_set_total(self._progress_done)
        logging.info(f"Image evaluation finished, {self._task_cnt} tasks")

    def _sample_keys(self, ctx: ImageContext, pixels):
        in_sample = {key for key in pixels if self._in_sample(ctx.time_of_flight[key], self._mask_vals[key])}
        masked = set(pixels).difference(in_sample)
        self._masked_keys.difference_update(pixels)
        self._masked_keys.update(masked)
        sample_keys = sorted(self._shrink_area(in_sample, self._masked_keys))
        skipped = set(pixels).difference(sample_keys)
        self._skipped_keys.difference_update(pixels)
        self._skipped_keys.update(skipped)
        if masked:
            reasons = f"time of flight below {self.opt.min_time_of_flight.to('ps').magnitude:g} ps"
            if self.use_value_mask:
                reasons += f" or {self.mask_quantity.value} outside of the mask value range"
            logging.info(f"Skipping {len(masked)} pixel(s) with a {reasons} (no sample)")
        if len(skipped) > len(masked):
            logging.info(f"Skipping {len(skipped) - len(masked)} pixel(s) next to them (shrunk sample area)")
        return sample_keys

    def _levels(self, sample_keys):
        levels = scan_levels(sample_keys, self.opt.seed_stride)
        logging.info(f"Evaluating {len(sample_keys)} pixels in {len(levels)} level(s): "
                     + ", ".join(f"{len(keys)} (spacing {stride})" for stride, keys in levels))
        return levels

    def _scan_levels(self, executor, ctx: ImageContext, levels):
        # pixels that already have a result (probe pixels) are skipped, pixels with evaluated neighbours (also outside
        # of the levels, e.g. around a re-evaluated area) get the window scan
        for level_idx, (stride, keys) in enumerate(levels):
            radius = levels[level_idx - 1][0] if level_idx else 1
            full, window = [], []
            for key in keys:
                if key in self.pixel_results:  # probe pixel
                    continue
                nbrs = self._neighbors(key, radius)
                target = self._window_target(ctx, key, nbrs) if nbrs else None
                if target is None:
                    full.append(key)
                else:
                    window.append(target)

            _, window_cost = self._pixel_costs(ctx)
            self._next_level_idx, self._level_phase = level_idx + 1, "before"
            self._level_window_cnt, self._level_follow_cost = len(window), 0.0
            self._stage_end = self._progress_done
            self._update_progress_total(ctx)

            label = f"Level {level_idx + 1} / {len(levels)}"
            self._run_full_scans(executor, ctx, full, f"{label}, full scan")
            self._level_phase, self._level_window_start = "window", self._progress_done
            untrusted = self._run_targets(executor, ctx, window, self.opt.nbr_iteration_count,
                                          f"{label}, neighbour window", full_scan=False)
            self._level_phase = "follow"
            full_keys = []
            self._level_follow_cost = self._follow_up_cost(ctx, untrusted, self.opt.max_window_shifts, full_keys)
            self._update_progress_total(ctx)
            for move_idx in range(self.opt.max_window_shifts):
                moves_left = self.opt.max_window_shifts - move_idx - 1
                full_keys.extend(t.key for t, reason in untrusted if reason != "edge")
                moved = [self._centered_target(ctx, t.key, float(t.best.d.to("µm").magnitude))
                         for t, reason in untrusted if reason == "edge"]
                # the moved windows themselves are in the pending tasks of their scan
                self._level_follow_cost = (self._follow_up_cost(ctx, [], moves_left, full_keys)
                                           + len(moved) * self._move_fail_rate() * self._retry_cost(ctx, moves_left))
                untrusted = self._run_targets(executor, ctx, moved, self.opt.nbr_iteration_count,
                                              f"{label}, window moved to the edge optimum ({move_idx + 1})",
                                              full_scan=False)
                self._move_stats = [self._move_stats[0] + len(untrusted), self._move_stats[1] + len(moved)]
                self._level_follow_cost = self._follow_up_cost(ctx, untrusted, moves_left, full_keys)
                self._update_progress_total(ctx)
            full_keys.extend(t.key for t, _ in untrusted)
            if full_keys:
                logging.info(f"{label}: no trusted window optimum for {len(full_keys)} / {len(window)} pixel(s)")
            self._level_follow_cost = 0.0
            self._run_full_scans(executor, ctx, full_keys, f"{label}, full scan (untrusted window result)")
            self._win_nominal += len(window) * window_cost
            self._win_actual += self._progress_done - self._level_window_start
            self._level_phase = None

    def _outlier_passes(self, executor, ctx: ImageContext, keys=None):
        # keys: only these pixels are checked (default all), ctx must hold their data
        threshold = self.opt.outlier_threshold.to("µm").magnitude
        self._outlier_estimate = 0.0
        for pass_idx in range(self.opt.outlier_passes):
            targets = []
            for key, res in list(self.pixel_results.items()):
                if keys is not None and key not in keys:
                    continue
                nbrs = self._neighbors(key, 1)
                if len(nbrs) < 3:
                    continue
                d_med, limit = self._deviation_limit(np.array([nbr.d for nbr in nbrs]))
                if abs(res.d - d_med) <= limit:
                    continue
                consistent = [nbr for nbr in nbrs if abs(nbr.d - d_med) <= limit] or nbrs
                target = self._window_target(ctx, key, consistent)
                if target is not None:
                    targets.append(target)

            logging.info(f"Outlier pass {pass_idx + 1} / {self.opt.outlier_passes}: {len(targets)} pixel(s) deviate "
                         f"more than max({threshold:g} µm, 3x the neighbour scatter) from the median of their "
                         f"neighbours")
            if not targets:
                break
            untrusted = self._run_targets(executor, ctx, targets, self.opt.nbr_iteration_count,
                                          f"Outlier pass {pass_idx + 1}", full_scan=False, store_untrusted=False)
            logging.info(f"Outlier pass {pass_idx + 1}: replaced {len(targets) - len(untrusted)} / {len(targets)}, "
                         f"kept the previous result where the window optimum is not trusted")

    def _check_reevaluation(self):
        if not self._run_info:
            raise ValueError("Evaluate the image (or load image results) first")
        if not np.array_equal(self.qs.freq_range_idx, self._freq_idx):
            raise ValueError("The fit range changed since the image evaluation, evaluate the image again")

    def _reevaluate_outliers_impl(self):
        qs = self.qs
        self._check_reevaluation()
        qs.reset_state()
        self._reset_counters()
        qs.progress_start(0)
        ctx = self._build_context(self._pixel_meas, self._run_info)
        with qs.process_pool() as executor:
            self._outlier_passes(executor, ctx)
        qs.progress_set_total(self._progress_done)

    def _reevaluate_area_impl(self):
        # the pixels of the area are evaluated again with the current settings (echo and knot count of the image
        # evaluation), pixels at the area border get window scans around their neighbours outside of the area
        self._check_reevaluation()
        pixels = self.pixel_measurements(confine=True, binning=self._run_binning())
        if not pixels:
            raise ValueError("No sample measurements in the area")
        for key in pixels:
            self.pixel_results.pop(key, None)
        self._provisional.difference_update(pixels)
        self._evaluate_pixels(pixels, "Area re-evaluation")

    def _continue_impl(self):
        # pixels of the image (or the area, if confined) without a final result, e.g. after a cancelled evaluation
        self._check_reevaluation()
        pixels = {key: meas_list for key, meas_list in
                  self.pixel_measurements(binning=self._run_binning()).items()
                  if (key not in self.pixel_results or key in self._provisional) and key not in self._skipped_keys}
        if not pixels:
            logging.info("All pixels are evaluated")
            return
        provisional_cnt = len(self._provisional.intersection(pixels))
        for key in pixels:
            self.pixel_results.pop(key, None)
        self._provisional.difference_update(pixels)
        logging.info(f"Continuing the image evaluation: {len(pixels)} pixel(s) without a result "
                     f"({provisional_cnt} with a provisional result whose follow-up scan was cancelled)")
        self._evaluate_pixels(pixels, "Continued evaluation")

    def _evaluate_pixels(self, pixels, label):
        qs = self.qs
        for key, meas_list in pixels.items():
            self._pixel_meas[key] = meas_list
            self._meas_pixel.update((meas, key) for meas in meas_list)
        qs.reset_state()
        self._reset_counters()
        ctx = self._build_context(pixels, self._run_info)
        if self._run_info.get("d_axis") is not None:
            ctx.d_axis = self._run_info["d_axis"]
        sample_keys = self._sample_keys(ctx, pixels)
        levels = self._levels(sample_keys)
        qs.progress_start(0)
        self._plan_progress(ctx, levels, len(sample_keys))
        with qs.process_pool() as executor:
            self._scan_levels(executor, ctx, levels)
            self._outlier_passes(executor, ctx, keys=set(sample_keys))
        qs.progress_set_total(self._progress_done)
        logging.info(f"{label} finished, {self._task_cnt} tasks")

    def _reevaluate_pixel_impl(self, key):
        # full scan of the configured thickness range
        qs = self.qs
        self._check_reevaluation()
        qs.reset_state()
        self._reset_counters()
        ctx = self._build_context({key: self._pixel_meas[key]}, self._run_info)
        qs.progress_start(0)
        with qs.process_pool() as executor:
            self._run_targets(executor, ctx, [self._target(ctx, key)], self._full_iterations(),
                              f"Pixel {self._pixel_label(key)}", full_scan=True)
        qs.progress_set_total(self._progress_done)
        res = self.pixel_results[key]
        logging.info(f"Pixel {self._pixel_label(key)} re-evaluated: d {res.d:.2f} µm, q {res.q_val:.3g}")

    def _uncertainty_impl(self, keys, label):
        # Monte Carlo thickness uncertainty of the pixels around their stored optimum, with the echo and knot count of
        # the image evaluation. In chunks of pixels, so the results of the replicates don't pile up
        qs = self.qs
        self._check_reevaluation()
        opt = self.settings.eval_opt.uncertainty
        estimator = estimator_for(self.dataset)
        qs.reset_state()
        self._reset_counters()
        qs.progress_start(len(keys) * opt.qspace_samples * opt.qspace_points)
        chunk = max(1, 200 // opt.qspace_samples)
        with qs.process_pool() as executor:
            for i in range(0, len(keys), chunk):
                chunk_keys = keys[i:i + chunk]
                ctx = self._build_context({key: self._pixel_meas[key] for key in chunk_keys}, self._run_info)
                specs, uncs = [], {}
                for key in chunk_keys:
                    meas_list = self._pixel_meas[key]
                    refs = self.measurement_selector.get_matching_refs(meas_list)
                    uncs[key] = estimator.transmission_uncertainty(meas_list, refs, qs.freq_range_idx, opt)
                    res = self.pixel_results[key]
                    specs.append((key, self._target(ctx, key).opt_config, ctx.t_exp[key], uncs[key], res.d,
                                  res.shift))
                for key, unc_res in qs.mc_thickness_uncertainty(executor, specs, label, add_progress=False).items():
                    self.pixel_results[key].d_uncert = unc_res.d_std
                    if len(keys) == 1 or unc_res.at_edge:
                        qs.log_thickness_uncertainty(f"Pixel {self._pixel_label(key)}", self.pixel_results[key].d,
                                                     unc_res, uncs[key])
                self._autosave()
        self._update_uncertainty_stats()

    def _update_uncertainty_stats(self):
        d_uncert = np.array([res.d_uncert for res in self.pixel_results.values()])
        d_uncert = d_uncert[np.isfinite(d_uncert)]
        if not d_uncert.size:
            self.set_trait("uncertainty_stats", "")
            return
        p5, p95 = np.percentile(d_uncert, [5, 95])
        text = (f"{d_uncert.size} pixels: median {np.median(d_uncert):.2f} µm, 5-95 % {p5:.2f} - {p95:.2f} µm")
        self.set_trait("uncertainty_stats", text)
        logging.info(f"Q-space thickness uncertainty of {text}")

    def _reset_counters(self):
        self._q_ref, self._task_cnt = {}, 0
        self._progress_done = 0  # tasks the progress has been advanced by
        self._stage_end = 0  # progress_done at the end of the running scan
        self._plan_levels, self._next_level_idx = [], 0
        self._level_phase = None  # "before" / "window" / "follow" its window scan, None outside of a level
        self._level_window_cnt, self._level_window_start, self._level_follow_cost = 0, 0, 0.0
        self._win_nominal, self._win_actual = 0, 0  # window pixels of the finished levels
        self._outlier_estimate = 0.0
        self._move_stats = [0, 0]  # untrusted / scanned moved windows
        self._last_autosave = time.monotonic()

    def _start(self, func):
        qs = self.qs
        if qs._active_eval is not None:
            logging.warning("Q-space evaluation is already running")
            return
        qs._cancel_event = threading.Event()
        qs._active_eval = self

        def bg_worker():
            state = "Failed"
            try:
                func()
                state = "Finished"
            except OptimizationCancelled:
                state = "Cancelled"
                logging.info("Image evaluation cancelled")
            except ValueError as e:
                logging.error(f"Image evaluation: {e}")
            except Exception:
                traceback.print_exc()
            finally:
                self._autosave(force=True)
                qs.progress_stop(state)
                if qs._active_eval is self:
                    qs._active_eval = None
                self._signals.finished.emit()

        if qs._thread_executor is None:
            qs._thread_executor = ThreadPoolExecutor(max_workers=1)
        qs._thread_executor.submit(bg_worker)

    def _on_finished(self):
        if self.pixel_results:
            full_cnt = sum(res.full_scan for res in self.pixel_results.values())
            median_width = self.median_width()
            reliable_cnt = sum(self.pixel_confidence(key, median_width).reliable for key in self.pixel_results)
            self.set_trait("evaluated_pixels", f"{len(self.pixel_results)} / {len(self._pixel_meas)} "
                                               f"({full_cnt} full scans, {len(self._skipped_keys)} skipped, "
                                               f"{reliable_cnt} reliable)")
        replot_key, self._replot_key = self._replot_key, None
        if replot_key in self.pixel_results:
            self.plot_pixel(replot_key)
        if not (self.plot_when_done and self.image_plot is not None and self.pixel_results):
            return
        if self.image_plot.selected_quantity_enum not in q_space_quantities:
            self.image_plot.selected_quantity_enum = QuantityEnum.QSpaceThickness
        self.image_plot.plot_image()

    def _pixel_label(self, key):
        x, y = self._pixel_pos(key)
        return f"({x:.2f}, {y:.2f}) mm"

    def _on_image_click(self, x, y, meas_list):
        # click handler of the image plot, True if the click was handled here
        if not self.opt.inspect_on_click or self.image_plot.selected_quantity_enum not in q_space_quantities:
            return False
        key = next((self._meas_pixel[meas] for meas in meas_list if meas in self._meas_pixel), None)
        if key not in self.pixel_results:
            return False
        self.plot_pixel(key)
        return True

    # three panels, the app's font sizes are made for single plots
    @mpl.rc_context(mpl_style_params({"font.size": 12, "legend.fontsize": 9, "lines.linewidth": 1.5,
                                      "lines.markersize": 4}))
    def plot_pixel(self, key):
        res = self.pixel_results[key]
        conf = self.pixel_confidence(key)
        nbrs = self._neighbors(key, 1)
        self._inspected_key = key
        d_text = f"{res.d:.2f} ± {res.d_uncert:.2f} µm" if np.isfinite(res.d_uncert) else f"{res.d:.2f} µm"
        self.set_trait("inspected_pixel", f"{self._pixel_label(key)}, d {d_text}, "
                                          + ("reliable" if conf.reliable else "not reliable"))

        fig = plt.figure("Q-space pixel", figsize=(14, 4.5))
        fig.clf()
        ax_q, ax_n, ax_a = fig.subplots(1, 3)
        fig.suptitle(f"Pixel {self._pixel_label(key)}, {'full' if res.full_scan else 'window'} scan, "
                     f"{'reliable' if conf.reliable else 'not reliable'}")

        if nbrs and np.isfinite(conf.d_limit):
            d_med = res.d - conf.d_deviation
            ax_q.axvspan(d_med - conf.d_limit, d_med + conf.d_limit, color="0.85", label="Allowed deviation")
        for i, nbr in enumerate(nbrs):
            ax_q.axvline(nbr.d, color="0.5", lw=0.8, ls="--", label="Neighbours" if i == 0 else None)
        ax_q.semilogy(res.profile_d, res.profile_q, "o-", ms=3, color="C0", label="Scanned")
        fit = min_parabola(res.profile_d, res.profile_q, res.d, side_cnt=2)
        if fit is not None and np.isfinite(conf.min_width):
            x, center, coeffs = fit
            xx = np.linspace(x[0], x[-1], 50)
            ax_q.semilogy(xx, np.polyval(coeffs, xx - center), color="C1", lw=1, label="Parabola")
        if np.isfinite(res.d_uncert):
            ax_q.axvspan(res.d - res.d_uncert, res.d + res.d_uncert, color="C3", alpha=0.15, lw=0,
                         label=f"Uncertainty ± {res.d_uncert:.2f} µm")
        ax_q.axvline(res.d, color="C3", label=f"Optimum {res.d:.2f} µm")
        ax_q.set_xlabel("Thickness (µm)")
        ax_q.set_ylabel("q-value")
        ax_q.set_title(f"q {res.q_val:.3g}, q / neighbours {conf.q_ratio:.2f}, d - neighbours "
                       f"{conf.d_deviation:+.1f} µm\nminimum width {conf.min_width:.1f} µm "
                       f"({conf.width_ratio:.2f} x median)")
        ax_q.legend()

        freq = self._freq_axis
        for i, nbr in enumerate(nbrs):
            label = "Neighbours" if i == 0 else None
            ax_n.plot(freq, nbr.n.real, color="0.7", lw=0.7, label=label)
            ax_a.plot(freq, alpha_of(freq, nbr.n), color="0.7", lw=0.7, label=label)
        ax_n.plot(freq, res.n.real, color="C0", label="Pixel")
        ax_a.plot(freq, alpha_of(freq, res.n), color="C0", label="Pixel")
        a_max = self._pixel_alpha_max(key, res)
        if a_max is not None:
            # the limit is far above alpha where the dynamic range is large, it shouldn't squeeze alpha
            top = 1.5 * np.nanmax(np.r_[alpha_of(freq, res.n).real, [nbr_a for nbr in nbrs
                                                                    for nbr_a in alpha_of(freq, nbr.n).real]])
            ax_a.plot(freq, a_max, color="C3", ls="--", label=f"Maximum measurable (min. {np.nanmin(a_max):.0f} 1/cm)")
            if np.isfinite(top) and top > ax_a.get_ylim()[0]:
                ax_a.set_ylim(top=top)
        ax_n.set_ylabel("Refractive index")
        ax_a.set_ylabel("Absorption coefficient (1/cm)")
        for ax in (ax_n, ax_a):
            ax.set_xlabel("Frequency (THz)")
            ax.legend()
        fig.tight_layout()
        plt.show()

    def _pixel_alpha_max(self, key, res):
        # alpha_max of the pixel (dynamic range of its reference spectrum, its measurements averaged), None if off
        if not self.settings.eval_opt.dynamic_range.enabled or self._freq_idx is None:
            return None
        meas_list = self._pixel_meas.get(key)
        if not meas_list:
            return None
        try:
            refs = self.dataset.measurement_selector.get_matching_refs(meas_list)
            ref_fd = self.dataset.get_multi_data(refs, Domain.Frequency)[:, self._freq_idx, 1]
            noise = noise_floor(self.dataset, len(meas_list))
        except Exception as e:
            logging.debug(f"No maximum measurable absorption for the pixel: {e}")
            return None
        return alpha_max(self._freq_axis, np.mean(np.abs(ref_fd), axis=0), noise, res.n.real, res.d)

    def _results_arrays(self):
        # everything of the image results as arrays for np.savez, including what loading them needs to continue or
        # re-evaluate (q(d) profiles nan padded, skipped pixels, echo / knot count and thickness range of the run)
        keys = sorted(self.pixel_results)
        res_list = [self.pixel_results[key] for key in keys]
        median_width = self.median_width()
        conf_list = [self.pixel_confidence(key, median_width) for key in keys]
        pos = np.array([self._pixel_pos(key) for key in keys]).reshape(-1, 2)
        prof_len = max((len(res.profile_d) for res in res_list), default=0)
        profile_d, profile_q = np.full((len(keys), prof_len), np.nan), np.full((len(keys), prof_len), np.nan)
        for i, res in enumerate(res_list):
            profile_d[i, :len(res.profile_d)] = res.profile_d
            profile_q[i, :len(res.profile_q)] = res.profile_q
        info = self._run_info
        settings = {"QSpaceEval": component_settings(self.qs), "AppSettings": component_settings(self.settings)}

        return dict(x=pos[:, 0], y=pos[:, 1],
                    ix=np.array([key[0] for key in keys], dtype=int),
                    iy=np.array([key[1] for key in keys], dtype=int),
                    d=np.array([res.d for res in res_list]),
                    shift=np.array([res.shift for res in res_list]),
                    q_val=np.array([res.q_val for res in res_list]),
                    full_scan=np.array([res.full_scan for res in res_list]),
                    reliable=np.array([conf.reliable for conf in conf_list]),
                    min_width=np.array([conf.min_width for conf in conf_list]),
                    width_ratio=np.array([conf.width_ratio for conf in conf_list]),
                    q_ratio=np.array([conf.q_ratio for conf in conf_list]),
                    d_deviation=np.array([conf.d_deviation for conf in conf_list]),
                    d_uncert=np.array([res.d_uncert for res in res_list]),
                    freq=self._freq_axis,
                    n=np.array([res.n for res in res_list]),
                    profile_d=profile_d, profile_q=profile_q,
                    pixels=np.array(sorted(self._pixel_meas), dtype=int).reshape(-1, 2),
                    skipped=np.array(sorted(self._skipped_keys), dtype=int).reshape(-1, 2),
                    masked=np.array(sorted(self._masked_keys), dtype=int).reshape(-1, 2),
                    provisional=np.array(sorted(self._provisional), dtype=int).reshape(-1, 2),
                    measurements=np.array(["; ".join(m.filepath.name for m in self._pixel_meas[key]) for key in keys]),
                    model=self.qs.transmission_model.name,
                    binning=self._run_binning(),
                    nfp=-1 if info.get("nfp") is None else int(info["nfp"]),
                    knot_count=-1 if info.get("knot_count") is None else int(info["knot_count"]),
                    run_d_axis=np.asarray(info.get("d_axis", np.empty(0)), dtype=float),
                    run_d_step=float(info.get("d_step", np.nan)),
                    settings=json.dumps(settings, default=str),
                    code_version=code_version(),
                    created=datetime.now().isoformat(timespec="seconds"))

    def _autosave(self, force=False):
        interval = self.opt.autosave_interval.to("s").magnitude
        if interval <= 0 or not self.pixel_results:
            return
        if not force and time.monotonic() - self._last_autosave < interval:
            return
        self._last_autosave = time.monotonic()
        path = Path(self.qs.result_saver.base_path) / "q_space_image_autosave.npz"
        tmp_path = path.with_name("q_space_image_autosave.tmp.npz")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(tmp_path, **self._results_arrays())
            # a crash while writing doesn't destroy the previous autosave
            os.replace(tmp_path, path)
        except OSError as e:
            logging.warning(f"Autosave of the image results failed: {e}")

    def _load_results(self, path):
        qs = self.qs
        with np.load(path) as npz:
            if "profile_d" not in npz.files:
                raise ValueError("saved by an older version without the scan data, it can't be loaded")
            data = {k: npz[k] for k in npz.files}
        freq = data["freq"]
        if len(freq) != len(qs.freq_axis) or not np.allclose(freq, qs.freq_axis):
            raise ValueError(f"its fit range ({freq[0]:.3f} - {freq[-1]:.3f} THz, {len(freq)} points) differs from "
                             f"the current fit range")
        binning = int(data["binning"])
        run_keys = {tuple(key) for key in data["pixels"].tolist()}
        pixels = {key: meas_list for key, meas_list in self.pixel_measurements(confine=False, binning=binning).items()
                  if key in run_keys}
        results, missing = {}, 0
        for i, key in enumerate(zip(data["ix"].tolist(), data["iy"].tolist())):
            if key not in pixels:
                missing += 1
                continue
            valid = np.isfinite(data["profile_d"][i])
            results[key] = PixelResult(d=float(data["d"][i]), shift=float(data["shift"][i]),
                                       q_val=float(data["q_val"][i]), n=data["n"][i],
                                       full_scan=bool(data["full_scan"][i]),
                                       profile_d=data["profile_d"][i][valid], profile_q=data["profile_q"][i][valid],
                                       min_width=float(data["min_width"][i]),
                                       d_uncert=float(data["d_uncert"][i]) if "d_uncert" in data else np.nan)
        if missing:
            logging.warning(f"{missing} pixel(s) of {path.name} are not part of the image grid of the data set")

        self.pixel_results = results
        self._pixel_meas = pixels
        self._meas_pixel = {meas: key for key, meas_list in pixels.items() for meas in meas_list}
        self._freq_idx, self._freq_axis = qs.freq_range_idx, qs.freq_axis
        self._skipped_keys = {tuple(key) for key in data["skipped"].tolist()}
        self._masked_keys = {tuple(key) for key in data["masked"].tolist()}
        self._provisional = {tuple(key) for key in data["provisional"].tolist()} if "provisional" in data else set()
        self._mask_vals = {}
        nfp, knot_count = int(data["nfp"]), int(data["knot_count"])
        self._run_info = {"nfp": None if nfp < 0 else nfp, "knot_count": None if knot_count < 0 else knot_count,
                          "d_axis": data["run_d_axis"], "d_step": float(data["run_d_step"]), "binning": binning}
        self._inspected_key = None
        self.set_trait("inspected_pixel", "")
        self._update_uncertainty_stats()
        logging.info(f"Loaded {len(results)} pixel results from {path} (created {data['created']}, code version "
                     f"{data['code_version'] or 'unknown'})")

    @action("Evaluate image", group=run_grp, priority=15)
    def evaluate_image(self):
        self._start(self._evaluate_image_impl)

    @action("Continue evaluation", group=run_grp, priority=16)
    def continue_evaluation(self):
        # evaluates the pixels without a result, with the echo / knot count and thickness range of the run
        if not self._pixel_meas:
            logging.warning("Evaluate the image or load image results first")
            return
        self._start(self._continue_impl)

    @action("Re-evaluate outliers", group=reeval_grp, priority=25)
    def reevaluate_outliers(self):
        if not self.pixel_results:
            logging.warning("Evaluate the image first")
            return
        self._start(self._reevaluate_outliers_impl)

    @action("Re-evaluate area", group=reeval_grp, priority=26)
    def reevaluate_area(self):
        # the area of the "Evaluated area" group, also if the image evaluation isn't confined to it
        if not self._pixel_meas:
            logging.warning("Evaluate the image first")
            return
        self._start(self._reevaluate_area_impl)

    @action("Re-evaluate inspected pixel", group=reeval_grp, priority=27)
    def reevaluate_inspected_pixel(self):
        if self._inspected_key not in self.pixel_results:
            logging.warning("Click a pixel in the image plot of a Q-space quantity first")
            return
        self._replot_key = self._inspected_key
        self._start(partial(self._reevaluate_pixel_impl, self._inspected_key))

    @action("Cancel", group=run_grp, priority=17)
    def cancel(self):
        self.qs.cancel_evaluation()

    @action("Thickness statistics", group=results_grp, priority=35)
    def plot_thickness_statistics(self):
        if not self.pixel_results:
            logging.warning("No image results")
            return
        keys = sorted(self.pixel_results)
        median_width = self.median_width()
        d = np.array([self.pixel_results[key].d for key in keys])
        reliable = np.array([self.pixel_confidence(key, median_width).reliable for key in keys])

        def stats(v):
            p5, p95 = np.percentile(v, [5, 95])
            return (f"mean {np.mean(v):.2f}, median {np.median(v):.2f}, std {np.std(v):.2f}, "
                    f"5-95 % {p5:.1f} - {p95:.1f} µm")

        text = f"{reliable.sum()} reliable pixels: {stats(d[reliable])}" if reliable.any() else "No reliable pixels"
        self.set_trait("thickness_stats", text)
        logging.info(f"Q-space thickness of all {len(d)} pixels: {stats(d)}. {text}")

        fig = plt.figure("Q-space thickness statistics")
        fig.clf()
        ax = fig.subplots()
        bins = np.histogram_bin_edges(d, bins=int(np.clip(np.sqrt(len(d)), 10, 100)))
        ax.hist(d, bins=bins, color="0.75", label=f"All ({len(d)})")
        ax.hist(d[reliable], bins=bins, color="C0", label=f"Reliable ({reliable.sum()})")
        median = np.median(d[reliable] if reliable.any() else d)
        ax.axvline(median, color="C3", label=f"Median {median:.2f} µm" + (" (reliable)" if reliable.any() else ""))
        ax.set_xlabel("Thickness (µm)")
        ax.set_ylabel("Pixels")
        ax.legend()
        fig.tight_layout()
        plt.show()

    @action("Thickness uncertainty of inspected pixel", group=uncertainty_grp, priority=45)
    def uncertainty_inspected_pixel(self):
        if self._inspected_key not in self.pixel_results:
            logging.warning("Click a pixel in the image plot of a Q-space quantity first")
            return
        key = self._inspected_key
        self._replot_key = key
        self._start(partial(self._uncertainty_impl, [key], f"Pixel {self._pixel_label(key)}"))

    @action("Thickness uncertainty of area", group=uncertainty_grp, priority=46,
            help="All evaluated pixels inside the area of the \"Evaluated area\" group (also if the evaluation "
                 "isn't confined to it). Costs Monte Carlo samples x thicknesses per window fits per pixel")
    def uncertainty_area(self):
        if not self.pixel_results:
            logging.warning("Evaluate the image first")
            return
        area = self.pixel_measurements(confine=True, binning=self._run_binning())
        keys = sorted(key for key in area if key in self.pixel_results)
        if not keys:
            logging.warning("No evaluated pixels in the area")
            return
        self._start(partial(self._uncertainty_impl, keys, "Area"))

    @action("Save image results", group=results_grp, priority=36)
    def save_results(self):
        if not self.pixel_results:
            logging.warning("No image results")
            return
        path = Path(self.qs.result_saver.base_path) / f"{datetime.now():%Y-%m-%d_%H-%M-%S}-q_space_image.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self._results_arrays())
        logging.info(f"Saved the q-space image results to {path}")

    @action("Load image results", group=results_grp, priority=37)
    def load_results(self):
        if self.qs._active_eval is not None:
            logging.warning("Wait until the running evaluation has finished")
            return
        path = self.results_file
        if not path.is_file():
            files = sorted(Path(self.qs.result_saver.base_path).glob("*q_space_image*.npz"),
                           key=lambda f: f.stat().st_mtime)
            if not files:
                logging.warning("No results file selected and no q-space image results in the result directory")
                return
            path = files[-1]
        try:
            self._load_results(path)
        except (ValueError, KeyError, OSError) as e:
            logging.error(f"Could not load {path}: {e}")
            return
        self._on_finished()

    @action("Clear image results", group=results_grp, priority=38)
    def clear_results(self):
        self.pixel_results = {}
        self._run_info = {}
        self._skipped_keys, self._masked_keys, self._provisional = set(), set(), set()
        self._inspected_key = None
        self.set_trait("evaluated_pixels", "")
        self.set_trait("inspected_pixel", "")
        self.set_trait("thickness_stats", "")
        self.set_trait("uncertainty_stats", "")
