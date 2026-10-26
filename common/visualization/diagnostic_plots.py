"""
Diagnostic plots of the selected measurements: dynamic range and the usable absorption range, transmission with its
uncertainty, Fabry-Perot echo visibility, Kramers-Kronig consistency of n and k, and a scatter plot of two quantities
over all measurements.
"""
import logging
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.signal import hilbert
from scipy.stats import pearsonr
from traitlets import Bool, Enum as TEnum, Int

from common.components import ComponentBase, action
from common.consts import c_thz
from common.default_appsettings import ComplexPart, Domain, QuantityEnum
from common.eval_component.dynamic_range import alpha_max, alpha_min, noise_floor, transmission_noise
from common.eval_component.q_value import echo_visibility, fp_round_trip_time
from mpl_settings import mpl_style_params

action = partial(action, check_init=True, rc_params=mpl_style_params)

WATER_LINES_THZ = (0.557, 0.752, 0.988, 1.097, 1.113, 1.163, 1.208, 1.229, 1.411, 1.602, 1.661, 1.670, 1.717,
                   1.762, 1.797, 1.867, 1.919, 2.041, 2.074, 2.164, 2.196, 2.222, 2.264, 2.344, 2.365, 2.391, 2.640,
                   2.685, 2.774)


def mark_water_lines(fig):
    # on the axes over frequency (x label "Frequency ...", or sharing x with such an axis), only inside the data range
    # so the x limits don't change
    axes = fig.get_axes()
    freq_axes = [ax for ax in axes if ax.get_xlabel().startswith("Frequency")]
    for ax in axes:
        if not any(ax is f_ax or ax.get_shared_x_axes().joined(ax, f_ax) for f_ax in freq_axes):
            continue
        done = getattr(ax, "_water_lines", set())
        lo, hi = sorted(ax.dataLim.intervalx)
        for f in WATER_LINES_THZ:
            if lo <= f <= hi and f not in done:
                ax.axvline(f, color="tab:cyan", alpha=0.35, lw=0.8, zorder=0, label="_water")
                done.add(f)
        ax._water_lines = done


def kramers_kronig_n(freq, k, n_anchor, anchor_idx):
    # singly subtractive Kramers-Kronig over the measured band: n(w) = n(w_a) + 2 / pi (I(w) - I(w_a)) with
    # I(w) = P int w' k(w') / (w'^2 - w^2) dw', the principal value by Maclaurin's rule (only the points an odd number
    # of steps away, twice the step)
    w = 2 * np.pi * np.asarray(freq, dtype=float)
    dw = float(np.mean(np.diff(w)))
    offset = np.arange(len(w))[None, :] - np.arange(len(w))[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(offset % 2 == 1, w[None, :] * k[None, :] / (w[None, :] ** 2 - w[:, None] ** 2), 0.0)
    integral = 2 * dw * terms.sum(axis=1)
    return n_anchor + 2 / np.pi * (integral - integral[anchor_idx])


class DiagnosticPlots(ComponentBase):
    diag_grp = "Diagnostic plots"
    marked_echoes = Int(4, min=1, group=diag_grp, help="Number of Fabry-Perot echo positions marked in the time "
                                                       "trace of the echo diagnostic").tag(
        name="Marked echoes", priority=0)

    scatter_grp = "Scatter plot"
    scatter_x = TEnum(QuantityEnum, QuantityEnum.TimeOfFlight, group=scatter_grp).tag(name="x quantity", priority=10)
    scatter_y = TEnum(QuantityEnum, QuantityEnum.P2P, group=scatter_grp).tag(name="y quantity", priority=11)
    scatter_complex_part = TEnum(ComplexPart, ComplexPart.abs, group=scatter_grp).tag(
        name="Complex value representation", priority=12)
    scatter_all = Bool(True, group=scatter_grp, help="All sample measurements, otherwise the selected ones. "
                                                     "Frequency dependent quantities at the selected frequency").tag(
        name="All sample measurements", priority=13)

    def __init__(self, plotter, **kwargs):
        super().__init__(**kwargs)
        self.plotter = plotter

    @property
    def dataset(self):
        return self.plotter.dataset

    def _selection(self):
        selection = self.plotter.measurement_selector.get_selection()
        if selection is None or not selection.sams:
            logging.warning("No sample measurement selected")
            return None, None
        sams = list(selection.sams)
        return sams, self.dataset.measurement_selector.get_matching_refs(sams)

    def _thickness(self):
        d = self.plotter.d_sample.to("µm").magnitude
        if d <= 1:
            logging.warning(f"Nominal sample thickness {d:g} µm (DataSet plotter): set the actual thickness")
            return None
        return d

    def _dynamic_range(self, sams, refs):
        # dB over the noise floor on the frequency range of the plotter (sample: mean of the selection, reference: of
        # its distinct references), and the mask of the contiguous band around the sample peak at least 10 dB above it
        ds, idx = self.dataset, self.plotter.freq_range_idx
        unique_refs = list(dict.fromkeys(refs))
        sam_amp = np.mean(np.abs(ds.get_multi_data(sams, Domain.Frequency)[:, idx, 1]), axis=0)
        ref_amp = np.mean(np.abs(ds.get_multi_data(unique_refs, Domain.Frequency)[:, idx, 1]), axis=0)
        noise_sam, noise_ref = noise_floor(ds, len(sams)), noise_floor(ds, len(unique_refs))
        dr_sam, dr_ref = 20 * np.log10(sam_amp / noise_sam), 20 * np.log10(ref_amp / noise_ref)
        band = np.zeros(len(dr_sam), dtype=bool)
        above = dr_sam >= 10
        if above.any():
            lo = hi = int(np.argmax(dr_sam))
            while lo > 0 and above[lo - 1]:
                lo -= 1
            while hi < len(band) - 1 and above[hi + 1]:
                hi += 1
            band[lo:hi + 1] = True
        return dr_sam, dr_ref, band, ref_amp, noise_sam

    def _usable_band(self, sams, refs, min_points=16):
        band = self._dynamic_range(sams, refs)[2]
        if band.sum() < min_points:
            logging.warning("The sample spectrum is less than 10 dB above the noise floor almost everywhere")
            return None
        freq = self.plotter.freq_axis[band]
        logging.info(f"Using {freq[0]:.2f} - {freq[-1]:.2f} THz, where the sample spectrum is at least 10 dB above the "
                     f"noise floor")
        return band

    def _show(self):
        if self.plotter.settings.plot_opt.auto_show:
            self.plotter.plt_show()

    @action("Dynamic range", group=diag_grp, priority=1)
    def plot_dynamic_range(self):
        sams, refs = self._selection()
        if sams is None:
            return
        ds, idx = self.dataset, self.plotter.freq_range_idx
        freq = self.plotter.freq_axis
        dr_sam, dr_ref, band, ref_amp, noise_sam = self._dynamic_range(sams, refs)

        d = self._thickness()
        fig = plt.figure("Dynamic range", figsize=(10, 9 if d else 5))
        fig.clf()
        axes = fig.subplots(2 if d else 1, 1, sharex=True, squeeze=False)[:, 0]
        ax = axes[0]
        ax.plot(freq, dr_ref, label="Reference")
        ax.plot(freq, dr_sam, label=f"Sample ({len(sams)} averaged)" if len(sams) > 1 else "Sample")
        ax.axhline(0, color="0.3", lw=1, label="Noise floor")
        ax.axhline(10, color="0.3", lw=1, ls="--", label="10 dB")
        ax.set_ylabel("Dynamic range (dB)")
        ax.legend(fontsize=10)

        if band.any():
            i_peak = int(np.argmax(dr_sam))
            f_band = freq[band]
            ax.axvspan(f_band[0], f_band[-1], color="tab:green", alpha=0.08, lw=0)
            logging.info(f"Dynamic range: the sample spectrum is at least 10 dB above the noise floor from "
                         f"{f_band[0]:.2f} to {f_band[-1]:.2f} THz (peak {dr_sam[i_peak]:.0f} dB at "
                         f"{freq[i_peak]:.2f} THz), reference peak {np.max(dr_ref):.0f} dB")
        else:
            logging.warning("Dynamic range: the sample spectrum is nowhere 10 dB above the noise floor")

        if d:
            res = ds.windowing_eval(sams, d=d)
            n_real = np.mean(res.refr_idx.real[:, idx], axis=0)
            alpha = np.mean(res.abs_coe[:, idx], axis=0)
            a_max = alpha_max(freq, ref_amp, noise_sam, n_real, d)
            t = np.mean(ds.transmission(sams, refs)[:, idx], axis=0)
            a_min = alpha_min(t, transmission_noise(ds, sams, refs, idx), d,
                              self.plotter.settings.eval_opt.dynamic_range.alpha_min_coverage)
            ax = axes[1]
            ax.plot(freq, alpha, color="C0", label="Absorption coefficient (windowed)")
            ax.plot(freq, a_max, color="C3", ls="--", label="Maximum measurable")
            ax.fill_between(freq, 0, a_min, color="C2", alpha=0.25, lw=0, label="Below minimum measurable")
            shown = band if band.any() else np.ones(len(freq), dtype=bool)
            top = np.nanmax(np.r_[alpha[shown], a_min[shown], a_max[shown]])
            if np.isfinite(top) and top > 0:
                ax.set_ylim(min(0.0, np.nanmin(alpha[shown])), 1.2 * top)
            ax.set_ylabel("Absorption coefficient (1/cm)")
            ax.set_title(f"d = {d:g} µm", fontsize=12)
            ax.legend(fontsize=10)
        axes[-1].set_xlabel("Frequency (THz)")
        fig.tight_layout()
        self._show()

    @action("Transmission with uncertainty", group=diag_grp, priority=2)
    def plot_transmission_uncertainty(self):
        sams, refs = self._selection()
        if sams is None:
            return
        ds, idx, freq = self.dataset, self.plotter.freq_range_idx, self.plotter.freq_axis
        t = np.mean(ds.transmission(sams, refs)[:, idx], axis=0)
        unc = transmission_noise(ds, sams, refs, idx)
        cov = self.plotter.settings.eval_opt.uncertainty.coverage_factor
        t_abs = np.abs(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            sig_abs = cov * np.sqrt(unc.variance / 2 + (unc.amplitude * t_abs) ** 2)
            sig_phi = cov * np.sqrt(unc.variance / (2 * t_abs ** 2) + (2 * np.pi * freq * unc.timing) ** 2)
        phi = np.unwrap(np.angle(t))

        fig = plt.figure("Transmission with uncertainty", figsize=(10, 8))
        fig.clf()
        ax_a, ax_p = fig.subplots(2, 1, sharex=True)
        label = f"Mean of {len(sams)}" if len(sams) > 1 else "Transmission"
        ax_a.plot(freq, t_abs, color="C0", label=label)
        ax_a.fill_between(freq, t_abs - sig_abs, t_abs + sig_abs, color="C0", alpha=0.25, lw=0,
                          label=f"± {cov:g} sigma")
        ax_a.set_ylabel("|T|")
        ax_p.plot(freq, phi, color="C1")
        ax_p.fill_between(freq, phi - sig_phi, phi + sig_phi, color="C1", alpha=0.25, lw=0)
        band = self._dynamic_range(sams, refs)[2]
        if band.sum() >= 2:
            for ax, y, sig in ((ax_a, t_abs, sig_abs), (ax_p, phi, sig_phi)):
                lo, hi = np.nanmin((y - sig)[band]), np.nanmax((y + sig)[band])
                pad = 0.05 * (hi - lo)
                ax.set_ylim(lo - pad, hi + pad)
            ax_a.axvspan(freq[band][-1], freq[-1], color="0.5", alpha=0.1, lw=0, label="< 10 dB above noise")
        ax_p.set_ylabel("Phase of T, unwrapped (rad)")
        ax_p.set_xlabel("Frequency (THz)")
        ax_a.legend(fontsize=10)
        ax_a.set_title(unc.describe(), fontsize=11)
        fig.tight_layout()
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.nanmedian(sig_abs / t_abs)
        logging.info(f"Transmission uncertainty ({cov:g} sigma) of {len(sams)} measurement(s): median "
                     f"{100 * rel:.2f} % of |T|, phase {1e3 * np.nanmedian(sig_phi):.2f} mrad; {unc.describe()}")
        self._show()

    @action("Echo diagnostic", group=diag_grp, priority=3)
    def plot_echo_diagnostic(self):
        sams, refs = self._selection()
        if sams is None:
            return
        d = self._thickness()
        if d is None:
            return
        band = self._usable_band(sams, refs)
        if band is None:
            return
        ds, freq = self.dataset, self.plotter.freq_axis[band]
        t = np.mean(ds.transmission(sams, refs)[:, np.asarray(self.plotter.freq_range_idx)[band]], axis=0)
        t_rt = fp_round_trip_time(freq, t, d)
        threshold = self.plotter.settings.eval_opt.q_value.fp_amp_threshold.to("").magnitude
        vis = echo_visibility(freq, t, d, t_rt, threshold)

        fig = plt.figure("Echo diagnostic", figsize=(10, 12))
        fig.clf()
        ax_r, ax_e, ax_t = fig.subplots(3, 1)
        if vis.tau is not None:
            shown = vis.tau <= 3.5 * t_rt
            ax_r.plot(vis.tau[shown], 100 * vis.ripple_spectrum[shown], color="C0", label="Measured")
            ax_r.axhline(100 * vis.expected_ripple, color="C1", ls="--", label="Expected (flat slab)")
            if vis.noise_ripple is not None:
                ax_r.axhline(100 * vis.noise_ripple, color="0.4", ls=":", label="Noise")
        for k in (1, 2, 3):
            ax_r.axvline(k * t_rt, color="C3", alpha=0.5, lw=1, label="Round trip time x k" if k == 1 else None)
        ax_r.set_xlabel("Delay (ps)")
        ax_r.set_ylabel("Ripple of ln|T| (%)")
        ax_r.set_title(f"d = {d:g} µm, group index {vis.n_group:.3f}, round trip {t_rt:.2f} ps", fontsize=12)
        ax_r.legend()

        ax_e.plot(freq, 100 * vis.first_echo, color="C0", label="First echo (flat slab)")
        ax_e.axhline(100 * threshold, color="C3", ls="--", label="Echo amplitude threshold")
        ax_e.set_yscale("log")
        ax_e.set_xlabel("Frequency (THz)")
        ax_e.set_ylabel("Echo / main pulse (%)")
        ax_e.legend()

        for meas_list, name, color in ((list(dict.fromkeys(refs)), "Reference", "0.5"), (sams, "Sample", "C0")):
            td = ds.get_multi_data(meas_list, Domain.Time)
            time, y = td[0, :, 0].real, td[:, :, 1].real.mean(axis=0)
            env = np.abs(hilbert(y))
            ax_t.plot(time, 20 * np.log10(np.maximum(env / env.max(), 1e-6)), color=color, lw=0.8, label=name)
            if name == "Sample":
                t_main = time[np.argmax(env)]
                for k in range(1, self.marked_echoes + 1):
                    ax_t.axvline(t_main + k * t_rt, color="C3", alpha=0.5, lw=1,
                                 label="Expected echoes" if k == 1 else None)
                ax_t.set_xlim(t_main - 5, t_main + (self.marked_echoes + 1) * t_rt)
        ax_t.set_ylim(-70, 3)
        ax_t.set_xlabel("Time (ps)")
        ax_t.set_ylabel("Envelope (dB)")
        ax_t.legend()
        fig.tight_layout()

        text = (f"Echo diagnostic: first echo at most {100 * np.max(vis.first_echo):.2f} %, above the threshold "
                + (f"up to {vis.f_max:.2f} THz" if vis.f_max is not None else "nowhere"))
        if vis.measured_ripple is not None:
            text += (f"; ripple measured {100 * vis.measured_ripple:.2f} %, expected {100 * vis.expected_ripple:.2f} %"
                     + (f", noise {100 * vis.noise_ripple:.2f} %" if vis.noise_ripple is not None else ""))
        logging.info(text)
        self._show()

    @action("Kramers-Kronig check", group=diag_grp, priority=4,
            help="n from the measured k (windowed evaluation at the nominal thickness) by a singly subtractive "
                 "Kramers-Kronig relation over the band where the sample spectrum is at least 10 dB above the noise "
                 "floor, anchored at its centre, compared with the measured n. Only the absorption inside the band "
                 "is included. Large deviations point to a wrong "
                 "thickness, phase unwrapping errors or scattering losses counted as absorption")
    def plot_kramers_kronig(self):
        sams, refs = self._selection()
        if sams is None:
            return
        d = self._thickness()
        if d is None:
            return
        band = self._usable_band(sams, refs)
        if band is None:
            return
        res = self.dataset.windowing_eval(sams, d=d)
        n_dict = {m: v[band] for m, v in self.plotter.format_meas_dict(sams, res.refr_idx.real).items()}
        k_dict = {m: v[band] for m, v in self.plotter.format_meas_dict(sams, res.refr_idx.imag).items()}

        fig = plt.figure("Kramers-Kronig check", figsize=(10, 8))
        fig.clf()
        ax_n, ax_d = fig.subplots(2, 1, sharex=True)
        for i, (meas, n_data) in enumerate(n_dict.items()):
            freq, n, k = n_data[:, 0].real, n_data[:, 1].real, k_dict[meas][:, 1].real
            anchor = len(freq) // 2
            n_kk = kramers_kronig_n(freq, k, n[anchor], anchor)
            label = self.plotter.get_legend_label(meas)
            line, = ax_n.plot(freq, n, label=f"{label}, measured")
            ax_n.plot(freq, n_kk, color=line.get_color(), ls="--", label=f"{label}, Kramers-Kronig")
            ax_d.plot(freq, n - n_kk, color=line.get_color(), label=label)
            logging.info(f"Kramers-Kronig check {label}: n - n_KK rms {np.sqrt(np.mean((n - n_kk) ** 2)):.4f}, "
                         f"largest {np.max(np.abs(n - n_kk)):.4f} (anchored at {freq[anchor]:.2f} THz)")
        ax_n.set_ylabel("Refractive index")
        ax_n.set_title(f"d = {d:g} µm", fontsize=12)
        ax_n.legend()
        ax_d.axhline(0, color="0.4", lw=1)
        ax_d.set_ylabel("n - n (Kramers-Kronig)")
        ax_d.set_xlabel("Frequency (THz)")
        fig.tight_layout()
        self._show()

    def _scatter_values(self, quantity, meas_list, batch=2000):
        func = self.dataset.quantity_func(quantity, freq_range=self.plotter.freq_range.magnitude,
                                          d_sam=self.plotter.d_sample)
        values = []
        for i in range(0, len(meas_list), batch):
            vals = np.asarray(func(meas_list[i:i + batch]))
            if vals.ndim == 0:
                raise ValueError(f"{quantity.value} is a single value, not one per measurement")
            if vals.ndim == 2:
                vals = vals[:, self.plotter.sel_freq_idx]
            values.append(vals)
        vals = np.concatenate(values)
        if not np.iscomplexobj(vals):
            return vals.astype(float)
        part = {ComplexPart.abs: np.abs, ComplexPart.real: np.real, ComplexPart.imag: np.imag,
                ComplexPart.phase: np.angle}[self.scatter_complex_part]
        return part(vals)

    @action("Scatter plot of two quantities", group=scatter_grp, priority=14)
    def plot_scatter(self):
        if self.scatter_all:
            meas_list = list(self.dataset.measurements["sams"])
        else:
            meas_list, _ = self._selection()
        if not meas_list:
            return
        try:
            x, y = (self._scatter_values(q, meas_list) for q in (self.scatter_x, self.scatter_y))
        except (KeyError, ValueError) as e:
            logging.warning(f"Scatter plot: {e}")
            return
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            logging.warning(f"Scatter plot: {valid.sum()} of {len(meas_list)} measurements have both values, at "
                            f"least two are needed")
            return

        def label(q):
            quant = q.value
            return f"{quant.label} ({quant.unit})" if quant.unit else quant.label

        fig = plt.figure(f"Scatter {self.scatter_y.name} vs {self.scatter_x.name}")
        fig.clf()
        ax = fig.add_subplot(111)
        ax.scatter(x[valid], y[valid], s=6, alpha=0.5, lw=0)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel(label(self.scatter_x))
        ax.set_ylabel(label(self.scatter_y))
        r = pearsonr(x[valid], y[valid])[0] if np.ptp(x[valid]) > 0 and np.ptp(y[valid]) > 0 else np.nan
        freq_dep = any(q.value.domain == Domain.Frequency and not q.value.uses_range
                       for q in (self.scatter_x, self.scatter_y))
        at_freq = f" at {self.plotter.freq:~P}" if freq_dep else ""
        ax.set_title(f"{valid.sum()} measurements{at_freq}, Pearson r = {r:.3f}", fontsize=12)
        fig.tight_layout()
        logging.info(f"Scatter plot of {self.scatter_y.value.label} vs {self.scatter_x.value.label}: {valid.sum()} of "
                     f"{len(meas_list)} measurements with both values{at_freq}, Pearson r = {r:.3f}")
        self._show()
