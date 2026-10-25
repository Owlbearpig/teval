"""
Time-frequency view of a trace: when which frequencies arrive. Water vapour absorption lines ring long after the pulse,
Fabry-Perot echoes repeat the pulse spectrum later, dispersion tilts the arrival time with frequency.
"""
import logging
from enum import Enum
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from traitlets import Bool, Float, Enum as TEnum
from common.components import ComponentBase, action
from common.default_appsettings import Domain
from common.traits import Q_, Quantity, ValueRange
from mpl_settings import mpl_style_params

action = partial(action, check_init=True, rc_params=mpl_style_params)


class TimeFrequencyMethod(Enum):
    stft = "Short-time Fourier transform (Gaussian window)"
    morlet = "Morlet wavelet transform"


def stft_gaussian(t, y, window_fwhm, freq, step=None):
    # |STFT| with a Gaussian window of the given FWHM (ps), window centres every step (default FWHM / 8)
    dt = float(np.median(np.diff(t)))
    step = step or window_fwhm / 8
    centers = np.arange(t[0], t[-1] + dt / 2, max(step, dt))
    sigma = window_fwhm / (2 * np.sqrt(2 * np.log(2)))
    n_fft = int(2 ** np.ceil(np.log2(4 * len(t))))
    f_fft = np.fft.rfftfreq(n_fft, dt)
    out = np.empty((len(freq), len(centers)))
    for i in range(0, len(centers), 64):
        c = centers[i:i + 64, None]
        windowed = y[None, :] * np.exp(-0.5 * ((t[None, :] - c) / sigma) ** 2)
        spec = np.abs(np.fft.rfft(windowed, n=n_fft, axis=1))
        out[:, i:i + len(c)] = np.array([np.interp(freq, f_fft, s) for s in spec]).T
    return centers, out


def morlet_cwt(t, y, freq, cycles):
    # |CWT| with complex Morlet wavelets of `cycles` periods (time width sigma_t = cycles / (2 pi f)), computed in
    # the frequency domain: the wavelet spectrum is a Gaussian at f with width f / cycles (analytic signal)
    dt = float(np.median(np.diff(t)))
    n_fft = int(2 ** np.ceil(np.log2(2 * len(t))))
    nu = np.fft.fftfreq(n_fft, dt)
    spec = np.fft.fft(y, n=n_fft)
    out = np.empty((len(freq), len(t)))
    for i, f in enumerate(freq):
        sigma_nu = f / cycles
        kernel = np.where(nu > 0, np.exp(-0.5 * ((nu - f) / sigma_nu) ** 2), 0.0)
        out[i] = np.abs(np.fft.ifft(spec * kernel)[:len(t)])
    return t, out


class Spectrogram(ComponentBase):
    spectrogram_grp = "Spectrogram"
    method = TEnum(TimeFrequencyMethod, TimeFrequencyMethod.stft).tag(name="Method", group=spectrogram_grp,
                                                                       priority=0)
    window_fwhm = Quantity(Q_(3.0, "ps"), help="Short-time Fourier transform: width (FWHM) of the Gaussian window; "
                                               "longer: finer frequency resolution, coarser time resolution").tag(
        name="Window width (FWHM)", group=spectrogram_grp, priority=1)
    wavelet_cycles = Float(6.0, min=1.0, help="Morlet wavelet: number of periods of the wavelet; more: finer "
                                              "frequency resolution, coarser time resolution").tag(
        name="Wavelet cycles", group=spectrogram_grp, priority=2)
    freq_range = ValueRange(default_value=[Q_(0.1, "THz"), Q_(3.0, "THz")]).tag(
        name="Frequency range", group=spectrogram_grp, priority=3)
    dynamic_range = Float(50.0, min=1.0).tag(name="Dynamic range (dB)", group=spectrogram_grp, priority=4)
    include_reference = Bool(True).tag(name="Also the reference", group=spectrogram_grp, priority=5)

    def __init__(self, plotter, **kwargs):
        super().__init__(**kwargs)
        self.plotter = plotter

    def _transform(self, t, y, freq):
        if self.method == TimeFrequencyMethod.stft:
            return stft_gaussian(t, y, self.window_fwhm.to("ps").magnitude, freq)
        return morlet_cwt(t, y, freq, self.wavelet_cycles)

    @action("Spectrogram of the selection", group=spectrogram_grp, priority=6)
    def plot_spectrogram(self):
        selection = self.plotter.measurement_selector.get_selection(must_match=False)
        if selection is None:
            return
        traces = {"Sample": selection.sams}
        if self.include_reference and selection.refs:
            traces = {"Reference": selection.refs, **traces}

        f0, f1 = sorted(q.to("THz").magnitude for q in self.freq_range)
        freq = np.linspace(max(f0, 1e-3), f1, 300)
        fig = plt.figure("Spectrogram", figsize=(9 * len(traces), 10))
        fig.clf()
        axes = fig.subplots(2, len(traces), sharex="col", gridspec_kw={"height_ratios": [1, 3]}, squeeze=False)
        results = {}
        for name, meas_list in traces.items():
            data = self.plotter.dataset.get_multi_data(list(meas_list), Domain.Time)
            t, y = data[0, :, 0].real, data[:, :, 1].real.mean(axis=0)
            results[name] = (t, y, *self._transform(t, y, freq))
        peak = max(float(np.max(r[3])) for r in results.values())
        for col, (name, (t, y, t_tf, mag)) in enumerate(results.items()):
            ax_trace, ax_tf = axes[0, col], axes[1, col]
            ax_trace.plot(t, y, color="0.3", lw=0.8)
            count = len(traces[name])
            x, y = traces[name][0].position
            ax_trace.set_title(f"{name}" + (f" (mean of {count})" if count > 1 else f" at ({x:g}, {y:g}) mm"),
                               fontsize=16)
            ax_trace.set_ylabel("Field (a.u.)")
            db = 20 * np.log10(np.maximum(mag / peak, 1e-12))
            img = ax_tf.pcolormesh(t_tf, freq, db, shading="auto", cmap="inferno", vmin=-self.dynamic_range, vmax=0)
            ax_tf.set_xlabel("Delay (ps)")
            ax_tf.set_ylabel("Frequency (THz)")
            ax_tf.xaxis.set_major_locator(MaxNLocator(5))
            fig.colorbar(img, ax=ax_tf, label="dB", orientation="horizontal", pad=0.18)
        method = (f"STFT, window {self.window_fwhm:~P}" if self.method == TimeFrequencyMethod.stft
                  else f"Morlet wavelet, {self.wavelet_cycles:g} cycles")
        fig.suptitle(f"Spectrogram ({method})")
        fig.tight_layout()
        logging.info(f"Spectrogram of {', '.join(f'{k.lower()} ({len(v)})' for k, v in traces.items())}: {method}")
        if self.plotter.settings.plot_opt.auto_show:
            self.plotter.plt_show()
