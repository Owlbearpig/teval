"""
Spectral image views: the selected frequency-domain quantity of the image plot (transmission, absorbance, n, alpha,
...) for every frequency of the frequency range with a slider, the spectrum of a clicked pixel, and a false-color RGB
image of three frequency bands (material differences that a single quantity at one frequency doesn't show).
"""
import logging
import warnings
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Slider
from traitlets import Bool
from common.components import ComponentBase, action
from common.default_appsettings import Domain
from common.functions import f_axis_idx_map
from common.traits import Q_, ValueRange
from common.visualization.image_grid import (image_measurements, to_grid, axes_extent, apply_axis_inversion,
                                             robust_limits)
from mpl_settings import mpl_style_params

action = partial(action, check_init=True, rc_params=mpl_style_params)

quantity_batch_size = 1000


class SpectralExplorer(ComponentBase):
    explorer_grp = "Frequency explorer"
    fixed_color_limits = Bool(False, help="The same color limits for all frequencies (1 - 99 % of all values), "
                                          "otherwise the limits follow the shown frequency").tag(
        name="Same color limits for all frequencies", group=explorer_grp, priority=0)

    rgb_grp = "False-color image"
    red_band = ValueRange(default_value=[Q_(0.3, "THz"), Q_(0.6, "THz")]).tag(
        name="Red band", group=rgb_grp, priority=10)
    green_band = ValueRange(default_value=[Q_(0.6, "THz"), Q_(1.0, "THz")]).tag(
        name="Green band", group=rgb_grp, priority=11)
    blue_band = ValueRange(default_value=[Q_(1.0, "THz"), Q_(1.5, "THz")]).tag(
        name="Blue band", group=rgb_grp, priority=12)
    rgb_common_scale = Bool(False, help="One scale for the three bands (absolute differences between the bands); "
                                        "otherwise each band is stretched to its own 1 - 99 % range (shows the "
                                        "spatial variation of each band best)").tag(
        name="Common scale for the bands", group=rgb_grp, priority=13)

    def __init__(self, image_plot, **kwargs):
        super().__init__(**kwargs)
        self.image_plot = image_plot
        self._widgets = []

    @property
    def dataset(self):
        return self.image_plot.dataset

    def _spectral_cube(self, freq_range):
        # (w, h, F) grid of the selected quantity on the frequency range, F the frequencies of the dataset axis
        ip = self.image_plot
        quantity = ip.selected_quantity
        if quantity.domain != Domain.Frequency:
            logging.warning(f"Select a frequency-domain quantity in the image plot ({quantity} is not)")
            return None
        meas_list = image_measurements(ip)
        if not meas_list:
            return None
        func = ip.get_selected_quantity_func(ip.grid_calc_freq_range)
        freq_idx = f_axis_idx_map(self.dataset.freq_axis, freq_range)
        freq = self.dataset.freq_axis[freq_idx]
        values = np.empty((len(meas_list), len(freq)))
        for i in range(0, len(meas_list), quantity_batch_size):
            batch = np.asarray(func(meas_list[i:i + quantity_batch_size]))
            if batch.ndim != 2:
                logging.warning(f"{quantity} is not a spectrum (one value per measurement), select a quantity with a "
                                f"value per frequency")
                return None
            values[i:i + len(batch)] = ip.to_real(batch[:, freq_idx])
        return freq, to_grid(ip, meas_list, values)

    @action("Frequency explorer", group=explorer_grp, priority=1)
    def plot_explorer(self):
        ip = self.image_plot
        rng = [q.to("THz").magnitude for q in ip.grid_calc_freq_range]
        res = self._spectral_cube(rng)
        if res is None:
            return
        freq, cube = res
        label = str(ip.selected_quantity)
        fixed = robust_limits(cube) if self.fixed_color_limits else None

        fig = plt.figure("Frequency explorer", figsize=(18, 8))
        fig.clf()
        ax_img = fig.add_axes((0.07, 0.22, 0.4, 0.7))
        ax_spec = fig.add_axes((0.62, 0.22, 0.36, 0.7))
        ax_slider = fig.add_axes((0.25, 0.05, 0.6, 0.03))

        i0 = int(np.argmin(np.abs(freq - ip.freq.to("THz").magnitude)))
        layer = cube[:, :, i0]
        img = ax_img.imshow(np.ma.masked_invalid(layer).T, origin="lower", extent=axes_extent(ip),
                            cmap=ip.plot_settings.color_map.value, interpolation="nearest",
                            clim=fixed or robust_limits(layer))
        fig.colorbar(img, ax=ax_img, label=label)
        ax_img.set_xlabel("x (mm)")
        ax_img.set_ylabel("y (mm)")
        ax_img.set_title(f"{label} at {freq[i0]:.3f} THz", fontsize=14)
        apply_axis_inversion(ip, ax_img)

        flat = cube.reshape(-1, len(freq))
        valid = np.isfinite(flat).any(axis=1)
        median = np.nanmedian(flat[valid], axis=0) if valid.any() else np.full(len(freq), np.nan)
        p_lo, p_hi = (np.nanpercentile(flat[valid], p, axis=0) for p in (10, 90)) if valid.any() else (median, median)
        ax_spec.fill_between(freq, p_lo, p_hi, color="0.85", label="10 - 90 % of the pixels")
        ax_spec.plot(freq, median, color="0.3", label="Median")
        (pixel_line,) = ax_spec.plot([], [], color="C3", label="Clicked pixel")
        f_marker = ax_spec.axvline(freq[i0], color="C0", lw=1)
        ax_spec.set_xlabel("Frequency (THz)")
        ax_spec.set_ylabel(label)
        ax_spec.xaxis.set_major_locator(MaxNLocator(6))
        ax_spec.legend(loc="best", fontsize=12)

        slider = Slider(ax_slider, "Frequency (THz)", float(freq[0]), float(freq[-1]), valinit=float(freq[i0]),
                        valstep=freq)

        def update(val):
            i = int(np.argmin(np.abs(freq - val)))
            layer_ = cube[:, :, i]
            img.set_data(np.ma.masked_invalid(layer_).T)
            img.set_clim(*(fixed or robust_limits(layer_)))
            ax_img.set_title(f"{label} at {freq[i]:.3f} THz")
            f_marker.set_xdata([freq[i], freq[i]])
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes is not ax_img or event.xdata is None:
                return
            shape = ip.img_shape
            xi = int(np.argmin(np.abs(shape["x_coords"] - event.xdata)))
            yi = int(np.argmin(np.abs(shape["y_coords"] - event.ydata)))
            pixel_line.set_data(freq, cube[xi, yi])
            pixel_line.set_label(f"Pixel ({shape['x_coords'][xi]:g}, {shape['y_coords'][yi]:g}) mm")
            ax_spec.legend(loc="best", fontsize=12)
            ax_spec.relim()
            ax_spec.autoscale_view()
            fig.canvas.draw_idle()

        slider.on_changed(update)
        # the widgets only work while references to them exist
        self._widgets = [slider, fig.canvas.mpl_connect("button_press_event", on_click)]
        logging.info(f"Frequency explorer of {label}, {len(freq)} frequencies {freq[0]:.2f} - {freq[-1]:.2f} THz; "
                     f"click a pixel for its spectrum")
        if ip.settings.plot_opt.auto_show:
            ip.plt_show()

    @action("False-color image", group=rgb_grp, priority=14)
    def plot_false_color(self):
        ip = self.image_plot
        bands = [sorted(q.to("THz").magnitude for q in band) for band in (self.red_band, self.green_band,
                                                                         self.blue_band)]
        res = self._spectral_cube([min(b[0] for b in bands), max(b[1] for b in bands)])
        if res is None:
            return
        freq, cube = res
        channels = []
        for lo, hi in bands:
            sel = (freq >= lo) & (freq <= hi)
            if not sel.any():
                sel[int(np.argmin(np.abs(freq - (lo + hi) / 2)))] = True
            with warnings.catch_warnings():
                # pixels without a value: nan, intended
                warnings.simplefilter("ignore", RuntimeWarning)
                channels.append(np.nanmean(cube[:, :, sel], axis=2))
        channels = np.stack(channels, axis=2)
        valid = np.isfinite(channels).all(axis=2)
        if self.rgb_common_scale:
            lo, hi = robust_limits(channels[valid])
            scaled = (channels - lo) / (hi - lo)
        else:
            scaled = np.empty_like(channels)
            for c in range(3):
                lo, hi = robust_limits(channels[:, :, c][valid])
                scaled[:, :, c] = (channels[:, :, c] - lo) / (hi - lo)
        rgba = np.zeros((*channels.shape[:2], 4))
        rgba[:, :, :3] = np.clip(np.nan_to_num(scaled), 0, 1)
        rgba[:, :, 3] = valid

        fig = plt.figure("False-color image", figsize=(11, 8))
        fig.clf()
        ax = fig.add_subplot(111)
        ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower", extent=axes_extent(ip), interpolation="nearest")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        band_text = ", ".join(f"{name} {lo:g}-{hi:g}" for name, (lo, hi) in zip("RGB", bands))
        ax.set_title(f"{ip.selected_quantity}: {band_text} THz", fontsize=14)
        apply_axis_inversion(ip, ax)
        fig.tight_layout()
        logging.info(f"False-color image of {ip.selected_quantity} ({band_text} THz, "
                     f"{'common scale' if self.rgb_common_scale else 'each band stretched'})")
        if ip.settings.plot_opt.auto_show:
            ip.plt_show()
