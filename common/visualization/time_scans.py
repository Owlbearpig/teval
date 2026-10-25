"""
B-scan and C-scan views of the time traces (THz imaging / NDT): along a line the traces as an image of position against
delay, where echoes and interfaces in the sample are lines; for the whole image the field, the envelope maximum or the
delay of the envelope maximum in a time gate, with a slider for the gate position (depth).
"""
import logging
from enum import Enum
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Slider
from scipy.signal import hilbert
from traitlets import Bool, Float, Enum as TEnum
from common.components import ComponentBase, action
from common.default_appsettings import Domain
from common.traits import Q_, Quantity, ValueRange
from common.visualization.image_grid import (image_measurements, grid_indices, axes_extent, apply_axis_inversion,
                                             robust_limits)
from mpl_settings import mpl_style_params

action = partial(action, check_init=True, rc_params=mpl_style_params)

trace_batch_size = 500


class CScanMode(Enum):
    field = "Field at the delay"
    envelope = "Envelope maximum in the gate"
    peak_delay = "Delay of the envelope maximum in the gate"


def envelope(traces, batch=1000):
    out = np.empty(traces.shape, dtype=np.float32)
    for i in range(0, len(traces), batch):
        out[i:i + batch] = np.abs(hilbert(traces[i:i + batch], axis=1))
    return out


class TimeScans(ComponentBase):
    bscan_grp = "B-scan"
    use_slice_line = Bool(True, help="Use the line of the slice plots (\"Line start / end point\"), otherwise the "
                                     "line below").tag(name="Use the slice plot line", group=bscan_grp, priority=0)
    line_start = ValueRange(default_value=[Q_(0.0, "mm"), Q_(0.0, "mm")]).tag(
        name="Line start point (x, y)", group=bscan_grp, priority=1)
    line_end = ValueRange(default_value=[Q_(0.0, "mm"), Q_(0.0, "mm")]).tag(
        name="Line end point (x, y)", group=bscan_grp, priority=2)
    bscan_envelope = Bool(False, help="Envelope (magnitude of the analytic signal) in dB relative to the maximum "
                                      "instead of the field: shows the echo positions independent of their sign "
                                      "and phase").tag(name="Envelope (dB)", group=bscan_grp, priority=3)
    bscan_dynamic_range = Float(40.0, min=1.0).tag(name="Envelope dynamic range (dB)", group=bscan_grp, priority=4)

    common_grp = "Time range"
    time_range = ValueRange(default_value=[Q_(0.0, "ps"), Q_(0.0, "ps")],
                            help="Shown delay range of the B- and C-scans (equal values: the whole trace). A "
                                 "shorter range also needs less memory for the C-scan").tag(
        name="Delay range", group=common_grp, priority=10)

    cscan_grp = "C-scan"
    cscan_mode = TEnum(CScanMode, CScanMode.envelope).tag(name="Shown value", group=cscan_grp, priority=20)
    gate_center = Quantity(Q_(0.0, "ps"),
                           help="Delay of the field, or start of the gate (also set by the slider of the C-scan "
                                "figure). 0: at the main pulse").tag(name="Gate position", group=cscan_grp, priority=21)
    gate_width = Quantity(Q_(1.0, "ps"), help="Width of the time gate for the envelope modes").tag(
        name="Gate width", group=cscan_grp, priority=22)

    def __init__(self, image_plot, **kwargs):
        super().__init__(**kwargs)
        self.image_plot = image_plot
        self._widgets = []

    @property
    def dataset(self):
        return self.image_plot.dataset

    def _line(self):
        if self.use_slice_line and self.image_plot.slice_plots is not None:
            return self.image_plot.slice_plots.line_start.magnitude, self.image_plot.slice_plots.line_end.magnitude
        return self.line_start.magnitude, self.line_end.magnitude

    def _time_mask(self, t):
        t0, t1 = sorted(q.to("ps").magnitude for q in self.time_range)
        return (t >= t0) & (t <= t1) if t1 > t0 else np.ones(len(t), bool)

    def _traces(self, meas_list):
        # preprocessed traces (the same as the waveform plots) on the delay range, in batches to bound the memory
        data = self.dataset.get_multi_data(meas_list[:1], Domain.Time)
        t = data[0, :, 0].real
        mask = self._time_mask(t)
        out = np.empty((len(meas_list), int(mask.sum())), dtype=np.float32)
        for i in range(0, len(meas_list), trace_batch_size):
            batch = self.dataset.get_multi_data(meas_list[i:i + trace_batch_size], Domain.Time)
            out[i:i + len(batch)] = batch[:, mask, 1].real
        return t[mask], out

    @staticmethod
    def _mean_per_point(keys, traces):
        # mean trace of the measurements with the same key (grid point / position), keys in order of appearance
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        total = np.zeros((len(unique), traces.shape[1]), dtype=np.float64)
        np.add.at(total, inverse, traces)
        counts = np.bincount(inverse, minlength=len(unique))
        return unique, (total / counts[:, None]).astype(np.float32)

    @action("Plot B-scan", group=bscan_grp, priority=5)
    def plot_bscan(self):
        start, end = (np.asarray(p, dtype=float) for p in self._line())
        if np.allclose(start, end):
            logging.warning("Set the start and end point of the B-scan line (slice plot line or B-scan line)")
            return
        shown = set(image_measurements(self.image_plot))
        meas_on_line, _ = self.image_plot.measurement_selector.get_arb_line(start, end)
        meas_on_line = [m for m in meas_on_line if m in shown]
        if len(meas_on_line) < 2:
            logging.warning("Less than two measurements on the B-scan line")
            return

        t, traces = self._traces(meas_on_line)
        direction = (end - start) / np.linalg.norm(end - start)
        dist = np.array([np.dot(np.asarray(m.position, dtype=float) - start, direction) for m in meas_on_line])
        positions, traces = self._mean_per_point(np.round(dist, 6).reshape(-1, 1), traces)
        positions = positions[:, 0]

        fig = plt.figure("B-scan", figsize=(12, 8))
        fig.clf()
        ax = fig.add_subplot(111)
        dp = float(np.median(np.diff(positions))) if len(positions) > 1 else 1.0
        extent = (positions[0] - dp / 2, positions[-1] + dp / 2, t[-1], t[0])
        if self.bscan_envelope:
            env = envelope(traces)
            db = 20 * np.log10(np.maximum(env / env.max(), 1e-12))
            img = ax.imshow(db.T, aspect="auto", extent=extent, cmap="inferno", vmin=-self.bscan_dynamic_range, vmax=0,
                            interpolation="nearest")
            label = "Envelope (dB)"
        else:
            lim = float(np.percentile(np.abs(traces), 99.5)) or 1.0
            img = ax.imshow(traces.T, aspect="auto", extent=extent, cmap="seismic", vmin=-lim, vmax=lim,
                            interpolation="nearest")
            label = "Field (a.u.)"
        fig.colorbar(img, ax=ax, label=label)
        ax.set_xlabel(f"Position along the line (mm) from ({start[0]:g}, {start[1]:g})")
        ax.set_ylabel("Delay (ps)")
        ax.set_title(f"B-scan ({start[0]:g}, {start[1]:g}) to ({end[0]:g}, {end[1]:g}) mm, "
                     f"{len(positions)} positions", fontsize=11)
        fig.tight_layout()
        self.image_plot._plot_meas_on_image(meas_on_line)
        logging.info(f"B-scan of {len(meas_on_line)} measurements at {len(positions)} positions")
        if self.image_plot.settings.plot_opt.auto_show:
            self.image_plot.plt_show()

    def _cscan_data(self):
        meas_list = image_measurements(self.image_plot)
        if not meas_list:
            return None
        t, traces = self._traces(meas_list)
        x_idx, y_idx = grid_indices(self.image_plot, meas_list)
        pixels, traces = self._mean_per_point(np.stack([x_idx, y_idx], axis=1), traces)
        need_envelope = self.cscan_mode != CScanMode.field
        values = envelope(traces) if need_envelope else traces
        logging.info(f"C-scan data: {len(pixels)} pixels x {len(t)} delays ({values.nbytes / 1e6:.0f} MB)")
        return {"t": t, "pixels": pixels, "values": values, "mode": self.cscan_mode, "mean_trace": traces.mean(axis=0)}

    def cscan_image(self, data, gate_start):
        t, values, mode = data["t"], data["values"], data["mode"]
        if mode == CScanMode.field:
            vals = values[:, int(np.argmin(np.abs(t - gate_start)))]
        else:
            width = max(self.gate_width.to("ps").magnitude, float(np.median(np.diff(t))))
            gate = (t >= gate_start) & (t <= gate_start + width)
            if not gate.any():
                gate[int(np.argmin(np.abs(t - gate_start)))] = True
            gated = values[:, gate]
            vals = gated.max(axis=1) if mode == CScanMode.envelope else t[gate][np.argmax(gated, axis=1)] - gate_start
        shape = self.image_plot.img_shape
        grid = np.full((shape["w"], shape["h"]), np.nan)
        grid[data["pixels"][:, 0], data["pixels"][:, 1]] = vals
        return grid

    @action("Plot C-scan", group=cscan_grp, priority=23)
    def plot_cscan(self):
        # computed again for every plot (dataset or preprocessing may have changed), the slider reuses it
        data = self._cscan_data()
        if data is None:
            logging.warning("No measurements for the C-scan")
            return
        t = data["t"]
        gate0 = float(np.clip(self.gate_center.to("ps").magnitude, t[0], t[-1]))
        if self.gate_center.magnitude == 0:
            # not set yet: at the main pulse
            gate0 = float(t[int(np.argmax(np.abs(data["mean_trace"])))])

        fig = plt.figure("C-scan", figsize=(11, 11))
        fig.clf()
        ax_img = fig.add_axes((0.14, 0.4, 0.78, 0.53))
        ax_trace = fig.add_axes((0.14, 0.14, 0.66, 0.17))
        ax_slider = fig.add_axes((0.25, 0.04, 0.55, 0.03))

        mode = data["mode"]
        grid = self.cscan_image(data, gate0)
        cmap = "seismic" if mode == CScanMode.field else self.image_plot.plot_settings.color_map.value
        if mode == CScanMode.field:
            lim = float(np.percentile(np.abs(data["values"]), 99.5)) or 1.0
            vmin, vmax = -lim, lim
        else:
            vmin, vmax = robust_limits(grid)
        img = ax_img.imshow(np.ma.masked_invalid(grid).T, origin="lower", extent=axes_extent(self.image_plot),
                            cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        cbar_label = {CScanMode.field: "Field (a.u.)", CScanMode.envelope: "Envelope maximum (a.u.)",
                      CScanMode.peak_delay: "Delay of the maximum after the gate start (ps)"}[mode]
        fig.colorbar(img, ax=ax_img, label=cbar_label)
        ax_img.set_xlabel("x (mm)")
        ax_img.set_ylabel("y (mm)")
        apply_axis_inversion(self.image_plot, ax_img)

        ax_trace.plot(t, data["mean_trace"], color="0.3", lw=0.8)
        ax_trace.set_ylabel("Mean")
        ax_trace.xaxis.set_major_locator(MaxNLocator(5))
        ax_trace.set_xlim(t[0], t[-1])
        width = self.gate_width.to("ps").magnitude
        marker = (ax_trace.axvline(gate0, color="C3") if mode == CScanMode.field
                  else ax_trace.axvspan(gate0, gate0 + width, color="C3", alpha=0.3))

        def title(gate):
            if mode == CScanMode.field:
                return f"C-scan: field at {gate:.2f} ps"
            return f"C-scan: {mode.value.lower()} {gate:.2f} - {gate + width:.2f} ps"

        ax_img.set_title(title(gate0), fontsize=11)
        slider = Slider(ax_slider, "Delay (ps)", float(t[0]), float(t[-1]), valinit=gate0)

        def update(val):
            new_grid = self.cscan_image(data, val)
            img.set_data(np.ma.masked_invalid(new_grid).T)
            if mode != CScanMode.field:
                img.set_clim(*robust_limits(new_grid))
            if mode == CScanMode.field:
                marker.set_xdata([val, val])
            else:
                marker.set_x(val)
            ax_img.set_title(title(val))
            self.set_trait("gate_center", Q_(float(val), "ps"))
            fig.canvas.draw_idle()

        slider.on_changed(update)
        # the slider only works while a reference to it exists
        self._widgets = [slider]
        logging.info(f"C-scan ({mode.value.lower()}), move the slider to change the delay")
        if self.image_plot.settings.plot_opt.auto_show:
            self.image_plot.plt_show()
