import numpy as np


def image_measurements(image_plot):
    measurements = image_plot.measurements
    return list(measurements["sams"] if image_plot.only_use_sample_set else measurements["all"])


def grid_indices(image_plot, meas_list):
    shape = image_plot.img_shape
    positions = np.array([m.position for m in meas_list], dtype=float).reshape(-1, 2)
    x_idx = np.argmin(np.abs(positions[:, 0, None] - shape["x_coords"]), axis=1)
    y_idx = np.argmin(np.abs(positions[:, 1, None] - shape["y_coords"]), axis=1)
    return x_idx, y_idx


def to_grid(image_plot, meas_list, values):
    # (N_meas, ...) values -> (w, h, ...) grid, the mean of the measurements of a grid point (the image plot keeps the
    # last one; for traces and spectra the mean is the better choice), nan where there is none
    values = np.asarray(values)
    shape = image_plot.img_shape
    x_idx, y_idx = grid_indices(image_plot, meas_list)
    dtype = complex if np.iscomplexobj(values) else float
    total = np.zeros((shape["w"], shape["h"], *values.shape[1:]), dtype=dtype)
    count = np.zeros((shape["w"], shape["h"]))
    np.add.at(total, (x_idx, y_idx), values)
    np.add.at(count, (x_idx, y_idx), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = total / count.reshape(*count.shape, *([1] * (values.ndim - 1)))
    grid[count == 0] = np.nan
    return grid


def axes_extent(image_plot):
    shape = image_plot.img_shape
    ext, dx, dy = shape["extent"], shape["dx"], shape["dy"]
    return float(ext[0] - dx / 2), float(ext[1] + dx / 2), float(ext[2] - dy / 2), float(ext[3] + dy / 2)


def apply_axis_inversion(image_plot, ax):
    if image_plot.plot_settings.invert_x:
        ax.invert_xaxis()
    if image_plot.plot_settings.invert_y:
        ax.invert_yaxis()


def robust_limits(vals, percentiles=(1.0, 99.0)):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return 0.0, 1.0
    lo, hi = (float(v) for v in np.percentile(vals, percentiles))
    if hi <= lo:
        pad = abs(lo) * 1e-3 or 1e-12
        lo, hi = lo - pad, hi + pad
    return lo, hi
