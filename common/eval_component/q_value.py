import numpy as np
from scipy.signal import detrend

from common.consts import c_thz
from common.eval_component.eval_result import SingleResultData
from common.eval_component.quantity_set import QuantityDataSet
from common.units import Q_


def fp_round_trip_time(freq_axis, t_exp, d, known_delay=0.0):
    # known_delay: part of the measured group delay caused by other (known) layers of the sample
    phi = np.unwrap(np.angle(t_exp))
    group_delay = np.abs(np.polyfit(2 * np.pi * freq_axis, phi, 1)[0])

    return 2 * (max(group_delay - known_delay, 0.0) + d / c_thz)


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


def remove_spikes(y, ratio=8.0):
    # isolated single points: deviation from the mean of their two neighbours above ratio x the robust scatter of
    # these deviations and well above the deviation of both neighbours (a single spike makes them deviate by half)
    y = np.array(y, dtype=float)
    if len(y) < 5:
        return y
    dev = np.abs(y[1:-1] - 0.5 * (y[:-2] + y[2:]))
    limit = ratio * 1.4826 * np.median(dev)
    for _ in range(len(y) // 4):
        dev = np.abs(y[1:-1] - 0.5 * (y[:-2] + y[2:]))
        nbr_dev = np.maximum(np.r_[0.0, dev[:-1]], np.r_[dev[1:], 0.0])
        spike = np.where((dev > limit) & (dev >= 1.5 * nbr_dev), dev, 0.0)
        i = int(np.argmax(spike))
        if spike[i] == 0:
            break
        y[i + 1] = 0.5 * (y[i] + y[i + 2])

    return y


def compute_q_value(res_data: SingleResultData, params):
    # q-value of a fit result: amplitude of the Fabry-Perot oscillation left in n / k / alpha. params: plain dict
    # (QSpaceEval.q_value_params) so it can be evaluated in the worker processes
    q_idx = params["q_idx"]
    freq_axis = res_data.freq_axis.magnitude[q_idx]

    n_full = res_data.datasets["n"].data.magnitude
    y = {"n": n_full.real, "k": n_full.imag}.get(params["quantity"])
    if y is None:
        y = res_data.datasets[params["quantity"]].data.magnitude
    y = y[q_idx]

    d = res_data.d.magnitude
    if params["fp_time"] is None:
        t_exp = res_data.datasets["t_exp"].data.magnitude[q_idx]
        t_fp = fp_round_trip_time(freq_axis, t_exp, d, params["known_delay"])
    else:
        t_fp = params["fp_time"]

    echo_amp = fp_echo_amplitude(freq_axis, n_full[q_idx], d, params["n_prev"], params["n_next"])
    q_freq_axis = freq_axis
    if params["auto_f_max"]:
        below = np.flatnonzero(echo_amp < params["amp_threshold"])
        f_max = freq_axis[below[0]] if len(below) else freq_axis[-1]
        # keep at least two FP periods
        f_max = min(max(f_max, freq_axis[0] + 2 / t_fp), freq_axis[-1])
        band = freq_axis <= f_max
        q_freq_axis, y = freq_axis[band], y[band]
    res_data.optimization_info["q_space_f_max"] = Q_(q_freq_axis[-1], "THz")
    if params.get("remove_spikes"):
        y = remove_spikes(y)

    q_axis, q_spec = q_space_spectrum(q_freq_axis, y)
    # half the main lobe width of the unwindowed transform, the peak position is only known to ~1 / bandwidth
    half_width = 0.5 / np.ptp(q_freq_axis)
    window = np.abs(q_axis - t_fp) <= half_width
    if not window.any():
        window[np.argmin(np.abs(q_axis - t_fp))] = True

    res_data.optimization_info["q_val"] = Q_(np.max(q_spec[window]), "")
    res_data.optimization_info["q_sum"] = Q_(np.sum(q_spec[window]), "")
    res_data.optimization_info["fp_time"] = Q_(t_fp, "ps")
    res_data.optimization_info["fp_period"] = Q_(1 / t_fp, "THz")

    res_data.datasets["q_space"] = QuantityDataSet(axes=[Q_(q_axis, "ps")],
                                                   data=Q_(q_spec, ""),
                                                   axes_labels=["Round trip time"],
                                                   data_label=f"Q-space |FFT({params['quantity']})|, "
                                                              f"FP time {t_fp:.3f} ps, "
                                                              f"{q_freq_axis[0]:.2f}-{q_freq_axis[-1]:.2f} THz")
    res_data.datasets["fp_echo_amp"] = QuantityDataSet(axes=[res_data.freq_axis[q_idx]],
                                                       data=Q_(100 * echo_amp, "%"),
                                                       axes_labels=["Frequency"],
                                                       data_label="Estimated Fabry-Perot echo amplitude")

    return res_data.optimization_info["q_val"]


def optimize_and_rate(d, shift, config_dict, opt_func, q_params, kept_datasets=None):
    # worker task: fit at (d, shift), q-value of the result, then drop the datasets the caller doesn't need so less
    # is sent back to the main process. kept_datasets: names of the datasets to keep (default all)
    res = opt_func(d, shift, config_dict)
    compute_q_value(res, q_params)
    if kept_datasets is not None:
        res.datasets = {k: v for k, v in res.datasets.items() if k in kept_datasets}

    return res
