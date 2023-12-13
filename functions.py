import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from numpy import array, nan_to_num, zeros, pi
from consts import c0, THz
from numpy.fft import irfft, rfft, rfftfreq
from scipy import signal


def do_fft(data_td):
    data_td = nan_to_num(data_td)

    dt = float(np.mean(np.diff(data_td[:, 0])))
    freqs, data_fd = rfftfreq(n=len(data_td[:, 0]), d=dt), np.conj(rfft(data_td[:, 1]))

    return array([freqs, data_fd]).T


def do_ifft(data_fd):
    f_axis, y_fd = data_fd[:, 0].real, data_fd[:, 1]

    y_td = irfft(np.conj(y_fd))
    df = np.mean(np.diff(f_axis))
    n = len(y_td)
    t = np.arange(0, n) / (n * df)

    return array([t, y_td]).T


def unwrap(data_fd):
    if data_fd.ndim == 2:
        y = nan_to_num(data_fd[:, 1])
    else:
        y = nan_to_num(data_fd)
        return np.unwrap(np.angle(y))

    return array([data_fd[:, 0].real, np.unwrap(np.angle(y))]).T


def phase_correction(data_fd, disable=False, fit_range=None, en_plot=False, extrapolate=False, rewrap=False,
                     ret_fd=False, both=False):
    freqs = data_fd[:, 0].real

    if disable:
        return array([freqs, np.unwrap(np.angle(data_fd[:, 1]))]).T

    phase_unwrapped = unwrap(data_fd)

    if fit_range is None:
        fit_range = [0.40, 0.75]

    fit_slice = (freqs >= fit_range[0]) * (freqs <= fit_range[1])
    p = np.polyfit(freqs[fit_slice], phase_unwrapped[fit_slice, 1], 1)

    phase_corrected = phase_unwrapped[:, 1] - p[1].real

    if en_plot:
        plt.figure("phase_correction")
        plt.plot(freqs, phase_unwrapped[:, 1], label="Unwrapped phase")
        plt.plot(freqs, phase_corrected, label="Shifted phase")
        # plt.plot(freqs, freqs * p[0].real, label="Lin. fit (slope*freq)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (rad)")
        plt.legend()

    if extrapolate:
        phase_corrected = p[0].real * freqs

    if rewrap:
        phase_corrected = np.angle(np.exp(1j * phase_corrected))

    y = np.abs(data_fd[:, 1]) * np.exp(1j * phase_corrected)
    if both:
        return do_ifft(array([freqs, y]).T), array([freqs, y]).T

    if ret_fd:
        return array([freqs, y]).T
    else:
        return array([freqs, phase_corrected]).T


def peak_cnt(data_td, threshold=1):
    y_ = data_td[:, 1]
    y_ -= (np.mean(y_[:10]) + np.mean(y_[-10:])) * 0.5

    y_[y_ < threshold] = 0
    peaks_idx = []
    for idx_ in range(1, len(y_)-1):
        if (y_[idx_-1] < y_[idx_]) * (y_[idx_] > y_[idx_+1]):
            peaks_idx.append(idx_)

    return len(peaks_idx)


def zero_pad(data_td, length=100):
    t, y = data_td[:, 0], data_td[:, 1]
    dt = np.mean(np.diff(data_td[:, 0]))
    cnt = int(length / dt)

    new_t = np.concatenate((t, np.arange(t[-1], t[-1] + cnt * dt, dt)))
    new_y = np.concatenate((y, np.zeros(cnt)))

    return array([new_t, new_y]).T


def window(data_td, win_len=None, win_start=None, shift=None, en_plot=False, slope=0.15):
    t, y = data_td[:, 0], data_td[:, 1]
    t -= t[0]
    pulse_width = 10  # ps
    dt = np.mean(np.diff(t))

    if win_len is None:
        win_len = int(pulse_width / dt)
    else:
        win_len = int(win_len / dt)

    if win_len > len(y):
        win_len = len(y)

    if win_start is None:
        win_center = np.argmax(np.abs(y))
        win_start = win_center - int(win_len / 2)
    else:
        win_start = int(win_start / dt)

    if win_start < 0:
        win_start = 0

    pre_pad = np.zeros(win_start)
    window_arr = signal.windows.tukey(win_len, slope)
    post_pad = np.zeros(len(y) - win_len - win_start)

    window_arr = np.concatenate((pre_pad, window_arr, post_pad))

    if shift is not None:
        window_arr = np.roll(window_arr, int(shift / dt))

    y_win = y * window_arr

    if en_plot:
        plt.figure("Windowing")
        plt.plot(t, y, label="Sam. before windowing")
        plt.plot(t, np.max(np.abs(y)) * window_arr, label="Window")
        plt.plot(t, y_win, label="Sam. after windowing")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (nA)")
        plt.legend()

    return np.array([t, y_win]).T


def calc_absorption(freqs, k):
    # Assuming freqs in range (0, 10 THz), returns a in units of 1/cm (1/m * 1/100)
    omega = 2 * pi * freqs * THz
    a = (2 * omega * k) / c0

    return a / 100


def cauchy_relation(freqs, p):
    lam = (c0 / freqs) * 10 ** -9

    n = np.zeros_like(lam)
    for i, coeff in enumerate(p):
        n += coeff * lam ** (-2 * i)

    return n


def add_noise(data_fd, enabled=True, scale=0.05, seed=None, en_plots=False):
    data_ret = nan_to_num(data_fd)

    np.random.seed(seed)

    if not enabled:
        return data_ret

    noise_phase = np.random.normal(0, scale * 0, len(data_fd[:, 0]))
    noise_amp = np.random.normal(0, scale * 1.5, len(data_fd[:, 0]))

    phi, magn = np.angle(data_fd[:, 1]), np.abs(data_fd[:, 1])

    phi_noisy = phi + noise_phase
    magn_noisy = magn * (1 + noise_amp)

    if en_plots:
        freqs = data_ret[:, 0]

        plt.figure("Phase")
        plt.plot(freqs, phi, label="Original data")
        plt.plot(freqs, phi_noisy, label="+ noise")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (rad)")
        plt.legend()

        plt.figure("Spectrum")
        plt.plot(freqs, magn, label="Original data")
        plt.plot(freqs, magn_noisy, label="+ noise")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (a.u.)")
        plt.legend()
        plt.show()

    noisy_data = magn_noisy * np.exp(1j * phi_noisy)

    data_ret[:, 1] = noisy_data.real + 1j * noisy_data.imag

    return data_ret


def pearson_corr_coeff(data0_fd, data1_fd):
    mod_td_y, sam_td_y = do_ifft(data0_fd)[:, 1], do_ifft(data1_fd)[:, 1]
    corr = pearsonr(mod_td_y.real, sam_td_y.real)

    return max(corr)


def chill():
    pass


# Polynomial Regression
def polyfit(x, y, degree, remove_worst_outlier=False):
    def _fit(x_, y_):
        res = {}

        coeffs = np.polyfit(x_, y_, degree)

        # Polynomial Coefficients
        res['polynomial'] = coeffs.tolist()

        # r-squared
        p = np.poly1d(coeffs)
        # fit values, and mean
        yhat = p(x_)  # or [p(z) for z in x]
        ybar = np.sum(y_) / len(y_)  # or sum(y)/len(y)
        ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
        sstot = np.sum((y_ - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])

        res['determination'] = ssreg / sstot

        return res

    def _remove_outlier(x_, y_):
        # len(x_) == len(y_)

        max_R, x_best, y_best = 0, None, None
        for i in range(len(x_)):
            x_test, y_test = np.delete(x_, i), np.delete(y_, i)

            res = _fit(x_test, y_test)
            if res["determination"] > max_R:
                max_R = res["determination"]
                x_best, y_best = x_test, y_test

        return x_best, y_best

    # https://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy

    slice_ = y > 0  # 1.5e5
    # slice_ = y > 1.5e5
    x, y = x[slice_], y[slice_]
    if True:
        x, y = _remove_outlier(x, y)
    results = _fit(x, y)

    return results


def to_db(data_fd):
    if data_fd.ndim == 2:
        return 20 * np.log10(np.abs(data_fd[:, 1]))
    else:
        return 20 * np.log10(np.abs(data_fd))


def get_noise_floor(data_fd, noise_start=6.0):
    return np.mean(20 * np.log10(np.abs(data_fd[data_fd[:, 0] > noise_start, 1])))


def zero_pad_fd(data0_fd, data1_fd):
    # expected data1_fd range: 0, 10 THz.
    df = np.mean(np.diff(data1_fd[:, 0].real))
    min_freq, max_freq = data0_fd[:, 0].real.min(), data0_fd[:, 0].real.max()
    pre_pad, post_pad = np.arange(0, min_freq, df), np.arange(max_freq, 10, df)
    padded_freqs = np.concatenate((pre_pad,
                                   data0_fd[:, 0].real,
                                   post_pad))
    padded_data = np.concatenate((zeros(len(pre_pad)),
                                  data0_fd[:, 1],
                                  zeros(len(post_pad))))
    return array([padded_freqs, padded_data]).T


def filtering(data_td, wn=(0.001, 9.999), filt_type="bandpass", order=5):
    dt = np.mean(np.diff(data_td[:, 0].real))
    fs = 1 / dt

    # sos = signal.butter(N=order, Wn=wn, btype=filt_type, fs=fs, output='sos')
    ba = signal.butter(N=order, Wn=wn, btype=filt_type, fs=fs, output='ba')
    # sos = signal.bessel(N=order, Wn=wn, btype=filt_type, fs=fs, output='ba')
    # data_td_filtered = signal.sosfilt(sos, data_td[:, 1])
    data_td_filtered = signal.filtfilt(*ba, data_td[:, 1])

    data_td_filtered = array([data_td[:, 0], data_td_filtered]).T

    return data_td_filtered


def f_axis_idx_map(freqs, freq_range=None):
    if freq_range is None:
        freq_range = (0.10, 4.00)
        f0_idx = int(np.argmin(np.abs(freqs - freq_range[0])))
        f1_idx = int(np.argmin(np.abs(freqs - freq_range[1])))
        f_idx = np.arange(f0_idx, f1_idx + 1)
    elif isinstance(freq_range, tuple):
        f0_idx = int(np.argmin(np.abs(freqs - freq_range[0])))
        f1_idx = int(np.argmin(np.abs(freqs - freq_range[1])))
        f_idx = np.arange(f0_idx, f1_idx + 1)
    else:
        single_freq = freq_range
        f_idx = np.array([int(np.argmin(np.abs(freqs - single_freq)))])

    return f_idx


def remove_spikes(arr):
    # TODO pretty bad
    diff = np.diff(arr)
    for i in range(1, len(arr) - 1):
        if i < 5:
            avg_diff = np.mean(np.diff(arr[i:i + 3]))
        else:
            avg_diff = np.mean(np.diff(arr[i - 3:i]))

        if diff[i] > avg_diff:
            arr[i + 1] = (arr[i] + arr[i + 2]) / 2

    return arr
