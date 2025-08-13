import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from numpy import array, nan_to_num, zeros, pi
from consts import c0, THz
from numpy.fft import irfft, rfft, rfftfreq
from scipy import signal


def do_fft(data_td):
    data_td = nan_to_num(data_td)

    dt = float(np.mean(np.diff(data_td[:, 0])))
    freqs, data_fd = rfftfreq(n=len(data_td[:, 0]), d=dt), rfft(data_td[:, 1])

    return array([freqs, data_fd]).T


def do_ifft(data_fd, out_len=None):
    f_axis, y_fd = data_fd[:, 0].real, data_fd[:, 1]

    y_td = irfft(np.conj(y_fd), n=out_len)
    df = np.mean(np.diff(f_axis))
    n = len(y_td)
    t = np.arange(0, n) / (n * df)

    data_td = array([t, y_td]).T

    return data_td


def unwrap(data_fd):
    if data_fd.ndim == 2:
        y = nan_to_num(data_fd[:, 1])
    else:
        y = nan_to_num(data_fd)
        return np.unwrap(np.angle(y))

    return array([data_fd[:, 0].real, np.unwrap(np.angle(y))]).T


def phase_correction(freq_axis_, phi, disable=False, fit_range=None, en_plot=False,
                     extrapolate=False, rewrap=False):
    if disable:
        return phi

    if fit_range is None:
        fit_range = [0.5, 1.50]

    fit_slice = (freq_axis_ >= fit_range[0]) * (freq_axis_ <= fit_range[1])
    p = np.polyfit(freq_axis_[fit_slice], phi[fit_slice], 1)

    phi_corrected = phi - p[1].real

    if en_plot:
        plt.figure("phase_correction")
        plt.plot(freq_axis_, phi, label="Unwrapped phase")
        plt.plot(freq_axis_, phi_corrected, label="Shifted phase")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (rad)")
        plt.legend()

    if extrapolate:
        phi_corrected = p[0].real * freq_axis_

    if rewrap:
        phi_corrected = np.angle(np.exp(1j * phi_corrected))

    return phi_corrected

def remove_offset(data_td):
    data_td[:, 1] -= np.mean(data_td[:10, 1])

    return data_td

def zero_pad(data_td, length=100):
    t, y = data_td[:, 0], data_td[:, 1]
    dt = np.mean(np.diff(data_td[:, 0]))
    cnt = int(length / dt)

    new_t = np.concatenate((t, np.arange(t[-1], t[-1] + cnt * dt, dt)))
    new_y = np.concatenate((y, np.zeros(cnt)))

    return array([new_t, new_y]).T


def window(data_td, win_width=None, win_start=None, shift=None, en_plot=False, slope=0.15, **k):
    t, y = data_td[:, 0], data_td[:, 1]
    t -= t[0]
    default_width = 10  # ps
    dt = np.mean(np.diff(t))

    if win_width is None:
        win_width = int(default_width / dt)
    else:
        win_width = int(win_width / dt)

    if win_width > len(y):
        win_width = len(y)

    if win_start is None:
        win_center = np.argmax(np.abs(y))
        win_start = win_center - int(win_width / 2)
    else:
        win_start = int(win_start / dt)

    window_arr = signal.windows.tukey(win_width, slope)
    window_mask = np.zeros(len(y))
    window_mask[:win_width] = window_arr

    window_mask = np.roll(window_mask, win_start)
    if win_start < 0:
        window_mask[len(y)+win_start:] = 0

    if shift is not None:
        window_mask = np.roll(window_mask, int(shift / dt))

    y_win = y * window_mask

    if en_plot:
        plt.figure("Windowing")
        plt.plot(t, y, label="Sam. before windowing")
        plt.plot(t, np.max(np.abs(y)) * window_mask, label="Window")
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


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    Output array is shifted by window_len
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    return y

def local_minima_1d(arr, en_plot=True):
    win_len = 25
    if len(arr) > 100:
        step = 10
        smooth_arr = smooth(arr, win_len)
    else:
        step = 1
        smooth_arr = arr

    minima_idx_smooth = []
    for i in range(step, len(smooth_arr) - step):
        prev_is_down_slope = all(np.diff(smooth_arr[i - step:i]) < 0)
        next_is_up_slope = all(np.diff(smooth_arr[i:i+step]) > 0)
        if smooth_arr[i - 1] > smooth_arr[i] < smooth_arr[i + 1]:
            if prev_is_down_slope and next_is_up_slope:
                minima_idx_smooth.append(i)

    minima_idx = np.array(minima_idx_smooth, dtype=int) - win_len // 2
    mean_period, std_period = np.mean(np.diff(minima_idx)), np.std(np.diff(minima_idx))

    if en_plot:
        plt.figure("local minima - smoothed")
        plt.plot(smooth_arr)
        x = np.arange(len(smooth_arr))
        plt.scatter(x[minima_idx_smooth], smooth_arr[minima_idx_smooth], c="red", s=15)

        plt.figure("local minima - original")
        plt.plot(arr)
        x = np.arange(len(arr))
        plt.scatter(x[minima_idx], arr[np.array(minima_idx)], c="red", s=15)

    return minima_idx, mean_period, std_period


def save_fig(fig_num_, mpl=None, save_dir=None, filename=None, **kwargs):
    if mpl is None:
        import matplotlib as mpl

    plt = mpl.pyplot

    rcParams = mpl.rcParams

    if save_dir is None:
        save_dir = Path(rcParams["savefig.directory"])

    fig = plt.figure(fig_num_)

    if filename is None:
        try:
            filename_s = str(fig.canvas.get_window_title())
        except AttributeError:
            filename_s = str(fig.canvas.manager.get_window_title())
    else:
        filename_s = str(filename)

    unwanted_chars = ["(", ")"]
    for char in unwanted_chars:
        filename_s = filename_s.replace(char, '')
    filename_s.replace(" ", "_")

    fig.set_size_inches((12, 9), forward=False)
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(save_dir / (filename_s + ".pdf"), bbox_inches='tight', dpi=300, pad_inches=0, **kwargs)


def plt_show(mpl_=None, en_save=False):
    if mpl_ is None:
        mpl_ = mpl
    plt_ = mpl_.pyplot
    for fig_num in plt_.get_fignums():
        fig = plt_.figure(fig_num)
        for ax in fig.get_axes():
            h, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend()

        if en_save:
            save_fig(fig_num, mpl_)
    plt_.show()
