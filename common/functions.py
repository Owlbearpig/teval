import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import numpy as np
from common.units import Q_
from scipy.stats import pearsonr
from numpy import array, nan_to_num, zeros, pi
from common.consts import c0, THz
from numpy.fft import irfft, rfft, rfftfreq
from scipy import signal
from enum import Enum
from matplotlib._pylab_helpers import Gcf
from scipy.signal import butter, filtfilt, get_window

class WindowTypes(Enum):
    tukey = "tukey"
    hannin = "hanning"
    rectangular = "rectangular"

def do_fft(data_td):
    data_td = nan_to_num(data_td)

    dt = float(np.mean(np.diff(data_td[:, 0])))
    freqs, data_fd = rfftfreq(n=len(data_td[:, 0]), d=dt), rfft(data_td[:, 1])

    return array([freqs, data_fd]).T


def do_ifft(data_fd, out_len=None, conj=True):
    f_axis, y_fd = data_fd[:, 0].real, data_fd[:, 1]

    if conj:
        y_td = irfft(np.conj(y_fd), n=out_len)
    else:
        y_td = irfft(y_fd, n=out_len)

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

def butter_filt(data_td, f_range):

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        f = filtfilt(b, a, data)

        return f

    fs = 1 / np.mean(np.diff(data_td[:, 0]))
    y = data_td[:, 1]
    data_filtered = butter_bandpass_filter(y, *f_range, fs)

    return np.array([data_td[:, 0], data_filtered]).T


def flip_phase(data_fd):
    freq_axis = data_fd[:, 0].real
    y_amp_ = np.abs(data_fd[:, 1])
    y_phi_ = np.unwrap(np.angle(data_fd[:, 1]))

    return np.array([freq_axis, y_amp_ * np.exp(-1j*y_phi_)]).T


def window(data_td, **kwargs):
    win_width = kwargs.get("win_width")
    shift = kwargs.get("shift")
    en_plot = kwargs.get("en_plot")
    slope = kwargs.get("slope")
    win_start = kwargs.get("win_start")

    t, y = data_td[:, 0], data_td[:, 1]
    t -= t[0]
    dt = np.mean(np.diff(t))

    win_width = int(win_width / dt)

    if win_width > len(y):
        win_width = len(y)

    win_center = np.argmax(np.abs(y))
    if win_start is None:
        win_start = win_center - int(win_width / 2)

    if "type" in kwargs:
        window = kwargs["type"]
        if window == WindowTypes.tukey:
            window_arr = get_window(window.name, win_width, slope)
        else:
            window_arr = get_window(window.name, win_width)

    window_mask = np.zeros(len(y))
    window_mask[:win_width] = window_arr

    window_mask = np.roll(window_mask, win_start)
    if win_start < 0:
        window_mask[len(y)+win_start:] = 0

    window_mask = np.roll(window_mask, int(shift / dt))

    y_win = y * window_mask

    if en_plot:
        if "fig_label" in kwargs:
            fig_label = f"_{kwargs['fig_label']}"
        else:
            fig_label = ""

        plt.figure("Window" + fig_label)
        plt.plot(t, y, label="Before windowing")
        plt.plot(t, np.max(np.abs(y)) * window_mask, label="Window")
        plt.plot(t, y_win, label="After windowing")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (nA)")
        plt.legend()

    return np.array([t, y_win]).T



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


def to_db(data_fd, normalize=False):
    res = np.abs(data_fd).astype(float)
    if normalize:
        res[..., 1] = res[..., 1] / np.max(res, axis=-1)
    res[..., 1] = 20 * np.log10(res[..., 1])

    return res



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
    if freqs is None:
        return None
    if isinstance(freq_range, (float, int, Q_)):
        single_freq = freq_range.magnitude if isinstance(freq_range, Q_) else freq_range
        f_idx = np.array([int(np.argmin(np.abs(freqs - single_freq)))])
    elif freq_range is None:
        freq_range = (0.10, 4.00)
        f0_idx = int(np.argmin(np.abs(freqs - freq_range[0])))
        f1_idx = int(np.argmin(np.abs(freqs - freq_range[1])))
        f_idx = np.arange(f0_idx, f1_idx + 1, dtype=int)
    elif len(freq_range) == 2:
        if isinstance(freq_range[0], Q_):
            freq_range = [freq_range[0].magnitude, freq_range[1].magnitude]

        sign = np.sign(freq_range[1] - freq_range[0])
        f0_idx = int(np.argmin(np.abs(freqs - freq_range[0])))
        f1_idx = int(np.argmin(np.abs(freqs - freq_range[1])))
        f_idx = np.arange(f0_idx, f1_idx + 1, sign, dtype=int)
    else:
        f_idx = np.ones_like(freqs, dtype=bool)

    return f_idx



def moving_average(a, n=3, iterations=1):
    a = np.array(a)
    if n%2==0:
        n += 1
    el = (n-1)//2 # edge_len

    for _ in range(iterations):
        a_padded = np.pad(a, (el, el), mode='reflect')
        ret = np.cumsum(a_padded, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        # a = np.concatenate((a[:el], ret[n - 1:] / n, a[-el:]))
        a = ret[n - 1:] / n

    return a

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

def round_dx(x, dx):
    return np.round(x / dx) * dx

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

def calculate_bandwidth(data_fd_1meas, noise_region_fraction = 0.20):
    freqs = data_fd_1meas[:, 0].real
    data_fd_1meas_db = to_db(data_fd_1meas, normalize=True)

    num_noise_samples = int(len(freqs) * noise_region_fraction)
    noise_floor_db = np.mean(data_fd_1meas_db[-num_noise_samples:, 1])

    valid_indices = np.where(data_fd_1meas_db[:, 1] < noise_floor_db + 6.0)[0]

    return {
        "bandwidth": freqs[valid_indices[0]],
        "snr": np.max(data_fd_1meas_db)-np.min(data_fd_1meas_db),
    }


def avg_data_array(data_arr):
    def _std(data_, **kwargs):
        if np.iscomplex(data_).any():
            a = np.std(data_.real, ddof=1, **kwargs)
            b = np.std(data_.imag, ddof=1, **kwargs)
            return a + 1j * b
        else:
            return np.std(data_, ddof=1, **kwargs)

    if data_arr.ndim < 2:
        return data_arr
    elif data_arr.ndim == 2:
        avg_std = np.mean(data_arr, axis=0)
        avg_std[:, 2] = _std(data_arr[:, 1], axis=0)
        return avg_std
    elif data_arr.ndim == 3: # [meas0, ..., meas_n][[x0, y0], ..., [xn, yn]]
        avg_std = np.mean(data_arr, axis=0)
        avg_std[:, 2] = _std(data_arr[:, :, 1], axis=0)
        return avg_std
    else: # [ref, sam][meas0, ..., meas_n][[x0, y0], ..., [xn, yn]]
        avg_std_ref = np.mean(data_arr[0], axis=0)
        avg_std_ref[:, :, 2] = _std(data_arr[0, :, :, 1], axis=0)

        avg_std_sam = np.mean(data_arr[1], axis=0)
        avg_std_sam[:, :, 2] = _std(data_arr[1, :, :, 1], axis=0)

        return np.array([avg_std_ref, avg_std_sam])




if __name__ == '__main__':
    arr = np.random.random((2, 12, 5, 3))
    barr = avg_data_array(arr)
    print(barr)
    print(arr.shape)
    print(barr.shape)