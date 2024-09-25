import itertools
import random
import re
import timeit
from functools import partial
from itertools import product
import matplotlib.pyplot as plt
import matplotlib as mpl
from .consts import plot_range1, plot_range, c_thz, plot_range2, c0
from numpy import array
from pathlib import Path
import numpy as np
import matplotlib.ticker as ticker
from .functions import do_fft, do_ifft, phase_correction, unwrap, window, polyfit, f_axis_idx_map, to_db, zero_pad, \
    peak_cnt, remove_spikes
from .measurements import get_all_measurements, MeasurementType
from .mpl_settings import mpl_style_params, fmt
from random import choice
# shgo = partial(shgo, workers=1)
from numpy import pi
from scipy.constants import epsilon_0
from teval.functions import phase_correction, window, do_fft, f_axis_idx_map, do_ifft, to_db
from teval.consts import c_thz, THz, plot_range1
from tmm import coh_tmm as coh_tmm_full
from scipy.optimize import shgo, minimize
from scipy.special import erfc
from enum import Enum

d_sub = 1000
angle_in = 0.0


def sub_refidx_a(img_, point=(22.5, 5)):
    #img_.plot_point(*point)
    #plt.show()

    sub_meas = img_.get_measurement(*point)
    sam_td = sub_meas.get_data_td()
    ref_td = img_.get_ref(point=point)

    sam_td = window(sam_td, en_plot=False, slope=0.99)
    ref_td = window(ref_td, en_plot=False, slope=0.99)

    ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

    freqs = ref_fd[:, 0].real
    omega = 2 * np.pi * freqs

    phi_ref = phase_correction(ref_fd, fit_range=(0.1, 1.2))
    phi_sam = phase_correction(sam_fd, fit_range=(0.1, 1.2))
    phi_diff = phi_sam[:, 1] - phi_ref[:, 1]

    n0 = 1 + c_thz * phi_diff / (omega * d_sub)
    k0 = -c_thz * np.log(np.abs(sam_fd[:, 1] / ref_fd[:, 1]) * (1 + n0) ** 2 / (4 * n0)) / (omega * d_sub)

    return np.array([freqs, n0 + 1j * k0], dtype=complex).T


def sub_refidx_tmm(img_, point, selected_freq_=None):
    initial_shgo_iters = 3
    freq_range = (0.25, 2.5)
    en_plot = False
    #img_.plot_point(*point)
    #plt.show()

    sub_meas = img_.get_measurement(*point)
    sam_td = sub_meas.get_data_td()
    ref_td = img_.get_ref(point=point)
    # ref_td = img_.get_ref()

    sam_td[:, 0] -= sam_td[0, 0]

    # sam_td = window(sam_td, en_plot=False, win_len=10)
    # ref_td = window(ref_td, en_plot=False, win_len=10)

    ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)
    phi = np.angle(sam_fd[:, 1] / ref_fd[:, 1])

    freqs = ref_fd[:, 0].real

    zero = np.zeros_like(freqs, dtype=complex)
    one = np.ones_like(freqs, dtype=complex)
    omega = 2 * np.pi * freqs

    if selected_freq_ is None:
        f_opt_idx = f_axis_idx_map(freqs, freq_range)
    else:
        f_opt_idx = list(f_axis_idx_map(freqs, selected_freq_))

    d_list = np.array([np.inf, d_sub, np.inf], dtype=float)

    phase_shift = np.exp(-1j * d_sub * omega / c_thz)

    # film_ref_interpol = self._ref_interpolation(measurement, selected_freq_=selected_freq_, ret_cart=True)

    def calc_model(n_model, ret_t=False, ret_T_and_R=False):
        n_list_ = np.array([one, n_model, one], dtype=complex).T

        R = np.zeros_like(freqs, dtype=complex)
        T = np.zeros_like(freqs, dtype=complex)
        ts_tmm_fd = np.zeros_like(freqs, dtype=complex)
        for f_idx_, freq_ in enumerate(freqs):
            if np.isclose(freq_, 0):
                continue
            lam_vac = c_thz / freq_
            n = n_list_[f_idx_]
            t_tmm_fd = coh_tmm_full("s", n, d_list, angle_in, lam_vac)["t"] * phase_shift[f_idx_]
            ts_tmm_fd[f_idx_] = t_tmm_fd
            if ret_T_and_R:
                dict_res = coh_tmm_full("s", n, d_list, angle_in, lam_vac)
                T[f_idx_] = dict_res["T"]
                R[f_idx_] = dict_res["R"]

        sam_tmm_fd_ = np.array([freqs, ts_tmm_fd * ref_fd[:, 1]]).T
        sam_tmm_td_ = do_ifft(sam_tmm_fd_)
        sam_tmm_td_[:, 0] -= sam_tmm_td_[0, 0]

        if ret_T_and_R:
            return T, R
        if ret_t:
            return ts_tmm_fd
        else:
            return sam_tmm_td_, sam_tmm_fd_

    def cost(p, freq_idx_):
        n = np.array([1, p[0] + 1j * p[1], 1], dtype=complex)
        # n = array([1, 1.9+1j*0.1, p[0] + 1j * p[1], 1])
        lam_vac = c_thz / freqs[freq_idx_]
        t_tmm_fd = coh_tmm_full("s", n, d_list, angle_in, lam_vac)["t"] * phase_shift[freq_idx_]

        sam_tmm_fd = t_tmm_fd * ref_fd[freq_idx_, 1]

        amp_loss = (np.abs(sam_tmm_fd) - np.abs(sam_fd[freq_idx_, 1])) ** 2
        phi_loss = (np.angle(t_tmm_fd) - phi[freq_idx_]) ** 2

        return amp_loss + phi_loss

    n_sub_a = sub_refidx_a(img_, point)

    res = None
    sigma, epsilon_r, n_opt = zero.copy(), zero.copy(), zero.copy()
    for f_idx_, freq in enumerate(freqs):
        if f_idx_ not in f_opt_idx:
            continue

        bounds_ = [(1.95, 2.00), (0.000, 0.010)]
        if freq > 1.7:
            bounds_ = [(1.965, 1.99), (0.007, 0.015)]
        n_sub_a_f_idx = n_sub_a[f_idx_, 1]
        bounds_ = [(n_sub_a_f_idx.real - 0.05, n_sub_a_f_idx.real + 0.05),
                   (n_sub_a_f_idx.imag - 0.005, n_sub_a_f_idx.imag + 0.005)]

        cost_ = cost
        if freq <= 0.150:
            res = shgo(cost_, bounds=bounds_, args=(f_idx_,), iters=initial_shgo_iters)
        elif freq <= 2.0:
            res = shgo(cost_, bounds=bounds_, args=(f_idx_,), iters=initial_shgo_iters)
            iters = initial_shgo_iters
            while res.fun > 1e-14:
                iters += 1
                res = shgo(cost_, bounds=bounds_, args=(f_idx_,), iters=iters)
                if iters >= initial_shgo_iters + 3:
                    break
        else:
            res = shgo(cost_, bounds=bounds_, args=(f_idx_,), iters=initial_shgo_iters)

        n_opt[f_idx_] = res.x[0] + 1j * res.x[1]
        epsilon_r[f_idx_] = n_opt[f_idx_] ** 2
        sigma[f_idx_] = 1j * (1 - epsilon_r[f_idx_]) * epsilon_0 * omega[f_idx_] * THz * 0.01  # "WORKS"
        # sigma[f_idx_] = 1j * (4 - epsilon_r[f_idx_]) * epsilon_0 * omega[f_idx_] * THz * 0.01  # 1/(Ohm cm)
        # sigma[f_idx_] = 1j * epsilon_r[f_idx_] * epsilon_0 * omega[f_idx_] * THz
        # sigma[f_idx_] = - 1j * epsilon_r[f_idx_] * epsilon_0 * omega[f_idx_] * THz
        print(f"Substrate result: {np.round(sigma[f_idx_], 1)} (S/cm), "
              f"n: {np.round(n_opt[f_idx_], 3)}, at {np.round(freqs[f_idx_], 3)} THz, "
              f"loss: {res.fun}")

    n_opt[:f_opt_idx[0]] = n_opt[f_opt_idx[0]] * np.ones_like(n_opt[:f_opt_idx[0]])
    n_opt[f_opt_idx[-1]:] = n_opt[f_opt_idx[-1]] * np.ones_like(n_opt[f_opt_idx[-1]:])

    if en_plot:
        sam_tmm_shgo_td, sam_tmm_shgo_fd = calc_model(n_opt)
        noise_floor = np.mean(20 * np.log10(np.abs(sam_fd[sam_fd[:, 0] > 6.0, 1])))
        plt.figure("Spectrum substrate")
        plt.title("Spectrum substrate")
        plt.plot(ref_fd[plot_range1, 0], to_db(ref_fd[plot_range1, 1]) - noise_floor, label="Reference")
        plt.plot(sam_fd[plot_range1, 0], to_db(sam_fd[plot_range1, 1]) - noise_floor, label="Substrate measurement")
        plt.plot(sam_tmm_shgo_fd[plot_range1, 0], to_db(sam_tmm_shgo_fd[plot_range1, 1]) - noise_floor,
                 label="TMM fit")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (dB)")

        t_tmm = calc_model(n_opt, ret_t=True)
        phi_tmm = np.angle(t_tmm)
        plt.figure("Phase substrate")
        plt.title("Phases substrate")
        plt.plot(freqs[plot_range1], phi[plot_range1], label="Substrate measurement", linewidth=2)
        plt.plot(freqs[plot_range1], phi_tmm[plot_range1], label="TMM", linewidth=2)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase difference (rad)")

        plt.figure("Time domain substrate")
        plt.plot(ref_td[:, 0], ref_td[:, 1], label="Ref Meas substrate", linewidth=2)
        plt.plot(sam_td[:, 0], sam_td[:, 1], label="Sam Meas substrate", linewidth=4)
        plt.plot(sam_tmm_shgo_td[:, 0], sam_tmm_shgo_td[:, 1], linewidth=2, label="TMM")

        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (arb. u.)")

    #n_opt.real[:-29] = np.convolve(n_opt.real, np.ones(30) / 30, mode='valid')
    #n_opt.imag[:-29] = np.convolve(n_opt.imag, np.ones(30) / 30, mode='valid')

    t_abs_meas = np.abs(sam_fd[:, 1] / ref_fd[:, 1])
    T, R = calc_model(n_opt, ret_T_and_R=True)
    t_abs = np.sqrt(T)
    t_abs = np.abs(calc_model(n_opt, ret_t=True))

    sigma_ret = sigma[f_opt_idx]
    epsilon_r_ret = epsilon_r[f_opt_idx]
    n = np.array([freqs, n_opt], dtype=complex).T
    t_abs = t_abs[f_opt_idx]
    R = R[f_opt_idx]
    t_abs_meas = t_abs_meas[f_opt_idx]

    eval_ret = {"n": n, "sam_fd": sam_fd, "ref_fd": ref_fd, "ref_td": ref_td, "sam_td": sam_td}

    return eval_ret


def conductivity(img_, measurement_, d_film_=None, selected_freq_=2.000, ref_idx=0,
                 tinkham_method=False, shift_sub=False, p2p_method=False):
    initial_shgo_iters = 3

    # sub_point = (49, -5)
    sub_point = (22, -5)
    sub_point = (40, -10)

    if shift_sub:
        from itertools import product
        from random import choice
        sub_points = list(product(range(21, 24), range(-14, 10)))
        sub_points.extend(list(product(range(46, 50), range(-14, 10))))
        sub_point = choice(sub_points)

        sub_point = (49, -5)
        sub_point = (22, -5)
        sub_point = (40, -10)

        print(f"New sub point: {[sub_point[0], sub_point[1]]}")

    if "sample3" in str(img_.data_path):
        d_film = 0.350
    elif "sample4" in str(img_.data_path):
        d_film = 0.250
    else:
        d_film = d_film_

    sub_eval_res = sub_refidx_tmm(img_, point=sub_point, selected_freq_=selected_freq_)
    n_sub = sub_eval_res["n"]
    sub_ref_fd = sub_eval_res["ref_fd"]
    sub_sam_fd = sub_eval_res["sam_fd"]
    sub_ref_td = sub_eval_res["ref_td"]
    sub_sam_td = sub_eval_res["sam_td"]

    shgo_bounds = [(1, 75), (1, 75)]

    if isinstance(measurement_, tuple):
        measurement_ = img_.get_measurement(*measurement_)

    film_td = measurement_.get_data_td()
    meas_pos = measurement_.position

    shifts = [[-4, 0], [-2, 0], [2, 0], [4, 0]]
    if ref_idx:
        try:
            shift = shifts[ref_idx]
        except IndexError:
            shift = [ref_idx, 0]
        meas_pos = (meas_pos[0] + shift[0], meas_pos[1] + shift[1])

    film_ref_td = img_.get_ref(both=False, point=meas_pos)
    # film_ref_td = img_.get_ref(both=False)

    # film_td = window(film_td, win_len=16, shift=0, en_plot=False, slope=0.99)
    # film_ref_td = window(film_ref_td, win_len=16, shift=0, en_plot=False, slope=0.99)

    pos_x = (measurement_.position[0] < 25) or (45 < measurement_.position[0])
    pos_y = (measurement_.position[1] < -11) or (9 < measurement_.position[1])
    if (np.max(film_td[:, 1]) / np.max(film_ref_td[:, 1]) > 0.50) or (pos_x and pos_y):
        return 1000, 0

    film_td[:, 0] -= film_td[0, 0]
    film_ref_td[:, 0] -= film_ref_td[0, 0]

    film_ref_fd, film_fd = do_fft(film_ref_td), do_fft(film_td)

    # film_ref_fd = phase_correction(film_ref_fd, fit_range=(0.1, 0.2), ret_fd=True, en_plot=False)
    # film_fd = phase_correction(film_fd, fit_range=(0.1, 0.2), ret_fd=True, en_plot=False)

    # phi = self.get_phase(point)
    phi = np.angle(film_fd[:, 1] / film_ref_fd[:, 1])

    freqs = film_ref_fd[:, 0].real
    zero = np.zeros_like(freqs, dtype=complex)
    one = np.ones_like(freqs, dtype=complex)
    omega = 2 * pi * freqs

    f_opt_idx = f_axis_idx_map(freqs, selected_freq_)

    d_list = np.array([np.inf, d_sub, d_film, np.inf], dtype=float)

    phase_shift = np.exp(-1j * (d_sub + np.sum(d_film)) * omega / c_thz)

    # film_ref_interpol = self._ref_interpolation(measurement, selected_freq_=selected_freq_, ret_cart=True)

    def tinkham_approx():
        T_sub = sub_sam_fd[:, 1] / sub_ref_fd[:, 1]
        T_sam = film_fd[:, 1] / film_ref_fd[:, 1]

        sigma_tink = 0.01 * (1 + n_sub[:, 1]) * epsilon_0 * c0 * (T_sub - T_sam) / (T_sam * d_film * 1e-6)
        sigma_tink = 1 / (sigma_tink * d_film * 1e-4)

        return np.array([sub_ref_fd[:, 0], sigma_tink], dtype=complex).T

    if tinkham_method:
        sigma = tinkham_approx()

        return 1 / (sigma[f_opt_idx[0], 1].real * d_film * 1e-4), 0

    def p2p_approx():
        T_sub = np.max(sub_sam_td[:, 1].real) / np.max(sub_ref_td[:, 1].real)
        T_sam = np.max(film_td[:, 1]) / np.max(film_ref_td[:, 1])

        sigma_tink = 0.01 * (1 + n_sub[:, 1]) * epsilon_0 * c0 * (T_sub - T_sam) / (T_sam * d_film * 1e-6)
        sigma_tink = 1 / (sigma_tink * d_film * 1e-4)

        return np.array([sub_ref_fd[:, 0], sigma_tink], dtype=complex).T

    if p2p_method:
        sigma = p2p_approx()

        return 0.01 / (sigma[f_opt_idx[0], 1].real * d_film * 1e-4), 0

    def cost(p, freq_idx_):
        n = np.array([1, n_sub[freq_idx_, 1], p[0] + 1j * p[1], 1], dtype=complex)
        # n = array([1, 1.9+1j*0.1, p[0] + 1j * p[1], 1])
        lam_vac = c_thz / freqs[freq_idx_]
        t_tmm_fd = coh_tmm_full("s", n, d_list, angle_in, lam_vac)["t"] * phase_shift[freq_idx_]

        sam_tmm_fd = t_tmm_fd * film_ref_fd[freq_idx_, 1]

        amp_loss = (np.abs(sam_tmm_fd) - np.abs(film_fd[freq_idx_, 1])) ** 2
        phi_loss = (np.angle(t_tmm_fd) - phi[freq_idx_]) ** 2

        return amp_loss + phi_loss

    res = None
    sigma, epsilon_r, n_opt = zero.copy(), zero.copy(), zero.copy()
    for f_idx_, freq in enumerate(freqs):
        if f_idx_ not in f_opt_idx:
            continue

        bounds_ = shgo_bounds

        cost_ = cost
        if freq <= 0.150:
            res = shgo(cost_, bounds=bounds_, args=(f_idx_,), iters=initial_shgo_iters)
        elif freq <= 2.0:
            res = shgo(cost_, bounds=bounds_, args=(f_idx_,), iters=initial_shgo_iters)
            iters = initial_shgo_iters
            while res.fun > 1e-14:
                iters += 1
                res = shgo(cost_, bounds=bounds_, args=(f_idx_,), iters=iters)
                if iters >= initial_shgo_iters + 3:
                    break
        else:
            res = shgo(cost_, bounds=bounds_, args=(f_idx_,), iters=initial_shgo_iters)

        n_opt[f_idx_] = res.x[0] + 1j * res.x[1]
        epsilon_r[f_idx_] = n_opt[f_idx_] ** 2
        sigma[f_idx_] = 1j * (1 - epsilon_r[f_idx_]) * epsilon_0 * omega[f_idx_] * THz * 0.01  # "WORKS"
        # sigma[f_idx_] = 1j * (4 - epsilon_r[f_idx_]) * epsilon_0 * omega[f_idx_] * THz * 0.01  # 1/(Ohm cm)
        # sigma[f_idx_] = 1j * epsilon_r[f_idx_] * epsilon_0 * omega[f_idx_] * THz
        # sigma[f_idx_] = - 1j * epsilon_r[f_idx_] * epsilon_0 * omega[f_idx_] * THz
        print(f"Result: {np.round(sigma[f_idx_], 1)} (S/cm), "
              f"n: {np.round(n_opt[f_idx_], 3)}, at {np.round(freqs[f_idx_], 3)} THz, "
              f"loss: {res.fun}")
        print(f"Substrate refractive index: {np.round(n_sub[f_idx_, 1], 3)}\n")

    # mean_cond = 0.5*(sigma[f_opt_idx[0]].real + sigma[f_opt_idx[0]].imag)

    return 1 / (sigma[f_opt_idx[0]].real * d_film * 1e-4), res.fun


class GridQuantity(Enum):
    P2P = 0
    Power = 1
    MeasTimeDeltaRef2Sam = 2
    RefAmp = 3
    RefArgmax = 4
    RefPhase = 5
    PeakCnt = 6
    TransmissionAmp = 7


class Image:
    plotted_ref = False
    noise_floor = None
    time_axis = None
    cache_path = None
    sample_idx = None
    all_points = None
    options = {}
    name = ""

    def __init__(self, data_path, sample_idx=None, options=None):
        self.data_path = data_path

        self.refs, self.sams, self.other = self._set_measurements()
        if sample_idx is not None:
            self.sample_idx = sample_idx

        self.image_info = self._set_info()
        self._set_options(options)
        self.image_data_td, self.image_data_fd = self._image_cache()
        self._evaluated_points = {}

    def _set_options(self, options_):
        if options_ is None:
            options_ = {}

        # set defaults if missing # TODO use default_dict ?
        if "excluded_areas" not in options_.keys():
            options_["excluded_areas"] = None
        if "one2onesub" not in options_.keys():
            options_["one2onesub"] = False

        if "cbar_min" not in options_.keys():
            options_["cbar_min"] = None
        if "cbar_max" not in options_.keys():
            options_["cbar_max"] = None

        if "log_scale" not in options_.keys():
            options_["log_scale"] = False

        if "color_map" not in options_.keys():
            # some options: ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
            options_["color_map"] = "autumn"

        if "invert_x" not in options_.keys():
            options_["invert_x"] = False
        if "invert_y" not in options_.keys():
            options_["invert_y"] = False

        if "en_window" not in options_.keys():
            options_["en_window"] = False

        if "selected_freq" not in options_.keys():
            options_["selected_freq"] = (1.0, 1.2)

        if "rcParams" not in options_.keys():
            options_["rcParams"] = {}

        self.options.update(options_)
        self._apply_options()

    def _apply_options(self):
        mpl.rcParams.update(self.options["rcParams"])

    def _set_measurements(self):
        all_measurements = get_all_measurements(data_dir_=self.data_path)
        refs, sams, other = self._filter_measurements(all_measurements)

        refs = tuple(sorted(refs, key=lambda meas: meas.meas_time))
        sams = tuple(sorted(sams, key=lambda meas: meas.meas_time))

        first_measurement = min(refs[0], sams[0], key=lambda meas: meas.meas_time)
        last_measurement = max(refs[-1], sams[-1], key=lambda meas: meas.meas_time)
        print(f"First measurement at: {first_measurement.meas_time}, last measurement: {last_measurement.meas_time}")
        time_del = last_measurement.meas_time - first_measurement.meas_time
        tot_hours, tot_mins = time_del.seconds // 3600, (time_del.seconds // 60) % 60
        print(f"Total measurement time: {tot_hours} hours and {tot_mins} minutes\n")

        return refs, sams, other

    def _find_refs(self, sample_measurements, ret_one=True):
        max_amp_meas = (None, -np.inf)
        for meas in sample_measurements:
            max_amp = np.max(meas.get_data_td())
            if max_amp > max_amp_meas[1]:
                max_amp_meas = (meas, max_amp)
        refs_ = [max_amp_meas[0]]
        print(f"Using reference measurement: {max_amp_meas[0].filepath.stem}")
        if not ret_one:
            for meas in sample_measurements:
                max_amp = np.max(meas.get_data_td())
                if max_amp > max_amp_meas[1] * 0.97:
                    refs_.append(meas)

        return refs_

    def _filter_measurements(self, measurements):
        refs, sams, other = [], [], []
        for measurement in measurements:
            if measurement.meas_type.value == MeasurementType.REF.value:
                refs.append(measurement)
            elif measurement.meas_type.value == MeasurementType.SAM.value:
                sams.append(measurement)
            else:
                other.append(measurement)

        if not [*refs, *sams, *other]:
            raise Exception("No measurements found. Check path or filenames")

        if not refs and sams:
            print("No references found. Using max amp. sample measurement")
            refs = self._find_refs(sams)

        return refs, sams, other

    def _set_info(self):
        if self.sample_idx is None:
            self.sample_idx = 0
        self.name = f"Sample {self.sample_idx}"

        sample_data_td = self.sams[0].get_data_td()
        samples = int(sample_data_td.shape[0])
        self.time_axis = sample_data_td[:, 0].real

        sample_data_fd = self.sams[0].get_data_fd()
        self.freq_axis = sample_data_fd[:, 0].real

        dt = np.mean(np.diff(self.time_axis))

        x_coords, y_coords = [], []
        for sam_measurement in self.sams:
            x_coords.append(sam_measurement.position[0])
            y_coords.append(sam_measurement.position[1])

        x_coords, y_coords = array(sorted(set(x_coords))), array(sorted(set(y_coords)))
        w, h = max(len(x_coords), 1), max(len(y_coords), 1)

        self.all_points = list(itertools.product(x_coords, y_coords))

        x_diff, y_diff = np.abs(np.diff(x_coords)), np.abs(np.diff(y_coords))

        if len(x_diff) > 0:
            x_diffs = x_diff[np.nonzero(x_diff)]
            # dx = np.mean(x_diffs)
            values, counts = np.unique(x_diffs, return_counts=True)
            dx = values[np.argmax(counts)]
        else:
            dx = 1

        if len(y_diff) > 0:
            y_diffs = y_diff[np.nonzero(y_diff)]
            # dy = np.mean(y_diffs)
            values, counts = np.unique(y_diffs, return_counts=True)
            dy = values[np.argmax(counts)]
        else:
            dy = 1

        dx, dy = round(dx, 3), round(dy, 3)

        extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

        self._empty_grid = np.zeros((w, h), dtype=complex)

        return {"w": w, "h": h, "dx": dx, "dy": dy, "dt": dt, "samples": samples, "extent": extent,
                "x_coords": x_coords, "y_coords": y_coords}

    def _coords_to_idx(self, x_, y_):
        x, y = self.image_info["x_coords"], self.image_info["y_coords"]
        x_idx, y_idx = np.argmin(np.abs(x_ - x)), np.argmin(np.abs(y_ - y))

        return x_idx, y_idx

    def _idx_to_coords(self, x_idx, y_idx):
        dx, dy = self.image_info["dx"], self.image_info["dy"]

        y = self.image_info["y_coords"][0] + y_idx * dy
        x = self.image_info["x_coords"][0] + x_idx * dx

        return x, y

    def _image_cache(self):
        """
        read all measurements into array and save as npy at location of first measurement
        """
        self.cache_path = Path(self.sams[0].filepath.parent / "cache")
        self.cache_path.mkdir(parents=True, exist_ok=True)

        try:
            img_data_td = np.load(str(self.cache_path / "_raw_img_td_cache.npy"))
            img_data_fd = np.load(str(self.cache_path / "_raw_img_fd_cache.npy"))
        except FileNotFoundError:
            w, h, samples = self.image_info["w"], self.image_info["h"], self.image_info["samples"]
            dx, dy = self.image_info["dx"], self.image_info["dy"]
            img_data_td = np.zeros((w, h, samples))
            fd_samples = (samples + 1) // 2 if (samples % 2) else 1 + samples // 2
            img_data_fd = np.zeros((w, h, fd_samples))
            min_x, max_x, min_y, max_y = self.image_info["extent"]

            for i, sam_measurement in enumerate(self.sams):
                if i % 100 == 0:
                    print(f"{round(100 * i / len(self.sams), 2)} % processed")

                x_pos, y_pos = sam_measurement.position
                x_idx, y_idx = int((x_pos - min_x) / dx), int((y_pos - min_y) / dy)
                img_data_td[x_idx, y_idx] = sam_measurement.get_data_td(get_raw=True)[:, 1]
                img_data_fd[x_idx, y_idx] = sam_measurement.get_data_fd()[:, 1]

            np.save(str(self.cache_path / "_raw_img_td_cache.npy"), img_data_td)
            np.save(str(self.cache_path / "_raw_img_fd_cache.npy"), img_data_fd)

        return img_data_td, img_data_fd

    def _is_excluded(self, idx_tuple):
        if self.options["excluded_areas"] is None:
            return False

        if np.array(self.options["excluded_areas"]).ndim == 1:
            areas = [self.options["excluded_areas"]]
        else:
            areas = self.options["excluded_areas"]

        for area in areas:
            x, y = self._idx_to_coords(*idx_tuple)
            return (area[0] <= x <= area[1]) * (area[2] <= y <= area[3])

        return False

    def _get_ref_pos(self, measurement_):
        ref_td = self.get_ref(point=measurement_.position)
        t, y = ref_td[:, 0], ref_td[:, 1]

        return t[np.argmax(y)]

    def _p2p(self):
        return np.max(self.image_data_td, axis=2) - np.min(self.image_data_td, axis=2)

    def _power(self, meas_, freq_range_):
        freq_slice = (freq_range_[0] < self.freq_axis) * (self.freq_axis < freq_range_[1])

        _, ref_fd = self.get_ref(point=meas_.position, both=True)
        sam_fd = meas_.get_data_fd()

        power_val_sam = np.sum(np.abs(sam_fd[freq_slice, 1])) / np.sum(freq_slice)
        power_val_ref = np.sum(np.abs(ref_fd[freq_slice, 1])) / np.sum(freq_slice)

        return (power_val_sam / power_val_ref) ** 2

    def _meas_time_delta(self, meas_):
        ref_meas = self.find_nearest_ref(meas_)

        return (meas_.meas_time - ref_meas.meas_time).total_seconds()

    def _ref_max(self, meas_, selected_freq_):
        amp_, _ = self._ref_interpolation(meas_, selected_freq_)

        return amp_

    def _ref_phase(self, meas_, selected_freq_):
        _, phi_ = self._ref_interpolation(meas_, selected_freq_)

        return phi_

    def _peak_cnt(self, meas_, threshold):
        data_td = meas_.get_data_td()
        y_ = data_td[:, 1]
        y_ -= (np.mean(y_[:10]) + np.mean(y_[-10:])) * 0.5

        y_[y_ < threshold] = 0
        peaks_idx = []
        for idx_ in range(1, len(y_) - 1):
            if (y_[idx_ - 1] < y_[idx_]) * (y_[idx_] > y_[idx_ + 1]):
                peaks_idx.append(idx_)

        return len(peaks_idx)

    def _amplitude_transmission(self, measurement_, selected_freq_):
        ref_td, ref_fd = self.get_ref(point=measurement_.position, both=True)
        freq_idx = f_axis_idx_map(ref_fd[:, 0].real, selected_freq_)

        sam_fd = measurement_.get_data_fd()
        power_val_sam = np.abs(sam_fd[freq_idx, 1])
        power_val_ref = np.abs(ref_fd[freq_idx, 1])

        return power_val_sam / power_val_ref

    def _calc_grid_vals(self, quantity=GridQuantity.P2P, freq_range=1.200):
        func_map = {GridQuantity.Power: partial(self._power, freq_range_=freq_range),
                    GridQuantity.MeasTimeDeltaRef2Sam: self._meas_time_delta,
                    GridQuantity.RefAmp: partial(self._ref_max, selected_freq_=freq_range),
                    GridQuantity.RefArgmax: self._get_ref_pos,
                    GridQuantity.RefPhase: self._ref_phase,
                    GridQuantity.PeakCnt: partial(self._peak_cnt, threshold=2.5),
                    GridQuantity.TransmissionAmp: partial(self._amplitude_transmission, selected_freq_=freq_range)
                    }

        if quantity == GridQuantity.P2P:
            return self._p2p()
        elif callable(quantity):
            func = quantity
        elif quantity in func_map:
            func = func_map[quantity]
        else:
            raise NotImplemented("quantity not implemented")

        grid_vals = self._empty_grid.copy()
        for i, measurement in enumerate(self.sams):
            print(f"{round(100 * i / len(self.sams), 2)} % done. "
                  f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")

            x_idx, y_idx = self._coords_to_idx(*measurement.position)

            grid_vals[x_idx, y_idx] = func(measurement)

        """
        elif quantity.lower() == "p2p_cond":
            grid_vals = self._empty_grid.copy()

            for i, measurement in enumerate(self.sams):
                print(f"{round(100 * i / len(self.sams), 2)} % done. "
                      f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")
                x_idx, y_idx = self._coords_to_idx(*measurement.position)
                sheet_resistance, _ = conductivity(self, measurement, p2p_method=True)
                grid_vals[x_idx, y_idx] = sheet_resistance
        elif quantity == "tinkham_cond":
            grid_vals = self._empty_grid.copy()

            for i, measurement in enumerate(self.sams):
                print(f"{round(100 * i / len(self.sams), 2)} % done. "
                      f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")
                x_idx, y_idx = self._coords_to_idx(*measurement.position)
                sheet_resistance, _ = conductivity(self, measurement, selected_freq_=selected_freq, tinkham_method=True)
                grid_vals[x_idx, y_idx] = sheet_resistance
        
        elif quantity.lower() == "conductivity":
            grid_vals = self._empty_grid.copy()

            for i, measurement in enumerate(self.sams):
                print(f"{round(100 * i / len(self.sams), 2)} % done. "
                      f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")
                x, y = measurement.position
                
                
                x_idx, y_idx = self._coords_to_idx(*measurement.position)
                sheet_resistance, err = conductivity(self, measurement, selected_freq_=selected_freq)

                retries = 1
                while (err > 1e-10) and (retries > 0):
                    print(f"Shifting ref. point (min. func. val: {err})")
                    sheet_resistance, err = conductivity(self, measurement, selected_freq_=selected_freq,
                                                         shift_sub=True)
                    retries -= 1


                #if sheet_resistance > self.options["cbar_max"]:
                #    sheet_resistance = self.options["cbar_max"]

                grid_vals[x_idx, y_idx] = sheet_resistance
        """

        return grid_vals.real

    def _exclude_pixels(self, grid_vals):
        filtered_grid = grid_vals.copy()
        dims = filtered_grid.shape
        for x_idx in range(dims[0]):
            for y_idx in range(dims[1]):
                if self._is_excluded((x_idx, y_idx)):
                    filtered_grid[x_idx, y_idx] = 0

        return filtered_grid

    def plot_image(self, selected_freq=None, quantity="p2p", img_extent=None):
        if selected_freq is None:
            selected_freq = self.options["selected_freq"]

        if quantity.lower() == "p2p":
            cbar_label = ""
        elif quantity.lower() == "p2p_cond":
            cbar_label = "p2p_cond"
        elif quantity.lower() == "tinkham_cond":
            cbar_label = f"Sheet resistance ($\Omega$/sq) @ {np.round(selected_freq, 3)} THz"
        elif quantity.lower() == "meas_time_delta":
            cbar_label = "Time to nearest ref. (s)"
        elif quantity.lower() == "ref_pos":
            cbar_label = "Ref. pulse pos. (ps)"
        elif quantity.lower() == "ref_amp":
            cbar_label = " Interpolated ref. amp. at " + str(np.round(selected_freq, 3)) + " THz"
        elif quantity == "Reference phase":
            cbar_label = " interpolated at " + str(np.round(selected_freq, 3)) + " THz"
        elif quantity.lower() == "power":
            cbar_label = f" ({selected_freq[0]}-{selected_freq[1]}) THz"
        elif quantity.lower() == "loss":
            cbar_label = " function value (log10)"
        elif quantity.lower() == "conductivity":
            cbar_label = f"Sheet resistance ($\Omega$/sq) @ {np.round(selected_freq, 3)} THz"
        elif quantity.lower() == "amplitude_transmission":
            cbar_label = f"Amplitude transmission @ {np.round(selected_freq, 2)} THz"
        elif quantity.lower() == "pulse_cnt":
            cbar_label = ""
        else:
            cbar_label = ""

        info = self.image_info
        if img_extent is None:
            w0, w1, h0, h1 = [0, info["w"], 0, info["h"]]
        else:
            dx, dy = info["dx"], info["dy"]
            w0, w1 = int((img_extent[0] - info["extent"][0]) / dx), int((img_extent[1] - info["extent"][0]) / dx)
            h0, h1 = int((img_extent[2] - info["extent"][2]) / dy), int((img_extent[3] - info["extent"][2]) / dy)

        grid_vals = self._calc_grid_vals(quantity=quantity, selected_freq=selected_freq)

        grid_vals = grid_vals[w0:w1, h0:h1]

        grid_vals = self._exclude_pixels(grid_vals)

        if self.options["log_scale"]:
            grid_vals = np.log10(grid_vals)

        fig = plt.figure(f"{self.name} {quantity}")
        ax = fig.add_subplot(111)
        ax.set_title(f"{self.name}")
        fig.subplots_adjust(left=0.2)

        if img_extent is None:
            img_extent = self.image_info["extent"]

        if self.options["cbar_min"] is None:
            cbar_min_val = np.min(grid_vals)
        else:
            cbar_min_val = self.options["cbar_min"]

        if self.options["cbar_max"] is None:
            cbar_max_val = np.max(grid_vals)
        else:
            cbar_max_val = self.options["cbar_max"]

        if self.options["log_scale"]:
            self.options["cbar_min"] = np.log10(cbar_min_val)
            self.options["cbar_max"] = np.log10(cbar_max_val)

        axes_extent = [img_extent[0] - self.image_info["dx"] / 2, img_extent[1] + self.image_info["dx"] / 2,
                       img_extent[2] - self.image_info["dy"] / 2, img_extent[3] + self.image_info["dy"] / 2]
        img = ax.imshow(grid_vals.transpose((1, 0)),
                        vmin=cbar_min_val, vmax=cbar_max_val,
                        origin="lower",
                        cmap=plt.get_cmap(self.options["color_map"]),
                        extent=axes_extent,
                        interpolation="hanning"
                        )
        if self.options["invert_x"]:
            ax.invert_xaxis()
        if self.options["invert_y"]:
            ax.invert_yaxis()

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        """
        if np.max(grid_vals) > 1000:
            cbar = fig.colorbar(img, format=ticker.FuncFormatter(fmt))
        else:
        """
        cbar = fig.colorbar(img)
        cbar.set_label(cbar_label, rotation=270, labelpad=30)

        """ # TODO separate method
        x_center, y_center = 0, 0
        if self.sample_idx == 3:
            y_center = -1.0
            x_center = 35.0
        if self.sample_idx == 4:
            y_center = -1.0
            x_center = 35.0

        x_coords, y_coords = self.image_info["x_coords"], self.image_info["y_coords"]
        x_idx, y_idx = self._coords_to_idx(x_center, y_center)

        plt.figure("x-slice")
        plt.plot(x_coords, grid_vals[:, x_idx], label=f"Sample {self.sample_idx} ({np.round(selected_freq, 3)} THz)")
        plt.xlabel("x (mm)")
        plt.ylabel(cbar_label)

        plt.figure("y-slice")
        plt.plot(y_coords, grid_vals[y_idx, :], label=f"Sample {self.sample_idx} ({np.round(selected_freq, 3)} THz)")
        plt.xlabel("y (mm)")
        plt.ylabel(cbar_label)
        """

    def get_measurement(self, x, y, meas_type=MeasurementType.SAM.value):
        if meas_type == MeasurementType.REF.value:
            meas_list = self.refs
        elif meas_type == MeasurementType.SAM.value:
            meas_list = self.sams
        else:
            meas_list = self.other

        closest_meas, best_fit_val = None, np.inf
        for meas in meas_list:
            val = abs(meas.position[0] - x) + \
                  abs(meas.position[1] - y)
            if val < best_fit_val:
                best_fit_val = val
                closest_meas = meas

        return closest_meas

    def get_point(self, x, y, normalize=False, sub_offset=False, both=False, add_plot=False):
        dx, dy, dt = self.image_info["dx"], self.image_info["dy"], self.image_info["dt"]

        x_idx, y_idx = self._coords_to_idx(x, y)
        y_ = self.image_data_td[x_idx, y_idx]

        if sub_offset:
            y_ -= (np.mean(y_[:10]) + np.mean(y_[-10:])) * 0.5

        if normalize:
            y_ *= 1 / np.max(y_)

        t = np.arange(0, len(y_)) * dt
        y_td = np.array([t, y_]).T

        if add_plot:
            self.plot_point(x, y, y_td)

        if not both:
            return y_td
        else:
            return y_td, do_fft(y_td)

    def find_nearest_ref(self, meas_):
        closest_ref, best_fit_val = None, np.inf
        for ref_meas in self.refs:
            dt = (meas_.meas_time - ref_meas.meas_time).total_seconds()
            if np.abs(dt) < np.abs(best_fit_val):
                best_fit_val = dt
                closest_ref = ref_meas

        print(f"Time between ref and sample: {best_fit_val} seconds")

        return closest_ref

    def get_ref(self, both=False, normalize=False, sub_offset=False, point=None, ret_meas=False):
        if point is not None:
            closest_sam = self.get_measurement(*point, meas_type=MeasurementType.SAM.value)

            chosen_ref = self.find_nearest_ref(closest_sam)
        else:
            chosen_ref = self.refs[-1]

        ref_td = chosen_ref.get_data_td()

        if sub_offset:
            ref_td[:, 1] -= (np.mean(ref_td[:10, 1]) + np.mean(ref_td[-10:, 1])) * 0.5

        if normalize:
            ref_td[:, 1] *= 1 / np.max(ref_td[:, 1])

        ref_td[:, 0] -= ref_td[0, 0]

        if ret_meas:
            return chosen_ref

        if both:
            ref_fd = do_fft(ref_td)
            return ref_td, ref_fd
        else:
            return ref_td

    def plot_point(self, x, y, sam_td=None, ref_td=None, sub_noise_floor=False, label="", td_scale=1):
        if (sam_td is None) and (ref_td is None):
            sam_td = self.get_point(x, y, sub_offset=True)
            ref_td = self.get_ref(sub_offset=True, point=(x, y))

            if self.options["en_window"]:
                sam_td = window(sam_td, win_len=25, shift=0, en_plot=False, slope=0.05)
                ref_td = window(ref_td, win_len=25, shift=0, en_plot=False, slope=0.05)

            ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

            # sam_td, sam_fd = phase_correction(sam_fd, fit_range=(0.8, 1.6), extrapolate=True, both=True)
            # ref_td, ref_fd = phase_correction(ref_fd, fit_range=(0.8, 1.6), extrapolate=True, both=True)

        else:
            ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

        phi_ref, phi_sam = unwrap(ref_fd), unwrap(sam_fd)

        noise_floor = np.mean(20 * np.log10(np.abs(ref_fd[ref_fd[:, 0] > 6.0, 1]))) * sub_noise_floor

        if not self.plotted_ref:
            plt.figure("Spectrum")
            plt.plot(ref_fd[plot_range1, 0], 20 * np.log10(np.abs(ref_fd[plot_range1, 1])) - noise_floor,
                     label="Reference")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Amplitude (dB)")

            plt.figure("Phase")
            plt.plot(ref_fd[plot_range1, 0], phi_ref[plot_range1, 1], label="Reference")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Phase (rad)")

            plt.figure("Time domain")
            plt.plot(ref_td[:, 0], ref_td[:, 1], label="Reference")
            plt.xlabel("Time (ps)")
            plt.ylabel("Amplitude (Arb. u.)")

            self.plotted_ref = True

        if not label:
            label += f" (x={x} (mm), y={y} (mm))"
        noise_floor = np.mean(20 * np.log10(np.abs(sam_fd[sam_fd[:, 0] > 6.0, 1]))) * sub_noise_floor

        plt.figure("Spectrum")
        plt.plot(sam_fd[plot_range1, 0], 20 * np.log10(np.abs(sam_fd[plot_range1, 1])) - noise_floor, label=label)

        plt.figure("Phase")
        plt.plot(sam_fd[plot_range1, 0], phi_sam[plot_range1, 1], label=label)

        plt.figure("Time domain")
        td_label = label
        if not np.isclose(td_scale, 1):
            td_label += f"\n(Amplitude x {td_scale})"
        plt.plot(sam_td[:, 0], td_scale * sam_td[:, 1], label=td_label)

        if not plt.fignum_exists("Amplitude transmission"):
            plt.figure("Amplitude transmission")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Amplitude transmission (%)")
        else:
            plt.figure("Amplitude transmission")
        absorb = np.abs(sam_fd[plot_range1, 1] / ref_fd[plot_range1, 1])
        plt.plot(sam_fd[plot_range1, 0], 100 * absorb, label=label)

        plt.figure("Absorbance")
        plt.plot(sam_fd[plot_range1, 0], -20 * np.log10(absorb), label=label)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Absorbance (dB)")

    def evaluate_point(self, point, d, label=None, en_plot=False):
        """
        evaluate and plot n, alpha and absorbance

        """
        sam_td, sam_fd = self.get_point(*point, both=True)
        ref_td, ref_fd = self.get_ref(point=point, both=True)

        omega = 2 * np.pi * ref_fd[:, 0].real

        phi_sam = phase_correction(sam_fd, en_plot=True)
        phi_ref = phase_correction(ref_fd, en_plot=True)

        phi = phi_sam[:, 1] - phi_ref[:, 1]

        n = 1 + phi * c_thz / (omega * d)

        alpha = (1 / 1e-4) * (-2 / d) * np.log(np.abs(sam_fd[:, 1] / ref_fd[:, 1]) * (n + 1) ** 2 / (4 * n))

        if en_plot:
            freq = ref_fd[:, 0].real
            plt.figure("Refractive index")
            plt.plot(freq[plot_range2], n[plot_range2], label=label)
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Refractive index")

            plt.figure("Absorption coefficient")
            plt.plot(freq[plot_range2], alpha[plot_range2], label=label)
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Absorption coefficient (1/cm)")

        return array([ref_fd[:, 0].real, n]).T, array([ref_fd[:, 0].real, alpha]).T

    def system_stability(self, selected_freq_=0.800):
        f_idx = np.argmin(np.abs(self.freq_axis - selected_freq_))

        ref_ampl_arr, ref_angle_arr, ref_pos = [], [], []

        t0 = self.refs[0].meas_time
        meas_times = [(ref.meas_time - t0).total_seconds() / 3600 for ref in self.refs]
        for i, ref in enumerate(self.refs):
            ref_td = ref.get_data_td()
            t, y = ref_td[:, 0], ref_td[:, 1]
            # ref_td = window(ref_td, win_len=12, shift=0, en_plot=False, slope=0.05)
            ref_fd = do_fft(ref_td)
            # ref_fd = phase_correction(ref_fd, fit_range=(0.8, 1.6), extrapolate=True, ret_fd=True, en_plot=False)

            ref_pos.append(t[np.argmax(y)])
            ref_ampl_arr.append(np.sum(np.abs(ref_fd[f_idx, 1])) / 1)
            phi = np.angle(ref_fd[f_idx, 1])
            """ ???
            if i and (abs(ref_angle_arr[-1] - phi) > pi):
                phi -= 2 * pi
            """
            ref_angle_arr.append(phi)
        ref_angle_arr = np.unwrap(ref_angle_arr)
        #ref_angle_arr -= np.mean(ref_angle_arr)
        #ref_ampl_arr -= np.mean(ref_ampl_arr)

        random.seed(10)
        rnd_sam = random.choice(self.sams)
        position1 = (19, 4)
        position2 = (20, 4)
        sam1 = self.get_measurement(*position1)  # rnd_sam
        sam2 = self.get_measurement(*position2)  # rnd_sam

        sam_t1 = (sam1.meas_time - t0).total_seconds() / 3600
        amp_interpol1, phi_interpol1 = self._ref_interpolation(sam1, ret_cart=False, selected_freq_=selected_freq_)

        sam_t2 = (sam2.meas_time - t0).total_seconds() / 3600
        amp_interpol2, phi_interpol2 = self._ref_interpolation(sam2, ret_cart=False, selected_freq_=selected_freq_)

        plt.figure("System stability reference pulse position")
        plt.title(f"Reference pulse position")
        plt.plot(meas_times, ref_pos, label=t0)
        plt.xlabel("Measurement time (hour)")
        plt.ylabel("Time (ps)")

        plt.figure("System stability amplitude")
        plt.title(f"Reference amplitude at {selected_freq_} THz")
        plt.plot(meas_times, ref_ampl_arr, label=t0)
        # plt.plot(sam_t1, amp_interpol1, marker="o", markersize=5, label=f"Interpol (x={position1[0]}, y={position1[1]}) mm")
        # plt.plot(sam_t2, amp_interpol2, marker="o", markersize=5, label=f"Interpol (x={position2[0]}, y={position2[1]}) mm")
        plt.xlabel("Measurement time (hour)")
        plt.ylabel("Amplitude (Arb. u.)")

        plt.figure("System stability angle")
        plt.title(f"Reference phase at {selected_freq_} THz")
        plt.plot(meas_times, ref_angle_arr, label=t0)
        # plt.plot(sam_t1, phi_interpol1, marker="o", markersize=5, label=f"Interpol (x={position1[0]}, y={position1[1]}) mm")
        # plt.plot(sam_t2, phi_interpol2, marker="o", markersize=5, label=f"Interpol (x={position2[0]}, y={position2[1]}) mm")
        plt.xlabel("Measurement time (hour)")
        plt.ylabel("Phase (rad)")

    def _ref_interpolation(self, sam_meas, selected_freq_, ret_cart=False):
        sam_meas_time = sam_meas.meas_time

        nearest_ref_idx, smallest_time_diff, time_diff = None, np.inf, 0
        for ref_idx in range(len(self.refs)):
            time_diff = (self.refs[ref_idx].meas_time - sam_meas_time).total_seconds()
            if abs(time_diff) < abs(smallest_time_diff):
                nearest_ref_idx = ref_idx
                smallest_time_diff = time_diff

        t0 = self.refs[0].meas_time
        if smallest_time_diff <= 0:
            # sample was measured after reference
            ref_before = self.refs[nearest_ref_idx]
            ref_after = self.refs[nearest_ref_idx + 1]
        else:
            ref_before = self.refs[nearest_ref_idx - 1]
            ref_after = self.refs[nearest_ref_idx]

        t = [(ref_before.meas_time - t0).total_seconds(), (ref_after.meas_time - t0).total_seconds()]
        ref_before_td, ref_after_td = ref_before.get_data_td(), ref_after.get_data_td()

        ref_before_fd, ref_after_fd = do_fft(ref_before_td), do_fft(ref_after_td)

        f_idx = np.argmin(np.abs(self.freq_axis - selected_freq_))
        y_amp = [np.sum(np.abs(ref_before_fd[f_idx, 1])) / 1,
                 np.sum(np.abs(ref_after_fd[f_idx, 1])) / 1]
        y_phi = [np.angle(ref_before_fd[f_idx, 1]), np.angle(ref_after_fd[f_idx, 1])]

        amp_interpol = np.interp((sam_meas_time - t0).total_seconds(), t, y_amp)
        phi_interpol = np.interp((sam_meas_time - t0).total_seconds(), t, y_phi)

        if ret_cart:
            return amp_interpol * np.exp(1j * phi_interpol)
        else:
            return amp_interpol, phi_interpol

    def knife_edge(self, x=None, y=None, coord_slice=None):
        if x is None and y is None:
            return

        x_coords, y_coords = self.image_info["y_coords"], self.image_info["x_coords"]
        freq_range = self.options["selected_freq"]

        # vertical direction / slice
        if x is not None:
            vertical_measurements = [self.get_measurement(x, y_) for y_ in y_coords]
            pos_axis, vals = y_coords, [self._power(meas_, freq_range) for meas_ in vertical_measurements]
            plot_x_label = "y (mm)"
        else:  # horizontal direction / slice
            horizontal_measurements = [self.get_measurement(x_, y) for x_ in x_coords]
            pos_axis, vals = x_coords, [self._power(meas_, freq_range) for meas_ in horizontal_measurements]
            plot_x_label = "x (mm)"

        vals = np.array(vals)
        pos_axis = pos_axis[np.nonzero(vals)]
        vals = vals[np.nonzero(vals)]

        if coord_slice is not None:
            val_mask = (coord_slice[0] < pos_axis) * (pos_axis < coord_slice[1])
            vals = vals[val_mask]
            pos_axis = pos_axis[val_mask]

        def _model(p):
            p_max, p_offset, w, h0 = p
            vals_model_ = p_offset + 0.5 * p_max * erfc(np.sqrt(2) * (pos_axis - h0) / w)

            return vals_model_

        def _cost(p):
            return np.sum((vals - _model(p)) ** 2)

        plt.figure("Knife edge")
        plt.xlabel(plot_x_label)
        plt.ylabel(f"Power (arb. u.) summed over {freq_range[0]}-{freq_range[1]} THz")

        p0 = np.array([vals[0], 0.0, 0.5, 34.0])
        opt_res = minimize(_cost, p0,
                           options={"iters": 100, "maxiter": np.inf, "maxev": np.inf, "maxfev": 1e3, "disp": False, },
                           tol=-1)
        # opt_res = minimize(_cost, p0,)

        opt_res = shgo(_cost, bounds=([p0[0] - 1, p0[0] + 1],
                                      [p0[1], p0[1] + 0.01],
                                      [p0[2] - 0.4, p0[2] + 2.0],
                                      [p0[3] - 2, p0[3] + 2]))

        plt.scatter(pos_axis, vals, label="Measurement", s=30, c="red", zorder=3)
        plt.plot(pos_axis, _model(opt_res.x), label="Optimization result")
        plt.plot(pos_axis, _model(p0), label="Initial guess")

        s = "\n".join(["".join(s) for s in
                       zip(["$P_{max}$: ", "$P_{offset}$: ", "Beam radius: ", "$h_0$: "],
                           [str(np.round(np.abs(param), 2)) for param in opt_res.x],
                           ["", "", " (mm)", " (mm)"])])
        plt.text(pos_axis[0], 0.04, s)

        return opt_res
