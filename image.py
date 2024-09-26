import itertools
import random
from functools import partial
import matplotlib.pyplot as plt
import matplotlib as mpl
from teval.consts import plot_range1, plot_range, c_thz, plot_range2, c0
from numpy import array
from pathlib import Path
import numpy as np
from teval.functions import unwrap, plt_show
from teval.measurements import get_all_measurements, MeasurementType
from teval.mpl_settings import mpl_style_params
from teval.functions import phase_correction, window, do_fft, f_axis_idx_map
from teval.consts import c_thz, THz, plot_range1
from scipy.optimize import shgo, minimize
from scipy.special import erfc
from enum import Enum
from measurements import Measurement


d_sub = 1000
angle_in = 0.0


class Quantity:
    func = None

    def __init__(self, label="label"):
        self.label = label

    def __repr__(self):
        return self.label

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class QuantityEnum(Enum):
    P2P = Quantity("P2p")
    Power = Quantity("Power")
    MeasTimeDeltaRef2Sam = Quantity("MeasTimeDeltaRef2Sam")
    RefAmp = Quantity("RefAmp")
    RefArgmax = Quantity("RefArgmax")
    RefPhase = Quantity("RefPhase")
    PeakCnt = Quantity("PeakCnt")
    TransmissionAmp = Quantity("TransmissionAmp")


class Image:
    plotted_ref = False
    noise_floor = None
    time_axis = None
    cache_path = None
    sample_idx = None
    all_points = None
    options = {}
    selected_freq = None
    selected_quantity = None
    name = ""

    def __init__(self, data_path, sample_idx=None, options=None):
        self.data_path = data_path

        self.refs, self.sams, self.other = self._set_measurements()
        if sample_idx is not None:
            self.sample_idx = sample_idx

        self.image_info = self._set_info()
        self._set_options(options)
        self._set_defaults()

        self.image_data_td, self.image_data_fd = self._image_cache()
        self._evaluated_points = {}

    def _set_defaults(self):
        if self.selected_freq is None:
            self.selected_freq = 1.000

        if self.selected_quantity is None:
            self.select_quantity(QuantityEnum.P2P)

    def _set_options(self, options_=None):
        if options_ is None:
            options_ = {}

        default_options = {"excluded_areas": None,
                           "one2onesub": False,
                           "cbar_lim": (None, None),
                           "log_scale": False,
                           "color_map": "autumn",
                           "invert_x": False, "invert_y": False,
                           "en_window": False,
                           "rcParams": mpl_style_params(),
                           }
        self.options.update(default_options)
        # some color_map options: ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

        self.options.update(options_)
        self._apply_options()

    def _apply_options(self):
        mpl.rcParams.update(self.options["rcParams"])

    def _set_measurements(self):
        if not isinstance(self.data_path, Path):
            self.data_path = Path(self.data_path)

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
        ref_td = self.get_ref_data(point=measurement_.position)
        t, y = ref_td[:, 0], ref_td[:, 1]

        return t[np.argmax(y)]

    def _p2p(self):
        return np.max(self.image_data_td, axis=2) - np.min(self.image_data_td, axis=2)

    def _power(self, meas_):
        if not isinstance(self.selected_freq, tuple):
            raise ValueError("selected_freq must be a tuple")

        freq_range_ = self.selected_freq
        freq_slice = (freq_range_[0] < self.freq_axis) * (self.freq_axis < freq_range_[1])

        _, ref_fd = self.get_ref_data(point=meas_.position, both=True)
        sam_fd = meas_.get_data_fd()

        power_val_sam = np.sum(np.abs(sam_fd[freq_slice, 1])) / np.sum(freq_slice)
        power_val_ref = np.sum(np.abs(ref_fd[freq_slice, 1])) / np.sum(freq_slice)

        return (power_val_sam / power_val_ref) ** 2

    def _meas_time_delta(self, meas_):
        ref_meas = self.find_nearest_ref(meas_)

        return (meas_.meas_time - ref_meas.meas_time).total_seconds()

    def _ref_max(self, meas_):
        amp_, _ = self._ref_interpolation(meas_, self.selected_freq)

        return amp_

    def _ref_phase(self, meas_):
        _, phi_ = self._ref_interpolation(meas_, self.selected_freq)

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

    def _amplitude_transmission(self, measurement_):
        ref_td, ref_fd = self.get_ref_data(point=measurement_.position, both=True)
        freq_idx = f_axis_idx_map(ref_fd[:, 0].real, self.selected_freq)

        sam_fd = measurement_.get_data_fd()
        power_val_sam = np.abs(sam_fd[freq_idx, 1])
        power_val_ref = np.abs(ref_fd[freq_idx, 1])

        return power_val_sam / power_val_ref

    def select_quantity(self, quantity):
        if isinstance(quantity, Quantity):
            self.selected_quantity = quantity
            return

        func_map = {QuantityEnum.P2P: lambda x: x,
                    QuantityEnum.Power: self._power,
                    QuantityEnum.MeasTimeDeltaRef2Sam: self._meas_time_delta,
                    QuantityEnum.RefAmp: self._ref_max,
                    QuantityEnum.RefArgmax: self._get_ref_pos,
                    QuantityEnum.RefPhase: self._ref_phase,
                    QuantityEnum.PeakCnt: partial(self._peak_cnt, threshold=2.5),
                    QuantityEnum.TransmissionAmp: self._amplitude_transmission,
                    }

        if quantity in func_map:
            quantity.value.func = func_map[quantity]
            self.selected_quantity = quantity.value

    def _calc_grid_vals(self):
        if self.selected_quantity == QuantityEnum.P2P:
            return np.real(self._p2p())

        grid_vals = self._empty_grid.copy()
        for i, measurement in enumerate(self.sams):
            print(f"{round(100 * i / len(self.sams), 2)} % done. "
                  f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")

            x_idx, y_idx = self._coords_to_idx(*measurement.position)

            grid_vals[x_idx, y_idx] = np.real(self.selected_quantity(measurement))

        return grid_vals.real

    def _exclude_pixels(self, grid_vals):
        filtered_grid = grid_vals.copy()
        dims = filtered_grid.shape
        for x_idx in range(dims[0]):
            for y_idx in range(dims[1]):
                if self._is_excluded((x_idx, y_idx)):
                    filtered_grid[x_idx, y_idx] = 0

        return filtered_grid

    def get_measurement(self, x, y, meas_type=MeasurementType.SAM.value) -> Measurement:
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

    def get_line(self, x=None, y=None):
        if x is None and y is None:
            return

        x_coords, y_coords = self.image_info["y_coords"], self.image_info["x_coords"]

        # vertical direction / slice
        if x is not None:
            return [self.get_measurement(x, y_) for y_ in y_coords], y_coords
        else:  # horizontal direction / slice
            return [self.get_measurement(x_, y) for x_ in x_coords], x_coords

    def find_nearest_ref(self, meas_):
        closest_ref, best_fit_val = None, np.inf
        for ref_meas in self.refs:
            dt = (meas_.meas_time - ref_meas.meas_time).total_seconds()
            if np.abs(dt) < np.abs(best_fit_val):
                best_fit_val = dt
                closest_ref = ref_meas

        print(f"Time between ref and sample: {best_fit_val} seconds")

        return closest_ref

    def get_ref_data(self, both=False, normalize=False, sub_offset=False, point=None, ret_meas=False):
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

    def evaluate_point(self, point, d, label=None, en_plot=False):
        """
        evaluate and plot n, alpha and absorbance

        """
        sam_meas = self.get_measurement(*point)
        sam_td, sam_fd = sam_meas.get_data_both_domains()
        ref_td, ref_fd = self.get_ref_data(point=point, both=True)

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

    def plot_point(self, x, y, sub_noise_floor=False, label="", td_scale=1):
        sam_meas = self.get_measurement(x, y)
        sam_td = sam_meas.get_data_td()

        ref_td = self.get_ref_data(sub_offset=True, point=(x, y))

        if self.options["en_window"]:
            sam_td = window(sam_td, win_len=25, shift=0, en_plot=False, slope=0.05)
            ref_td = window(ref_td, win_len=25, shift=0, en_plot=False, slope=0.05)

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

    def plot_system_stability(self):
        selected_freq_ = self.selected_freq
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

    def plot_image(self, img_extent=None):
        cbar_label = str(self.selected_quantity)

        info = self.image_info
        if img_extent is None:
            w0, w1, h0, h1 = [0, info["w"], 0, info["h"]]
        else:
            dx, dy = info["dx"], info["dy"]
            w0, w1 = int((img_extent[0] - info["extent"][0]) / dx), int((img_extent[1] - info["extent"][0]) / dx)
            h0, h1 = int((img_extent[2] - info["extent"][2]) / dy), int((img_extent[3] - info["extent"][2]) / dy)

        grid_vals = self._calc_grid_vals()

        grid_vals = grid_vals[w0:w1, h0:h1]

        grid_vals = self._exclude_pixels(grid_vals)

        if self.options["log_scale"]:
            grid_vals = np.log10(grid_vals)

        fig = plt.figure(f"{self.name} {self.selected_quantity}")
        ax = fig.add_subplot(111)
        ax.set_title(f"{self.name}")
        fig.subplots_adjust(left=0.2)

        if img_extent is None:
            img_extent = self.image_info["extent"]

        cbar_min, cbar_max = self.options["cbar_lim"]
        if cbar_min is None:
            cbar_min = np.min(grid_vals)
        if cbar_max is None:
            cbar_max = np.max(grid_vals)

        if self.options["log_scale"]:
            self.options["cbar_min"] = np.log10(cbar_min)
            self.options["cbar_max"] = np.log10(cbar_max)

        axes_extent = (float(img_extent[0] - self.image_info["dx"] / 2),
                       float(img_extent[1] + self.image_info["dx"] / 2),
                       float(img_extent[2] - self.image_info["dy"] / 2),
                       float(img_extent[3] + self.image_info["dy"] / 2))
        img = ax.imshow(grid_vals.transpose((1, 0)),
                        vmin=cbar_min, vmax=cbar_max,
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

    def plot_line(self, x=None, y=None):
        measurements, coords = self.get_line(x, y)

        vals = []
        for i, measurement in enumerate(measurements):
            print(f"{round(100 * i / len(measurements), 2)} % done. "
                  f"(Measurement: {i}/{len(measurements)}, {measurement.position} mm)")

            vals.append(np.real(self.selected_quantity.func(measurement)))

        label_ = f"Sample {self.sample_idx} ({np.round(self.selected_freq, 3)} THz)"

        plt.figure("x-slice")
        plt.plot(coords, vals, label=label_)
        plt.xlabel("x (mm)")
        plt.ylabel(str(self.selected_quantity))

        plt.figure("y-slice")
        plt.plot(coords, vals, label=label_)
        plt.xlabel("y (mm)")
        plt.ylabel(str(self.selected_quantity))

    def knife_edge(self, x=None, y=None, coord_slice=None):
        if not isinstance(self.selected_freq, tuple):
            raise ValueError("selected_freq must be a tuple")

        measurements, coords = self.get_line(x, y)
        vals = np.array([self._power(meas_) for meas_ in measurements])

        pos_axis = coords[np.nonzero(vals)]
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

        # vertical direction / slice
        if x is not None:
            plot_x_label = "y (mm)"
        else:  # horizontal direction / slice
            plot_x_label = "x (mm)"

        plt.figure("Knife edge")
        plt.xlabel(plot_x_label)
        plt.ylabel(f"Power (arb. u.) summed over {self.selected_freq[0]}-{self.selected_freq[1]} THz")

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


if __name__ == '__main__':
    img = Image(r"/home/ftpuser/ftp/Data/HHI_Aachen/remeasure_02_09_2024/sample3/img3")
    img.select_quantity(QuantityEnum.MeasTimeDeltaRef2Sam)
    img.plot_image()

    plt_show()
