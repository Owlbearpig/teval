import itertools
import random
from functools import partial
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import array
from pathlib import Path
import numpy as np
from teval.functions import unwrap, plt_show, remove_offset, window
from teval.measurements import MeasurementType, Measurement, Domain
from teval.mpl_settings import mpl_style_params
from teval.functions import phase_correction, do_fft, f_axis_idx_map
from teval.consts import c_thz, plot_range1, plot_range2
from scipy.optimize import shgo
from scipy.special import erfc
from enum import Enum
import logging
from datetime import datetime


class Quantity:
    func = None

    def __init__(self, label="label", func=None, domain=Domain.TimeDomain):
        self.label = label
        self.domain = domain
        if func is not None:
            self.func = func

    def __repr__(self):
        return self.label

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class QuantityEnum(Enum):
    P2P = Quantity("P2p")
    Power = Quantity("Power", domain=Domain.FrequencyDomain)
    MeasTimeDeltaRef2Sam = Quantity("MeasTimeDeltaRef2Sam")
    RefAmp = Quantity("RefAmp", domain=Domain.FrequencyDomain)
    RefArgmax = Quantity("RefArgmax")
    RefPhase = Quantity("RefPhase", domain=Domain.FrequencyDomain)
    PeakCnt = Quantity("PeakCnt")
    TransmissionAmp = Quantity("TransmissionAmp", domain=Domain.FrequencyDomain)


class Image:
    plotted_ref = False
    noise_floor = None
    time_axis = None
    cache_path = None
    sample_name = None
    all_points = None
    options = {}
    selected_freq = None
    selected_quantity = None
    grid_func = None
    measurements = None

    def __init__(self, data_path=None, options=None):
        if data_path is None:
            return

        self.data_path = data_path

        self.measurements, self.dataset_info = self._parse_measurements()

        self.grid_info = self._set_grid_properties()
        self._set_options(options)
        self._set_defaults()

        self.data_td, self.data_fd = self._cache()

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
                           "pixel_interpolation": None,
                           "rcParams": mpl_style_params(),
                           "sample_name": "",
                           "window_config": {},

                           }
        default_options.update(options_)
        # some color_map options: ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

        self.options.update(default_options)
        self._apply_options()

    def _apply_options(self):
        self.sample_name = self.options["sample_name"]
        mpl.rcParams.update(self.options["rcParams"])

    def _read_data_dir(self):
        glob = self.data_path.glob("**/*.txt")

        file_list = list(glob)

        measurements = []
        for i, file_path in enumerate(file_list):
            if file_path.is_file():
                try:
                    measurements.append(Measurement(filepath=file_path))
                except Exception as err:
                    if i == len(file_list) - 1:
                        logging.warning(f"No readable files found in {self.data_path}")
                        raise err
                    logging.info(f"Skipping {file_path}. {err}")

        return measurements

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
            logging.warning("No references found. Using max amp. sample measurement")
            refs = self._find_refs(sams)

        return refs, sams, other

    def _parse_measurements(self):
        if not isinstance(self.data_path, Path):
            self.data_path = Path(self.data_path)

        all_measurements = self._read_data_dir()
        refs, sams, other = self._filter_measurements(all_measurements)

        refs = tuple(sorted(refs, key=lambda meas: meas.meas_time))
        sams = tuple(sorted(sams, key=lambda meas: meas.meas_time))

        first_measurement = sorted(all_measurements, key=lambda meas: meas.meas_time)[0]
        last_measurement = sorted(all_measurements, key=lambda meas: meas.meas_time)[-1]
        logging.info(f"First measurement at: {first_measurement.meas_time}, "
                     f"last measurement: {last_measurement.meas_time}")
        time_del = last_measurement.meas_time - first_measurement.meas_time
        td_secs = time_del.seconds
        tot_hours, min_part = time_del.seconds // 3600, (td_secs // 60) % 60
        sec_part = time_del.seconds % 60

        logging.info(f"Total measurement time: {tot_hours} hours, "
                     f"{min_part} minutes and {sec_part} seconds ({td_secs} seconds)\n")

        info = {"id_map": dict(zip([id_.identifier for id_ in sorted(all_measurements, key=lambda x: x.identifier)],
                                   range(len(all_measurements))))}

        return {"refs": refs, "sams": sams, "other": other, "all": all_measurements}, info

    def _find_refs(self, sample_measurements, ret_one=True):
        max_amp_meas = (None, -np.inf)
        for meas in sample_measurements:
            data_td = self.get_meas_data(meas)
            max_amp = np.max(np.abs(data_td[:, 1]))
            if max_amp > max_amp_meas[1]:
                max_amp_meas = (meas, max_amp)
        refs_ = [max_amp_meas[0]]

        logging.debug(f"Using reference measurement: {max_amp_meas[0].filepath.stem}")

        if not ret_one:
            for meas in sample_measurements:
                data_td = self.get_meas_data(meas)
                max_amp = np.max(np.abs(data_td[:, 1]))
                if max_amp > max_amp_meas[1] * 0.97:
                    refs_.append(meas)

        return refs_

    def _set_grid_properties(self):
        sample_data_td = self.measurements["all"][0].get_data_td()
        samples = int(sample_data_td.shape[0])
        self.time_axis = sample_data_td[:, 0].real

        sample_data_fd = self.measurements["all"][0].get_data_fd()
        self.freq_axis = sample_data_fd[:, 0].real

        dt = np.mean(np.diff(self.time_axis))

        x_coords, y_coords = [], []
        for sam_measurement in self.measurements["sams"]:
            x_coords.append(sam_measurement.position[0])
            y_coords.append(sam_measurement.position[1])

        x_coords, y_coords = array(sorted(set(x_coords))), array(sorted(set(y_coords)))

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

        if not x_coords:
            x_coords = [0]
        if not y_coords:
            y_coords = [0]

        w = int(1 + np.ceil((max(x_coords) - min(x_coords)) / dx))
        h = int(1 + np.ceil((max(y_coords) - min(y_coords)) / dy))

        y_coords = np.round(np.arange(min(y_coords), max(y_coords) + dy, dy), 1)
        x_coords = np.round(np.arange(min(x_coords), max(x_coords) + dx, dx), 1)

        extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

        self._empty_grid = np.zeros((w, h), dtype=complex)

        return {"w": w, "h": h, "dx": dx, "dy": dy, "dt": dt, "samples": samples, "extent": extent,
                "x_coords": x_coords, "y_coords": y_coords}

    def _coords_to_idx(self, x_, y_):
        x, y = self.grid_info["x_coords"], self.grid_info["y_coords"]
        x_idx, y_idx = np.argmin(np.abs(x_ - x)), np.argmin(np.abs(y_ - y))

        return x_idx, y_idx

    def _idx_to_coords(self, x_idx, y_idx):
        dx, dy = self.grid_info["dx"], self.grid_info["dy"]

        y = self.grid_info["y_coords"][0] + y_idx * dy
        x = self.grid_info["x_coords"][0] + x_idx * dx

        return x, y

    def get_meas_data(self, meas, domain=Domain.TimeDomain):
        idx = self.dataset_info["id_map"][meas.identifier]
        meas_td, meas_fd = self.data_td[idx], self.data_fd[idx]

        if domain == Domain.TimeDomain:
            return meas_td
        elif domain == Domain.FrequencyDomain:
            return meas_fd
        else:
            return meas_td, meas_fd

    def _cache(self):
        self.cache_path = Path(self.measurements["all"][0].filepath.parent / "cache")
        self.cache_path.mkdir(parents=True, exist_ok=True)

        try:
            data_td = np.load(str(self.cache_path / "_td_cache.npy"))
            data_fd = np.load(str(self.cache_path / "_fd_cache.npy"))
        except FileNotFoundError:
            measurements = self.measurements["all"]

            y_td, y_fd = measurements[0].get_data_td(), measurements[0].get_data_fd()
            data_td = np.zeros((len(measurements), *y_td.shape), dtype=y_td.dtype)
            data_fd = np.zeros((len(measurements), *y_fd.shape), dtype=y_fd.dtype)

            for i, meas in enumerate(measurements):
                if i % 100 == 0:
                    logging.info(f"Reading files. {round(100 * i / len(measurements), 2)} %")
                idx = self.dataset_info["id_map"][meas.identifier]
                data_td[idx], data_fd[idx] = meas.get_data_td(), meas.get_data_fd()

            np.save(str(self.cache_path / "_td_cache.npy"), data_td)
            np.save(str(self.cache_path / "_fd_cache.npy"), data_fd)

        return data_td, data_fd

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

    def _p2p(self, meas_):
        y_td = self.get_meas_data(meas_)
        return np.max(y_td[:, 1]) - np.min(y_td[:, 1])

    def _power(self, meas_):
        if not isinstance(self.selected_freq, tuple):
            self.selected_freq = (1.0, 1.2)
            logging.warning(f"Selected_freq must be a tuple. Using default range ({self.selected_freq})")

        freq_range_ = self.selected_freq
        freq_slice = (freq_range_[0] < self.freq_axis) * (self.freq_axis < freq_range_[1])

        ref_fd = self.get_ref_data(point=meas_.position, domain=Domain.FrequencyDomain)
        sam_fd = meas_.get_data_fd()

        power_val_sam = np.sum(np.abs(sam_fd[freq_slice, 1])) / np.sum(freq_slice)
        power_val_ref = np.sum(np.abs(ref_fd[freq_slice, 1])) / np.sum(freq_slice)

        return (power_val_sam / power_val_ref) ** 2

    def _meas_time_delta(self, meas_):
        ref_meas = self.find_nearest_ref(meas_)

        return (meas_.meas_time - ref_meas.meas_time).total_seconds()

    def _ref_max(self, meas_):
        amp_, _ = self._ref_interpolation(meas_)

        return amp_

    def _ref_phase(self, meas_):
        _, phi_ = self._ref_interpolation(meas_)

        return phi_

    def _peak_cnt(self, meas_, threshold):
        data_td = self.get_meas_data(meas_)
        y_ = data_td[:, 1]
        y_ -= (np.mean(y_[:10]) + np.mean(y_[-10:])) * 0.5

        y_[y_ < threshold] = 0
        peaks_idx = []
        for idx_ in range(1, len(y_) - 1):
            if (y_[idx_ - 1] < y_[idx_]) * (y_[idx_] > y_[idx_ + 1]):
                peaks_idx.append(idx_)

        return len(peaks_idx)

    def _amplitude_transmission(self, measurement_):
        ref_td, ref_fd = self.get_ref_data(point=measurement_.position, domain=Domain.Both)
        freq_idx = f_axis_idx_map(ref_fd[:, 0].real, self.selected_freq)

        sam_fd = measurement_.get_data_fd()
        power_val_sam = np.abs(sam_fd[freq_idx, 1])
        power_val_ref = np.abs(ref_fd[freq_idx, 1])

        return power_val_sam / power_val_ref

    def _calc_grid_vals(self):
        grid_vals = self._empty_grid.copy()
        for i, measurement in enumerate(self.sams):
            if not i % (len(self.sams) // 100) or i == len(self.sams)-1:
                logging.info(f"{round(100 * i / len(self.sams), 2)} % done. "
                             f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")

            x_idx, y_idx = self._coords_to_idx(*measurement.position)

            grid_vals[x_idx, y_idx] = self.grid_func(measurement)

        return grid_vals.real

    def select_quantity(self, quantity, label=""):
        if isinstance(quantity, Quantity):
            if not callable(Quantity.func):
                logging.warning("Func of Quantity must be callable")
            self.grid_func = quantity.func
            self.selected_quantity = quantity

        if callable(quantity):
            self.grid_func = quantity
            self.selected_quantity = Quantity(label, func=quantity)

        func_map = {QuantityEnum.P2P: self._p2p,
                    QuantityEnum.Power: self._power,
                    QuantityEnum.MeasTimeDeltaRef2Sam: self._meas_time_delta,
                    QuantityEnum.RefAmp: self._ref_max,
                    QuantityEnum.RefArgmax: self._get_ref_pos,
                    QuantityEnum.RefPhase: self._ref_phase,
                    QuantityEnum.PeakCnt: partial(self._peak_cnt, threshold=2.5),
                    QuantityEnum.TransmissionAmp: self._amplitude_transmission,
                    }

        if quantity in func_map:
            self.grid_func = lambda x: np.real(func_map[quantity](x))
            self.selected_quantity = quantity.value

    def get_measurement(self, x, y, meas_type=MeasurementType.SAM.value) -> Measurement:
        if meas_type == MeasurementType.REF.value:
            meas_list = self.measurements["refs"]
        elif meas_type == MeasurementType.SAM.value:
            meas_list = self.measurements["sams"]
        else:
            meas_list = self.measurements["other"]

        closest_meas, best_fit_val = None, np.inf
        for meas in meas_list:
            val = abs(meas.position[0] - x) + \
                  abs(meas.position[1] - y)
            if val < best_fit_val:
                best_fit_val = val
                closest_meas = meas

        return closest_meas

    def _exclude_pixels(self, grid_vals):
        filtered_grid = grid_vals.copy()
        dims = filtered_grid.shape
        for x_idx in range(dims[0]):
            for y_idx in range(dims[1]):
                if self._is_excluded((x_idx, y_idx)):
                    filtered_grid[x_idx, y_idx] = 0

        return filtered_grid

    def get_line(self, x=None, y=None):
        if x is None and y is None:
            return

        x_coords, y_coords = self.grid_info["x_coords"], self.grid_info["y_coords"]

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

        logging.debug(f"Time between ref and sample: {best_fit_val} seconds")

        return closest_ref

    def get_ref_data(self, domain=Domain.TimeDomain, point=None):
        if point is not None:
            closest_sam = self.get_measurement(*point)
            chosen_ref = self.find_nearest_ref(closest_sam)
        else:
            chosen_ref = self.refs[-1]

        ref_td = self.get_meas_data(chosen_ref)
        ref_fd = self.get_meas_data(chosen_ref, domain=Domain.FrequencyDomain)

        if domain == Domain.TimeDomain:
            return ref_td
        elif domain == Domain.FrequencyDomain:
            return ref_fd
        else:
            return ref_td, ref_fd

    def get_ref_sam_meas(self, point):
        sam_meas = self.get_measurement(*point)
        ref_meas = self.find_nearest_ref(sam_meas)

        return ref_meas, sam_meas

    def evaluate_point(self, point, d, plot_label=None, en_plot=False):
        """
        evaluate and plot n, alpha and absorbance
        # d in um
        """
        if plot_label is None:
            plot_label = str(point)

        ref_meas, sam_meas = self.get_ref_sam_meas(point)

        ref_td, sam_td = ref_meas.get_data_td(), sam_meas.get_data_td()

        ref_td = remove_offset(ref_td)
        sam_td = remove_offset(sam_td)

        ref_td = window(ref_td, **self.options["window_config"])
        sam_td = window(sam_td, **self.options["window_config"])

        ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

        freq_axis = ref_fd[:, 0].real
        omega = 2 * np.pi * freq_axis

        phi_sam = phase_correction(sam_fd, en_plot=en_plot)
        phi_ref = phase_correction(ref_fd, en_plot=en_plot)

        phi = phi_sam[:, 1] - phi_ref[:, 1]

        n = 1 + phi * c_thz / (omega * d)
        kap = -c_thz * np.log(np.abs(sam_fd[:, 1] / ref_fd[:, 1]) * (1 + n) ** 2 / (4 * n)) / (omega * d)

        # 1/cm
        alph = (1 / 1e-4) * (-2 / d) * np.log(np.abs(sam_fd[:, 1] / ref_fd[:, 1]) * (n + 1) ** 2 / (4 * n))

        absorb = np.abs(sam_fd[:, 1] / ref_fd[:, 1])

        if en_plot:
            plt.figure("Refractive index")
            plt.plot(freq_axis[plot_range2], n[plot_range2], label=plot_label + " (Real)")
            plt.plot(freq_axis[plot_range2], kap[plot_range2], label=plot_label + " (Imag)")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Refractive index")

            plt.figure("Absorption coefficient")
            plt.plot(freq_axis[plot_range2], alph[plot_range2], label=plot_label)
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Absorption coefficient (1/cm)")

        res = {"n": array([freq_axis, n + 1j*kap]).T, "a": array([freq_axis, alph]).T,
               "absorb": array([freq_axis, absorb]).T}

        return res

    def _ref_interpolation(self, sam_meas, ret_cart=False):
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
        ref_before_td, ref_after_td = self.get_meas_data(ref_before), self.get_meas_data(ref_after)

        ref_before_fd, ref_after_fd = do_fft(ref_before_td), do_fft(ref_after_td)

        f_idx = np.argmin(np.abs(self.freq_axis - self.selected_freq))
        y_amp = [np.sum(np.abs(ref_before_fd[f_idx, 1])) / 1,
                 np.sum(np.abs(ref_after_fd[f_idx, 1])) / 1]
        y_phi = [np.angle(ref_before_fd[f_idx, 1]), np.angle(ref_after_fd[f_idx, 1])]

        amp_interpol = np.interp((sam_meas_time - t0).total_seconds(), t, y_amp)
        phi_interpol = np.interp((sam_meas_time - t0).total_seconds(), t, y_phi)

        if ret_cart:
            return amp_interpol * np.exp(1j * phi_interpol)
        else:
            return amp_interpol, phi_interpol

    def plot_point(self, point=None, **kwargs_):
        kwargs = {"label": "",
                  "sub_noise_floor": False,
                  "td_scale": 1,
                  "apply_window": False,
                  "remove_offset": False, }
        kwargs.update(kwargs_)

        label = kwargs["label"]
        sub_noise_floor = kwargs["sub_noise_floor"]
        td_scale = kwargs["td_scale"]
        apply_window = kwargs["apply_window"]
        remove_offset_ = kwargs["remove_offset"]

        if point is None:
            sam_meas = self.sams[0]
            point = sam_meas.position
        else:
            sam_meas = self.get_measurement(*point)
        ref_meas = self.find_nearest_ref(sam_meas)

        ref_td = self.get_meas_data(ref_meas)
        sam_td = self.get_meas_data(sam_meas)

        if remove_offset_:
            ref_td = remove_offset(ref_td)
            sam_td = remove_offset(sam_td)

        if apply_window:
            ref_td = window(ref_td, **self.options["window_config"])
            sam_td = window(sam_td, **self.options["window_config"])

        ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)
        freq_axis = ref_fd[:, 0].real

        phi_ref, phi_sam = unwrap(ref_fd), unwrap(sam_fd)

        noise_floor = np.mean(20 * np.log10(np.abs(ref_fd[ref_fd[:, 0] > 6.0, 1]))) * sub_noise_floor

        if not self.plotted_ref:
            y_db = (20 * np.log10(np.abs(ref_fd[plot_range1, 1])) - noise_floor).real
            plt.figure("Spectrum")
            plt.plot(freq_axis[plot_range1], y_db, label="Reference")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Amplitude (dB)")

            plt.figure("Phase")
            plt.plot(freq_axis[plot_range1], phi_ref[plot_range1, 1], label="Reference")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Phase (rad)")

            plt.figure("Time domain")
            plt.plot(ref_td[:, 0], ref_td[:, 1], label="Reference")
            plt.xlabel("Time (ps)")
            plt.ylabel("Amplitude (Arb. u.)")

            self.plotted_ref = True

        if not label:
            label += f" (x={point[0]} (mm), y={point[1]} (mm))"

        freq_axis = sam_fd[:, 0].real
        noise_floor = np.mean(20 * np.log10(np.abs(sam_fd[sam_fd[:, 0] > 6.0, 1]))) * sub_noise_floor

        plt.figure("Spectrum")
        y_db = (20 * np.log10(np.abs(sam_fd[plot_range1, 1])) - noise_floor).real
        plt.plot(freq_axis[plot_range1], y_db, label=label)

        plt.figure("Phase")
        plt.plot(freq_axis[plot_range1], phi_sam[plot_range1, 1], label=label)

        plt.figure("Time domain")
        td_label = label
        if not np.isclose(td_scale, 1):
            td_label += f"\n(Amplitude x {td_scale})"
        plt.plot(sam_td[:, 0], td_scale * sam_td[:, 1], label=td_label)

        if not plt.fignum_exists("Amplitude transmission"):
            plt.figure("Amplitude transmission")
            plt.xlabel("Frequency (THz)")
            plt.ylabel(r"Amplitude transmission ($\%$)")
        else:
            plt.figure("Amplitude transmission")
        absorb = np.abs(sam_fd[plot_range1, 1] / ref_fd[plot_range1, 1])
        plt.plot(freq_axis[plot_range1], 100 * absorb, label=label)

        plt.figure("Absorbance")
        plt.plot(freq_axis[plot_range1], -20 * np.log10(absorb), label=label)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Absorbance (dB)")

    def plot_system_stability(self):
        selected_freq_ = self.selected_freq
        f_idx = np.argmin(np.abs(self.freq_axis - selected_freq_))

        ref_ampl_arr, ref_angle_arr, ref_pos = [], [], []

        if not self.measurements["refs"]:
            meas_set = self.measurements["all"]
        else:
            meas_set = self.measurements["refs"]

        t0 = meas_set[0].meas_time
        meas_times = [(meas.meas_time - t0).total_seconds() / 3600 for meas in meas_set]
        for i, ref in enumerate(meas_set):
            ref_td = ref.get_data_td()
            t, y = ref_td[:, 0], ref_td[:, 1]
            ref_fd = do_fft(ref_td)

            ref_pos.append(t[np.argmax(y)])
            ref_ampl_arr.append(np.sum(np.abs(ref_fd[f_idx, 1])) / 1)
            phi = np.angle(ref_fd[f_idx, 1])

            ref_angle_arr.append(phi)
        ref_angle_arr = np.unwrap(ref_angle_arr)

        plt.figure("Stability ref pulse pos")
        plt.title(f"Reference pulse position")
        plt.plot(meas_times, ref_pos, label=t0)
        plt.xlabel("Measurement time (hour)")
        plt.ylabel("Time (ps)")

        plt.figure("Stability amplitude")
        plt.title(f"Amplitude of reference measurement at {selected_freq_} THz")
        plt.plot(meas_times, ref_ampl_arr, label=t0)
        plt.xlabel("Measurement time (hour)")
        plt.ylabel("Amplitude (Arb. u.)")

        plt.figure("Stability phase")
        plt.title(f"Phase of reference measurement at {selected_freq_} THz")
        plt.plot(meas_times, ref_angle_arr, label=t0)
        plt.xlabel("Measurement time (hour)")
        plt.ylabel("Phase (rad)")

    def plot_temperature(self, log_file):
        def read_log_file(log_file_):
            def read_line(line_):
                parts = line_.split(" ")
                t = datetime.strptime(f"{parts[0]} {parts[1]}", '%Y-%m-%d %H:%M:%S')
                return t, float(parts[4]), float(parts[-3])

            with open(log_file_) as file:
                meas_time_, temp_, humidity_ = [], [], []
                for i, line in enumerate(file):
                    if i % 250:
                        continue
                    try:
                        res = read_line(line)
                        meas_time_.append(res[0])
                        temp_.append(res[1])
                        humidity_.append(res[2])
                    except IndexError:
                        continue

            return meas_time_, temp_, humidity_

        meas_time, temp, humidity = read_log_file(log_file)

        if self.refs is not None:
            t0 = self.refs[0].meas_time
        else:
            t0 = meas_time[0]

        meas_time_diff = [(t - t0).total_seconds() / 3600 for t in meas_time]

        stability_figs = ["Stability ref pulse pos", "Stability amplitude", "Stability phase"]
        for fig_label in stability_figs:
            if plt.fignum_exists(fig_label):
                fig = plt.figure(fig_label)
                ax_list = fig.get_axes()
                ax1 = ax_list[0]
                ax1.tick_params(axis="y", colors="blue")
                ax1.set_ylabel(ax1.get_ylabel(), c="blue")
                ax1.grid(c="blue")

                ax2 = ax1.twinx()
                ax2.plot(meas_time_diff, temp, c="red")
                ax2.set_ylabel("Temperature (°C)", c="red")
                ax2.tick_params(axis="y", colors="red")
                ax2.grid(c="red")

        if not plt.fignum_exists(stability_figs[0]):
            fig, ax1 = plt.subplots(num="Temperature plot")
            ax1.plot(meas_time_diff, temp, label=f"Start: {t0}")
            ax1.set_xlabel("Measurement time (hour)")
            ax1.set_ylabel("Temperature (°C)")

    def plot_image(self, img_extent=None):
        info = self.grid_info
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

        fig_num = " ".join([str(self.sample_name), str(self.selected_quantity)])
        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111)
        ax.set_title(f"{self.sample_name}")
        fig.subplots_adjust(left=0.2)

        if img_extent is None:
            img_extent = self.grid_info["extent"]

        cbar_min, cbar_max = self.options["cbar_lim"]
        if cbar_min is None:
            cbar_min = np.min(grid_vals)
        if cbar_max is None:
            cbar_max = np.max(grid_vals)

        if self.options["log_scale"]:
            self.options["cbar_min"] = np.log10(cbar_min)
            self.options["cbar_max"] = np.log10(cbar_max)

        axes_extent = (float(img_extent[0] - self.grid_info["dx"] / 2),
                       float(img_extent[1] + self.grid_info["dx"] / 2),
                       float(img_extent[2] - self.grid_info["dy"] / 2),
                       float(img_extent[3] + self.grid_info["dy"] / 2))
        img_ = ax.imshow(grid_vals.transpose((1, 0)),
                         vmin=cbar_min, vmax=cbar_max,
                         origin="lower",
                         cmap=plt.get_cmap(self.options["color_map"]),
                         extent=axes_extent,
                         interpolation=self.options["pixel_interpolation"]
                         )
        if self.options["invert_x"]:
            ax.invert_xaxis()
        if self.options["invert_y"]:
            ax.invert_yaxis()

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        en_freq_label = Domain.FrequencyDomain == self.selected_quantity.domain
        if isinstance(self.selected_freq, tuple):
            freq_label = f"({self.selected_freq[0]}-{self.selected_freq[1]} THz)"
        else:
            freq_label = f"({self.selected_freq} THz)"

        cbar_label = " ".join([str(self.selected_quantity), freq_label * en_freq_label])

        cbar = fig.colorbar(img_)
        cbar.set_label(cbar_label, rotation=270, labelpad=30)

    def plot_line(self, x=None, y=None):
        measurements, coords = self.get_line(x, y)

        vals = []
        for i, measurement in enumerate(measurements):
            logging.info(f"{round(100 * i / len(measurements), 2)} % done. "
                         f"(Measurement: {i}/{len(measurements)}, {measurement.position} mm)")

            vals.append(self.grid_func(measurement))

        label_ = self.sample_name

        if y is not None:
            plt.figure("x-slice")
            plt.title(f"y={y} mm")
            plt.plot(coords, vals, label=label_)
            plt.xlabel("x (mm)")
            plt.ylabel(str(self.selected_quantity))

        else:
            plt.figure("y-slice")
            plt.title(f"x={x} mm")
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

        # vertical direction
        if x is not None:
            plot_x_label = "y (mm)"
        else:  # horizontal direction
            plot_x_label = "x (mm)"

        plt.figure("Knife edge")
        plt.xlabel(plot_x_label)
        plt.ylabel(f"Power (arb. u.) summed over {self.selected_freq[0]}-{self.selected_freq[1]} THz")

        p0 = np.array([vals[0], 0.0, 0.5, 34.0])
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
    from random import choice

    logging.basicConfig(level=logging.INFO)
    options = {}
    # img = Image(r"/home/ftpuser/ftp/Data/HHI_Aachen/remeasure_02_09_2024/sample3/img3", options)
    # img = Image(r"/home/ftpuser/ftp/Data/SemiconductorSamples/Wafer_25_and_wafer_19073", options)
    img = Image(r"E:\measurementdata\Stability\31-10-2024_L1\air")

    # img.select_quantity()
    # img.plot_image()
    # img.window_all()
    # all_meas = img.all_measurements(sort_key=lambda x: x.identifier)

    # point = choice(img.all_points)
    # img.window_all()
    # img.plot_point()
    # img.evaluate_point(point, 1000, en_plot=True)
    # img.selected_freq = 0.5
    img.plot_system_stability()

    plt_show(en_save=False)
