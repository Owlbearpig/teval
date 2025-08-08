import itertools
import os
import random
from functools import partial
import consts
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from numpy import array
from pathlib import Path
import numpy as np
from functions import unwrap, plt_show, window, local_minima_1d
from measurements import MeasurementType, Measurement, Domain
from mpl_settings import mpl_style_params
from functions import phase_correction, do_fft, f_axis_idx_map, remove_offset
from consts import c_thz, eps0
from scipy.optimize import shgo
from scipy.special import erfc
from enum import Enum
import logging
from datetime import datetime
from tqdm import tqdm


"""
TODO: 
How are measurements mapped when multiple measurements are performed at the same x-y coordinates? 

ideas: add teralyzer evaluation (time consuming)
- Add plt_show here
- interactive imshow plots

# units:
[l] = um, [t] = ps, [alpha] = 1/cm (absorption coeff.), [sigma] = S/cm, [eps0] = Siemens * second

"""

class PixelInterpolation(Enum):
    # imshow(interpolation=pixel_interpolation)
    none = None
    antialiased = 'antialiased'
    nearest = 'nearest'
    bilinear = 'bilinear'
    bicubic = 'bicubic'
    spline16 = 'spline16'
    spline36 = 'spline36'
    hanning = 'hanning'
    hamming = 'hamming'
    hermite = 'hermite'
    kaiser = 'kaiser'
    quadric = 'quadric'
    catrom = 'catrom'
    gaussian = 'gaussian'
    bessel = 'bessel'
    mitchell = 'mitchell'
    sinc = 'sinc'
    lanczos = 'lanczos'
    blackman = 'blackman'

class ClimateQuantity(Enum):
    Temperature = 0
    Humidity = 1


class Dist(Enum):
    Position = lambda meas1, meas2: (abs(meas1.position[0] - meas2.position[0]) +
                                     abs(meas1.position[1] - meas2.position[1]))
    Time = lambda meas1, meas2: (meas1.meas_time - meas2.meas_time).total_seconds()


class Direction(Enum):
    Horizontal = 0
    Vertical = 1


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
    P2P = Quantity("Peak to peak")
    Power = Quantity("Power", domain=Domain.FrequencyDomain)
    MeasTimeDeltaRef2Sam = Quantity("Time delta Ref. to Sam.")
    RefAmp = Quantity("Ref. Amp", domain=Domain.FrequencyDomain)
    RefArgmax = Quantity("Ref. Argmax")
    RefPhase = Quantity("Ref. Phase", domain=Domain.FrequencyDomain)
    PeakCnt = Quantity("Peak Cnt")
    TransmissionAmp = Quantity("Amplitude Transmission", domain=Domain.FrequencyDomain)
    RefractiveIdx = Quantity("Refractive idx", domain=Domain.FrequencyDomain)
    AbsorptionCoe = Quantity("Absorption coe", domain=Domain.FrequencyDomain)


class LogLevel(Enum):
    info = logging.INFO
    debug = logging.DEBUG
    warning = logging.WARNING
    error = logging.ERROR
    critical = logging.CRITICAL

class DataSet:
    def __init__(self, data_path=None, options_=None):
        self.plotted_ref = False
        self.noise_floor = None
        self.time_axis = None
        self.raw_data_cache = {}
        self.options = {}
        self.img_properties = {}
        self.selected_freq = None
        self.selected_quantity = None
        self.grid_func = None
        self.grid_vals = None
        self.measurements = {"refs": (), "sams": (), "all": ()}
        self.sub_dataset = None

        self.data_path = Path(data_path)
        self._check_path()

        self._set_options(options_)
        self._set_defaults()

        self._parse_measurements()

        self._set_img_properties()

        self._check_refs_exist()

    def _check_path(self):
        if self.data_path is None:
            raise ValueError("Path cannot be None")
        if not self.data_path.exists():
            raise ValueError(f"Path {self.data_path} does not exist")
        if not self.data_path.is_dir():
            raise ValueError(f"Path {self.data_path} is not a directory")
        if not list(self.data_path.glob("*")):
            raise ValueError(f"Path {self.data_path} is empty")


    def _set_defaults(self):
        if self.selected_freq is None:
            self.selected_freq = 1.000

        if self.selected_quantity is None:
            self.select_quantity(QuantityEnum.P2P)

    def _set_options(self, options_=None):
        if options_ is None:
            options_ = {}

        # some color_map options: ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        default_options = {"log_level": LogLevel.info,
                           "result_dir": consts.result_dir,
                           "excluded_areas": None,
                           "cbar_lim": (None, None),
                           "log_scale": False,
                           "color_map": "autumn",
                           "invert_x": False, "invert_y": False,
                           "pixel_interpolation": PixelInterpolation.none,
                           "rcParams": mpl_style_params(),
                           "fig_label": "",
                           "img_title": "",
                           "en_cbar_label": False,
                           "plot_range": slice(0, 900),
                           "ref_pos": (None, None),
                           "ref_threshold": 0.95,
                           "dist_func": Dist.Position,
                           "pp_opt": {
                               "window_opt": {"enabled": False, "win_width": None, "win_start": None,
                                              "shift": None, "en_plot": False, "slope": 0.15},
                               "remove_dc": True,
                               "dt": 0,
                                      },
                           "sample_properties": {"d": 1000, "layers": 1, "default_values": True},
                           }
        if "sample_properties" in options_:
            default_options["sample_properties"]["default_values"] = False

        def check_values(new_options, default):
            for k in default:
                if k not in new_options:
                    new_options[k] = default[k]
                elif isinstance(default[k], dict) and isinstance(new_options[k], dict):
                    check_values(new_options[k], default[k])
        check_values(options_, default_options)

        if not isinstance(options_["result_dir"], Path):
            options_["result_dir"] = Path(options_["result_dir"])

        self.options.update(options_)
        self._apply_options()

    def _apply_options(self):
        self.options["rcParams"]["savefig.directory"] = self.options["result_dir"]
        mpl.rcParams.update(self.options["rcParams"])

        logging.basicConfig(level=self.options["log_level"].value)

    def _read_data_dir(self):
        glob = self.data_path.glob("**/*.txt")

        file_list = list(glob)

        measurements = []
        for i, file_path in enumerate(file_list):
            if file_path.is_file() and ".txt" in file_path.name:
                try:
                    measurements.append(Measurement(filepath=file_path))
                except Exception as err:
                    if i == len(file_list) - 1:
                        logging.warning(f"No readable files found in {self.data_path}")
                        raise err
                    logging.info(f"Skipping {file_path}. {err}")

        return measurements

    def _sort_measurements(self, measurements):
        refs, sams = [], []
        for measurement in measurements:
            if measurement.meas_type.value == MeasurementType.REF.value:
                refs.append(measurement)
            else:
                sams.append(measurement)

        if not [*refs, *sams]:
            raise Exception("No measurements found. Check path or filenames")

        def sort(meas_list):
            if meas_list:
                return tuple(sorted(meas_list, key=lambda meas: meas.meas_time))
            else:
                return ()

        refs_sorted, sams_sorted, all_sorted = sort(refs), sort(sams), sort([*refs, *sams])

        self.measurements["refs"] = refs_sorted
        self.measurements["sams"] = sams_sorted
        self.measurements["all"] = all_sorted

    def _generate_coord_map(self, all_measurements_):
        # keys are of the format "xy" where x, y are the positions cut/0 filled to 3 dec. places
        self.raw_data_cache["coord_map"] = {}
        for meas in all_measurements_:
            key = "".join([f"{val:.3f}" for val in meas.position])
            self.raw_data_cache["coord_map"][key] = meas

    def _parse_measurements(self):
        if not isinstance(self.data_path, Path):
            self.data_path = Path(self.data_path)

        all_measurements = self._read_data_dir()

        self.raw_data_cache["id_map"] = dict(zip([id_.identifier for id_ in
                                                  sorted(all_measurements, key=lambda x: x.identifier)],
                                                 range(len(all_measurements))))

        self._fill_cache(all_measurements)

        self._sort_measurements(all_measurements)

        self._generate_coord_map(all_measurements)

        first_measurement = self.measurements["all"][0]
        last_measurement = self.measurements["all"][-1]

        logging.info(f"Dataset contains {len(all_measurements)} measurements")
        logging.info(f"First measurement at: {first_measurement.meas_time}, "
                     f"last measurement: {last_measurement.meas_time}")
        time_del = last_measurement.meas_time - first_measurement.meas_time
        td_secs = time_del.seconds + 24 * 3600 * time_del.days
        tot_hours = td_secs // 3600
        min_part = (td_secs // 60) % 60
        sec_part = time_del.seconds % 60

        logging.info(f"Total measurement time: {tot_hours} hours, "
                     f"{min_part} minutes and {sec_part} seconds ({td_secs} seconds)\n")

        time_diffs = [(self.measurements["all"][i + 1].meas_time -
                       self.measurements["all"][i].meas_time).total_seconds()
                      for i in range(0, len(self.measurements["all"]) - 1)]

        mean_time_diff = np.mean(time_diffs)

        logging.info(f"Mean time between measurements: {np.round(mean_time_diff, 2)} seconds")
        logging.info(f"Min and max time between measurements: "
                     f"({np.min(time_diffs)}, {np.max(time_diffs)}) seconds\n")

        all_meas = self.measurements["all"]
        max_amp_meas = (all_meas[0], -np.inf)
        for meas in all_meas:
            data_td = self.get_data(meas)
            max_amp = np.max(np.abs(data_td[:, 1]))
            if max_amp > max_amp_meas[1]:
                max_amp_meas = (meas, max_amp)
        logging.info(f"Maximum amplitude measurement: {max_amp_meas[0].filepath.name}\n")
        self.measurements["max_amp_meas"] = max_amp_meas[0]

    def _check_refs_exist(self):
        if self.measurements["refs"] or not self.measurements["sams"]:
            return
        logging.info(f"No explicit references in the dataset. Using ref_pos option")

        threshold = self.options["ref_threshold"]

        max_amp_meas = self.measurements["max_amp_meas"]
        max_amp = np.max(np.abs(self.get_data(max_amp_meas)[:, 1]))

        manual_pos = self.options["ref_pos"]

        refs_ = []
        if (manual_pos[0] is not None) and (manual_pos[1] is not None):
            refs_ = [self.get_measurement(*manual_pos)]
            logging.warning(f"Using measurement at {manual_pos} as ref.")
        elif not all([pos is None for pos in manual_pos]):
            if manual_pos[0] is None:
                y = manual_pos[1]
                logging.info(f"Selecting measurements along horizontal line at y={y} mm")
                ref_line, x_coords = self.get_line(y=y)
            else:
                x = manual_pos[0]
                logging.info(f"Selecting measurements along vertical line at x={x} mm")
                ref_line, y_coords = self.get_line(x=x)

            for meas in ref_line:
                data_td = self.get_data(meas)
                if np.max(np.abs(data_td[:, 1])) > threshold * max_amp:
                    refs_.append(meas)

        if len(refs_) > 1:
            logging.info(f"Using reference measurements: {refs_[0].filepath.stem} to {refs_[-1].filepath.stem}")

        if not refs_:
            for meas in self.measurements["all"]:
                data_td = self.get_data(meas)
                amp = np.max(np.abs(data_td[:, 1]))
                if amp > threshold * max_amp:
                    refs_.append(meas)

            logging.info(f"Using max amplitude measurements as ref. (Threshold: {threshold})")

        if not refs_:
            logging.warning(f"No suitable refs found. Check ref_pos option or ref_threshold.")

        logging.info("######################################################\n")

        self.measurements["refs"] = tuple(refs_)

    def _set_img_properties(self):
        # first_meas = self.measurements["all"][0]
        # sample_data_td = first_meas.get_data_td()
        sample_data_td, sample_data_fd = self.get_data(self.measurements["all"][0], domain=Domain.Both)
        samples = int(sample_data_td.shape[0])

        self.time_axis = sample_data_td[:, 0].real
        self.freq_axis = sample_data_fd[:, 0].real

        dt = np.mean(np.diff(self.time_axis))

        x_coords, y_coords = [], []
        for sam_measurement in self.measurements["sams"]:
            x_coords.append(sam_measurement.position[0])
            y_coords.append(sam_measurement.position[1])

        if not x_coords: x_coords.append(0)
        if not y_coords: y_coords.append(0)

        x_coords, y_coords = array(sorted(set(x_coords))), array(sorted(set(y_coords)))

        all_points = list(itertools.product(x_coords, y_coords))

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

        w = int(1 + np.ceil((max(x_coords) - min(x_coords)) / dx))
        h = int(1 + np.ceil((max(y_coords) - min(y_coords)) / dy))

        y_coords = np.round(np.arange(min(y_coords), max(y_coords) + dy, dy), 1)
        x_coords = np.round(np.arange(min(x_coords), max(x_coords) + dx, dx), 1)

        extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

        self._empty_grid = np.zeros((w, h), dtype=complex)

        self.img_properties = {"w": w, "h": h, "dx": dx, "dy": dy, "dt": dt, "samples": samples, "extent": extent,
                               "x_coords": x_coords, "y_coords": y_coords, "all_points": all_points, "img_ax": None}
        self._update_fig_num()

    def _coords_to_idx(self, x_, y_):
        x, y = self.img_properties["x_coords"], self.img_properties["y_coords"]
        x_idx, y_idx = np.argmin(np.abs(x_ - x)), np.argmin(np.abs(y_ - y))

        return x_idx, y_idx

    def _idx_to_coords(self, x_idx, y_idx):
        dx, dy = self.img_properties["dx"], self.img_properties["dy"]

        y = self.img_properties["y_coords"][0] + y_idx * dy
        x = self.img_properties["x_coords"][0] + x_idx * dx

        return x, y

    def _update_fig_num(self):
        en_freq_label = Domain.FrequencyDomain == self.selected_quantity.domain
        fig_num = ""
        if self.options["fig_label"]:
            fig_num += self.options["fig_label"] + " "
        fig_num += str(self.selected_quantity)

        if isinstance(self.selected_freq, tuple):
            f1, f2 = int(self.selected_freq[0] * 1e3), int(self.selected_freq[1] * 1e3)
            fig_num += en_freq_label * f" {f1}-{f2} GHz"
        else:
            fig_num += en_freq_label * f" {int(self.selected_freq * 1e3)}GHz"
        fig_num = fig_num.replace(" ", "_")

        self.img_properties["fig_num"] = fig_num

    def select_freq(self, freq):
        self.selected_freq = freq
        self._update_fig_num()

    def _pre_process(self, meas_):
        pp_opt = self.options["pp_opt"]

        idx = self.raw_data_cache["id_map"][meas_.identifier]
        data_td = self.raw_data_cache["td"][idx]

        if pp_opt["remove_dc"]:
            data_td = remove_offset(data_td)

        if pp_opt["window_opt"]["enabled"]:
            data_td = window(data_td, **pp_opt["window_opt"])

        return data_td

    def get_data(self, meas, domain=None):
        if domain is None:
            domain = Domain.TimeDomain

        data_td = self._pre_process(meas)

        if domain == Domain.TimeDomain:
            return data_td
        elif domain == Domain.FrequencyDomain:
            return do_fft(data_td)
        else:
            return data_td, do_fft(data_td)

    def _fill_cache(self, all_measurements):
        cache_path = Path(self.data_path / "_cache")
        cache_path.mkdir(parents=True, exist_ok=True)

        y_td, y_fd = all_measurements[0].get_data_td(), all_measurements[0].get_data_fd()
        td_cache_shape = (len(all_measurements), *y_td.shape)
        fd_cache_shape = (len(all_measurements), *y_fd.shape)

        try:
            data_td = np.load(str(cache_path / "_td_cache.npy"))
            data_fd = np.load(str(cache_path / "_fd_cache.npy"))
            shape_match = (data_td.shape == td_cache_shape) * (data_fd.shape == fd_cache_shape)
            if not shape_match:
                logging.error("Data <-> cache shape mismatch. Rebuilding cache.")
                raise FileNotFoundError
        except FileNotFoundError:
            data_td = np.zeros(td_cache_shape, dtype=y_td.dtype)
            data_fd = np.zeros(fd_cache_shape, dtype=y_fd.dtype)

            iter_ = tqdm(enumerate(all_measurements), total=len(all_measurements),
                         desc="Saving as npy", colour="green")
            for i, meas in iter_:
                idx = self.raw_data_cache["id_map"][meas.identifier]
                data_td[idx], data_fd[idx] = meas.get_data_td(), meas.get_data_fd()

            np.save(str(cache_path / "_td_cache.npy"), data_td)
            np.save(str(cache_path / "_fd_cache.npy"), data_fd)

        self.raw_data_cache["td"], self.raw_data_cache["fd"] = data_td, data_fd

    def _get_ref_argmax(self, measurement_):
        ref_td = self.get_data(measurement_)
        t, y = ref_td[:, 0], ref_td[:, 1]

        return t[np.argmax(y)]

    def _get_zero_crossing(self, measurement_):
        data_td = self.get_data(measurement_)
        t, y = data_td[:, 0], data_td[:, 1]
        a_max = np.argmax(y)

        zero_crossing_idx = 0
        for i in range(len(y)):
            if i < a_max or i == len(y) - 1:
                continue
            if y[i - 1] > 0 > y[i]:
                zero_crossing_idx = i
                break

        y1, y2 = y[zero_crossing_idx - 1], y[zero_crossing_idx]
        x1, x2 = t[zero_crossing_idx - 1], t[zero_crossing_idx]
        t0 = (y1 * x2 - x1 * y2) / (y1 - y2)

        return t0

    def _p2p(self, meas_):
        y_td = self.get_data(meas_)
        return np.max(y_td[:, 1]) - np.min(y_td[:, 1])

    def _power(self, meas_):
        if not isinstance(self.selected_freq, tuple):
            self.select_freq((1.0, 1.2))
            logging.warning(f"Selected_freq must be a tuple. Using default range ({self.selected_freq})")

        freq_range_ = self.selected_freq
        freq_slice = (freq_range_[0] < self.freq_axis) * (self.freq_axis < freq_range_[1])

        ref_fd, sam_fd = self.get_ref_data(point=meas_.position, domain=Domain.Both)

        power_val_sam = np.sum(np.abs(sam_fd[freq_slice, 1])**2)
        power_val_ref = np.sum(np.abs(ref_fd[freq_slice, 1])**2)

        return power_val_sam / power_val_ref

    def _meas_time_delta(self, meas_):
        ref_meas = self.find_nearest_ref(meas_)

        return (meas_.meas_time - ref_meas.meas_time).total_seconds()

    def _ref_max(self, meas_):
        amp_, _ = self._ref_interpolation(meas_)

        return amp_

    def _ref_phase(self, meas_):
        _, phi_ = self._ref_interpolation(meas_)

        return phi_

    def _simple_peak_cnt(self, meas_, threshold):
        data_td = self.get_data(meas_)
        y_ = data_td[:, 1]
        y_ -= (np.mean(y_[:10]) + np.mean(y_[-10:])) * 0.5

        y_[y_ < threshold] = 0
        peaks_idx = []
        for idx_ in range(1, len(y_) - 1):
            if (y_[idx_ - 1] < y_[idx_]) * (y_[idx_] > y_[idx_ + 1]):
                peaks_idx.append(idx_)

        return len(peaks_idx)

    def _transmission(self, meas_, freq_range_=None):
        ref_fd = self.get_ref_data(point=meas_.position, domain=Domain.FrequencyDomain)

        sam_fd = self.get_data(meas_, Domain.FrequencyDomain)

        if freq_range_ is not None:
            t = sam_fd[:, 1] / ref_fd[:, 1]
        else:
            freq_idx = f_axis_idx_map(ref_fd[:, 0].real, self.selected_freq)
            t = sam_fd[freq_idx, 1] / ref_fd[freq_idx, 1]


        return t

    def _amplitude_transmission(self, meas_):
        t = self._transmission(meas_)

        return np.abs(t)[0]

    def _cmplx_refractive_idx(self, meas_, freq_range=None):
        if self.options["sample_properties"]["default_values"]:
            logging.warning(f"Using default sample properties: {self.options['sample_properties']}")

        d = self.options["sample_properties"]["d"]
        ref_fd = self.get_ref_data(point=meas_.position, domain=Domain.FrequencyDomain)
        sam_fd = self.get_data(meas_, Domain.FrequencyDomain)

        freq_axis = ref_fd[:, 0].real

        phi_ref = np.unwrap(np.angle(ref_fd[:, 1]))
        phi_sam = np.unwrap(np.angle(sam_fd[:, 1]))

        phi = - (phi_sam - phi_ref)
        phi_corrected = phase_correction(freq_axis, phi)

        if freq_range is None:
            freq_idx = f_axis_idx_map(freq_axis, freq_range=self.selected_freq)
        else:
            freq_idx = f_axis_idx_map(freq_axis, freq_range=freq_range)

        freq_axis = freq_axis[freq_idx]
        omega = 2 * np.pi * freq_axis
        phi_ref = phi_ref[freq_idx]
        phi_sam = phi_sam[freq_idx]
        phi = phi[freq_idx]
        phi_corrected = phi_corrected[freq_idx]

        # phi =  - (phi_sam_corrected[freq_idx, 1] - phi_ref_corrected[freq_idx, 1])

        n = 1 + phi_corrected * c_thz / (omega * d)
        n[n < 0] = 1
        kap = -c_thz * np.log(np.abs(sam_fd[freq_idx, 1] / ref_fd[freq_idx, 1]) * (1 + n) ** 2 / (4 * n)) / (omega * d)
        # kap = -c_thz * np.log(np.abs(sam_fd[freq_idx, 1] / ref_fd[freq_idx, 1])) / (omega * d)

        ret = {"freq_axis": freq_axis,
               "refr_idx": n + 1j * kap,
               "phi_ref": phi_ref, "phi_sam": phi_sam, "phi": phi, "phi_corrected": phi_corrected,
               }

        return ret

    def _refractive_idx(self, meas_):
        return np.mean(np.real(self._cmplx_refractive_idx(meas_)["refr_idx"]))

    def _extinction_coe(self, meas_):
        return np.mean(np.imag(self._cmplx_refractive_idx(meas_)["refr_idx"]))

    def _absorption_coef(self, meas_, freq_range=None):
        n_cmplx_res = self._cmplx_refractive_idx(meas_, freq_range)
        freq_axis = n_cmplx_res["freq_axis"]
        kap = n_cmplx_res["refr_idx"].imag

        omega = 2 * np.pi * freq_axis
        alph = (1 / 1e-4) * 2 * kap * omega / c_thz # 1/cm

        if freq_range is None:
            return np.mean(alph)
        else:
            return alph

    def _conductivity(self, meas_, freq_range=None):
        sub_pnt = (70, 10)
        sub_meas = self.sub_dataset.get_measurement(*sub_pnt)
        t_sub = self.sub_dataset._transmission(sub_meas, 1)
        t_sam = self._transmission(meas_, 1)
        sub_res = self.sub_dataset._cmplx_refractive_idx(sub_meas, freq_range=(0, 10))

        f_axis = sub_res["freq_axis"]
        n_sub = sub_res["refr_idx"]
        d_film = self.options["sample_properties"]["d_film"]

        # [eps0] = second * Siemens, [c_thz] = um / ps, [1/d_film] = 1/um -> conversion: 1e12 * 1e-2 (S/cm)
        sigma = 1e10 * (1/d_film) * eps0 * c_thz * (1 + n_sub) * (t_sub/t_sam - 1)

        # phase correction, [dt] = fs
        dt = self.options["pp_opt"]["dt"]
        dt *= 1e-3
        sigma *= np.exp(-1j*dt*2*np.pi*f_axis)

        sigma.imag *= 1

        return sigma

    def _calc_grid_vals(self):
        if self.grid_vals is not None:
            return self.grid_vals

        grid_vals = self._empty_grid.copy()
        sam_meas = self.measurements["sams"]

        iter_ = tqdm(enumerate(sam_meas), total=len(sam_meas),
                     desc="Evaluating measurements", colour="green")
        for i, measurement in iter_:
            x_idx, y_idx = self._coords_to_idx(*measurement.position)

            grid_vals[x_idx, y_idx] = self.grid_func(measurement)

        return grid_vals

    def select_quantity(self, quantity, label=""):
        if isinstance(quantity, Quantity):
            if not callable(Quantity.func):
                logging.warning("Func of Quantity must be callable")
            self.grid_func = quantity.func
            self.selected_quantity = quantity

        if callable(quantity):
            self.grid_func = quantity
            self.selected_quantity = Quantity(label, func=quantity)

        single_freq_quant = [QuantityEnum.RefractiveIdx, QuantityEnum.AbsorptionCoe]
        if quantity in single_freq_quant and isinstance(self.selected_freq, tuple):
            logging.warning(f"Selected_freq is a tuple. Averaging range ({self.selected_freq})")

        func_map = {QuantityEnum.P2P: self._p2p,
                    QuantityEnum.Power: self._power,
                    QuantityEnum.MeasTimeDeltaRef2Sam: self._meas_time_delta,
                    QuantityEnum.RefAmp: self._ref_max,
                    QuantityEnum.RefArgmax: self._get_ref_argmax,
                    QuantityEnum.RefPhase: self._ref_phase,
                    QuantityEnum.PeakCnt: partial(self._simple_peak_cnt, threshold=2.5),
                    QuantityEnum.TransmissionAmp: self._amplitude_transmission,
                    QuantityEnum.RefractiveIdx: self._refractive_idx,
                    QuantityEnum.AbsorptionCoe: self._absorption_coef,
                    }

        if quantity in func_map:
            self.grid_func = lambda x: np.real(func_map[quantity](x))
            self.selected_quantity = quantity.value

        self._update_fig_num()

    def get_measurement(self, x, y) -> Measurement:
        meas_list = self.measurements["all"]
        pnt = (x, y)
        try:
            key = "".join([f"{val:.3f}" for val in pnt])
            closest_meas = self.raw_data_cache["coord_map"][key]
        except KeyError:
            closest_meas, best_fit_val = None, np.inf
            for meas in meas_list:
                val = abs(meas.position[0] - pnt[0]) + abs(meas.position[1] - pnt[1])
                if val < best_fit_val:
                    best_fit_val = val
                    closest_meas = meas

        return closest_meas

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
            return None

        x_coords, y_coords = self.img_properties["x_coords"], self.img_properties["y_coords"]

        # vertical direction / slice
        if x is not None:
            return [self.get_measurement(x, y_) for y_ in y_coords], y_coords
        else:  # horizontal direction / slice
            return [self.get_measurement(x_, y) for x_ in x_coords], x_coords

    def find_nearest_ref(self, meas_):
        dist_func = self.options["dist_func"]

        closest_ref, best_fit_val = None, np.inf
        for ref_meas in self.measurements["refs"]:
            dist_val = dist_func(ref_meas, meas_)

            if np.abs(dist_val) < np.abs(best_fit_val):
                best_fit_val = dist_val
                closest_ref = ref_meas

        logging.debug(f"Sam: {meas_})")
        logging.debug(f"Ref: {closest_ref})")
        if self.options["dist_func"] == Dist.Time:
            logging.debug(f"Time between ref and sample: {best_fit_val} seconds")
        else:
            logging.debug(f"Distance between ref and sample: {best_fit_val} mm")

        return closest_ref

    def get_ref_data(self, domain=Domain.TimeDomain, point=None):
        if point is not None:
            closest_sam = self.get_measurement(*point)
            chosen_ref = self.find_nearest_ref(closest_sam)
        else:
            chosen_ref = self.measurements["refs"][-1]

        if domain in [Domain.TimeDomain, Domain.FrequencyDomain]:
            return self.get_data(chosen_ref, domain=domain)
        else:
            return self.get_data(chosen_ref, domain=Domain.Both)

    def get_ref_sam_meas(self, point):
        sam_meas = self.get_measurement(*point)
        ref_meas = self.find_nearest_ref(sam_meas)

        return ref_meas, sam_meas

    def _ref_interpolation(self, sam_meas, ret_cart=False):
        sam_meas_time = sam_meas.meas_time

        nearest_ref_idx, smallest_time_diff, time_diff = None, np.inf, 0
        for ref_idx in range(len(self.measurements["refs"])):
            time_diff = (self.measurements["refs"][ref_idx].meas_time - sam_meas_time).total_seconds()
            if abs(time_diff) < abs(smallest_time_diff):
                nearest_ref_idx = ref_idx
                smallest_time_diff = time_diff

        t0 = self.measurements["refs"][0].meas_time
        if smallest_time_diff <= 0:
            # sample was measured after reference
            ref_before = self.measurements["refs"][nearest_ref_idx]
            ref_after = self.measurements["refs"][nearest_ref_idx + 1]
        else:
            ref_before = self.measurements["refs"][nearest_ref_idx - 1]
            ref_after = self.measurements["refs"][nearest_ref_idx]

        t = [(ref_before.meas_time - t0).total_seconds(), (ref_after.meas_time - t0).total_seconds()]
        ref_before_td, ref_after_td = self.get_data(ref_before), self.get_data(ref_after)

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

    def average_area(self, pnt_bot_left, pnt_top_right, label="A1"):
        assert (pnt_bot_left[0] <= pnt_top_right[0]) and (pnt_bot_left[1] <= pnt_top_right[1])

        x_coords, y_coords = self.img_properties["x_coords"], self.img_properties["y_coords"]
        pnt_bot_left = (x_coords[np.argmin(np.abs(pnt_bot_left[0] - x_coords))],
                        y_coords[np.argmin(np.abs(pnt_bot_left[1] - y_coords))])
        pnt_top_right = (x_coords[np.argmin(np.abs(pnt_top_right[0] - x_coords))],
                         y_coords[np.argmin(np.abs(pnt_top_right[1] - y_coords))])

        x0_idx, y0_idx = self._coords_to_idx(*pnt_bot_left)
        x1_idx, y1_idx = self._coords_to_idx(*pnt_top_right)

        if self.grid_vals is None:
            self._calc_grid_vals()

        grid_vals = self.grid_vals[x0_idx:x1_idx+1, y0_idx:y1_idx+1]
        mean_val, std_val = np.mean(grid_vals), np.std(grid_vals)

        mean_s, std_s = str(np.round(mean_val, 4)), str(np.round(std_val, 4))
        min_s, max_s = str(np.round(np.min(grid_vals), 4)), str(np.round(np.max(grid_vals), 4))

        meas_cnt = grid_vals.shape[0] * grid_vals.shape[1]
        logging.info(f"Average value of area {label} ({meas_cnt} measurements): {mean_s}±{std_s}")
        logging.info(f"Min: {min_s}, max: {max_s}\n")

        ret = {"mean": mean_val, "std": std_val, "min": min_s, "max": max_s}
        if not plt.fignum_exists(num=self.img_properties["fig_num"]):
            return ret

        plt.figure(self.img_properties["fig_num"])
        ax = self.img_properties["img_ax"]

        dx, dy = self.img_properties["dx"], self.img_properties["dy"]
        # pixels are centered around each coordinate, unlike patches.Rectangle
        pnt_bot_left = (pnt_bot_left[0] - dx / 2, pnt_bot_left[1] - dy / 2)
        pnt_top_right = (pnt_top_right[0] + dx / 2, pnt_top_right[1] + dy / 2)

        # draw rectangle
        rect_width = pnt_top_right[0] - pnt_bot_left[0]
        rect_height = pnt_top_right[1] - pnt_bot_left[1]
        rect = patches.Rectangle(
            pnt_bot_left,  # bottom left
            rect_width,  # width
            rect_height,  # height
            linewidth=2, edgecolor="black", facecolor="none"
        )
        ax.add_patch(rect)

        # decide where to put rect label
        img_extent = self.img_properties["extent"]
        t_x, t_y = pnt_bot_left[0] + rect_width / 2, pnt_bot_left[1] + rect_height / 2
        if t_x < img_extent[0] + 1:
            t_x = pnt_bot_left[0] + rect_width + 1.5
        if t_x > img_extent[1] - 1:
            t_x = pnt_top_right[0] - rect_width - 1.5

        t_y_below = pnt_top_right[1] - rect_height - 1.5
        t_y_above = pnt_bot_left[1] + rect_height + 1.5
        if t_y < img_extent[2] + 1:
            t_y = t_y_above
        if t_y > img_extent[3] - 1:
            t_y = t_y_below

        # if rect too small, place above or below
        if rect_height < 3.0 or rect_width < 3.0:
            # if in top half, place below, else place above
            if t_y > img_extent[2] + 0.5 * abs(img_extent[3] - img_extent[2]):
                t_y = t_y_below
            else:
                t_y = t_y_above

        # add label
        ax.text(t_x, t_y, label,
                color="black", fontsize=18, ha="center", va="center", fontweight="bold")

        return ret


    def plot_point(self, point=None, en_td_plot=True, **kwargs_):
        kwargs = {"label": "",
                  "sub_noise_floor": False,
                  "td_scale": 1,
                  "remove_t_offset": False, }
        kwargs.update(kwargs_)

        label = kwargs["label"]
        sub_noise_floor = kwargs["sub_noise_floor"]
        td_scale = kwargs["td_scale"]
        remove_t_offset = kwargs["remove_t_offset"]

        plot_range = self.options["plot_range"]

        if point is None:
            sam_meas = self.measurements["all"][0]
            point = sam_meas.position
        else:
            sam_meas = self.get_measurement(*point)
        ref_meas = self.find_nearest_ref(sam_meas)

        ref_td, ref_fd = self.get_data(ref_meas, domain=Domain.Both)
        sam_td, sam_fd = self.get_data(sam_meas, domain=Domain.Both)

        if remove_t_offset:
            ref_td[:, 0] -= ref_td[0, 0]
            sam_td[:, 0] -= sam_td[0, 0]

        freq_axis = ref_fd[:, 0].real

        t = sam_fd[:, 1] / ref_fd[:, 1]
        absorb = np.abs(1/t)

        f_min, f_max = freq_axis[plot_range][0], freq_axis[plot_range][-1]
        simple_eval_res = self._cmplx_refractive_idx(sam_meas, (f_min, f_max))
        phi = simple_eval_res["phi"]
        phi_corrected = simple_eval_res["phi_corrected"]

        refr_idx = simple_eval_res["refr_idx"]
        alph = self._absorption_coef(sam_meas, (f_min, f_max))

        ret = {"freq_axis": freq_axis, "absorb": absorb, "t": t, "ref_fd": ref_fd, "sam_fd": sam_fd}

        noise_floor = np.mean(20 * np.log10(np.abs(ref_fd[ref_fd[:, 0] > 6.0, 1]))) * sub_noise_floor

        if not self.plotted_ref:
            y_db = (20 * np.log10(np.abs(ref_fd[plot_range, 1])) - noise_floor).real
            plt.figure("Spectrum")
            plt.plot(freq_axis[plot_range], y_db, label="Reference")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Amplitude (dB)")

            plt.figure("Time domain")
            plt.plot(ref_td[:, 0], ref_td[:, 1], label="Reference")
            plt.xlabel("Time (ps)")
            plt.ylabel("Amplitude (Arb. u.)")

            self.plotted_ref = True

        if not label:
            label += f" (x,y)=({point[0]}, {point[1]}) mm"

        freq_axis = sam_fd[:, 0].real
        noise_floor = np.mean(20 * np.log10(np.abs(sam_fd[sam_fd[:, 0] > 6.0, 1]))) * sub_noise_floor

        plt.figure("Spectrum")
        y_db = (20 * np.log10(np.abs(sam_fd[plot_range, 1])) - noise_floor).real
        plt.plot(freq_axis[plot_range], y_db, label=label)

        plt.figure("Phase")
        plt.plot(freq_axis[plot_range], phi, label="Original", ls="dashed")
        plt.plot(freq_axis[plot_range], phi_corrected, label="Corrected")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (rad)")

        plt.figure("phase slope")
        plt.plot(freq_axis[plot_range][:-1], np.diff(phi_corrected))

        plt.figure("Time domain")
        td_label = label
        if not np.isclose(td_scale, 1):
            td_label += f"\n(Amplitude x {td_scale})"
        if en_td_plot:
            plt.plot(sam_td[:, 0], td_scale * sam_td[:, 1], label=td_label)

        if not plt.fignum_exists("Amplitude transmission"):
            plt.figure("Amplitude transmission")
            plt.xlabel("Frequency (THz)")
            plt.ylabel(r"Amplitude transmission")
            plt.ylim((-0.05, 1.10))
        else:
            plt.figure("Amplitude transmission")

        plt.plot(freq_axis[plot_range], (1/absorb[plot_range]), label=label)

        plt.figure("Absorbance")
        plt.plot(freq_axis[plot_range], 20 * np.log10(absorb[plot_range]), label=label)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Absorbance (dB)")

        plt.figure("Refractive index")
        plt.plot(freq_axis[plot_range], refr_idx.real, label="Real part")
        plt.plot(freq_axis[plot_range], refr_idx.imag, label="Imaginary part")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Refractive index")

        plt.figure("Absorption coefficient")
        plt.plot(freq_axis[plot_range], alph)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Absorption coefficient (1/cm)")

        if self.sub_dataset is not None:
            sigma = self._conductivity(sam_meas)

            plt.figure("Conductivity")
            plt.plot(freq_axis[plot_range], sigma[plot_range].real, label="Real part")
            plt.plot(freq_axis[plot_range], sigma[plot_range].imag, label="Imaginary part")
            plt.ylim((-1e3, 1.5e5))
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Conductivity (S/cm)")

        return ret

    def plot_system_stability(self):
        first_meas = self.measurements["all"][0]
        if all([first_meas.position == meas.position for meas in self.measurements["all"]]):
            meas_set = self.measurements["all"]
        else:
            meas_set = self.measurements["refs"]

        selected_freq_ = self.selected_freq
        f_idx = np.argmin(np.abs(self.freq_axis - selected_freq_))

        ref_ampl_arr, ref_angle_arr, ref_zero_crossing = np.zeros((3, len(meas_set)))
        t0 = meas_set[0].meas_time
        meas_times = np.array([(ref.meas_time - t0).total_seconds() / 3600 for ref in meas_set])

        for i, ref in enumerate(meas_set):
            ref_td, ref_fd = self.get_data(ref, domain=Domain.Both)

            ref_zero_crossing[i] = self._get_zero_crossing(ref)
            ref_ampl_arr[i] = np.sum(np.abs(ref_fd[f_idx, 1]))
            ref_angle_arr[i] = np.angle(ref_fd[f_idx, 1])

        meas_interval = np.mean(np.diff(meas_times))
        ref_angle_arr = np.unwrap(ref_angle_arr)

        minima = local_minima_1d(ref_angle_arr, en_plot=False)
        period, period_std = minima[1] * meas_interval * 60, minima[2] * meas_interval * 60

        ref_zero_crossing = (ref_zero_crossing - ref_zero_crossing[0]) * 1000

        # correction
        # ref_angle_arr -= 2*np.pi*self.freq_axis[f_idx]*(ref_zero_crossing/1000)

        max_diff_0x = np.max(np.diff(ref_zero_crossing))
        logging.info(f"Largest jump: {np.round(max_diff_0x, 2)} fs")
        max_diff = np.max(np.diff(ref_angle_arr))
        logging.info(f"Largest phase jump: {np.round(max_diff, 2)} rad")

        avg_amp_change = np.mean(np.abs(np.diff(ref_ampl_arr)))
        max_amp_change = np.max(np.diff(ref_ampl_arr))

        logging.info(f"Largest amplitude change: {np.round(max_amp_change, 2)} (Arb. u.)")
        logging.info(f"Mean absolute amplitude change: {np.round(avg_amp_change, 2)} (Arb. u.)")
        logging.info(f"Measurement interval: {np.round(meas_interval * 60, 2)} min.")
        logging.info(f"Period: {np.round(period, 2)}±{np.round(period_std, 2)} min.")

        plt.figure("fft")
        phi_fft = np.fft.rfft(ref_angle_arr)
        phi_fft_f = np.fft.rfftfreq(len(ref_angle_arr), d=meas_interval * 3600)

        plt.plot(phi_fft_f[1:], np.abs(phi_fft)[1:])
        plt.xlabel("Frequency (1/hour)")
        plt.ylabel("Magnitude")

        if meas_times.max() < 5 / 60:
            meas_times *= 3600
            mt_unit = "seconds"
        elif 5 / 60 <= meas_times.max() < 0.5:
            meas_times *= 60
            mt_unit = "minutes"
        else:
            mt_unit = "hours"

        plt.figure("Reference zero crossing")
        plt.title(f"Reference zero crossing change")
        plt.plot(meas_times, ref_zero_crossing, label=t0)
        plt.xlabel(f"Measurement time ({mt_unit})")
        plt.ylabel("Time (fs)")

        plt.figure("Stability amplitude")
        plt.title(f"Amplitude of reference measurement at {selected_freq_} THz")
        plt.plot(meas_times, ref_ampl_arr, label=t0)
        plt.xlabel(f"Measurement time ({mt_unit})")
        plt.ylabel("Amplitude (Arb. u.)")

        plt.figure("Stability phase")
        plt.title(f"Phase of reference measurement at {selected_freq_} THz")
        plt.plot(meas_times, ref_angle_arr, label=t0)
        plt.xlabel(f"Measurement time ({mt_unit})")
        plt.ylabel("Phase (rad)")

        if plt.fignum_exists(self.img_properties["fig_num"]):
            self.plot_refs()

    def plot_frequency_noise(self):
        ref_meas_set = self.measurements["refs"]
        freq_axis = self.freq_axis

        ampl_arr_db = np.zeros((len(ref_meas_set), len(freq_axis)))
        for i, ref in enumerate(ref_meas_set):
            ref_td, ref_fd = self.get_data(ref, domain=Domain.Both)
            ampl_arr_db[i] = 20*np.log10(np.abs(ref_fd[:, 1]))


        plt.figure("Amplitude noise")
        #plt.title(f"Amplitude of reference measurement at {selected_freq_} THz")
        plt.plot(freq_axis, np.std(ampl_arr_db, axis=0))

        plt.xlabel(f"Frequency (THz)")
        plt.ylabel("Amplitude (dB)")

    def plot_climate(self, log_file, quantity=ClimateQuantity.Temperature):
        def read_log_file(log_file_):
            def read_line(line_):
                parts = line_.split(" ")
                t = datetime.strptime(f"{parts[0]} {parts[1]}", '%Y-%m-%d %H:%M:%S')
                return t, float(parts[4]), float(parts[-3])

            with open(log_file_) as file:
                meas_time_, temp_, humidity_ = [], [], []
                for i, line in enumerate(file):
                    if i % 250:
                        pass
                    try:
                        res = read_line(line)
                        meas_time_.append(res[0])
                        temp_.append(res[1])
                        humidity_.append(res[2])
                    except IndexError:
                        continue

            return meas_time_, temp_, humidity_

        meas_time, temp, humidity = read_log_file(log_file)

        if self.measurements["refs"] is not None:
            t0 = self.measurements["refs"][0].meas_time
        else:
            t0 = meas_time[0]

        meas_time_diff = [(t - t0).total_seconds() / 3600 for t in meas_time]

        if quantity == ClimateQuantity.Temperature:
            quant = temp
            y_label = "Temperature (°C)"
        else:
            quant = humidity
            y_label = "Humidity (\%)"

        stability_figs = ["Reference zero crossing", "Stability amplitude", "Stability phase"]
        for fig_label in stability_figs:
            if plt.fignum_exists(fig_label):
                fig = plt.figure(fig_label)
                ax_list = fig.get_axes()
                ax1 = ax_list[0]
                ax1.tick_params(axis="y", colors="blue")
                ax1.set_ylabel(ax1.get_ylabel(), c="blue")
                ax1.grid(c="blue")

                ax2 = ax1.twinx()
                ax2.scatter(meas_time_diff, quant, c="red")
                ax2.set_ylabel(y_label, c="red")
                ax2.tick_params(axis="y", colors="red")
                ax2.grid(False)

        if not plt.fignum_exists(stability_figs[0]):
            fig, ax1 = plt.subplots(num="Climate plot")
            ax1.scatter(meas_time_diff, quant, label=f"Start: {t0}")
            ax1.set_xlabel("Measurement time (hour)")
            ax1.set_ylabel(y_label)

    def plot_image(self, img_extent=None):
        info = self.img_properties
        if img_extent is None:
            w0, w1, h0, h1 = [0, info["w"], 0, info["h"]]
        else:
            dx, dy = info["dx"], info["dy"]
            w0, w1 = int((img_extent[0] - info["extent"][0]) / dx), int((img_extent[1] - info["extent"][0]) / dx)
            h0, h1 = int((img_extent[2] - info["extent"][2]) / dy), int((img_extent[3] - info["extent"][2]) / dy)

        self.grid_vals = self._calc_grid_vals()

        shown_grid_vals = self.grid_vals.real
        shown_grid_vals = shown_grid_vals[w0:w1, h0:h1]
        shown_grid_vals = self._exclude_pixels(shown_grid_vals)

        if self.options["log_scale"]:
            shown_grid_vals = np.log10(shown_grid_vals)

        fig = plt.figure(self.img_properties["fig_num"])
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.2)

        if img_extent is None:
            img_extent = self.img_properties["extent"]

        cbar_min, cbar_max = self.options["cbar_lim"]
        if cbar_min is None:
            cbar_min = np.min(shown_grid_vals)
        if cbar_max is None:
            cbar_max = np.max(shown_grid_vals)

        if self.options["log_scale"]:
            self.options["cbar_min"] = np.log10(cbar_min)
            self.options["cbar_max"] = np.log10(cbar_max)

        axes_extent = (float(img_extent[0] - self.img_properties["dx"] / 2),
                       float(img_extent[1] + self.img_properties["dx"] / 2),
                       float(img_extent[2] - self.img_properties["dy"] / 2),
                       float(img_extent[3] + self.img_properties["dy"] / 2))
        img_ = ax.imshow(shown_grid_vals.transpose((1, 0)),
                         vmin=cbar_min, vmax=cbar_max,
                         origin="lower",
                         cmap=plt.get_cmap(self.options["color_map"]),
                         extent=axes_extent,
                         interpolation=self.options["pixel_interpolation"].value
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

        quantity_label = " ".join([str(self.selected_quantity), freq_label * en_freq_label])
        img_title_option = str(self.options["img_title"])
        ax.set_title(" ".join([quantity_label, img_title_option]))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = fig.colorbar(img_, cax=cax)
        cbar.set_ticks(np.round(np.linspace(cbar_min, cbar_max, 4), 3))

        if self.options["en_cbar_label"]:
            cbar.set_label(quantity_label, rotation=270, labelpad=30)

        self.img_properties["img_ax"] = ax

    def plot_refs(self):
        if not plt.fignum_exists(self.img_properties["fig_num"]):
            return

        plt.figure(num=self.img_properties["fig_num"])
        img_ax = self.img_properties["img_ax"]

        ref_x_coords, ref_y_coords = [], []
        for ref in self.measurements["refs"]:
            ref_x_coords.append(ref.position[0])
            ref_y_coords.append(ref.position[1])

        plt_fun = img_ax.scatter

        plt_fun(ref_x_coords, ref_y_coords, color="black", linewidth=0.4)

    def plot_line(self, line_coords=None, direction=Direction.Horizontal, **kwargs):
        if line_coords is None:
            line_coords = [0.0]
        if isinstance(line_coords, (int, float)):
            line_coords = [line_coords]

        horizontal = direction == direction.Horizontal

        if horizontal:
            fig_num = "x-slice"
            plt.xlabel("x (mm)")
        else:
            fig_num = "y-slice"
            plt.xlabel("y (mm)")

        fig_num += "_" + self.img_properties["quantity_label"].replace(" ", "_")
        plt.figure(fig_num)
        plt.title(f"Line scan ({direction.name})")
        plt.ylabel(self.img_properties["quantity_label"])

        for line_coord in line_coords:
            if horizontal:
                measurements, coords = self.get_line(None, line_coord)
            else:
                measurements, coords = self.get_line(line_coord, None)

            vals = []
            for i, measurement in enumerate(measurements):
                logging.info(f"{round(100 * i / len(measurements), 2)} % done. "
                             f"(Measurement: {i}/{len(measurements)}, {measurement.position} mm)")

                vals.append(self.grid_func(measurement))

            if horizontal:
                kwargs["label"] = f"y={line_coord} (mm)"
                plt.plot(coords, vals, **kwargs)
            else:
                kwargs["label"] = f"x={line_coord} (mm)"
                plt.plot(coords, vals, **kwargs)

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

    def link_sub_dataset(self, dataset_):
        self.sub_dataset = dataset_



    def plot_jitter(self):
        x = [25, 50, 100, 200]
        y = [113.8, 39.8, 12.47, 6.17]

        plt.figure("Jitter")
        plt.plot(x, y)
        plt.xlabel("Measurement window (ps)")
        plt.ylabel("Largest jump (fs)")


if __name__ == '__main__':
    options = {
        # "cbar_lim": (0.52, 0.60), # 2.5 THz
        # "cbar_lim": (0.64, 0.66), # img4
        "cbar_lim": (0.60, 0.66), # img5 1.5 THz
        # "cbar_lim": (0.55, 0.62), # img5 2.0 THz
        "plot_range": slice(30, 650),
        "ref_pos": (10.0, 0.0),
        "fig_label": "",
        "dist_func": Dist.Position,

    }

    logging.basicConfig(level=logging.INFO)
    # dataset = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Thinfilm_solarcell")
    # dataset = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Laser_crystallized_Si")
    # dataset = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Wood/S1")

    # dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img2", options)
    # dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img2", options)
    # dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img5", options)
    dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img5", options)

    # dataset = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Leaf/scan3", options)
    # dataset = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Leaf/scan2", options)
    # dataset = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Graphene/scan2", options)

    dataset.select_freq(2.00)
    # dataset.select_quantity(QuantityEnum.P2P)
    dataset.select_quantity(QuantityEnum.TransmissionAmp)
    # dataset.plot_point((16, 10), apply_window=False)
    dataset.plot_point((35, 0), apply_window=False)
    # dataset.plot_line()
    # dataset.plot_point((30, 14), label="x=30 mm")
    # dataset.plot_point((40, 14), label="x=40 mm")
    # dataset.plot_point((44, 14), label="x=44 mm")

    # dataset.plot_point((35, 0), label="x=35 mm, side branch")
    # dataset.plot_point((47, 0), label="x=47 mm, leaf")

    dataset.plot_image()
    # dataset.average_area((19, -2), (32, 5), label="2") # img3
    # dataset.average_area((72, -2), (85, 3), label="9")
    # dataset.average_area((72, -1), (85, 2), label="9") # img3
    # dataset.average_area((25, -10), (48, 3), label="7") # img4
    # dataset.average_area((62, -10), (83, 3), label="8") # img4
    dataset.average_area((30, -10), (40, -3.5), label="12.1")  # img5
    dataset.average_area((23, 2), (28, 4), label="12.2") # img5

    # img 1
    #dataset.average_area((4, 8), (20, 13), label="8")
    #dataset.average_area((36, 8), (51, 13), label="7")
    #dataset.average_area((35, -13.5), (50, -11), label="4")
    #dataset.average_area((70, -14.25), (79, -13), label="2")
    #dataset.average_area((64, 13.50), (77, 14.25), label="9")

    # img 2
    #dataset.average_area((9, -10), (30, 2), label="A1") # VR04
    #dataset.average_area((45, -10), (60, 0), label="A2") # VR01_1
    #dataset.average_area((68, -12), (70, -1.5), label="A3") # VR01_2

    #dataset.plot_point((10, 14), label="SC 1")
    #dataset.plot_point((40, 14), label="SC 2")
    #dataset.plot_point((10, -14), label="SC 3")
    #dataset.plot_point((40, -14), label="SC 4")

    # dataset2 = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Laser_crystallized_Si")

    # point = choice(dataset.img_properties["all_points"])
    # img.window_all()
    #dataset.plot_point((45, 14), label="Wood substrate (x=45 mm)")
    #dataset.plot_point((50, 14), label="Wood substrate (x=50 mm)")
    # dataset.plot_point((5, 14), label="Reference (air)")
    # dataset2.plot_point((60, -14), label="Borofloat sub.")
    # dataset2.plot_point((30, -14), label="Sub. + SiN sub.")
    # dataset2.plot_point((5, -14), label="Si B-doped")
    # dataset2.plot_point((60, 14), label="Si + SiN interlayer + Borofloat")
    # dataset2.plot_point((30, 14), label="Si n-doped + SiN interlayer + Borofloat")

    # dataset.plot_line(y=-14.00, label="y=-14 mm")
    # dataset.plot_line(y=14.00, label="y=14 mm")

    # dataset2 = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Wood/S5")
    #dataset2.plot_line(y=-12.00, label="y=-12")
    #dataset2.plot_line(y=12.00, label="y=12")

    #dataset2.plot_point((60, 12), label="Wood + Lacquer (Pure)")
    #dataset2.plot_point((60, -12), label="Wood + Lac. (1:1)")
    #dataset2.plot_point((40, 12), label="Wood + Lac. (1:2)")
    #dataset2.plot_point((40, -12), label="Wood + Lac. (1:5)")

    # dataset.plot_point((50, 14), label="Wood substrate (x=50 mm)")
    # dataset = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Wood/S2")
    # dataset2.plot_point((50, 14), label="Wood substrate")
    # dataset.evaluate_point(point, 1000, en_plot=True)
    # dataset.selected_freq = 2.0
    # dataset.plot_line(y=14.00)
    # dataset.plot_system_stability()
    # dataset.plot_jitter()
    # dataset.plot_climate(r"/home/ftpuser/ftp/Data/Stability/T_RH_sensor_logs/2024-11-20 17-27-58_log.txt", quantity=ClimateQuantity.Temperature)

    plt_show(en_save=False)
