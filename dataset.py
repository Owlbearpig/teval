import itertools
import os
import random
from copy import deepcopy
from functools import partial
import consts
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import array
from pathlib import Path
import numpy as np
from functions import unwrap, window, local_minima_1d, check_dict_values, WindowTypes, butter_filt
from measurements import MeasurementType, Measurement, Domain
from mpl_settings import mpl_style_params
import matplotlib as mpl
from functions import phase_correction, do_fft, do_ifft, f_axis_idx_map, remove_offset
from consts import c_thz, eps0_thz
from scipy.optimize import shgo
from scipy.special import erfc
from enum import Enum
import logging
from datetime import datetime
from tqdm import tqdm


"""
TODOs: 
- How are measurements mapped when multiple measurements are performed at the same x-y coordinates?
- Setting cbar lims sucks... set lims based on area min max?
- Fix runtime / use cache for t calc?
- window function (fuctions.py): allow negative values (wrap around) + fix plot (clipping)
- Logging is messy, also fix log levels and RuntimeWarnings
- freq_range variable in transmission function (and other functions?)

New ideas: add teralyzer evaluation (time consuming)
- Add plt_show here (done)
- Possibly add marker in image to show where .plot_point() is
- interactive imshow plots -> maybe connect to .plot_point()

# units:
[l] = µm, [t] = ps, [alpha] = 1/cm (absorption coe.), [sigma] = S/cm, [eps0] = Siemens * ps,
[f] = THz (1/ps), [c_thz] = µm/ps
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

    def __init__(self, label="label", func=None, domain=None):
        self.label = label
        if domain is None:
            self.domain = Domain.Time
        else:
            self.domain = domain
        if func is not None:
            self.func = func

    def __repr__(self):
        return self.label

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class QuantityEnum(Enum):
    P2P = Quantity("Peak to peak")
    Power = Quantity("Power", domain=Domain.Frequency)
    Phase = Quantity("Phase", domain=Domain.Frequency)
    MeasTimeDeltaRef2Sam = Quantity("Time delta Ref. to Sam.")
    RefAmp = Quantity("Ref. Amp", domain=Domain.Frequency)
    RefArgmax = Quantity("Ref. Argmax")
    RefPhase = Quantity("Ref. Phase", domain=Domain.Frequency)
    PeakCnt = Quantity("Peak Cnt")
    ZeroCrossing = Quantity("Zero Crossing", domain=Domain.Time)
    TimeOfFlight = Quantity("Time of Flight", domain=Domain.Time)
    TransmissionAmp = Quantity("Amplitude transmission", domain=Domain.Frequency)
    TransmissionPhase = Quantity("Phase transmission", domain=Domain.Frequency)
    RefractiveIdx = Quantity("Refractive idx", domain=Domain.Frequency)
    AbsorptionCoe = Quantity("Absorption coe", domain=Domain.Frequency)


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
        self.freq_axis = None
        self.raw_data_cache = {}
        self.options = {}
        self.img_properties = {}
        self.selected_freq = None
        self.selected_freq_idx = None
        self.selected_quantity = None
        self.grid_func = None
        self.grid_vals = None
        self.measurements = {"refs": None, "sams": None, "all": None}
        self.sub_dataset = None
        self._is_sub_dataset = False

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
                           "save_plots": False,
                           "excluded_areas": None,
                           "cbar_lim": (None, None),
                           "log_scale": False,
                           "color_map": "autumn",
                           "invert_x": False, "invert_y": False,
                           "pixel_interpolation": PixelInterpolation.none,
                           "fig_label": "",
                           "img_title": "",
                           "en_cbar_label": False,
                           "plot_range": slice(0, 900),
                           "ref_pos": (None, None),
                           "ref_threshold": 0.95,
                           "dist_func": Dist.Position,
                           "pp_opt": {
                               "window_opt": {"enabled": False, "win_width": None, "win_start": None,
                                              "shift": None, "slope": 0.15, "en_plot": False,
                                              "type": WindowTypes.tukey},
                               "filter_opt": {"enabled": False, "f_range": (0.3, 3.0), },
                               "remove_dc": True,
                               "dt": 0,
                           },
                           "sample_properties": {"d": 1000, "layers": 1, "default_values": True},
                           "eval_opt": {"dt": 0,  # dt in fs
                                        "sub_pnt": (0, 0),
                                        "fit_range": (0.50, 2.20),
                                        },
                           "only_shown_figures": [],
                           "shown_plots": {
                               "Window": True,
                               "Time domain": True,
                               "Spectrum": True,
                               "Phase": True,
                               "Phase slope": False,
                               "Amplitude transmission": False,
                               "Absorbance": False,
                               "Refractive index": False,
                               "Absorption coefficient": False,
                               "Conductivity": False,
                           },
                           "plot_opt": {"shift_sam2ref": False,}
                           }
        if "sample_properties" in options_:
            default_options["sample_properties"]["default_values"] = False

        check_dict_values(options_, default_options)

        if not isinstance(options_["result_dir"], Path):
            options_["result_dir"] = Path(options_["result_dir"])

        self.options.update(options_)
        self._apply_options()

    def _apply_options(self):
        new_rc_params = {"savefig.directory": self.options["result_dir"]}

        mpl.rcParams.update(mpl_style_params(new_rc_params))
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

        logging.info(f"Dataset contains {len(all_measurements)} measurements")
        if not self.measurements["refs"]:
            logging.info("No reference measurements found based on filename (ref)")
        else:
            logging.info(f"{len(self.measurements['refs'])} reference measurements ")
            logging.info(f"{len(self.measurements['sams'])} sample measurements")

        first_measurement = self.measurements["all"][0]
        last_measurement = self.measurements["all"][-1]

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
            data_td = self._get_data(meas)
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
        max_amp = np.max(np.abs(self._get_data(max_amp_meas)[:, 1]))

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
                data_td = self._get_data(meas)
                if np.max(np.abs(data_td[:, 1])) > threshold * max_amp:
                    refs_.append(meas)

        if len(refs_) > 1:
            logging.info(f"Using reference measurements: {refs_[0].filepath.stem} to {refs_[-1].filepath.stem}")

        if not refs_:
            for meas in self.measurements["all"]:
                data_td = self._get_data(meas)
                amp = np.max(np.abs(data_td[:, 1]))
                if amp > threshold * max_amp:
                    refs_.append(meas)

            logging.info(f"Using max amplitude measurements as ref. (Threshold: {threshold})")

        if not refs_:
            logging.warning(f"No suitable refs found. Check ref_pos option or ref_threshold.")

        self.measurements["refs"] = tuple(refs_)

        logging.info(f"Found {len(self.measurements['refs'])} possible reference measurements")
        logging.info("######################################################\n")


    def _set_img_properties(self):
        # first_meas = self.measurements["all"][0]
        # sample_data_td = first_meas.get_data_td()
        sample_data_td, sample_data_fd = self._get_data(self.measurements["all"][0], domain=Domain.Both)
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
                               "x_coords": x_coords, "y_coords": y_coords, "all_points": all_points,
                               "img_ax": None}
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
        en_freq_label = Domain.Frequency == self.selected_quantity.domain
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

        self.img_properties["fig_num"] = fig_num + self._is_sub_dataset * "_subset"

        self._update_quantity_label()

    def _update_quantity_label(self):
        en_freq_label = Domain.Frequency == self.selected_quantity.domain
        if isinstance(self.selected_freq, tuple):
            freq_label = f"({self.selected_freq[0]}-{self.selected_freq[1]} THz)"
        else:
            freq_label = f"({self.selected_freq} THz)"

        self.img_properties["quantity_label"] = " ".join([str(self.selected_quantity), freq_label * en_freq_label])

    def select_freq(self, freq):
        self.selected_freq = freq
        self.selected_freq_idx = self._selected_freq_idx()
        self._update_fig_num()

    def _selected_freq_idx(self):
        if self.selected_freq_idx is None:
            self.selected_freq_idx = np.argmin(np.abs(self.freq_axis - self.selected_freq))

        return self.selected_freq_idx

    def _pre_process(self, meas_):
        pp_opt = self.options["pp_opt"]

        idx = self.raw_data_cache["id_map"][meas_.identifier]
        data_td = self.raw_data_cache["td"][idx]

        if pp_opt["remove_dc"]:
            data_td = remove_offset(data_td)

        if pp_opt["window_opt"]["enabled"]:
            data_td = window(data_td, **pp_opt["window_opt"])

        if pp_opt["filter_opt"]["enabled"]:
            data_td = butter_filt(data_td, **pp_opt["filter_opt"])

        return data_td

    def _get_data(self, meas, domain=None):
        if domain is None:
            domain = Domain.Time

        data_td = self._pre_process(meas)

        if domain == Domain.Time:
            return data_td
        elif domain == Domain.Frequency:
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
        ref_td = self._get_data(measurement_)
        t, y = ref_td[:, 0], ref_td[:, 1]

        return t[np.argmax(y)]

    def _get_zero_crossing(self, measurement_):
        data_td = self._get_data(measurement_)
        t, y = data_td[:, 0], data_td[:, 1]
        a_max = np.argmax(y)

        zero_crossing_idx = 1
        for i in range(len(y)):
            if i < a_max or i == len(y) - 1:
                continue
            if y[i - 1] > 0 > y[i]:
                zero_crossing_idx = i
                break

        y1, y2 = y[zero_crossing_idx - 1], y[zero_crossing_idx]
        x1, x2 = t[zero_crossing_idx - 1], t[zero_crossing_idx]

        if np.isclose(y2-y1, 0):
            return 0
        else:
            return (y1 * x2 - x1 * y2) / (y1 - y2)

    def _p2p(self, meas_):
        y_td = self._get_data(meas_)
        return np.max(y_td[:, 1]) - np.min(y_td[:, 1])

    def _phase(self, meas_):
        y_fd = self._get_data(meas_, domain=Domain.Frequency)
        return np.angle(y_fd[self.selected_freq_idx, 1])

    def _power(self, meas_):
        if not isinstance(self.selected_freq, tuple):
            self.select_freq((1.0, 1.2))
            logging.warning(f"Selected_freq must be a tuple. Using default range ({self.selected_freq})")

        freq_range_ = self.selected_freq
        freq_slice = (freq_range_[0] < self.freq_axis) * (self.freq_axis < freq_range_[1])

        ref_fd = self.get_ref_data(point=meas_.position, domain=Domain.Frequency)
        sam_fd = self._get_data(meas_, domain=Domain.Frequency)

        power_val_ref = np.sum(np.abs(ref_fd[freq_slice, 1]) ** 2)
        power_val_sam = np.sum(np.abs(sam_fd[freq_slice, 1]) ** 2)

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
        data_td = self._get_data(meas_)
        y_ = data_td[:, 1]
        y_ -= (np.mean(y_[:10]) + np.mean(y_[-10:])) * 0.5

        y_[y_ < threshold] = 0
        peaks_idx = []
        for idx_ in range(1, len(y_) - 1):
            if (y_[idx_ - 1] < y_[idx_]) * (y_[idx_] > y_[idx_ + 1]):
                peaks_idx.append(idx_)

        return len(peaks_idx)

    def transmission(self, meas_, freq_range_=None, phase_sign=1):
        ref_fd = self.get_ref_data(point=meas_.position, domain=Domain.Frequency)

        sam_fd = self._get_data(meas_, Domain.Frequency)

        if freq_range_ is not None:
            t = sam_fd[:, 1] / ref_fd[:, 1]
        else:
            freq_idx = f_axis_idx_map(self.freq_axis, self.selected_freq)
            t = sam_fd[freq_idx, 1] / ref_fd[freq_idx, 1]

        if phase_sign != 1:
            phi = np.unwrap(np.angle(t))
            t = np.abs(t) * np.exp(phase_sign * 1j * phi)

        return t

    def _amplitude_transmission(self, meas_):
        t = self.transmission(meas_)

        return np.abs(t)[0]

    def _phase_transmission(self, meas_):
        t = self.transmission(meas_)

        return np.angle(t)[0]

    def _time_of_flight(self, meas_):
        closest_ref = self.find_nearest_ref(meas_)

        t_zero_ref = self._get_zero_crossing(closest_ref)
        t_zero_sam = self._get_zero_crossing(meas_)

        return np.abs(t_zero_ref - t_zero_sam)

    def single_layer_eval(self, meas_, freq_range=None):
        if self.options["sample_properties"]["default_values"]:
            logging.warning(f"Using default sample properties: {self.options['sample_properties']}")

        d = self.options["sample_properties"]["d"]

        og_win_setting = deepcopy(self.options["pp_opt"]["window_opt"])

        self.options["pp_opt"]["window_opt"]["enabled"] = True
        self.options["pp_opt"]["window_opt"]["win_width"] = 10
        self.options["pp_opt"]["window_opt"]["en_plot"] = False

        ref_fd = self.get_ref_data(point=meas_.position, domain=Domain.Frequency)
        sam_fd = self._get_data(meas_, Domain.Frequency)

        self.options["pp_opt"]["window_opt"] = og_win_setting

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
        alpha = 1e4 * 2 * omega * kap / c_thz
        # kap = -c_thz * np.log(np.abs(sam_fd[freq_idx, 1] / ref_fd[freq_idx, 1])) / (omega * d)

        ret = {"freq_axis": freq_axis,
               "refr_idx": n + 1j * kap, "alpha": alpha,
               "phi_ref": phi_ref, "phi_sam": phi_sam, "phi": phi, "phi_corrected": phi_corrected,
               }

        return ret

    def _refractive_idx(self, meas_):
        return np.mean(np.real(self.single_layer_eval(meas_)["refr_idx"]))

    def _extinction_coe(self, meas_):
        return np.mean(np.imag(self.single_layer_eval(meas_)["refr_idx"]))

    def _absorption_coef(self, meas_, freq_range=None):
        n_cmplx_res = self.single_layer_eval(meas_, freq_range)
        freq_axis = n_cmplx_res["freq_axis"]
        kap = n_cmplx_res["refr_idx"].imag

        omega = 2 * np.pi * freq_axis
        alph = (1 / 1e-4) * 2 * kap * omega / c_thz # 1/cm

        if freq_range is None:
            return np.mean(alph)
        else:
            return alph

    def _eval_sub(self):
        sub_pnt = self.options["eval_opt"]["sub_pnt"]
        sub_meas = self.sub_dataset.get_measurement(*sub_pnt)
        sub_res = self.sub_dataset.single_layer_eval(sub_meas, freq_range=(0, 10))
        sub_res["t_sub"] = self.sub_dataset.transmission(sub_meas, 1)

        return sub_res

    def _conductivity(self, meas_):
        sub_res = self._eval_sub()
        t_sam = self.transmission(meas_, 1)

        n_sub = sub_res["refr_idx"]
        t_sub = sub_res["t_sub"]
        d_film = self.options["sample_properties"]["d_film"]

        # [eps0_thz] = ps * Siemens / µm, [c_thz] = µm / ps, [1/d_film] = 1/um -> conversion: 1e4 (S/cm)
        # 1 / µm = 1 / (1e-6 m) = 1 / (1e-6 * 1e2 cm) = 1 / (1e-4 cm) = 1e4 * 1 / cm
        sigma = 1e4 * (1/d_film) * eps0_thz * c_thz * (1 + n_sub) * (t_sub/t_sam - 1)

        # phase correction, [dt] = fs
        dt = self.options["eval_opt"]["dt"]
        dt *= 1e-3
        sigma *= np.exp(-1j*dt*2*np.pi*self.freq_axis)

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
                    QuantityEnum.Phase: self._phase,
                    QuantityEnum.Power: self._power,
                    QuantityEnum.MeasTimeDeltaRef2Sam: self._meas_time_delta,
                    QuantityEnum.RefAmp: self._ref_max,
                    QuantityEnum.RefArgmax: self._get_ref_argmax,
                    QuantityEnum.RefPhase: self._ref_phase,
                    QuantityEnum.PeakCnt: partial(self._simple_peak_cnt, threshold=2.5),
                    QuantityEnum.ZeroCrossing: self._get_zero_crossing,
                    QuantityEnum.TimeOfFlight: self._time_of_flight,
                    QuantityEnum.TransmissionAmp: self._amplitude_transmission,
                    QuantityEnum.TransmissionPhase: self._phase_transmission,
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
        from random import choice
        # closest_ref = choice(self.measurements["refs"])

        logging.debug(f"Sam: {meas_})")
        logging.debug(f"Ref: {closest_ref})")
        if self.options["dist_func"] == Dist.Time:
            logging.debug(f"Time between ref and sample: {best_fit_val} seconds")
        else:
            logging.debug(f"Distance between ref and sample: {best_fit_val} mm")

        return closest_ref

    def get_ref_data(self, domain=Domain.Time, point=None, ref_idx=None, ret_meas=False):
        if point is not None:
            closest_sam = self.get_measurement(*point)
            chosen_ref = self.find_nearest_ref(closest_sam)
        else:
            if ref_idx is None:
                ref_idx = -1
            chosen_ref = self.measurements["refs"][ref_idx]

        # chosen_ref = np.random.choice(self.measurements["refs"])

        if domain in [Domain.Time, Domain.Frequency]:
            ret = self._get_data(chosen_ref, domain=domain)
        else:
            ret = self._get_data(chosen_ref, domain=Domain.Both)

        if ret_meas:
            return ret, chosen_ref
        else:
            return ret

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
        ref_before_td, ref_after_td = self._get_data(ref_before), self._get_data(ref_after)

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

    def meas_time_diff(self, m1, m2):
        # meas time difference in hours
        return (m2.meas_time - m1.meas_time).total_seconds() / 3600

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

        logging.info(f"Plotting point {point}")
        logging.info(f"Reference measurement: {ref_meas}")
        logging.info(f"Sample measurement: {sam_meas}")

        # TODO redo window plotting
        show_win_plot = deepcopy(self.options["pp_opt"]["window_opt"]["en_plot"])
        if self.options["shown_plots"]["Window"]:
            self.options["pp_opt"]["window_opt"]["en_plot"] = True

        self.options["pp_opt"]["window_opt"]["fig_label"] = "ref"
        ref_td, ref_fd = self._get_data(ref_meas, domain=Domain.Both)

        #ref_fd[:, 1] = np.abs(ref_fd[:, 1]) * np.exp(-1j*np.angle(ref_fd[:, 1]))
        #ref_td = do_ifft(ref_fd, conj=False)
        self.options["pp_opt"]["window_opt"]["fig_label"] = "sam"
        sam_td, sam_fd = self._get_data(sam_meas, domain=Domain.Both)

        if self.options["plot_opt"]["shift_sam2ref"]:
            shift_t = np.abs(np.argmax(ref_td[:, 1]) - np.argmax(sam_td[:, 1]))
            sam_td[:, 1] = np.roll(sam_td[:, 1], -shift_t)

        self.options["pp_opt"]["window_opt"]["en_plot"] = show_win_plot

        if remove_t_offset:
            ref_td[:, 0] -= ref_td[0, 0]
            sam_td[:, 0] -= sam_td[0, 0]

        freq_axis = ref_fd[:, 0].real

        t = sam_fd[:, 1] / ref_fd[:, 1]
        absorb = np.abs(1/t)

        f_min, f_max = freq_axis[plot_range][0], freq_axis[plot_range][-1]
        simple_eval_res = self.single_layer_eval(sam_meas, (f_min, f_max))
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
            label = f"(x,y)=({point[0]}, {point[1]}) mm"

        freq_axis = sam_fd[:, 0].real
        noise_floor = np.mean(20 * np.log10(np.abs(sam_fd[sam_fd[:, 0] > 6.0, 1]))) * sub_noise_floor

        plt.figure("Spectrum")
        y_db = (20 * np.log10(np.abs(sam_fd[plot_range, 1])) - noise_floor).real
        plt.plot(freq_axis[plot_range], y_db, label=label)

        plt.figure("Phase")
        plt.plot(freq_axis[plot_range], phi, label=label + " (Original)", ls="dashed")
        plt.plot(freq_axis[plot_range], phi_corrected, label=label + " (Corrected)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (rad)")

        plt.figure("Phase slope")
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
            # plot_range = slice(30, 550)
            plt.figure("Conductivity")
            plt.title(label)
            plt.plot(freq_axis[plot_range], sigma[plot_range].real, label="Real part")
            plt.plot(freq_axis[plot_range], sigma[plot_range].imag, label="Imaginary part")
            # plt.ylim((-1e3, 1.5e5))
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Conductivity (S/cm)")

        return ret

    def plot_meas_phi_diff(self, pnt0, pnt1, label=""):
        plot_range = self.options["plot_range"]

        sam_meas0 = self.get_measurement(*pnt0)
        ref_meas0 = self.find_nearest_ref(sam_meas0)

        sam_meas1 = self.get_measurement(*pnt1)

        ref_fd = self._get_data(ref_meas0, domain=Domain.Frequency)
        freq_axis = ref_fd[:, 0].real

        f_min, f_max = freq_axis[plot_range][0], freq_axis[plot_range][-1]
        simple_eval_res0 = self.single_layer_eval(sam_meas0, (f_min, f_max))
        simple_eval_res1 = self.single_layer_eval(sam_meas1, (f_min, f_max))
        phi0 = simple_eval_res0["phi_corrected"]
        phi1 = simple_eval_res1["phi_corrected"]

        plt.figure("Phi difference")
        plt.plot(freq_axis[plot_range], phi0-phi1, label=label)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase difference (rad)")

    def plot_system_stability(self, climate_log_file=None):
        first_meas = self.measurements["all"][0]
        if all([first_meas.position == meas.position for meas in self.measurements["all"]]):
            meas_set = self.measurements["all"]
            logging.info("Using the full dataset")
        else:
            meas_set = self.measurements["refs"]
            logging.info("Using reference set")

        selected_freq_ = self.selected_freq
        if isinstance(selected_freq_, tuple):
            selected_freq_ = selected_freq_[0]
        f_idx = np.argmin(np.abs(self.freq_axis - selected_freq_))

        ref_ampl_arr, ref_angle_arr, ref_zero_crossing = np.zeros((3, len(meas_set)))
        t0 = meas_set[0].meas_time

        meas_times = np.array([self.meas_time_diff(meas_set[0], m) for m in meas_set])

        if meas_times.max() < 5 / 60:
            meas_times *= 3600
            mt_unit = "seconds"
        elif 5 / 60 <= meas_times.max() < 0.5:
            meas_times *= 60
            mt_unit = "minutes"
        else:
            mt_unit = "hours"

        for i, ref in enumerate(meas_set):
            ref_td, ref_fd = self._get_data(ref, domain=Domain.Both)

            ref_zero_crossing[i] = self._get_zero_crossing(ref)
            ref_ampl_arr[i] = np.sum(np.abs(ref_fd[f_idx, 1]))
            ref_angle_arr[i] = -np.angle(ref_fd[f_idx, 1])

        ref_ampl_arr = 100 * (ref_ampl_arr[0] - ref_ampl_arr) / ref_ampl_arr[0]

        meas_interval = np.mean(np.diff(meas_times))
        ref_angle_arr = np.unwrap(ref_angle_arr)

        minima = local_minima_1d(ref_angle_arr, en_plot=False)
        period, period_std = minima[1] * meas_interval * 60, minima[2] * meas_interval * 60

        ref_zero_crossing = (ref_zero_crossing - ref_zero_crossing[0]) * 1000

        # correction
        # ref_angle_arr -= 2*np.pi*self.freq_axis[f_idx]*(ref_zero_crossing/1000)

        abs_p_shifts = np.abs(np.diff(ref_zero_crossing))
        logging.info(f"Mean pulse shift: {np.round(np.mean(abs_p_shifts), 2)} fs")
        max_diff_0x, min_diff_0x = np.max(abs_p_shifts), np.min(abs_p_shifts)
        logging.info(f"Largest/smallest shift: {np.round(max_diff_0x, 2)}/{np.round(min_diff_0x, 2)} fs")

        max_diff, argmax_diff = np.max(np.diff(ref_angle_arr)), np.argmax(np.diff(ref_angle_arr))
        phase_str = f"Largest phase jump: {np.round(max_diff, 2)} rad"
        phase_str += f" (time: {np.round(meas_times[argmax_diff], 2)} {mt_unit})"
        phase_str += f" (at {selected_freq_} THz)"
        logging.info(phase_str)

        avg_amp_change = np.mean(np.abs(np.diff(ref_ampl_arr)))
        max_amp_change = np.max(np.diff(ref_ampl_arr))

        logging.info(f"Largest amplitude change: {np.round(max_amp_change, 2)} (Arb. u.)")
        logging.info(f"Mean absolute amplitude change: {np.round(avg_amp_change, 2)} (Arb. u.)")
        logging.info(f"Mean measurement interval: {np.round(meas_interval*3600, 2)} sec.")
        logging.info(f"Period (estimation): {np.round(period, 2)}±{np.round(period_std, 2)} min.")

        plt.figure("fft")
        phi_fft = np.fft.rfft(ref_angle_arr)
        phi_fft_f = np.fft.rfftfreq(len(ref_angle_arr), d=meas_interval * 3600)

        plt.plot(phi_fft_f[1:], np.abs(phi_fft)[1:])
        plt.xlabel("Frequency (1/hour)")
        plt.ylabel("Magnitude")

        from random import choice
        idx = choice(range(len(meas_set)))
        ref0, ref1 =  meas_set[idx], meas_set[idx+1]
        ref0_fd, ref1_fd = self._get_data(ref0, domain=Domain.Frequency), self._get_data(ref1, domain=Domain.Frequency)
        phi0, phi1 = np.angle(ref0_fd[:, 1]), np.angle(ref1_fd[:, 1])
        amp0, amp1 = np.abs(ref0_fd[:, 1]), np.abs(ref1_fd[:, 1])
        w = 2*np.pi*self.freq_axis

        plt.figure("Pulse shift")
        plt.plot(self.freq_axis, 1e3*(phi0-phi1)/w, label=idx)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Time (fs)")

        plt.figure("Amp change")
        plt.plot(self.freq_axis, amp0-amp1)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude change (Arb. u.)")

        plt.figure("Reference zero crossing")
        plt.title(f"Reference zero crossing\n(relative to first measurement)")
        plt.plot(meas_times, ref_zero_crossing, label=t0)
        plt.xlabel(f"Measurement time ({mt_unit})")
        plt.ylabel("Time (fs)")

        plt.figure("Reference zero crossing change")
        plt.title(f"Reference zero crossing change")
        plt.plot(meas_times[1:], abs_p_shifts, label=t0)
        phase_change = np.abs(np.diff(ref_angle_arr))
        # plt.plot(meas_times[1:], 1e3*phase_change/(2*3.1415*selected_freq_), label=t0)
        plt.xlabel(f"Measurement time ({mt_unit})")
        plt.ylabel("Time (fs)")

        plt.figure("Stability amplitude")
        plt.title(f"Amplitude of reference measurement at {selected_freq_} THz")
        plt.plot(meas_times, ref_ampl_arr)
        plt.xlabel(f"Measurement time ({mt_unit})")
        plt.ylabel("Relative amplitude change (%)")

        plt.figure("Stability phase")
        plt.title(f"Phase of reference measurement at {selected_freq_} THz")
        plt.plot(meas_times, ref_angle_arr)
        plt.xlabel(f"Measurement time ({mt_unit})")
        plt.ylabel("Phase (rad)")

        plt.figure("Time between reference measurements")
        plt.title(f"Time between reference measurements")
        plt.plot(meas_times[1:], np.diff(meas_times)*3600)
        plt.ylabel("Time difference (s)")
        plt.xlabel("Time since first measurement (h)")

        if climate_log_file is not None:
            self.plot_climate(climate_log_file)

    def plot_frequency_noise(self):
        ref_meas_set = self.measurements["refs"]
        freq_axis = self.freq_axis

        ampl_arr_db = np.zeros((len(ref_meas_set), len(freq_axis)))
        for i, ref in enumerate(ref_meas_set):
            ref_td, ref_fd = self._get_data(ref, domain=Domain.Both)
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
                    if i % 150: # Sampling time: 2 sec (= 0.5 Hz) -> 300 * 2 = 600 sec
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

        if self.measurements["refs"] is not None:
            t0 = self.measurements["refs"][0].meas_time
            tf = self.measurements["refs"][-1].meas_time
            tf_idx = np.argmin(np.abs([(tf - t).total_seconds() for t in meas_time]))
        else:
            t0 = meas_time[0]
            tf_idx = len(meas_time)

        meas_time_diff = [(t - t0).total_seconds() / 3600 for t in meas_time]

        if quantity == ClimateQuantity.Temperature:
            quant = temp
            y_label = "Temperature (°C)"
        else:
            quant = humidity
            y_label = "Humidity (\\%)"

        meas_time_diff = meas_time_diff[:tf_idx]
        quant = quant[:tf_idx]

        stability_figs = ["Reference zero crossing", "Stability amplitude", "Stability phase"]
        for fig_label in stability_figs:
            if plt.fignum_exists(fig_label):
                fig = plt.figure(fig_label)
                ax_list = fig.get_axes()
                ax1 = ax_list[0]
                ax1.tick_params(axis="y", colors="blue")
                ax1.set_ylabel(ax1.get_ylabel(), c="blue")
                # ax1.grid(c="blue")
                ax1.grid(False)

                ax2 = ax1.twinx()
                ax2.plot(meas_time_diff, quant, c="red")
                ax2.set_ylabel(y_label, c="red")
                ax2.tick_params(axis="y", colors="red")
                ax2.grid(False)

        if not plt.fignum_exists(stability_figs[0]):
            fig, ax1 = plt.subplots(num="Climate plot")
            ax1.scatter(meas_time_diff, quant, label=f"Start: {t0}")
            ax1.set_xlabel("Measurement time (hour)")
            ax1.set_ylabel(y_label)

    def plot_image(self, img_extent=None):
        self._update_fig_num()
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

        self._update_quantity_label()
        quantity_label = self.img_properties["quantity_label"]

        img_title_option = str(self.options["img_title"])
        ax.set_title(" ".join([quantity_label, img_title_option]))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = fig.colorbar(img_, cax=cax)
        cbar.set_ticks(np.round(np.linspace(cbar_min, cbar_max, 4), 3))

        if self.options["en_cbar_label"]:
            cbar.set_label(quantity_label, rotation=270, labelpad=30)

        self.img_properties["img_ax"] = ax
        self.img_properties["plotted_image"] = True

    def _plot_meas_on_image(self, measurements):
        if not plt.fignum_exists(self.img_properties["fig_num"]):
            return

        plt.figure(num=self.img_properties["fig_num"])
        img_ax = self.img_properties["img_ax"]

        meas_x_coords, meas_y_coords = [], []
        for m in measurements:
            meas_x_coords.append(m.position[0])
            meas_y_coords.append(m.position[1])

        plt_fun = img_ax.scatter

        plt_fun(meas_x_coords, meas_y_coords, color="black", linewidth=0.4)

    def plot_refs(self):
        self._plot_meas_on_image(self.measurements["refs"])

    def plot_line(self, line_coords=None, direction=Direction.Horizontal, **plot_kwargs):
        if line_coords is None:
            line_coords = [0.0]
        if isinstance(line_coords, (int, float)):
            line_coords = [line_coords]

        horizontal = direction == direction.Horizontal

        if horizontal:
            fig_num = "x-slice"
            x_label = "x (mm)"
        else:
            fig_num = "y-slice"
            x_label = "y (mm)"

        fig_num += "_" + self.img_properties["quantity_label"].replace(" ", "_")
        plt.figure(fig_num)
        plt.title(f"Line scan ({direction.name})")
        plt.xlabel(x_label)
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
                plot_kwargs["label"] = f"y={line_coord} (mm)"
                plt.plot(coords, vals, **plot_kwargs)
            else:
                plot_kwargs["label"] = f"x={line_coord} (mm)"
                plt.plot(coords, vals, **plot_kwargs)

        self._plot_meas_on_image(measurements)

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
        dataset_._is_sub_dataset = True
        self.sub_dataset = dataset_

    def _is_figure_open(self, num):
        # not used
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            if num == fig.get_label():
                return True
        return False

    def save_fig(self, fig_num_, filename=None, **kwargs):
        save_dir = Path(self.options["result_dir"])

        fig = plt.figure(fig_num_)

        if filename is None:
            try:
                filename_s = str(fig.canvas.get_window_title())
            except AttributeError:
                filename_s = str(fig.canvas.manager.get_window_title())
        else:
            filename_s = str(filename)

        illegal_chars = ["(", ")"]
        for char in illegal_chars:
            filename_s = filename_s.replace(char, '')
        filename_s.replace(" ", "_")

        fig.set_size_inches((12, 9), forward=False)
        plt.subplots_adjust(wspace=0.3)
        plt.savefig(save_dir / (filename_s + ".pdf"), bbox_inches='tight', dpi=300, pad_inches=0, **kwargs)

    def plt_show(self):

        # fig_labels = [plt.figure(fig_num).get_label() for fig_num in plt.get_fignums()]
        only_shown_fig_nums = []
        if self.options["only_shown_figures"]:
            only_shown_fig_nums = self.options["only_shown_figures"]
            logging.warning(f"Only showing figures {self.options['only_shown_figures']}")

        not_shown = []
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            fig_label = fig.get_label()
            for ax in fig.get_axes():
                h, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend()

            if self.options["save_plots"]:
                self.save_fig(fig_num)

            if only_shown_fig_nums and fig_label not in only_shown_fig_nums:
                not_shown.append(fig_label)
                plt.close(fig_num)
                continue
            if fig_label in self.options["shown_plots"]:
                if not self.options["shown_plots"][fig_label]:
                    not_shown.append(fig_label)
                    plt.close(fig_num)

        logging.info(f"Not showing plots: {', '.join(not_shown)}")
        plt.show()

    def ref_difference_plot(self,):
        ref1, ref2 = self.measurements["refs"][11], self.measurements["refs"][16]

        print(ref1)
        print(ref2)

        ref1_fd = self._get_data(ref1, domain=Domain.Frequency)
        ref2_fd = self._get_data(ref2, domain=Domain.Frequency)

        freq = ref1_fd[:, 0]

        phi1 = np.angle(ref1_fd[:, 1])
        phi2 = np.angle(ref2_fd[:, 1])

        phi1_unwrap = np.unwrap(phi1)
        phi2_unwrap = np.unwrap(phi2)

        diff = phi1_unwrap - phi2_unwrap
        diff_diff = np.append(np.diff(diff), 0)

        plt.figure("Unwrapped phases")
        plt.plot(freq, phi1_unwrap, label="Ref 1")
        plt.plot(freq, phi2_unwrap, label="Ref 2")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Unwrapped phase (rad)")

        plt.figure("Phase difference")
        plt.scatter(freq, diff_diff)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase difference (rad)")

        mark_x = [self.meas_time_diff(self.measurements["refs"][0], ref1),
                  self.meas_time_diff(self.measurements["refs"][0], ref2)]
        mark_y = [phi1[self._selected_freq_idx],
                  phi2[self._selected_freq_idx]]

        plt.figure("Stability phase")
        plt.scatter(mark_x, mark_y, color="red", s=30, zorder=99)

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

    dataset.plt_show()
