from common.components import ComponentBase
from common.default_appsettings import QuantityFunc, ClimateQuantity, Direction, Dist
from common.settings import Settings
from copy import deepcopy
from pathlib import Path
import numpy as np
from functions import unwrap, window, local_minima_1d, WindowTypes, butter_filt
from functions import phase_correction, do_fft, do_ifft, f_axis_idx_map, remove_offset
from functions import arr_statistics
from common.measurements import Measurement, meas_id_func
from mpl_settings import mpl_style_params
import matplotlib as mpl
from common.consts import c_thz, eps0_thz
import logging
import colorlog
from datetime import datetime
from common.dataset_cache import DatasetCache
import pandas as pd
from scipy.signal import correlate
from scipy.stats import pearsonr
from q_space_eval import QSpaceEval
from common.default_appsettings import Domain, MeasurementType, QuantityEnum, AppSettings
from common.components import action
import itertools

"""
TODOs: 
- How are measurements mapped when multiple measurements are performed at the same x-y coordinates?
- Setting cbar lims sucks... set lims based on area min max?
- Fix runtime / use cache for t calc?
- check if filesizes? match when making npy array
- window function (fuctions.py): allow negative values (wrap around) + fix plot (clipping)
- Logging is messy, also fix log levels and RuntimeWarnings
- freq_range variable in transmission function (and other functions?)
- combine the different reference select settings into one dict
- Fix unit labeling
- Split DataSet into multiple smaller classes (e.g. a plotting class) "Classes should do one thing each."
- should phi correction be a part of the pre-processing?
- rename some keys in options dict e.g. "eval_opt" to "eval"
- make Dataset(dict)?
- q-eval: svmaf, estimate #FP reflections and FP spacing for q-space freq range
- fix default_options["sample_properties"]["default_values"]

New ideas: add teralyzer evaluation (time consuming)
- Add plt_show here (done)
- Possibly add marker in image to show where .plot_point() is
- interactive imshow plots -> maybe connect to .plot_point()

# units:
[l] = µm, [t] = ps, [alpha] = 1/cm (absorption coe.), [sigma] = S/cm, [eps0] = Siemens * ps,
[f] = THz (1/ps), [c_thz] = µm/ps
"""


def logger_config(settings):
    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s: %(message)s",
        log_colors={
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )

    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.addHandler(handler)

    log_level = settings.log_level.value
    logger.setLevel(log_level)

class DataSet(ComponentBase):
    def __init__(self, data_path : Path | str, settings : Settings):
        super().__init__()
        self.plotted_ref = False
        self.noise_floor = None
        self.time_axis = None
        self.freq_axis = None
        self.properties = {"data": {}, "shape": {}, }
        self.measurements = {"refs": [], "sams": [], "all": []}

        self.sub_dataset = None
        self.is_sub_dataset = False

        self.data_path = self._set_path(data_path)

        self.settings = settings

        self._parse_measurements()

        self._check_refs_exist()

    @action("print settings")
    def print_settings(self):
        print(self.settings.pp_opt.filter_enabled)

    def _data_properties(self):
        sample_data_td, sample_data_fd = self.get_data(self.measurements["all"][0], domain=Domain.Both)
        samples = int(sample_data_td.shape[0])

        self.time_axis = sample_data_td[:, 0].real
        self.freq_axis = sample_data_fd[:, 0].real

        dt = np.mean(np.diff(self.time_axis))

        self.properties["data"] = {"dt": dt, "samples": samples}

    def _shape_properties(self):
        x_coords, y_coords = [], []
        for sam_measurement in self.measurements["sams"]:
            x_coords.append(sam_measurement.position[0])
            y_coords.append(sam_measurement.position[1])

        if not x_coords: x_coords.append(0)
        if not y_coords: y_coords.append(0)

        x_coords = np.array(sorted(set(x_coords)))
        y_coords = np.array(sorted(set(y_coords)))

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

        self.properties["shape"] = {"w": w, "h": h, "dx": dx, "dy": dy, "extent": extent,
                                    "x_coords": x_coords, "y_coords": y_coords, "all_points": all_points}

    def _set_path(self, data_path):
        data_path = Path(data_path)

        if not data_path.exists():
            raise ValueError(f"Path {data_path} does not exist")
        if not data_path.is_dir():
            raise ValueError(f"Path {data_path} is not a directory")
        if not list(data_path.glob("*")):
            raise ValueError(f"Path {data_path} is empty")

        return data_path

    def find_climate_log_file(self, climate_log_file):
        checked_dirs = [self.data_path, self.data_path.parent]
        log_files = []
        log_files.extend([file for file in checked_dirs[0].iterdir() if "log" in file.name])
        log_files.extend([file for file in checked_dirs[1].iterdir() if "log" in file.name])

        for log_file in log_files:
            if str(climate_log_file) in log_file.name:
                return log_file

        return None

    def _apply_options(self):
        new_rc_params = {"savefig.directory": self.settings.result_dir}

        mpl.rcParams.update(mpl_style_params(new_rc_params))
        logger_config(self.settings)

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

        return {"refs": refs_sorted, "sams": sams_sorted, "all": all_sorted}

    def _timestamp2id(self, timestamp_str):
        timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S.%f")

        return meas_id_func(timestamp_dt)

    def _parse_measurements(self):
        if not isinstance(self.data_path, Path):
            self.data_path = Path(self.data_path)

        all_measurements = self._read_data_dir()

        self.measurements = self._sort_measurements(all_measurements)

        self.cache = DatasetCache(self.measurements["all"], self.data_path)

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
                     f"{min_part} minute(s) and {sec_part} second(s) ({td_secs} seconds)\n")

        time_diffs = [(self.measurements["all"][i + 1].meas_time -
                       self.measurements["all"][i].meas_time).total_seconds()
                      for i in range(0, len(self.measurements["all"]) - 1)]

        mean_time_diff = np.mean(time_diffs)

        self.properties["mean_time_diff"] = mean_time_diff
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

        self._data_properties()
        self._shape_properties()

    def _check_refs_exist(self):
        if self.measurements["refs"] or not self.measurements["sams"]:
            return
        logging.info(f"No explicit references in the dataset. Using ref_pos option")

        threshold = self.settings.ref_threshold

        max_amp_meas = self.measurements["max_amp_meas"]
        max_amp = np.max(np.abs(self.get_data(max_amp_meas)[:, 1]))

        manual_pos = self.settings.ref_pos

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

        self.measurements["refs"] = tuple(refs_)

        logging.info(f"Found {len(self.measurements['refs'])} possible reference measurements")
        logging.info("######################################################\n")

    def _pre_process(self, meas_):
        pp_opt = self.settings.pp_opt

        cache_idx = self.cache.id_map[meas_.identifier]
        data_td = self.cache.raw_data_td[cache_idx]

        if pp_opt.remove_dc:
            data_td = remove_offset(data_td)

        if pp_opt.window_enabled:
            data_td = window(data_td, **pp_opt.traits())

        if pp_opt.filter_enabled:
            data_td = butter_filt(data_td, pp_opt.f_range)

        return data_td

    def get_data(self, meas, domain=None):
        if domain is None:
            domain = Domain.Time

        data_td = self._pre_process(meas)

        if domain == Domain.Time:
            return data_td
        elif domain == Domain.Frequency:
            return do_fft(data_td)
        else:
            return data_td, do_fft(data_td)

    def _get_multi_data(self, meas_list, domain=Domain.Both):
        y0_td = self.get_data(meas_list[0])
        y0_fd = do_fft(y0_td)

        t_axis, f_axis = y0_td[:, 0], y0_fd[:, 0]

        data_td = np.zeros([len(meas_list), *y0_td.shape])
        data_fd = np.zeros([len(meas_list), *y0_fd.shape], dtype=complex)

        for meas_idx, meas in enumerate(meas_list):
            data_td[meas_idx] = self._pre_process(meas)
            data_fd[meas_idx] = do_fft(data_td[meas_idx])

        if domain == Domain.Time:
            return data_td
        elif domain == Domain.Frequency:
            return data_fd
        else:
            return data_td, data_fd

    def get_ref_argmax(self, measurement_):
        ref_td = self.get_data(measurement_)
        t, y = ref_td[:, 0], ref_td[:, 1]

        return t[np.argmax(y)]

    def spectral_similarity(self, meas_0, meas_1, freq_min=0.15, freq_max=2.00):
        data_fd_0 = self.get_data(meas_0, domain=Domain.Frequency)
        data_fd_1 = self.get_data(meas_1, domain=Domain.Frequency)

        f_idx_range = f_axis_idx_map(self.freq_axis, (freq_min, freq_max))

        x, y = np.abs(data_fd_0[f_idx_range, 1]), np.abs(data_fd_1[f_idx_range, 1])

        return 1+np.log(np.abs(pearsonr(x, y).statistic))

    def delay_from_phaseslope(self, meas_0, meas_1, freq_min=0.15, freq_max=0.85):
        data_td_0 = self.get_data(meas_0)
        t0, y0 = data_td_0[:, 0], data_td_0[:, 1]

        data_td_1 = self.get_data(meas_1)
        t1, y1 = data_td_1[:, 0], data_td_1[:, 1]

        dt = t0[1] - t0[0]
        N = len(t0)

        y0 = y0 - np.mean(y0)
        y1 = y1 - np.mean(y1)

        Y0 = np.fft.fft(y0)
        Y1 = np.fft.fft(y1)

        # Y1 = (0.95*np.abs(Y1)) * np.exp(1j*np.angle(Y1))

        freqs = np.fft.fftfreq(N, dt)
        omega = 2 * np.pi * freqs

        H = Y1 / Y0

        phase = np.unwrap(np.angle(H))

        mask = freqs > 0

        if freq_min is not None:
            mask &= freqs > freq_min

        if freq_max is not None:
            mask &= freqs < freq_max

        omega_fit = omega[mask]
        phase_fit = phase[mask]

        p = np.polyfit(omega_fit, phase_fit, 1)

        slope = p[0]

        tau = -slope
        # print(tau)
        return tau

    def _get_cross_correlation_delay(self, meas_0, meas_1):
        upsample = 10
        data_td_0 = self.get_data(meas_0)
        t0, y0 = data_td_0[:, 0], data_td_0[:, 1]

        data_td_1 = self.get_data(meas_1)
        t1, y1 = data_td_1[:, 0], data_td_1[:, 1]

        dt = t0[1] - t0[0]

        y0 = y0 - np.mean(y0)
        y1 = y1 - np.mean(y1)

        n = len(y0) + len(y1)

        Y0 = np.fft.fft(y0, n * upsample)
        Y1 = np.fft.fft(y1, n * upsample)

        corr = np.fft.ifft(Y1 * np.conj(Y0))
        corr = np.abs(np.fft.fftshift(corr))

        lags = np.arange(-len(corr) // 2, len(corr) // 2)

        k = np.argmax(corr)

        lag = lags[k] / upsample

        delay = lag * dt
        # print(delay)
        return delay

    def get_zero_crossing(self, measurement_):
        data_td = self.get_data(measurement_)
        t, y = data_td[:, 0], data_td[:, 1]
        y_abs_max = np.argmax(np.abs(y))
        # y_abs_max = np.argmax(y)

        zero_crossing_idx = 1
        for i in range(len(y)):
            if i < y_abs_max or i == len(y) - 1:
                continue
            if np.sign(y[i - 1]) * np.sign(y[i]) < 0: # sign change ++=+, --=+, +-=-, -+=-
            # if y[i - 1] > 0 > y[i]:  # sign change ++=+, --=+, +-=-, -+=-
                zero_crossing_idx = i
                break

        y1, y2 = y[zero_crossing_idx - 1], y[zero_crossing_idx]
        x1, x2 = t[zero_crossing_idx - 1], t[zero_crossing_idx]

        if np.isclose(y2-y1, 0):
            return 0
        else:
            zero_crossing_interp = (y1 * x2 - x1 * y2) / (y1 - y2)
            # print(zero_crossing_interp)
            return zero_crossing_interp

    def p2p(self, meas_: Measurement):
        y_td = self.get_data(meas_)
        return np.max(y_td[:, 1]) - np.min(y_td[:, 1])

    def phase(self, meas_: Measurement):
        y_fd = self.get_data(meas_, domain=Domain.Frequency)
        return np.angle(y_fd[:, 1])

    def power(self, meas_: Measurement, freq_range: tuple):
        freq_slice = (freq_range[0] < self.freq_axis) * (self.freq_axis < freq_range[1])

        ref_fd = self.get_ref_data(point=meas_.position, domain=Domain.Frequency)
        sam_fd = self.get_data(meas_, domain=Domain.Frequency)

        power_val_ref = np.sum(np.abs(ref_fd[freq_slice, 1]) ** 2)
        power_val_sam = np.sum(np.abs(sam_fd[freq_slice, 1]) ** 2)

        return power_val_sam / power_val_ref

    def meas_time_delta(self, meas_: Measurement):
        ref_meas = self.find_nearest_ref(meas_)

        return (meas_.meas_time - ref_meas.meas_time).total_seconds()

    def ref_max(self, meas_: Measurement):
        amp_, _ = self._ref_interpolation(meas_)

        return amp_

    def ref_phase(self, meas_: Measurement):
        _, phi_ = self._ref_interpolation(meas_)

        return phi_

    def simple_peak_cnt(self, meas_: Measurement, threshold: float):
        data_td = self.get_data(meas_)
        y_ = data_td[:, 1]
        y_ -= (np.mean(y_[:10]) + np.mean(y_[-10:])) * 0.5

        y_[y_ < threshold] = 0
        peaks_idx = []
        for idx_ in range(1, len(y_) - 1):
            if (y_[idx_ - 1] < y_[idx_]) * (y_[idx_] > y_[idx_ + 1]):
                peaks_idx.append(idx_)

        return len(peaks_idx)

    def transmission(self, meas_, phase_sign=1):
        ref_fd = self.get_ref_data(point=meas_.position, domain=Domain.Frequency)

        sam_fd = self.get_data(meas_, Domain.Frequency)

        t = sam_fd[:, 1] / ref_fd[:, 1]

        if phase_sign != 1:
            phi = np.unwrap(np.angle(t))
            t = np.abs(t) * np.exp(phase_sign * 1j * phi)

        return t

    def amplitude_transmission(self, meas_):
        t = self.transmission(meas_)

        return np.abs(t)

    def phase_transmission(self, meas_):
        t = self.transmission(meas_)

        return np.angle(t)

    def time_of_flight(self, meas_):
        closest_ref = self.find_nearest_ref(meas_)

        t_zero_ref = self.get_zero_crossing(closest_ref)
        t_zero_sam = self.get_zero_crossing(meas_)

        return np.abs(t_zero_ref - t_zero_sam)

    def _calc_ndim_quant(self, *arrs, op, out_like=None):
        ndim = arrs[0].ndim

        if out_like is not None:
            val = out_like.copy()
        else:
            val = arrs[0].copy()

        if ndim == 1:
            val = op(*arrs)
        elif ndim == 2:
            val[:, 1] = op(*(a for a in arrs))
        else:
            for i in range(arrs[0].shape[0]):
                val[i, :, 1] = op(*(a[i, :, :] for a in arrs))

        return val

    def _calc_phi(self, ref_td_, sam_td_, ref_fd_, sam_fd_):
        phi_fit_range = self.settings.eval_opt.phi_fit_range

        t_axis, f_axis = ref_td_[:, 0], ref_fd_[:, 0].real
        w_axis = 2*np.pi*f_axis

        t0_ref_idx = np.argmax(np.abs(ref_td_[:, 1]))
        t0_sam_idx = np.argmax(np.abs(sam_td_[:, 1]))

        t0_ref, t0_sam = t_axis[t0_ref_idx], t_axis[t0_sam_idx]
        t_offset = ref_td_[0, 0] - sam_td_[0, 0]

        phi0_ref, phi0_sam = w_axis * t0_ref, w_axis * t0_sam

        phi_r_ref = np.angle(ref_fd_[:, 1] * np.exp(-1j * phi0_ref))
        phi_r_sam = np.angle(sam_fd_[:, 1] * np.exp(-1j * w_axis * t0_sam))

        phi0_star = np.unwrap(phi_r_sam - phi_r_ref)

        fit_slice = (f_axis >= phi_fit_range[0]) * (f_axis <= phi_fit_range[1])
        p = np.polyfit(f_axis[fit_slice], phi0_star[fit_slice], 1)

        phi0 = phi0_star - 2*np.pi * int(p[1] / (2*np.pi))

        if not self.settings.eval_opt.phi_offset_correction:
            return phi0_star

        phi = phi0 - phi0_ref + phi0_sam + t_offset

        return phi

    def calc_meas_quantities(self, ref_meas_, meas_):
        meas_quants = {}
        meas_quants["freq_axis"] = self.freq_axis

        is_avg_eval = self.settings.eval_opt.average
        if not is_avg_eval:
            logging.info("Single measurement evaluation")
            logging.info(f"Reference measurement: {ref_meas_}")
            logging.info(f"Sample measurement: {meas_}")

            ref_td, ref_fd = self.get_data(ref_meas_, Domain.Both)
            sam_td, sam_fd = self.get_data(meas_, Domain.Both)

        else:
            ref_meas_list = self.get_consecutive_meas(ref_meas_)
            sam_meas_list = self.get_measurement(*meas_.position, return_single=False)

            logging.info("Average measurement evaluation")
            logging.info(f"Reference measurement list (count: {len(ref_meas_list)}):")
            logging.info(f"{[meas.filepath.name for meas in ref_meas_list]}")
            logging.info(f"Sample measurement list (count {len(sam_meas_list)}): ")
            logging.info(f"{[meas.filepath.name for meas in sam_meas_list]}")

            ref_td, ref_fd = self._get_multi_data(ref_meas_list, Domain.Both)
            sam_td, sam_fd = self._get_multi_data(sam_meas_list, Domain.Both)

        meas_quants["ref_td"], meas_quants["ref_td_std"] = arr_statistics(ref_td)
        meas_quants["sam_td"], meas_quants["sam_td_std"] = arr_statistics(sam_td)
        meas_quants["ref_fd"], meas_quants["ref_fd_std"] = arr_statistics(ref_fd)
        meas_quants["sam_fd"], meas_quants["sam_fd_std"] = arr_statistics(sam_fd)

        t_exp_amp = self._calc_ndim_quant(sam_fd, ref_fd, op=lambda a, b: np.abs(np.divide(a[:, 1], b[:, 1])))
        t_exp_phi = self._calc_ndim_quant(ref_td, sam_td, ref_fd, sam_fd, op=self._calc_phi, out_like=ref_fd)
        t_exp = self._calc_ndim_quant(t_exp_amp, t_exp_phi, op=lambda a, b: a[:, 1] * np.exp(-1j*b[:, 1]))

        meas_quants["t_exp_amp"], meas_quants["t_exp_amp_std"] = arr_statistics(t_exp_amp)
        meas_quants["t_exp_phi"], meas_quants["t_exp_phi_std"] = arr_statistics(t_exp_phi)
        meas_quants["t_exp"], meas_quants["t_exp_std"] = arr_statistics(t_exp)

        return meas_quants

    def single_layer_eval(self, meas_):
        if self.settings.sample_properties.default_values:
            logging.warning(f"Using default sample properties: {self.settings.sample_properties}")

        d = self.settings.sample_properties.d

        og_pp_opt = deepcopy(self.settings.pp_opt)

        self.settings.pp_opt.enabled = True
        self.settings.pp_opt.win_width = 10
        self.settings.pp_opt.win_start = None
        self.settings.pp_opt.en_plot = False

        ref_td, ref_fd = self.get_ref_data(point=meas_.position, domain=Domain.Both)
        sam_td, sam_fd = self.get_data(meas_, Domain.Both)

        self.settings.pp_opt = og_pp_opt

        freq_axis = self.freq_axis

        phi_ref = np.unwrap(np.angle(ref_fd[:, 1]))
        phi_sam = np.unwrap(np.angle(sam_fd[:, 1]))

        phi = - (phi_sam - phi_ref)
        phi_corrected = self._calc_phi(ref_td, sam_td, ref_fd, sam_fd)
        phi_corrected = np.abs(phi_corrected)
        # phi_corrected = phase_correction(freq_axis, phi, extrapolate=False)

        omega = 2 * np.pi * freq_axis

        # phi =  - (phi_sam_corrected[freq_idx, 1] - phi_ref_corrected[freq_idx, 1])

        with np.errstate(divide='ignore', invalid='ignore'):
            n = 1 + phi_corrected * c_thz / (omega * d)
            n[0] = n[1]
            n[n < 0] = 1
            kap = -c_thz * np.log(np.abs(sam_fd[:, 1] / ref_fd[:, 1]) * (1 + n) ** 2 / (4 * n)) / (omega * d)
            alpha = 1e4 * 2 * omega * kap / c_thz
            refr_idx = n + 1j * kap

        ret = {"freq_axis": freq_axis,"refr_idx": refr_idx, "alpha": alpha,
               "phi_ref": phi_ref, "phi_sam": phi_sam, "phi": phi, "phi_corrected": phi_corrected,
               }

        return ret

    def refractive_idx(self, meas_):
        return np.real(self.single_layer_eval(meas_)["refr_idx"])

    def _extinction_coe(self, meas_):
        return np.imag(self.single_layer_eval(meas_)["refr_idx"])

    def absorption_coef(self, meas_):
        n_cmplx_res = self.single_layer_eval(meas_)
        freq_axis = n_cmplx_res["freq_axis"]
        kap = n_cmplx_res["refr_idx"].imag

        omega = 2 * np.pi * freq_axis
        alph = (1 / 1e-4) * 2 * kap * omega / c_thz # 1/cm

        return alph

    def _eval_sub(self):
        sub_pnt = self.settings.eval_opt.sub_pnt
        sub_meas = self.sub_dataset.get_measurement(*sub_pnt)
        sub_res = self.sub_dataset.single_layer_eval(sub_meas)
        sub_res["t_sub"] = self.sub_dataset.transmission(sub_meas)

        return sub_res

    def conductivity(self, meas_):
        sub_res = self._eval_sub()
        t_sam = self.transmission(meas_, 1)

        n_sub = sub_res["refr_idx"]
        t_sub = sub_res["t_sub"]
        d_film = self.settings.sample_properties.d_film

        # [eps0_thz] = ps * Siemens / µm, [c_thz] = µm / ps, [1/d_film] = 1/um -> conversion: 1e4 (S/cm)
        # 1 / µm = 1 / (1e-6 m) = 1 / (1e-6 * 1e2 cm) = 1 / (1e-4 cm) = 1e4 * 1 / cm
        sigma = 1e4 * (1/d_film) * eps0_thz * c_thz * (1 + n_sub) * (t_sub/t_sam - 1)

        # phase correction, [dt] = fs
        dt = self.settings.eval_opt.dt
        dt *= 1e-3
        sigma *= np.exp(-1j*dt*2*np.pi*self.freq_axis)

        sigma.imag *= 1

        return sigma

    def get_measurement(self, x: float, y: float, return_single=True):
        meas_list = self.measurements["all"]
        pnt = (x, y)
        try:
            key = self.cache.coord_map_key_func(pnt)
            found_meas_list = self.cache.coord_map[key]
        except KeyError:
            found_meas_list, best_fit_val = None, np.inf
            for meas in meas_list:
                val = abs(meas.position[0] - pnt[0]) + abs(meas.position[1] - pnt[1])
                if val < best_fit_val:
                    best_fit_val = val
                    found_meas_list = [meas]

        if return_single:
            return found_meas_list[0]
        else:
            return found_meas_list

    def get_measurement_from_timestamp(self, timestamp_str):
        meas_id_ = self._timestamp2id(timestamp_str)

        found_meas = None
        for meas in self.measurements["all"]:
            if meas.identifier == meas_id_:
                found_meas = meas

        if found_meas is None:
            logging.warning(f"Measurement with timestamp: {timestamp_str} (id: {meas_id_}) not found in dataset")

        return found_meas

    def get_consecutive_meas(self, meas_):
        # measurements with same position as meas_ sampled without interruption (compared to avg meas time)
        coord_map_key = self.cache.coord_map_key_func(meas_.position)
        meas_at_pos = self.cache.coord_map[coord_map_key]
        if len(meas_at_pos) == 1:
            return meas_at_pos

        meas_idx0 = meas_at_pos.index(meas_)
        max_dist = 2*self.properties["mean_time_diff"]

        time_diff = np.diff([meas.meas_time for meas in meas_at_pos])
        time_diff_sec = [t_diff.total_seconds() for t_diff in time_diff]
        jump_idx_list = np.where(time_diff_sec > max_dist)[0]

        interval_idx = np.digitize(meas_idx0, jump_idx_list, right=True)
        if interval_idx == 0:
            meas_idx_range = np.arange(0, jump_idx_list[0]+1)
        elif interval_idx == len(jump_idx_list):
            meas_idx_range = np.arange(jump_idx_list[-1]+1, len(meas_at_pos))
        else:
            meas_idx_range = np.arange(jump_idx_list[interval_idx-1]+1, jump_idx_list[interval_idx]+1)

        found_meas = np.array(meas_at_pos)[meas_idx_range]

        return found_meas

    def get_line(self, x=None, y=None, limits=None):
        shape_properties = self.properties["shape"]
        if x is None and y is None:
            return None

        x_coords, y_coords = shape_properties["x_coords"], shape_properties["y_coords"]

        # vertical direction / slice
        if x is not None:
            ret = [self.get_measurement(x, y_) for y_ in y_coords], y_coords
        else:  # horizontal direction / slice
            ret = [self.get_measurement(x_, y) for x_ in x_coords], x_coords

        if limits is None:
            return ret
        else:
            measurements, coords = ret
            meas_in_limit_range = []
            for i, coord in enumerate(coords):
                if (limits[0] < coord) and (coord < limits[1]):
                    meas_in_limit_range.append(measurements[i])

            return meas_in_limit_range, coords

    def find_nearest_ref(self, meas_, dist_func=None) -> Measurement:
        if not dist_func:
            dist_func = self.settings.dist_func.value

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
        if self.settings.dist_func == Dist.Time:
            logging.debug(f"Time between ref and sample: {best_fit_val} seconds")
        else:
            logging.debug(f"Distance between ref and sample: {best_fit_val} mm")

        return closest_ref

    def get_ref_data(self, domain=Domain.Time, point=None, ref_idx=None, ret_meas=False):
        if self.settings.fix_ref is not False:
            chosen_ref = self.measurements["refs"][self.settings.fix_ref]
        elif point is not None:
            closest_sam = self.get_measurement(*point)
            chosen_ref = self.find_nearest_ref(closest_sam)
        else:
            if ref_idx is None:
                ref_idx = -1
            chosen_ref = self.measurements["refs"][ref_idx]

        # chosen_ref = np.random.choice(self.measurements["refs"])

        if domain in [Domain.Time, Domain.Frequency]:
            ret = self.get_data(chosen_ref, domain=domain)
        else:
            ret = self.get_data(chosen_ref, domain=Domain.Both)

        if ret_meas:
            return ret, chosen_ref
        else:
            return ret

    def get_ref_sam_meas(self, point):
        sam_meas = self.get_measurement(*point)
        ref_meas = self.find_nearest_ref(sam_meas)

        return ref_meas, sam_meas

    def _ref_interpolation(self, sam_meas):
        sam_meas_time = sam_meas.meas_time
        nearest_ref = self.find_nearest_ref(sam_meas, dist_func=Dist.Time)

        sam_idx = self.measurements["all"].index(sam_meas)
        ref_idx = self.measurements["all"].index(nearest_ref)

        ref_list_idx = self.measurements["refs"].index(nearest_ref)
        if sam_idx < ref_idx:
            # nearest_ref was measured after sample
            ref_before = self.measurements["refs"][ref_list_idx - 1]
            ref_after = self.measurements["refs"][ref_list_idx]
        else: # nearest_ref was measured before sample
            ref_before = self.measurements["refs"][ref_list_idx]
            ref_after = self.measurements["refs"][ref_list_idx + 1]

        ref_before_fd = self.get_data(ref_before, domain=Domain.Frequency)
        ref_after_fd =  self.get_data(ref_after, domain=Domain.Frequency)

        t0 = self.measurements["refs"][0].meas_time
        t = [(ref_before.meas_time - t0).total_seconds(), (ref_after.meas_time - t0).total_seconds()]

        y_fd_interpol = np.zeros(len(self.freq_axis), dtype=complex)
        for f_idx in range(len(self.freq_axis)):
            y_fd = [ref_before_fd[f_idx, 1], ref_after_fd[f_idx, 1]]

            y_fd_interpol[f_idx] = np.interp((sam_meas_time - t0).total_seconds(), t, y_fd)

        ref_interpol_fd = np.array([self.freq_axis, y_fd_interpol], dtype=complex).T

        return ref_interpol_fd


    def meas_time_diff(self, m1, m2):
        # meas time difference in hours
        return (m2.meas_time - m1.meas_time).total_seconds() / 3600

    def export_as_csv(self, dict_, file_app=""):
        save_dir = self.settings.result_dir
        save_path = save_dir / f"plotted_data_{file_app}.csv"
        logging.info(f"Exporting data to {save_path}")

        exp_dict = {}
        for k in dict_:
            if isinstance(dict_[k], np.ndarray):
                if dict_[k].ndim != 1:
                    arr = dict_[k][:, 1]
                else:
                    arr = dict_[k]
                if len(dict_[k]) == len(dict_["freq_axis"]):
                    exp_dict[k] = arr

        df = pd.DataFrame(exp_dict)
        df.to_csv(save_path, index=False)

    def print_ret(self, ret_, label=""):
        if label:
            logging.info(label)
        freq_axis = ret_["freq_axis"].real

        printed_freq_list = self.settings.eval_opt.printed_freqs
        f_idx_list = np.array([np.argmin(np.abs(f-freq_axis)) for f in printed_freq_list])
        if "k" in ret_:
            ret_["n"] = ret_["n"] + 1j * ret_["k"]

        printed_quantities = ["d", "n", "alpha"]
        uncert_map = {"d": self.settings.eval_opt.delta_d, "n": ret_["delta_n"], "alpha": ret_["delta_alpha"]}
        for quantity in ret_:
            if quantity not in printed_quantities:
                continue

            val = ret_[quantity]
            uncert = uncert_map[quantity]
            msg = "\n"
            if isinstance(val, np.ndarray):
                for f_idx in f_idx_list:
                    if val.ndim == 2:
                        freq = val[f_idx, 0]
                        val_ = val[f_idx, 1]
                        uncert_ = uncert[f_idx, 1]
                    else:
                        freq = freq_axis[f_idx]
                        val_ = val[f_idx]
                        uncert_ = uncert[f_idx]
                    val_ = np.round(val_, 2)
                    uncert_ = np.round(uncert_, 2)
                    freq = np.round(freq, 2)
                    msg += f"{quantity}: {val_}±{uncert_} at {freq} THz\n"
            else:
                val_ = np.round(val, 2)
                uncert_ = np.round(uncert, 2)
                msg = f"{quantity}: {val_}±{uncert_}\n"

            if quantity == list(ret_.keys())[-1]:
                msg += "\n"

            logging.info(msg)

    def link_sub_dataset(self, dataset_):
        dataset_.is_sub_dataset = True
        self.sub_dataset = dataset_


if __name__ == '__main__':
    options = {
        # "cbar_lim": (0.52, 0.60), # 2.5 THz
        # "cbar_lim": (0.64, 0.66), # img4
        "cbar_lim": (0.60, 0.66), # img5 1.5 THz
        # "cbar_lim": (0.55, 0.62), # img5 2.0 THz
        "plot_opt": {"plot_range": slice(30, 650), },
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
    dataset.plot_meas((35, 0), apply_window=False)
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
