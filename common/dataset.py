import traceback

from common.components import ComponentBase
from common.default_appsettings import Dist, ReferenceClassification, Domain, QuantityEnum
from common.settings import Settings
from common.measurement_selection import MeasurementSelection, get_coordinate_line
from pathlib import Path
import numpy as np
from common.functions import (window, butter_filt, do_fft, f_axis_idx_map,
                              remove_offset, avg_data_array, calculate_bandwidth)
from common.measurements import Measurement
from common.consts import c_thz, eps0_thz
import logging
from datetime import datetime
from common.dataset_cache import DatasetCache
import pandas as pd
from common.traits import Q_, Path as TPath, Quantity
from scipy.stats import pearsonr
from common.components import action
from traitlets import Unicode, observe, Float, Bool, Enum as TEnum, Instance, Int, Dict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from PySide6.QtCore import QObject, Signal
from functools import cached_property

"""
TODOs: 
- How are measurements mapped when multiple measurements are performed at the same x-y coordinates?
- Setting cbar lims sucks... set lims based on area min max?
- Fix runtime / use cache for t calc?
- check if filesizes? match when making npy array
- window function (fuctions.py): allow negative values (wrap around) + fix plot (clipping)
- self.logger is messy, also fix log levels and RuntimeWarnings
- freq_range variable in transmission function (and other functions?)
- combine the different reference select settings into one dict
- Fix unit labeling
- Split DataSet into multiple smaller classes (e.g. a plotting class) "Classes should do one thing each."
- should phi correction be a part of the pre-processing?
- rename some keys in options dict e.g. "eval_opt" to "eval"
- make Dataset(dict)? -> No
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

def format_meas_dict(meas_list, data_array, average_only=False):
    if len(meas_list) == 1:
        return {meas_list[0]: data_array[0]}

    y_meas_dict_ = {"Average": avg_data_array(data_array)}

    if not average_only:
        individual_y_meas = {
            meas: data_array[meas_idx]
            for meas_idx, meas in enumerate(meas_list)
        }
        y_meas_dict_.update(individual_y_meas)

    return y_meas_dict_

class WindowEvalResult:
    def __init__(self, **kwargs):
        self.meas_list = kwargs.get("meas", None)
        self.t = kwargs.get("t", None)
        self.freq_axis = kwargs.get("freq_axis", None)
        self.refr_idx = kwargs.get("refr_idx", None)
        self.abs_coe = kwargs.get("abs_coe", None)
        self.phi = kwargs.get("phi", None)
        self.phi_corrected = kwargs.get("phi_corrected", None)

class DataSetInfoPane(ComponentBase):

    max_amp_meas_info = Unicode("", group="Dataset info", read_only=True).tag(name="Max amplitude measurement")
    first_meas_info = Unicode("", group="Dataset info", read_only=True).tag(name="First measurement")
    last_meas_info = Unicode("", group="Dataset info", read_only=True).tag(name="Last measurement")

    all_meas_cnt_info = Unicode("", group="Measurement classification", read_only=True).tag(
        name="Number of measurements")
    ref_meas_cnt_info = Unicode("", group="Measurement classification", read_only=True).tag(
        name="Number of reference measurements")
    sam_meas_cnt_info = Unicode("", group="Measurement classification", read_only=True).tag(
        name="Number of sample measurements")

    meas_time_info = Unicode("", group="Measurement time info", read_only=True).tag(name="Total measurement time")
    mean_meas_time_info = Unicode("", group="Measurement time info", read_only=True).tag(name="Mean measurement time")

    shape_w_info = Unicode("", group="Shape info", read_only=True).tag(name="Width (mm)")
    shape_h_info = Unicode("", group="Shape info", read_only=True).tag(name="Height (mm)")
    shape_dx_info = Unicode("", group="Shape info", read_only=True).tag(name="x-step (mm)")
    shape_dy_info = Unicode("", group="Shape info", read_only=True).tag(name="y-step (mm)")
    pixel_cnt = Unicode("", group="Shape info", read_only=True).tag(name="Pixel count", priority=99)

    x_coord_extrema = Unicode("", group="Coordinate info",
                              read_only=True).tag(name="x min, max (mm)")
    y_coord_extrema = Unicode("", group="Coordinate info",
                              read_only=True).tag(name="y min, max (mm)")

    sampling_start = Quantity(Q_(0, "ps"), read_only=True).tag(name="Sampling start", group="Data info")
    sampling_end = Quantity(Q_(0, "ps"), read_only=True).tag(name="Sampling end", group="Data info")
    sampling_period = Quantity(Q_(0, "ps"), read_only=True).tag(name="Sampling period", group="Data info")
    sample_count = Int(0, read_only=True).tag(name="Time samples", group="Data info")
    sampling_window = Quantity(Q_(0, "ps"), read_only=True).tag(name="Sampling window", group="Data info")

    frequency_resolution = Quantity(Q_(0, "THz"), read_only=True).tag(name="Frequency resolution", group="Data info")
    spectral_line_cnt = Int(0, read_only=True).tag(name="Frequency samples", group="Data info")
    nyquist_frequency = Quantity(Q_(0, "THz"), read_only=True).tag(name="Nyquist frequency", group="Data info")

    peak2peak = Float(0.0,read_only=True).tag(name="Peak to peak", group="Maximum amplitude measurement")
    snr = Quantity(Q_(0.0, "dB"), read_only=True).tag(name="Signal to noise ratio",
                                                      group="Maximum amplitude measurement")
    signal_bandwidth = Quantity(Q_(0.0, "THz"), read_only=True).tag(name="Signal bandwidth",
                                                                    group="Maximum amplitude measurement")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ParsingSignalCarrier(QObject):
    progress_signal = Signal(float)
    initialization_complete_signal = Signal(str, bool)
    measurements_parsed_signal = Signal(dict)

class DataSet(ComponentBase):
    data_path = TPath(Path(r""), is_file=False).tag(priority=-100, name="Dataset path",
                                                    fullwidth=True).tag(priority=-2)
    caching_progress = Float(0, min=0, max=1, read_only=True).tag(name="Caching progress", priority=-1)
    is_initialized = Bool(False, read_only=True).tag(name="Cache Initialized")

    sub_linked = Bool(False, read_only=True).tag(name="Substrate dataset linked")

    reference_filter_group = "Reference classification"
    ref_classifier = TEnum(ReferenceClassification,
                           default_value=ReferenceClassification.from_file_name,
                           group=reference_filter_group).tag(name="Reference classification criteria", priority=-1)
    ref_threshold = Float(0.999, group=reference_filter_group, min=0, max=1,
                          help="Threshold relative to maximum amplitude measurement").tag(name="Reference threshold")
    horizontal_ref_coord = Quantity(Q_(0.0, "mm")).tag(group=reference_filter_group, name="Horizontal line coordinate")
    vertical_ref_coord = Quantity(Q_(0.0, "mm")).tag(group=reference_filter_group, name="Vertical line coordinate")
    ref_id_str = Unicode("ref").tag(group=reference_filter_group, name="Filter string")

    data_export_grp = "Data export"
    export_csv_dir = TPath(Path(""), is_file=False).tag(name="Save directory", group=data_export_grp)
    data_export_label = Unicode("", group=data_export_grp).tag(name="Data export label")

    info_pane = Instance(DataSetInfoPane)
    measurement_selector = Instance(MeasurementSelection)

    measurements = Dict(default_value={"refs": [], "sams": [], "all": ()})

    def __init__(self, settings : Settings, **kwargs):
        super().__init__(**kwargs)
        self._parse_lock = Lock()
        self._is_parsing = False
        self.set_trait("is_initialized", False)

        self.info_pane = DataSetInfoPane(object_name="Info pane")

        self.measurements = {"refs": [], "sams": [], "all": ()}

        self.cache = None
        self.sub_dataset = None

        self.settings = settings

        self.measurement_selector = MeasurementSelection(self)

        self.freq_axis = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.settings.save_configuration(self)

    def __enter__(self):
        self.settings.load_configuration(self)
        self._set_observers()
        self.measurement_selector.set_observers()
        self._parse_measurements()

        return self

    @property
    def func_map(self):
        func_map = {QuantityEnum.P2P: self.p2p, # 1D (N_meas)
                    QuantityEnum.Phase: self.phase, # 2D (N_meas, Freq_slice)
                    QuantityEnum.MeasTimeDeltaRef2Sam: self.meas_time_delta, # 0D scalar
                    QuantityEnum.Power: self.power, # 2D (N_meas, Freq_slice)
                    QuantityEnum.RefAmp: self.ref_max, # 2D (N_meas, Freq_slice)
                    QuantityEnum.RefArgmax: self.get_ref_abs_argmax, # 1D (N_meas)
                    QuantityEnum.RefPhase: self.ref_phase, # 2D (N_meas, Freq_slice)
                    QuantityEnum.ZeroCrossing: self.get_zero_crossing, # 1D (N_meas)
                    QuantityEnum.TimeOfFlight: self.time_of_flight, # 1D (N_meas)
                    QuantityEnum.Transmission: self.transmission, # 2D (N_meas, Freq_slice)
                    QuantityEnum.TransmissionAmp: self.amplitude_transmission, # 2D (N_meas, Freq_slice)
                    QuantityEnum.TransmissionPhase: self.phase_difference, # 2D (N_meas, Freq_slice)
                    QuantityEnum.RefractiveIdx: self.refractive_idx, # 2D (N_meas, Freq_slice)
                    QuantityEnum.AbsorptionCoe: self.absorption_coef, # 2D (N_meas, Freq_slice)
                    QuantityEnum.Conductivity: self.conductivity, # 2D (N_meas, Freq_slice)
                    }
        return func_map

    @cached_property
    def shape_properties(self):
        return self._update_shape_properties()

    def refresh_shape_properties(self):
        self.__dict__.pop("shape_properties", None)
        self._update_shape_properties()

    @property
    def time_diffs(self):
        time_diffs = []
        for i in range(0, len(self.measurements["all"]) - 1):
            m0, m1 = self.measurements["all"][i], self.measurements["all"][i + 1]
            time_diffs.append(self.meas_time_diff(m0, m1).magnitude)

        return time_diffs

    @property
    def mean_time_diff(self):
        return np.mean(self.time_diffs)

    def selected_measurements(self):
        return self.measurement_selector.selected_measurements

    @observe("data_path")
    def _set_path(self, change=None):
        if change is None:
            return
        self.data_path = change["new"]
        self._parse_measurements(self.data_path)

        self.set_trait("caching_progress", 1)

    def _set_observers(self):
        reference_filter_trait_names = self.trait_names(group=self.reference_filter_group)
        self.observe(self.update_meas_sorting, names=reference_filter_trait_names)

    def update_cache_progress(self, progress):
        self.set_trait("caching_progress", progress)

    @action("Clear cache")
    def clear_cache(self):
        if self.cache is None:
            self.logger.info("Cache is None. Reset dataset path")
        else:
            self.cache.clear_cache()
            self.set_trait("is_initialized", False)

    @action("Reparse measurements")
    def reparse_measurements(self):
        self.clear_cache()
        self._parse_measurements()

    def _pre_process(self, meas_):
        pp_opt = self.settings.pp_opt

        cache_idx = self.cache.id_map[meas_.identifier]
        data_td = self.cache.raw_data_td[cache_idx]

        if pp_opt.remove_dc:
            data_td = remove_offset(data_td)

        if pp_opt.window_enabled:
            win_kwargs = {k: getattr(pp_opt, k) for k in pp_opt.traits()}
            data_td = window(data_td, **win_kwargs)

        if pp_opt.filter_enabled:
            data_td = butter_filt(data_td, pp_opt.f_range.magnitude)

        return data_td

    def _single_meas_data(self, meas, domain=Domain.Time):
        data_td = self._pre_process(meas)

        if domain == Domain.Time:
            data = data_td
        else:
            data = do_fft(data_td)
            self.freq_axis = data[:, 0].real

        data = np.insert(data, 2, 0, axis=1)

        return data

    def get_multi_data(self, meas_list, domain=Domain.Time):
        meas_list = [meas_list] if isinstance(meas_list, Measurement) else meas_list

        sample_data = self._single_meas_data(meas_list[0], domain=domain)
        shape = sample_data.shape

        data_arr = np.zeros([len(meas_list), *shape], dtype=sample_data.dtype)
        data_arr[0] = sample_data
        if len(meas_list) == 1:
            return data_arr
        else:
            for meas_idx, meas in enumerate(meas_list[1:]):
                data_arr[meas_idx] = self._single_meas_data(meas, domain=domain)

        return data_arr

    def max_amp_meas_filter(self, meas_list, threshold=1.0):
        meas_list = np.array(meas_list, dtype=object)
        data_td = self.get_multi_data(meas_list, domain=Domain.Time)
        max_per_measurement = np.max(np.abs(data_td[:, :, 1]), axis=1)
        threshold_mask = max_per_measurement >= (threshold * np.max(max_per_measurement))

        ret_list = list(meas_list[threshold_mask])

        if threshold < 1:
            return ret_list
        else:
            return ret_list[0]

    def _update_measurement_dict(self, new_measurement_dict):
        new_measurement_dict["max_amp_meas"] = self.max_amp_meas_filter(new_measurement_dict["all"])

        self._sort_meas_type(new_measurement_dict)

        self._set_dataset_info()
        self.refresh_shape_properties()

    def _parse_measurements(self, data_path=None):
        if data_path is None:
            data_path = self.data_path
        if data_path is None:
            return

        signal_carrier = ParsingSignalCarrier()
        signal_carrier.progress_signal.connect(self.update_cache_progress)
        signal_carrier.initialization_complete_signal.connect(self.set_trait)
        signal_carrier.measurements_parsed_signal.connect(self._update_measurement_dict)

        def bg_worker(target_path):
            if not self._parse_lock.acquire(blocking=False):
                self.logger.debug("Parse already in progress")
                return
            try:
                if not target_path or not target_path.exists():
                    raise ValueError(f"Path {target_path} does not exist")
                if not target_path.is_dir():
                    raise ValueError(f"Path {target_path} is not a directory")
                if not list(target_path.glob("*")):
                    raise ValueError(f"Path {target_path} is empty")

                new_measurements = {"all": self._read_data_dir()}
                if not new_measurements["all"]:
                    return

                self.cache = DatasetCache(self, new_measurements, target_path, signal_carrier)

                signal_carrier.measurements_parsed_signal.emit(new_measurements)
                signal_carrier.initialization_complete_signal.emit("is_initialized", True)
            except Exception as e:
                tb_str = traceback.format_exc()
                print(tb_str)
                self.logger.error(f"Error parsing path {target_path}: {e}", exc_info=True)
            finally:
                self._parse_lock.release()

        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(bg_worker, data_path)

    def update_meas_sorting(self, ref_filter_change):
        self._update_measurement_dict(self.measurements)

    def _update_shape_properties(self):
        x_coords, y_coords = [], []
        for sam_measurement in self.measurements["sams"]:
            x_coords.append(sam_measurement.position[0])
            y_coords.append(sam_measurement.position[1])

        if not x_coords: x_coords.append(0)
        if not y_coords: y_coords.append(0)

        x_coords = np.array(sorted(set(x_coords)))
        y_coords = np.array(sorted(set(y_coords)))

        all_points = [meas.position for meas in self.measurements["all"]]

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

        shape = {"w": w, "h": h, "dx": dx, "dy": dy, "extent": extent,
                 "x_coords": x_coords, "y_coords": y_coords, "all_points": all_points}
        self.set_shape_info_traits(shape)

        return shape

    def set_shape_info_traits(self, shape):
        self.info_pane.set_trait("shape_w_info", str(shape["w"]))
        self.info_pane.set_trait("shape_h_info", str(shape["h"]))
        self.info_pane.set_trait("shape_dx_info", str(shape["dx"]))
        self.info_pane.set_trait("shape_dy_info", str(shape["dy"]))
        self.info_pane.set_trait("pixel_cnt", str(int((shape["h"]/shape["dy"])*(shape["w"]/shape["dx"]))))

        x_extrema_str = f"{np.min(shape['x_coords'])}, {np.max(shape['x_coords'])}"
        self.info_pane.set_trait("x_coord_extrema", x_extrema_str)

        y_extrema_str = f"{np.min(shape['y_coords'])}, {np.max(shape['y_coords'])}"
        self.info_pane.set_trait("y_coord_extrema", y_extrema_str)

    def find_climate_log_file(self, climate_log_file):
        log_file = Path(climate_log_file)

        target_log_file = log_file.name

        checked_dirs = [self.data_path, self.data_path.parent]
        log_files = []
        log_files.extend([file for file in checked_dirs[0].iterdir() if "log" in file.name])
        log_files.extend([file for file in checked_dirs[1].iterdir() if "log" in file.name])

        for log_file in log_files:
            if str(target_log_file) in log_file.name:
                return log_file

        return None

    def _read_data_dir(self):
        glob = self.data_path.glob("**/*.txt")

        measurements = []
        for i, file_path in enumerate(glob):
            if file_path.is_file() and ".txt" in file_path.name:
                try:
                    measurements.append(Measurement(filepath=file_path))
                except Exception as err:
                    self.logger.info(f"Skipping {file_path}. {err}")

        if not measurements:
            self.logger.warning("No measurements found. Check path or filenames")

        measurements = tuple(sorted(measurements, key=lambda meas: meas.meas_time))

        return measurements

    def _set_dataset_info(self):
        max_amp_meas = self.measurements["max_amp_meas"]
        self.logger.debug(f"Maximum amplitude measurement: {max_amp_meas.filepath.name}\n")
        self.info_pane.set_trait("max_amp_meas_info", max_amp_meas.filepath.stem)

        all_cnt = len(self.measurements["all"])
        ref_cnt = len(self.measurements["refs"])
        sam_cnt = len(self.measurements["sams"])
        self.logger.debug(f"Dataset contains {all_cnt} measurements")
        self.logger.debug(f"{ref_cnt} reference measurements ")
        self.logger.debug(f"{sam_cnt} sample measurements")

        first_measurement = self.measurements["all"][0]
        last_measurement = self.measurements["all"][-1]

        self.logger.debug(f"First measurement at: {first_measurement.meas_time}, "
                     f"last measurement: {last_measurement.meas_time}")
        time_del = last_measurement.meas_time - first_measurement.meas_time
        td_secs = time_del.seconds + 24 * 3600 * time_del.days
        tot_hours = td_secs // 3600
        min_part = (td_secs // 60) % 60
        sec_part = time_del.seconds % 60

        self.logger.debug(f"Total measurement time: {tot_hours} hours, "
                     f"{min_part} minute(s) and {sec_part} second(s) ({td_secs} seconds)\n")

        mean_time_diff = self.mean_time_diff
        diffs = self.time_diffs
        self.logger.debug(f"Mean time between measurements: {np.round(mean_time_diff, 2)} seconds")
        self.logger.debug(f"Min and max time between measurements: "
                     f"({np.min(diffs)}, {np.max(diffs)}) seconds\n")

        self.info_pane.set_trait("all_meas_cnt_info", str(all_cnt))
        self.info_pane.set_trait("ref_meas_cnt_info", str(ref_cnt))
        self.info_pane.set_trait("sam_meas_cnt_info", str(sam_cnt))

        self.info_pane.set_trait("first_meas_info", first_measurement.filepath.stem)
        self.info_pane.set_trait("last_meas_info", last_measurement.filepath.stem)

        self.info_pane.set_trait("meas_time_info", f"{tot_hours:02}:{min_part:02}:{sec_part:02}")
        self.info_pane.set_trait("mean_meas_time_info", f"{np.round(mean_time_diff, 2)} seconds")

        data_td = self.get_multi_data(max_amp_meas)
        data_fd = self.get_multi_data(max_amp_meas, Domain.Frequency)
        p2p = self.p2p(max_amp_meas)
        signal_properties = calculate_bandwidth(data_fd[0])

        t_start, t_end = Q_(data_td[0, 0, 0], "ps"), Q_(data_td[0, -1, 0], "ps")
        self.info_pane.set_trait("sampling_start", t_start)
        self.info_pane.set_trait("sampling_end", t_end)
        self.info_pane.set_trait("sample_count", data_td.shape[1])
        self.info_pane.set_trait("sampling_window", t_end-t_start)
        self.info_pane.set_trait("sampling_period", (t_end-t_start)/data_td.shape[1])

        self.info_pane.set_trait("frequency_resolution", Q_(np.mean(np.diff(data_fd[0, :, 0].real)), "THz"))
        self.info_pane.set_trait("spectral_line_cnt", data_fd.shape[1])
        self.info_pane.set_trait("nyquist_frequency", Q_(data_fd[0, -1, 0].real, "THz"))

        self.info_pane.set_trait("peak2peak", p2p[0])

        self.info_pane.set_trait("snr", Q_(signal_properties["snr"], "dB"))
        self.info_pane.set_trait("signal_bandwidth", Q_(signal_properties["bandwidth"], "THz"))

    def _sort_meas_type(self, new_measurements):
        all_measurements = new_measurements["all"]
        id_str = self.ref_id_str

        refs_ = []
        ref_line = []
        match self.ref_classifier:
            case ReferenceClassification.from_file_name:
                for meas in all_measurements:
                    if id_str in str(meas.filepath.stem).lower():
                        refs_.append(meas)
                if not refs_:
                    self.logger.info(f"No explicit references found in the dataset based on filename ({id_str}).")
            case ReferenceClassification.above_threshold:
                refs_ = self.max_amp_meas_filter(all_measurements, self.ref_threshold)
                if isinstance(refs_, Measurement):
                    refs_ = [refs_]
                self.logger.info(f"Using measurements near max amplitude as ref. (Threshold: {self.ref_threshold})")
            case ReferenceClassification.horizontal_line_as_ref:
                y = self.horizontal_ref_coord
                self.logger.info(f"Selecting measurements along horizontal line closest to y={y}")
                ref_line = get_coordinate_line(all_measurements, y=y)
            case ReferenceClassification.vertical_line_as_ref:
                x = self.vertical_ref_coord
                self.logger.info(f"Selecting measurements along vertical line closest to x={x}")
                ref_line = get_coordinate_line(all_measurements, x=x)

        if ref_line:
            refs_ = self.max_amp_meas_filter(ref_line, self.ref_threshold)
            if len(refs_) > 1:
                self.logger.info(f"Using reference measurements: {refs_[0].filepath.stem} to {refs_[-1].filepath.stem}")

        if not refs_:
            self.logger.warning(f"{self.object_name}: No refs found matching reference classification criteria.")

        sams_ = []
        for meas in all_measurements:
            if meas not in refs_:
                sams_.append(meas)

        measurements = {
            "all": all_measurements,
            "refs": refs_,
            "sams": sams_,
            "max_amp_meas": new_measurements.get("max_amp_meas")
        }

        self.logger.info(f"Classified {len(measurements['refs'])} measurement(s) as reference\n")

        self.set_trait("measurements", measurements)

    def link_sub_dataset(self, dataset_):
        if dataset_ is None:
            return

        self.sub_dataset = dataset_
        self.set_trait("sub_linked", True)

    def windowing_eval(self, meas_):
        ref_meas = self.measurement_selector.get_matching_ref(meas_)

        with self.settings.pp_opt.override(window_enabled=True, win_width=10, win_start=None):
            ref_fd = self.get_multi_data(ref_meas, Domain.Frequency)
            sam_fd = self.get_multi_data(meas_, Domain.Frequency)

            phi_corrected = self.phase_difference(meas_)

        freq_axis = self.freq_axis
        omega = 2 * np.pi * freq_axis

        d = self.settings.eval_opt.d.magnitude
        with np.errstate(divide='ignore', invalid='ignore'):
            n = 1 + phi_corrected * c_thz / (omega * d)
            n = np.nan_to_num(n, nan=1)

            n[:, 0] = n[:, 1]
            n[n < 0] = 1
            ext_coe = -c_thz * np.log(np.abs(sam_fd[:, :, 1] / ref_fd[:, :, 1]) * (1 + n) ** 2 / (4 * n)) / (omega * d)
            ext_coe = np.nan_to_num(ext_coe, nan=0)
            abs_coe = 1e4 * 2 * omega * ext_coe / c_thz
            refr_idx = n + 1j * ext_coe

        ret = {"freq_axis": freq_axis,
               "refr_idx": refr_idx,
               "abs_coe": abs_coe,
               "phi_corrected": phi_corrected,
               }

        return WindowEvalResult(**ret)

    def refractive_idx(self, meas_):
        return np.real(self.windowing_eval(meas_).refr_idx)

    def absorption_coef(self, meas_):
        return self.windowing_eval(meas_).abs_coe

    def extinction_coe(self, meas_):
        return np.imag(self.windowing_eval(meas_).refr_idx)

    def get_single_layer_properties(self):
        sub_pnt = self.settings.eval_opt.sub_pnt
        if self.settings.eval_opt.use_sub_dataset:
            meas_list = self.sub_dataset.measurement_selector.get_measurements_from_point(*sub_pnt)
            window_eval_res = self.sub_dataset.windowing_eval(meas_list)
            t = self.sub_dataset.transmission(meas_list)
        else:
            meas_list = self.measurement_selector.get_measurements_from_point(*sub_pnt)
            window_eval_res = self.windowing_eval(meas_list)
            t = self.transmission(meas_list)

        window_eval_res.meas_list = meas_list
        window_eval_res.t = t

        return window_eval_res

    def get_ref_abs_argmax(self, meas_):
        ref_meas = self.measurement_selector.get_matching_ref(meas_)
        ref_td = self.get_multi_data(ref_meas)
        t, y = ref_td[:, :, 0], ref_td[:, :, 1]

        return t[np.arange(ref_td.shape[0]), np.argmax(np.abs(y), axis=1)]

    def spectral_similarity(self, meas_0, meas_1, freq_min=0.15, freq_max=2.00):
        data_fd_0 = self.get_multi_data(meas_0, domain=Domain.Frequency)
        data_fd_1 = self.get_multi_data(meas_1, domain=Domain.Frequency)

        f_idx_range = f_axis_idx_map(self.freq_axis, (freq_min, freq_max))
        x, y = np.abs(data_fd_0[:, f_idx_range, 1]), np.abs(data_fd_1[:, f_idx_range, 1])

        return 1+np.log(np.abs(pearsonr(x, y, axis=1).statistic))

    def delay_from_phase_slope(self, meas_0, meas_1, freq_min=0.15, freq_max=0.85):
        phi_diff = self.phase_difference(meas_0, meas_1)
        mask = (freq_min <= self.freq_axis) & (self.freq_axis <= freq_max)

        p = np.polyfit(self.freq_axis[mask], phi_diff[:, mask].T, 1)

        tau = -p[0] / (2*np.pi)

        return tau

    def get_zero_crossing(self, meas_):
        data_td = self.get_multi_data(meas_)
        t, y = data_td[:, :, 0], data_td[:, :, 1]

        y_abs_max = np.argmax(np.abs(y), axis=1)

        sign_changes = (y[:, :-1] * y[:, 1:]) < 0

        time_indices = np.arange(y.shape[1] - 1)
        valid_mask = sign_changes & (time_indices >= y_abs_max[:, None])

        first_cross_idx = np.argmax(valid_mask, axis=1)

        meas_idx = np.arange(y.shape[0])

        y1 = y[meas_idx, first_cross_idx]
        y2 = y[meas_idx, first_cross_idx + 1]
        x1 = t[meas_idx, first_cross_idx]
        x2 = t[meas_idx, first_cross_idx + 1]

        dy = y1 - y2

        has_zero_crossing = valid_mask[meas_idx, first_cross_idx] & ~np.isclose(dy, 0)

        zero_crossing_interp = np.zeros(y.shape[0])
        zero_crossing_interp[has_zero_crossing] = ((y1 * x2 - x1 * y2) / dy)[has_zero_crossing]

        return zero_crossing_interp

    def p2p(self, meas_: Measurement):
        y_td = self.get_multi_data(meas_)
        return np.max(y_td[:, :, 1], axis=1) - np.min(y_td[:, :, 1], axis=1)

    def phase(self, meas_: Measurement):
        y_fd = self.get_multi_data(meas_, domain=Domain.Frequency)
        return np.angle(y_fd[:, :, 1])

    def power(self, meas_: Measurement):
        ref_meas = self.measurement_selector.get_matching_ref(meas_)
        ref_fd = self.get_multi_data(ref_meas, domain=Domain.Frequency)
        sam_fd = self.get_multi_data(meas_, domain=Domain.Frequency)

        power_val_ref = np.abs(ref_fd[:, :, 1])
        power_val_sam = np.abs(sam_fd[:, :, 1])

        return (power_val_sam / power_val_ref) ** 2

    def power_int(self, meas_: Measurement, freq_range):
        freq_slice = (freq_range[0] < self.freq_axis) * (self.freq_axis < freq_range[1])

        ref_meas = self.measurement_selector.get_matching_ref(meas_)
        ref_fd = self.get_multi_data(ref_meas, domain=Domain.Frequency)
        sam_fd = self.get_multi_data(meas_, domain=Domain.Frequency)

        power_val_ref = np.sum(np.abs(ref_fd[:, freq_slice, 1]) ** 2, axis=1)
        power_val_sam = np.sum(np.abs(sam_fd[:, freq_slice, 1]) ** 2, axis=1)

        return power_val_sam / power_val_ref

    def meas_time_delta(self, meas_: Measurement):
        ref_meas = self.measurement_selector.get_nearest_ref(meas_)

        return (meas_.meas_time - ref_meas.meas_time).total_seconds()

    def ref_max(self, meas_: Measurement):
        y_fd = self._ref_interpolation(meas_)

        return np.abs(y_fd[:, :, 1])

    def ref_phase(self, meas_: Measurement):
        y_fd = self._ref_interpolation(meas_)

        return np.angle(y_fd[:, :, 1])

    def simple_peak_cnt(self, meas_: Measurement, threshold: float):
        data_td = self.get_multi_data(meas_)
        y_ = data_td[:, :, 1]

        baseline = 0.5 * (
                np.mean(y_[:, :10], axis=1, keepdims=True) +
                np.mean(y_[:, -10:], axis=1, keepdims=True)
        )
        y_ -= baseline

        threshold_slice = y_[:, 1:-1] > threshold
        peak_idx = (y_[:, :-2] < y_[:, 1:-1]) & (y_[:, 1:-1] > y_[:, 2:])
        peak_cnt = np.sum(peak_idx & threshold_slice, axis=1)

        return peak_cnt

    def simple_phase_difference(self, m0, m1=None):
        if m1 is None:
            m1 = self.measurement_selector.get_matching_ref(m0)

        m1_fd = self.get_multi_data(m1, Domain.Frequency)
        m0_fd = self.get_multi_data(m0, Domain.Frequency)

        phi_r_m1 = np.angle(m1_fd[:, :, 1])
        phi_r_m0 = np.angle(m0_fd[:, :, 1])

        return np.unwrap(phi_r_m0 - phi_r_m1, axis=1)


    def phase_difference(self, m0, m1=None):
        # phase unwrapping Phase Retrieval in Terahertz Time-Domain
        # Measurements: a how to Tutorial P. Uhd Jepsen https://doi.org/10.1007/s10762-019-00578-0
        if m1 is None:
            m1 = self.measurement_selector.get_matching_ref(m0)

        m1_fd = self.get_multi_data(m1, Domain.Frequency)
        m0_fd = self.get_multi_data(m0, Domain.Frequency)
        m1_td = self.get_multi_data(m1, Domain.Time)
        m0_td = self.get_multi_data(m0, Domain.Time)

        phi_fit_range = self.settings.eval_opt.phi_fit_range.magnitude

        f_axis = self.freq_axis
        w_axis = 2 * np.pi * f_axis

        t0_m1_idx = np.argmax(np.abs(m1_td[:, :, 1]), axis=1)
        t0_m0_idx = np.argmax(np.abs(m0_td[:, :, 1]), axis=1)

        t0_m1 = m1_td[np.arange(m1_td.shape[0]), t0_m1_idx, 0]
        t0_m0 = m0_td[np.arange(m0_td.shape[0]), t0_m0_idx, 0]
        t_offset = m1_td[:, 0, 0] - m0_td[:, 0, 0]

        phi0_m1, phi0_m0 = w_axis * t0_m1[:, None], w_axis * t0_m0[:, None]

        phi_r_m1 = np.angle(m1_fd[:, :, 1] * np.exp(-1j * phi0_m1))
        phi_r_m0 = np.angle(m0_fd[:, :, 1] * np.exp(-1j * phi0_m0))

        phi0_star = np.unwrap(phi_r_m0 - phi_r_m1, axis=1)

        fit_slice = (f_axis >= phi_fit_range[0]) * (f_axis <= phi_fit_range[1])
        p = np.polyfit(f_axis[fit_slice], phi0_star[:, fit_slice].T, 1)

        phi0 = phi0_star - 2 * np.pi * np.round(p[1][:, None] / (2 * np.pi))

        if not self.settings.eval_opt.phi_offset_correction:
            return phi0_star

        phi = phi0 - phi0_m1 + phi0_m0 + w_axis * t_offset[:, None]

        return phi

    def amplitude_transmission(self, meas_):
        ref_meas_ = self.measurement_selector.get_matching_ref(meas_)
        ref_fd = self.get_multi_data(ref_meas_, Domain.Frequency)
        sam_fd = self.get_multi_data(meas_, Domain.Frequency)

        t = sam_fd[:, :, 1] / ref_fd[:, :, 1]

        return np.abs(t)

    def transmission(self, meas_):
        t_abs = self.amplitude_transmission(meas_)
        phase_difference = self.phase_difference(meas_)

        t = t_abs * np.exp(1j * phase_difference)

        return t

    def time_of_flight(self, meas_):
        closest_ref = self.measurement_selector.get_nearest_ref(meas_)

        t_zero_ref = self.get_zero_crossing(closest_ref)
        t_zero_sam = self.get_zero_crossing(meas_)

        return t_zero_ref - t_zero_sam

    def conductivity(self, meas_):
        sub_properties = self.get_single_layer_properties()

        t_sam = self.transmission(meas_)

        n_sub = sub_properties.refr_idx
        t_sub = sub_properties.t

        d_film = self.settings.eval_opt.d_film.magnitude
        if np.isclose(d_film, 0):
            d_film = 1e-3

        # [eps0_thz] = ps * Siemens / µm, [c_thz] = µm / ps, [1/d_film] = 1/um -> conversion: 1e4 (S/cm)
        # 1 / µm = 1 / (1e-6 m) = 1 / (1e-6 * 1e2 cm) = 1 / (1e-4 cm) = 1e4 * 1 / cm
        sigma = 1e4 * (1/d_film) * eps0_thz * c_thz * (1 + n_sub) * (t_sub/t_sam - 1)

        # phase correction, [dt] = fs
        dt = self.settings.eval_opt.dt.magnitude * 1e-3
        sigma *= np.exp(-1j*dt*2*np.pi*self.freq_axis[None, :])

        return sigma

    def sigma_to_n(self, freq, sigma):
        w = 2 * np.pi * freq

        sigma *= 1e-4 # S/cm -> S/µm
        n_ = (1 + 1j) * np.sqrt(sigma/(2*w*eps0_thz))

        return n_

    def _ref_interpolation(self, sam_meas):
        if len(self.measurements["refs"]) < 2:
            single_ref_fd = self.get_multi_data([self.measurements["refs"][0]], domain=Domain.Frequency)
            return np.repeat(single_ref_fd, len(sam_meas), axis=0)

        refs_before = []
        refs_after = []
        sam_times = []

        for meas in sam_meas:
            nearest_ref = self.measurement_selector.get_nearest_ref(meas, dist_func=Dist.Time.value)
            sam_idx = self.measurements["all"].index(meas)
            ref_idx = self.measurements["all"].index(nearest_ref)
            ref_list_idx = self.measurements["refs"].index(nearest_ref)

            if sam_idx < ref_idx:
                # nearest_ref was measured after sample
                refs_before.append(self.measurements["refs"][ref_list_idx - 1])
                refs_after.append(self.measurements["refs"][ref_list_idx])
            else:
                # nearest_ref was measured before sample
                refs_before.append(self.measurements["refs"][ref_list_idx])
                refs_after.append(self.measurements["refs"][ref_list_idx + 1])

            sam_times.append(meas.meas_time)

        fd_before = self.get_multi_data(refs_before, domain=Domain.Frequency)[:, :, 1]
        fd_after = self.get_multi_data(refs_after, domain=Domain.Frequency)[:, :, 1]

        t_before = np.array([(rb.meas_time - st).total_seconds() for rb, st in zip(refs_before, sam_times)])
        t_span = np.array([(ra.meas_time - rb.meas_time).total_seconds() for ra, rb in zip(refs_after, refs_before)])

        y_fd_interpol = fd_before -((t_before / t_span)[:, None]) * (fd_after - fd_before)

        freq_grid = np.tile(self.freq_axis, (len(sam_meas), 1))
        std_grid = np.tile(np.zeros_like(self.freq_axis), (len(sam_meas), 1))

        return np.stack((freq_grid, y_fd_interpol, std_grid), axis=-1)

    def meas_time_diff(self, m1, m2):
        val = (m2.meas_time - m1.meas_time).total_seconds()
        return Q_(val, "s")

    @action("Export measurement data", group=data_export_grp)
    def export_measurement_data(self):
        selected_meas = self.selected_measurements
        ref_meas = self.measurement_selector.get_matching_ref(selected_meas)
        data_arrays = {
            "ref_td": self.get_multi_data(ref_meas, domain=Domain.Time),
            "ref_fd": self.get_multi_data(ref_meas, domain=Domain.Frequency),
            "sam_td": self.get_multi_data(selected_meas, domain=Domain.Time),
            "sam_fd": self.get_multi_data(selected_meas, domain=Domain.Frequency),
            "t": self.transmission(selected_meas),
        }

        save_dir = self.export_csv_dir
        file_app = self.data_export_label
        if not file_app:
            file_app = datetime.now().isoformat().replace(':', '-')
        save_path = save_dir / f"exported_data_{file_app}.csv"
        self.logger.info(f"Exporting data to {save_path}")

        exp_dict = {"freq_axis": data_arrays["ref_fd"][0, :, 0].real,
                    "time_axis": data_arrays["ref_td"][0, :, 0].real}
        for k, data_arr in data_arrays.items():
            for meas_idx in data_arr.shape[0]:
                exp_dict[f"{k}_{meas_idx}"] = data_arr[meas_idx, :, 1]

        df = pd.DataFrame(exp_dict)
        df = df.astype(str)
        df = df.apply(lambda col: col.str.replace(r"^\((.*)\)$", r"\1", regex=True))
        df.to_csv(save_path, index=False)

    def print_ret(self, ret_, label=""):
        if label:
            self.logger.info(label)
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

            self.logger.info(msg)

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

    logging.basicConfig(level=self.logger.INFO)
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
