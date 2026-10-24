from __future__ import annotations
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from functools import cached_property, partial
from pathlib import Path
from threading import Lock
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Signal
from common.components import ComponentBase
from common.components import action
from common.consts import c_thz
from common.dataset_cache import DatasetCache, remove_cache_dir
from common.default_appsettings import Dist, ReferenceClassification, Domain, QuantityEnum, WindowTypes, ComponentToggle
from common.functions import (butter_filt, do_fft, f_axis_idx_map,
                              remove_offset, avg_data_array, calculate_bandwidth, to_db)
from common.measurement_selection import MeasurementSelection, get_coordinate_line
from common.measurements import Measurement
from common.settings import Settings
from common.traits import Q_, Path as TPath, Quantity, ValueRange
from scipy.signal import get_window
from scipy.stats import pearsonr
from traitlets import Unicode, observe, Float, Bool, Enum as TEnum, Instance, Int, Dict

"""
TODOs: 
- How are measurements mapped when multiple measurements are performed at the same x-y coordinates?
- Setting cbar lims sucks... set lims based on area min max?
- Fix runtime / use cache for t calc?
- check if filesizes? match when making npy array
- window function (fuctions.py): allow negative values (wrap around) + fix plot (clipping)
- self.logger is messy, also fix log levels and RuntimeWarnings
- Fix unit labeling
- should phi correction be a part of the pre-processing?
- q-eval: svmaf, estimate #FP reflections and FP spacing for q-space freq range

# units:
[l] = µm, [t] = ps, [alpha] = 1/cm (absorption coe.), [sigma] = S/cm, [eps0] = Siemens * ps,
[f] = THz (1/ps), [c_thz] = µm/ps
"""

class Averaging(Enum):
    only_avg = "Average only"
    only_individual = "Individual measurements only"
    both = "Average and individual measurements"


class DataSetFormatter(ComponentBase):
    freq_range = None
    averaging = None

    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.measurement_selector = dataset.measurement_selector

    def __enter__(self, *args) -> DataSetFormatter:
        if self.settings is not None:
            self.settings.load_configuration(self)

        return self

    def __exit__(self, *args):
        if self.settings is not None:
            self.settings.save_configuration(self)

    @property
    def measurements(self):
        return self.dataset.measurements

    @property
    def settings(self):
        return self.dataset.settings

    @property
    def freq_axis(self):
        f_axis = self.dataset.freq_axis[self.freq_range_idx]
        return f_axis

    @property
    def freq_range_idx(self):
        return f_axis_idx_map(self.dataset.freq_axis, self.freq_range)

    def format_meas_dict(self, meas_list, data_array: np.ndarray | None, is_td=False):
        if data_array is None:
            return {}
        if not is_td and data_array.shape[1] != len(self.freq_axis):
            if data_array.ndim == 2:
                data_array = data_array[:, self.freq_range_idx]
            elif data_array.ndim == 3:
                data_array = data_array[:, self.freq_range_idx, :]

        if data_array.ndim == 2:
            freq_tile = np.tile(self.freq_axis, (len(meas_list), 1))
            data_array = np.stack((freq_tile, data_array, np.zeros_like(freq_tile)), axis=2)

        if len(meas_list) == 1:
            return {meas_list[0]: data_array[0]}

        average_y_meas = {"Average": avg_data_array(data_array)}
        individual_y_meas = {
            meas: data_array[meas_idx]
            for meas_idx, meas in enumerate(meas_list)
        }
        match self.averaging:
            case Averaging.only_avg:
                return average_y_meas
            case Averaging.only_individual:
                return individual_y_meas
            case Averaging.both:
                return {**average_y_meas, **individual_y_meas}
        return {}


class WindowEvalResult:
    def __init__(self, **kwargs):
        super().__init__()
        self.meas_list = kwargs.get("meas", None)
        self.t = kwargs.get("t", None)
        self.freq_axis = kwargs.get("freq_axis", None)
        self.refr_idx = kwargs.get("refr_idx", None)
        self.abs_coe = kwargs.get("abs_coe", None)
        self.phi = kwargs.get("phi", None)
        self.phi_corrected = kwargs.get("phi_corrected", None)
        self.d = kwargs.get("d", None)


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

    td_info_grp = "Time domain info"
    sampling_start = Quantity(Q_(0, "ps"), read_only=True).tag(name="Sampling start", group=td_info_grp, priority=1000)
    sampling_end = Quantity(Q_(0, "ps"), read_only=True).tag(name="Sampling end", group=td_info_grp, priority=1001)
    sampling_period = Quantity(Q_(0, "fs"), read_only=True).tag(name="Sampling period", group=td_info_grp,
                                                                priority=1002)
    sample_count = Int(0, read_only=True).tag(name="Time samples", group=td_info_grp, priority=1003)
    sampling_window = Quantity(Q_(0, "ps"), read_only=True).tag(name="Sampling window", group=td_info_grp,
                                                                priority=1004)

    fd_info_grp = "Frequency domain info"
    min_freq = Quantity(Q_(0, "THz"), read_only=True).tag(name="Minimum frequency axis value", group=fd_info_grp,
                                                          priority=1000)
    max_freq = Quantity(Q_(0, "THz"), read_only=True).tag(name="Maximum frequency axis value", group=fd_info_grp,
                                                          priority=1001)
    frequency_resolution = Quantity(Q_(0, "GHz"), read_only=True).tag(name="Frequency resolution", group=fd_info_grp,
                                                                      priority=1002)
    spectral_line_cnt = Int(0, read_only=True).tag(name="Frequency samples", group=fd_info_grp, priority=1003)
    nyquist_frequency = Quantity(Q_(0, "THz"), read_only=True).tag(name="Nyquist frequency", group=fd_info_grp,
                                                                   priority=1004)

    peak2peak = Float(0.0, read_only=True).tag(name="Peak to peak", group="Maximum amplitude measurement")
    snr = Quantity(Q_(0.0, "dB"), read_only=True).tag(name="Signal to noise ratio",
                                                      group="Maximum amplitude measurement")
    signal_bandwidth = Quantity(Q_(0.0, "THz"), read_only=True).tag(name="Signal bandwidth",
                                                                    group="Maximum amplitude measurement")

    selected_measurement_info = Unicode("", read_only=True).tag(name="Selected measurements",
                                                                group="Selected measurement")

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
    ref_point = ValueRange([Q_(0.0, "mm"), Q_(0.0, "mm")]).tag(group=reference_filter_group,
                                                               name="Single reference point")

    data_export_grp = "Data export"
    export_csv_dir = TPath(Path(""), is_file=False).tag(name="Save directory", group=data_export_grp)
    data_export_label = Unicode("", group=data_export_grp).tag(name="Data export label")

    info_pane = Instance(DataSetInfoPane, allow_none=True).tag(toggle=ComponentToggle.en_info_pane)
    measurement_selector = Instance(MeasurementSelection)

    measurements = Dict(default_value={"refs": [], "sams": [], "all": ()})

    def __init__(self, settings: Settings, **kwargs):
        super().__init__(**kwargs)
        self._parse_lock = Lock()
        self._is_parsing = False
        self.set_trait("is_initialized", False)

        self.measurements = {"refs": [], "sams": [], "all": (), "max_amp_meas": None}

        self.cache = None
        self.sub_dataset = None
        self.quantity_funcs = {}

        self.settings: Settings = settings

        self.measurement_selector = MeasurementSelection(self, object_name="Measurement selection")

        self._freq_axis = None

    def __enter__(self, *args):
        super().__enter__(self, *args)
        if self.settings is not None:
            self.settings.load_configuration(self)
        self._set_observers()
        self.measurement_selector.set_observers()

        if self.settings.component_toggle.en_info_pane:
            self.info_pane = DataSetInfoPane(object_name="Info pane")

        self._parse_measurements()

        return self

    def __exit__(self, *args):
        super().__exit__(self, *args)
        if self.settings is not None:
            self.settings.save_configuration(self)

    @property
    def freq_axis(self):
        if self._freq_axis is None:
            self._single_meas_data(self.measurements["all"][0], domain=Domain.Frequency)

        return self._freq_axis

    @property
    def _func_map(self):
        func_map = {QuantityEnum.P2P: self.p2p,  # 1D (N_meas)
                    QuantityEnum.Phase: self.phase,  # 2D (N_meas, Freq_slice)
                    QuantityEnum.MeasTimeDeltaRef2Sam: self.meas_time_delta,  # 0D scalar
                    QuantityEnum.Power: self.power,  # 2D (N_meas, Freq_slice)
                    QuantityEnum.Absorbance: self.absorbance,  # 2D (N_meas, Freq_slice)
                    QuantityEnum.RefAmp: self.ref_max,  # 2D (N_meas, Freq_slice)
                    QuantityEnum.RefArgmax: self.get_ref_abs_argmax,  # 1D (N_meas)
                    QuantityEnum.RefPhase: self.ref_phase,  # 2D (N_meas, Freq_slice)
                    QuantityEnum.ZeroCrossing: self.get_zero_crossing,  # 1D (N_meas)
                    QuantityEnum.TimeOfFlight: self.time_of_flight,  # 1D (N_meas)
                    QuantityEnum.PhaseToF: self.delay_from_phase_slope,  # 1D (N_meas)
                    QuantityEnum.Transmission: self.transmission,  # 2D (N_meas, Freq_slice)
                    QuantityEnum.TransmissionAmp: self.amplitude_transmission,  # 2D (N_meas, Freq_slice)
                    QuantityEnum.TransmissionPhase: self.phase_difference,  # 2D (N_meas, Freq_slice)
                    }
        return func_map

    def quantity_func(self, quantity: QuantityEnum, freq_range=None, d_sam=None):
        func_map = self._func_map
        func_map[QuantityEnum.PowerInt] = partial(self.power_int, freq_range=freq_range)  # 1D (N_meas)
        func_map[QuantityEnum.PeakCnt] = partial(self.simple_peak_cnt, threshold=2.5)  # 1D (N_meas)

        func_map[QuantityEnum.RIEstimate] = partial(self.refractive_index_estimate, d=d_sam)  # 2D (N_meas, Freq_slice)
        func_map[QuantityEnum.RefractiveIdx] = partial(self.refractive_idx, d_sam=d_sam)  # 2D (N_meas, Freq_slice)
        func_map[QuantityEnum.ExtCoe] = partial(self.extinction_coe, d_sam=d_sam)  # 2D (N_meas, Freq_slice)
        func_map[QuantityEnum.AbsorptionCoe] = partial(self.absorption_coef, d_sam=d_sam)  # 2D (N_meas, Freq_slice)

        # quantities provided by evaluation components, e.g. Conductivity (ConductivityEval)
        func_map.update(self.quantity_funcs)

        if quantity not in func_map:
            raise KeyError(f"No function for {quantity.name}, is the evaluation component providing it enabled?")

        return func_map[quantity]

    def register_quantity_func(self, quantity: QuantityEnum, func):
        # func(meas_list) -> (N_meas) or (N_meas, Freq) array
        self.quantity_funcs[quantity] = func

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

    @observe("data_path")
    def _set_path(self, change=None):
        if change is None:
            return
        self.data_path = change["new"]
        self._parse_measurements(self.data_path)

        self.set_trait("caching_progress", 1)

    def is_ref(self, meas: Measurement) -> bool:
        if not isinstance(meas, Measurement):
            return False
        return meas in self.measurements["refs"]

    def _set_observers(self):
        reference_filter_trait_names = self.trait_names(group=self.reference_filter_group)
        self.observe(self.update_meas_sorting, names=reference_filter_trait_names)

    def update_cache_progress(self, progress):
        self.set_trait("caching_progress", progress)

    @action("Clear cache")
    def clear_cache(self):
        if self.cache is not None:
            self.cache.clear_cache()
        elif self.data_path:
            # also when building the cache failed, e.g. on a cache file of an older format
            try:
                self.logger.info(f"Cleared {remove_cache_dir(self.data_path)}")
            except OSError as e:
                self.logger.warning(f"Could not remove the cache directory: {e}")
        self.set_trait("is_initialized", False)

    @action("Reparse measurements")
    def reparse_measurements(self):
        self.clear_cache()
        self._parse_measurements()

    def _apply_window(self, data_td):
        t, y = data_td[:, 0], data_td[:, 1]
        dt = np.mean(np.diff(t))

        win_opt = self.settings.pp_opt.window
        win_enum = win_opt.type
        sym = win_opt.symmetric
        win_width = win_opt.win_width
        win_start = win_opt.win_start
        center_window = win_opt.center_on_pulse
        shift = win_opt.shift
        en_plot = win_opt.en_plot

        HFT90D = [1, 1.942604, 1.340318, 0.440811, 0.043097]

        param_map = {
            WindowTypes.tukey: (win_opt.tukey_alpha,),
            WindowTypes.chebwin: (win_opt.chebwin_at,),
            WindowTypes.dpss: (win_opt.dpss_nw,),
            WindowTypes.exponential: (win_opt.exp_center if not sym else None,
                                      win_opt.exp_tau,),
            WindowTypes.gaussian: (win_opt.gaussian_std,),
            WindowTypes.general_cosine: (HFT90D,),
            WindowTypes.general_gaussian: (win_opt.general_gauss_p, win_opt.general_gauss_sig,),
            WindowTypes.general_hamming: (win_opt.general_hamming_alpha,),
            WindowTypes.kaiser: (win_opt.kaiser_beta,),
            WindowTypes.kaiser_bessel_derived: (win_opt.kaiser_bessel_beta,),
            WindowTypes.taylor: (win_opt.taylor_nbar, win_opt.taylor_sll, win_opt.taylor_norm,),
        }
        win_param = (win_enum.name, *param_map.get(win_enum, ()))

        win_width = int(win_width.magnitude / dt)

        if win_width > len(y):
            win_width = len(y)

        win_arr = get_window(win_param, win_width, fftbins=not sym)

        if center_window:
            win_center = np.argmax(np.abs(y))
            win_start = win_center - int(win_width / 2)
        else:
            win_start = int(win_start.magnitude / dt)

        window_mask = np.zeros_like(y)
        window_mask[:win_width] = win_arr

        window_mask = np.roll(window_mask, win_start)
        if win_start < 0:
            window_mask[len(y) + win_start:] = 0

        window_mask = np.roll(window_mask, int(shift.magnitude / dt))

        y_win = y * window_mask

        fig_num = "Window"
        if en_plot and not plt.fignum_exists(fig_num):
            plt.figure(fig_num)
            plt.plot(t, y, label="Before windowing")
            plt.plot(t, np.max(np.abs(y)) * window_mask, label="Window")
            plt.plot(t, y_win, label="After windowing")
            plt.xlabel("Time (ps)")
            plt.ylabel("Amplitude (nA)")
            plt.legend()

        return np.array([t, y_win]).T

    def _pre_process(self, meas_):
        pp_opt = self.settings.pp_opt

        cache_idx = self.cache.id_map[meas_.identifier]
        data_td = np.array([self.cache.time_axis,
                            self.cache.raw_data_td[cache_idx]], dtype=float).T

        if pp_opt.normalize_data:
            data_td[:, 1] = data_td[:, 1] / np.max(data_td[:, 1])

        if pp_opt.remove_dc:
            data_td = remove_offset(data_td)

        if pp_opt.window.window_enabled:
            data_td = self._apply_window(data_td)

        if pp_opt.filter_enabled:
            data_td = butter_filt(data_td, pp_opt.f_range.magnitude)

        return data_td

    def _single_meas_data(self, meas, domain=Domain.Time):
        data_td = self._pre_process(meas)

        if domain == Domain.Time:
            data = data_td
        else:
            data = do_fft(data_td)
            if self._freq_axis is None:
                self._freq_axis = data[:, 0].real
        data = np.insert(data, 2, 0, axis=1)

        return data

    def get_multi_data(self, meas_list, domain=Domain.Time) -> np.ndarray:
        meas_list = [meas_list] if isinstance(meas_list, Measurement) else meas_list

        sample_data = self._single_meas_data(meas_list[0], domain=domain)
        shape = sample_data.shape

        data_arr = np.zeros([len(meas_list), *shape], dtype=sample_data.dtype)
        data_arr[0] = sample_data
        if len(meas_list) == 1:
            return data_arr
        else:
            for meas_idx, meas in enumerate(meas_list[1:]):
                data_arr[meas_idx + 1] = self._single_meas_data(meas, domain=domain)

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

        if self.info_pane is not None:
            self._set_dataset_info()
        self.refresh_shape_properties()

    def _parse_measurements(self, data_path=None):
        if data_path is None:
            data_path = self.data_path
        if data_path is None:
            return

        self.set_trait("is_initialized", False)
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

    def update_meas_sorting(self, change=None):
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

        if self.info_pane is not None:
            self.set_shape_info_traits(shape)

        return shape

    def set_shape_info_traits(self, shape) -> None:
        self.info_pane.set_trait("shape_w_info", str(shape["w"]))
        self.info_pane.set_trait("shape_h_info", str(shape["h"]))
        self.info_pane.set_trait("shape_dx_info", str(shape["dx"]))
        self.info_pane.set_trait("shape_dy_info", str(shape["dy"]))
        self.info_pane.set_trait("pixel_cnt", str(int((shape["h"] / shape["dy"]) * (shape["w"] / shape["dx"]))))

        x_extrema_str = f"{np.min(shape['x_coords'])}, {np.max(shape['x_coords'])}"
        self.info_pane.set_trait("x_coord_extrema", x_extrema_str)

        y_extrema_str = f"{np.min(shape['y_coords'])}, {np.max(shape['y_coords'])}"
        self.info_pane.set_trait("y_coord_extrema", y_extrema_str)

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
        self.info_pane.set_trait("sampling_window", t_end - t_start)
        self.info_pane.set_trait("sampling_period", (t_end - t_start) / data_td.shape[1])

        self.info_pane.set_trait("frequency_resolution", Q_(1e3 * np.mean(np.diff(data_fd[0, :, 0].real)), "GHz"))
        self.info_pane.set_trait("spectral_line_cnt", data_fd.shape[1])
        self.info_pane.set_trait("nyquist_frequency", Q_(data_fd[0, -1, 0].real, "THz"))
        self.info_pane.set_trait("min_freq", Q_(data_fd[0, 0, 0].real, "THz"))
        self.info_pane.set_trait("max_freq", Q_(data_fd[0, -1, 0].real, "THz"))

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
            case ReferenceClassification.single_point:
                refs_ = self.measurement_selector.get_measurements_from_point(*self.ref_point)
            case ReferenceClassification.max_amp:
                refs_ = new_measurements["max_amp_meas"]
                if isinstance(refs_, Measurement):
                    refs_ = [refs_]
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

    def windowing_eval(self, meas_, d=None) -> WindowEvalResult:
        ref_meas = self.measurement_selector.get_matching_refs(meas_)

        with self.settings.pp_opt.window.override(window_enabled=True, win_width=Q_(10, "ps")):
            ref_fd = self.get_multi_data(ref_meas, Domain.Frequency)
            sam_fd = self.get_multi_data(meas_, Domain.Frequency)

            phi_corrected = np.abs(self.phase_difference(meas_))

        freq_axis = self.freq_axis
        omega = 2 * np.pi * freq_axis

        if d is None:
            self.logger.warning("No sample thickness passed to windowing_eval, using settings default")
            d = self.settings.eval_opt.d
        d = d.magnitude if hasattr(d, "magnitude") else d
        d = 1.0 if np.isclose(d, 0) else d
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
               "d": d,
               }

        return WindowEvalResult(**ret)

    def refractive_idx(self, meas_, d_sam=None):
        return np.real(self.windowing_eval(meas_, d=d_sam).refr_idx)

    def absorption_coef(self, meas_, d_sam=None):
        return self.windowing_eval(meas_, d=d_sam).abs_coe

    def extinction_coe(self, meas_, d_sam=None):
        return np.imag(self.windowing_eval(meas_, d=d_sam).refr_idx)

    def get_ref_abs_argmax(self, meas_):
        ref_meas = self.measurement_selector.get_matching_refs(meas_)
        ref_td = self.get_multi_data(ref_meas)
        t, y = ref_td[:, :, 0], ref_td[:, :, 1]

        return t[np.arange(ref_td.shape[0]), np.argmax(np.abs(y), axis=1)]

    def spectral_similarity(self, meas_0=None, meas_1=None, data_arrays_fd=None):
        if (meas_0 is None) and (meas_1 is None) and (data_arrays_fd is None):
            return None

        if data_arrays_fd is None:
            data_fd_0 = self.get_multi_data(meas_0, domain=Domain.Frequency)
            if meas_1 is None:
                meas_1 = self.measurement_selector.get_matching_refs(meas_0)
            data_fd_1 = self.get_multi_data(meas_1, domain=Domain.Frequency)
        else:
            data_fd_0, data_fd_1 = data_arrays_fd

        freq_min, freq_max = self.settings.eval_opt.spectral_similarity_range.magnitude
        f_idx_range = f_axis_idx_map(self.freq_axis, (freq_min, freq_max))
        x, y = np.abs(data_fd_0[:, f_idx_range, 1]), np.abs(data_fd_1[:, f_idx_range, 1])

        return 1 + np.log(np.abs(pearsonr(x, y, axis=1).statistic))

    def delay_from_phase_slope(self, meas_0=None, meas_1=None, data_arrays=None):
        if data_arrays is None:
            phi_diff = self.phase_difference(meas_0, meas_1)
        else:
            phi_diff = self.phase_difference(data_arrays=data_arrays)
        freq_min, freq_max = self.settings.eval_opt.avg_grp_delay_fit_range.magnitude
        mask = (freq_min <= self.freq_axis) & (self.freq_axis <= freq_max)

        p = np.polyfit(self.freq_axis[mask], phi_diff[:, mask].T, 1)

        tau = -p[0] / (2 * np.pi)

        return tau

    def get_zero_crossing(self, meas_=None, data_td=None):
        if data_td is None:
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

        zero_crossing_interp = np.zeros(y.shape[0], dtype=float)
        zero_crossing_interp[has_zero_crossing] = ((y1 * x2 - x1 * y2) / dy)[has_zero_crossing]

        return zero_crossing_interp

    def pulse_amplitude(self, meas_: Measurement):
        y_td = self.get_multi_data(meas_)
        return np.max(y_td[:, :, 1], axis=1)

    def p2p(self, meas_: Measurement):
        y_td = self.get_multi_data(meas_)
        return np.max(y_td[:, :, 1], axis=1) - np.min(y_td[:, :, 1], axis=1)

    def phase(self, meas_: Measurement):
        y_fd = self.get_multi_data(meas_, domain=Domain.Frequency)
        return np.angle(y_fd[:, :, 1])

    def unwrapped_phase(self, meas_: Measurement):
        phi = self.phase(meas_)
        return np.unwrap(phi, axis=1)

    def absorbance(self, meas_: Measurement, ref_meas_=None):
        return -to_db(self.power(meas_, ref_meas_)) / 20

    def power(self, meas_: Measurement, ref_meas_=None):
        ref_meas = ref_meas_ if ref_meas_ else self.measurement_selector.get_matching_refs(meas_)
        ref_fd = self.get_multi_data(ref_meas, domain=Domain.Frequency)
        sam_fd = self.get_multi_data(meas_, domain=Domain.Frequency)

        power_val_ref = np.abs(ref_fd[:, :, 1])
        power_val_sam = np.abs(sam_fd[:, :, 1])

        return (power_val_sam / power_val_ref) ** 2

    def power_int(self, meas_: Measurement, ref_meas_=None, freq_range=None):
        if freq_range is None:
            self.logger.warning("No frequency range passed, using full range for power integration")
            freq_range = (0, len(self.freq_axis))
        freq_slice = (freq_range[0] < self.freq_axis) * (self.freq_axis < freq_range[1])

        ref_meas = ref_meas_ if ref_meas_ else self.measurement_selector.get_matching_refs(meas_)
        ref_fd = self.get_multi_data(ref_meas, domain=Domain.Frequency)
        sam_fd = self.get_multi_data(meas_, domain=Domain.Frequency)

        power_val_ref = np.sum(np.abs(ref_fd[:, freq_slice, 1]) ** 2, axis=1)
        power_val_sam = np.sum(np.abs(sam_fd[:, freq_slice, 1]) ** 2, axis=1)

        return power_val_sam / power_val_ref

    def meas_time_delta(self, meas_: Measurement, ref_meas_=None):
        meas_list = [meas_] if isinstance(meas_, Measurement) else meas_

        get_nearest_ref = self.measurement_selector.get_nearest_ref
        ref_list = ref_meas_ if ref_meas_ else [get_nearest_ref(meas) for meas in meas_list]

        meas_times = [(meas.meas_time - ref_meas.meas_time).total_seconds()
                      for (meas, ref_meas) in zip(meas_list, ref_list)]
        return np.array(meas_times, dtype=float)

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
            m1 = self.measurement_selector.get_matching_refs(m0)

        m1_fd = self.get_multi_data(m1, Domain.Frequency)
        m0_fd = self.get_multi_data(m0, Domain.Frequency)

        phi_r_m1 = np.angle(m1_fd[:, :, 1])
        phi_r_m0 = np.angle(m0_fd[:, :, 1])
        return np.unwrap(phi_r_m0 - phi_r_m1, axis=1)

    def phase_difference(self, m0=None, m1=None, data_arrays=None):
        # phase unwrapping Phase Retrieval in Terahertz Time-Domain
        # Measurements: a how to Tutorial P. Uhd Jepsen https://doi.org/10.1007/s10762-019-00578-0
        if data_arrays is None:
            if m1 is None:
                m1 = self.measurement_selector.get_matching_refs(m0)

            m0_fd = self.get_multi_data(m0, Domain.Frequency)
            m1_fd = self.get_multi_data(m1, Domain.Frequency)
            m0_td = self.get_multi_data(m0, Domain.Time)
            m1_td = self.get_multi_data(m1, Domain.Time)
        else:
            m0_td, m0_fd, m1_td, m1_fd = data_arrays

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

        if self.settings.eval_opt.phi_offset_correction:
            fit_slice = (f_axis >= phi_fit_range[0]) * (f_axis <= phi_fit_range[1])
            p = np.polyfit(f_axis[fit_slice], phi0_star[:, fit_slice].T, 1)
            phi0 = phi0_star - 2 * np.pi * np.round(p[1][:, None] / (2 * np.pi))
        else:
            phi0 = phi0_star

        # add the removed linear phase (pulse positions) back
        phi = phi0 - phi0_m1 + phi0_m0 + w_axis * t_offset[:, None]

        return phi

    def amplitude_transmission(self, meas_, ref_meas_=None):
        if ref_meas_ is None:
            ref_meas_ = self.measurement_selector.get_matching_refs(meas_)

        ref_fd = self.get_multi_data(ref_meas_, Domain.Frequency)
        sam_fd = self.get_multi_data(meas_, Domain.Frequency)

        return np.abs(sam_fd[:, :, 1] / ref_fd[:, :, 1])

    def _raw_transmission(self, meas_, ref_meas_=None):
        t_abs = self.amplitude_transmission(meas_, ref_meas_)
        phase_difference = self.phase_difference(meas_, ref_meas_)

        return t_abs * np.exp(1j * phase_difference)

    def transmission(self, meas_, ref_meas_=None):
        t = self._raw_transmission(meas_, ref_meas_)

        drift = self.settings.pp_opt.ref_drift
        if (drift.interpolate_refs or drift.block_median) and isinstance(meas_, (list, tuple, np.ndarray)) and len(meas_):
            ref_list = list(ref_meas_) if ref_meas_ else self.measurement_selector.get_matching_refs(meas_)
            t = self._drift_corrected_transmission(list(meas_), ref_list, t, drift)

        return t

    def _reference_blocks(self, drift):
        # references sorted by time, split into blocks (each reference its own block without the block median), the
        # mean measurement time of each block and the block index of each reference
        refs = sorted(self.measurements["refs"], key=lambda m: m.meas_time)
        times = np.array([r.meas_time.timestamp() for r in refs])
        if drift.block_median:
            splits = np.flatnonzero(np.diff(times) > drift.block_gap.to("s").magnitude) + 1
        else:
            splits = np.arange(1, len(refs))
        bounds = list(zip(np.r_[0, splits], np.r_[splits, len(refs)]))
        blocks = [refs[a:b] for a, b in bounds]
        centers = np.array([times[a:b].mean() for a, b in bounds])
        block_of = {ref: bi for bi, block in enumerate(blocks) for ref in block}

        return blocks, centers, block_of

    @staticmethod
    def _geometric_interp(t0, t1, w):
        # t0 * (t1 / t0)^w, w: (N, 1). The principal phase of the ratio is enough for drifts of ~100 fs up to 3 THz
        ratio = t1 / np.where(t0 == 0, 1, t0)
        return t0 * np.abs(ratio) ** w * np.exp(1j * w * np.angle(ratio))

    def _drift_corrected_transmission(self, meas_list, ref_list, t, drift):
        # t of each sample against its reference divided by the block median of that reference's block (relative to
        # the reference), then interpolated in time to the reference (block) on the other side of the sample
        blocks, centers, block_of = self._reference_blocks(drift)
        if not blocks:
            return t
        median_cache = {}

        def block_factor(bi, anchor):
            # median over the block of R_k / R_anchor, so t(s, anchor) / factor = S / R_median
            if not drift.block_median or len(blocks[bi]) < 2:
                return 1.0
            if (bi, anchor) not in median_cache:
                ratios = self._raw_transmission(blocks[bi], [anchor] * len(blocks[bi]))
                median_cache[(bi, anchor)] = (np.median(np.abs(ratios), axis=0)
                                              * np.exp(1j * np.median(np.angle(ratios), axis=0)))
            return median_cache[(bi, anchor)]

        max_gap = drift.max_interp_gap.to("s").magnitude
        t = np.array(t, dtype=complex)
        partner_idx, partner_meas, partner_refs, weights, partner_blocks = [], [], [], [], []
        for i, (meas, ref) in enumerate(zip(meas_list, ref_list)):
            bi = block_of.get(ref)
            if bi is None:
                continue
            t[i] = t[i] / block_factor(bi, ref)
            if not drift.interpolate_refs:
                continue
            t_s = meas.meas_time.timestamp()
            bj = bi + 1 if t_s >= centers[bi] else bi - 1
            if not 0 <= bj < len(blocks) or abs(centers[bj] - centers[bi]) > max_gap:
                continue
            partner = min(blocks[bj], key=lambda r: abs(r.meas_time.timestamp() - t_s))
            partner_idx.append(i)
            partner_meas.append(meas)
            partner_refs.append(partner)
            partner_blocks.append(bj)
            weights.append(np.clip((t_s - centers[bi]) / (centers[bj] - centers[bi]), 0.0, 1.0))

        if partner_idx:
            t_partner = self._raw_transmission(partner_meas, partner_refs)
            for k, (bj, partner) in enumerate(zip(partner_blocks, partner_refs)):
                t_partner[k] = t_partner[k] / block_factor(bj, partner)
            t[partner_idx] = self._geometric_interp(t[partner_idx], t_partner, np.array(weights)[:, None])

        return t

    def time_of_flight(self, meas_, ref_meas_=None):
        closest_ref = ref_meas_ if ref_meas_ else self.measurement_selector.get_matching_refs(meas_)

        t_zero_ref = self.get_zero_crossing(closest_ref)
        t_zero_sam = self.get_zero_crossing(meas_)

        return t_zero_sam - t_zero_ref

    def refractive_index_estimate(self, meas_, ref_meas_=None, d=None):
        ref_list = ref_meas_ if ref_meas_ else self.measurement_selector.get_matching_refs(meas_)
        dt = self.time_of_flight(meas_, ref_list)

        if d is None:
            self.logger.warning("No sample thickness passed to windowing_eval, using settings default")
            d = self.settings.eval_opt.d
        d = d.magnitude if hasattr(d, "magnitude") else d
        d = 1.0 if np.isclose(d, 0) else d

        w = 2 * np.pi * self.freq_axis
        w = np.where(w == 0, w[1] if len(w) > 1 else 1e-12, w)

        with self.settings.pp_opt.window.override(window_enabled=True, win_width=Q_(10, "ps")):
            t_amp = self.amplitude_transmission(meas_, ref_list)
            phi_diff = np.abs(self.phase_difference(meas_, ref_list))

        n_real = 1.0 + (c_thz * phi_diff / (w * d))
        t_interface_amp = np.abs(4 * n_real / (n_real + 1) ** 2)

        n_imag = -(c_thz / (w * d)) * np.log(t_amp / t_interface_amp)

        n_real = 1.0 + (c_thz * dt / d)[:, None]

        return n_real + 1j * n_imag

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

        y_fd_interpol = fd_before - ((t_before / t_span)[:, None]) * (fd_after - fd_before)

        freq_grid = np.tile(self.freq_axis, (len(sam_meas), 1))
        std_grid = np.tile(np.zeros_like(self.freq_axis), (len(sam_meas), 1))

        return np.stack((freq_grid, y_fd_interpol, std_grid), axis=-1)

    def meas_time_diff(self, m1, m2):
        val = (m2.meas_time - m1.meas_time).total_seconds()
        return Q_(val, "s")

    @action("Export measurement data", group=data_export_grp)
    def export_measurement_data(self):
        selection = self.measurement_selector.get_selection()
        if selection is None:
            return
        selected_meas, ref_meas = selection.sams, selection.refs

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


if __name__ == '__main__':
    options = {
        # "cbar_lim": (0.52, 0.60), # 2.5 THz
        # "cbar_lim": (0.64, 0.66), # img4
        "cbar_lim": (0.60, 0.66),  # img5 1.5 THz
        # "cbar_lim": (0.55, 0.62), # img5 2.0 THz
        "plot_opt": {"plot_range": slice(30, 650), },
        "ref_pos": (10.0, 0.0),
        "fig_label": "",
        "dist_func": Dist.Position,

    }

    # logging.basicConfig(level=self.logger.INFO)
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
    dataset.average_area((23, 2), (28, 4), label="12.2")  # img5

    # img 1
    # dataset.average_area((4, 8), (20, 13), label="8")
    # dataset.average_area((36, 8), (51, 13), label="7")
    # dataset.average_area((35, -13.5), (50, -11), label="4")
    # dataset.average_area((70, -14.25), (79, -13), label="2")
    # dataset.average_area((64, 13.50), (77, 14.25), label="9")

    # img 2
    # dataset.average_area((9, -10), (30, 2), label="A1") # VR04
    # dataset.average_area((45, -10), (60, 0), label="A2") # VR01_1
    # dataset.average_area((68, -12), (70, -1.5), label="A3") # VR01_2

    # dataset.plot_point((10, 14), label="SC 1")
    # dataset.plot_point((40, 14), label="SC 2")
    # dataset.plot_point((10, -14), label="SC 3")
    # dataset.plot_point((40, -14), label="SC 4")

    # dataset2 = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Laser_crystallized_Si")

    # point = choice(dataset.img_properties["all_points"])
    # img.window_all()
    # dataset.plot_point((45, 14), label="Wood substrate (x=45 mm)")
    # dataset.plot_point((50, 14), label="Wood substrate (x=50 mm)")
    # dataset.plot_point((5, 14), label="Reference (air)")
    # dataset2.plot_point((60, -14), label="Borofloat sub.")
    # dataset2.plot_point((30, -14), label="Sub. + SiN sub.")
    # dataset2.plot_point((5, -14), label="Si B-doped")
    # dataset2.plot_point((60, 14), label="Si + SiN interlayer + Borofloat")
    # dataset2.plot_point((30, 14), label="Si n-doped + SiN interlayer + Borofloat")

    # dataset.plot_line(y=-14.00, label="y=-14 mm")
    # dataset.plot_line(y=14.00, label="y=14 mm")

    # dataset2 = DataSet(r"/home/ftpuser/ftp/Data/IPHT2/Wood/S5")
    # dataset2.plot_line(y=-12.00, label="y=-12")
    # dataset2.plot_line(y=12.00, label="y=12")

    # dataset2.plot_point((60, 12), label="Wood + Lacquer (Pure)")
    # dataset2.plot_point((60, -12), label="Wood + Lac. (1:1)")
    # dataset2.plot_point((40, 12), label="Wood + Lac. (1:2)")
    # dataset2.plot_point((40, -12), label="Wood + Lac. (1:5)")

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
