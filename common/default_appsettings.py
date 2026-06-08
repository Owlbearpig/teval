import logging
from pathlib import Path
from enum import Enum
from traitlets import (
    HasTraits, Bool, Int, Float, Unicode, Tuple,
    List as TList, Dict as TDict, Enum as TEnum, Instance, Any as TAny
)

import common.consts
from common import traits
from common.components import ComponentBase

class Domain(Enum):
    Time = 0
    Frequency = 1
    Both = 2


class MeasurementType(Enum):
    REF = 1
    SAM = 2


class PixelInterpolation(Enum):
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


class WindowTypes(Enum):
    tukey = "tukey"
    hannin = "hanning"
    rectangular = "rectangular"


class QuantityFunc:
    func = None

    def __init__(self, label="label", func=None, domain=None, unit=""):
        self.label = label
        if domain is None:
            self.domain = Domain.Time
        else:
            self.domain = domain
        if func is not None:
            self.func = func
        self.unit = bool(unit)*" (" + unit + bool(unit)*")"

    def __repr__(self):
        return self.label

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class QuantityEnum(Enum):
    P2P = QuantityFunc("Peak to peak", domain=Domain.Time)
    Power = QuantityFunc("Power", domain=Domain.Frequency)
    Phase = QuantityFunc("Phase", domain=Domain.Frequency, unit="rad")
    MeasTimeDeltaRef2Sam = QuantityFunc("Time delta Ref. to Sam.", domain=Domain.Time)
    RefAmp = QuantityFunc("Ref. Amp", domain=Domain.Frequency)
    RefArgmax = QuantityFunc("Ref. Argmax", domain=Domain.Time)
    RefPhase = QuantityFunc("Ref. Phase", domain=Domain.Frequency)
    PeakCnt = QuantityFunc("Peak Cnt", domain=Domain.Time)
    ZeroCrossing = QuantityFunc("Zero Crossing", domain=Domain.Time, unit="ps")
    TimeOfFlight = QuantityFunc("Time of Flight", domain=Domain.Time, unit="ps")
    TransmissionAmp = QuantityFunc("Amplitude transmission", domain=Domain.Frequency)
    TransmissionPhase = QuantityFunc("Phase transmission", domain=Domain.Frequency, unit="rad")
    RefractiveIdx = QuantityFunc("Refractive idx", domain=Domain.Frequency)
    AbsorptionCoe = QuantityFunc("Absorption coe", domain=Domain.Frequency, unit="1/cm")


class LogLevel(Enum):
    info = logging.INFO
    debug = logging.DEBUG
    warning = logging.WARNING
    error = logging.ERROR
    critical = logging.CRITICAL



class EvalOpt(ComponentBase):
    dt = Int(0)
    sub_pnt = Tuple(Float(), Float(), default_value=(0.0, 0.0))
    fit_range = Tuple(Float(), Float(), default_value=(0.50, 2.20))
    q_space_range = Tuple(Float(), Float(), default_value=(0.75, 2.00))
    phi_fit_range = Tuple(Float(), Float(), default_value=(0.47, 1.05))
    average = Bool(False)
    delta_d = Float(2.0)
    phi_offset_correction = Bool(True)
    printed_freqs = TList(trait=Float(), default_value=[1.000, 2.000])
    d_opt_axis = TAny(None, allow_none=True)

class PpOpt(ComponentBase):
    enabled = Bool(False)
    dt = Int(0)

    window_group = "Window options"
    enabled = Bool(False, group=window_group)
    win_width = Int(10, group=window_group)
    shift = Float(0, group=window_group)
    slope = Float(0.15, group=window_group)
    en_plot = Bool(False, group=window_group)
    type = TEnum(WindowTypes, default_value=WindowTypes.tukey, group=window_group)

    filter_group = "Filter options"
    f_range = Tuple(Float(), Float(), default_value=(0.3, 3.0), group=filter_group)
    remove_dc = Bool(True, group=filter_group)

class SampleProperties(ComponentBase):
    d = Int(1000)
    layers = Int(1)
    default_values = Bool(True)


class SavePlotsSettings(ComponentBase):
    path = Instance(Path, default_value=common.consts.result_dir)
    filetype = Unicode("pdf")
    suffix = Unicode("")
    bbox_inches = Unicode("tight")
    dpi = Int(300)
    pad_inches = Int(0)
    set_size_inches = Tuple(Float(), Float(), default_value=(12.0, 9.0))


class ShownPlots(ComponentBase):
    window = Bool(True)
    time_domain = Bool(True)
    spectrum = Bool(True)
    phase = Bool(True)
    phase_slope = Bool(False)
    amplitude_transmission = Bool(False)
    absorbance = Bool(False)
    refractive_index = Bool(False)
    absorption_coefficient = Bool(False)
    conductivity = Bool(False)


class PlotOpt(ComponentBase):
    plot_range = Tuple(Float(), Float(), default_value=(0.05, 3.5), metadata={"priority": 1, "readonly": False})
    shift_sam2ref = Bool(False)
    label = Unicode("")
    sub_noise_floor = Bool(False)
    td_scale = Int(1)
    remove_t_offset = Bool(False)
    err_bar_limits = TAny(None, allow_none=True)
    ref_err_bars = Bool(False)
    stability_plot_rel_change = Bool(False)
    subtract_mean = Bool(False)
    temp_sensor_idx = Int(-1)
    plot_zero_crossing = Bool(False)
    disable_legend = TList(Int(), default_value=[])
    clip_climate_data = Bool(False)
    redp_sensor_labels = TDict(
        key_trait=Unicode(),
        value_trait=Unicode(),
        default_value={
            "Redp idx 0": r"$\theta_{system}$",
            "Redp idx 1": r"$\theta_{air}$",
            "Redp idx 2": r"$\theta_{fiber}$",
            "Redp idx 3": r"$\theta_{box}$",
        }
    )


class AppSettings(ComponentBase):
    log_level = TEnum(LogLevel, default_value=LogLevel.info)
    result_dir = Instance(Path, default_value=common.consts.result_dir)
    save_plots = Bool(False)
    excluded_areas = TAny(None, allow_none=True)
    cbar_lim = Tuple(TAny(allow_none=True), TAny(allow_none=True), default_value=(None, None))
    log_scale = Bool(False)
    color_map = Unicode("autumn")
    invert_x = Bool(False)
    invert_y = Bool(False)
    pixel_interpolation = TEnum(PixelInterpolation, default_value=PixelInterpolation.none)
    fig_label = Unicode("")
    img_title = Unicode("")
    en_cbar_label = Bool(True)
    ref_pos = Tuple(TAny(allow_none=True), TAny(allow_none=True), default_value=(None, None))
    ref_threshold = Float(0.95)
    fix_ref = Bool(False)
    dist_func = TEnum(Dist, default_value=Dist.Time)

    save_plots_settings = Instance(SavePlotsSettings, args=())
    pp_opt = Instance(PpOpt, args=())
    sample_properties = Instance(SampleProperties, args=())
    eval_opt = Instance(EvalOpt, args=())
    plot_opt = Instance(PlotOpt, args=())

    enable_q_eval = Bool(False)
    only_shown_figures = TList(TAny(), default_value=[])
    shown_plots = Instance(ShownPlots, args=())