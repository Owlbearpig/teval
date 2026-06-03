from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional, List, Dict, Any
import common.consts
import logging
from pathlib import Path

from common.components import ComponentBase


class Domain(Enum):
    Time = 0
    Frequency = 1
    Both = 2


class MeasurementType(Enum):
    REF = 1
    SAM = 2

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

class WindowTypes(Enum):
    tukey = "tukey"
    hannin = "hanning"
    rectangular = "rectangular"

class Quantity:
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
    P2P = Quantity("Peak to peak")
    Power = Quantity("Power", domain=Domain.Frequency)
    Phase = Quantity("Phase", domain=Domain.Frequency, unit="rad")
    MeasTimeDeltaRef2Sam = Quantity("Time delta Ref. to Sam.")
    RefAmp = Quantity("Ref. Amp", domain=Domain.Frequency)
    RefArgmax = Quantity("Ref. Argmax")
    RefPhase = Quantity("Ref. Phase", domain=Domain.Frequency)
    PeakCnt = Quantity("Peak Cnt")
    ZeroCrossing = Quantity("Zero Crossing", domain=Domain.Time, unit="ps")
    TimeOfFlight = Quantity("Time of Flight", domain=Domain.Time, unit="ps")
    TransmissionAmp = Quantity("Amplitude transmission", domain=Domain.Frequency)
    TransmissionPhase = Quantity("Phase transmission", domain=Domain.Frequency, unit="rad")
    RefractiveIdx = Quantity("Refractive idx", domain=Domain.Frequency)
    AbsorptionCoe = Quantity("Absorption coe", domain=Domain.Frequency, unit="1/cm")


class LogLevel(Enum):
    info = logging.INFO
    debug = logging.DEBUG
    warning = logging.WARNING
    error = logging.ERROR
    critical = logging.CRITICAL

@dataclass
class WindowOpt(ComponentBase):
    enabled: bool = False
    win_width: Optional[int] = None
    win_start: Optional[int] = None
    shift: Optional[float] = None
    slope: float = 0.15
    en_plot: bool = False
    type: WindowTypes = WindowTypes.tukey

@dataclass
class FilterOpt(ComponentBase):
    enabled: bool = False
    f_range: Tuple[float, float] = (0.3, 3.0)

@dataclass
class EvalOpt(ComponentBase):
    dt: int = 0  # dt in fs
    sub_pnt: Tuple[float, float] = (0, 0)
    fit_range: Tuple[float, float] = (0.50, 2.20)
    q_space_range: Tuple[float, float] = (0.75, 2.00)
    phi_fit_range: Tuple[float, float] = (0.47, 1.05)
    average: bool = False
    delta_d: float = 2.0
    phi_offset_correction: bool = True
    printed_freqs: Optional[List[float]] = None
    d_opt_axis: Optional[Any] = None

@dataclass
class PpOpt(ComponentBase):
    window_opt: WindowOpt = field(default_factory=WindowOpt)
    filter_opt: FilterOpt = field(default_factory=FilterOpt)
    remove_dc: bool = True
    dt: int = 0

@dataclass
class SampleProperties(ComponentBase):
    d: int = 1000
    layers: int = 1
    default_values: bool = True

@dataclass
class SavePlotsSettings(ComponentBase):
    path: Path = common.consts.result_dir
    filetype: str = "pdf"
    suffix: str = ""
    bbox_inches: str = "tight"
    dpi: int = 300
    pad_inches: int = 0
    set_size_inches: Tuple[float, float] = (12, 9)

@dataclass
class ShownPlots(ComponentBase):
    window: bool = True
    time_domain: bool = True
    spectrum: bool = True
    phase: bool = True
    phase_slope: bool = False
    amplitude_transmission: bool = False
    absorbance: bool = False
    refractive_index: bool = False
    absorption_coefficient: bool = False
    conductivity: bool = False

@dataclass
class PlotOpt(ComponentBase):
    plot_range: Tuple[float, float] = field(default=(0.05, 3.5), metadata={"priority": 1, "readonly": False})
    shift_sam2ref: bool = False
    label: str = ""
    sub_noise_floor: bool = False
    td_scale: int = 1
    remove_t_offset: bool = False
    err_bar_limits: Optional[Any] = None
    ref_err_bars: bool = False
    stability_plot_rel_change: bool = False
    subtract_mean: bool = False
    temp_sensor_idx: int = -1
    plot_zero_crossing: bool = False
    disable_legend: List[int] = field(default_factory=list)
    clip_climate_data: bool = False
    redp_sensor_labels: Dict[str, str] = field(
        default_factory=lambda: {
            "Redp idx 0": r"$\theta_{system}$",
            "Redp idx 1": r"$\theta_{air}$",
            "Redp idx 2": r"$\theta_{fiber}$",
            "Redp idx 3": r"$\theta_{box}$",
        }
    )


@dataclass
class AppSettings(ComponentBase):
    log_level: LogLevel = LogLevel.info
    result_dir: Path = common.consts.result_dir
    save_plots: bool = False
    excluded_areas: Optional[Any] = None
    cbar_lim: Tuple[Optional[float], Optional[float]] = (None, None)
    log_scale: bool = False
    color_map: str = "autumn"
    invert_x: bool = False
    invert_y: bool = False
    pixel_interpolation: PixelInterpolation = PixelInterpolation.none
    fig_label: str = ""
    img_title: str = ""
    en_cbar_label: bool = True
    ref_pos: Tuple[Optional[float], Optional[float]] = (None, None)
    ref_threshold: float = 0.95
    fix_ref: bool = False
    dist_func: Dist = Dist.Time

    save_plots_settings: SavePlotsSettings = field(default_factory=SavePlotsSettings)
    pp_opt: PpOpt = field(default_factory=PpOpt)
    sample_properties: SampleProperties = field(default_factory=SampleProperties)
    eval_opt: EvalOpt = field(default_factory=EvalOpt)
    plot_opt: PlotOpt = field(default_factory=PlotOpt)

    enable_q_eval: bool = False
    only_shown_figures: List[Any] = field(default_factory=list)
    shown_plots: ShownPlots = field(default_factory=ShownPlots)
