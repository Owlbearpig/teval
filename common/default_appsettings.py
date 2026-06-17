import logging
from enum import Enum, member
from pathlib import Path
import traitlets
from traitlets import (
    HasTraits, Bool, Int, Float, Unicode, Tuple,
    List as TList, Dict as TDict, Enum as TEnum, Instance, Any as TAny
)
import common.consts
from common import traits
from common.components import ComponentBase
from common.traits import ValueRange, Path as TPath, Q_, Quantity as TQuantity


class LogLevel(Enum):
    info = logging.INFO
    debug = logging.DEBUG
    warning = logging.WARNING
    error = logging.ERROR
    critical = logging.CRITICAL


class ReferenceSelection(Enum):
    from_file_name = 0
    point_as_ref = 1
    horizontal_line_as_ref = 2
    vertical_line_as_ref = 3
    above_threshold = 4

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

class Domain(Enum):
    Time = 0
    Frequency = 1
    Both = 2

class Dist(Enum):
    Position = member(lambda meas1, meas2: (abs(meas1.position[0] - meas2.position[0]) +
                                     abs(meas1.position[1] - meas2.position[1])))
    Time = member(lambda meas1, meas2: (meas1.meas_time - meas2.meas_time).total_seconds())

class Direction(Enum):
    Horizontal = 0
    Vertical = 1


class WindowTypes(Enum):
    tukey = "tukey"
    hannin = "hanning"
    rectangular = "rectangular"

class ColorMaps(Enum):
    magma = "magma"
    inferno = "inferno"
    plasma = "plasma"
    viridis = "viridis"
    cividis = "cividis"
    twilight = "twilight"
    twilight_shifted = "twilight_shifted"
    turbo = "turbo"
    berlin = "berlin"
    managua = "managua"
    vanimo = "vanimo"
    Blues = "Blues"
    BrBG = "BrBG"
    BuGn = "BuGn"
    BuPu = "BuPu"
    CMRmap = "CMRmap"
    GnBu = "GnBu"
    Greens = "Greens"
    Greys = "Greys"
    OrRd = "OrRd"
    Oranges = "Oranges"
    PRGn = "PRGn"
    PiYG = "PiYG"
    PuBu = "PuBu"
    PuBuGn = "PuBuGn"
    PuOr = "PuOr"
    PuRd = "PuRd"
    Purples = "Purples"
    RdBu = "RdBu"
    RdGy = "RdGy"
    RdPu = "RdPu"
    RdYlBu = "RdYlBu"
    RdYlGn = "RdYlGn"
    Reds = "Reds"
    Spectral = "Spectral"
    Wistia = "Wistia"
    YlGn = "YlGn"
    YlGnBu = "YlGnBu"
    YlOrBr = "YlOrBr"
    YlOrRd = "YlOrRd"
    afmhot = "afmhot"
    autumn = "autumn"
    binary = "binary"
    bone = "bone"
    brg = "brg"
    bwr = "bwr"
    cool = "cool"
    coolwarm = "coolwarm"
    copper = "copper"
    cubehelix = "cubehelix"
    flag = "flag"
    gist_earth = "gist_earth"
    gist_gray = "gist_gray"
    gist_heat = "gist_heat"
    gist_ncar = "gist_ncar"
    gist_rainbow = "gist_rainbow"
    gist_stern = "gist_stern"
    gist_yarg = "gist_yarg"
    gnuplot = "gnuplot"
    gnuplot2 = "gnuplot2"
    gray = "gray"
    hot = "hot"
    hsv = "hsv"
    jet = "jet"
    nipy_spectral = "nipy_spectral"
    ocean = "ocean"
    pink = "pink"
    prism = "prism"
    rainbow = "rainbow"
    seismic = "seismic"
    spring = "spring"
    summer = "summer"
    terrain = "terrain"
    winter = "winter"
    Accent = "Accent"
    Dark2 = "Dark2"
    Paired = "Paired"
    Pastel1 = "Pastel1"
    Pastel2 = "Pastel2"
    Set1 = "Set1"
    Set2 = "Set2"
    Set3 = "Set3"
    tab10 = "tab10"
    tab20 = "tab20"
    tab20b = "tab20b"
    tab20c = "tab20c"
    grey = "grey"
    gist_grey = "gist_grey"
    gist_yerg = "gist_yerg"
    Grays = "Grays"
    magma_r = "magma_r"
    inferno_r = "inferno_r"
    plasma_r = "plasma_r"
    viridis_r = "viridis_r"
    cividis_r = "cividis_r"
    twilight_r = "twilight_r"
    twilight_shifted_r = "twilight_shifted_r"
    turbo_r = "turbo_r"
    berlin_r = "berlin_r"
    managua_r = "managua_r"
    vanimo_r = "vanimo_r"
    Blues_r = "Blues_r"
    BrBG_r = "BrBG_r"
    BuGn_r = "BuGn_r"
    BuPu_r = "BuPu_r"
    CMRmap_r = "CMRmap_r"
    GnBu_r = "GnBu_r"
    Greens_r = "Greens_r"
    Greys_r = "Greys_r"
    OrRd_r = "OrRd_r"
    Oranges_r = "Oranges_r"
    PRGn_r = "PRGn_r"
    PiYG_r = "PiYG_r"
    PuBu_r = "PuBu_r"
    PuBuGn_r = "PuBuGn_r"
    PuOr_r = "PuOr_r"
    PuRd_r = "PuRd_r"
    Purples_r = "Purples_r"
    RdBu_r = "RdBu_r"
    RdGy_r = "RdGy_r"
    RdPu_r = "RdPu_r"
    RdYlBu_r = "RdYlBu_r"
    RdYlGn_r = "RdYlGn_r"
    Reds_r = "Reds_r"
    Spectral_r = "Spectral_r"
    Wistia_r = "Wistia_r"
    YlGn_r = "YlGn_r"
    YlGnBu_r = "YlGnBu_r"
    YlOrBr_r = "YlOrBr_r"
    YlOrRd_r = "YlOrRd_r"
    afmhot_r = "afmhot_r"
    autumn_r = "autumn_r"
    binary_r = "binary_r"
    bone_r = "bone_r"
    brg_r = "brg_r"
    bwr_r = "bwr_r"
    cool_r = "cool_r"
    coolwarm_r = "coolwarm_r"
    copper_r = "copper_r"
    cubehelix_r = "cubehelix_r"
    flag_r = "flag_r"
    gist_earth_r = "gist_earth_r"
    gist_gray_r = "gist_gray_r"
    gist_heat_r = "gist_heat_r"
    gist_ncar_r = "gist_ncar_r"
    gist_rainbow_r = "gist_rainbow_r"
    gist_stern_r = "gist_stern_r"
    gist_yarg_r = "gist_yarg_r"
    gnuplot_r = "gnuplot_r"
    gnuplot2_r = "gnuplot2_r"
    gray_r = "gray_r"
    hot_r = "hot_r"
    hsv_r = "hsv_r"
    jet_r = "jet_r"
    nipy_spectral_r = "nipy_spectral_r"
    ocean_r = "ocean_r"
    pink_r = "pink_r"
    prism_r = "prism_r"
    rainbow_r = "rainbow_r"
    seismic_r = "seismic_r"
    spring_r = "spring_r"
    summer_r = "summer_r"
    terrain_r = "terrain_r"
    winter_r = "winter_r"
    Accent_r = "Accent_r"
    Dark2_r = "Dark2_r"
    Paired_r = "Paired_r"
    Pastel1_r = "Pastel1_r"
    Pastel2_r = "Pastel2_r"
    Set1_r = "Set1_r"
    Set2_r = "Set2_r"
    Set3_r = "Set3_r"
    tab10_r = "tab10_r"
    tab20_r = "tab20_r"
    tab20b_r = "tab20b_r"
    tab20c_r = "tab20c_r"
    grey_r = "grey_r"
    gist_grey_r = "gist_grey_r"
    gist_yerg_r = "gist_yerg_r"
    Grays_r = "Grays_r"

class QuantityFunc:
    func = None

    def __init__(self, label="label", func=None, domain=None, unit=""):
        self.label = label
        self.domain = Domain.Time if domain is None else domain
        self.func = func if func is not None else lambda x: x

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
    Transmission = QuantityFunc("Transmission", domain=Domain.Frequency)
    TransmissionAmp = QuantityFunc("Amplitude transmission", domain=Domain.Frequency)
    TransmissionPhase = QuantityFunc("Phase transmission", domain=Domain.Frequency, unit="rad")
    RefractiveIdx = QuantityFunc("Refractive idx", domain=Domain.Frequency)
    AbsorptionCoe = QuantityFunc("Absorption coe", domain=Domain.Frequency, unit="1/cm")
    Conductivity = QuantityFunc("Conductivity", domain=Domain.Frequency, unit="S/cm")


class EvalOpt(ComponentBase):
    fit_quantity = TEnum(QuantityEnum, default_value=QuantityEnum.TransmissionAmp).tag(name="Fit quantity")
    dt = TQuantity(Q_(0.0, "fs"))
    fit_range = ValueRange([Q_(0.50, "THz"), Q_(2.20, "THz")])
    q_space_range = ValueRange([Q_(0.75, "THz"), Q_(2.00, "THz")])
    phi_fit_range = ValueRange([Q_(0.47, "THz"), Q_(1.05, "THz")])
    average = Bool(False)
    delta_d = Float(2.0)
    fp_count = Int(0).tag(name="Number of Fabry-Perots")
    phi_offset_correction = Bool(True)
    printed_freqs = TList(trait=Float(), default_value=[1.000, 2.000])
    d_opt_axis = TAny(None, allow_none=True)

    use_sub_dataset = Bool(False, group="Conductivity").tag(name="Use substrate dataset")
    sub_pnt = ValueRange([0, 0], group="Conductivity").tag(name="Substrate point")


class PpOpt(ComponentBase):
    window_group = "Window options"
    window_enabled = Bool(False, group=window_group, name="Enabled")
    win_width = Int(10, group=window_group)
    shift = Float(0, group=window_group).tag(name="Window shift")
    slope = Float(0.15, group=window_group)
    en_plot = Bool(False, group=window_group)
    type = TEnum(WindowTypes, default_value=WindowTypes.tukey, group=window_group)

    filter_group = "Filter options"
    filter_enabled = Bool(False, group=filter_group, name="Enabled")
    f_range = Tuple(Float(), Float(), default_value=(0.3, 3.0), group=filter_group)
    remove_dc = Bool(True, group=filter_group)

class SampleProperties(ComponentBase):
    d = TQuantity(Q_(0.0, "µm"))
    d_film = TQuantity(Q_(0.0, "µm")).tag(name="Film thickness")
    layers = Int(1)
    default_values = Bool(True, read_only=True)


class SaveSettings(ComponentBase):
    path = TPath(default_value=common.consts.result_dir)
    filetype = Unicode("pdf")
    suffix = Unicode("")
    bbox_inches = Unicode("tight")
    dpi = Int(300)
    pad_inches = Int(0)
    set_size_inches = TList([12.0, 9.0])
    save_plots = Bool(False)
    en_csv_export = Bool(False)

class PlotOpt(ComponentBase):
    plot_range = ValueRange([Q_(0.05, "THz"), Q_(3.5, "THz")], metadata={"priority": 1, "readonly": False})

    climate_group = "Stability and climate"
    stability_plot_rel_change = Bool(False, group=climate_group)
    subtract_mean = Bool(False, group=climate_group)
    temp_sensor_idx = Int(-1, group=climate_group)
    climate_file = TPath(Path(), group=climate_group)
    clip_climate_data = Bool(False, group=climate_group)
    redp_sensor_labels = TDict(
        key_trait=Unicode(),
        value_trait=Unicode(),
        default_value={
            "Redp idx 0": r"$\theta_{system}$",
            "Redp idx 1": r"$\theta_{air}$",
            "Redp idx 2": r"$\theta_{fiber}$",
            "Redp idx 3": r"$\theta_{box}$",
        },
        group=climate_group
    )

    shift_sam2ref = Bool(False)
    label = Unicode("")
    sub_noise_floor = Bool(False)
    td_scale = Float(1.0)
    remove_t_offset = Bool(False)
    err_bar_limits = ValueRange([90, 110])
    ref_err_bars = Bool(False)
    fig_num_ext = Unicode("")

    plot_zero_crossing = Bool(False)
    disable_legend = TList(Int(), default_value=[])

    image_group = "Image"
    excluded_areas = TAny(None, allow_none=True, group=image_group)
    cbar_lim = ValueRange(default_value=[0, 0], group=image_group)
    log_scale = Bool(False, group=image_group)
    color_map = traitlets.Enum(ColorMaps, default_value=ColorMaps.autumn).tag(name="Colormaps", group=image_group)
    invert_x = Bool(False, group=image_group)
    invert_y = Bool(False, group=image_group)
    pixel_interpolation = TEnum(PixelInterpolation, default_value=PixelInterpolation.none, group=image_group)
    fig_label = Unicode("", group=image_group)
    img_title = Unicode("", group=image_group)
    en_cbar_label = Bool(True, group=image_group)

    shown_plots_group = "Shown plots"
    window = Bool(True, group=shown_plots_group)
    time_domain = Bool(True, group=shown_plots_group)
    spectrum = Bool(True, group=shown_plots_group)
    phase = Bool(True, group=shown_plots_group)
    phase_slope = Bool(False, group=shown_plots_group)
    amplitude_transmission = Bool(False, group=shown_plots_group)
    absorbance = Bool(False, group=shown_plots_group)
    refractive_index = Bool(False, group=shown_plots_group)
    absorption_coefficient = Bool(False, group=shown_plots_group)
    conductivity = Bool(False, group=shown_plots_group)

    only_shown_figures = TList(TAny(), default_value=[])

class DatasetOpt(ComponentBase):
    dist_func = TEnum(Dist, default_value=Dist.Time)

    reference_filter_group = "Reference filter"
    ref_selection = TEnum(ReferenceSelection,
                          default_value=ReferenceSelection.point_as_ref,
                          group=reference_filter_group)
    ref_pos = ValueRange([0, 0], group=reference_filter_group)
    fix_ref = Bool(False, group=reference_filter_group)
    ref_threshold = Float(0.95, group=reference_filter_group, min=0, max=1)

class AppSettings(ComponentBase):
    log_level = TEnum(LogLevel, default_value=LogLevel.info)
    result_dir = TPath(default_value=common.consts.result_dir)

    save_settings = Instance(SaveSettings, args=())
    pp_opt = Instance(PpOpt, args=())
    sample_properties = Instance(SampleProperties, args=())
    eval_opt = Instance(EvalOpt, args=())
    plot_opt = Instance(PlotOpt, args=())
    dataset_opt = Instance(DatasetOpt, args=())

    enable_q_eval = Bool(False)
