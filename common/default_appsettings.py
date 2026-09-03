import logging
from enum import Enum, member
from pathlib import Path
import traitlets
from traitlets import (Bool, Int, Float, Unicode, Tuple,
    List as TList, Dict as TDict, Enum as TEnum, Instance, Any as TAny
)
from common.eval_component.shgo_settings import SHGOOptions
from common.components import ComponentBase
from common.traits import ValueRange, Path as TPath, Q_, Quantity as TQuantity

class SimRISelection(Enum):
    const = 0

class LogLevel(Enum):
    info = logging.INFO
    debug = logging.DEBUG
    warning = logging.WARNING
    error = logging.ERROR
    critical = logging.CRITICAL


class ReferenceClassification(Enum):
    from_file_name = "From file name"
    horizontal_line_as_ref = "Horizontal line"
    vertical_line_as_ref = "Vertical line"
    above_threshold = "Above threshold"
    single_point = "Single point"


class WindowTypes(Enum):
    tukey = "tukey"
    chebwin = "chebwin"
    dpss = "dpss"
    gaussian = "gaussian"
    general_cosine = "general_cosine"
    general_gaussian = "general_gaussian"
    general_hamming = "general_hamming"
    kaiser = "kaiser"
    kaiser_bessel_derived = "kaiser_bessel_derived"
    taylor = "taylor"
    hannin = "hanning"
    rectangular = "rectangular"
    barthann = "barthann"
    bartlett = "bartlett"
    blackman = "blackman"
    blackmanharris = "blackmanharris"
    bohman = "bohman"
    boxcar = "boxcar"
    cosine = "cosine"
    exponential = "exponential"
    flattop = "flattop"
    hamming = "hamming"
    hann = "hann"
    lanczos = "lanczos"
    nuttall = "nuttall"
    parzen = "parzen"
    triang = "triang"

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

class Dist(Enum):
    Position = member(lambda meas1, meas2: (abs(meas1.position[0] - meas2.position[0]) +
                                     abs(meas1.position[1] - meas2.position[1])))
    Time = member(lambda meas1, meas2: (meas1.meas_time - meas2.meas_time).total_seconds())

class Direction(Enum):
    Horizontal = 0
    Vertical = 1

class Filetype(Enum):
    pdf = ".pdf"
    png = ".png"
    jpg = ".jpg"

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
    def __init__(self, label="label", func=None, domain=None, unit=""):
        self.label = label
        self.domain = Domain.Time if domain is None else domain
        self.func = func if func is not None else lambda x: x
        self.unit = unit

    def __repr__(self):
        return self.label

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class QuantityEnum(Enum):
    P2P = QuantityFunc("Peak to peak", domain=Domain.Time)
    PowerInt = QuantityFunc("Integrated power", domain=Domain.Frequency)
    Power = QuantityFunc("Power", domain=Domain.Frequency)
    Absorbance = QuantityFunc("Absorbance", domain=Domain.Frequency, unit="dB")
    Phase = QuantityFunc("Phase", domain=Domain.Frequency, unit="rad")
    MeasTimeDeltaRef2Sam = QuantityFunc("Time delta Ref. to Sam.", domain=Domain.Time)
    RefAmp = QuantityFunc("Ref. Amp", domain=Domain.Frequency)
    RefArgmax = QuantityFunc("Ref. Argmax", domain=Domain.Time)
    RefPhase = QuantityFunc("Ref. Phase", domain=Domain.Frequency, unit="rad")
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
    dt = TQuantity(Q_(0.0, "fs")).tag(name="Conductivity pulse shift")
    fit_range = ValueRange([Q_(0.50, "THz"), Q_(2.20, "THz")]).tag(name="Fit range")
    q_space_range = ValueRange([Q_(0.75, "THz"), Q_(2.00, "THz")]).tag(name="Q-space minimization range")
    phi_fit_range = ValueRange([Q_(0.47, "THz"), Q_(1.05, "THz")]).tag(name="Phase correction fit range")
    average = Bool(False, help="Average over consecutive measurements "
                               "with same position").tag(name="Average measurements")
    delta_d = TQuantity(Q_(2.0, "µm")).tag(name="Thickness uncertainty")
    fp_count = Int(0).tag(name="Number of Fabry-Perots")
    phi_offset_correction = Bool(True).tag(name="Phase offset correction")
    printed_freqs = Unicode(default_value="1.0, 2.0").tag(name="Printed frequencies (THz)")

    d = TQuantity(Q_(0.0, "µm")).tag(name="Sample thickness")
    d_film = TQuantity(Q_(0.0, "µm")).tag(name="Film thickness")
    fp_spacing = TQuantity(Q_(12, "fs")).tag(name="Approximate Fabry-perot spacing")

    transmission_sim_grp = "Transmission simulation"
    sim_d = TQuantity(Q_(100, "µm"), group=transmission_sim_grp).tag(name="Simulation thickness")
    sim_h = TQuantity(Q_(1, "µm"), group=transmission_sim_grp).tag(name="Simulation film thickness")
    sim_nfp = Int(8, group=transmission_sim_grp).tag(name="Fabry perot count")
    sim_shift = TQuantity(Q_(0, "fs"), group=transmission_sim_grp).tag(name="Added phase shift")
    sim_n_sub = ValueRange([1, 0],
                           group=transmission_sim_grp).tag(name="Refractive index sub. (real, imag)")
    sim_n_film = ValueRange([1, 0],
                           group=transmission_sim_grp).tag(name="Refractive index film (real, imag)")
    sim_n_selection = TEnum(SimRISelection, SimRISelection.const,
                            group=transmission_sim_grp).tag(name="Simulation refractive index")

    conductivity_calc_grp = "Conductivity calculation"
    use_sub_dataset = Bool(False, group=conductivity_calc_grp).tag(name="Use separate substrate dataset")
    sub_pnt = ValueRange([0, 0], group=conductivity_calc_grp).tag(name="Substrate point")

class PpOpt(ComponentBase):
    remove_dc = Bool(True).tag(name="Subtract DC")
    normalize_data = Bool(False).tag(name="Normalize waveform")

    window_group = "Window options"
    window_enabled = Bool(False, group=window_group, name="Enabled").tag(name="Enable window")
    win_width = Int(10, group=window_group).tag(name="Window width").tag(name="Window width")
    shift = Float(0, group=window_group).tag(name="Window shift").tag(name="Window shift")
    en_plot = Bool(False, group=window_group).tag(name="Enable window plot")
    type = TEnum(WindowTypes, default_value=WindowTypes.tukey, group=window_group).tag(name="Window type")
    symmetric = Bool(True, group=window_group).tag(name="Symmetric window")

    #specific window parameters
    window_param_group = "Window parameters"
    tukey_alpha = Float(0.50, group=window_param_group).tag(name="Tukey slope", priority=1999)
    chebwin_at = Float(100.0, group=window_param_group).tag(name="Chebwin attenuation", priority=2000)
    dpss_nw = Float(2.50, group=window_param_group).tag(name="Standardized half bandwidth", priority=2000)
    exp_center = Float(0.0, group=window_param_group).tag(name="Exponential center", priority=2000)
    exp_tau = Float(1.00, group=window_param_group).tag(name="Exponential decay tau", priority=2000)
    gaussian_std = Float(0.0, group=window_param_group).tag(name="Gaussian standard deviation", priority=2000)
    general_gauss_p = Float(1.0, group=window_param_group).tag(name="Gen. Gaussian shape parameter", priority=2000)
    general_gauss_sig = Float(10.0, group=window_param_group).tag(name="Gen. Gaussian sigma", priority=2000)
    general_hamming_alpha = Float(0.54, group=window_param_group).tag(name="Gen. Hamming alpha", priority=2000)
    kaiser_beta = Float(14.0, group=window_param_group).tag(name="Kaiser beta", priority=2000)
    kaiser_bessel_beta = Float(4.0, group=window_param_group).tag(name="Kaiser-Bessel beta", priority=2000)
    taylor_nbar = Int(4, group=window_param_group).tag(name="Taylor nbar", priority=2000)
    taylor_sll = Float(30, group=window_param_group).tag(name="Taylor sll", priority=2000)
    taylor_norm = Bool(True, group=window_param_group).tag(name="Taylor normalize", priority=2000)

    filter_group = "Frequency filter options"
    filter_enabled = Bool(False, group=filter_group, name="Enabled").tag(name="Enable filter")
    f_range = ValueRange([Q_(0.35, "THz"), Q_(3.0, "THz")], group=filter_group).tag(name="Filter range")


class SaveSettings(ComponentBase):
    path = TPath(Path(""), is_file=False).tag(name="Figure save directory")
    filetype = TEnum(Filetype, Filetype.jpg).tag(name="File type")
    suffix = Unicode("").tag(name="Filename suffix")
    bbox_inches = Unicode("tight", help="Should be set to tight").tag(name="Bounding box")
    dpi = Int(300).tag(name="DPI")
    pad_inches = Int(0).tag(name="Pad inches")
    set_width_size_inches = Float(12.0).tag(name="Set width size inches")
    set_height_size_inches = Float(9.0).tag(name="Set height size inches")
    save_plots = Bool(False).tag(name="Enable saving of plots")
    only_save_plots = Bool(False).tag(name="Hide plots (only save if enabled)")

class PlotOpt(ComponentBase):
    plot_range = ValueRange([Q_(0.05, "THz"), Q_(3.5, "THz")],
                            metadata={"priority": 1, "readonly": False}).tag(name="Plot range")

    climate_group = "Stability and climate"
    stability_plot_rel_change = Bool(False, group=climate_group).tag(name="Convert to percent")
    subtract_mean = Bool(False, group=climate_group).tag(name="Subtract mean value")
    temp_sensor_idx = Int(-1, group=climate_group, help="-1 selects all sensors").tag(name="Select sensor index")
    climate_file = TPath(Path(), group=climate_group).tag(name="Climate log file")
    clip_climate_data = Bool(False, group=climate_group, help="Clip climate data to THz measurement",
                             ).tag(name="Clip climate data")
    climate_quantity = TEnum(ClimateQuantity, ClimateQuantity.Temperature,
                             group=climate_group).tag(name="Climate quantity")
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

    shift_sam2ref = Bool(False).tag(name="Shift sample pulse to ref")
    sub_noise_floor = Bool(False).tag(name="Subtract spectrum noise floor")
    td_scale = Float(1.0).tag(name="Scale waveform")
    remove_t_offset = Bool(False).tag(name="Start t-axis at 0 ps")
    fig_num_ext = Unicode("").tag(name="Figure number extension")

    plot_zero_crossing = Bool(False).tag(name="Plot zero crossing")

    image_group = "Image"
    cbar_lim = ValueRange(default_value=[0.0, 0.0], group=image_group).tag(name="Custom color bar limits")
    log_scale = Bool(False, group=image_group).tag(name="Log scale")
    color_map = traitlets.Enum(ColorMaps, default_value=ColorMaps.autumn).tag(name="Colormaps", group=image_group)
    invert_x = Bool(False, group=image_group).tag(name="Invert x")
    invert_y = Bool(False, group=image_group).tag(name="Invert y")
    pixel_interpolation = TEnum(PixelInterpolation, default_value=PixelInterpolation.none,
                                group=image_group).tag(name="Pixel interpolation")
    img_fig_num_ext = Unicode("", group=image_group).tag(name="Figure number extension")
    img_title = Unicode("", group=image_group).tag(name="Image title")
    en_cbar_label = Bool(True, group=image_group).tag(name="Enable color bar label")
    en_cbar_lim = Bool(default_value=False, group=image_group).tag(name="Enable custom color bar limits")

class AppSettings(ComponentBase):
    log_level = TEnum(LogLevel, default_value=LogLevel.info).tag(name="Log level")

    save_settings = Instance(SaveSettings, args=())
    pp_opt = Instance(PpOpt, args=())
    eval_opt = Instance(EvalOpt, args=())
    plot_opt = Instance(PlotOpt, args=())
    shgo_options = Instance(SHGOOptions, args=())

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.shgo_options = SHGOOptions()

if __name__ == '__main__':
    settings = AppSettings()

    print([getattr(settings.eval_opt, k) for k in settings.eval_opt.traits()])
