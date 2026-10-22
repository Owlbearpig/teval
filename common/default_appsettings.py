import logging
from enum import Enum, member
from pathlib import Path
import traitlets
from traitlets import (Bool, Int, Float, Unicode, Dict as TDict, Enum as TEnum, Instance
                       )
from common.eval_component.shgo_settings import SHGOOptions
from common.components import ComponentBase
from common.traits import ValueRange, Path as TPath, Q_, Quantity as TQuantity

class QSpaceQuantity(Enum):
    n = "n"
    k = "k"
    alpha = "alpha"

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
    max_amp = "Maximum amplitude"
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

class PlotReferenceWaveform(Enum):
    always = "Always"
    only_first = "Only first"
    never = "Never"

class ComplexPart(Enum):
    abs = "Absolute value"
    real = "Real part"
    imag = "Imaginary part"
    phase = "Phase"

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

class DerivedQuantity:
    def __init__(self, label="label", domain=None, unit="", uses_range=False):
        self.label = label
        self.domain = Domain.Time if domain is None else domain
        self.unit = unit
        self.uses_range = uses_range

    def __repr__(self):
        return self.label

class QuantityEnum(Enum):
    P2P = DerivedQuantity("Peak to peak", domain=Domain.Time)
    PowerInt = DerivedQuantity("Integrated power", domain=Domain.Frequency, uses_range=True)
    Power = DerivedQuantity("Power", domain=Domain.Frequency)
    Absorbance = DerivedQuantity("Absorbance", domain=Domain.Frequency, unit="dB")
    Phase = DerivedQuantity("Phase", domain=Domain.Frequency, unit="rad")
    MeasTimeDeltaRef2Sam = DerivedQuantity("Time delta Ref. to Sam.", unit="s", domain=Domain.Time)
    RefAmp = DerivedQuantity("Ref. Amp", domain=Domain.Frequency)
    RefArgmax = DerivedQuantity("Ref. Argmax", domain=Domain.Time)
    RefPhase = DerivedQuantity("Ref. Phase", domain=Domain.Frequency, unit="rad")
    PeakCnt = DerivedQuantity("Peak Cnt", domain=Domain.Time)
    ZeroCrossing = DerivedQuantity("Zero Crossing", domain=Domain.Time, unit="ps")
    TimeOfFlight = DerivedQuantity("Time of Flight", domain=Domain.Time, unit="ps")
    PhaseToF = DerivedQuantity("Phase slope ToF", domain=Domain.Time, unit="ps")
    RIEstimate = DerivedQuantity("Refractive index estimate", domain=Domain.Time)
    Transmission = DerivedQuantity("Transmission", domain=Domain.Frequency)
    TransmissionAmp = DerivedQuantity("Amplitude transmission", domain=Domain.Frequency)
    TransmissionPhase = DerivedQuantity("Phase transmission", domain=Domain.Frequency, unit="rad")
    RefractiveIdx = DerivedQuantity("Refractive idx", domain=Domain.Frequency)
    ExtCoe = DerivedQuantity("Extinction coefficient", domain=Domain.Frequency)
    AbsorptionCoe = DerivedQuantity("Absorption coe.", domain=Domain.Frequency, unit="1/cm")
    Conductivity = DerivedQuantity("Conductivity", domain=Domain.Frequency, unit="S/cm")


class EvalOpt(ComponentBase):
    phi_fit_range = ValueRange([Q_(0.47, "THz"), Q_(1.05, "THz")]).tag(name="Phase correction fit range")
    avg_grp_delay_fit_range = ValueRange([Q_(0.25, "THz"), Q_(0.85, "THz")]).tag(name="Group delay fit range")
    spectral_similarity_range = ValueRange([Q_(0.15, "THz"), Q_(2.00, "THz")]).tag(name="Spectral similarity range")

    delta_d = TQuantity(Q_(2.0, "µm")).tag(name="Thickness uncertainty")
    auto_fp_count = Bool(True, help="Number of Fabry-Perot echoes inside the recorded time window, estimated from "
                                    "the trace length after the main pulse and the round trip time. Only used by "
                                    "the truncated echo models (model_1layer_fp, t_model_2layer)").tag(
        name="Use estimated number of Fabry-Perot echoes")
    fp_count = Int(25, min=-1, help="-1: infinite number of echoes").tag(
        name="Assumed number of Fabry-Perot echoes")
    phi_offset_correction = Bool(True).tag(name="Phase offset correction")

    d = TQuantity(Q_(0.0, "µm")).tag(name="Default nominal sample thickness")
    d_film = TQuantity(Q_(0.0, "µm")).tag(name="Film thickness")

    transmission_sim_grp = "Transmission simulation"
    add_sim_to_res = Bool(False, group=transmission_sim_grp).tag(name="Add simulated transmission to result")
    sim_d = TQuantity(Q_(100, "µm"), group=transmission_sim_grp).tag(name="Simulation thickness")
    sim_nfp = Int(8, group=transmission_sim_grp).tag(name="Fabry perot count")
    sim_h = TQuantity(Q_(1, "µm"), group=transmission_sim_grp).tag(name="Simulation film thickness")
    sim_shift = TQuantity(Q_(0, "fs"), group=transmission_sim_grp).tag(name="Added phase shift")
    sim_n_sub = ValueRange([1, 0],
                           group=transmission_sim_grp).tag(name="Refractive index sub. (real, imag)")
    sim_n_film = ValueRange([1, 0],
                           group=transmission_sim_grp).tag(name="Refractive index film (real, imag)")
    sim_n_selection = TEnum(SimRISelection, SimRISelection.const,
                            group=transmission_sim_grp).tag(name="Simulation refractive index")

    q_space_eval_grp = "Q-value calculation"
    normalize_q_vals = Bool(False, group=q_space_eval_grp).tag(name="Normalize q-values")
    use_separate_range = Bool(False, group=q_space_eval_grp).tag(name="Use separate range for calculation")
    q_space_range = ValueRange([Q_(0.75, "THz"), Q_(2.00, "THz")]).tag(name="Separate q-space minimization range",
                                                                       group=q_space_eval_grp)
    q_space_quantity = TEnum(QSpaceQuantity, QSpaceQuantity.n).tag(name="Quantity used for q-value calculation",
                                                                   group=q_space_eval_grp)
    auto_fp_time = Bool(True, group=q_space_eval_grp,
                        help="Estimate the Fabry-Perot round trip time 2 n_g d / c for each thickness from the "
                             "group delay of the measured transmission").tag(name="Estimate Fabry-Perot round trip time")
    fp_spacing = TQuantity(Q_(12, "ps"), group=q_space_eval_grp,
                           help="Used when the estimate is disabled; position of the Fabry-Perot peak in the "
                                "q-space spectrum").tag(name="Fabry-Perot round trip time")
    auto_q_space_f_max = Bool(True, group=q_space_eval_grp,
                              help="Limit the q-value range to frequencies where the estimated Fabry-Perot echo "
                                   "amplitude |r|^2 exp(-2 alpha d) is above the threshold, above that n and k "
                                   "contain no Fabry-Perot information").tag(name="Limit q-value range to visible echo")
    fp_amp_threshold = TQuantity(Q_(0.3, "%"), group=q_space_eval_grp).tag(name="Echo amplitude threshold")

    regulated_spline_grp = "Regulated spline optimization"
    knot_cnt = Int(100, min=2).tag(name="Spline count", group=regulated_spline_grp)
    auto_knot_cnt = Bool(True, group=regulated_spline_grp,
                         help="Set the spline count from the estimated Fabry-Perot period at the upper thickness "
                              "bound, the spline count above is then ignored").tag(name="Automatic spline count")
    knots_per_fp_period = Float(4.0, min=1.0, group=regulated_spline_grp).tag(name="Splines per Fabry-Perot period")
    reg_k = Float(1e-10).tag(name="k regularization", group=regulated_spline_grp, significant_figures=3)
    reg_n = Float(1e-10).tag(name="n regularization", group=regulated_spline_grp, significant_figures=3)
    max_nfev = Int(100, min=1).tag(name="Maximum number of function evaluations", group=regulated_spline_grp)
    per_freq_refine = Bool(True, group=regulated_spline_grp,
                           help="Solve n per frequency (Newton) starting from the spline fit, keeps the Fabry-Perot "
                                "residuals the q-value is based on").tag(name="Per frequency refinement")

    smoothing_avg_grp = "Smoothing average"
    smoothing_avg_en = Bool(False).tag(name="Enable smoothing average of n and k", group=smoothing_avg_grp)
    smoothing_avg_n = Int(3).tag(name="Smoothing average window size", group=smoothing_avg_grp)
    smoothing_avg_iters = Int(3).tag(name="Smoothing average iterations", group=smoothing_avg_grp)

    shift_grp = "Shift axis settings"
    enable_shift_opt = Bool(False).tag(group=shift_grp).tag(name="Enable pulse shift optimization")
    shift_opt_axis_bounds = ValueRange([Q_(0, "fs"), Q_(0, "fs", )],
                                       group=shift_grp).tag(name="Shift axis bounds")
    shift_opt_axis_step = TQuantity(Q_(1, "fs"), group=shift_grp).tag(name="Shift axis step")

class PpOpt(ComponentBase):
    remove_dc = Bool(True).tag(name="Subtract DC")
    normalize_data = Bool(False).tag(name="Normalize waveform")

    window_group = "Window options"
    window_enabled = Bool(False, group=window_group).tag(name="Enable window")
    win_width = TQuantity(Q_(10, "ps"), group=window_group).tag(name="Window width")
    win_start = TQuantity(Q_(0, "ps"), group=window_group).tag(name="Window start")
    center_on_pulse = Bool(True, group=window_group).tag(name="Center window on pulse")
    shift = TQuantity(Q_(0, "ps"), group=window_group).tag(name="Window shift")
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
    shift_sam2ref = Bool(False).tag(name="Shift sample pulse to ref")
    sub_noise_floor = Bool(False).tag(name="Subtract spectrum noise floor")
    td_scale = Float(1.0).tag(name="Scale waveform")
    remove_t_offset = Bool(False).tag(name="Start t-axis at 0 ps")
    fig_num_ext = Unicode("").tag(name="Figure number extension")
    plot_zero_crossing = Bool(False).tag(name="Plot zero crossing")
    add_mean_to_quant_plot = Bool(True).tag(name="Add mean value to quantity plot")
    auto_show = Bool(True).tag(name="Automatically show plots")
    plot_standard_deviation = Bool(True).tag(name="Plot standard deviation")
    plot_ref_waveform = TEnum(PlotReferenceWaveform, PlotReferenceWaveform.only_first).tag(
        name="Plot reference with sample")

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

class ComponentToggle(ComponentBase):
    image_plot_grp = "Image plot"
    en_image_plot = Bool(True).tag(name="Main component", group=image_plot_grp, priority=1000)
    en_avg_val_rect = Bool(True).tag(name="Average value rectangle", group=image_plot_grp, priority=1001)
    en_slice_plots = Bool(True).tag(name="Slice plots", group=image_plot_grp, priority=1002)

    dataset_plotter_grp = "Dataset plots"
    en_extended_dataset_plots = Bool(True).tag(name="Main component", group=dataset_plotter_grp, priority=1000)
    en_stability_plots = Bool(True).tag(name="Climate plots", group=dataset_plotter_grp, priority=1001)

    measurement_selection = "Measurement selection"
    en_queue = Bool(True).tag(name="Queue", group=measurement_selection)

    info_pane = "DataSet info pane"
    en_info_pane = Bool(True).tag(name="Info pane", group=info_pane)

    q_space_eval = "Q-Space evaluation"
    en_q_space_eval = Bool(True).tag(name="Main component", group=q_space_eval, priority=1000)
    en_preprocessing = Bool(True).tag(name="Preprocessing view", group=q_space_eval, priority=1001)

    regression = "Regression"
    en_regression = Bool(True).tag(name="Main component", group=regression, priority=1000)

    conductivity_eval = "Conductivity evaluation"
    en_conductivity_eval = Bool(True).tag(name="Main component", group=conductivity_eval, priority=1000)

    sparse_deconvolution = "Sparse deconvolution"
    en_sparse_deconvolution = Bool(True).tag(name="Main component", group=sparse_deconvolution, priority=1000)

class AppSettings(ComponentBase):
    log_level = TEnum(LogLevel, default_value=LogLevel.info).tag(name="Log level")

    save_settings = Instance(SaveSettings, args=())
    pp_opt = Instance(PpOpt, args=())
    eval_opt = Instance(EvalOpt, args=())
    plot_opt = Instance(PlotOpt, args=())
    shgo_options = Instance(SHGOOptions, args=())
    component_toggle = Instance(ComponentToggle, args=())

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.shgo_options = SHGOOptions()

if __name__ == '__main__':
    settings = AppSettings()
    print([getattr(settings.eval_opt, k) for k in settings.eval_opt.traits()])
