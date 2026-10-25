import logging
from enum import Enum, member
from pathlib import Path
import traitlets
from traitlets import (Bool, Int, Float, Unicode, Dict as TDict, Enum as TEnum, Instance
                       )
from common.eval_component.shgo_settings import SHGOOptions
from common.components import ComponentBase
from common.remote.options import RemoteOpt
from common.traits import ValueRange, Path as TPath, Q_, Quantity as TQuantity

class QSpaceQuantity(Enum):
    n = "n"
    k = "k"
    alpha = "alpha"

class TrustRegionSolver(Enum):
    # tr_solver of scipy.optimize.least_squares, member names are passed as is
    exact = "Exact (SVD)"
    lsmr = "Iterative (LSMR)"


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
    QSpaceThickness = DerivedQuantity("Q-space thickness", domain=Domain.Time, unit="µm")
    QSpaceShift = DerivedQuantity("Q-space time shift", domain=Domain.Time, unit="fs")
    QSpaceQVal = DerivedQuantity("Q-space q-value", domain=Domain.Time)
    QSpaceRefractiveIdx = DerivedQuantity("Q-space refractive index", domain=Domain.Frequency)
    QSpaceAbsorptionCoe = DerivedQuantity("Q-space absorption coe.", domain=Domain.Frequency, unit="1/cm")
    QSpaceSampleMask = DerivedQuantity("Q-space sample mask", domain=Domain.Time)
    QSpaceReliable = DerivedQuantity("Q-space reliable", domain=Domain.Time)
    QSpaceMinWidth = DerivedQuantity("Q-space minimum width", domain=Domain.Time, unit="µm")
    QSpaceQRatio = DerivedQuantity("Q-space q-value / neighbours", domain=Domain.Time)
    QSpaceDDeviation = DerivedQuantity("Q-space thickness - neighbours", domain=Domain.Time, unit="µm")
    QSpaceFullScan = DerivedQuantity("Q-space full scan", domain=Domain.Time)
    QSpaceBandRefractiveIdx = DerivedQuantity("Q-space refractive index (band mean)", domain=Domain.Time)
    QSpaceBandAbsorptionCoe = DerivedQuantity("Q-space absorption coe. (band mean)", domain=Domain.Time,
                                              unit="1/cm")
    QSpaceThicknessUncertainty = DerivedQuantity("Q-space thickness uncertainty", domain=Domain.Time, unit="µm")


class QValueOpt(ComponentBase):
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
    remove_q_spikes = Bool(False, group=q_space_eval_grp,
                           help="Replace isolated single-frequency spikes (e.g. a per-frequency solution on another "
                                "branch, or noise) in the quantity by the mean of their neighbours before the "
                                "q-value is calculated. The extracted n and k are not changed").tag(
        name="Remove single-frequency spikes")


class SplineFitOpt(ComponentBase):
    regulated_spline_grp = "Regulated spline optimization"
    knot_cnt = Int(100, min=2).tag(name="Spline count", group=regulated_spline_grp)
    auto_knot_cnt = Bool(True, group=regulated_spline_grp,
                         help="Set the spline count from the estimated Fabry-Perot period at the upper thickness "
                              "bound, the spline count above is then ignored").tag(name="Automatic spline count")
    knots_per_fp_period = Float(4.0, min=1.0, group=regulated_spline_grp).tag(name="Splines per Fabry-Perot period")
    reg_k = Float(1e-10).tag(name="k regularization", group=regulated_spline_grp, significant_figures=3)
    reg_n = Float(1e-10).tag(name="n regularization", group=regulated_spline_grp, significant_figures=3)
    max_nfev = Int(100, min=1).tag(name="Maximum number of function evaluations", group=regulated_spline_grp)
    tr_solver = TEnum(TrustRegionSolver, default_value=TrustRegionSolver.exact, group=regulated_spline_grp,
                      help="Solver of the trust region subproblem in each spline fit step. Exact: SVD of the "
                           "Jacobian, the largest part of a fit. Iterative: LSMR, cheaper per step. With the per "
                           "frequency refinement both give the same thickness and q-value").tag(
        name="Trust region solver")
    use_compiled_kernels = Bool(True, group=regulated_spline_grp,
                                help="Compiled (numba) model, initial guess scan and per frequency refinement for "
                                     "the single layer model, same results as the numpy code. Other models and a "
                                     "missing numba installation use the numpy code").tag(
        name="Use compiled kernels (numba)")
    per_freq_refine = Bool(True, group=regulated_spline_grp,
                           help="Solve n per frequency (Newton) starting from the spline fit, keeps the Fabry-Perot "
                                "residuals the q-value is based on").tag(name="Per frequency refinement")


class TransmissionSimOpt(ComponentBase):
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


class ImageEvalOpt(ComponentBase):
    # tuning of the q-space image evaluation (QSpaceImageEval), rarely changed per image
    nbr_grp = "Neighbour guided scan"
    seed_stride = Int(4, min=1, group=nbr_grp,
                      help="Every n-th pixel in x and y gets the full thickness scan (settings of the q-space "
                           "evaluation), then the spacing is halved and each new pixel only scans the window "
                           "around its evaluated neighbours. 1: full scan for every pixel").tag(
        name="Seed pixel spacing", priority=0)
    nbr_d_window = TQuantity(Q_(10.0, "µm"), group=nbr_grp,
                             help="Half width of the scanned thickness window around the median thickness of the "
                                  "evaluated neighbours, should cover the pixel to pixel scatter of the thickness. "
                                  "If the optimum ends up at the edge of the window, the window is moved there").tag(
        name="Thickness window around neighbours", priority=1)
    nbr_d_step = TQuantity(Q_(5.0, "µm"), group=nbr_grp).tag(name="Thickness window step", priority=2)
    max_window_shifts = Int(2, min=0, group=nbr_grp,
                            help="How often the window is moved to an optimum at its edge before the pixel gets "
                                 "the full thickness scan").tag(name="Maximum window moves", priority=3)
    nbr_iteration_count = Int(1, min=1, group=nbr_grp,
                              help="Coarse window scan plus refinement iterations, the refinement uses the "
                                   "thickness step shrink factor of the q-space evaluation. With the "
                                   "interpolated minimum, refinement iterations are rarely worth their cost").tag(
        name="Iteration count (window scan)", priority=4)
    max_q_ratio = Float(1.5, group=nbr_grp,
                        help="A window optimum whose q-value is more than this factor above the median q-value of "
                             "the neighbours is not trusted (the window probably misses a thickness step): the pixel "
                             "gets the full thickness scan, in the outlier re-evaluation the previous result is "
                             "kept").tag(name="Maximum q-value ratio to neighbours", priority=5)

    search_grp = "Thickness search"
    probe_pixel_count = Int(32, min=0, group=search_grp,
                            help="Number of random pixels that get the full thickness scan of the q-space "
                                 "evaluation first. All later full scans only cover the thickness range of the "
                                 "probe results (pixels with a q-value above twice the median are ignored) plus "
                                 "the margin; an optimum at the edge of that range gets the full range scan. "
                                 "0: always scan the full range").tag(name="Probe pixels", priority=10)
    probe_range_margin = TQuantity(Q_(20.0, "µm"), group=search_grp).tag(name="Probe range margin", priority=11)
    interpolate_minimum = Bool(True, group=search_grp,
                               help="The thickness of the minimum is the vertex of the parabola through the q-values "
                                    "of the best scanned thickness and its two neighbours, followed by one fit at "
                                    "that thickness (for n and q). Replaces the refinement iterations of the full "
                                    "scans").tag(name="Interpolate the q-value minimum", priority=12)

    outlier_grp = "Outlier re-evaluation"
    outlier_passes = Int(1, min=0, group=outlier_grp,
                         help="After the scan, pixels whose thickness deviates from the median of their 8 "
                              "neighbours by more than the threshold (and by more than 3x the scatter of the "
                              "neighbours) are scanned again in the window around the consistent neighbours. The "
                              "new result is only used if its optimum lies inside the window and passes the "
                              "q-value check, real thickness steps keep their value").tag(
        name="Re-evaluation passes", priority=20)
    outlier_threshold = TQuantity(Q_(10.0, "µm"), group=outlier_grp).tag(name="Thickness deviation threshold",
                                                                        priority=21)

    inspect_grp = "Reliability and inspection"
    max_width_ratio = Float(2.0, min=1.0, group=inspect_grp,
                            help="A pixel is reliable if its q(d) minimum is about as sharp as in the rest of the "
                                 "image (width: distance at which a parabola through the minimum doubles the "
                                 "q-value, at most this factor times the median width of all pixels), its q-value is "
                                 "at most the maximum q-value ratio above the median of its 8 neighbours and its "
                                 "thickness deviates from theirs by at most the outlier threshold (or 3x their "
                                 "scatter). Image plot quantities: Q-space reliable, minimum width, q-value / "
                                 "neighbours, thickness - neighbours").tag(
        name="Maximum width of the q-value minimum / median", priority=30)
    inspect_on_click = Bool(True, group=inspect_grp,
                            help="Clicking a pixel in the image plot of a Q-space quantity plots its q(d) curve, n "
                                 "and alpha instead of the default on-click plot").tag(
        name="Inspect pixel on click", priority=31)

    general_grp = "General"
    min_time_of_flight = TQuantity(Q_(0.1, "ps"), group=general_grp,
                                   help="Pixels whose mean time of flight (sample to reference delay) is below this "
                                        "value have no sample in the beam (air) and are not evaluated. 0: evaluate "
                                        "all pixels").tag(name="Minimum time of flight", priority=40)
    autosave_interval = TQuantity(Q_(60.0, "s"), group=general_grp,
                                  help="The results are saved to q_space_image_autosave.npz in the result directory "
                                       "during the evaluation at this interval and when it ends (also when cancelled "
                                       "or failed). Load it and continue the evaluation to resume. 0: off").tag(
        name="Autosave interval", priority=41)


class UncertaintyOpt(ComponentBase):
    # Monte Carlo uncertainty of the evaluations (common.eval_component.uncertainty)
    sources_grp = "Uncertainty sources"
    reference_drift = Bool(True, group=sources_grp,
                           help="Change of the reference pulse (arrival time, amplitude) between the reference and "
                                "the sample measurement, interpolated in the time series of all references. With the "
                                "reference drift correction (Settings -> Preprocessing -> Reference drift) the "
                                "residual error of the correction, from predicting each reference block from its "
                                "neighbours. The measurement noise (additive, amplitude and timing jitter) is always included, "
                                "fitted to consecutive references").tag(name="Include the reference drift",
                                                                        priority=0)
    extra_timing = TQuantity(Q_(0.0, "fs"), group=sources_grp,
                             help="Added (in quadrature) to the timing uncertainty of the sample against the "
                                  "reference, e.g. for a sample holder that changes the beam path").tag(
        name="Additional timing uncertainty", priority=1)
    extra_amplitude = Float(0.0, min=0.0, group=sources_grp,
                            help="Added (in quadrature) to the relative amplitude uncertainty of the reference").tag(
        name="Additional amplitude uncertainty (%)", priority=2)
    random_seed = Int(1, min=0, group=sources_grp,
                      help="Seed of the random perturbations, 0: different each run").tag(name="Random seed",
                                                                                         priority=3)

    qspace_grp = "Q-space thickness"
    qspace_enabled = Bool(False, group=qspace_grp,
                          help="After the thickness scan of each measurement (\"Fit transmission model\", queue), "
                               "the transmission is perturbed randomly with the estimated uncertainty and the "
                               "thickness window around the optimum is scanned again for each perturbation. The "
                               "spread of their q-value minima is the thickness uncertainty, the spread of their "
                               "n(f) the uncertainty of n and alpha. Costs samples x points fits per "
                               "measurement").tag(name="Thickness uncertainty (Monte Carlo)", priority=10)
    qspace_samples = Int(20, min=3, group=qspace_grp).tag(name="Monte Carlo samples", priority=11)
    qspace_window = TQuantity(Q_(3.0, "µm"), group=qspace_grp,
                              help="Half width of the thickness window scanned per sample, around the optimum. "
                                   "A sample with its minimum at the end of the window continues the window on that "
                                   "side (up to 4 times)").tag(
        name="Thickness window (half width)", priority=12)
    qspace_points = Int(7, min=3, group=qspace_grp).tag(name="Thicknesses per window", priority=13)

    cond_grp = "Conductivity"
    cond_enabled = Bool(True, group=cond_grp,
                        help="Uncertainty of the conductivity from the measurement noise, the reference drift and "
                             "the film / substrate thickness uncertainty (Conductivity evaluation)").tag(
        name="Conductivity uncertainty (Monte Carlo)", priority=20)
    cond_samples = Int(200, min=10, group=cond_grp).tag(name="Monte Carlo samples", priority=21)


class EvalOpt(ComponentBase):
    # sub-tabs of Settings -> Evaluation; settings saved before they existed are moved there on load
    # (common.settings.migrate_config)
    q_value = Instance(QValueOpt, kw={"object_name": "Q-value calculation"})
    spline_fit = Instance(SplineFitOpt, kw={"object_name": "Spline fit"})
    transmission_sim = Instance(TransmissionSimOpt, kw={"object_name": "Transmission simulation"})
    image_eval = Instance(ImageEvalOpt, kw={"object_name": "Image evaluation"})
    uncertainty = Instance(UncertaintyOpt, kw={"object_name": "Uncertainty"})

    phi_fit_range = ValueRange([Q_(0.47, "THz"), Q_(1.05, "THz")]).tag(name="Phase correction fit range")
    avg_grp_delay_fit_range = ValueRange([Q_(0.25, "THz"), Q_(0.85, "THz")]).tag(name="Group delay fit range")
    spectral_similarity_range = ValueRange([Q_(0.15, "THz"), Q_(2.00, "THz")]).tag(name="Spectral similarity range")

    delta_d = TQuantity(Q_(2.0, "µm"), help="Used for the uncertainty of n and alpha of averaged measurements "
                                            "(scatter of the measurements), unless the Monte Carlo thickness "
                                            "uncertainty (Uncertainty tab) replaces it at the optimum").tag(
        name="Thickness uncertainty")
    auto_fp_count = Bool(True, help="Number of Fabry-Perot echoes inside the recorded time window, estimated from "
                                    "the trace length after the main pulse and the round trip time. Only used by "
                                    "the truncated echo models (model_1layer_fp, t_model_2layer)").tag(
        name="Use estimated number of Fabry-Perot echoes")
    fp_count = Int(25, min=-1, help="-1: infinite number of echoes").tag(
        name="Assumed number of Fabry-Perot echoes")
    phi_offset_correction = Bool(True).tag(name="Phase offset correction")

    d = TQuantity(Q_(0.0, "µm")).tag(name="Default nominal sample thickness")
    d_film = TQuantity(Q_(0.0, "µm")).tag(name="Film thickness")

    smoothing_avg_grp = "Smoothing average"
    smoothing_avg_en = Bool(False).tag(name="Enable smoothing average of n and k", group=smoothing_avg_grp)
    smoothing_avg_n = Int(3).tag(name="Smoothing average window size", group=smoothing_avg_grp)
    smoothing_avg_iters = Int(3).tag(name="Smoothing average iterations", group=smoothing_avg_grp)

    shift_grp = "Shift axis settings"
    enable_shift_opt = Bool(False).tag(group=shift_grp).tag(name="Enable pulse shift optimization")
    shift_opt_axis_bounds = ValueRange([Q_(0, "fs"), Q_(0, "fs", )],
                                       group=shift_grp).tag(name="Shift axis bounds")
    shift_opt_axis_step = TQuantity(Q_(1, "fs"), group=shift_grp).tag(name="Shift axis step")

class StripeDirection(Enum):
    rows = "Rows (constant y)"
    columns = "Columns (constant x)"


class RefDriftOpt(ComponentBase):
    # correction of a drift of the THz pulse between references, e.g. references measured once per scan row
    interpolate_refs = Bool(False,
                            help="The transmission of a sample uses its matching reference and the reference (block) "
                                 "measured on the other side of it in time, interpolated to the sample's "
                                 "measurement time (amplitude and phase). Compensates slow drifts of the THz pulse "
                                 "between references, which shift the thickness of a q-space evaluation by about "
                                 "0.3 um per fs of timing drift").tag(name="Interpolate reference in time", priority=0)
    max_interp_gap = TQuantity(Q_(30.0, "min"),
                               help="Interpolate only between references (blocks) at most this far apart in time, "
                                    "otherwise the matching reference is used alone").tag(
        name="Maximum time between interpolated references", priority=1)
    block_median = Bool(False,
                        help="References measured in a row (less than the block gap apart, e.g. the air spots of one "
                             "opening of a sample holder) form a block, and the reference is the median of the "
                             "block (amplitude and phase). Keeps single references, e.g. partly blocked ones at the "
                             "edge of an opening, from setting the amplitude").tag(
        name="Median of reference blocks", priority=2)
    block_gap = TQuantity(Q_(60.0, "s")).tag(name="Maximum time between references of a block", priority=3)


class WindowOpt(ComponentBase):
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


class PpOpt(ComponentBase):
    # sub-tab of Settings -> Preprocessing; settings saved before it existed are moved there on load
    # (common.settings.migrate_config)
    window = Instance(WindowOpt, kw={"object_name": "Window"})
    ref_drift = Instance(RefDriftOpt, kw={"object_name": "Reference drift"})

    remove_dc = Bool(True).tag(name="Subtract DC")
    normalize_data = Bool(False).tag(name="Normalize waveform")

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
    auto_cbar_lim = Bool(False, group=image_group,
                         help="Color bar limits from percentiles of the shown pixels, so that outliers (sample "
                              "edges, holder, single bad pixels) don't stretch the color scale. Pixels without a value "
                              "(e.g. not evaluated) are ignored. The limits are logged, they can be copied to the "
                              "custom limits. Takes precedence over the custom limits").tag(
        name="Automatic color bar limits")
    auto_cbar_percentiles = ValueRange([1.0, 99.0], group=image_group).tag(
        name="Automatic limits: percentiles (%)")

    destripe_group = "Image destriping"
    destripe = Bool(False, group=destripe_group,
                    help="Removes stripes along the scan lines (e.g. from a drift of the system between scan rows): "
                         "the median of each line within the region is compared to the median of the neighbouring "
                         "lines, and the difference is removed from the whole line. Assumes that the sample in the "
                         "region has no real structure from line to line, real steps along the lines stay").tag(
        name="Remove stripes", priority=0)
    destripe_direction = TEnum(StripeDirection, StripeDirection.rows, group=destripe_group).tag(
        name="Stripe direction", priority=1)
    destripe_window = Int(15, min=3, group=destripe_group,
                          help="Number of neighbouring lines whose median is the reference of a line").tag(
        name="Lines for the trend", priority=2)
    destripe_x_range = ValueRange([Q_(0.0, "mm"), Q_(0.0, "mm")], group=destripe_group,
                                  help="Region used to determine the line offsets, equal bounds: the whole image").tag(
        name="Region x-range", priority=3)
    destripe_y_range = ValueRange([Q_(0.0, "mm"), Q_(0.0, "mm")], group=destripe_group).tag(
        name="Region y-range", priority=4)
    destripe_relative = Bool(False, group=destripe_group,
                             help="Divide the lines by their relative offset instead of subtracting it, for "
                                  "amplitude like quantities").tag(name="Relative offsets", priority=5)

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
    en_multilayer = Bool(True).tag(name="Multi-layer sample", group=q_space_eval, priority=1002)
    en_q_space_image = Bool(True).tag(name="Image evaluation", group=q_space_eval, priority=1003)

    regression = "Regression"
    en_regression = Bool(True).tag(name="Main component", group=regression, priority=1000)

    conductivity_eval = "Conductivity evaluation"
    en_conductivity_eval = Bool(True).tag(name="Main component", group=conductivity_eval, priority=1000)

    sparse_deconvolution = "Sparse deconvolution"
    en_sparse_deconvolution = Bool(True).tag(name="Main component", group=sparse_deconvolution, priority=1000)

class AppSettings(ComponentBase):
    save_settings_on_exit = Bool(True, help="Off: the settings are not written to the config file when the app closes "
                                            "(also not by the command line interface), changes are lost. Only this "
                                            "option itself is stored, so saving stays off until it is switched on "
                                            "again").tag(name="Save settings on exit")
    log_level = TEnum(LogLevel, default_value=LogLevel.info,
                      help="Lowest level of the messages shown in the message panel (debug: also the diagnostic "
                           "messages)").tag(name="Log level")
    max_log_lines = Int(250, min=10, help="Number of the newest messages kept in the message panel, older ones "
                                          "are removed").tag(name="Maximum message lines")

    save_settings = Instance(SaveSettings, args=())
    pp_opt = Instance(PpOpt, args=())
    eval_opt = Instance(EvalOpt, args=())
    plot_opt = Instance(PlotOpt, args=())
    shgo_options = Instance(SHGOOptions, args=())
    component_toggle = Instance(ComponentToggle, args=())
    remote = Instance(RemoteOpt, kw={"object_name": "Remote workers"})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.shgo_options = SHGOOptions()

if __name__ == '__main__':
    settings = AppSettings()
    print([getattr(settings.eval_opt, k) for k in settings.eval_opt.traits()])
