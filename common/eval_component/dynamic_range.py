"""
Largest measurable absorption coefficient (Jepsen & Fischer, Opt. Lett. 30, 29 (2005)): where the sample attenuates
the reference spectrum down to the noise floor, the extracted alpha stops following the sample and only reflects the
noise. alpha_max d = 2 ln(DR 4 n / (n + 1)^2), with DR the reference amplitude spectrum over the noise floor (the
additive noise of the noise model, divided by sqrt of the number of averaged sample measurements).

Smallest measurable absorption: alpha = -2 / d ln(|T| / t_Fresnel), so an uncertainty of ln|T| gives the absorption
coefficient an uncertainty 2 / d sigma(ln|T|), independent of alpha. Below k times that the extracted alpha can't be
told apart from 0. sigma(ln|T|)^2 = E|dT|^2 / (2 |T|^2) + a^2: the noise of the (mean) transmission (component along T)
and the relative uncertainty a of the reference amplitude (drift, extra amplitude uncertainty), see uncertainty.py.
"""
import numpy as np
from traitlets import Bool, Float
from common.components import ComponentBase
from common.eval_component.quantity_set import QuantityDataSet
from common.units import Q_


class DynamicRangeOpt(ComponentBase):
    enabled = Bool(True, help="Adds the largest measurable absorption coefficient alpha_max to the q-space results "
                              "(quantity \"alpha_max\", check it together with alpha in the result view) and to the "
                              "pixel view of the image evaluation. Above it the absorption is set by the noise, not "
                              "by the sample").tag(name="Maximum measurable absorption")
    alpha_min_enabled = Bool(True, help="Adds the smallest measurable absorption coefficient alpha_min (quantity "
                                        "\"alpha_min\") to the q-space results and the pixel view of the image "
                                        "evaluation: coverage factor x the uncertainty of alpha from the transmission "
                                        "amplitude noise and the reference amplitude drift (Settings -> Evaluation -> "
                                        "Uncertainty). Below it alpha isn't distinguishable from 0").tag(
        name="Minimum measurable absorption")
    alpha_min_coverage = Float(2.0, min=0.1, help="alpha_min is this many standard uncertainties of alpha").tag(
        name="Coverage factor (alpha_min)")


def alpha_max(freq, ref_amplitude, noise_amplitude, n_real, d_um):
    # 1/cm; freq only for the shape. Where the reference is below the noise floor alpha_max is 0
    dr = np.abs(ref_amplitude) / max(float(noise_amplitude), 1e-300)
    n_real = np.asarray(n_real, dtype=float)
    arg = dr * 4 * n_real / (n_real + 1) ** 2
    return 2 / (d_um * 1e-4) * np.log(np.maximum(arg, 1.0))


def alpha_min(t, unc, d_um, coverage):
    # 1/cm; t: (mean) transmission on the frequencies of unc (TransmissionUncertainty)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_var = np.asarray(unc.variance) / (2 * np.abs(t) ** 2)
    return coverage * 2 / (d_um * 1e-4) * np.sqrt(rel_var + unc.amplitude ** 2)


def transmission_noise(dataset, meas_list, ref_list, freq_idx):
    from common.eval_component.uncertainty import estimator_for
    return estimator_for(dataset).transmission_uncertainty(meas_list, ref_list, freq_idx,
                                                           dataset.settings.eval_opt.uncertainty)


def noise_floor(dataset, sample_cnt=1):
    # amplitude of the additive spectrum noise of the averaged sample measurements (the noise model of the references)
    # imported here: the uncertainty module imports the settings, which hold DynamicRangeOpt
    from common.eval_component.uncertainty import estimator_for
    estimator = estimator_for(dataset)
    estimator.update()
    return estimator.noise_model.additive / np.sqrt(max(int(sample_cnt), 1))


def alpha_max_dataset(freq_axis, ref_amplitude, noise_amplitude, n_real, d_um):
    values = alpha_max(np.asarray(freq_axis.magnitude), ref_amplitude, noise_amplitude, n_real, d_um)
    return QuantityDataSet(axes=[freq_axis], data=Q_(values, "1/cm"), axes_labels=["Frequency"],
                           data_label="Maximum measurable absorption")


def alpha_min_dataset(freq_axis, t, unc, d_um, coverage):
    return QuantityDataSet(axes=[freq_axis], data=Q_(alpha_min(t, unc, d_um, coverage), "1/cm"),
                           axes_labels=["Frequency"], data_label="Minimum measurable absorption")


def below_limit(freq, alpha, alpha_lim):
    # frequencies where alpha is below alpha_min: count and range, None if none
    below = np.asarray(alpha) < np.asarray(alpha_lim)
    if not below.any():
        return None
    return int(below.sum()), float(freq[below][0]), float(freq[below][-1])


def limit_frequency(freq, alpha, alpha_lim, fraction=0.9):
    # first frequency where alpha reaches fraction x alpha_max, None if it stays below
    reached = np.flatnonzero(np.asarray(alpha) >= fraction * np.asarray(alpha_lim))
    return float(freq[reached[0]]) if len(reached) else None
