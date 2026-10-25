"""
Largest measurable absorption coefficient (Jepsen & Fischer, Opt. Lett. 30, 29 (2005)): where the sample attenuates
the reference spectrum down to the noise floor, the extracted alpha stops following the sample and only reflects the
noise. alpha_max d = 2 ln(DR 4 n / (n + 1)^2), with DR the reference amplitude spectrum over the noise floor (the
additive noise of the noise model, divided by sqrt of the number of averaged sample measurements).
"""
import numpy as np
from traitlets import Bool
from common.components import ComponentBase
from common.eval_component.quantity_set import QuantityDataSet
from common.units import Q_


class DynamicRangeOpt(ComponentBase):
    enabled = Bool(True, help="Adds the largest measurable absorption coefficient alpha_max to the q-space results "
                              "(quantity \"alpha_max\", check it together with alpha in the result view) and to the "
                              "pixel view of the image evaluation. Above it the absorption is set by the noise, not "
                              "by the sample").tag(name="Maximum measurable absorption")


def alpha_max(freq, ref_amplitude, noise_amplitude, n_real, d_um):
    # 1/cm; freq only for the shape. Where the reference is below the noise floor alpha_max is 0
    dr = np.abs(ref_amplitude) / max(float(noise_amplitude), 1e-300)
    n_real = np.asarray(n_real, dtype=float)
    arg = dr * 4 * n_real / (n_real + 1) ** 2
    return 2 / (d_um * 1e-4) * np.log(np.maximum(arg, 1.0))


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


def limit_frequency(freq, alpha, alpha_lim, fraction=0.9):
    # first frequency where alpha reaches fraction x alpha_max, None if it stays below
    reached = np.flatnonzero(np.asarray(alpha) >= fraction * np.asarray(alpha_lim))
    return float(freq[reached[0]]) if len(reached) else None
