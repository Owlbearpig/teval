"""
Effective medium models of a mixture of a host (matrix, e.g. PE powder or air for porosity) and inclusions (the
material of interest) with volume fraction f: the mixture's refractive index from both, the volume fraction from the
measured mixture index, or the inclusion index from the mixture index at a known fraction. With q-space image results,
a volume fraction map.

Refractive indices in the convention of the evaluation results (n = n' + i k, k > 0), eps = n^2.
"""
import logging
from enum import Enum
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from traitlets import Float, Enum as TEnum, Unicode
from common.components import action
from common.dataset import DataSetFormatter
from common.default_appsettings import QuantityEnum
from common.eval_component.layer_stack import load_layer_result
from common.eval_component.q_space_image import alpha_of
from common.traits import Q_, ValueRange, Path as TPath
from mpl_settings import mpl_style_params

action = partial(action, check_init=True, rc_params=mpl_style_params)

fraction_grid = np.linspace(0.0, 1.0, 2001)


class EMTModel(Enum):
    bruggeman = "Bruggeman (symmetric, any fraction)"
    maxwell_garnett = "Maxwell Garnett (dilute spherical inclusions)"
    looyenga = "Landau-Lifshitz-Looyenga"
    linear_n = "Linear in n (CRIM)"


class EMTTask(Enum):
    fraction = "Volume fraction from the mixture"
    inclusion = "Inclusion index from the mixture"
    mixture = "Mixture index from the fraction"


class IndexSource(Enum):
    constant = "Constant"
    result = "Result file"


def _root_physical(eps_a, eps_b):
    # of two roots the one with Im(eps) >= 0 (passive medium), and Re(n) > 0 of its square root
    pick_b = (eps_a.imag < -1e-12) | ((np.abs(eps_a.imag) <= 1e-12) & (eps_a.real < eps_b.real))
    return np.where(pick_b, eps_b, eps_a)


def _sqrt_n(eps):
    n = np.sqrt(np.asarray(eps, dtype=complex))
    return np.where(n.real < 0, -n, n)


def mixture_index(model, n_host, n_incl, f):
    # n of the mixture, broadcast over frequency (and fraction) arrays
    f = np.asarray(f, dtype=float)
    if model == EMTModel.linear_n:
        return f * n_incl + (1 - f) * n_host
    e_h, e_i = np.asarray(n_host, dtype=complex) ** 2, np.asarray(n_incl, dtype=complex) ** 2
    if model == EMTModel.maxwell_garnett:
        beta = (e_i - e_h) / (e_i + 2 * e_h)
        return _sqrt_n(e_h * (1 + 2 * f * beta) / (1 - f * beta))
    if model == EMTModel.looyenga:
        return _sqrt_n((f * e_i ** (1 / 3) + (1 - f) * e_h ** (1 / 3)) ** 3)
    # Bruggeman: f (e_i - e) / (e_i + 2 e) + (1 - f) (e_h - e) / (e_h + 2 e) = 0  ->  2 e^2 - b e - e_i e_h = 0
    b = (3 * f - 1) * e_i + (2 - 3 * f) * e_h
    root = np.sqrt(b ** 2 + 8 * e_i * e_h)
    return _sqrt_n(_root_physical((b + root) / 4, (b - root) / 4))


def inclusion_index(model, n_host, n_mix, f):
    # n of the inclusions that gives the mixture index at the volume fraction f (inverse of mixture_index)
    if f <= 0:
        raise ValueError("The volume fraction must be above 0")
    if model == EMTModel.linear_n:
        return (n_mix - (1 - f) * n_host) / f
    e_h, e = np.asarray(n_host, dtype=complex) ** 2, np.asarray(n_mix, dtype=complex) ** 2
    if model == EMTModel.maxwell_garnett:
        beta = (e - e_h) / (f * (e + 2 * e_h))
        return _sqrt_n(e_h * (1 + 2 * beta) / (1 - beta))
    if model == EMTModel.looyenga:
        return _sqrt_n(((e ** (1 / 3) - (1 - f) * e_h ** (1 / 3)) / f) ** 3)
    ratio = (1 - f) * (e_h - e) / (f * (e_h + 2 * e))
    return _sqrt_n(e * (1 - 2 * ratio) / (1 + ratio))


def fit_fraction(model, n_host, n_incl, n_mix, weights=None):
    # volume fraction that fits the mixture index best (least squares of n' and k over the frequencies, on a fine
    # fraction grid), and the rms residual of n
    n_mix = np.atleast_2d(np.asarray(n_mix, dtype=complex))
    models = mixture_index(model, np.asarray(n_host)[None, :], np.asarray(n_incl)[None, :], fraction_grid[:, None])
    w = np.ones(n_mix.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(n_mix).all(axis=0) & np.isfinite(models).all(axis=0), w, 0.0)
    n_mix = np.nan_to_num(n_mix)
    models = np.nan_to_num(models)
    # sum_f w |m - n|^2 for all fractions and mixtures at once: |m|^2 - 2 Re(m conj(n)) + |n|^2
    costs = ((np.abs(models) ** 2) @ w)[:, None] - 2 * np.real((models * w) @ np.conj(n_mix).T) \
        + ((np.abs(n_mix) ** 2) @ w)[None, :]
    best = np.argmin(costs, axis=0)
    rms = np.sqrt(np.maximum(costs[best, np.arange(len(best))], 0) / max(np.sum(w), 1e-300))
    return fraction_grid[best], rms


class EffectiveMedium(DataSetFormatter):
    model_grp = "Model"
    model = TEnum(EMTModel, EMTModel.bruggeman,
                  help="Bruggeman: both phases equivalent, valid for any fraction (pellets of mixed powders). "
                       "Maxwell Garnett: isolated spherical inclusions in the host, small fractions. "
                       "Looyenga: independent of the particle shape, often used for powders and porosity. Linear "
                       "in n: the time of flight adds up (layers along the beam)").tag(
        name="Effective medium model", group=model_grp, priority=0)
    task = TEnum(EMTTask, EMTTask.fraction).tag(name="Calculate", group=model_grp, priority=1)
    volume_fraction = Float(0.1, min=0.0, max=1.0, help="Volume fraction of the inclusions (for the inclusion "
                                                        "index and the mixture index)").tag(
        name="Volume fraction of the inclusions", group=model_grp, priority=2)
    freq_range = ValueRange(default_value=[Q_(0.3, "THz"), Q_(1.5, "THz")],
                            help="Frequency range of the plots and of the volume fraction fit").tag(
        name="Frequency range", group=model_grp, priority=3)

    mixture_grp = "Mixture (measured)"
    mixture_path = TPath(Path("."), must_exist=False,
                         help="Q-space evaluation result file (HDF5) of the mixture, its optimum (or average) result "
                              "is used").tag(name="Mixture result file", group=mixture_grp, priority=10)

    host_grp = "Host (matrix)"
    host_source = TEnum(IndexSource, IndexSource.constant).tag(name="Host index from", group=host_grp, priority=20)
    host_n = ValueRange([1.52, 0.0], help="n and k of the host, e.g. PE about 1.52, air 1.0").tag(
        name="Host n, k", group=host_grp, priority=21)
    host_path = TPath(Path("."), must_exist=False).tag(name="Host result file", group=host_grp, priority=22)

    incl_grp = "Inclusions"
    incl_source = TEnum(IndexSource, IndexSource.constant).tag(name="Inclusion index from", group=incl_grp,
                                                               priority=30)
    incl_n = ValueRange([2.0, 0.01]).tag(name="Inclusion n, k", group=incl_grp, priority=31)
    incl_path = TPath(Path("."), must_exist=False).tag(name="Inclusion result file", group=incl_grp, priority=32)

    result_grp = "Result"
    result_text = Unicode("", read_only=True).tag(name="Result", group=result_grp, priority=40, en_save=False)

    def __init__(self, dataset, q_space_eval=None, **kwargs):
        super().__init__(dataset, **kwargs)
        self.q_space_eval = q_space_eval
        dataset.register_quantity_func(QuantityEnum.EMTVolumeFraction, self.fraction_quantity)

    def _band(self, freq):
        f0, f1 = sorted(q.to("THz").magnitude for q in self.freq_range)
        return (freq >= f0) & (freq <= f1)

    @staticmethod
    def _index(source, const, path, freq, label):
        if source == IndexSource.constant:
            return np.full(len(freq), float(const[0]) + 1j * float(const[1]))
        f_res, n_res, _, meas = load_layer_result(path)
        if freq[0] < f_res[0] or freq[-1] > f_res[-1]:
            logging.warning(f"The {label} result covers {f_res[0]:.2f} - {f_res[-1]:.2f} THz only, its index is "
                            f"extended with the end values")
        return np.interp(freq, f_res, n_res.real) + 1j * np.interp(freq, f_res, n_res.imag)

    def _components(self, freq):
        n_host = self._index(self.host_source, self.host_n, self.host_path, freq, "host")
        n_incl = None
        if self.task != EMTTask.inclusion:
            n_incl = self._index(self.incl_source, self.incl_n, self.incl_path, freq, "inclusion")
        return n_host, n_incl

    def _mixture(self):
        freq, n_mix, d, meas = load_layer_result(self.mixture_path)
        band = self._band(freq)
        if not band.any():
            raise ValueError(f"The mixture result has no frequencies in the frequency range "
                             f"({freq[0]:.2f} - {freq[-1]:.2f} THz)")
        return freq[band], n_mix[band], meas

    @action("Calculate", group=result_grp, priority=41)
    def calculate(self):
        try:
            if self.task == EMTTask.mixture:
                f0, f1 = sorted(q.to("THz").magnitude for q in self.freq_range)
                freq = np.linspace(f0, f1, 300)
                n_mix_meas, meas = None, None
            else:
                freq, n_mix_meas, meas = self._mixture()
            n_host, n_incl = self._components(freq)
        except (ValueError, OSError, KeyError) as e:
            logging.error(f"Effective medium: {e}")
            return

        fig = plt.figure("Effective medium", figsize=(16, 7))
        fig.clf()
        ax_n, ax_a = fig.subplots(1, 2)
        model = self.model.value.split(" (")[0]

        def plot(n, label, **kw):
            ax_n.plot(freq, n.real, label=label, **kw)
            ax_a.plot(freq, alpha_of(freq, n).real, label=label, **kw)

        if self.task == EMTTask.fraction:
            fractions, rms = fit_fraction(self.model, n_host, n_incl, n_mix_meas)
            f_fit = float(fractions[0])
            # the fit is on a grid of the fraction, at its ends the measured index is outside of the model range
            edge = " (at the end of 0 - 1: the mixture index is outside of what the model gives)" * (
                f_fit in (fraction_grid[0], fraction_grid[-1]))
            text = f"Volume fraction {f_fit:.3f} ({model}), rms deviation of n {rms[0]:.4f}{edge}"
            plot(n_mix_meas, f"Measured ({meas})", color="C0")
            plot(mixture_index(self.model, n_host, n_incl, f_fit), f"Model, f = {f_fit:.3f}", color="C3", ls="--")
        elif self.task == EMTTask.inclusion:
            n_incl = inclusion_index(self.model, n_host, n_mix_meas, self.volume_fraction)
            text = (f"Inclusion index at f = {self.volume_fraction:g} ({model}): n {np.mean(n_incl.real):.4f}, "
                    f"alpha {np.mean(alpha_of(freq, n_incl).real):.2f} 1/cm (means over the range)")
            plot(n_mix_meas, f"Mixture ({meas})", color="C0")
            plot(n_incl, "Inclusions", color="C3")
        else:
            n_mix = mixture_index(self.model, n_host, n_incl, self.volume_fraction)
            text = (f"Mixture index at f = {self.volume_fraction:g} ({model}): n {np.mean(n_mix.real):.4f}, "
                    f"alpha {np.mean(alpha_of(freq, n_mix).real):.2f} 1/cm (means over the range)")
            plot(n_mix, "Mixture (model)", color="C3")
        plot(n_host, "Host", color="0.5", lw=0.8)
        if n_incl is not None and self.task != EMTTask.inclusion:
            plot(n_incl, "Inclusions", color="0.3", lw=0.8, ls=":")
        ax_n.set_ylabel("Refractive index")
        ax_a.set_ylabel("Absorption coefficient (1/cm)")
        for ax in (ax_n, ax_a):
            ax.set_xlabel("Frequency (THz)")
            ax.legend(fontsize=12)
        fig.suptitle(text, fontsize=14)
        fig.tight_layout()
        self.set_trait("result_text", text)
        logging.info(f"Effective medium: {text}")
        if self.settings.plot_opt.auto_show:
            plt.show()

    def fraction_quantity(self, meas_list):
        # volume fraction per pixel from the n(f) of the q-space image results (host and inclusion index of the
        # settings, frequency range), nan without a result
        out = np.full(len(meas_list), np.nan)
        image_eval = getattr(self.q_space_eval, "image_eval", None)
        if image_eval is None or image_eval._freq_axis is None:
            return out
        freq = image_eval._freq_axis
        band = self._band(freq)
        if not band.any():
            logging.warning("The frequency range of the effective medium has no frequencies of the image evaluation")
            return out
        try:
            n_host, n_incl = self._index(self.host_source, self.host_n, self.host_path, freq[band], "host"), \
                self._index(self.incl_source, self.incl_n, self.incl_path, freq[band], "inclusion")
        except (ValueError, OSError, KeyError) as e:
            logging.error(f"Effective medium: {e}")
            return out
        results = image_eval._lookup(meas_list)
        rows = [i for i, res in enumerate(results) if res is not None]
        if not rows:
            return out
        cache = {}
        for i in rows:
            res = results[i]
            if id(res) not in cache:
                cache[id(res)] = res.n[band]
        unique_ids = list(cache)
        fractions, _ = fit_fraction(self.model, n_host, n_incl, np.array([cache[k] for k in unique_ids]))
        by_id = dict(zip(unique_ids, fractions))
        for i in rows:
            out[i] = by_id[id(results[i])]
        return out
