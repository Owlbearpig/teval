import ast
import numpy as np
from numpy import sqrt
from common.consts import eps0_thz, c_thz, pi
from common.traits import FitParameter
from common.eval_component.model_latex import wrap_expression
from dataclasses import dataclass
from enum import Enum
from typing import Callable


def sigma_to_n(freq, sig):
    # [eps0_thz] = ps * S / µm
    # sig_ in S/cm -> S/µm ( 1/(1e6 µm) = 1/m = 1/(1e2 cm) => 1e-4/µm = 1/cm)
    sig *= 1e-4

    w = 2*pi*freq

    sig *= 1e-4 # S/cm -> S/µm
    n = sqrt(1 + 1j * sig / (eps0_thz * w))

    return n

def n_to_sigma(freq, n):
    # [eps0_thz] = ps * S / µm
    # sig_ in S/cm -> S/µm ( 1/(1e6 µm) = 1/m = 1/(1e2 cm) => 1e-4/µm = 1/cm)

    w = 2 * pi * freq

    sig = -1j*(n ** 2 - 1) * w * eps0_thz
    sig *= 1e4 # S/µm -> S/cm

    return sig

def drude(freq, sig0, tau):
    # [tau] = fs, [sig0] = S/cm,
    # => [sig_cc_] = S/cm
    tau *= 1e-3  # fs = 1e-3 ps
    tau /= 2 * pi

    scale = 1
    sig0 *= scale

    w = 2 * pi * freq
    sig_cc_ = sig0 / (1 - 1j * tau * w)

    return sig_cc_


def drude2(freq_, sig0, tau, wp, eps_inf):
    tau *= 1e-3
    tau /= 2 * pi
    w = 2 * pi * freq_

    return sig0 * wp ** 2 / (tau - 1j * w) - 1j * eps0_thz * w * (eps_inf - 1)

def drude_smith(freq_, sig0, tau, c1):
    tau *= 1e-3
    tau /= 2 * pi
    w = 2 * pi * freq_

    sig_drude = sig0 / (1 - 1j * tau * w)
    smith_term = sig0 * c1 / (1 - 1j * tau * w)**2

    return sig_drude + smith_term

def lattice_contrib(freq_, tau, wp, eps_s, eps_inf):
    tau *= 1e-3  # fs = 1e-3 ps
    tau /= 2 * pi

    wp *= 2 * pi

    w = 2 * pi * freq_

    eps_l = eps_inf - ((eps_s - eps_inf) * wp ** 2) / (w ** 2 - wp ** 2 + 1j * w / tau)

    return eps_l


def total_response(freq, sig0, tau, wp, eps_s, eps_inf, c1):
    # [freq] = THz, [tau] = fs, [sig0] = S/cm, [wp] = THz. Dimensionless: eps_inf, eps_s
    sig_cc = drude(freq, sig0, tau)
    # sig_cc = drude_smith(freq, tau, sig0, c1)
    eps_l = lattice_contrib(freq, tau, wp, eps_s, eps_inf)

    n = sqrt(eps_l)

    sig_tot = n_to_sigma(freq, n) + sig_cc

    return sig_tot


@dataclass(frozen=True)
class RegressionModel:
    # func(freq, **params): freq in THz, each parameter value in the unit of its default FitParameter
    func: Callable | None
    parameters: tuple[FitParameter, ...] = ()
    latex: tuple[str, ...] = ()  # formula lines as computed by func


_sig0 = FitParameter("sig0", "S/cm", 10, 20, "σ₀")
_tau = FitParameter("tau", "fs", 10, 1000, "τ")
_wp = FitParameter("wp", "THz", -10, 100, "ωₚ")
_eps_inf = FitParameter("eps_inf", "", 1, 100, "ε_inf")
_eps_s = FitParameter("eps_s", "", 1, 100, "ε_s")
_c1 = FitParameter("c1", "", -1, 1, "c₁")


_drude_latex = r"\sigma_D(f) = \frac{\sigma_0}{1 - i\,\omega\,\tau^\prime}"
_defs_latex = r"\omega = 2\pi f, \quad \tau^\prime = \frac{\tau}{2\pi}"


class RegressionModels(Enum):
    drude = RegressionModel(drude, (_sig0, _tau), (_drude_latex, _defs_latex))
    drude2 = RegressionModel(drude2, (_sig0, _tau, _wp, _eps_inf), (
        r"\sigma(f) = \frac{\sigma_0\,\omega_p^2}{\tau^\prime - i\,\omega} - i\,\varepsilon_0\,\omega"
        r"\left(\varepsilon_\infty - 1\right)", _defs_latex))
    lattice_drude = RegressionModel(total_response, (_sig0, _tau, _wp, _eps_s, _eps_inf, _c1), (
        r"\sigma(f) = \sigma_D(f) - i\,\varepsilon_0\,\omega\left(\varepsilon_L(f) - 1\right)",
        _drude_latex,
        r"\varepsilon_L(f) = \varepsilon_\infty - \frac{\left(\varepsilon_s - \varepsilon_\infty\right)"
        r"\,\omega_p^{\prime 2}}{\omega^2 - \omega_p^{\prime 2} + i\,\omega / \tau^\prime}",
        _defs_latex + r", \quad \omega_p^\prime = 2\pi\,\omega_p \quad (c_1 \mathrm{\ unused})"))
    drude_smith = RegressionModel(drude_smith, (_sig0, _tau, _c1), (
        r"\sigma(f) = \frac{\sigma_0}{1 - i\,\omega\,\tau^\prime} "
        r"+ \frac{\sigma_0\,c_1}{\left(1 - i\,\omega\,\tau^\prime\right)^2}", _defs_latex))
    custom = RegressionModel(None)


def parameter_label(name):
    for model in RegressionModels:
        for p in model.value.parameters:
            if p.name == name:
                return p.label
    return ""


# names usable in custom model expressions (besides freq and the free parameters)
expression_namespace = {
    "np": np, "pi": pi, "c_thz": c_thz, "eps0_thz": eps0_thz,
    "sqrt": np.emath.sqrt, "log": np.emath.log, "log10": np.emath.log10, "exp": np.exp,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "arctan": np.arctan, "sinh": np.sinh, "cosh": np.cosh,
    "tanh": np.tanh, "abs": np.abs, "real": np.real, "imag": np.imag, "conj": np.conj, "angle": np.angle,
    "drude": drude, "drude2": drude2, "drude_smith": drude_smith, "lattice_contrib": lattice_contrib,
    "total_response": total_response, "n_to_sigma": n_to_sigma,
}


def expression_parameters(expression):
    # free parameters: all names in the expression which are neither freq nor in the namespace, in order of appearance
    tree = ast.parse(wrap_expression(expression), mode="eval")
    name_nodes = sorted((node for node in ast.walk(tree) if isinstance(node, ast.Name)),
                        key=lambda node: (node.lineno, node.col_offset))

    names = []
    for node in name_nodes:
        if node.id != "freq" and node.id not in expression_namespace and node.id not in names:
            names.append(node.id)

    return names


def make_expression_model(expression):
    param_names = expression_parameters(expression)
    code = compile(wrap_expression(expression), "<regression model>", "eval")
    global_ns = {"__builtins__": {}, **expression_namespace}

    def expression_model(freq, **params):
        res = eval(code, global_ns, {"freq": freq, **params})
        return np.broadcast_to(res, np.shape(freq))

    return expression_model, param_names
