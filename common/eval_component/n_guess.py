import numpy as np
from enum import Enum
from scipy.optimize import minimize_scalar
from common.eval_component.cost_functions import abs_cost_fun, phi_cost_fun


class NGuessMethod(Enum):
    tof = "Time of flight (fixed thickness)"
    cost_scan = "Model cost scan (per thickness)"


def _model_grid(model, n_grid, freq_axis, model_kwargs):
    try:
        t_mod = model(n_grid, freq_axis[None, :], **model_kwargs)
        if np.shape(t_mod) == n_grid.shape:
            return t_mod
    except Exception:
        pass

    return np.array([model(n_row, freq_axis, **model_kwargs) for n_row in n_grid])


def cost_scan_n_guess(d, shift, config_dict):
    freq_axis = np.asarray(config_dict["freq_axis"])
    t_exp = np.asarray(config_dict["t_exp"])[:, 1]
    model = config_dict["transmission_model"]
    n_min, n_max = config_dict["n_scan_range"]
    n_step = config_dict["n_scan_step"]
    k_max = config_dict["k_scan_max"]

    model_kwargs_keys = ["n_sub", "n1", "n4", "h", "nfp", "layers"]
    model_kwargs = {k: config_dict[k] for k in model_kwargs_keys if k in config_dict}
    model_kwargs["shift"] = shift
    model_kwargs["d"] = d

    one = np.ones_like(freq_axis, dtype=complex)

    n_axis = np.arange(n_min, n_max + n_step, n_step)
    phi_cost = np.sum(phi_cost_fun(t_exp, _model_grid(model, n_axis[:, None] * one, freq_axis, model_kwargs)), axis=1)
    n_coarse = n_axis[np.argmin(phi_cost)]

    def band_cost(n_):
        return np.sum(phi_cost_fun(t_exp, model(n_ * one, freq_axis, **model_kwargs)))

    n_real = minimize_scalar(band_cost, bounds=(n_coarse - n_step, n_coarse + n_step), method="bounded").x

    k_cnt = 100
    k_axis = np.linspace(k_max / k_cnt, k_max, k_cnt)
    n_grid = n_real + 1j * k_axis[:, None] * one
    abs_cost = abs_cost_fun(t_exp, _model_grid(model, n_grid, freq_axis, model_kwargs))
    k = k_axis[np.argmin(abs_cost, axis=0)]
    k = np.clip(np.polyval(np.polyfit(freq_axis, k, 2), freq_axis), k_axis[0], k_max)

    return np.column_stack((freq_axis, n_real + 1j * k))


def get_n_guess(d, shift, config_dict):
    if config_dict.get("n_guess_method") == NGuessMethod.cost_scan:
        return cost_scan_n_guess(d, shift, config_dict)

    return np.asarray(config_dict["n_guess"])
