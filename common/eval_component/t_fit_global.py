import numpy as np
import logging
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from common.consts import c_thz
from common.eval_component.eval_result import SingleResultData
from common.units import Q_
from common.eval_component.quantity_set import QuantityDataSet
from datetime import datetime

def spline_transmission_optimization(d, shift, config_dict):
    freq_axis = np.asarray(config_dict["freq_axis"])
    n0 = np.asarray(config_dict["n_guess"])
    t_exp = np.asarray(config_dict["t_exp"])
    transmission_model = config_dict["transmission_model"]
    cost_fun = config_dict["cost_fun"]
    n_knots = config_dict["knot_count"]
    lambda_n = config_dict["reg_n"]
    lambda_k = config_dict["reg_k"]
    max_nfev = config_dict["max_nfev"]

    model_kwargs_keys = ["n_sub", "n1", "n4", "h", "nfp"]
    model_kwargs = {k: config_dict[k] for k in model_kwargs_keys if k in config_dict}
    model_kwargs["shift"] = shift
    model_kwargs["d"] = d

    t_exp = t_exp[:, 1]

    n0 = n0[:, 1]

    n_scale = max(np.ptp(n0.real), np.mean(np.abs(n0.real)) * 0.01, 1e-6)
    k_scale = max(np.ptp(n0.imag), np.mean(np.abs(n0.imag)) * 0.01, 1e-6)

    knot_idx = np.linspace(0, len(freq_axis) - 1, n_knots, dtype=int)
    if n_knots > len(freq_axis):
        msg = f"n_knots={n_knots} is larger than the number of frequency points={len(freq_axis)}"
        logging.error(msg)
        raise ValueError(msg)

    knot_idx = np.round(np.linspace(0, len(freq_axis) - 1, n_knots)).astype(int)

    freq_knots = freq_axis[knot_idx]

    def make_splines(p):
        n_knots_values = p[:n_knots]
        k_knots_values = p[n_knots:]

        spline_n = CubicSpline(freq_knots, n_knots_values, bc_type="natural")
        spline_k = CubicSpline(freq_knots, k_knots_values, bc_type="natural")

        return spline_n, spline_k

    def calculate_model(p):
        spline_n, spline_k = make_splines(p)

        n = spline_n(freq_axis) + 1j * spline_k(freq_axis)
        t_mod = transmission_model(n, freq_axis, **model_kwargs)

        return n, t_mod

    def residual_fun(p):
        n, t_mod = calculate_model(p)

        data_residual = cost_fun(t_exp, t_mod)

        spline_n, spline_k = make_splines(p)

        d2n = spline_n(freq_axis, 2)
        d2k = spline_k(freq_axis, 2)

        d2n_normalized = d2n / n_scale
        d2k_normalized = d2k / k_scale

        residuals = np.concatenate([data_residual,
                                    np.sqrt(lambda_n) * d2n_normalized,
                                    np.sqrt(lambda_k) * d2k_normalized,
        ])

        return residuals

    n_real_0 = n0.real[knot_idx]
    kappa_0 = n0.imag[knot_idx]

    p0 = np.concatenate([n_real_0, kappa_0])
    """
    best = (np.inf, 1, 1)
    for x in np.arange(0.50, 2.0, 0.10):
        for y in np.arange(0.50, 2.0, 0.10):
            p0_test = np.concatenate([x*n_real_0, y*kappa_0])
            s = np.sum(residual_fun(p0_test))
            if s < best[0]:
                best = (s, x, y)
    """
    n0_real_knots = n0.real[knot_idx]
    n0_imag_knots = n0.imag[knot_idx]

    n_lower, n_upper = 0.50 * n0_real_knots, 2.00 * n0_real_knots
    k_lower, k_upper = 0.50 * n0_imag_knots, 2.00 * n0_imag_knots

    lower_bounds = np.clip(np.concatenate([n_lower, k_lower]), 0, a_max=np.inf)
    upper_bounds = np.clip(np.concatenate([n_upper, k_upper]), 1e-6, a_max=np.inf)

    p0 = np.clip(p0, lower_bounds + 1e-12, upper_bounds - 1e-12)

    opt_res = least_squares(
        residual_fun,
        p0,
        bounds=(lower_bounds, upper_bounds),
        x_scale="jac",
        max_nfev = max_nfev,
    )

    n_opt_res, t_mod_res = calculate_model(opt_res.x)

    gof = np.sum(cost_fun(t_exp, t_mod_res))

    alpha_ = (freq_axis * 4 * np.pi * n_opt_res.imag / (1e-4 * c_thz))

    result_data = SingleResultData()
    result_data.d = Q_(d, "µm")
    result_data.shift = Q_(shift, "fs")

    result_data.freq_axis = Q_(freq_axis, "THz")

    result_data.optimization_info["gof"] = Q_(gof / len(freq_axis), "")

    result_data.optimization_info["timestamp"] = str(datetime.now().isoformat())
    result_data.optimization_info["cost"] = opt_res.cost
    result_data.optimization_info["optimality"] = opt_res.optimality
    result_data.optimization_info["nfev"] = opt_res.nfev
    result_data.optimization_info["njev"] = opt_res.njev
    result_data.optimization_info["status"] = opt_res.status
    result_data.optimization_info["optimizer_message"] = opt_res.message
    result_data.optimization_info["converged"] = opt_res.success

    result_data.optimization_info["n_knots"] = n_knots
    result_data.optimization_info["lambda_n"] = lambda_n
    result_data.optimization_info["lambda_k"] = lambda_k

    result_data.datasets["t_exp"] = QuantityDataSet(
        axes=[Q_(freq_axis, "THz")],
        data=Q_(t_exp,""),
        axes_labels=["Frequency"],
        data_label="Transmission coefficient experimental"
    )

    result_data.datasets["n0"] = QuantityDataSet(
        axes=[Q_(freq_axis, "THz")],
        data=Q_(n0,""),
        axes_labels=["Frequency"],
        data_label="Refractive index"
    )

    result_data.datasets["n"] = QuantityDataSet(
        axes=[Q_(freq_axis, "THz")],
        data=Q_(n_opt_res,""),
        axes_labels=["Frequency"],
        data_label="Refractive index"
    )

    result_data.datasets["alpha"] = QuantityDataSet(
        axes=[Q_(freq_axis, "THz")],
        data=Q_(alpha_,"1/cm"),
        axes_labels=["Frequency"],
        data_label="Absorption coefficient"
    )

    return result_data
