import numpy as np
import logging
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from common.consts import c_thz
from common.eval_component.n_guess import get_n_guess
from common.eval_component.eval_result import SingleResultData
from common.units import Q_
from common.eval_component.quantity_set import QuantityDataSet
from datetime import datetime


def refine_n_per_frequency(n, freq_axis, t_exp, transmission_model, model_kwargs, max_iter=50, max_step=0.01,
                           tol=1e-10):
    # Newton on Re/Im of the model residual per frequency (2x2 real system, so conjugated models work too),
    # started from the spline solution to stay on its branch. Returns n and the per frequency convergence mask
    def residual(n_):
        return transmission_model(n_, freq_axis, **model_kwargs) - t_exp

    n_ref = n.copy()
    h = 1e-7
    step_size = np.full(n.shape, np.inf)
    for _ in range(max_iter):
        r = residual(n)
        dr_dn = (residual(n + h) - residual(n - h)) / (2 * h)
        dr_dk = (residual(n + 1j * h) - residual(n - 1j * h)) / (2 * h)

        det = dr_dn.real * dr_dk.imag - dr_dk.real * dr_dn.imag
        det = np.where(np.abs(det) < 1e-30, np.nan, det)
        step_n = (dr_dk.imag * r.real - dr_dk.real * r.imag) / det
        step_k = (dr_dn.real * r.imag - dr_dn.imag * r.real) / det
        step = np.nan_to_num(step_n + 1j * step_k)

        step_size = np.abs(step)
        step = np.where(step_size > max_step, max_step * step / np.maximum(step_size, 1e-30), step)
        n = n - step
        if np.max(step_size) < tol:
            break

    worse = ~np.isfinite(n) | (np.abs(residual(n)) > np.abs(residual(n_ref)))
    n[worse] = n_ref[worse]
    # the finite difference Jacobian limits the final step to ~h, a nonzero det marks a regular solution
    converged = ~worse & (step_size < 1e-6) & np.isfinite(det)

    return n, converged

def spline_transmission_optimization(d, shift, config_dict):
    freq_axis = np.asarray(config_dict["freq_axis"])
    n0 = get_n_guess(d, shift, config_dict)
    t_exp = np.asarray(config_dict["t_exp"])
    transmission_model = config_dict["transmission_model"]
    cost_fun = config_dict["cost_fun"]
    n_knots = config_dict["knot_count"]
    lambda_n = config_dict["reg_n"]
    lambda_k = config_dict["reg_k"]
    max_nfev = config_dict["max_nfev"]

    model_kwargs_keys = ["n_sub", "n1", "n4", "h", "nfp", "layers"]
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

    unit_splines = CubicSpline(freq_knots, np.eye(n_knots), bc_type="natural")
    basis = unit_splines(freq_axis)
    basis_d2 = unit_splines(freq_axis, 2)

    # ||basis_d2 @ p|| == ||r_d2 @ p||, the K x K factor keeps the Jacobian (and its SVD) small
    r_d2 = np.linalg.qr(basis_d2, mode="r")
    reg_n_mat = np.sqrt(lambda_n) / n_scale * r_d2
    reg_k_mat = np.sqrt(lambda_k) / k_scale * r_d2
    reg_jac = np.block([[reg_n_mat, np.zeros_like(r_d2)],
                        [np.zeros_like(r_d2), reg_k_mat]])

    def calculate_model(p):
        n = basis @ p[:n_knots] + 1j * (basis @ p[n_knots:])
        t_mod = transmission_model(n, freq_axis, **model_kwargs)

        return n, t_mod

    def residual_fun(p):
        n, t_mod = calculate_model(p)

        data_residual = cost_fun(t_exp, t_mod)

        residuals = np.concatenate([data_residual,
                                    reg_n_mat @ p[:n_knots],
                                    reg_k_mat @ p[n_knots:],
        ])

        return residuals

    def jac_fun(p):
        n = basis @ p[:n_knots] + 1j * (basis @ p[n_knots:])
        h = 1e-7

        def cost_at(n_):
            return cost_fun(t_exp, transmission_model(n_, freq_axis, **model_kwargs))

        dc_dn = (cost_at(n + h) - cost_at(n - h)) / (2 * h)
        dc_dk = (cost_at(n + 1j * h) - cost_at(n - 1j * h)) / (2 * h)

        data_jac = np.hstack((dc_dn[:, None] * basis, dc_dk[:, None] * basis))

        return np.vstack((data_jac, reg_jac))

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
        jac=jac_fun,
        bounds=(lower_bounds, upper_bounds),
        x_scale="jac",
        max_nfev = max_nfev,
    )

    n_opt_res, t_mod_res = calculate_model(opt_res.x)
    n_spline = n_opt_res
    converged = opt_res.success
    if config_dict.get("per_freq_refine", False):
        # the spline fit is only the start value here, hitting max_nfev doesn't matter
        n_opt_res, converged_f = refine_n_per_frequency(n_spline, freq_axis, t_exp, transmission_model, model_kwargs)
        converged = bool(np.all(converged_f))
        t_mod_res = transmission_model(n_opt_res, freq_axis, **model_kwargs)

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
    result_data.optimization_info["converged"] = converged
    result_data.optimization_info["spline_converged"] = opt_res.success

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

    if config_dict.get("per_freq_refine", False):
        result_data.datasets["n_spline"] = QuantityDataSet(
            axes=[Q_(freq_axis, "THz")],
            data=Q_(n_spline, ""),
            axes_labels=["Frequency"],
            data_label="Refractive index (spline)"
        )

    result_data.datasets["n"] = QuantityDataSet(
        axes=[Q_(freq_axis, "THz")],
        data=Q_(n_opt_res,""),
        axes_labels=["Frequency"],
        data_label="Refractive index"
    )

    result_data.datasets["alpha"] = QuantityDataSet(
        axes=[Q_(np.real(freq_axis), "THz")],
        data=Q_(np.real(alpha_),"1/cm"),
        axes_labels=["Frequency"],
        data_label="Absorption coefficient"
    )

    return result_data
