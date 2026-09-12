import numpy as np

from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

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

    model_kwargs_keys = ["n_sub", "n1", "n4", "h", "nfp"]
    model_kwargs = {k: config_dict[k] for k in model_kwargs_keys if k in config_dict}
    model_kwargs["shift"] = shift
    model_kwargs["d"] = d

    freq = freq_axis
    y_meas = t_exp[:, 1]

    n0_complex = n0[:, 1]

    sort_idx = np.argsort(freq)

    freq = freq[sort_idx]
    y_meas = y_meas[sort_idx]
    n0_complex = n0_complex[sort_idx]

    knot_idx = np.linspace(0, len(freq) - 1, n_knots, dtype=int)

    freq_knots = freq[knot_idx]

    n_real_0 = n0_complex.real[knot_idx]
    kappa_0 = n0_complex.imag[knot_idx]

    p0 = np.concatenate([n_real_0, kappa_0])
    n0_real_knots = n0_complex.real[knot_idx]
    n0_imag_knots = n0_complex.imag[knot_idx]

    n_lower = 0.90 * n0_real_knots
    n_upper = 1.10 * n0_real_knots

    k_lower = 0.10 * n0_imag_knots
    k_upper = 1.10 * n0_imag_knots

    lower_bounds = np.concatenate([n_lower, k_lower])
    upper_bounds = np.concatenate([n_upper, k_upper])

    p0 = np.clip(p0, lower_bounds + 1e-12, upper_bounds - 1e-12)

    n_scale = max(np.ptp(n0_complex.real), np.mean(np.abs(n0_complex.real)) * 0.01, 1e-6)
    k_scale = max(np.ptp(n0_complex.imag), np.mean(np.abs(n0_complex.imag)) * 0.01, 1e-6)

    def make_splines(p):
        n_knots_values = p[:n_knots]
        k_knots_values = p[n_knots:]

        spline_n = CubicSpline(freq_knots, n_knots_values, bc_type="natural")
        spline_k = CubicSpline(freq_knots, k_knots_values, bc_type="natural")

        return spline_n, spline_k

    def calculate_model(p):
        spline_n, spline_k = make_splines(p)

        n_real = spline_n(freq)
        k = spline_k(freq)

        n_complex = n_real + 1j * k
        t_mod = transmission_model(n_complex, freq, **model_kwargs)

        return n_complex, t_mod

    def residual_fun(p):
        n_complex, t_mod = calculate_model(p)

        #amplitude_residual = (np.abs(y_meas) - np.abs(t_mod))
        #phase_residual = np.angle(y_meas / t_mod)
        residual = cost_fun(y_meas, t_mod)

        spline_n, spline_k = make_splines(p)

        d2n = spline_n(freq, 2)
        d2k = spline_k(freq, 2)

        d2n_normalized = d2n / n_scale
        d2k_normalized = d2k / k_scale

        residuals = np.concatenate([residual,
                                    np.sqrt(lambda_n) * d2n_normalized,
                                    np.sqrt(lambda_k) * d2k_normalized,
        ])

        return residuals

    opt_res = least_squares(
        residual_fun,
        p0,
        bounds=(lower_bounds, upper_bounds),
        x_scale="jac",
        verbose=0,
    )

    n_opt_sorted, t_mod_sorted = calculate_model(opt_res.x)
    inverse_sort = np.argsort(sort_idx)

    n_opt_res_ = n_opt_sorted[inverse_sort]

    t_mod = t_mod_sorted[inverse_sort]

    gof = np.sum(cost_fun(y_meas, t_mod_sorted))

    alpha_ = (freq_axis * 4 * np.pi * n_opt_res_.imag / (1e-4 * c_thz))
    result_data = SingleResultData()

    result_data.d = Q_(d, "µm")
    result_data.shift = Q_(shift, "fs")

    result_data.freq_axis = Q_(freq_axis, "THz")

    result_data.optimization_info["gof"] = Q_(gof / len(freq_axis), "")

    result_data.optimization_info["converged"] = opt_res.success
    result_data.optimization_info["timestamp"] = str(datetime.now().isoformat())
    result_data.optimization_info["optimizer_message"] = opt_res.message

    result_data.optimization_info["n_knots"] = n_knots
    result_data.optimization_info["lambda_n"] = lambda_n
    result_data.optimization_info["lambda_k"] = lambda_k

    result_data.datasets["t_exp"] = QuantityDataSet(
        axes=[Q_(freq_axis, "THz")],
        data=Q_(t_exp[:, 1],""),
        axes_labels=["Frequency"],
        data_label="Transmission coefficient experimental"
    )

    result_data.datasets["n0"] = QuantityDataSet(
        axes=[Q_(freq_axis, "THz")],
        data=Q_(n0[:, 1],""),
        axes_labels=["Frequency"],
        data_label="Refractive index"
    )

    result_data.datasets["n"] = QuantityDataSet(
        axes=[Q_(freq_axis, "THz")],
        data=Q_(n_opt_res_,""),
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
