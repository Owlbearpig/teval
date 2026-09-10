import numpy as np
from common.eval_component.shgo import shgo
import time
from common.consts import c_thz
from common.eval_component.eval_result import SingleResultData
from common.units import Q_
from common.eval_component.quantity_set import QuantityDataSet
from datetime import datetime

def optimize_transmission(d, shift, config_dict) -> SingleResultData:
    freq_axis = config_dict["freq_axis"]
    n0 = config_dict["n_guess"]
    t_exp = config_dict["t_exp"]
    transmission_model = config_dict["transmission_model"]
    cost_fun = config_dict["cost_fun"]
    minimizer_kwargs = config_dict["minimizer_kwargs"]
    shgo_options = config_dict["shgo_options"]

    model_kwargs_keys = ["n_sub", "n1", "n4", "h", "nfp"]
    model_kwargs = {k: config_dict[k] for k in model_kwargs_keys if k in config_dict}
    model_kwargs["shift"] = shift
    model_kwargs["d"] = d

    gof = 0
    convergence_results = np.zeros_like(freq_axis, dtype=bool)
    n_opt_res_ = np.zeros_like(freq_axis, dtype=complex)
    for f_idx, freq in enumerate(freq_axis):
        def opt_fun(p):
            n = p[0] + 1j * p[1]
            t_mod = transmission_model(n, freq, **model_kwargs)
            return cost_fun(t_exp[f_idx, 1], t_mod)

        n0_f_idx = n0[f_idx, 1]
        n_min, n_max = 0.90 * n0_f_idx.real, 1.10 * n0_f_idx.real
        k_min, k_max = 0.10 * n0_f_idx.imag, 1.10 * n0_f_idx.imag
        bounds = [(n_min, n_max), (k_min, k_max)]

        conv, i_ = False, 0
        while not conv:
            i_ += 1
            shgo_opt_res_ = shgo(opt_fun,
                                 bounds=bounds,
                                 #minimizer_kwargs=minimizer_kwargs,
                                 #options=shgo_options,
                                 n=1, iters=20,
                                 )
            
            x = shgo_opt_res_.x
            gof += shgo_opt_res_.fun
            convergence_results[f_idx] = shgo_opt_res_.success
            n_opt_res_[f_idx] = x[0] + 1j * x[1]

            # n_opt_res_[f_idx] = n0_f_idx
            # x = np.random.random(2)
            # n_opt_res_[f_idx] = x[0] + 0.001 * 1j * x[1]
            # n_opt_res_[f_idx] = 1.5 + 1j * 0.015

            if f_idx == 0:
                break

            diff = (n_opt_res_[f_idx] - n_opt_res_[f_idx - 1]).real
            if np.abs(diff) < 0.10:
                conv = True
            else:
                c0, c1 = 0.90 + i_ * 0.02, 1.10 - i_ * 0.02
                n_bounds = (n0_f_idx.real * c0, n0_f_idx.real * c1)
                k_bounds = (n0_f_idx.imag * c0, n0_f_idx.imag * c1)

                bounds = [(min(n_bounds), max(n_bounds)), (min(k_bounds), max(k_bounds))]
            if i_ > 4:
                n_prev = n_opt_res_[f_idx - 1]
                c0, c1 = 0.90 + i_ * 0.01, 1.10 - i_ * 0.01
                n_bounds = (n_prev.real * c0, n_prev.real * c1)
                k_bounds = (n_prev.imag * c0, n_prev.imag * c1)

                bounds = [(min(n_bounds), max(n_bounds)), (min(k_bounds), max(k_bounds))]
            if i_ > 5:
                break

    alpha_ = freq_axis * 4 * np.pi * n_opt_res_.imag / (1e-4 * c_thz)

    result_data = SingleResultData()
    result_data.d = Q_(d, "µm")
    result_data.shift = Q_(shift, "µm")
    result_data.freq_axis = Q_(freq_axis, "THz")
    result_data.optimization_info["gof"] = Q_(gof / len(freq_axis), "")
    result_data.optimization_info["converged"] = np.all(convergence_results)
    result_data.optimization_info["timestamp"] = str(datetime.now().isoformat())
    result_data.datasets["t_exp"] = QuantityDataSet(axes=[Q_(freq_axis, "THz")],
                                                    data=Q_(t_exp[:, 1], ""),
                                                    axes_labels=["Frequency"],
                                                    data_label="Transmission coefficient experimental")
    result_data.datasets["n0"] = QuantityDataSet(axes=[Q_(freq_axis, "THz")],
                                                 data=Q_(n0[:, 1], ""),
                                                 axes_labels=["Frequency"],
                                                 data_label="Refractive index")
    result_data.datasets["n"] = QuantityDataSet(axes=[Q_(freq_axis, "THz")],
                                                data=Q_(n_opt_res_, ""),
                                                axes_labels=["Frequency"],
                                                data_label="Refractive index")
    result_data.datasets["alpha"] = QuantityDataSet(axes=[Q_(freq_axis, "THz")],
                                                data=Q_(alpha_, "1/cm"),
                                                axes_labels=["Frequency"],
                                                data_label="Absorption coefficient")

    return result_data

