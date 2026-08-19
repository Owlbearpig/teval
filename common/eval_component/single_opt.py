import numpy as np
from common.eval_component.shgo import shgo
import time
from common.consts import c_thz

def optimize_transmission(d, shift, config_dict):
    f_idx_range_ = config_dict["f_idx_range_"]
    freq_axis = config_dict["freq_axis"]
    n0_ = config_dict["n_guess"]
    t_exp = config_dict["t_exp"]
    transmission_model = config_dict["transmission_model"]
    cost_fun = config_dict["cost_fun"]
    minimizer_kwargs = config_dict["minimizer_kwargs"]
    shgo_options = config_dict["shgo_options"]

    freq_axis_slice = freq_axis[f_idx_range_]
    n0_ = n0_[f_idx_range_]
    t_exp_ = t_exp[f_idx_range_, 1]

    time.sleep(1000/(min(500, d)))

    model_kwargs_keys = ["d", "n_sub", "n1", "n4", "h", "nfp"]
    model_kwargs = {k: config_dict[k] for k in model_kwargs_keys if k in config_dict}
    model_kwargs["shift"] = shift

    gof = 0
    n_opt_res_ = np.zeros_like(freq_axis_slice, dtype=complex)
    for f_idx, f_ in enumerate(freq_axis_slice):
        def opt_fun(p):
            n = p[0] + 1j * p[1]
            t_mod = transmission_model(n, f_, **model_kwargs)
            return cost_fun(t_exp_[f_idx], t_mod)

        n0_f_idx = n0_[f_idx]
        n_min, n_max = 0.90 * n0_f_idx.real, 1.10 * n0_f_idx.real
        k_min, k_max = 0.10 * n0_f_idx.imag, 1.10 * n0_f_idx.imag
        bounds = [(n_min, n_max), (k_min, k_max)]

        conv, i_ = False, 0
        while not conv:
            i_ += 1
            """
            shgo_opt_res_ = shgo(opt_fun,
                                 bounds=bounds,
                                 minimizer_kwargs=minimizer_kwargs,
                                 options=shgo_options,
                                 # n=1, iters=200,
                                 )
            
            x = shgo_opt_res_.x
            gof += shgo_opt_res_.fun
            """
            x = [1, 1]
            n_opt_res_[f_idx] = x[0] + 1j * x[1]

            if f_idx == 0:
                break

            # diff = (n_opt_res_[f_idx].real - n_opt_res_[f_idx - 1].real)
            diff = 0
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

    alpha_ = freq_axis_slice * 4 * np.pi * n_opt_res_.imag / (1e-4 * c_thz)

    return {
        "d": d, "shift": shift, "freq_axis": freq_axis_slice, "gof": gof / len(freq_axis_slice),
        "n": n_opt_res_.real, "k": n_opt_res_.imag, "alpha": alpha_, "n0_real": n0_.real, "n0_imag": n0_.imag,
    }

