from dataset import DataSet
from functions import check_dict_values
from scipy.optimize import shgo
from functools import partial
import numpy as np
from scipy.constants import c, epsilon_0
import matplotlib.pyplot as plt


class DatasetEval(DataSet):

    def __init__(self, dataset_path, sub_dataset_path=None, options_=None):
        super().__init__(dataset_path, options_)
        self._link_sub_dataset(sub_dataset_path)

    def _link_sub_dataset(self, sub_dataset_path):
        if sub_dataset_path is None:
            return
        sub_dataset = DataSet(sub_dataset_path, self.options)
        self.link_sub_dataset(sub_dataset)

    def _drude(self, freq_, tau, sig0):
        scale = 1
        sig0 *= scale
        tau *= 1e-3
        w = 2 * np.pi * freq_
        return sig0 / (1 - 1j * tau * w)

    def _lattice_contrib(self, freq_, tau, wp, eps_inf, eps_s):
        tau *= 1e-3
        w = 2 * np.pi * freq_
        return eps_inf - (eps_s - eps_inf) * wp ** 2 / (w ** 2 - wp ** 2 + 1j * w / tau)

    def _total_response(self, freq, tau, sig0, wp, eps_inf, eps_s, c1=None):
        # [freq] = THz, [tau] = fs, [sig0] = S/cm, [wp] = THz. Dimensionless: eps_inf, eps_s
        sig_cc = self._drude(freq, tau, sig0)
        # sig_cc = drude_smith(freq, tau, sig0, c1)
        eps_L = self._lattice_contrib(freq, tau, wp, eps_inf, eps_s)

        w_ = 2 * np.pi * freq
        return ((1 - 1e12 * w_ * epsilon_0 * eps_L) * 1j + 100 * sig_cc) / 100

    def _opt_fun(self, x_, sigma_):
        # if any([p < 0 for p in x_]):
        #    return np.inf

        model = partial(self._total_response, freq=self.freq_axis)
        # model = drude_smith
        mask = (0.35 <= self.freq_axis) * (self.freq_axis < 2.0)  # 2.2
        real_part = (model(*x_).real - sigma_.real) ** 2
        imag_part = (model(*x_).imag - sigma_.imag) ** 2

        return np.sum(real_part[mask] + imag_part[mask]) / (len(self.freq_axis[mask]) * 1000)

    def _fit_conductivity_model(self, sigma_exp_):
        cost = partial(self._opt_fun, sigma_=sigma_exp_)

        min_kwargs = {"method": "Nelder-Mead",
                      "options": {
                          "maxev": np.inf,
                          "maxiter": 4000,
                          "tol": 1e-12,
                          "fatol": 1e-12,
                          "xatol": 1e-12,
                      }
                      }
        shgo_options = {
            # "maxfev": np.inf,
            # "f_tol": 1e-12,
            # "maxiter": 4000,
            # "ftol": 1e-12,
            # "xtol": 1e-12,
            # "maxev": 4000,
            # "minimize_every_iter": True,
            # "disp": True
        }
        opt_res_ = shgo(cost,
                        bounds=[(1, 800), (5, 150), (0.1, 100), (0, 100), (0, 100)],
                        # bounds=[(1, 8000), (10, 150), (0.1, 100), (0, 100), (0, 100), (-1, 1)],
                        # bounds = [(1, 800), (10, 150), (-1, 1)],
                        # n=1, iters=200,
                        minimizer_kwargs=min_kwargs,
                        options=shgo_options,
                        )
        return opt_res_

    def conductivity_model(self, sigma_exp):
        opt_res = self._fit_conductivity_model(sigma_exp)
        x = opt_res.x
        # x = [-1.588e-02,  5.044e+01,  920.8,  40.55, -43.13] # fit result
        # x = [-0.01588,  50.44,  920.8,  40.55, -43.13]
        sigma_model_ = self._total_response(*x)

        return sigma_model_

    def eval_point(self, pnt):
        f_axis = self.freq_axis
        plot_range = self.options["plot_range"]

        meas = self.get_measurement(*pnt)
        sigma_exp = self._conductivity(meas)
        sigma_model = self.conductivity_model(sigma_exp)

        plt.figure("Conductivity_")
        plt.plot(f_axis[plot_range], sigma_exp[plot_range].real, label="Exp (real)")
        plt.plot(f_axis[plot_range], sigma_exp[plot_range].imag, label="Exp (imag)")
        plt.plot(f_axis[plot_range], sigma_model[plot_range].real, label="Fit (real)")
        plt.plot(f_axis[plot_range], sigma_model[plot_range].imag, label="Fit (imag)")
        plt.xlim((0.20, 2.55))
        # plt.ylim((-10, 200))
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Conductivity (S/cm)")


if __name__ == "__main__":
    pass


