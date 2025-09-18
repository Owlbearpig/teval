from dataset import DataSet
from functions import check_dict_values
from scipy.optimize import shgo
from functools import partial
import numpy as np
from consts import eps0_thz, c_thz
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.WARNING)

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
        tau /= 2*np.pi
        w = 2 * np.pi * freq_
        return sig0 / (1 + 1j * tau * w)

    def _drude2(self, freq_, tau, sig0, wp, eps_inf):
        tau *= 1e-3
        tau /= 2 * np.pi
        w = 2 * np.pi * freq_

        return sig0 * wp**2 / (tau - 1j * w) - 1j * eps0_thz * w * (eps_inf - 1)

    def _lattice_contrib(self, freq_, tau, wp, eps_inf, eps_s):
        tau *= 1e-3
        # tau *= 2 * np.pi
        wp *= 2 * np.pi
        w = 2 * np.pi * freq_
        eps_l = eps_inf - (eps_s - eps_inf) * wp ** 2 / (w ** 2 - wp ** 2 - 1j * w / tau)
        sigma_l_ = 1e-2 * (1 - w * eps0_thz * eps_l) * 1j

        return sigma_l_

    def _total_response(self, freq, tau, sig0, wp, eps_s, c1=None):
        # [freq] = THz, [tau] = fs, [sig0] = S/cm, [wp] = THz. Dimensionless: eps_inf, eps_s
        eps_inf = 9
        tau = tau
        sig_cc = self._drude(freq, tau, sig0)
        # sig_cc = drude_smith(freq, tau, sig0, c1)
        sigma_l = self._lattice_contrib(freq, tau, wp, eps_inf, eps_s)

        model_ = sig_cc + sigma_l

        return model_.real + 1j * model_.imag

    def _model_2layer(self, freq, n_sub, n_film, shift_=0):
        # n_sub += 0.01
        n1, n4 = 1, 1
        d = self.options["sample_properties"]["d"]
        h = self.options["sample_properties"]["d_film"]
        w_ = 2 * np.pi * freq
        t12 = 2 * n1 / (n1 + n_film)
        t23 = 2 * n_film / (n_film + n_sub)
        t34 = 2 * n_sub / (n_sub + n4)

        r12 = (n1 - n_film) / (n1 + n_film)
        r23 = (n_film - n_sub) / (n_film + n_sub)
        r34 = (n_sub - n4) / (n_sub + n4)

        exp1 = np.exp(1j * (h * w_ / c_thz) * n_film)
        exp2 = np.exp(1j * (d * w_ / c_thz) * n_sub)

        e_sam_num = t12 * t23 * t34 * exp1 * exp2
        e_sam_den = 1 + r12 * r23 * exp1 ** 2 + r23 * r34 * exp2 ** 2 + r12 * r34 * exp1 ** 2 * exp2 ** 2
        e_sam = e_sam_num / e_sam_den

        e_ref = np.exp(1j * ((d + h) * w_ / c_thz))
        phase_shift = np.exp(1j * shift_ * 1e-3 * w_)  # 6, 16

        t = phase_shift * e_ref / e_sam

        return np.nan_to_num(t)

    def _opt_fun(self, x_, sigma_):
        # if any([p < 0 for p in x_]):
        #    return np.inf

        freq = self.freq_axis
        model = self._total_response
        # model = partial(self._drude2, eps_inf=9)
        # model = drude_smith
        f0, f1 = self.options["eval_opt"]["fit_range"]
        mask = (f0 <= freq) * (freq < f1)  # 2.2

        real_part = (model(freq, *x_).real - sigma_.real) ** 2
        imag_part = (model(freq, *x_).imag - sigma_.imag) ** 2

        return np.sum(real_part[mask] + imag_part[mask]) / (len(freq[mask]) * 1000)

    def _fit_conductivity_model(self, sigma_exp_):
        cost = partial(self._opt_fun, sigma_=sigma_exp_)

        min_kwargs = {"method": "Nelder-Mead",
                      "options": {
                          "maxev": np.inf,
                          "maxiter": 40000,
                          "tol": 1e-13,
                          "fatol": 1e-13,
                          "xatol": 1e-13,
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
                        # bounds=[(1, 800), (0.1, 10), (1, 800), (0, 100), (0, 100)],
                        # bounds=[(1, 80), (1, 30), (1, 8000), (0, 1000), (0, 1000)], # drude
                        bounds=[(1, 80), (1, 30), (1, 8000), (0, 1000)],
                        # bounds=[(1, 80), (1, 30), (1, 8000)],  # drude2
                        # bounds=[(1, 8000), (10, 150), (0.1, 100), (0, 100), (0, 100), (-1, 1)],
                        # bounds = [(1, 800), (10, 150), (-1, 1)],
                        # n=1, iters=200,
                        minimizer_kwargs=min_kwargs,
                        options=shgo_options,
                        )
        return opt_res_

    def _opt_fun_ri(self, x_, freq, n_sub_, t_exp_):
        if x_[0] < 1 or x_[1] < 0:
            return np.inf

        n_film_ = x_[0] + 1j * x_[1]

        t_mod = self._model_2layer(freq, n_sub=n_sub_, n_film=n_film_)

        mag = (np.abs(t_mod) - np.abs(t_exp_))**2
        ang = (np.angle(t_mod) - np.angle(t_exp_))**2

        # real_part = (t_mod.real - t_exp_.real) ** 2
        # imag_part = (t_mod.imag - t_exp_.imag) ** 2

        return mag + ang

    def _fit_refractive_index(self, t_exp_, sub_res):
        min_kwargs = {"method": "Nelder-Mead",
                      "options": {
                          "maxev": np.inf,
                          "maxiter": 400,
                          "tol": 1e-10,
                          "fatol": 1e-10,
                          "xatol": 1e-10,
                      }
                      }
        shgo_options = {
            "maxfev": np.inf,
            # "f_tol": 1e-12,
            "maxiter": 50,
            # "ftol": 1e-12,
            # "xtol": 1e-12,
            # "maxev": 4000,
            # "minimize_every_iter": True,
            "disp": False
        }

        f0, f1 = self.options["eval_opt"]["fit_range"]
        freq_mask = (f0 <= self.freq_axis) * (self.freq_axis < f1)

        n_opt = np.zeros_like(self.freq_axis, dtype=complex)
        for f_idx in np.arange(len(self.freq_axis))[freq_mask]:
            args_ = (self.freq_axis[f_idx], sub_res["refr_idx"][f_idx], t_exp_[f_idx])

            opt_res_ = shgo(self._opt_fun_ri,
                            bounds=[(1, 10), (1, 10)],
                            minimizer_kwargs=min_kwargs,
                            options=shgo_options,
                            args=args_,
                            )
            n_opt[f_idx] = opt_res_.x[0] + 1j * opt_res_.x[1]
            print(self.freq_axis[f_idx], opt_res_.x, opt_res_.fun)

        return n_opt


    def conductivity_model(self, sigma_exp):
        opt_res = self._fit_conductivity_model(sigma_exp)
        x = opt_res.x # x = [tau, sig0, wp, eps_inf, eps_s]
        print(x)
        # x = [1, 100, 4*np.pi, 10, 20]
        # x = [1, 100, 2, 16.8, 20] # ulatowski plot
        # x = [-1.588e-02,  5.044e+01,  920.8,  40.55, -43.13] # fit result
        # x = [-0.01588,  50.44,  920.8,  40.55, -43.13]
        # x = [40, 3, 50, 9]
        sigma_model_ = self._total_response(self.freq_axis, *x)
        # sigma_model_ = self._drude2(self.freq_axis, *x)

        return sigma_model_

    def eval_point(self, pnt):
        f_axis = self.freq_axis
        plot_range = self.options["plot_range"]

        meas = self.get_measurement(*pnt)
        sigma_exp = self._conductivity(meas)
        sigma_model = self.conductivity_model(sigma_exp)

        sub_res = self._eval_sub()
        t_exp = self.transmission(meas, 1)
        n_film = self._fit_refractive_index(t_exp, sub_res)

        t_mod = self._model_2layer(self.freq_axis, n_sub=sub_res["refr_idx"], n_film=n_film, shift_=0)
        f0, f1 = self.options["eval_opt"]["fit_range"]
        f_mask = (f0 < self.freq_axis)*(self.freq_axis < f1)

        plt.figure("n_opt")
        plt.plot(self.freq_axis, n_film.real, label="real")
        plt.plot(self.freq_axis, n_film.imag, label="imag")

        plt.figure("Transmission fit")
        plt.plot(self.freq_axis[f_mask], np.abs(t_exp[f_mask]), label="Experiment")
        plt.plot(self.freq_axis[f_mask], np.abs(t_mod[f_mask]), label="Model")

        plt.figure("Transmission fit angle")
        plt.plot(self.freq_axis[f_mask], np.angle(t_exp[f_mask]), label="Experiment")
        plt.plot(self.freq_axis[f_mask], np.angle(t_mod[f_mask]), label="Model")

    def plot_eval_res(self):
        pass
        """
        plt.figure("Conductivity")
        # plt.plot(f_axis[plot_range], sigma_exp[plot_range].real, label="Exp (real)")
        # plt.plot(f_axis[plot_range], sigma_exp[plot_range].imag, label="Exp (imag)")
        plt.plot(f_axis[plot_range], sigma_model[plot_range].real, label="Fit (real)")
        plt.plot(f_axis[plot_range], sigma_model[plot_range].imag, label="Fit (imag)")

        # plt.plot(f_axis, sigma_model.real, label="Fit (real)")
        # plt.plot(f_axis, sigma_model.imag, label="Fit (imag)")
        # plt.xlim((0.20, 2.55))
        # plt.ylim((-10, 200))
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Conductivity (S/cm)")
        """

if __name__ == "__main__":
    pass


