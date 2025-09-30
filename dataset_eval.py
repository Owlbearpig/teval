from dataset import DataSet, Domain
from functions import window
from scipy.optimize import shgo
from functools import partial
import numpy as np
from consts import eps0_thz, c_thz
import matplotlib.pyplot as plt
import logging
from scipy.signal import iirnotch, filtfilt
from numpy import polyfit
from functions import phase_correction

# logging.basicConfig(level=logging.WARNING)

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

    def _model_1layer(self, freqs, n3_):
        d = self.options["sample_properties"]["d_1"]
        shift_ = self.options["eval_opt"]["shift_sub"]
        nfp = self.options["eval_opt"]["nfp"]

        n1 = 1
        w_ = 2 * np.pi * freqs
        t_as = 2 * n1 / (n1 + n3_)
        t_sa = 2 * n3_ / (n1 + n3_)
        r_as = (n1 - n3_) / (n1 + n3_)
        r_sa = (n3_ - n1) / (n1 + n3_)

        exp = np.exp(1j * (d * w_ / c_thz) * n3_)

        if nfp == "inf":
            fp_factor = 1 / (1 + r_as * r_sa * exp ** 2)
        else:
            fp_factor = 0
            for fp_idx in range(0, nfp+1):
                fp_factor += (r_as ** 2 * exp ** 2) ** fp_idx

        e_sam = t_as * t_sa * exp * fp_factor
        e_ref = np.exp(1j * (d * w_ / c_thz))

        phase_shift = np.exp(1j * shift_ * 1e-3 * w_)

        t = phase_shift * e_sam / e_ref

        return np.nan_to_num(t)

    def _model_2layer(self, freq, n_sub, n_film):
        # n_sub += 0.01
        n1, n4 = 1, 1
        d = self.options["sample_properties"]["d_2"]
        h = self.options["sample_properties"]["d_film"]
        shift_ = self.options["eval_opt"]["shift_film"]
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

        t = phase_shift * e_sam / e_ref

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

    def _opt_fun_1layer(self, x_, freq_, t_exp_, bounds_=None):
        if bounds_ is not None:
            for i, p_ in enumerate(x_):
                if not (bounds_[i][0] <= p_) * (p_ <= bounds_[i][1]):
                    return np.inf

        n = x_[0] + 1j * x_[1]
        t_mod = self._model_1layer(freq_, n)

        abs_loss = (np.abs(t_mod) - np.abs(t_exp_)) ** 2
        ang_loss = (np.angle(t_mod) - np.angle(t_exp_)) ** 2

        return abs_loss + ang_loss

    def _opt_fun_2layer(self, x_, freq, n_sub_, t_exp_):
        #if x_[0] < 1 or x_[1] < 0:
        #    return np.inf

        n_film_ = x_[0] + 1j * x_[1]

        t_mod = self._model_2layer(freq, n_sub=n_sub_, n_film=n_film_)

        mag = (np.abs(t_mod) - np.abs(t_exp_))**2
        ang = (np.angle(t_mod) - np.angle(t_exp_))**2

        # real_part = (t_mod.real - t_exp_.real) ** 2
        # imag_part = (t_mod.imag - t_exp_.imag) ** 2

        return mag + ang

    def _gof(self, n_film_, n_sub_, t_exp_):
        f0, f1 = self.options["eval_opt"]["fit_range_film"]
        freq_mask = (f0 <= self.freq_axis) * (self.freq_axis <= f1)

        s = 0
        for f_idx in np.arange(len(self.freq_axis))[freq_mask]:
            n_f = (n_film_[f_idx].real, n_film_[f_idx].imag)
            s += self._opt_fun_2layer(n_f, self.freq_axis[f_idx], n_sub_[f_idx], t_exp_[f_idx])

        return s / len(self.freq_axis[freq_mask])

    def _fit_2layer(self, t_exp_, n_sub_):
        bounds_ = [(1, 15), (1, 15)]
        min_kwargs = {"method": "Nelder-Mead",
                      "options": {
                          #"maxev": np.inf,
                          "maxiter": 400,
                          "tol": 1e-6,
                          "fatol": 1e-6,
                          "xatol": 1e-6,
                      }
                      }
        shgo_options = {
            # "maxfev": np.inf,
            # "f_tol": 1e-12,
            # "maxiter": 50,
            # "ftol": 1e-12,
            # "xtol": 1e-12,
            # "maxev": 4000,
            # "minimize_every_iter": True,
            "disp": False
        }

        f0, f1 = self.options["eval_opt"]["fit_range_film"]
        freq_mask = (f0 <= self.freq_axis) * (self.freq_axis <= f1)

        best_ = (None, np.inf)
        n_opt_best = None
        for d_sub in [650]:#[*np.arange(640.00, 655, 5.0)]: # [656.8]:# 650
            for shift in [30]:  # [61.1]: # 30
                # self.options["sample_properties"]["d_film"] = d_film
                self.options["sample_properties"]["d_2"] = d_sub
                self.options["eval_opt"]["shift_film"] = shift

                gof = 0
                n_opt = np.zeros_like(self.freq_axis, dtype=complex)
                for f_idx in np.arange(len(self.freq_axis))[freq_mask]:
                    args_ = (self.freq_axis[f_idx], n_sub_[f_idx], t_exp_[f_idx])

                    opt_res_ = shgo(self._opt_fun_2layer,
                                    bounds=bounds_,
                                    minimizer_kwargs=min_kwargs,
                                    options=shgo_options,
                                    args=args_,
                                    iters=2,
                                    )
                    n_opt[f_idx] = opt_res_.x[0] + opt_res_.x[1] * 1j
                    # print(self.freq_axis[f_idx], n_opt[f_idx])
                    gof += opt_res_.fun

                n_imag = n_opt.imag
                # n_imag = n_imag - np.mean(n_imag[freq_mask])

                win_settings = {"en_plot": False, "fig_label": "FFT",
                                "win_start": 1.00*f0, "win_width": 1.00*(f1 - f0), }
                n_imag = window(np.array([self.freq_axis, n_imag]).T, **win_settings)
                n_imag = n_imag[:, 1]

                fft_ = np.fft.rfft(n_imag[freq_mask])
                fft_freq_axis = np.fft.rfftfreq(len(n_imag[freq_mask]),
                                                d=np.mean(np.diff(self.freq_axis)))
                # mask_ = (10 < fft_freq_axis) * (fft_freq_axis < 20)

                mask_ = (10 < fft_freq_axis) * (fft_freq_axis < 20)
                peak_val, peak_idx = np.max(np.abs(fft_[mask_])), np.argmax(np.abs(fft_[mask_]))

                gof = gof / np.sum(freq_mask)
                print("Film:", peak_val, shift, d_sub, gof)

                #plt.figure("TEST3")
                #plt.plot(fft_freq_axis, np.abs(fft_), label=f"shift {shift}")


                fs = 1 / np.mean(np.diff(self.freq_axis))
                Q = 0.3  # quality factor: higher = narrower # 0.5 def

                peak_freq = fft_freq_axis[mask_][peak_idx]
                # print(peak_freq, fs)
                b, a = iirnotch(peak_freq / (fs / 2), Q)

                #n_opt.real = filtfilt(b, a, n_opt.real)
                #n_opt.imag = filtfilt(b, a, n_opt.imag)

                #plt.figure("TEST2")
                #plt.plot(n_imag[freq_mask], label="Before filter")
                #plt.plot(n_opt.imag, label="After filter")

                if peak_val < best_[1]:
                    best_ = ((shift, d_sub), peak_val, gof)
                    n_opt_best = n_opt

        print("Best film: ", best_)
        return n_opt_best

    def _fit_1layer(self, t_exp_):
        bounds = [(3.05, 3.12), (0.000, 0.015)]
        #bounds = [(3.05, 3.13), (0.0025, 0.0050)]
        min_kwargs = {"method": "Nelder-Mead",
                      "options": {
                          #"maxev": np.inf,
                          #"maxiter": 4000,
                          "tol": 1e-12,
                          "fatol": 1e-12,
                          "xatol": 1e-12,
                      }
                      }

        #t_exp_phi_ = phase_correction(self.freq_axis, np.unwrap(np.angle(t_exp_)), en_plot=True)
        #t_exp_ = np.abs(t_exp_) * np.exp(1j * t_exp_phi_)

        f0, f1 = self.options["eval_opt"]["fit_range_sub"]
        freq_mask = (f0 <= self.freq_axis) * (self.freq_axis <= f1)

        shift = 15.3 # 15.3  # 30
        best_ = ((None, None), np.inf)
        n_opt_best = None
        for d_sub in [639.5]:#[639.5]:#[*np.arange(639, 640, 0.5)]:# [*np.arange(637, 643, 1)]:#[*np.arange(639.5, 640, 0.1)]: # 639 (Teralyzer)
            for shift in [-3]:#[-3]:#[*np.arange(-5, 5, 1)]:#[*np.arange(-3, 3, 1)]:#[*np.arange(-5, 4, 1)]: # [15.6]: # 28 # 22
                self.options["sample_properties"]["d_1"] = d_sub
                self.options["eval_opt"]["shift_sub"] = shift
                gof = 0
                n_opt = np.zeros_like(self.freq_axis, dtype=complex)
                for f_idx in np.arange(len(self.freq_axis))[freq_mask]:

                    cost = partial(self._opt_fun_1layer,
                                   freq_=self.freq_axis[f_idx],
                                   t_exp_=t_exp_[f_idx],
                                   bounds_=bounds)

                    opt_res_ = shgo(cost, bounds, minimizer_kwargs=min_kwargs, iters=3)
                    n_opt[f_idx] = opt_res_.x[0] + opt_res_.x[1] * 1j
                    gof += opt_res_.fun

                n_imag = n_opt.imag
                n_imag_mean = np.mean(n_imag[freq_mask])

                win_settings = {"en_plot": False, "win_start": f0, "win_width": f1-f0, "fig_label": "FFT"}
                n_imag = window(np.array([self.freq_axis, n_imag]).T, **win_settings)
                n_imag = n_imag[:, 1]

                fft_ = np.fft.rfft(n_imag[freq_mask] - n_imag_mean)
                fft_freq_axis = np.fft.rfftfreq(len(n_imag[freq_mask]),
                                                d=np.mean(np.diff(self.freq_axis)))

                # mask_ = (10 < fft_freq_axis) * (fft_freq_axis < 20)
                mask_ = (11 <= fft_freq_axis)
                peak_val, peak_idx = np.max(np.abs(fft_[mask_])), np.argmax(np.abs(fft_[mask_]))
                sum_val = np.sum(np.abs(fft_))
                gof = gof / np.sum(freq_mask)
                print("Sub:", sum_val, peak_val, shift, d_sub, gof)

                plt.figure("TESTFFT")
                plt.plot(fft_freq_axis, np.abs(fft_), label=f"shift {shift}")

                fs = 1/np.mean(np.diff(self.freq_axis))
                Q = 0.5  # quality factor: higher = narrower

                peak_freq = fft_freq_axis[mask_][peak_idx]
                # print(peak_freq, fs)
                b, a = iirnotch(peak_freq / (fs / 2), Q)

                #n_opt.real = filtfilt(b, a, n_opt.real)
                #n_opt.imag = filtfilt(b, a, n_opt.imag)

                # plt.figure("TEST2")
                # plt.plot(n_imag[freq_mask], label="Before filter")
                # plt.plot(n_opt.imag, label="After filter")

                plt.figure("n_sub")
                plt.plot(self.freq_axis, n_opt.real, label=shift)
                plt.plot(self.freq_axis, n_opt.imag, label=shift)

                opt_val = peak_val
                if opt_val < best_[1]:
                    best_ = ((shift, d_sub), opt_val)
                    n_opt_best = n_opt

        print("Best sub: ", best_)
        self.options["sample_properties"]["d_1"] = best_[0][1]
        self.options["eval_opt"]["shift_sub"] = best_[0][0]

        return n_opt_best

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

    def phase_fit_(self, freq, phi):
        z = polyfit(freq, phi, 1)
        plt.figure("TEST")
        plt.plot(freq, z[0]*freq + z[1])
        print(z)

    def transmission_sim(self):
        if not self.options["sim_opt"]["enabled"]:
            exit("BB")

        self.options["eval_opt"]["shift_sub"] = self.options["sim_opt"]["shift_sim"]
        n_sub = self.options["sim_opt"]["n_sub"]
        nfp_og = self.options["eval_opt"]["nfp"]

        self.options["eval_opt"]["nfp"] = nfp_og
        t_sim = np.zeros_like(self.freq_axis, dtype=complex)
        for f_idx, freq in enumerate(self.freq_axis):
            t_sim[f_idx] = self._model_1layer(freq, n_sub)
        t_sim = np.abs(t_sim) * np.exp(-1j * np.angle(t_sim))

        self.options["eval_opt"]["nfp"] = nfp_og

        return t_sim

    def ref_std(self, en_plot=False):
        all_refs = self.measurements["refs"]
        ref_data = np.zeros((len(all_refs), len(self.freq_axis)), dtype=complex)
        for ref_idx, ref_meas in enumerate(all_refs):
            ref_fd = self._get_data(ref_meas, domain=Domain.Frequency)
            ref_data[ref_idx] = ref_fd[:, 1]

        freq_range = (0.35 < self.freq_axis)*(self.freq_axis < 4.0)
        amp_argmin = np.argmin(np.abs(ref_data[:, freq_range]))
        amp_argmin = np.unravel_index(amp_argmin, ref_data[:, freq_range].shape)[0]
        amp_min = ref_data[amp_argmin]

        amp_argmax = np.argmax(np.abs(ref_data[:, freq_range]))
        amp_argmax = np.unravel_index(amp_argmax, ref_data[:, freq_range].shape)[0]
        amp_max = ref_data[amp_argmax]

        amp_mean, amp_std = np.mean(np.abs(ref_data), axis=0), np.std(np.abs(ref_data), axis=0)
        phi_mean, phi_std = np.mean(np.angle(ref_data), axis=0), np.std(np.angle(ref_data), axis=0)

        def dB(y):
            return 20*np.log10(y)

        if en_plot:
            plt.figure("Ref Standard deviation")
            #plt.plot(self.freq_axis, dB(np.abs(amp_min)), label="Min")
            #plt.plot(self.freq_axis, dB(amp_mean), label="Mean")
            #plt.plot(self.freq_axis, dB(np.abs(amp_max)), label="Max")
            plt.plot(self.freq_axis, amp_std, label="Std")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Amplitude (dB)")

        return amp_std

    def meas_sim(self):
        t_sim = self.transmission_sim()

        ref1_fd, ref1_meas = self.get_ref_data(Domain.Frequency, ref_idx=10, ret_meas=True)
        ref2_fd, ref2_meas = self.get_ref_data(Domain.Frequency, ref_idx=10, ret_meas=True)

        meas_time_diff = (ref1_meas.meas_time - ref2_meas.meas_time).total_seconds()
        print("ref1 - ref2 measurement time difference (seconds): ", np.round(meas_time_diff, 2))

        ref_amp_std = self.ref_std(en_plot=False)

        t_sim_meas = t_sim * ref1_fd[:, 1] / ref2_fd[:, 1]

        ref_amp = np.abs(ref2_fd[:, 1])
        ref_phi = np.angle(ref2_fd[:, 1])
        t_sim_meas = t_sim * ref_amp * np.exp(1j * ref_phi) / ref1_fd[:, 1]

        return t_sim_meas



    def eval_point(self, pnt):
        f_axis = self.freq_axis
        plot_range = self.options["plot_range"]

        meas_sub = self.get_measurement(*self.options["eval_opt"]["sub_pnt"])
        meas_film = self.get_measurement(*pnt)
        sigma_exp = self._conductivity(meas_film)
        sigma_model = self.conductivity_model(sigma_exp)

        # t_exp_1layer = self.meas_sim()

        self.sub_dataset.options["pp_opt"]["window_opt"]["enabled"] = True
        self.sub_dataset.options["pp_opt"]["window_opt"]["en_plot"] = True
        self.sub_dataset.options["pp_opt"]["window_opt"]["fig_label"] = "sub"

        # single_layer_eval = self.sub_dataset.single_layer_eval(meas_sub, (0, 10))
        t_exp_1layer = self.sub_dataset.transmission(meas_sub, 1)
        # t_exp_1layer = self.transmission_sim()

        self.sub_dataset.options["pp_opt"]["window_opt"]["enabled"] = False
        phi = -np.unwrap(np.angle(t_exp_1layer))
        phi = phase_correction(self.freq_axis, phi, en_plot=True, fit_range=(0.3, 0.6))
        # phi -= 0.03
        t_exp_1layer = np.abs(t_exp_1layer) * np.exp(1j * phi)

        n_sub = self._fit_1layer(t_exp_1layer)

        self.options["pp_opt"]["window_opt"]["enabled"] = True
        self.options["pp_opt"]["window_opt"]["en_plot"] = True
        self.options["pp_opt"]["window_opt"]["fig_label"] = "film"
        t_exp_2layer = self.transmission(meas_film, 1)
        self.options["pp_opt"]["window_opt"]["enabled"] = False
        t_exp_2layer = np.abs(t_exp_2layer) * np.exp(-1j * (np.angle(t_exp_2layer)))

        n_film = self._fit_2layer(t_exp_2layer, n_sub)
        # self.plt_show()

        w = 2 * np.pi * np.array(self.freq_axis)
        sigma = 1e4 * 2 * eps0_thz * n_film ** 2 * w / (1 + 1j)  # S/cm

        t_mod_sub = self._model_1layer(self.freq_axis, n_sub)
        t_mod_film = self._model_2layer(self.freq_axis, n_sub=n_sub, n_film=n_film)

        f0, f1 = self.options["eval_opt"]["fit_range_sub"]
        f_mask = (f0 < self.freq_axis)*(self.freq_axis < f1)


        plt.figure("Conductivity(fit)")
        plt.plot(self.freq_axis, sigma.real, label="Real part")
        plt.plot(self.freq_axis, sigma.imag, label="Imaginary part")

        plt.figure("n_sub")
        plt.plot(self.freq_axis, n_sub.real, label="Real part")
        plt.plot(self.freq_axis, n_sub.imag, label="Imaginary part")
        # plt.plot(self.freq_axis, single_layer_eval["refr_idx"].imag, label="Imaginary part (1 layer eval)")
        plt.ylim((0, 0.012))
        plt.xlim(self.options["eval_opt"]["fit_range_sub"])
        plt.xlabel("Frequency (THz)")

        plt.figure("absorption coefficient")
        plt.plot(self.freq_axis, 4*np.pi*n_sub.imag*self.freq_axis/0.03)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Absorption coefficient (1/cm)")

        plt.figure("n_film")
        plt.plot(self.freq_axis, n_film.real, label="real")
        plt.plot(self.freq_axis, n_film.imag, label="imag")
        plt.xlabel("Frequency (THz)")

        plt.figure("Transmission fit abs film")
        plt.plot(self.freq_axis[f_mask], np.abs(t_exp_2layer[f_mask]), label="Experiment")
        plt.plot(self.freq_axis[f_mask], np.abs(t_mod_film[f_mask]), label="Model")
        plt.xlabel("Frequency (THz)")

        plt.figure("Transmission fit angle film")
        plt.plot(self.freq_axis[f_mask], np.angle(t_exp_2layer[f_mask]), label="Experiment")
        plt.plot(self.freq_axis[f_mask], np.angle(t_mod_film[f_mask]), label="Model")
        plt.xlabel("Frequency (THz)")

        plt.figure("TEST2")
        phi_1l =  np.unwrap(np.angle(t_exp_1layer[f_mask]))
        phi_2l = np.unwrap(np.angle(t_exp_2layer[f_mask]))
        plt.plot(self.freq_axis[f_mask], phi_1l, label="Experiment 1l")
        plt.plot(self.freq_axis[f_mask], phi_2l, label="Experiment 2l")
        #self.phase_fit_(self.freq_axis[f_mask], phi_1l)
        # self.phase_fit_(self.freq_axis[f_mask], phi_2l)

        plt.figure("Transmission fit abs sub")
        plt.plot(self.freq_axis[f_mask], np.log10(np.abs(t_exp_1layer[f_mask])), label="Experiment")
        plt.plot(self.freq_axis[f_mask], np.log10(np.abs(t_mod_sub[f_mask])), label="Model")

        plt.figure("Transmission fit abs sub dev")
        diff = (np.abs(t_exp_1layer[f_mask]) - np.abs(t_mod_sub[f_mask]))**2
        plt.plot(self.freq_axis[f_mask], np.log10(diff), label="Log squared difference")
        # plt.plot(self.freq_axis[f_mask], n_sub.imag[f_mask] / np.max(n_sub.imag[f_mask]), label="n_sub.imag")

        plt.figure("Transmission fit angle sub")
        plt.plot(self.freq_axis[f_mask], np.angle(t_exp_1layer[f_mask]), label="Experiment")
        plt.plot(self.freq_axis[f_mask], np.angle(t_mod_sub[f_mask]), label="Model")

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


