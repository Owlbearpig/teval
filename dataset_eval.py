from dataset import DataSet, Domain
from functions import window, do_ifft
from scipy.optimize import shgo
from functools import partial
import numpy as np
from consts import eps0_thz, c_thz
import matplotlib.pyplot as plt
import logging
from scipy.signal import iirnotch, filtfilt
from numpy import polyfit
from functions import phase_correction
from tmm_impl import coh_tmm
from enum import Enum

# logging.basicConfig(level=logging.WARNING)

class DataSetType(Enum):
    Main = "main"
    Sub = "sub"
    Other = "other"

class DatasetEval(DataSet):

    def __init__(self, dataset_path, sub_dataset_path=None, options_=None):
        super().__init__(dataset_path, options_)
        self._check_options()
        self._link_sub_dataset(sub_dataset_path)
        self._opt_consts = {}

    def _link_sub_dataset(self, sub_dataset_path, options=None):
        if sub_dataset_path is None:
            return
        if options is None:
            options = self.options

        sub_dataset = DataSet(sub_dataset_path, options)
        self.link_sub_dataset(sub_dataset)

    def _check_options(self):
        # TODO automate type cast. (list -> array)
        self.options["eval_opt"]["film_bounds"] = np.array(self.options["eval_opt"]["film_bounds"])

    def _get_dataset(self, which=DataSetType.Main):
        if which == DataSetType.Main:
            return self
        elif which == DataSetType.Sub:
            if self.sub_dataset is None:
                raise ValueError("No sub-dataset linked.")
            return self.sub_dataset
        else:
            return self, self.sub_dataset
    """
    def plot_point(self, *args, **kwargs):
        ds = self._get_dataset(which)
        ds.plot_point(*args, **kwargs)
    """

    def _drude(self, freq_, sig0, tau):
        # [tau] = fs, [sig0] = S/cm
        tau *= 1e-3  # fs = 1e-3 ps
        tau /= 2 * np.pi

        scale = 1
        sig0 *= scale

        w = 2 * np.pi * freq_
        return sig0 / (1 - 1j * tau * w)

    def _drude2(self, freq_, sig0, tau, wp, eps_inf):
        tau *= 1e-3
        tau /= 2 * np.pi
        w = 2 * np.pi * freq_

        return sig0 * wp**2 / (tau - 1j * w) - 1j * eps0_thz * w * (eps_inf - 1)

    def _drude_smith(self, freq_, wp, tau, eps_inf, c1=-0.99):
        w = 2 * np.pi * freq_
        #tau *= 1e-3
        #tau /= 2 * np.pi
        wt = 1 / tau
        f1 = wp**2 / (w**2 + 1j * w * wt)

        eps_ds = eps_inf - f1 * (1 + c1 * wt / (wt - 1j*w))

        return np.sqrt(eps_ds)

    def _lattice_contrib(self, freq_, tau, wp, eps_s, eps_inf):
        tau *= 1e-3 # fs = 1e-3 ps
        tau /= 2 * np.pi

        wp *= 2 * np.pi

        w = 2 * np.pi * freq_

        eps_l = eps_inf - ((eps_s - eps_inf) * wp ** 2) / (w ** 2 - wp ** 2 + 1j * w / tau)

        return eps_l

    def _total_response(self, freq, sig0, tau, wp, eps_s=None, eps_inf=None, c1=None):
        # [freq] = THz, [tau] = fs, [sig0] = S/cm, [wp] = THz. Dimensionless: eps_inf, eps_s
        if eps_s is None:
            eps_s = self._opt_consts["eps_s"]
        if eps_inf is None:
            eps_inf = self._opt_consts["eps_inf"]
        w = 2 * np.pi * freq

        sig_cc = self._drude(freq, sig0, tau)
        # sig_cc = drude_smith(freq, tau, sig0, c1)
        eps_l = self._lattice_contrib(freq, tau, wp, eps_s, eps_inf)

        sig_tot = self._n_to_sigma(self.freq_axis, np.sqrt(eps_l)) + sig_cc
        n_ = self._sigma_to_n(self.freq_axis, sig_tot)

        # n_ = np.sqrt(eps_l + 1j*sig_cc/(w*(eps0_thz * 1e-4))) # eps0_thz * 1e-4

        return n_.real + 1j * n_.imag

    def _sigma_to_n(self, freq_, sig_):
        # [eps0_thz] = ps * S / µm
        # sig_ in S/cm -> S/µm ( 1/(1e6 µm) = 1/m = 1/(1e2 cm) => 1e-4/µm = 1/cm)
        sig_ *= 1e-4

        w = 2*np.pi*freq_

        n_ = np.sqrt(1 + 1j * sig_ / (eps0_thz * w))

        return n_

    def _n_to_sigma(self, freq_, n_):
        # [eps0_thz] = ps * S / µm
        # sig_ in S/cm -> S/µm ( 1/(1e6 µm) = 1/m = 1/(1e2 cm) => 1e-4/µm = 1/cm)

        w = 2 * np.pi * freq_

        sig_ = -1j*(n_**2 - 1) * w * eps0_thz
        sig_ *= 1e4

        return sig_

    def _sigma_dc(self, freq, sig0):
        w = 2 * np.pi * freq
        sig0 *= 1e-4
        n_ = (1 + 1j) * np.sqrt(sig0/(2*w*eps0_thz))

        return n_

    def _t_cond_model(self, freq_, p_):
        n_sub_ = self._opt_consts["n_sub"]
        # tau = self._opt_consts["tau"]
        n_film_ = self.selected_n_model(freq_, *p_)

        # n_film_ = self._sigma_to_n(freq_, sig_model_)
        t_mod_ = self._t_model_2layer(freq_, n_sub_, n_film_)

        return t_mod_


    def _t_tmm_model_1layer(self, freq, n3_):
        d = self.options["sample_properties"]["d_1"]
        shift_ = self.options["eval_opt"]["shift_sub"]
        pol = "s"
        n_list = [1, n3_, 1]
        d_list = [np.inf, d, np.inf]
        th_0 = 0 * np.pi / 180
        lam_vac = c_thz / freq
        w_ = 2 * np.pi * freq

        phase_shift = np.exp(1j * shift_ * 1e-3 * w_)

        e_sam = coh_tmm(pol, n_list, d_list, th_0, lam_vac)
        e_ref = np.exp(1j * (d * w_ / c_thz))

        t = phase_shift * e_sam / e_ref

        return np.nan_to_num(t)

    def _t_model_1layer(self, freqs, n3_):
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

        r_geo = r_as * r_sa * exp ** 2 # r in geometric series
        if nfp == "inf":
            fp_factor = 1 / (1 + r_geo)
        else:
            fp_factor = 0
            for fp_idx in range(0, nfp+1):
                fp_factor += r_geo ** fp_idx

        e_sam = t_as * t_sa * exp * fp_factor
        e_ref = np.exp(1j * (d * w_ / c_thz))

        phase_shift = np.exp(1j * shift_ * 1e-3 * w_)

        t = phase_shift * e_sam / e_ref

        return np.nan_to_num(t)

    def _t_model_2layer(self, freq, n_sub, n_film):
        # n_sub += 0.01
        n1, n4 = 1, 1
        d = self.options["sample_properties"]["d_2"]
        h = self.options["sample_properties"]["d_film"]
        shift_ = self.options["eval_opt"]["shift_film"]
        nfp = self.options["eval_opt"]["nfp"]

        w_ = 2 * np.pi * freq
        t12 = 2 * n1 / (n1 + n_film)
        t23 = 2 * n_film / (n_film + n_sub)
        t34 = 2 * n_sub / (n_sub + n4)

        r12 = (n1 - n_film) / (n1 + n_film)
        r23 = (n_film - n_sub) / (n_film + n_sub)
        r34 = (n_sub - n4) / (n_sub + n4)

        exp1 = np.exp(1j * (h * w_ / c_thz) * n_film)
        exp2 = np.exp(1j * (d * w_ / c_thz) * n_sub)

        # r in geometric series
        r_geo = r12 * r23 * exp1 ** 2 + r23 * r34 * exp2 ** 2 + r12 * r34 * exp1 ** 2 * exp2 ** 2
        if nfp == "inf":
            fp_factor = 1 / (1 + r_geo)
        else:
            fp_factor = 0
            for fp_idx in range(0, nfp+1):
                fp_factor += r_geo ** fp_idx
        
        e_sam = t12 * t23 * t34 * exp1 * exp2 * fp_factor

        e_ref = np.exp(1j * ((d + h) * w_ / c_thz))
        phase_shift = np.exp(1j * shift_ * 1e-3 * w_)  # 6, 16

        t = phase_shift * e_sam / e_ref

        return np.nan_to_num(t)

    def _opt_fun_freq_model(self, p_):
        """
        bounds_ = self.options["eval_opt"]["freq_model_bounds"]
        for i in range(len(p_)):
            if (p_[i] < bounds_[i][0]) or (p_[i] > bounds_[i][1]):
                return np.inf
        """
        #if any([p < 0 for p in x_]):
        #    return np.inf

        freq = self.freq_axis
        model = self._t_cond_model
        # model = partial(self._drude2, eps_inf=9)
        # model = drude_smith

        y_mod = model(freq, p_)
        cost = self._abs_phi_cost_fun

        return cost(y_mod)

    def _abs_phi_cost_fun(self, y_mod_):
        f0, f1 = self.options["eval_opt"]["fit_range"]
        mask = (f0 <= self.freq_axis) * (self.freq_axis < f1)  # 2.2

        y_meas_ = self._opt_consts["y_meas"]

        phi_diff = (np.angle(y_mod_) - np.angle(y_meas_)) ** 2
        l_phi = np.sum(phi_diff[mask]) / len(self.freq_axis[mask])

        l_abs = self._abs_cost_fun(y_mod_)

        return l_abs + l_phi

    def _abs_cost_fun(self, y_mod_):
        f0, f1 = self.options["eval_opt"]["fit_range"]
        mask = (f0 <= self.freq_axis) * (self.freq_axis < f1)  # 2.2

        y_meas_ = self._opt_consts["y_meas"]
        abs_diff = (np.abs(y_mod_) - np.abs(y_meas_)) ** 2
        l_abs = np.sum(abs_diff[mask]) / len(self.freq_axis[mask])

        return l_abs

    def _fit_freq_model(self):
        self.selected_n_model = self._drude_smith
        # self.selected_n_model = self._sigma_dc

        bounds_ = self.options["eval_opt"]["freq_model_bounds"]

        min_kwargs = {"method": "Nelder-Mead",
                      "options": {
                          "maxev": np.inf,
                          "maxiter": 40000,
                          "tol": 1e-13,
                          "fatol": 1e-13,
                          "xatol": 1e-13,
                      },
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
        opt_res_ = shgo(self._opt_fun_freq_model,
                        bounds=bounds_,
                        n=2, iters=200,
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
        t_mod = self._t_model_1layer(freq_, n)
        # t_mod = self._tmm_model_1layer(freq_, n)

        abs_loss = (np.abs(t_mod) - np.abs(t_exp_)) ** 2
        ang_loss = (np.angle(t_mod) - np.angle(t_exp_)) ** 2

        return abs_loss + ang_loss

    def _opt_fun_2layer(self, x_, freq, n_sub_, t_exp_):
        #if x_[0] < 1 or x_[1] < 0:
        #    return np.inf

        n_film_ = x_[0] + 1j * x_[1]

        t_mod = self._t_model_2layer(freq, n_sub=n_sub_, n_film=n_film_)

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

    def _q_val(self, n_imag_):
        f0, f1 = self.options["eval_opt"]["fit_range_film"]
        freq_mask = (f0 <= self.freq_axis) * (self.freq_axis <= f1)

        n_imag_mean = np.mean(n_imag_[freq_mask])

        win_settings = {"en_plot": False, "win_start": f0, "win_width": f1 - f0, "fig_label": "FFT"}
        n_imag_ = window(np.array([self.freq_axis, n_imag_]).T, **win_settings)
        n_imag_ = n_imag_[:, 1]

        fft_ = np.fft.rfft(n_imag_[freq_mask] - n_imag_mean)
        fft_freq_axis = np.fft.rfftfreq(len(n_imag_[freq_mask]),
                                        d=np.mean(np.diff(self.freq_axis)))

        # mask_ = (10 < fft_freq_axis) * (fft_freq_axis < 20)
        mask_ = (11 <= fft_freq_axis)
        peak_val, peak_idx = np.max(np.abs(fft_[mask_])), np.argmax(np.abs(fft_[mask_]))
        sum_val = np.sum(np.abs(fft_[7 <= fft_freq_axis]))

        # plt.figure("TESTFFT")
        # plt.plot(fft_freq_axis, np.abs(fft_), label=f"shift {shift}")

        fs = 1/np.mean(np.diff(self.freq_axis))
        Q = 0.5  # quality factor: higher = narrower

        peak_freq = fft_freq_axis[mask_][peak_idx]
        # print(peak_freq, fs)
        b, a = iirnotch(peak_freq / (fs / 2), Q)

        n_imag_filt_ = filtfilt(b, a, n_imag_)

        return peak_val, sum_val, n_imag_filt_

    def _fit_2layer(self, t_exp_, n_sub_):
        pnt = self.options["eval_opt"]["sub_pnt"]
        bounds_ = self.options["eval_opt"]["film_bounds"]
        # bounds_ = [(1, 15), (1, 15)]

        min_kwargs = {"method": "Nelder-Mead",
                      "options": {
                          # "maxev": np.inf,
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
            "disp": False,
        }

        f0, f1 = self.options["eval_opt"]["fit_range_film"]
        freq_mask = (f0 <= self.freq_axis) * (self.freq_axis <= f1)

        best_ = (None, np.inf)
        n_opt_best = None
        d_sub0 = self.options["sample_properties"]["d_1"]
        for d_sub in [*np.arange(d_sub0, d_sub0 + 1, 1.0)]:#[*np.arange(640.00, 655, 5.0)]: # [656.8]:# 650
            for shift in [0]:  # [61.1]: # 30
                # self.options["sample_properties"]["d_film"] = d_film
                self.options["sample_properties"]["d_2"] = d_sub
                self.options["eval_opt"]["shift_film"] = shift

                gof = 0
                n_opt = np.zeros((len(self.freq_axis), len(bounds_)), dtype=complex)
                for f_idx in np.arange(len(self.freq_axis))[freq_mask]:
                    args_ = (self.freq_axis[f_idx], n_sub_[f_idx], t_exp_[f_idx])

                    opt_res_ = shgo(self._opt_fun_2layer,
                                    bounds=bounds_,
                                    minimizer_kwargs=min_kwargs,
                                    options=shgo_options,
                                    args=args_,
                                    iters=2,
                                    )
                    n_opt[f_idx] = opt_res_.x
                    # print(self.freq_axis[f_idx], n_opt[f_idx])
                    gof += opt_res_.fun

                n_opt = n_opt[:, 0] + 1j * n_opt[:, 1]
                n_imag = n_opt.imag

                peak_val, sum_val, n_imag_filt_ = self._q_val(n_imag)

                if peak_val < best_[1]:
                    best_ = ((shift, d_sub), peak_val, gof)
                    n_opt_best = n_opt

        print("Best q-val film: ", best_)
        return n_opt_best

    def _fit_1layer(self, t_exp_):
        pnt = self.options["eval_opt"]["sub_pnt"]
        bounds = self.options["eval_opt"]["sub_bounds"]
        #bounds = [(3.05, 3.13), (0.0025, 0.0050)]
        min_kwargs = {"method": "Nelder-Mead",
                      "options": {
                          #"maxev": np.inf,
                          # "maxiter": 20,
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
        d_sub0 = self.options["sample_properties"]["d_1"]
        for d_sub in [*np.arange(d_sub0, d_sub0+1, 1.0)]:#[639.5]:#[*np.arange(639, 640, 0.5)]: # 639 / 640 (Teralyzer)
            for shift in [*np.arange(-0, 1, 1.0)]:# 28, 22, -3
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
                peak_val, sum_val, n_imag_filt = self._q_val(n_imag)

                gof = gof / np.sum(freq_mask)

                print("Sub:", sum_val, peak_val, shift, d_sub, gof)

                res = ((shift, d_sub), peak_val, sum_val)
                if peak_val < best_[1]:
                    best_ = res
                    n_opt_best = n_opt

                with open("debug_out", "a") as f:
                    res_line = f"{pnt} {str(res)}\n"
                    f.write(res_line)

        print(f"Best q-val sub: {best_} @{pnt}")
        self.options["sample_properties"]["d_1"] = best_[0][1]
        self.options["eval_opt"]["shift_sub"] = best_[0][0]

        with open("result_out", "a") as f:
            res_line = f"{pnt} {str(best_)}\n"
            f.write(res_line)

        with open("debug_out", "a") as f:
            f.write("\n")

        return n_opt_best

    def conductivity_model(self, sigma_exp):
        self._opt_consts["sigma_exp"] = sigma_exp
        opt_res = self._fit_freq_model()
        p = opt_res.x # x = [tau, sig0, wp, eps_inf, eps_s]
        # p = [1, 100, 4*np.pi, 10, 20]
        # p = [1, 100, 2, 16.8, 20] # ulatowski plot
        # p = [-1.588e-02,  5.044e+01,  920.8,  40.55, -43.13] # fit result
        # p = [-0.01588,  50.44,  920.8,  40.55, -43.13]
        # p = [40, 3, 50, 9]
        sigma_model_ = self._total_response(self.freq_axis, *p) # TODO _total_response returns n

        return sigma_model_

    def phase_fit_(self, freq, phi):
        z = polyfit(freq, phi, 1)
        plt.figure("TEST")
        plt.plot(freq, z[0]*freq + z[1])
        print(z)

    def t_sim_1layer(self):
        if not self.options["sim_opt"]["enabled"]:
            exit("BB")

        self.options["eval_opt"]["shift_sub"] = self.options["sim_opt"]["shift_sim"]
        n_sub = self.options["sim_opt"]["n_sub"]
        nfp_og = self.options["eval_opt"]["nfp"]

        self.options["eval_opt"]["nfp"] = self.options["sim_opt"]["nfp_sim"]
        t_sim = np.zeros_like(self.freq_axis, dtype=complex)
        for f_idx, freq in enumerate(self.freq_axis):
            n_sub_ = n_sub#0.03*freq + n_sub.real + 1j * freq * 0.001
            t_sim[f_idx] = self._t_model_1layer(freq, n_sub_)
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
        phi = np.unwrap(np.angle(ref_data))
        phi_mean, phi_std = np.mean(phi, axis=0), np.std(phi, axis=0)

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

    def sub_meas_sim(self):
        t_sim = self.t_sim_1layer()

        sub_pnt = self.options["eval_opt"]["sub_pnt"]

        # ref1_fd, ref1_meas = self.get_ref_data(Domain.Frequency, ref_idx=10, ret_meas=True)
        ref1_fd, ref1_meas = self.sub_dataset.get_ref_data(Domain.Frequency, point=sub_pnt, ret_meas=True)
        ref2_fd, ref2_meas = self.sub_dataset.get_ref_data(Domain.Frequency, ref_idx=10, ret_meas=True)

        meas_time_diff = (ref1_meas.meas_time - ref2_meas.meas_time).total_seconds()
        print("ref1 - ref2 measurement time difference (seconds): ", np.round(meas_time_diff, 2))

        ref_amp_std = self.ref_std(en_plot=False)

        ref_amp = np.abs(ref2_fd[:, 1])
        ref_phi = np.angle(ref2_fd[:, 1])
        ref2_fd[:, 1] = ref_amp * np.exp(1j * ref_phi)

        t_sim_meas = t_sim * ref1_fd[:, 1] / ref2_fd[:, 1]

        sam_sim = t_sim * ref1_fd[:, 1]
        sam_sim_fd = np.array([self.freq_axis, sam_sim], dtype=complex).T

        t_sim_meas = sam_sim_fd[:, 1] / ref2_fd[:, 1]

        sam_sim_td = do_ifft(sam_sim_fd, conj=False)

        plt.figure("Time domain")
        plt.plot(sam_sim_td[:, 0], sam_sim_td[:, 1], label="Model")

        return t_sim_meas


    def eval_point_n_fit(self, film_pnt=None):
        """
        Fit refractive index to the substrate measurement (n_sub)
        then use n_sub in the fit of the refractive index to the film measurement (n_film)
        calculate sigma from n_film
        """
        sub_pnt = self.options["eval_opt"]["sub_pnt"]
        if film_pnt is None:
            film_pnt = sub_pnt
        res = {}

        meas_sub = self.sub_dataset.get_measurement(*sub_pnt)
        meas_film = self.get_measurement(*film_pnt)

        single_layer_eval_res = self.sub_dataset.single_layer_eval(meas_sub, (0, 10))
        res["alpha"] = single_layer_eval_res["alpha"]

        res["sigma_exp"] = self._conductivity(meas_film)
        res["sigma_mod"] = self.conductivity_model(res["sigma_exp"])

        #self.sub_dataset.options["pp_opt"]["window_opt"]["enabled"] = True
        self.sub_dataset.options["pp_opt"]["window_opt"]["en_plot"] = False
        self.sub_dataset.options["pp_opt"]["window_opt"]["fig_label"] = "sub"
        t_exp_1layer = self.sub_dataset.transmission(meas_sub, 1, phase_sign=-1)
        # t_exp_1layer = self.transmission_sim()
        # t_exp_1layer = self.sub_meas_sim()

        #self.sub_dataset.options["pp_opt"]["window_opt"]["enabled"] = False

        # phi = np.unwrap(np.angle(t_exp_1layer))
        # phi = phase_correction(self.freq_axis, phi, en_plot=True, fit_range=(0.3, 0.6))
        # phi -= 0.03
        # t_exp_1layer = np.abs(t_exp_1layer) * np.exp(1j * phi)

        res["t_exp_1layer"] = t_exp_1layer

        n_sub = self._fit_1layer(t_exp_1layer)
        res["n_sub"] = n_sub
        self._opt_consts["n_sub"] = n_sub

        if self.options["eval_opt"]["area_fit"]:
            return res

        self.options["pp_opt"]["window_opt"]["fig_label"] = "film"
        t_exp_2layer = self.transmission(meas_film, 1, phase_sign=-1)

        res["t_exp_2layer"] = t_exp_2layer

        n_film = self._fit_2layer(t_exp_2layer, n_sub)

        res["t_mod_film"] = self._t_model_2layer(self.freq_axis, n_sub=n_sub, n_film=n_film)

        w = 2 * np.pi * np.array(self.freq_axis)
        res["sigma_n_film"] = 1e4 * 2 * eps0_thz * n_film ** 2 * w / (1 + 1j)  # S/cm

        return res

    def eval_point_model_fit(self, film_pnt=None):
        """
        Fit model with frequency independent parameters to the full spectrum
        """
        sub_pnt = self.options["eval_opt"]["sub_pnt"]
        if film_pnt is None:
            film_pnt = sub_pnt
        res = {}

        meas_sub = self.sub_dataset.get_measurement(*sub_pnt)
        meas_film = self.get_measurement(*film_pnt)

        res["t_exp_1layer"] = self.sub_dataset.transmission(meas_sub, 1, phase_sign=-1)
        n_sub = self._fit_1layer(res["t_exp_1layer"])
        res["n_sub"] = n_sub

        res["t_exp_2layer"] = self.transmission(meas_film, 1, phase_sign=-1)

        # n_sub.imag = 0.023*self.freq_axis

        self._opt_consts["y_meas"] = res["t_exp_2layer"]
        self._opt_consts["n_sub"] = n_sub
        self._opt_consts["eps_s"] = 5
        self._opt_consts["eps_inf"] = 50
        # self._opt_consts["tau"] = 100 * 10

        freq_fit_res = self._fit_freq_model()
        p_opt = freq_fit_res.x
        print(freq_fit_res)
        res["t_mod_film"] = self._t_cond_model(self.freq_axis, p_opt)
        # [ 2.503e+03  6.897e+03 -4.208e+00  2.306e+03 -2.305e+03]
        # best res: [ 2.720e+04 -1.130e+03  9.979e-01 -2.997e+03  4.591e+04] or [ 3.714e+04  4.980e+02  1.801e+00  5.587e+04  1.945e+00]
        # p_opt = [100, 1000, 2, 20, 0.025*16.8] # sig0, tau, wp, eps_s, eps_inf = 16.8 # 0.025*16.8
        p_opt = [ 3.714e+04,  4.980e+02,  1.801e+00, 5.587e+04,  1.945e+00]
        # p_opt = [*p_opt, self._opt_consts["eps_s"], self._opt_consts["eps_inf"]]
        sig_cc = self._drude(self.freq_axis, p_opt[0], p_opt[1])
        eps_l = self._lattice_contrib(self.freq_axis, *p_opt[1:])
        n_film = self._total_response(self.freq_axis, *p_opt)
        sig_tot = self._n_to_sigma(self.freq_axis, n_film) + sig_cc

        # res["t_mod_film"] = self._t_cond_model(self.freq_axis, p_opt)

        # n_film = self._sigma_to_n(self.freq_axis, sig_tot)

        plt.figure("_drude_cc_part")
        plt.title("Ulatowski check (publication values) drude part")
        plt.plot(self.freq_axis, sig_cc.real, label="real part")
        plt.plot(self.freq_axis, sig_cc.imag, label="imag part")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Sigma_cc (S/cm)")

        plt.figure("_drude_l_part")
        plt.title("Ulatowski check (publication values) lattice part")
        plt.plot(self.freq_axis, eps_l.real, label="real part")
        plt.plot(self.freq_axis, eps_l.imag, label="imag part")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("eps_l")

        plt.figure("_total_response")
        plt.title("Ulatowski check (publication values) total response")
        plt.plot(self.freq_axis, sig_tot.real, label="real part")
        plt.plot(self.freq_axis, sig_tot.imag, label="imag part")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("sig_tot (S/cm)")

        """
        plt.figure("Transmission fit abs film")
        for sig0_ in [1e4, 1e5]:
            f0, f1 = self.options["eval_opt"]["fit_range_sub"]
            f_mask = (f0 < self.freq_axis) * (self.freq_axis < f1)
            t_mod = self.selected_model(self.freq_axis, sig0_, tau=100)
            plt.plot(self.freq_axis[f_mask], np.abs(t_mod[f_mask]), label=f"Model sigma0={sig0_}")
            print(self._opt_fun_freq_model([sig0_, 100]), sig0_, 100)
        """


        self.plot_eval_res(res)

        return res

    def plot_eval_res(self, res):

        n_sub = res["n_sub"]

        t_exp_1layer, t_exp_2layer = res["t_exp_1layer"], res["t_exp_2layer"]

        f0, f1 = self.options["eval_opt"]["fit_range_sub"]
        f_mask = (f0 < self.freq_axis) * (self.freq_axis < f1)

        plt.figure("TEST2")
        phi_1l = np.unwrap(np.angle(t_exp_1layer[f_mask]))
        phi_2l = np.unwrap(np.angle(t_exp_2layer[f_mask]))
        plt.plot(self.freq_axis[f_mask], phi_1l, label="Experiment 1l")
        plt.plot(self.freq_axis[f_mask], phi_2l, label="Experiment 2l")
        # self.phase_fit_(self.freq_axis[f_mask], phi_1l)
        # self.phase_fit_(self.freq_axis[f_mask], phi_2l)

        plt.figure("n_sub")
        plt.plot(self.freq_axis, n_sub.real, label="Real part")
        plt.plot(self.freq_axis, n_sub.imag, label="Imaginary part")
        # plt.ylim((-0.005, 0.020))
        plt.xlim(self.options["eval_opt"]["fit_range_sub"])
        plt.xlabel("Frequency (THz)")

        t_mod_sub = self._t_model_1layer(self.freq_axis, n_sub)
        plt.figure("Transmission fit abs sub")
        plt.plot(self.freq_axis[f_mask], np.log10(np.abs(t_exp_1layer[f_mask])), label="Experiment")
        plt.plot(self.freq_axis[f_mask], np.log10(np.abs(t_mod_sub[f_mask])), label="Model")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("log10(|t|)")

        plt.figure("Transmission fit angle sub")
        plt.plot(self.freq_axis[f_mask], np.angle(t_exp_1layer[f_mask]), label="Experiment")
        plt.plot(self.freq_axis[f_mask], np.angle(t_mod_sub[f_mask]), label="Model")
        plt.xlabel("Frequency (THz)")

        plt.figure("Transmission fit abs sub diff")
        log_diff = np.log10((np.abs(t_exp_1layer[f_mask]) - np.abs(t_mod_sub[f_mask])) ** 2)
        plt.plot(self.freq_axis[f_mask], log_diff, label="Log squared difference")
        # plt.plot(self.freq_axis[f_mask], n_sub.imag[f_mask] / np.max(n_sub.imag[f_mask]), label="n_sub.imag")
        plt.xlabel("Frequency (THz)")

        t_mod_film = res["t_mod_film"]
        plt.figure("Transmission fit abs film")
        plt.plot(self.freq_axis[f_mask], np.abs(t_exp_2layer[f_mask]), label="Experiment", ls="dashed")
        plt.plot(self.freq_axis[f_mask], np.abs(t_mod_film[f_mask]), label="Model (fit)")
        plt.ylim((-0.05, 0.45))
        plt.xlabel("Frequency (THz)")
        plt.ylabel("|t|")

        plt.figure("Transmission fit angle film")
        plt.plot(self.freq_axis[f_mask], np.angle(t_exp_2layer[f_mask]), label="Experiment", ls="dashed")
        plt.plot(self.freq_axis[f_mask], np.angle(t_mod_film[f_mask]), label="Model (fit)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("angle(t)")

        plt.figure("Transmission fit angle film")
        plt.plot(self.freq_axis[f_mask], np.angle(t_exp_2layer[f_mask]), label="Experiment")
        plt.plot(self.freq_axis[f_mask], np.angle(t_mod_film[f_mask]), label="Model")
        plt.xlabel("Frequency (THz)")

        if "sigma_n_film" in res:
            sigma_n_film = res["sigma_n_film"]
            plt.figure("Conductivity(fit)")
            plt.plot(self.freq_axis, sigma_n_film.real, label="Real part")
            plt.plot(self.freq_axis, sigma_n_film.imag, label="Imaginary part")

        if "single_layer_eval" in res:
            single_layer_eval = res["single_layer_eval"]

            plt.figure("n_sub")
            # plt.plot(self.freq_axis, single_layer_eval["refr_idx"].imag, label="Imaginary part (1 layer eval)")

            plt.figure("absorption coefficient")
            plt.plot(self.freq_axis, 4 * np.pi * n_sub.imag * self.freq_axis / 0.03, label="fit")
            plt.plot(self.freq_axis, single_layer_eval["alpha"], label="single layer eval")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Absorption coefficient (1/cm)")
            plt.ylim((0, 0.012))
            plt.xlim((-0.05, 3.5))

        if "n_film" in res:
            n_film = res["n_film"]
            plt.figure("n_film")
            plt.plot(self.freq_axis, n_film.real, label="real")
            plt.plot(self.freq_axis, n_film.imag, label="imag")
            plt.xlabel("Frequency (THz)")

        if "sigma_exp" in res and "sigma_mod" in res:
            sigma_exp, sigma_mod = res["sigma_exp"], res["sigma_mod"]
            plt.figure("Conductivity")
            plt.plot(self.freq_axis[f_mask], sigma_exp[f_mask].real, label="Exp (real)")
            plt.plot(self.freq_axis[f_mask], sigma_exp[f_mask].imag, label="Exp (imag)")
            plt.plot(self.freq_axis[f_mask], sigma_mod[f_mask].real, label="Fit (real)")
            plt.plot(self.freq_axis[f_mask], sigma_mod[f_mask].imag, label="Fit (imag)")
            # plt.xlim((0.20, 2.55))
            # plt.ylim((-10, 200))
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Conductivity (S/cm)")

if __name__ == "__main__":
    pass


