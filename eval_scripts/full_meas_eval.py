import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0
from dataset import DataSet, plt_show, QuantityEnum
from scipy.optimize import shgo, curve_fit
from functools import partial
import logging
from functions import do_ifft
from numpy import polyfit

c = c / 1e12

n1, n4 = 1, 1
f0 = 1.25
h0 = 300 * 1e-9
d0 = 645 * 1e-6

en_sub_window = True
en_film_window = True

def model_1layer(n3_, d=d0, f=f0, shift_=0):
    w_ = 2 * pi * f
    t_as = 2 * n1 / (n1 + n3_)
    t_sa = 2 * n3_ / (n1 + n3_)
    r_as = (n1 - n3_) / (n1 + n3_)
    r_sa = (n3_ - n1) / (n1 + n3_)

    exp = np.exp(1j * (d * w_ / c) * n3_)
    e_sam = t_as * t_sa * exp / (1 + r_as * r_sa * exp ** 2)
    e_ref = np.exp(1j * (d * w_ / c))
    phase_shift = np.exp(1j * shift_ * 1e-3 * w_)
    t = phase_shift * e_sam / e_ref

    return np.nan_to_num(t)


def model_2layer(n2_, n3_, h=h0, d=d0, f=f0, shift_=0):
    # n3_ += 0.01
    w_ = 2 * pi * f
    t12 = 2 * n1 / (n1 + n2_)
    t23 = 2 * n2_ / (n2_ + n3_)
    t34 = 2 * n3_ / (n3_ + n4)

    r12 = (n1 - n2_) / (n1 + n2_)
    r23 = (n2_ - n3_) / (n2_ + n3_)
    r34 = (n3_ - n4) / (n3_ + n4)

    exp1 = np.exp(1j * (h * w_ / c) * n2_)
    exp2 = np.exp(1j * (d * w_ / c) * n3_)

    e_sam = t12 * t23 * t34 * exp1 * exp2 / (1 + r12 * r23 * exp1 ** 2 + r23 * r34 * exp2 ** 2 + r12 * r34 * exp1 ** 2 * exp2 ** 2)
    e_ref = np.exp(1j * ((d0 + h) * w_ / c))
    phase_shift = np.exp(1j * shift_ * 1e-3 * w_) # 6, 16

    t = phase_shift * e_sam / e_ref

    return np.nan_to_num(t)


def cost_1layer(p, freq_, t_exp_, shift=0, d=d0, bounds_=None):
    if bounds_ is not None:
        for i, p_ in enumerate(p):
            if not (bounds_[i][0] < p_) * (p_ < bounds_[i][1]):
                return np.inf

    n = p[0] + 1j * p[1]
    t_mod = model_1layer(n, f=freq_, shift_=shift, d=d)

    abs_loss = (np.abs(t_mod) - np.abs(t_exp_)) ** 2
    ang_loss = (np.angle(t_mod) - np.angle(t_exp_)) ** 2

    return abs_loss + ang_loss


def cost_2layer(p, freq_, t_exp_, n_sub_, d=d0, shift=0):
    if p[0] < 0:
        return np.inf
    n = p[0] + 1j * p[1]
    t_mod = model_2layer(n, n3_=n_sub_, f=freq_, shift_=shift, d=d)

    #real_loss = np.real(t_mod - t_exp_)**2
    #imag_loss = np.imag(t_mod - t_exp_)**2
    #return real_loss + imag_loss

    abs_loss = (np.abs(t_mod) - np.abs(t_exp_)) ** 2
    ang_loss = (np.angle(t_mod) - np.angle(t_exp_)) ** 2

    return abs_loss + ang_loss


def optimize_1layer(f_axis_, t_exp_):
    bounds = [(3.00, 3.15), (0.0, 0.020)]
    min_kwargs = {"method": "Nelder-Mead",
                  "options": {
                      "maxev": np.inf,
                      "maxiter": 4000,
                      "tol": 1e-12,
                      "fatol": 1e-12,
                      "xatol": 1e-12,
                  }
    }

    shift = 15.3 # 30
    d = d0
    #for d in np.arange(640, 655, 1.0) * 1e-6:
    for shift in [shift]:#, *np.arange(14.5, 15.9, 0.1)]:
        n_opt_res = np.zeros_like(f_axis_, dtype=complex)
        for f_idx, f in enumerate(f_axis_):
            if (f > 3.5) or (f < 0.0):
                continue

            cost = partial(cost_1layer, freq_=f, t_exp_=t_exp_[f_idx],
                           shift=shift, d=d, bounds_=bounds)

            opt_res_ = shgo(cost, bounds, minimizer_kwargs=min_kwargs)
            n_opt_res[f_idx] = opt_res_.x[0] + opt_res_.x[1] * 1j

        f_mask = (0.60 < f_axis_) * (f_axis_ < 2.5)
        fft_ = np.fft.rfft(n_opt_res[f_mask].imag)
        fft_freq_axis = np.fft.rfftfreq(len(n_opt_res[f_mask].imag),
                                        d=np.mean(np.diff(f_axis_)))
        #mask_ = (10 < fft_freq_axis) * (fft_freq_axis < 20)
        #print(np.max(np.abs(fft_[mask_])), d, shift)
        #plt.figure("fft")
        #plt.plot(fft_freq_axis, np.abs(fft_), label=f"shift {shift}")

    return n_opt_res


def optimize_2layer(f_axis_, t_exp_, n_sub):
    bounds = [(2, 30), (-10.0, 30.0)]
    min_kwargs = {"method": "Nelder-Mead",
                  "options": {
                      "maxev": np.inf,
                      "maxiter": 4000,
                      "tol": 1e-12,
                      "fatol": 1e-12,
                      "xatol": 1e-12,
                  }
                  }

    # direct sub ri: use minima at 5.0
    # for (23, 2.5): d=641.2 µm, dt=31 fs
    for d in [644*1e-6]: #np.arange(643, 645.1, 0.5) * 1e-6: # opt [642*1e-6] h=400 nm
        for shift in [17]: #[*np.arange(5, 20.1, 1)]: # opt [31]
            n_opt_res = np.zeros_like(f_axis_, dtype=complex)
            for f_idx, f in enumerate(f_axis_):
                if (f > 2.5) or (f < 0.0):
                    continue
                """
                if f_idx != 0:
                    prev_n = n_opt_res[f_idx-1]
                    bounds = [(prev_n.real*0.9, prev_n.real*1.1),
                              (bounds[1][0], bounds[1][1])]
                """
                cost = partial(cost_2layer, freq_=f, t_exp_=t_exp_[f_idx],
                               n_sub_=n_sub[f_idx], shift=shift, d=d)

                opt_res_ = shgo(cost, bounds, minimizer_kwargs=min_kwargs)
                n_opt_res[f_idx] = opt_res_.x[0] + opt_res_.x[1] * 1j

            #"""
            f_mask = (0.5 < f_axis_) * (f_axis_ < 1.75)
            fft_ = np.fft.rfft(n_opt_res[f_mask].imag)
            fft_freq_axis = np.fft.rfftfreq(len(n_opt_res[f_mask].imag),
                                            d=np.mean(np.diff(f_axis_)))
            mask_ = (6 < fft_freq_axis) * (fft_freq_axis < 20)
            print(np.max(np.abs(fft_[mask_])), d*1e6, shift)
            plt.figure("fft_film")
            plt.plot(fft_freq_axis, np.abs(fft_), label=f"shift {shift}")
            #"""
    plt.legend()
    #plt.show()

    return n_opt_res

def ri_fit(freq_axis_, n_, ret_alpha_fit=False):
    f_mask = (0.60 < freq_axis_) * (freq_axis_ < 2.5)
    coe_i = polyfit(freq_axis_[f_mask], n_[f_mask].imag, 2)
    parabola = lambda x_, a_: a_*x_**2

    cf_res = curve_fit(parabola, freq_axis_[f_mask], n_[f_mask].imag, p0=coe_i[:1])

    n_sub_imag_cf = cf_res[0][0] * freq_axis_ ** 2
    coe_r = polyfit(freq_axis_[f_mask], n_[f_mask].real, 2)
    n_sub_imag_fit = coe_i[0] * freq_axis_ ** 2 + coe_i[1] * freq_axis_ + coe_i[2]
    n_sub_real_fit = coe_r[0] * freq_axis_ ** 2 + coe_r[1] * freq_axis_ + coe_r[2]

    alpha = 1e-2 * 2 * 2 * np.pi * freq_axis_ * n_.imag / c
    alpha_cf_res = curve_fit(parabola, freq_axis_[f_mask], alpha[f_mask], p0=1)

    alpha_cf = alpha_cf_res[0][0] * freq_axis_ ** 2
    n_sub_imag_alpha_cf = 1e2 * alpha_cf * c / (2 * 2 * np.pi * freq_axis_)

    fft_ = np.fft.rfft(n_[f_mask].imag)
    freq_ = np.fft.rfftfreq(len(n_[f_mask].imag), d=np.mean(np.diff(freq_axis_)))

    plt.figure("fft")
    plt.plot(freq_, np.abs(fft_))
    plt.legend()
    plot_range = (0.35 < freq_axis_) * (freq_axis_ < 2.5)

    #"""
    plt.figure("Absorption coefficient")
    plt.plot(freq_axis_, alpha)
    plt.plot(freq_axis_, alpha_cf)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Absorption coefficient (1/cm)")

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 4), height_ratios=[1, 2], num="ri sub fit")
    ax1.plot(freq_axis_[plot_range], n_.real[plot_range])
    ax2.plot(freq_axis_[plot_range], n_.imag[plot_range], color="red")
    ax1.set_ylim(3.04, 3.12)
    ax2.set_ylim(-0.001, 0.015)

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labelbottom=False)

    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    #ax1.plot((-d, +d), (-d, +d), **kwargs)  # top-left
    #ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right

    kwargs.update(transform=ax2.transAxes)  # switch to ax2
    #ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left
    #ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right

    ax2.set_xlim((-0.05, 3.05))
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.05)

    #plt.plot(freq_axis_, n_sub_real_fit, label="fit (real)")
    #plt.plot(freq_axis_, n_sub_imag_fit, label="fit (imag)")
    #plt.plot(freq_axis_, n_sub_imag_cf, label="kappa fit (imag)")
    #plt.plot(freq_axis_, n_sub_imag_alpha_cf, label="alpha fit (imag)")
    ax2.set_xlabel("Frequency (THz)")
    ax1.set_ylabel("$n_{s,real}$")
    ax2.set_ylabel("$n_{s,imag}$")

    #ax1.legend()
    #ax2.legend()
    # plt.show()
    #"""
    if ret_alpha_fit:
        return n_sub_real_fit + 1j * n_sub_imag_alpha_cf
    else:
        return n_sub_real_fit + 1j * n_sub_imag_fit

def drude(freq_axis, tau, sig0):
    scale = 1
    sig0 *= scale
    tau *= 1e-3
    w = 2 * np.pi * freq_axis
    return sig0 / (1 - 1j * tau * w)


def lattice_contrib(freq_axis, tau, wp, eps_inf, eps_s):
    tau *= 1e-3
    w = 2 * np.pi * freq_axis
    return eps_inf - (eps_s - eps_inf) * wp ** 2 / (w ** 2 - wp ** 2 + 1j * w / tau)


def total_response(freq, tau, sig0, wp, eps_inf, eps_s, c1=None):
    # [freq] = THz, [tau] = fs, [sig0] = S/cm, [wp] = THz. Dimensionless: eps_inf, eps_s
    sig_cc = drude(freq, tau, sig0)
    # sig_cc = drude_smith(freq, tau, sig0, c1)
    eps_L = lattice_contrib(freq, tau, wp, eps_inf, eps_s)

    w_ = 2 * np.pi * freq
    return ((1 - 1e12 * w_ * epsilon_0 * eps_L) * 1j + 100 * sig_cc) / 100

def drude_smith(freq, tau, sig0, c1):
    tau *= 1e-3
    w = 2 * np.pi * freq
    f = (1 + c1 / (1 - 1j * w * tau))
    return drude(freq, tau*1e3, sig0) * f

def opt_fun(x_, freq_axis, sigma_):
    #if any([p < 0 for p in x_]):
    #    return np.inf

    model = total_response
    # model = drude_smith
    mask = (0.35 <= freq_axis) * (freq_axis < 2.0) # 2.2
    real_part = (model(freq_axis, *x_).real - sigma_.real) ** 2
    imag_part = (model(freq_axis, *x_).imag - sigma_.imag) ** 2

    return np.sum(real_part[mask] + imag_part[mask]) / (len(freq_axis[mask]) * 1000)

def fit_conductivity_model(freq_axis_, sigma_):
    cost = partial(opt_fun, freq_axis=freq_axis_, sigma_=sigma_)
    logging.basicConfig(level=logging.INFO)
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
        #"maxfev": np.inf,
        #"f_tol": 1e-12,
        #"maxiter": 4000,
        #"ftol": 1e-12,
        #"xtol": 1e-12,
        #"maxev": 4000,
        #"minimize_every_iter": True,
        #"disp": True
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

def tinkham_eval(f_axis_, t_sub_, t_film_, n_sub_):
    w_ = 2 * np.pi * f_axis_
    shift = np.exp(1j * 0 * 1e-3 * w_)
    sigma_ = 1e-2 * shift * np.abs(t_sub_ / t_film_ - 1) * epsilon_0 * (1e12 * c) * (1 + n_sub_) / h0
    return sigma_ # S/cm

## measurement
options = {
    "ref_pos": (10, None),  # img5
    "cbar_lim": (0.56, 0.60),
    "window_options": {"en_plot": False, "win_width": 2*50},
}
dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img5", options)
dataset.select_freq(f0)

res_sub = dataset.plot_point((70, 2.5), en_td_plot=False,
                             apply_window=en_sub_window, remove_t_offset=True, label="Sub. (exp.)")
res_film = dataset.plot_point((40, 2.5), en_td_plot=True, # (grey area: (23, 2.5))
                              apply_window=en_film_window, remove_t_offset=True, label="film")

#dataset.plot_system_stability()
#dataset.select_quantity(QuantityEnum.TransmissionAmp)
#dataset.plot_image()
#plt_show()

f_axis = res_sub["freq_axis"]
w = 2*f_axis*np.pi

t_exp_sub = res_sub["t"]
n_sub_res = optimize_1layer(f_axis, t_exp_sub)
n_sub_res_fit = ri_fit(f_axis, n_sub_res, ret_alpha_fit=True)
# n_sub_res = n_sub_res_fit
# n_sub_res = 3.0806 * np.ones_like(n_sub_res)

t_sub_mod = np.array([model_1layer(n3_=n_sub_res[f_idx], f=f) for f_idx, f in enumerate(f_axis)])

sub_fd_mod = np.array([res_sub["ref_fd"][:, 0], res_sub["ref_fd"][:, 1] * t_sub_mod]).T
sam_td_sub_mod = do_ifft(sub_fd_mod, out_len=4001)

t_exp_film = res_film["t"]
n_film_res = optimize_2layer(f_axis, t_exp_film, n_sub_res)
n_film_res_fit = ri_fit(f_axis, n_film_res, ret_alpha_fit=False)
# n_film_res = n_film_res_fit

t_film_mod = np.array([model_2layer(n2_=n_film_res[f_idx], n3_=n_sub_res[f_idx], f=f) for f_idx, f in enumerate(f_axis)])

film_fd_mod = np.array([res_film["ref_fd"][:, 0], res_film["ref_fd"][:, 1] * t_film_mod]).T
sam_td_film_mod = do_ifft(film_fd_mod, out_len=4001)

sigma_tnk = tinkham_eval(f_axis, t_exp_sub, t_exp_film, n_sub_res)

sigma = 1e-2 * 2 * epsilon_0 * n_film_res**2 * w * 1e12 / (1 + 1j)  # S/cm

opt_res = fit_conductivity_model(f_axis, sigma)
x = opt_res.x
# x = [-1.588e-02,  5.044e+01,  920.8,  40.55, -43.13] # fit result
# x = [-0.01588,  50.44,  920.8,  40.55, -43.13]
sigma_fit = total_response(f_axis, *x)
# sigma_fit = drude(f_axis, 100, 50.44)

print(opt_res)
print(opt_fun(opt_res.x, f_axis, sigma))

plot_range = (0.35 < f_axis) * (f_axis < 2.5)

plt.figure("Substrate refractive index")
plt.plot(f_axis, n_sub_res.real, label="exp (real)")
plt.plot(f_axis, n_sub_res.imag, label="exp (imag)")
plt.plot(f_axis, n_sub_res_fit.real, label="fit (real)")
plt.plot(f_axis, n_sub_res_fit.imag, label="fit (imag)")
plt.xlabel("Frequency (THz)")
plt.ylabel("Refractive index")
plt.xlim((-0.05, 3.05))
# plt.ylim((-0.01, 0.03))

plt.figure("Film refractive index")
plt.plot(f_axis[plot_range], n_film_res[plot_range].real, label="exp (real)")
plt.plot(f_axis[plot_range], n_film_res[plot_range].imag, label="exp (imag)")
plt.plot(f_axis[plot_range], n_film_res_fit[plot_range].real, label="fit (real)")
plt.plot(f_axis[plot_range], n_film_res_fit[plot_range].imag, label="fit (imag)")
plt.xlabel("Frequency (THz)")
plt.ylabel("Refractive index")

plt.figure("Phase")
plt.plot(f_axis, np.unwrap(np.angle(sub_fd_mod[:, 1])), label="sub (model)")
plt.plot(f_axis, np.unwrap(np.angle(film_fd_mod[:, 1])), label="film (model)")
plt.xlabel("Frequency (THz)")
plt.ylabel("Phase (rad)")

plt.figure("Spectrum")
plt.plot(f_axis, 20 * np.log10(np.abs(sub_fd_mod[:, 1])), label="Sub. (model)")
plt.plot(f_axis, 20 * np.log10(np.abs(film_fd_mod[:, 1])), label="Film (model)")
plt.xlabel("Frequency (THz)")
plt.ylabel("Amplitude (dB)")

plt.figure("Time domain")
# plt.plot(sam_td_sub_mod[:, 0], sam_td_sub_mod[:, 1], label="Sub. (model)")
plt.plot(sam_td_film_mod[:, 0], sam_td_film_mod[:, 1], label="film (model)")
plt.xlabel("Time (ps)")
plt.ylabel("Amplitude (arb. u.)")
"""
plt.figure("t_film exp vs mod real")
plt.plot(f_axis, t_exp_film.real)
plt.plot(f_axis, t_film_mod.real)
plt.xlabel("Frequency (THz)")
plt.ylabel("Transmission coefficient")

plt.figure("t_film exp vs mod imag")
plt.plot(f_axis, t_exp_film.imag)
plt.plot(f_axis, t_film_mod.imag)
plt.xlabel("Frequency (THz)")
plt.ylabel("Transmission coefficient")
"""
plt.figure("Conductivity")
plt.plot(f_axis[plot_range], sigma[plot_range].real, label="Exp (real)")
plt.plot(f_axis[plot_range], sigma[plot_range].imag, label="Exp (imag)")
plt.plot(f_axis[plot_range], sigma_fit[plot_range].real, label="Fit (real)")
plt.plot(f_axis[plot_range], sigma_fit[plot_range].imag, label="Fit (imag)")
plt.xlim((0.20, 2.55))
# plt.ylim((-10, 200))
plt.xlabel("Frequency (THz)")
plt.ylabel("Conductivity (S/cm)")
"""
plt.figure("Conductivity Tinkham")
plt.plot(f_axis, sigma.real, label="Exp (real)")
plt.plot(f_axis, sigma.imag, label="Exp (imag)")
plt.plot(f_axis, sigma_tnk.real, label="Tinkham (real)")
plt.plot(f_axis, sigma_tnk.imag, label="Tinkham (imag)")
plt.xlim((-0.05, 2.5))
plt.ylim((-10, 500))
plt.xlabel("Frequency (THz)")
plt.ylabel("Conductivity (S/cm)")
"""
plt_show()
