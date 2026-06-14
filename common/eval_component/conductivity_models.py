from numpy import sqrt
from common.consts import eps0_thz, pi

def sigma_to_n(freq, sig):
    # [eps0_thz] = ps * S / µm
    # sig_ in S/cm -> S/µm ( 1/(1e6 µm) = 1/m = 1/(1e2 cm) => 1e-4/µm = 1/cm)
    sig *= 1e-4

    w = 2*pi*freq

    sig *= 1e-4 # S/cm -> S/µm
    n = sqrt(1 + 1j * sig / (eps0_thz * w))

    return n

def n_to_sigma(freq, n):
    # [eps0_thz] = ps * S / µm
    # sig_ in S/cm -> S/µm ( 1/(1e6 µm) = 1/m = 1/(1e2 cm) => 1e-4/µm = 1/cm)

    w = 2 * pi * freq

    sig = -1j*(n ** 2 - 1) * w * eps0_thz
    sig *= 1e4 # S/µm -> S/cm

    return sig

def drude(freq, sig0, tau):
    # [tau] = fs, [sig0] = S/cm,
    # => [sig_cc_] = S/cm
    tau *= 1e-3  # fs = 1e-3 ps
    tau /= 2 * pi

    scale = 1
    sig0 *= scale

    w = 2 * pi * freq
    sig_cc_ = sig0 / (1 - 1j * tau * w)

    return sig_cc_


def drude2(freq_, sig0, tau, wp, eps_inf):
    tau *= 1e-3
    tau /= 2 * pi
    w = 2 * pi * freq_

    return sig0 * wp ** 2 / (tau - 1j * w) - 1j * eps0_thz * w * (eps_inf - 1)

def drude_smith(freq_, sig0, tau, c1):
    tau *= 1e-3
    tau /= 2 * pi
    return sig0 * c1 / (tau - 1j * tau) # TODO CHECK THIS!!!

def lattice_contrib(freq_, tau, wp, eps_s, eps_inf):
    tau *= 1e-3  # fs = 1e-3 ps
    tau /= 2 * pi

    wp *= 2 * pi

    w = 2 * pi * freq_

    eps_l = eps_inf - ((eps_s - eps_inf) * wp ** 2) / (w ** 2 - wp ** 2 + 1j * w / tau)

    return eps_l


def total_response(freq, sig0, tau, wp, eps_s, eps_inf, c1):
    # [freq] = THz, [tau] = fs, [sig0] = S/cm, [wp] = THz. Dimensionless: eps_inf, eps_s
    sig_cc = drude(freq, sig0, tau)
    # sig_cc = drude_smith(freq, tau, sig0, c1)
    eps_l = lattice_contrib(freq, tau, wp, eps_s, eps_inf)

    n = sqrt(eps_l)

    sig_tot = n_to_sigma(freq, n) + sig_cc

    return sig_tot
