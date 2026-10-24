from tmm import array, list_snell, seterr, cos, zeros, interface_r, interface_t, make_2x2_array, exp
import numpy as np
from common.consts import c_thz

def _is_forward_angle(n, theta):
    # vectorized version of tmm.is_forward_angle
    ncostheta = n * np.cos(theta)
    return np.where(np.abs(ncostheta.imag) > 100 * np.finfo(float).eps, ncostheta.imag > 0, ncostheta.real > 0)


def coh_tmm_wrapper(pol, n_list, d_list, th_0, lam_vac):
    # same result as coh_tmm, but vectorized: lam_vac and each entry of n_list can be arrays of any
    # broadcastable shape (e.g. frequencies, or scan points x frequencies)
    lam_vac = np.asarray(lam_vac, dtype=float)
    n_arr = np.array(np.broadcast_arrays(*[np.asarray(n, dtype=complex) for n in n_list], lam_vac)[:-1])
    d_list = np.asarray(d_list, dtype=float)
    num_layers = len(n_arr)

    if pol not in ("s", "p"):
        raise ValueError("Polarization must be 's' or 'p'")

    with np.errstate(invalid="ignore", over="ignore"):
        if th_0 == 0:
            # normal incidence: every angle is 0 (pi for a backward outer medium), skips arcsin / cos of the
            # complex arrays which dominate the cost of the scan grids
            cos_arr = np.ones(n_arr.shape, dtype=complex)
            for i in (0, -1):
                cos_arr[i] = np.where(_is_forward_angle(n_arr[i], 0.0), 1.0, -1.0)
        else:
            th_arr = np.arcsin(n_arr[0] * np.sin(th_0) / n_arr)
            for i in (0, -1):
                th_arr[i] = np.where(_is_forward_angle(n_arr[i], th_arr[i]), th_arr[i], np.pi - th_arr[i])
            cos_arr = np.cos(th_arr)

        kz_arr = 2 * np.pi * n_arr * cos_arr / lam_vac

        def interface(i):
            # Fresnel amplitudes, same as tmm.interface_t / interface_r with the cosines computed once
            n_i, n_f, c_i, c_f = n_arr[i], n_arr[i + 1], cos_arr[i], cos_arr[i + 1]
            if pol == "s":
                denom = n_i * c_i + n_f * c_f
                return 2 * n_i * c_i / denom, (n_i * c_i - n_f * c_f) / denom
            denom = n_f * c_i + n_i * c_f
            return 2 * n_i * c_i / denom, (n_f * c_i - n_i * c_f) / denom

        # Mtilde = (1/t01) [[1, r01], [r01, 1]] @ M_1 @ ... @ M_(num_layers-2), elementwise 2x2 products
        t_, r_ = interface(0)
        m00, m01, m10, m11 = 1 / t_, r_ / t_, r_ / t_, 1 / t_
        for i in range(1, num_layers - 1):
            t_, r_ = interface(i)
            e_m, e_p = np.exp(-1j * kz_arr[i] * d_list[i]), np.exp(1j * kz_arr[i] * d_list[i])
            a00, a01, a10, a11 = e_m / t_, r_ * e_m / t_, r_ * e_p / t_, e_p / t_
            m00, m01, m10, m11 = (m00 * a00 + m01 * a10, m00 * a01 + m01 * a11,
                                  m10 * a00 + m11 * a10, m10 * a01 + m11 * a11)

    return 1 / m00

def coh_tmm(pol, n_list, d_list, th_0, lam_vac):

    # Convert lists to numpy arrays if they're not already.
    n_list = array(n_list)
    d_list = array(d_list, dtype=float)

    num_layers = n_list.size

    # th_list is a list with, for each layer, the angle that the light travels
    # through the layer. Computed with Snell's law. Note that the "angles" may be
    # complex!
    th_list = list_snell(n_list, th_0)

    # kz is the z-component of (complex) angular wavevector for forward-moving
    # wave. Positive imaginary part means decaying.
    kz_list = 2 * np.pi * n_list * cos(th_list) / lam_vac

    # delta is the total phase accrued by traveling through a given layer.
    # Ignore warning about inf multiplication
    olderr = seterr(invalid='ignore')
    delta = kz_list * d_list
    seterr(**olderr)


    # t_list[i,j] and r_list[i,j] are transmission and reflection amplitudes,
    # respectively, coming from i, going to j. Only need to calculate this when
    # j=i+1. (2D array is overkill but helps avoid confusion.)
    t_list = zeros((num_layers, num_layers), dtype=complex)
    r_list = zeros((num_layers, num_layers), dtype=complex)
    for i in range(num_layers-1):
        t_list[i,i+1] = interface_t(pol, n_list[i], n_list[i+1],
                                    th_list[i], th_list[i+1])
        r_list[i,i+1] = interface_r(pol, n_list[i], n_list[i+1],
                                    th_list[i], th_list[i+1])
    # At the interface between the (n-1)st and nth material, let v_n be the
    # amplitude of the wave on the nth side heading forwards (away from the
    # boundary), and let w_n be the amplitude on the nth side heading backwards
    # (towards the boundary). Then (v_n,w_n) = M_n (v_{n+1},w_{n+1}). M_n is
    # M_list[n]. M_0 and M_{num_layers-1} are not defined.
    # My M is a bit different than Sernelius's, but Mtilde is the same.
    M_list = zeros((num_layers, 2, 2), dtype=complex)
    for i in range(1, num_layers-1):
        M_list[i] = (1/t_list[i,i+1]) * np.dot(
            make_2x2_array(exp(-1j*delta[i]), 0, 0, exp(1j*delta[i]),
                           dtype=complex),
            make_2x2_array(1, r_list[i,i+1], r_list[i,i+1], 1, dtype=complex))
    Mtilde = make_2x2_array(1, 0, 0, 1, dtype=complex)
    for i in range(1, num_layers-1):
        Mtilde = np.dot(Mtilde, M_list[i])
    Mtilde = np.dot(make_2x2_array(1, r_list[0,1], r_list[0,1], 1,
                                   dtype=complex)/t_list[0,1], Mtilde)

    # Net complex transmission amplitudes
    t = 1/Mtilde[0,0]

    return t

if __name__ == '__main__':
    def shift_t(freq, t, shift=0.0):
        return t * np.exp(1j * 2 * np.pi * (shift * 1e-3) * freq)

    def t_tmm_model_1layer(n, freq, **opt_kwargs):
        d = opt_kwargs["d"]
        shift = opt_kwargs["shift"]

        pol = "s"
        n_list = [1, n, 1]
        d_list = [np.inf, d, np.inf]
        th_0 = 0 * np.pi / 180
        lam_vac = c_thz / freq
        w_ = 2 * np.pi * freq

        e_sam = coh_tmm_wrapper(pol, n_list, d_list, th_0, lam_vac)
        e_ref = np.exp(1j * (d * w_ / c_thz))

        t = e_sam / e_ref

        t = shift_t(freq, t, shift)

        return np.nan_to_num(t)


