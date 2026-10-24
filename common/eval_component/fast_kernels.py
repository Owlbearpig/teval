# Compiled (numba) versions of the hot parts of a q-space fit for the single layer model (t_tmm_model_1layer):
# the model itself, the cost scan initial guess and the per frequency Newton refinement. Same algorithms as the numpy
# versions (n_guess.cost_scan_n_guess, t_fit_global.refine_n_per_frequency), results agree to rounding. Used by the
# worker processes when numba is installed and "Use compiled kernels" is enabled, everything else keeps the numpy code.
import numpy as np

from common.consts import c_thz
from common.eval_component.transfer_functions import t_tmm_model_1layer

try:
    from numba import njit, vectorize, complex128, float64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def fast_path_available(transmission_model):
    return NUMBA_AVAILABLE and transmission_model is t_tmm_model_1layer


if NUMBA_AVAILABLE:
    _FLOAT_MAX = np.finfo(float).max

    @njit(cache=True)
    def _nan_to_num(x):
        if np.isnan(x):
            return 0.0
        if np.isinf(x):
            return _FLOAT_MAX if x > 0 else -_FLOAT_MAX
        return x

    @njit(cache=True)
    def _slab(n, f, d, shift):
        # t_tmm_model_1layer for one frequency: coh_tmm_wrapper("s", [1, n, 1], [inf, d, inf], 0, c / f) relative to
        # the same distance in air, conjugated, time shifted, nan_to_num
        t01 = 2.0 / (1.0 + n)
        r01 = (1.0 - n) / (1.0 + n)
        t12 = 2.0 * n / (n + 1.0)
        r12 = (n - 1.0) / (n + 1.0)
        kz_d = 2.0 * np.pi * n / (c_thz / f) * d
        e_m = np.exp(-1j * kz_d)
        e_p = np.exp(1j * kz_d)
        m00 = (1.0 / t01) * (e_m / t12) + (r01 / t01) * (r12 * e_p / t12)
        t = 1.0 / m00
        e_ref = np.exp(1j * (d * 2.0 * np.pi * f / c_thz))
        t = np.conj(t / e_ref) * np.exp(1j * 2.0 * np.pi * (shift * 1e-3) * f)
        return complex(_nan_to_num(t.real), _nan_to_num(t.imag))

    @vectorize([complex128(complex128, float64, float64, float64)], cache=True)
    def _slab_ufunc(n, f, d, shift):
        return _slab(n, f, d, shift)

    def t_slab_model(n, freq, **kwargs):
        # drop-in for t_tmm_model_1layer, broadcasts like it
        return _slab_ufunc(np.asarray(n, dtype=complex), np.asarray(freq, dtype=float),
                           float(kwargs["d"]), float(kwargs["shift"]))

    t_slab_model.layer_cnt = 1

    @njit(cache=True)
    def _phi_cost_sum(t_exp, freq, n, d, shift):
        # sum of phi_cost_fun over the frequencies for a constant n
        s = 0.0
        for i in range(freq.shape[0]):
            a = np.angle(t_exp[i] / _slab(n + 0j, freq[i], d, shift))
            s += a * a
        return s

    @njit(cache=True)
    def _fminbound(t_exp, freq, d, shift, a, b, xatol=1e-5, maxfun=500):
        # scipy.optimize.minimize_scalar(method="bounded") of _phi_cost_sum (Brent's method on a bounded interval)
        sqrt_eps = np.sqrt(2.2e-16)
        golden_mean = 0.5 * (3.0 - np.sqrt(5.0))
        fulc = a + golden_mean * (b - a)
        nfc, xf = fulc, fulc
        rat = e = 0.0
        x = xf
        fx = _phi_cost_sum(t_exp, freq, x, d, shift)
        num = 1
        ffulc = fnfc = fx
        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1
        while abs(xf - xm) > (tol2 - 0.5 * (b - a)):
            golden = True
            if abs(e) > tol1:
                golden = False
                r = (xf - nfc) * (fx - ffulc)
                q = (xf - fulc) * (fx - fnfc)
                p = (xf - fulc) * q - (xf - nfc) * r
                q = 2.0 * (q - r)
                if q > 0.0:
                    p = -p
                q = abs(q)
                r = e
                e = rat
                if (abs(p) < abs(0.5 * q * r)) and (p > q * (a - xf)) and (p < q * (b - xf)):
                    rat = p / q
                    x = xf + rat
                    if ((x - a) < tol2) or ((b - x) < tol2):
                        rat = tol1 * (np.sign(xm - xf) + ((xm - xf) == 0))
                else:
                    golden = True
            if golden:
                e = a - xf if xf >= xm else b - xf
                rat = golden_mean * e
            x = xf + (np.sign(rat) + (rat == 0)) * max(abs(rat), tol1)
            fu = _phi_cost_sum(t_exp, freq, x, d, shift)
            num += 1
            if fu <= fx:
                if x >= xf:
                    a = xf
                else:
                    b = xf
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = xf, fx
                xf, fx = x, fu
            else:
                if x < xf:
                    a = x
                else:
                    b = x
                if (fu <= fnfc) or (nfc == xf):
                    fulc, ffulc = nfc, fnfc
                    nfc, fnfc = x, fu
                elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                    fulc, ffulc = x, fu
            xm = 0.5 * (a + b)
            tol1 = sqrt_eps * abs(xf) + xatol / 3.0
            tol2 = 2.0 * tol1
            if num >= maxfun:
                break
        return xf

    @njit(cache=True)
    def _cost_scan_core(t_exp, freq, d, shift, n_axis, n_step, k_axis):
        # n_guess.cost_scan_n_guess up to the k smoothing: constant n from the phase cost (scan, then bounded
        # minimization), k per frequency from the amplitude cost
        best, n_coarse = np.inf, n_axis[0]
        for n_ in n_axis:
            c = _phi_cost_sum(t_exp, freq, n_, d, shift)
            if c < best:
                best, n_coarse = c, n_
        n_real = _fminbound(t_exp, freq, d, shift, n_coarse - n_step, n_coarse + n_step)

        k = np.empty(freq.shape[0])
        for i in range(freq.shape[0]):
            best, k_best = np.inf, k_axis[0]
            for k_ in k_axis:
                c = abs(t_exp[i]) - abs(_slab(n_real + 1j * k_, freq[i], d, shift))
                c = c * c
                if c < best:
                    best, k_best = c, k_
            k[i] = k_best
        return n_real, k

    def cost_scan_n_guess(d, shift, config_dict):
        # same interface and result as n_guess.cost_scan_n_guess
        freq_axis = np.asarray(config_dict["freq_axis"], dtype=float)
        t_exp = np.asarray(config_dict["t_exp"])[:, 1].astype(complex)
        n_min, n_max = config_dict["n_scan_range"]
        n_step = config_dict["n_scan_step"]
        k_max = config_dict["k_scan_max"]
        n_axis = np.arange(n_min, n_max + n_step, n_step)
        k_cnt = 100
        k_axis = np.linspace(k_max / k_cnt, k_max, k_cnt)

        n_real, k = _cost_scan_core(t_exp, freq_axis, float(d), float(shift), n_axis, float(n_step), k_axis)
        k = np.clip(np.polyval(np.polyfit(freq_axis, k, 2), freq_axis), k_axis[0], k_max)

        return np.column_stack((freq_axis, n_real + 1j * k))

    @njit(cache=True)
    def _refine_core(n, freq, t_exp, d, shift, max_iter, max_step, tol):
        h = 1e-7
        out = n.copy()
        converged = np.zeros(n.shape[0], dtype=np.bool_)
        for i in range(n.shape[0]):
            f, te, n_i = freq[i], t_exp[i], n[i]
            r_ref = abs(_slab(n_i, f, d, shift) - te)
            step_size, det = np.inf, 1.0
            for _ in range(max_iter):
                r = _slab(n_i, f, d, shift) - te
                dr_dn = (_slab(n_i + h, f, d, shift) - _slab(n_i - h, f, d, shift)) / (2 * h)
                dr_dk = (_slab(n_i + 1j * h, f, d, shift) - _slab(n_i - 1j * h, f, d, shift)) / (2 * h)
                det = dr_dn.real * dr_dk.imag - dr_dk.real * dr_dn.imag
                if abs(det) < 1e-30:
                    det = np.nan
                    step = 0j
                else:
                    step = complex((dr_dk.imag * r.real - dr_dk.real * r.imag) / det,
                                   (dr_dn.real * r.imag - dr_dn.imag * r.real) / det)
                step_size = abs(step)
                if step_size > max_step:
                    step = max_step * step / max(step_size, 1e-30)
                n_i = n_i - step
                if step_size < tol:
                    break
            worse = (not (np.isfinite(n_i.real) and np.isfinite(n_i.imag))
                     or abs(_slab(n_i, f, d, shift) - te) > r_ref)
            out[i] = n[i] if worse else n_i
            converged[i] = (not worse) and step_size < 1e-6 and np.isfinite(det)
        return out, converged

    def refine_n_per_frequency(n, freq_axis, t_exp, d, shift, max_iter=50, max_step=0.01, tol=1e-10):
        # t_fit_global.refine_n_per_frequency for the single layer model, iterates each frequency until it converged
        return _refine_core(np.asarray(n, dtype=complex), np.asarray(freq_axis, dtype=float),
                            np.asarray(t_exp, dtype=complex), float(d), float(shift), max_iter, max_step, tol)
