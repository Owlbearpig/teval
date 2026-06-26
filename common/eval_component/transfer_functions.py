import numpy as np
from common.consts import c_thz
from common.eval_component.tmm_impl import coh_tmm

# n is the free parameter of the unknown layer

def t_tmm_model_1layer(n, freq, **opt_kwargs):
    d = opt_kwargs["d"]
    pol = "s"
    n_list = [1, n, 1]
    d_list = [np.inf, d, np.inf]
    th_0 = 0 * np.pi / 180
    lam_vac = c_thz / freq
    w_ = 2 * np.pi * freq

    e_sam = coh_tmm(pol, n_list, d_list, th_0, lam_vac)
    e_ref = np.exp(1j * (d * w_ / c_thz))

    t = e_sam / e_ref

    return np.nan_to_num(t)

def t_tmm_model_2layer(n, freq, **opt_kwargs):
    d = opt_kwargs["d"]
    n_sub = opt_kwargs["n_sub"]
    h = opt_kwargs["h"]
    n1 = opt_kwargs["n1"]
    n4 = opt_kwargs["n4"]

    pol = "s"
    n_list = [n1, n, n_sub, n4]
    d_list = [np.inf, h, d, np.inf]
    th_0 = 0 * np.pi / 180
    lam_vac = c_thz / freq
    w_ = 2 * np.pi * freq

    e_sam = coh_tmm(pol, n_list, d_list, th_0, lam_vac)
    e_ref = np.exp(1j * ((d + h) * w_ / c_thz))

    t = e_sam / e_ref

    return np.nan_to_num(t)

def model_1layer(n, freq, **opt_kwargs):
    d = opt_kwargs["d"]
    nfp = opt_kwargs["nfp"]
    n1 = opt_kwargs["n1"]

    w_ = 2 * np.pi * freq
    t_as = 2 * n1 / (n1 + n)
    t_sa = 2 * n / (n1 + n)
    r_as = (n1 - n) / (n1 + n)
    r_sa = (n - n1) / (n1 + n)

    """
    exp = np.exp(1j * (d * w_ / c_thz) * n3_)
    e_sam = t_as * t_sa * exp / (1 + r_as * r_sa * exp ** 2)
    e_ref = np.exp(1j * (d * w_ / c_thz))

    t = e_sam / e_ref
    """
    #"""
    exp1 = np.exp(1j * (d * w_ / c_thz) * (n - 1))
    exp2 = np.exp(1j * 2 * (d * w_ / c_thz) * n)

    s = 0
    for i in range(nfp):
        s += (r_as**2 * exp2)**i

    t = (1 - r_as ** 2) * exp1 * s
    #"""
    return np.nan_to_num(t)


def model_2layer(n, freq, **opt_kwargs):
    d = opt_kwargs["d"]
    n_sub = opt_kwargs["n_sub"]
    h = opt_kwargs["h"]
    n1 = opt_kwargs["n1"]
    n4 = opt_kwargs["n4"]

    w_ = 2 * np.pi * freq
    t12 = 2 * n1 / (n1 + n)
    t23 = 2 * n / (n + n_sub)
    t34 = 2 * n_sub / (n_sub + n4)

    r12 = (n1 - n) / (n1 + n)
    r23 = (n - n_sub) / (n + n_sub)
    r34 = (n_sub - n4) / (n_sub + n4)

    exp1 = np.exp(1j * (h * w_ / c_thz) * n)
    exp2 = np.exp(1j * (d * w_ / c_thz) * n_sub)

    e_sam = t12 * t23 * t34 * exp1 * exp2 / (1 + r12 * r23 * exp1 ** 2 + r23 * r34 * exp2 ** 2 + r12 * r34 * exp1 ** 2 * exp2 ** 2)
    e_ref = np.exp(1j * ((d + h) * w_ / c_thz))

    t = e_sam / e_ref

    return np.nan_to_num(t)

def _t_model_2layer(n, freq, **opt_kwargs):
    d = opt_kwargs["d"]
    n_sub = opt_kwargs["n_sub"]
    h = opt_kwargs["h"]
    n1 = opt_kwargs["n1"]
    n4 = opt_kwargs["n4"]
    nfp = opt_kwargs["nfp"]

    w_ = 2 * np.pi * freq
    t12 = 2 * n1 / (n1 + n)
    t23 = 2 * n / (n + n_sub)
    t34 = 2 * n_sub / (n_sub + n4)

    r12 = (n1 - n) / (n1 + n)
    r23 = (n - n_sub) / (n + n_sub)
    r34 = (n_sub - n4) / (n_sub + n4)

    exp1 = np.exp(1j * (h * w_ / c_thz) * n)
    exp2 = np.exp(1j * (d * w_ / c_thz) * n_sub)

    # r in geometric series
    r_geo = r12 * r23 * exp1 ** 2 + r23 * r34 * exp2 ** 2 + r12 * r34 * exp1 ** 2 * exp2 ** 2
    if nfp == -1:
        fp_factor = 1 / (1 + r_geo)
    else:
        fp_factor = 0
        for fp_idx in range(0, nfp + 1):
            fp_factor += r_geo ** fp_idx

    e_sam = t12 * t23 * t34 * exp1 * exp2 * fp_factor
    e_ref = np.exp(1j * ((d + h) * w_ / c_thz))

    t = e_sam / e_ref

    return np.nan_to_num(t)


def transferfunction_error(sam_fd, ref_fd, ref_fd_std, sam_fd_std, noise_freq=5.0):
    # H=Esam/Eref=(a+ib)/(c+id)
    # f=Re(H)
    # g=Im(H)
    a = sam_fd[:, 1].real
    b = sam_fd[:, 1].imag
    c = ref_fd[:, 1].real
    d = ref_fd[:, 1].imag

    noiseposition = np.argmin(np.abs(ref_fd[:, 0] - noise_freq))

    anoise = a[noiseposition::]
    bnoise = b[noiseposition::]
    cnoise = c[noiseposition::]
    dnoise = d[noiseposition::]

    anstd = np.std(anoise)
    bnstd = np.std(bnoise)
    cnstd = np.std(cnoise)
    dnstd = np.std(dnoise)

    Astd = sam_fd_std[:, 1].real
    Bstd = sam_fd_std[:, 1].imag
    Cstd = ref_fd_std[:, 1].real
    Dstd = ref_fd_std[:, 1].imag

    DeltaA = (Astd * Astd + anstd * anstd) ** 0.5
    DeltaB = (Bstd * Bstd + bnstd * bnstd) ** 0.5
    DeltaC = (Cstd * Cstd + cnstd * cnstd) ** 0.5
    DeltaD = (Dstd * Dstd + dnstd * dnstd) ** 0.5

    dfda = c / (c*c + d*d)
    dfdb = d / (c*c + d*d)
    dfdc = (a*d*d - a*c*c -2*b*c*d) / (c*c + d*d)**2
    dfdd = (b*c*d-b*d*d-2*a*c*d) / (c*c+d*d)**2

    dgda = -d / (c*c + d*d)
    dgdb = c / (c*c+d*d)
    dgdc = -(b*c*c -b*d*d - 2*a*c*d) / (c*c+d*d)**2
    dgdd = (a*d*d - a*c*c-2*b*c*d) / (c*c+d*d)**2

    DeltaRealH = ((DeltaA * dfda) ** 2 + (DeltaB * dfdb) ** 2 + (DeltaC * dfdc) ** 2 + (DeltaD * dfdd) ** 2) ** 0.5
    DeltaImagH = ((DeltaA * dgda) ** 2 + (DeltaB * dgdb) ** 2 + (DeltaC * dgdc) ** 2 + (DeltaD * dgdd) ** 2) ** 0.5

    return DeltaRealH + 1j * DeltaImagH

def dtdn(n, d, freq):
    w_ = 2 * np.pi * freq
    b = 1j * (d * w_ / c_thz)
    e = np.exp(b * n)
    f = 4*n*e
    df = 4*(1+n*b)*e
    g = (1+n)**2 - (1-n)**2 * e**2
    dg = 2*(1+n) - 2*(1-n)*e**2 - (1-n)**2*2*b*e**2

    return (df*g-f*dg) / (g*g)

def dtdd(n, d, freq):
    w_ = 2 * np.pi * freq
    b = 1j * (d * w_ / c_thz)
    e = np.exp(b * n)
    f = 4 * n * e
    g = (1 + n) ** 2 - (1 - n) ** 2 * e ** 2
    dgdd = -(1-n)**2 * e**2 * 2*b*n/d
    dfdd = f*b/d

    return (dfdd*g-f*dgdd) / (g*g)

