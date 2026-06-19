import numpy as np
from common.consts import c_thz
from common.eval_component.tmm_impl import coh_tmm


def t_tmm_model_1layer(freq, n3_r, n3_i, d):
    pol = "s"
    n3_ = n3_r + 1j * n3_i
    n_list = [1, n3_, 1]
    d_list = [np.inf, d, np.inf]
    th_0 = 0 * np.pi / 180
    lam_vac = c_thz / freq
    w_ = 2 * np.pi * freq

    e_sam = coh_tmm(pol, n_list, d_list, th_0, lam_vac)
    e_ref = np.exp(1j * (d * w_ / c_thz))

    t = e_sam / e_ref

    return np.nan_to_num(t)

def t_tmm_model_2layer(freq, n2_r, n2_i, n3_r, n3_i, h, d, n1=1, n4=1):
    pol = "s"
    n2_ = n2_r + 1j * n2_i
    n3_ = n3_r + 1j * n3_i
    n_list = [n1, n2_, n3_, n4]
    d_list = [np.inf, h, d, np.inf]
    th_0 = 0 * np.pi / 180
    lam_vac = c_thz / freq
    w_ = 2 * np.pi * freq

    e_sam = coh_tmm(pol, n_list, d_list, th_0, lam_vac)
    e_ref = np.exp(1j * ((d + h) * w_ / c_thz))

    t = e_sam / e_ref

    return np.nan_to_num(t)

def model_1layer(freq, n3_r, n3_i, d, nfp=3, n1=1):
    n3_ = n3_r + 1j * n3_i
    w_ = 2 * np.pi * freq
    t_as = 2 * n1 / (n1 + n3_)
    t_sa = 2 * n3_ / (n1 + n3_)
    r_as = (n1 - n3_) / (n1 + n3_)
    r_sa = (n3_ - n1) / (n1 + n3_)

    """
    exp = np.exp(1j * (d * w_ / c_thz) * n3_)
    e_sam = t_as * t_sa * exp / (1 + r_as * r_sa * exp ** 2)
    e_ref = np.exp(1j * (d * w_ / c_thz))

    t = e_sam / e_ref
    """
    #"""
    exp1 = np.exp(1j * (d * w_ / c_thz) * (n3_ - 1))
    exp2 = np.exp(1j * 2 * (d * w_ / c_thz) * n3_)

    s = 0
    for i in range(nfp):
        s += (r_as**2 * exp2)**i

    t = (1 - r_as ** 2) * exp1 * s
    #"""
    return np.nan_to_num(t)


def model_2layer(freq, n2_r, n2_i, n3_r, n3_i, h, d, n1=1, n4=1):
    n2_ = n2_r + 1j * n2_i
    n3_ = n3_r + 1j * n3_i
    w_ = 2 * np.pi * freq
    t12 = 2 * n1 / (n1 + n2_)
    t23 = 2 * n2_ / (n2_ + n3_)
    t34 = 2 * n3_ / (n3_ + n4)

    r12 = (n1 - n2_) / (n1 + n2_)
    r23 = (n2_ - n3_) / (n2_ + n3_)
    r34 = (n3_ - n4) / (n3_ + n4)

    exp1 = np.exp(1j * (h * w_ / c_thz) * n2_)
    exp2 = np.exp(1j * (d * w_ / c_thz) * n3_)

    e_sam = t12 * t23 * t34 * exp1 * exp2 / (1 + r12 * r23 * exp1 ** 2 + r23 * r34 * exp2 ** 2 + r12 * r34 * exp1 ** 2 * exp2 ** 2)
    e_ref = np.exp(1j * ((d + h) * w_ / c_thz))

    t = e_sam / e_ref

    return np.nan_to_num(t)

def _t_model_2layer(freq, n_sub_r, n_sub_i, n_film_r, n_film_i, d, h, nfp):
    n_sub = n_sub_r + 1j * n_sub_i
    n_film = n_film_r + 1j * n_film_i
    n1, n4 = 1, 1

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

