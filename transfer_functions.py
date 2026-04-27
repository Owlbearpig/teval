import numpy as np
from consts import c_thz


def model_1layer(n3_, d, f, n1=1, shift_=0):
    w_ = 2 * np.pi * f
    t_as = 2 * n1 / (n1 + n3_)
    t_sa = 2 * n3_ / (n1 + n3_)
    r_as = (n1 - n3_) / (n1 + n3_)
    r_sa = (n3_ - n1) / (n1 + n3_)

    exp = np.exp(1j * (d * w_ / c_thz) * n3_)
    e_sam = t_as * t_sa * exp / (1 + r_as * r_sa * exp ** 2)
    e_ref = np.exp(1j * (d * w_ / c_thz))

    phase_shift = np.exp(1j * shift_ * 1e-3 * w_)

    t = phase_shift * e_sam / e_ref

    return np.nan_to_num(t)


def model_2layer(n2_, n3_, h, d, f, n1=1, n4=1, shift_=0):
    # n3_ += 0.01
    w_ = 2 * np.pi * f
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

    phase_shift = np.exp(1j * shift_ * 1e-3 * w_) # 6, 16
    t = phase_shift * e_sam / e_ref

    return np.nan_to_num(t)
