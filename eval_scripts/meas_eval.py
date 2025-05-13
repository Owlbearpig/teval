import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0
from dataset import DataSet

sig_v2o5_20degC_amorph = 5e-7 # S/cm
sig_v2o5_20degC_amorph = 80 # S/cm

n1, n4 = 1, 1
n3 = 3.07726 + 4.144e-003j # 1.25 THz
# n3 = 3.07695 + 4.373e-003j # 1.25 THz
# n3 = 3.080 + 3.904e-3*1j # 1.5 THz
# n3 = 3.086 + 5.1542e-3*1j # 2.0 THz
f0 = 1.25 * 1e12
h = 300 * 1e-9
d0 = 645 * 1e-6

sigma0 = sig_v2o5_20degC_amorph * 100 # conversion to S/m
sigma0 = 2485.10

def load_teralyzer_result():
    pass

def model_1layer(d=d0, f=f0):
    w = 2 * pi * f
    t_as = 2 * n1 / (n1 + n3)
    t_sa = 2 * n3 / (n1 + n3)
    r_as = (n1 - n3) / (n1 + n3)
    r_sa = (n3 - n1) / (n1 + n3)

    exp = np.exp(1j * (d * w / c) * n3)

    t_abs = np.abs(t_as * t_sa * exp / (1 + r_as * r_sa * exp**2))

    return np.nan_to_num(t_abs)

def model_2layer(sigma=sigma0, f=f0):
    w = 2 * pi * f
    n2 = (1+1j) * np.sqrt(sigma / (4 * pi * epsilon_0 * f))
    t12 = 2 * n1 / (n1 + n2)
    t23 = 2 * n2 / (n2 + n3)
    t34 = 2 * n3 / (n3 + n4)

    r12 = (n1 - n2) / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)
    r34 = (n3 - n4) / (n3 + n4)

    exp1 = np.exp(1j * (h * w / c) * n2)
    exp2 = np.exp(1j * (d0 * w / c) * n3)

    t_abs = np.abs(t12 * t23 * t34 * exp1 * exp2 / (1 + r12*r23*exp1**2 + r23*r34*exp2**2 + r12*r34*exp1**2*exp2**2))

    return np.nan_to_num(t_abs)

d_arr = np.linspace(250, 750, 10000) * 1e-6
f_arr = np.linspace(0.25, 3.5, 10000) * 1e12

t_abs_arr_d = np.zeros_like(d_arr)
for i, d_ in enumerate(d_arr):
    t_abs_arr_d[i] = model_1layer(d=d_)

t_abs_arr_f = np.zeros_like(f_arr)
for i, f_ in enumerate(f_arr):
    t_abs_arr_f[i] = model_1layer(f=f_)

t_abs_film_arr_f = np.zeros_like(f_arr)
for i, f_ in enumerate(f_arr):
    t_abs_film_arr_f[i] = model_2layer(f=f_)

## measurement
options = {
    "ref_pos": (5, None), # img5
}
dataset = DataSet(r"/home/ftpuser/ftp/Data/Furtwangen/Vanadium Oxide/img5", options)
dataset.select_freq(f0/1e12)
res_sub = dataset.plot_point((70, -4), apply_window=False)
res_film = dataset.plot_point((25, 2.5), apply_window=False)
# dataset.plot_image()

plt.figure()
plt.plot(d_arr * 1e6, t_abs_arr_d)
plt.xlabel("d_sub (µm)")
plt.ylabel("t_abs at 1.5 THz")

plt.figure()
plt.plot(f_arr * 1e-12, 100*t_abs_arr_f, label="Model (sub)")
plt.plot(f_arr * 1e-12, 100*t_abs_film_arr_f, label="Model (film)")
plt.plot(res_sub["freq_axis"], 100 * (1 / res_sub["absorb"]), label="Experiment (sub)")
plt.plot(res_sub["freq_axis"], 100 * (1 / res_film["absorb"]), label="Experiment (film)")
plt.xlabel("frequency (THz)")
plt.ylabel("t_abs for d=645 µm (\\%)")
plt.ylim((-5, 110))
plt.legend()

print("Sub transmission: ", model_1layer())

f0_idx = np.argmin(np.abs(res_sub["freq_axis"] - f0 / 1e12))
x = np.linspace(0, 2, 20000) * 1e4

best_fit_val, best_fit = np.inf, None
for sig_ in x:
    t_mod = model_2layer(sigma=sig_)
    t_exp = (1 / res_film["absorb"])[f0_idx]
    diff = (t_mod - t_exp)**2
    if diff < best_fit_val:
        best_fit = sig_
        best_fit_val = diff

print(best_fit)

#plt.plot(x, y)
plt.show()
