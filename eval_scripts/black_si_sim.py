from tmm import coh_tmm
import numpy as np
from consts import c_thz
import matplotlib.pyplot as plt
from pathlib import Path
from functions import do_fft, do_ifft, window, flip_phase

data_dir = Path(r"/home/ftpuser/ftp/Data/Black_Si/NoPattern_Linescan2_NoNitrogen/")
ref_file = data_dir / r"2026-01-14T17-35-01.585294-BlackSi_NoPattern_NoN2_100avg-ref-X_132.000 mm-Y_16.000 mm.txt"

ref_td = np.loadtxt(ref_file)
ref_td = window(ref_td)

ref_fd = do_fft(ref_td)
ref_fd = flip_phase(ref_fd)

#### Transmission coefficient sim

pol = "s"
th_0 = 0 * np.pi / 180

def tmm_wrapper(freq_axis, d_list_, refr_idx_):
    refr_idx_arr = np.array(refr_idx_).T

    t = np.zeros_like(freq_axis, dtype=complex)
    for f_idx, freq_ in enumerate(freq_axis):
        lam_vac = c_thz / freq_
        n_list = refr_idx_arr[f_idx]

        t[f_idx] = coh_tmm(pol, n_list, d_list_, th_0, lam_vac)["t"]

    return t

# freqs = np.linspace(0.001, 10, 5000)
freqs = ref_fd[:, 0].real
w = 2 * np.pi * freqs

one = np.ones_like(freqs, dtype=complex)

d0, dBlackSi = 534, 10 # µm
n0 = 3.42 * one # Si ref_idx, no dispersion, no absorption

nB0 = 0.5 * 3.42 * one
nB1 = (1/3) * 3.42 * one

samples = {"Si": ([np.inf, d0, np.inf], [one, n0, one]),
           "BlackSi-1": ([np.inf, dBlackSi, d0-dBlackSi, np.inf], [one, nB0, n0, one]),
           "BlackSi-2": ([np.inf, dBlackSi, d0-dBlackSi, np.inf], [one, nB1, n0, one]),
           }

results = {}
for sample in samples:
    results[sample] = tmm_wrapper(freqs, *samples[sample])

fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, num="Transmission coefficient")
for sample in results:
    t = results[sample]
    t_amp = np.abs(t)
    t_phi = np.unwrap(np.angle(t))

    ax0.plot(freqs, t_amp, label=sample)
    ax1.plot(freqs, t_phi, label=sample)

ax0.set_ylabel("Amplitude")
ax1.set_ylabel("Phase (rad)")
ax1.set_xlabel("Frequency (THz)")

ax0.legend()
ax1.legend()

#### Measurement sim => sam_fd = t * ref_fd * np.exp(-1j*empty_phase(d0))
phase_empty = np.exp(-1j * d0 * w / c_thz)

fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, num="Measurements (Frequency domain)")
fig, ax2 = plt.subplots(nrows=1, ncols=1, num="Measurements (Time domain)")
for sample in results:
    t = results[sample]

    sam_fd = np.array([freqs, t * ref_fd[:, 1] * phase_empty]).T

    sam_td = do_ifft(sam_fd, conj=False)

    y_fd = sam_fd[:, 1]
    t, y_td = sam_td[:, 0], sam_td[:, 1]
    y_td = np.flip(y_td)

    y_amp = np.abs(y_fd)
    y_phi = np.unwrap(np.angle(y_fd))

    ax0.plot(freqs, 20*np.log10(y_amp), label=sample)
    ax1.plot(freqs, y_phi, label=sample)

    ax2.plot(t, y_td, label=sample)

ax0.set_ylabel("Amplitude (dB)")
ax1.set_ylabel("Phase (rad)")
ax1.set_xlabel("Frequency (THz)")

ax0.plot(ref_fd[:, 0], 20*np.log10(np.abs(ref_fd[:, 1])), label="Reference")
ax1.plot(ref_fd[:, 0], np.unwrap(np.angle(ref_fd[:, 1])), label="Reference")
ax2.plot(ref_td[:, 0], ref_td[:, 1], label="Reference")
ax2.set_ylabel("Amplitude")
ax2.set_xlabel("Time (ps)")

ax0.legend()
ax1.legend()
ax2.legend()
plt.show()
