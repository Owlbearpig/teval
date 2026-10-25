"""
Sharpening of images by deconvolution with the beam profile. The measured value of a pixel is (approximately) the
sample property weighted with the intensity profile of the focus, a Gaussian exp(-2 r^2 / w^2) with the 1/e^2
intensity radius w that the knife-edge fit of the integrated power gives.
"""
from enum import Enum
import numpy as np
from scipy.signal import fftconvolve


class DeconvolutionMethod(Enum):
    richardson_lucy = "Richardson-Lucy (iterative)"
    wiener = "Wiener filter"


def gaussian_psf(dx, dy, beam_radius):
    # normalized intensity profile on the pixel grid, out to 2 w (beyond: < e^-8 of the peak)
    rx, ry = max(1, int(np.ceil(2 * beam_radius / dx))), max(1, int(np.ceil(2 * beam_radius / dy)))
    x = np.arange(-rx, rx + 1) * dx
    y = np.arange(-ry, ry + 1) * dy
    psf = np.exp(-2 * (x[:, None] ** 2 + y[None, :] ** 2) / beam_radius ** 2)
    return psf / psf.sum()


def _convolve(img, psf):
    return fftconvolve(img, psf, mode="same")


def richardson_lucy(img, valid, psf, iterations):
    # masked Richardson-Lucy: pixels without a value don't constrain the estimate (data weight 0) and don't pull
    # their neighbours towards 0. Needs positive values, so they are shifted (the PSF is normalized, so a constant
    # offset passes the blur unchanged)
    weights = valid.astype(float)
    shown = np.where(valid, img, 0.0)
    offset = 0.0
    if valid.any():
        lo, hi = float(np.min(img[valid])), float(np.max(img[valid]))
        offset = -lo + 1e-3 * (hi - lo or abs(lo) or 1.0)
    data = np.where(valid, shown + offset, 0.0)
    norm = _convolve(weights, psf[::-1, ::-1])
    norm = np.where(norm > 1e-12, norm, 1.0)
    estimate = np.where(valid, data, np.mean(data[valid]) if valid.any() else 1.0)
    for _ in range(int(iterations)):
        blurred = _convolve(estimate, psf)
        ratio = np.where(valid, data / np.maximum(blurred, 1e-12), 0.0)
        estimate = estimate * _convolve(ratio, psf[::-1, ::-1]) / norm
    return estimate - offset


def wiener(img, valid, psf, snr):
    # pixels without a value are filled with the PSF weighted mean of their neighbours (normalized convolution), then
    # the Wiener filter with a constant noise to signal power ratio 1 / snr
    weights = _convolve(valid.astype(float), psf)
    fallback = float(np.mean(img[valid])) if valid.any() else 0.0
    # far from valid pixels the weight is FFT round-off (~1e-17), not 0: dividing by it explodes
    near = weights > 1e-6
    filled = np.where(near, _convolve(np.where(valid, img, 0.0), psf) / np.where(near, weights, 1.0), fallback)
    data = np.where(valid, img, filled)
    mean = float(np.mean(data))
    data = data - mean
    shape = [s + p for s, p in zip(data.shape, psf.shape)]
    kernel = np.zeros(shape)
    kernel[:psf.shape[0], :psf.shape[1]] = psf
    kernel = np.roll(kernel, (-(psf.shape[0] // 2), -(psf.shape[1] // 2)), axis=(0, 1))
    padded = np.pad(data, [(0, p) for p in psf.shape], mode="reflect")
    h = np.fft.rfft2(kernel)
    g = np.fft.rfft2(padded)
    restored = np.fft.irfft2(g * np.conj(h) / (np.abs(h) ** 2 + 1 / snr), s=shape)
    return restored[:data.shape[0], :data.shape[1]] + mean


def deconvolve_image(img, valid, dx, dy, beam_radius, method=DeconvolutionMethod.richardson_lucy, iterations=20,
                     snr=100.0):
    psf = gaussian_psf(dx, dy, beam_radius)
    if method == DeconvolutionMethod.richardson_lucy:
        out = richardson_lucy(img, valid, psf, iterations)
    else:
        out = wiener(img, valid, psf, snr)
    return np.where(valid, out, img)
