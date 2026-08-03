"""The sharpened intensity mapping (``sharpen_hist``).

The mathematical core of N3.  The bias field blurs the intensity histogram, and
the paper's central assumption is that the blur is approximately a Gaussian
convolution.  The measured histogram is therefore deconvolved to recover the
distribution the tissues would have had, and each measured intensity is mapped
to the expected true intensity under that distribution,

.. math::  U(v) = E[u \\mid v] = \\frac{(u f) * g}{f * g}

with ``f`` the deconvolved distribution and ``g`` the Gaussian.  Passing a
volume through ``U`` sharpens its histogram, and the difference between the
volume and its sharpened self is attributed to the field.

Ported from ``legacy/N3/src/SharpenHist/sharpen_hist.cc:98-233``.  Everything
happens on the *bin* grid, so the kernel width is ``fwhm/slope`` bins.
"""

import math

import torch


def sharpen_lut(counts, value_range, fwhm, noise, deblur=False):
    """N3's sharpened intensity mapping, one value per histogram bin.

    ``fwhm`` is the assumed width of the blur in intensity units (log intensity,
    in the pipeline) and ``noise`` is the additive constant of the Wiener
    restoration filter.  ``deblur=True`` reproduces the legacy ``-blur`` flag,
    which skips the deconvolution and only smooths.
    """
    counts = torch.as_tensor(counts, dtype=torch.float64).reshape(-1)
    bins = counts.numel()
    low, high = float(value_range[0]), float(value_range[1])
    if bins <= 1 or low == high:
        raise ValueError("sharpen_lut: degenerate histogram")

    # The transform is padded to twice the next power of two, with the
    # histogram sitting in the middle -- at 200 bins that is 512, not 256.
    padded = int(pow(2, math.ceil(math.log(bins) / math.log(2.0)) + 1) + 0.5)
    offset = (padded - bins) // 2

    slope = (high - low) / (bins - 1)  # intensity per bin
    blur = torch.fft.fft(_gaussian(fwhm / slope, padded, counts.device))
    restore = blur.conj() / (blur.conj() * blur + noise)

    padded_counts = torch.zeros(padded, dtype=counts.dtype, device=counts.device)
    padded_counts[offset:offset + bins] = counts

    if deblur:
        distribution = padded_counts
    else:
        distribution = torch.fft.ifft(
            torch.fft.fft(padded_counts) * restore).real.clamp(min=0.0)

    intensity = low + (torch.arange(padded, dtype=counts.dtype,
                                    device=counts.device) - offset) * slope
    moment = intensity * distribution

    mapping = (_blurred(moment, blur) / _blurred(distribution, blur))
    mapping = torch.where(torch.isfinite(mapping), mapping,
                          torch.zeros_like(mapping))
    return mapping[offset:offset + bins]


def _blurred(values, blur):
    """``values`` convolved with the Gaussian, via its transform ``blur``."""
    return torch.fft.ifft(torch.fft.fft(values) * blur).real


def _gaussian(fwhm, size, device):
    """A unit-area Gaussian of width ``fwhm`` bins, centred on index 0.

    Centred on 0 and wrapped around the end of the array, so that convolving
    by multiplication in the Fourier domain does not shift the result.
    """
    factor = 4.0 * math.log(2.0) / (fwhm * fwhm)
    scale = 2.0 * math.sqrt(math.log(2.0) / math.pi) / fwhm

    kernel = torch.zeros(size, dtype=torch.float64, device=device)
    kernel[0] = scale
    half = torch.arange(1, (size - 1) // 2 + 1, dtype=torch.float64,
                        device=device)
    tail = scale * torch.exp(-half * half * factor)
    kernel[1:half.numel() + 1] = tail
    kernel[size - half.numel():] = tail.flip(0)

    if size % 2 == 0:  # the wrap-around point belongs to both halves
        kernel[size // 2] = scale * math.exp(-size * size * factor / 4.0)
    return kernel
