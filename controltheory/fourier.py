# -*- coding: utf-8 -*-
"""
Tools for the Fourier domain.
"""

import numpy as np
import astropy.units as u

__all__ = ['frequencies']

def frequencies(length, rate):
    """Generate the frequencies for a FFT at a given rate."""
    rate = u.Quantity(rate, u.Hz)
    return np.fft.fftshift(np.fft.fftfreq(length)) * rate