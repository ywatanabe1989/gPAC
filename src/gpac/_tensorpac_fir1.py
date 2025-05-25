#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-25 11:55:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/src/gpac/_tensorpac_fir1.py
# ----------------------------------------
"""
TensorPAC-compatible FIR filter implementation for gPAC.

This module provides TensorPAC's custom fir1 implementation
to ensure exact compatibility for fair comparisons.
"""

import numpy as np
import torch
from ._decorators import torch_fn


def fir_order(fs, sizevec, flow, cycle=3):
    """
    Calculate filter order using TensorPAC's method.
    
    Parameters
    ----------
    fs : float
        Sampling frequency
    sizevec : int
        Signal length
    flow : float
        Lower frequency bound
    cycle : int
        Number of cycles
        
    Returns
    -------
    int
        Filter order
    """
    if cycle is None:
        filtorder = 3 * np.fix(fs / flow)
    else:
        filtorder = cycle * (fs // flow)
        
        if (sizevec < 3 * filtorder):
            filtorder = (sizevec - 1) // 3
    
    return int(filtorder)


def n_odd_fcn(f, o, w, l):
    """Odd case for TensorPAC fir1."""
    # Variables :
    b0 = 0
    m = np.array(range(int(l + 1)))
    k = m[1:len(m)]
    b = np.zeros(k.shape)

    # Run Loop :
    for s in range(0, len(f), 2):
        m = (o[s + 1] - o[s]) / (f[s + 1] - f[s])
        b1 = o[s] - m * f[s]
        b0 = b0 + (b1 * (f[s + 1] - f[s]) + m / 2 * (
            f[s + 1] * f[s + 1] - f[s] * f[s])) * abs(
            np.square(w[round((s + 1) / 2)]))
        b = b + (m / (4 * np.pi * np.pi) * (
            np.cos(2 * np.pi * k * f[s + 1]) - np.cos(2 * np.pi * k * f[s])
        ) / (k * k)) * abs(np.square(w[round((s + 1) / 2)]))
        b = b + (f[s + 1] * (m * f[s + 1] + b1) * np.sinc(2 * k * f[
            s + 1]) - f[s] * (m * f[s] + b1) * np.sinc(2 * k * f[s])) * abs(
            np.square(w[round((s + 1) / 2)]))

    b = np.insert(b, 0, b0)
    a = (np.square(w[0])) * 4 * b
    a[0] = a[0] / 2
    aud = np.flipud(a[1:len(a)]) / 2
    a2 = np.insert(aud, len(aud), a[0])
    h = np.concatenate((a2, a[1:] / 2))

    return h


def n_even_fcn(f, o, w, l):
    """Even case for TensorPAC fir1."""
    # Variables :
    k = np.array(range(0, int(l) + 1, 1)) + 0.5
    b = np.zeros(k.shape)

    # # Run Loop :
    for s in range(0, len(f), 2):
        m = (o[s + 1] - o[s]) / (f[s + 1] - f[s])
        b1 = o[s] - m * f[s]
        b = b + (m / (4 * np.pi * np.pi) * (np.cos(2 * np.pi * k * f[
            s + 1]) - np.cos(2 * np.pi * k * f[s])) / (
            k * k)) * abs(np.square(w[round((s + 1) / 2)]))
        b = b + (f[s + 1] * (m * f[s + 1] + b1) * np.sinc(2 * k * f[
            s + 1]) - f[s] * (m * f[s] + b1) * np.sinc(2 * k * f[s])) * abs(
            np.square(w[round((s + 1) / 2)]))

    a = (np.square(w[0])) * 4 * b
    h = 0.5 * np.concatenate((np.flipud(a), a))

    return h


def firls(n, f, o):
    """TensorPAC's firls implementation."""
    # Variables definition :
    w = np.ones(round(len(f) / 2))
    n += 1
    f /= 2
    lo = (n - 1) / 2

    nodd = bool(n % 2)

    if nodd:  # Odd case
        h = n_odd_fcn(f, o, w, lo)
    else:  # Even case
        h = n_even_fcn(f, o, w, lo)

    return h


def fir1(n, wn):
    """
    TensorPAC's fir1 implementation.
    
    Parameters
    ----------
    n : int
        Filter order
    wn : array_like
        Normalized frequency boundaries [low, high] / (fs/2)
        
    Returns
    -------
    b : array_like
        Filter coefficients
    a : float
        Always 1 for FIR filters
    """
    # Variables definition :
    nbands = len(wn) + 1
    ff = np.array((0, wn[0], wn[0], wn[1], wn[1], 1))

    f0 = np.mean(ff[2:4])
    lo = n + 1

    mags = np.array(range(nbands)).reshape(1, -1) % 2
    aa = np.ravel(np.tile(mags, (2, 1)), order='F')

    # Get filter coefficients :
    h = firls(lo - 1, ff, aa)

    # Apply a window to coefficients :
    wind = np.hamming(lo)
    b = h * wind
    c = np.exp(-1j * 2 * np.pi * (f0 / 2) * np.array(range(lo)))
    b /= abs(c @ b)

    return b, 1


@torch_fn
def design_filter_tensorpac(sig_len, fs, low_hz=None, high_hz=None, cycle=3, is_bandstop=False):
    """
    Design a filter using TensorPAC's fir1 implementation.
    
    This is a drop-in replacement for gPAC's design_filter function
    that uses TensorPAC's custom fir1 implementation instead of scipy.firwin.
    
    Parameters
    ----------
    sig_len : int
        Signal length
    fs : float
        Sampling frequency
    low_hz : float, optional
        Lower frequency bound
    high_hz : float, optional
        Upper frequency bound
    cycle : int
        Number of cycles (default: 3)
    is_bandstop : bool
        Whether to create a bandstop filter (not implemented)
        
    Returns
    -------
    torch.Tensor
        Filter coefficients
    """
    if is_bandstop:
        raise NotImplementedError("Bandstop filters not implemented in TensorPAC mode")
    
    fs_f = float(fs)
    if fs_f <= 0:
        raise ValueError("fs must be positive.")
    
    nyq = fs_f / 2.0
    low_hz_f = float(low_hz) if low_hz is not None else None
    high_hz_f = float(high_hz) if high_hz is not None else None
    
    # Validate inputs
    if low_hz_f is None or high_hz_f is None:
        raise ValueError("Both low_hz and high_hz must be provided for bandpass filter")
    
    if not (0 < low_hz_f < nyq and 0 < high_hz_f < nyq):
        raise ValueError(f"Frequencies must be > 0 and < Nyquist ({nyq}).")
    
    if low_hz_f >= high_hz_f:
        raise ValueError(f"Low frequency {low_hz_f} must be < high {high_hz_f}.")
    
    # Calculate filter order using TensorPAC's method
    filter_order = fir_order(fs_f, sig_len, low_hz_f, cycle=cycle)
    
    # Get filter coefficients using TensorPAC's fir1
    wn = np.array([low_hz_f, high_hz_f]) / nyq
    b_coeff, a_coeff = fir1(filter_order, wn)
    
    # Convert to torch tensor
    h_np_contiguous = np.ascontiguousarray(b_coeff, dtype=np.float32)
    return torch.from_numpy(h_np_contiguous)


# Export the main function
__all__ = ['design_filter_tensorpac', 'fir_order', 'fir1']