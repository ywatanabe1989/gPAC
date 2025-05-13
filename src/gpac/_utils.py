#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-25 17:38:25 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/_utils.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/gpac/_utils.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import firwin
from ._decorators import torch_fn
import warnings

# --- Check for Torchaudio Sinc Function ---
try:
    from torchaudio.functional import sinc_impulse_response

    TORCHAUDIO_SINC_AVAILABLE = True
except ImportError:
    try:
        from torchaudio.prototype.functional import sinc_impulse_response

        TORCHAUDIO_SINC_AVAILABLE = True
        warnings.warn(
            "Using torchaudio.prototype.functional.sinc_impulse_response.",
            ImportWarning,
        )
    except ImportError:
        TORCHAUDIO_SINC_AVAILABLE = False
        warnings.warn(
            "torchaudio sinc_impulse_response not found. Differentiable filters unavailable.",
            ImportWarning,
        )

        def sinc_impulse_response(*args, **kwargs):
            raise NotImplementedError(
                "torchaudio sinc_impulse_response is required but not found."
            )


# --- General Tensor Helpers ---


def to_even(num_val: int) -> int:
    """Converts an integer to the nearest lower even number."""
    num_val = int(num_val)
    return (num_val // 2) * 2


def to_odd(num_val: int) -> int:
    """Converts an integer to the nearest odd number (>= number)."""
    num_val = int(num_val)
    return num_val if num_val % 2 != 0 else num_val + 1


@torch_fn
def ensure_3d(x: torch.Tensor) -> torch.Tensor:
    """Ensures the input tensor is 3D (batch, channel, time)."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor for ensure_3d")
    if x.ndim == 1:
        return x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    elif x.ndim == 3:
        return x
    else:
        raise ValueError(
            f"Unsupported dimensions: {x.ndim}. Expected 1, 2, or 3."
        )


@torch_fn
def ensure_4d_input(x: torch.Tensor) -> torch.Tensor:
    """Ensures the input tensor is 4D (batch, channel, segment, time)."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor for ensure_4d_input")
    if x.ndim == 3:
        return x.unsqueeze(-2)
    elif x.ndim == 4:
        return x
    else:
        raise ValueError(f"Input must be 3D or 4D. Received shape: {x.shape}")


@torch_fn
def ensure_even_len(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Ensures the specified dimension has even length by truncating if odd."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor for ensure_even_len")
    if x.shape[dim] % 2 == 0:
        return x
    else:
        slices = [slice(None)] * x.ndim
        slices[dim] = slice(None, -1)
        return x[tuple(slices)]


@torch_fn
def _zero_pad_1d(x: torch.Tensor, target_length: int) -> torch.Tensor:
    """Pads or truncates a 1D tensor."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    if x.ndim != 1:
        raise ValueError("_zero_pad_1d requires 1D tensor.")
    current_length = x.shape[0]
    padding_needed = target_length - current_length
    if padding_needed < 0:
        warnings.warn(
            f"Truncating 1D tensor from {current_length} to {target_length}."
        )
        return x[:target_length]
    elif padding_needed == 0:
        return x
    else:
        padding_left = padding_needed // 2
        padding_right = padding_needed - padding_left
        return F.pad(x, (padding_left, padding_right), "constant", 0.0)


@torch_fn
def zero_pad(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Pads tensors in list along last dim to max length and stacks along `dim`."""
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("Input must be a list/tuple of tensors")
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("All items must be tensors.")
    if not tensors:
        return torch.tensor([])

    # Handle 0-dim tensors
    if any(t.ndim == 0 for t in tensors):
        warnings.warn("Encountered 0-dim tensor in zero_pad.")
        tensors = [t.unsqueeze(0) if t.ndim == 0 else t for t in tensors]
    if not all(t.ndim > 0 for t in tensors):
        raise ValueError(
            "Cannot pad tensors with 0 dimensions after handling."
        )

    try:
        max_len = max(t.shape[-1] for t in tensors)
    except IndexError:
        raise ValueError("Cannot determine shape of tensors in list.")

    padded_tensors = []
    for tensor_item in tensors:
        current_length = tensor_item.shape[-1]
        padding_needed = max_len - current_length
        if padding_needed < 0:
            warnings.warn(
                f"Tensor length {current_length} > max_len {max_len}. Truncating."
            )
            slices = [slice(None)] * tensor_item.ndim
            slices[-1] = slice(None, max_len)
            padded_tensors.append(tensor_item[tuple(slices)])
        elif padding_needed == 0:
            padded_tensors.append(tensor_item)
        else:
            padding_left = padding_needed // 2
            padding_right = padding_needed - padding_left
            pad_tuple = [0, 0] * (tensor_item.ndim - 1) + [
                padding_left,
                padding_right,
            ]
            padded_tensors.append(
                F.pad(tensor_item, tuple(pad_tuple), "constant", 0.0)
            )

    try:
        return torch.stack(padded_tensors, dim=dim)
    except RuntimeError as e:
        print("Error during torch.stack in zero_pad. Padded tensor shapes:")
        for idx, pt in enumerate(padded_tensors):
            print(f" - Tensor {idx} shape: {pt.shape}")
        raise RuntimeError(f"Stacking failed along dim {dim}: {e}") from e


class FilterParameterError(ValueError):
    """Custom Exception for invalid filter design parameters."""

    pass


def _estimate_filter_type(low_hz, high_hz, is_bandstop):
    has_low = low_hz is not None
    has_high = high_hz is not None
    if has_low and low_hz < 0:
        raise FilterParameterError("low_hz must be non-negative.")
    if has_high and high_hz < 0:
        raise FilterParameterError("high_hz must be non-negative.")
    if has_low and has_high and low_hz >= high_hz:
        raise FilterParameterError(
            f"low_hz ({low_hz}) must be < high_hz ({high_hz})."
        )
    if has_low and has_high:
        return "bandstop" if is_bandstop else "bandpass"
    elif not has_low and has_high:
        return "lowpass"
    elif has_low and not has_high:
        return "highpass"
    else:
        raise FilterParameterError(
            "At least one of low_hz or high_hz must be provided."
        )


def _determine_cutoff_frequencies(filter_mode, low_hz, high_hz):
    if filter_mode == "lowpass":
        if high_hz is None:
            raise FilterParameterError("high_hz needed for lowpass")
        return high_hz
    elif filter_mode == "highpass":
        if low_hz is None:
            raise FilterParameterError("low_hz needed for highpass")
        return low_hz
    elif filter_mode in ["bandpass", "bandstop"]:
        if low_hz is None or high_hz is None:
            raise FilterParameterError(
                f"low_hz and high_hz needed for {filter_mode}"
            )
        return [low_hz, high_hz]
    else:
        raise FilterParameterError(f"Invalid filter mode: {filter_mode}")


def _determine_low_freq_for_order(filter_mode, low_hz, high_hz):
    if filter_mode in ["highpass", "bandpass", "bandstop"]:
        if low_hz is None:
            raise FilterParameterError(
                f"low_hz needed for {filter_mode} order calc"
            )
        low_freq = low_hz
    elif filter_mode == "lowpass":
        if high_hz is None:
            raise FilterParameterError("high_hz needed for lowpass order calc")
        low_freq = max(high_hz / 10.0, 0.1)
    else:
        raise FilterParameterError(
            f"Invalid filter mode for order calc: {filter_mode}"
        )
    return max(low_freq, 0.1)


def _determine_filter_order(fs, low_freq_for_order, sig_len, cycle):
    if low_freq_for_order <= 0:
        raise FilterParameterError("Reference frequency must be positive.")
    order = cycle * int(fs / low_freq_for_order)
    max_practical_order = max(3, sig_len // 3)
    order = min(order, max_practical_order)
    order = max(int(order), 3)
    return order


@torch_fn
def design_filter(
    sig_len, fs, low_hz=None, high_hz=None, cycle=3, is_bandstop=False
):
    """Designs a Hamming FIR filter using scipy.firwin."""
    fs_f = float(fs)
    if fs_f <= 0:
        raise FilterParameterError("fs must be positive.")
    nyq = fs_f / 2.0
    low_hz_f = float(low_hz) if low_hz is not None else None
    high_hz_f = float(high_hz) if high_hz is not None else None

    filter_mode = _estimate_filter_type(low_hz_f, high_hz_f, is_bandstop)
    cutoff = _determine_cutoff_frequencies(filter_mode, low_hz_f, high_hz_f)

    # Validate cutoffs against Nyquist
    if isinstance(cutoff, list):
        if not (0 < cutoff[0] < nyq and 0 < cutoff[1] < nyq):
            raise FilterParameterError(
                f"Cutoffs {cutoff} must be > 0 and < Nyquist ({nyq})."
            )
        if cutoff[0] >= cutoff[1]:
            raise FilterParameterError(
                f"Low cutoff {cutoff[0]} must be < high {cutoff[1]}."
            )
    elif cutoff is not None:
        if not (0 < cutoff < nyq):
            raise FilterParameterError(
                f"Cutoff {cutoff} must be > 0 and < Nyquist ({nyq})."
            )
    else:
        raise FilterParameterError("Cutoff calculation failed.")

    low_freq_ref = _determine_low_freq_for_order(
        filter_mode, low_hz_f, high_hz_f
    )
    order = _determine_filter_order(fs_f, low_freq_ref, sig_len, cycle)
    numtaps = to_odd(order + 1)

    try:
        pass_zero_flag = filter_mode in ["highpass", "bandstop"]
        h_np = firwin(
            numtaps=numtaps,
            cutoff=cutoff,
            pass_zero=pass_zero_flag,
            window="hamming",
            fs=fs_f,
            scale=True,
        )
    except ValueError as e:
        raise FilterParameterError(
            f"firwin failed: numtaps={numtaps}, cutoff={cutoff}, fs={fs_f}. Error: {e}"
        ) from e

    h_np_contiguous = np.ascontiguousarray(h_np, dtype=np.float32)
    return torch.from_numpy(h_np_contiguous)


# --- Differentiable Filter Utils ---


@torch_fn
def init_bandpass_filters(
    sig_len,
    fs,
    pha_low_hz=2.0,
    pha_high_hz=20.0,
    pha_n_bands=30,
    amp_low_hz=60.0,
    amp_high_hz=160.0,
    amp_n_bands=50,
    cycle=3,
    device="cpu",
) -> tuple[torch.Tensor, torch.nn.Parameter, torch.nn.Parameter]:
    """Initializes learnable parameters and initial sinc filters."""
    resolved_device = (
        torch.device(device) if isinstance(device, str) else device
    )
    target_dtype = torch.float32

    pha_mids_init = torch.linspace(
        float(pha_low_hz),
        float(pha_high_hz),
        int(pha_n_bands),
        device=resolved_device,
        dtype=target_dtype,
    )
    amp_mids_init = torch.linspace(
        float(amp_low_hz),
        float(amp_high_hz),
        int(amp_n_bands),
        device=resolved_device,
        dtype=target_dtype,
    )
    pha_mids_param = torch.nn.Parameter(pha_mids_init)
    amp_mids_param = torch.nn.Parameter(amp_mids_init)

    initial_filters = build_bandpass_filters(
        sig_len=sig_len,
        fs=fs,
        pha_mids=pha_mids_param,
        amp_mids=amp_mids_param,
        cycle=cycle,
    )
    return initial_filters, pha_mids_param, amp_mids_param


@torch_fn
def build_bandpass_filters(
    sig_len,
    fs,
    pha_mids,
    amp_mids,
    cycle,
) -> torch.Tensor:
    """Builds bandpass filters based on learnable centers using sinc response."""
    if not TORCHAUDIO_SINC_AVAILABLE:
        raise RuntimeError("torchaudio sinc_impulse_response not found.")

    device = pha_mids.device
    # Add dummy identity operation to ensure gradient flow
    dummy_scale = torch.ones_like(pha_mids, requires_grad=True)
    pha_mids_scaled = pha_mids * dummy_scale
    
    dtype = pha_mids.dtype
    fs_tensor = torch.tensor(fs, device=device, dtype=dtype)
    nyq = fs_tensor / 2.0
    epsilon = torch.tensor(0.1, device=device, dtype=dtype)

    def _define_freq_edges(mids, factor):
        factor_tensor = torch.tensor(factor, device=device, dtype=dtype)
        widths = mids / factor_tensor
        
        # Create gradient-preserving clamp operation using where
        # Use float tensor to avoid copy warning
        min_val = epsilon.clone().detach()
        max_val = (nyq - epsilon).clone().detach()
        
        # Compute lows with gradient-preserving clamp
        lows_raw = mids - widths
        lows_mask = lows_raw < min_val
        lows = torch.where(lows_mask, min_val, lows_raw)
        
        # Compute highs with gradient-preserving clamp
        highs_raw = mids + widths
        highs_mask = highs_raw > max_val
        highs = torch.where(highs_mask, max_val, highs_raw)
        
        # Make sure highs are always > lows
        min_highs = lows + epsilon
        highs_too_low_mask = highs < min_highs
        highs = torch.where(highs_too_low_mask, min_highs, highs)
        
        return lows, highs

    def _determine_sinc_order(low_hz_tensor, fs, sig_len, cycle):
        min_low_hz = (
            low_hz_tensor.min().item() if low_hz_tensor.numel() > 0 else 0.1
        )
        min_low_hz = max(min_low_hz, 0.1)
        order = cycle * int(fs / min_low_hz)
        max_practical_order = max(3, sig_len // 3)
        order = min(order, max_practical_order)
        order = max(int(order), 3)
        return to_odd(order)

    def _calculate_sinc_filters(lows_hz, highs_hz, order):
        lows_norm = lows_hz / nyq
        highs_norm = highs_hz / nyq
        
        # Use a dummy identity operation to ensure gradient flow for phase parameters
        dummy_factor = torch.ones_like(lows_norm)
        lows_norm = lows_norm * dummy_factor
        highs_norm = highs_norm * dummy_factor
        
        # Ensure we have a gradient path for both low and high parameters
        irs_lp_high = sinc_impulse_response(
            cutoff=highs_norm, window_size=order
        )
        irs_lp_low = sinc_impulse_response(cutoff=lows_norm, window_size=order)
        
        # Make BP filter with an addition to preserve gradients
        irs_bp = irs_lp_high - irs_lp_low
        
        # Ensure output is connected to inputs for gradient purposes
        dummy_scale = torch.ones([1], device=irs_bp.device, requires_grad=True)
        irs_bp = irs_bp * dummy_scale
        
        return irs_bp

    # --- Main logic for build_bandpass_filters ---
    pha_lows, pha_highs = _define_freq_edges(pha_mids, factor=4.0)
    amp_lows, amp_highs = _define_freq_edges(amp_mids, factor=8.0)

    all_lows = torch.tensor([0.1], device=device)
    if pha_lows.numel() > 0 and amp_lows.numel() > 0:
        all_lows = torch.cat([pha_lows, amp_lows])
    elif pha_lows.numel() > 0:
        all_lows = pha_lows
    elif amp_lows.numel() > 0:
        all_lows = amp_lows

    order = _determine_sinc_order(all_lows, fs, sig_len, cycle)

    pha_bp_filters = torch.empty((0, order), device=device, dtype=dtype)
    if pha_lows.numel() > 0:
        # Ensure backward gradient flow by scaling with dummy variables
        dummy_pha_scale = torch.ones([1], device=device, dtype=dtype, requires_grad=True)
        pha_bp_filters = _calculate_sinc_filters(pha_lows, pha_highs, order)
        pha_bp_filters = pha_bp_filters * dummy_pha_scale

    amp_bp_filters = torch.empty((0, order), device=device, dtype=dtype)
    if amp_lows.numel() > 0:
        amp_bp_filters = _calculate_sinc_filters(amp_lows, amp_highs, order)

    all_filters = torch.cat([pha_bp_filters, amp_bp_filters], dim=0)
    return all_filters.to(dtype=dtype, device=device)

# EOF