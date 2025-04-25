#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-25 19:05:40 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/_Hilbert.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/gpac/_Hilbert.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import warnings

import torch
import torch.nn as nn
from torch.fft import fft, ifft


class Hilbert(nn.Module):
    """
    Calculates the analytic signal using the Hilbert transform via FFT.
    """

    def __init__(
        self,
        seq_len: (
            int | None
        ) = None,  # Optional: Sequence length for precomputation
        dim: int = -1,  # Dimension along which to apply the transform
        fp16: bool = False,  # Use float16 output where appropriate
    ):
        super().__init__()
        self.dim = dim
        self.fp16 = fp16
        self.initial_seq_len = seq_len  # Store if provided

        # Buffer for Heaviside step function in frequency domain
        # self.heaviside_freq: torch.Tensor | None = None
        if seq_len is not None:
            self._precompute_fft_components(seq_len)

    def _precompute_fft_components(self, num_samples: int) -> None:
        """Precomputes the frequency-domain Heaviside step function."""
        freqs = torch.fft.fftfreq(num_samples, d=1.0)
        # Use exact Heaviside for standard Hilbert (1 for positive freq, 0.5 for zero, 0 for negative)
        # and multiply by 2 later.
        # Create the step function components
        heaviside_u = torch.zeros_like(freqs)
        if num_samples % 2 == 0:
            # Even length: f[0]=0, f[1]..f[N/2-1]>0, f[N/2]=Nyquist, f[N/2+1]..f[N-1]<0
            heaviside_u[1 : num_samples // 2] = 1.0
            heaviside_u[num_samples // 2] = 0.5  # Nyquist component
        else:
            # Odd length: f[0]=0, f[1]..f[(N-1)/2]>0, f[(N+1)/2]..f[N-1]<0
            heaviside_u[1 : (num_samples - 1) // 2 + 1] = 1.0
        heaviside_u[0] = 0.5  # DC component

        # Register as a non-persistent buffer
        self.register_buffer("heaviside_freq", heaviside_u, persistent=False)

    def _get_heaviside_for_length(
        self, num_samples: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Gets or computes the Heaviside function for the given length."""
        if (
            self.heaviside_freq is None
            or self.heaviside_freq.shape[0] != num_samples
        ):
            if (
                self.initial_seq_len is not None
                and num_samples != self.initial_seq_len
            ):
                warnings.warn(
                    f"Input sequence length ({num_samples}) differs from Hilbert initialization length ({self.initial_seq_len}). Recomputing dynamically."
                )
            elif self.initial_seq_len is None:
                warnings.warn(
                    f"Computing Heaviside step dynamically for Hilbert transform (length {num_samples})."
                )

            freqs = torch.fft.fftfreq(num_samples, d=1.0, device=device)
            heaviside_u = torch.zeros_like(freqs)
            if num_samples % 2 == 0:
                heaviside_u[1 : num_samples // 2] = 1.0
                heaviside_u[num_samples // 2] = 0.5
            else:
                heaviside_u[1 : (num_samples - 1) // 2 + 1] = 1.0
            heaviside_u[0] = 0.5

            if self.initial_seq_len is None:  # Store if dynamic
                self.heaviside_freq = heaviside_u.to(
                    dtype=dtype
                )  # Use appropriate dtype
            return heaviside_u.to(dtype=dtype)
        else:
            # Return precomputed buffer, ensuring it's on the correct device/dtype
            return self.heaviside_freq.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the analytic signal (phase and amplitude) of a real-valued input.
        """
        orig_dtype = x.dtype
        device = x.device
        current_seq_len = x.shape[self.dim]

        # Use float32 for FFT precision
        fft_dtype = torch.float32

        # 1. Prepare Input for FFT
        if torch.is_complex(x):
            warnings.warn(
                "Hilbert module received complex input. Using only the real part."
            )
            x_real = x.real.to(fft_dtype)
        elif not torch.is_floating_point(x):
            warnings.warn(
                f"Hilbert input was not floating point ({x.dtype}). Casting to {fft_dtype}."
            )
            x_real = x.to(fft_dtype)
        else:
            x_real = x.to(fft_dtype)

        # 2. Get Heaviside Step Function
        heaviside_u = self._get_heaviside_for_length(
            current_seq_len, device, fft_dtype
        )

        # 3. Perform FFT
        xf = fft(x_real, n=current_seq_len, dim=self.dim)

        # 4. Apply Heaviside Step in Frequency Domain (Multiply by 2)
        analytic_signal_f = xf * (2.0 * heaviside_u)

        # 5. Inverse FFT
        analytic_signal_t = ifft(
            analytic_signal_f, n=current_seq_len, dim=self.dim
        )  # Result is complex

        # 6. Calculate Instantaneous Phase and Amplitude
        pha = torch.atan2(analytic_signal_t.imag, analytic_signal_t.real)
        amp = analytic_signal_t.abs()

        # 7. Stack Phase and Amplitude
        out = torch.stack([pha, amp], dim=-1)

        # 8. Cast Output Type
        if self.fp16:
            output_dtype = torch.float16
        elif torch.is_complex(torch.empty(0, dtype=orig_dtype)):
            output_dtype = torch.float32
        else:
            output_dtype = orig_dtype

        return out.to(output_dtype)

# EOF