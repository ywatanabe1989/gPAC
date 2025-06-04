#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-26 10:33:30 (ywatanabe)"
# File: ./mngs_repo/src/mngs/nn/_PAC.py

THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/nn/_PAC.py"

# Imports
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ._BandPassFilter import BandPassFilter
from ._Hilbert import Hilbert
from ._ModulationIndex import ModulationIndex


# Functions
class PAC(nn.Module):
    def __init__(
        self,
        seq_len: int,
        fs: float,
        pha_start_hz: float = 2,
        pha_end_hz: float = 20,
        pha_n_bands: int = 50,
        amp_start_hz: float = 60,
        amp_end_hz: float = 160,
        amp_n_bands: int = 30,
        n_perm: Union[int, None] = 0,
        trainable: bool = False,
        in_place: bool = True,
        fp16: bool = False,
        edge_mode: str = "auto",
        edge_length: Optional[Union[int, float]] = None,
    ) -> None:
        super().__init__()

        # Input validation
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if fs <= 0:
            raise ValueError(f"fs must be positive, got {fs}")
        if pha_start_hz <= 0 or pha_end_hz <= pha_start_hz:
            raise ValueError(
                f"Invalid phase frequency range: [{pha_start_hz}, {pha_end_hz}] Hz"
            )
        if amp_start_hz <= 0 or amp_end_hz <= amp_start_hz:
            raise ValueError(
                f"Invalid amplitude frequency range: [{amp_start_hz}, {amp_end_hz}] Hz"
            )
        if pha_n_bands <= 0 or amp_n_bands <= 0:
            raise ValueError(
                f"Number of bands must be positive, got pha: {pha_n_bands}, amp: {amp_n_bands}"
            )
        # Handle n_perm=0 as None
        if n_perm == 0:
            n_perm = None
        if n_perm is not None and (not isinstance(n_perm, int) or n_perm <= 0):
            raise ValueError(f"n_perm must be a positive integer or None, got {n_perm}")

        self.fp16 = fp16
        self.n_perm = n_perm
        self.trainable = trainable
        self.edge_mode = edge_mode
        self.edge_length = edge_length

        # caps amp_end_hz to avoid aliasing
        factor = 0.8
        nyquist = fs / 2
        amp_end_hz = int(min(nyquist / (1 + factor) - 1, amp_end_hz))

        # Also adjust amp_start_hz if it's now >= amp_end_hz
        if amp_start_hz >= amp_end_hz:
            # Set amp_start_hz to a reasonable value below amp_end_hz
            amp_start_hz = max(1.0, amp_end_hz * 0.5)

        # Check frequency bounds against Nyquist
        if pha_end_hz > nyquist:
            raise ValueError(
                f"Phase end frequency {pha_end_hz} Hz exceeds Nyquist frequency {nyquist} Hz"
            )
        if amp_end_hz > nyquist:
            raise ValueError(
                f"Amplitude end frequency {amp_end_hz} Hz exceeds Nyquist frequency {nyquist} Hz"
            )

        # Determine padding mode based on edge_mode if not specified directly
        padding_mode = "reflect"  # default
        if edge_mode == "zero":
            padding_mode = "zero"
        elif edge_mode == "replicate":
            padding_mode = "replicate"

        self.bandpass = BandPassFilter(
            seq_len,
            fs,
            pha_start_hz=pha_start_hz,
            pha_end_hz=pha_end_hz,
            pha_n_bands=pha_n_bands,
            amp_start_hz=amp_start_hz,
            amp_end_hz=amp_end_hz,
            amp_n_bands=amp_n_bands,
            fp16=fp16,
            trainable=trainable,
            padding_mode=padding_mode,
        )

        # Set PHA_MIDS_HZ and AMP_MIDS_HZ from the bandpass filter
        self.PHA_MIDS_HZ = self.bandpass.pha_mids
        self.AMP_MIDS_HZ = self.bandpass.amp_mids

        self.hilbert = Hilbert(seq_len, dim=-1, fp16=fp16)

        self.modulation_index = ModulationIndex(n_bins=18, temperature=0.1)

        # No need for DimHandler - we'll use simple reshaping in generate_surrogates

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute PAC values from input signal.

        Parameters
        ----------
        x : torch.Tensor
            Input signal with shape:
            - (batch_size, n_chs, seq_len) or
            - (batch_size, n_chs, n_segments, seq_len)

        Returns
        -------
        dict
            Dictionary containing:
            - 'pac': PAC values (modulation index) with shape (batch, channels, freqs_phase, freqs_amplitude)
            - 'phase_frequencies': Center frequencies for phase bands
            - 'amplitude_frequencies': Center frequencies for amplitude bands
            - 'mi_per_segment': None (for efficiency - use forward_full() if needed)
            - 'amplitude_distributions': None (for efficiency - use forward_full() if needed)
            - 'phase_bin_centers': None (for efficiency - use forward_full() if needed)
            - 'phase_bin_edges': None (for efficiency - use forward_full() if needed)
            - 'pac_z': Z-scored PAC values (if n_perm is specified)
            - 'surrogates': Surrogate PAC values (if n_perm is specified)
            - 'surrogate_mean': Mean of surrogates (if n_perm is specified)
            - 'surrogate_std': Std of surrogates (if n_perm is specified)
        """
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        if x.ndim not in [3, 4]:
            raise ValueError(
                f"Input must be 3D or 4D tensor, got {x.ndim}D tensor with shape {x.shape}"
            )

        # Constants for clarity
        PHASE_IDX = 0
        AMPLITUDE_IDX = 1

        with torch.set_grad_enabled(bool(self.trainable)):
            # Ensure 4D input: (batch, channels, segments, time)
            x = self._ensure_4d_input(x)
            batch_size, n_chs, n_segments, seq_len = x.shape

            # Process each batch-channel combination together
            # This reshape is necessary for the bandpass filter
            x = x.reshape(batch_size * n_chs, n_segments, seq_len)

            # Apply bandpass filtering with configurable edge handling
            # For now, we don't remove edges in the filter itself
            x = self.bandpass(x, edge_len=0)
            # Now: (batch*chs, segments, n_bands, time)

            # Extract phase and amplitude via Hilbert transform
            x = self.hilbert(x)
            # Now: (batch*chs, segments, n_bands, time, 2) where last dim is [phase, amplitude]

            # Restore batch dimension
            x = x.reshape(batch_size, n_chs, n_segments, -1, seq_len, 2)
            # Now: (batch, chs, segments, n_bands, time, 2)

            # Split into phase and amplitude bands
            n_pha_bands = len(self.PHA_MIDS_HZ)
            n_amp_bands = len(self.AMP_MIDS_HZ)

            # Extract phase from phase bands
            pha = x[:, :, :, :n_pha_bands, :, PHASE_IDX]
            # Extract amplitude from amplitude bands
            amp = x[:, :, :, n_pha_bands:, :, AMPLITUDE_IDX]

            # Rearrange dimensions for ModulationIndex
            # ModulationIndex expects: (batch, chs, freqs, segments, time)
            pha = pha.permute(0, 1, 3, 2, 4)
            amp = amp.permute(0, 1, 3, 2, 4)

            # Remove edge artifacts based on configuration
            edge_len = self._calculate_edge_length(seq_len)
            if edge_len > 0:
                pha = pha[..., edge_len:-edge_len]
                amp = amp[..., edge_len:-edge_len]

            # Convert to half precision if needed
            if self.fp16:
                pha = pha.half()
                amp = amp.half()

            # Calculate modulation index (only compute MI, not distributions)
            mi_results = self.modulation_index(pha, amp)

            # Extract the primary PAC values
            pac_values = mi_results["mi"]

            # Prepare output dictionary with fixed structure
            output = {
                "pac": pac_values,
                "phase_frequencies": self.PHA_MIDS_HZ.detach().cpu(),
                "amplitude_frequencies": self.AMP_MIDS_HZ.detach().cpu(),
                "mi_per_segment": None,  # Not computed for efficiency
                "amplitude_distributions": None,  # Not computed for efficiency
                "phase_bin_centers": None,  # Not computed for efficiency
                "phase_bin_edges": None,  # Not computed for efficiency
            }

            # Apply surrogate statistics if requested
            if self.n_perm is not None:
                z_scores, surrogates = self.to_z_using_surrogate(pha, amp, pac_values)
                output["pac_z"] = z_scores
                output["surrogates"] = surrogates
                output["surrogate_mean"] = surrogates.mean(dim=2)
                output["surrogate_std"] = surrogates.std(dim=2)

            return output

    def to_z_using_surrogate(
        self, pha: torch.Tensor, amp: torch.Tensor, observed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate z-scores using surrogate distribution.

        Returns
        -------
        tuple
            (z_scores, surrogates)
        """
        surrogates = self.generate_surrogates(pha, amp)
        mm = surrogates.mean(dim=2).to(observed.device)
        ss = surrogates.std(dim=2).to(observed.device)
        z_scores = (observed - mm) / (ss + 1e-5)
        return z_scores, surrogates

    def generate_surrogates(
        self, pha: torch.Tensor, amp: torch.Tensor, batch_size: int = 1
    ) -> torch.Tensor:
        """
        Generate surrogate PAC values by circular shifting the phase signal.

        Parameters
        ----------
        pha : torch.Tensor
            Phase signal with shape (batch, channels, freqs_pha, segments, time)
        amp : torch.Tensor
            Amplitude signal with shape (batch, channels, freqs_amp, segments, time)
        batch_size : int
            Batch size for processing surrogates to manage memory

        Returns
        -------
        torch.Tensor
            Surrogate PAC values with shape (batch, channels, n_perm, freqs_pha, freqs_amp)
        """
        # Get dimensions
        batch, n_chs, n_freqs_pha, n_segments, seq_len = pha.shape
        n_freqs_amp = amp.shape[2]

        # Generate random circular shift points for each permutation
        shift_points = torch.randint(seq_len, (self.n_perm,), device=pha.device)

        # Store surrogate PAC values
        surrogate_pacs = []

        # Process each permutation
        with torch.no_grad():
            for perm_idx, shift in enumerate(shift_points):
                # Circular shift the phase signal
                pha_shifted = torch.roll(pha, shifts=int(shift), dims=-1)

                # Calculate PAC for this permutation
                # Process in smaller batches if needed for memory
                pac_perm = []
                for i in range(0, batch, batch_size):
                    end_idx = min(i + batch_size, batch)
                    mi_results = self.modulation_index(
                        pha_shifted[i:end_idx], amp[i:end_idx]
                    )
                    pac_perm.append(mi_results["mi"].cpu())

                # Combine batches
                pac_perm = torch.cat(pac_perm, dim=0)
                surrogate_pacs.append(pac_perm)

        # Stack all permutations: (batch, channels, n_perm, freqs_pha, freqs_amp)
        surrogate_pacs = torch.stack(surrogate_pacs, dim=2)

        # Clear GPU cache if we used it
        if pha.is_cuda:
            torch.cuda.empty_cache()

        return surrogate_pacs

    # The init_bandpass method is no longer needed as BandPassFilter handles both static and trainable modes

    # Band calculation methods are now in BandPassFilter

    def _calculate_edge_length(self, seq_len: int) -> int:
        """
        Calculate edge length based on configuration.

        Parameters
        ----------
        seq_len : int
            Sequence length

        Returns
        -------
        int
            Number of samples to remove from each edge
        """
        if self.edge_mode == "none":
            return 0
        elif self.edge_mode == "auto":
            # Default behavior: remove 1/8 of signal
            return seq_len // 8
        elif self.edge_mode == "adaptive":
            # Adaptive based on filter order and frequency
            # Estimate based on lowest frequency component
            min_freq = min(
                self.bandpass.pha_bands[0, 0].item(),
                self.bandpass.amp_bands[0, 0].item(),
            )
            # Rule of thumb: 3 cycles of lowest frequency
            edge_samples = int(3 * self.bandpass.fs / min_freq)
            return min(edge_samples, seq_len // 4)  # Cap at 1/4 of signal
        elif self.edge_mode == "fixed":
            if self.edge_length is None:
                raise ValueError("edge_length must be specified when edge_mode='fixed'")
            if isinstance(self.edge_length, float):
                # Interpret as fraction of signal
                return int(self.edge_length * seq_len)
            else:
                # Direct sample count
                return int(self.edge_length)
        else:
            raise ValueError(f"Unknown edge_mode: {self.edge_mode}")

    @staticmethod
    def _ensure_4d_input(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            message = f"Input tensor must be 4D with the shape (batch_size, n_chs, n_segments, seq_len). Received shape: {x.shape}"

        if x.ndim == 3:
            # warnings.warn(
            #     "'n_segments' was determined to be 1, assuming your input is (batch_size, n_chs, seq_len).",
            #     UserWarning,
            # )
            x = x.unsqueeze(-2)

        if x.ndim != 4:
            raise ValueError(message)

        return x


# Main block removed - example usage is in documentation/tests

# EOF
