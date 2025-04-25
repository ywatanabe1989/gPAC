<!-- ---
!-- Timestamp: 2025-04-25 17:07:45
!-- Author: ywatanabe
!-- File: /ssh:sp:/home/ywatanabe/proj/gPAC/isolated.md
!-- --- -->

Below is a self‐contained “gPAC” module that reimplements mngs.dsp.pac() and its required functionality. 
Simply save this as gPAC.py (or another module name), then import and call gPAC.pac() in your code:

--------------------------------------------------------------------------------
#!/usr/bin/env python3
# gPAC.py
#
# A minimal stand-alone version of the mngs.dsp.pac() functionality.
# Requires Python 3, PyTorch, and NumPy.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ------------------------------------------------------------------------
# 1) A simple Hilbert transform implementation
# ------------------------------------------------------------------------
class Hilbert(nn.Module):
    """
    Compute the analytic signal (phase + amplitude) via a Hilbert transform.

    Usage:
       h = Hilbert(seq_len)  # seq_len is the time dimension
       y = h(x)  # x.shape => (batch, ..., seq_len)
       # y.shape => same as x except an extra dimension for (phase, amplitude).
       # The last dimension of size 2 is [phase, amplitude].

    The transform is approximate and uses a smooth step in the frequency domain
    for differentiability. If you only need a standard transform, you can replace
    that logic with torch.fft-based sign manipulations.
    """

    def __init__(self, seq_len: int, dim: int = -1, fp16: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.fp16 = fp16

        # Frequencies for the entire spectral range
        # shape == (seq_len, )
        f = torch.cat([
            torch.arange(0, (seq_len - 1) // 2 + 1),
            torch.arange(-(seq_len // 2), 0)
        ]) / float(seq_len)
        self.register_buffer("f", f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Output shape is (..., 2), where the last dimension 2 is [phase, amplitude].
        """
        if self.fp16:
            x = x.half()

        # x_fft: complex64/128 => shape same as x
        Xf = torch.fft.fft(x.float(), n=self.seq_len, dim=self.dim)

        # Build a smooth step from 0 to 1 across positive frequencies
        # so that the transform is differentiable
        steepness = 50.0
        # shape: self.f
        hi = torch.sigmoid(steepness * self.f).to(Xf.device)

        # Multiply by 2 * step to get an analytic half
        Xf_trans = Xf * (2.0 * hi)

        # Go back to time domain
        x_analytic = torch.fft.ifft(Xf_trans, dim=self.dim)

        # Real/imag => amplitude + phase
        # For each point: amplitude = abs, phase = arctan2(imag, real)
        phase = torch.atan2(x_analytic.imag, x_analytic.real)
        amplitude = x_analytic.abs()

        # Concatenate along a new dimension => last dimension has size=2
        out = torch.stack([phase, amplitude], dim=-1)

        return out.float() if self.fp16 else out


# ------------------------------------------------------------------------
# 2) Modulation Index (MI) from phase and amplitude
# ------------------------------------------------------------------------
class ModulationIndex(nn.Module):
    """
    Computes the Modulation Index (Tort et al. 2010) for Phase-Amplitude Coupling.
    Optionally returns the amplitude distribution (amp_prob=True) instead of MI.
    """

    def __init__(self, n_bins: int = 18, fp16: bool = False, amp_prob: bool = False):
        super().__init__()
        self.n_bins = n_bins
        self.fp16 = fp16
        self.amp_prob = amp_prob
        # Phase bin cutoffs: from -pi to +pi
        pha_bins = torch.linspace(-np.pi, np.pi, n_bins + 1)
        self.register_buffer("pha_bin_cutoffs", pha_bins)

    def forward(self, pha: torch.Tensor, amp: torch.Tensor) -> torch.Tensor:
        """
        pha.shape => (B, ch, freq_pha, n_seg, T)
        amp.shape => (B, ch, freq_amp, n_seg, T)
        Return => MI with shape (B, ch, freq_pha, freq_amp)
                  or amplitude distribution if amp_prob=True.
        """
        if self.fp16:
            pha = pha.half()
            amp = amp.half()
        else:
            pha = pha.float()
            amp = amp.float()

        # One‐hot for each time point => which bin the phase belongs in
        n_bins = self.n_bins
        # bin_indices => which bin each time pt goes to
        bin_indices = (torch.bucketize(pha, self.pha_bin_cutoffs) - 1
                       ).clamp_(0, n_bins - 1)
        # shape => same as pha
        # Turn into one‐hot along last dimension
        # => shape (B, ch, freq_pha, n_seg, T, n_bins)
        one_hot_bins = F.one_hot(bin_indices.long(), num_classes=n_bins).bool()

        # We want to broadcast freq_amp dimension => insert it
        # amp => shape (B, ch, freq_amp, n_seg, T)
        # => (B, ch, freq_amp, n_seg, T, 1) so we can multiply with one_hot_bins
        amp_ex = amp.unsqueeze(2 - 2)  # hack: just unify them
        # Actually we want: freq_pha dimension is index=2, freq_amp dimension does not exist => we'll handle this carefully
        # We'll do a simpler approach: expand them to 6 dims to match one_hot_bins

        # Expand amp => (B, ch, freq_amp, n_seg, T, 1)
        amp_ex = amp_ex.unsqueeze(-1)
        # Expand one_hot_bins => (B, ch, freq_pha, n_seg, T, n_bins) => plus freq_amp axis
        # => insert freq_amp dimension after freq_pha => shape: (B, ch, freq_pha, freq_amp, n_seg, T, n_bins)
        # Easiest is to do: one_hot_bins => (B, ch, freq_pha, n_seg, T) => unsqueeze(3)
        one_hot_bins = one_hot_bins.unsqueeze(3)
        # => (B, ch, freq_pha, 1, n_seg, T, n_bins)
        # Now broadcast freq_amp: (B, ch, 1, freq_amp, n_seg, T, 1)
        # We'll reorder so that freq_amp is next to freq_pha
        amp_ex = amp_ex.unsqueeze(2)
        # => shape => (B, ch, 1, freq_amp, n_seg, T, 1)

        # Multiply => shape => (B, ch, freq_pha, freq_amp, n_seg, T, n_bins)
        amp_in_bins = amp_ex * one_hot_bins

        # Average amplitude in each bin => sum over time dimension => shape => (B, ch, freq_pha, freq_amp, n_seg, n_bins)
        # time dimension is -2 => T => we sum axis=-2
        # first sum amplitude within each bin
        sum_amp = amp_in_bins.sum(dim=-2)
        # also sum how many points are in each bin
        count_per_bin = one_hot_bins.sum(dim=-2) + 1e-9

        mean_amp_bins = sum_amp / count_per_bin

        # Probability distribution across bins => shape => (B, ch, freq_pha, freq_amp, n_seg, n_bins)
        prob_amp = mean_amp_bins / (mean_amp_bins.sum(dim=-1, keepdim=True) + 1e-9)

        if self.amp_prob:
            # shape => (B, ch, freq_pha, freq_amp, n_seg, n_bins)
            return prob_amp

        # Calculate the Tort-based Modulation Index => sum(prob * ln(prob))
        # Then we normalize by log(n_bins)
        log_nbins = float(np.log(n_bins))
        pac = (  # sum(prob ln(prob)) => negative => plus log(n_bins) => scaled
            torch.log(n_bins + 1e-9) + (prob_amp * (prob_amp + 1e-9).log()).sum(dim=-1)
        ) / log_nbins

        # => shape => (B, ch, freq_pha, freq_amp, n_seg)
        # Average across n_seg => shape => (B, ch, freq_pha, freq_amp)
        return pac.mean(dim=-1)


# ------------------------------------------------------------------------
# 3) BandPassFilter and optional DifferentiableBandPassFilter
# ------------------------------------------------------------------------
def design_filter_fir(numtaps, cutoff, fs, pass_zero=False):
    """
    A minimal FIR filter design using window method (like scipy.signal.firwin).
    """
    # Convert cutoff freq => fraction of Nyquist
    nyq = fs / 2.0
    if isinstance(cutoff, float):
        cutoff = [cutoff / nyq]
    else:
        cutoff = [c / nyq for c in cutoff]

    # Window function
    # use Hamming
    # Beta factor not needed for Hamming
    # Build filter using sinc => naive approach
    # We'll just rely on torch for a minimal version or do our own for bigger code
    # For a small example, let's do a direct construction:
    t = torch.arange(numtaps)
    mid = (numtaps - 1) / 2.0
    # sinc-based
    # We'll do pass_zero => True => "lowpass"
    # pass_zero => False => "highpass"
    if len(cutoff) == 1:
        # single cutoff => low or high
        # We skip the advanced weighting. For a real design, use windowed-sinc or torch builtin
        # This minimal version is purely illustrative.
        freq = cutoff[0]
        # sinc => sin(2*pi*freq*(t-mid)) / (t-mid)
        sinc_func = (2.0 * freq) * torch.sinc(2.0 * freq * (t - mid))
        # Hamming window
        win = 0.54 - 0.46 * torch.cos(2.0 * np.pi * (t / (numtaps - 1)))
        h = sinc_func * win
        if not pass_zero:
            # highpass => spectral inversion
            h = -h
            h[int(mid)] += 1.0
        return h / h.sum()

    else:
        # bandpass or bandstop => do difference of 2 lowpass
        # for simplicity
        freq1, freq2 = cutoff
        # lowpass at freq2 => h2
        h2 = design_filter_fir(numtaps, freq2 * nyq, fs, pass_zero=True)
        # lowpass at freq1 => h1
        h1 = design_filter_fir(numtaps, freq1 * nyq, fs, pass_zero=True)
        bp = h2 - h1
        if pass_zero:
            # bandstop => invert
            bp = -bp
            bp[int((numtaps - 1) / 2)] += 1.0
        return bp / bp.sum()


class BandPassFilter(nn.Module):
    """
    A simple multi‐band FIR filter (static, non‐trainable) for band‐pass or band‐stop.

    bands: shape (N, 2), each row = [low_hz, high_hz].
    fs   : sampling rate
    seq_len: used to guess order of filter
    """

    def __init__(self, bands: np.ndarray, fs: float, seq_len: int, fp16=False):
        super().__init__()
        self.fp16 = fp16

        # Build a stack of filter kernels => shape (N, kernel_size)
        # For each row in bands => design a FIR
        # We pick an order ~ min( ~3 cycles at the lowest freq, seq_len/4, etc. ) etc
        # As a minimal approach, we do a small fixed order:
        order = min(seq_len // 2, 512)
        kernels = []
        for row in bands:
            low, high = row
            # if user sets "bandstop" they'd define the same row but set is_bandstop => omitted for brevity
            # just do a bandpass
            cutoff = [low, high]
            # pass_zero => "False" => bandpass
            h = design_filter_fir(order, cutoff, fs, pass_zero=False)
            kernels.append(h)
        # => shape => (n_bands, order)
        self.register_buffer("kernels", torch.stack(kernels, dim=0).float())

    def forward(self, x: torch.Tensor, edge_len=0):
        """
        x.shape => (batch_size, n_segments, seq_len)
        Return => shape (batch_size, n_segments, n_filters, seq_len)
        """
        if self.fp16:
            x = x.half()
        B, S, T = x.shape
        # pad => replicate edge => "manual" or we do a small pad
        pad_len = self.kernels.shape[1] // 2
        xpad_left = x[..., :pad_len].flip(dims=[-1])
        xpad_right = x[..., -pad_len:].flip(dims=[-1])
        xext = torch.cat([xpad_left, x, xpad_right], dim=-1)
        # => shape => (B,S,T+2*pad_len)
        xext = xext.reshape(B * S, 1, T + 2 * pad_len)

        # conv => for each filter
        w = self.kernels.unsqueeze(1)  # => (n_filts,1,kernel_size)
        y = F.conv1d(xext.float(), w.float(), padding=0)
        # result => shape => (B*S, n_filts, T+2*pad_len - kernel_size+1 )
        # kernel_size = ...
        Lout = y.shape[-1]
        # remove the same pad => we want original length T
        # T + 2*pad_len - (order) + 1 => T if order=2*pad_len
        # so we should get shape => (B*S, n_filts, T)
        y = y[..., :T]
        y = y.reshape(B, S, w.shape[0], T)
        return y


class DifferentiableBandPassFilter(nn.Module):
    """
    A “trainable” multi‐band filter that splits into phase bands & amplitude bands,
    typically for Phase‐Amplitude Coupling. The center frequencies are parameters
    that can be learned, but the underlying filter shape is built from a minimal
    windowed‐sinc approach. This is more of a demonstration, not a production design.

    For typical usage, you can skip “trainable” mode and just use BandPassFilter.
    """

    def __init__(
        self,
        sig_len: int,
        fs: float,
        pha_low_hz=2.0,
        pha_high_hz=20.0,
        pha_n_bands=30,
        amp_low_hz=60.0,
        amp_end_hz=160.0,
        amp_n_bands=50,
        cycle=3,
        fp16=False,
    ):
        super().__init__()
        self.sig_len = sig_len
        self.fs = fs
        self.cycle = cycle
        self.fp16 = fp16

        # Define center frequencies as trainable
        self.pha_mids = nn.Parameter(
            torch.linspace(pha_low_hz, pha_high_hz, pha_n_bands)
        )
        self.amp_mids = nn.Parameter(
            torch.linspace(amp_low_hz, amp_end_hz, amp_n_bands)
        )

        # Build initial set of filters
        self.register_buffer("kernels", self._build_filters())

    def forward(self, x: torch.Tensor, edge_len=0):
        # rebuild kernels on each forward => center freq might have changed
        self.kernels = self._build_filters().to(x.device)
        # shape => (n_filters, kernel_size)

        if self.fp16:
            x = x.half()

        B, S, T = x.shape
        pad_len = self.kernels.shape[1] // 2
        xpad_left = x[..., :pad_len].flip(dims=[-1])
        xpad_right = x[..., -pad_len:].flip(dims=[-1])
        xext = torch.cat([xpad_left, x, xpad_right], dim=-1)
        xext = xext.reshape(B * S, 1, T + 2 * pad_len)

        w = self.kernels.unsqueeze(1)  # => shape => (n_filts,1,kernel_size)
        y = F.conv1d(xext.float(), w.float(), padding=0)
        # => shape => (B*S, n_filts, T)
        y = y[..., :T]
        y = y.reshape(B, S, w.shape[0], T)
        return y

    def _build_filters(self):
        # For each center freq => define band edges => build FIR
        # For “phase” we do a narrow band => e.g. +/- 25% of center
        # For “amp” we do e.g. +/- 12.5% of center
        # Here we’ll do a minimal approach
        # Collect them all into a single list => final shape => stacked
        f_pha_lo = self.pha_mids - self.pha_mids / 4.0
        f_pha_hi = self.pha_mids + self.pha_mids / 4.0
        f_amp_lo = self.amp_mids - self.amp_mids / 8.0
        f_amp_hi = self.amp_mids + self.amp_mids / 8.0

        # Combine all
        all_lo = torch.cat([f_pha_lo, f_amp_lo], dim=0)
        all_hi = torch.cat([f_pha_hi, f_amp_hi], dim=0)

        # pick a filter order
        # For demonstration, fix some default
        order = min(self.sig_len // 2, 512)

        all_kernels = []
        for low, high in zip(all_lo, all_hi):
            # clamp
            low = max(low.item(), 0.1)
            high = max(high.item(), low + 1.0)
            if high > (self.fs / 2.0 - 0.1):
                high = self.fs / 2.0 - 0.1

            cutoff = [low, high]
            h = design_filter_fir(order, cutoff, self.fs, pass_zero=False)
            all_kernels.append(h)
        return torch.stack(all_kernels, dim=0)


# ------------------------------------------------------------------------
# 4) The main PAC class
# ------------------------------------------------------------------------
class PAC(nn.Module):
    """
    A neural network module that computes Phase-Amplitude Coupling (PAC):
      1) Bandpass filter the input into phase bands and amplitude bands
      2) Hilbert transform: obtains phase and amplitude
      3) Evaluate Modulation Index (or amplitude distribution) across “pha x amp” freq pairs
      4) Optionally produce surrogate-based z‐scores

    Typically used internally by the “pac(...)” function.
    """

    def __init__(
        self,
        seq_len: int,
        fs: float,
        pha_start_hz=2.0,
        pha_end_hz=20.0,
        pha_n_bands=50,
        amp_start_hz=60.0,
        amp_end_hz=160.0,
        amp_n_bands=30,
        n_perm: int = None,
        trainable: bool = False,
        fp16: bool = False,
        amp_prob: bool = False,
    ):
        super().__init__()
        self.fp16 = fp16
        self.n_perm = n_perm
        self.amp_prob = amp_prob
        self.trainable = trainable

        # Pick a bandpass method
        if trainable:
            self.bandpass = DifferentiableBandPassFilter(
                seq_len,
                fs,
                pha_start_hz,
                pha_end_hz,
                pha_n_bands,
                amp_start_hz,
                amp_end_hz,
                amp_n_bands,
                fp16=fp16,
            )
            # Expose the actual mid frequencies
            self.PHA_MIDS_HZ = self.bandpass.pha_mids
            self.AMP_MIDS_HZ = self.bandpass.amp_mids
        else:
            # create “static" sets
            def build_bands(lo, hi, n, scale="phase"):
                # narrower for low freq => e.g. +/- 25%
                # or amplitude => +/- 12.5%
                if n < 1 or lo >= hi:
                    return np.array([[2, 3]])
                mid = np.linspace(lo, hi, n)
                if scale == "phase":
                    # +/-25%
                    arr = np.column_stack([mid - mid / 4, mid + mid / 4])
                else:
                    # +/-12.5%
                    arr = np.column_stack([mid - mid / 8, mid + mid / 8])
                arr[arr < 0.1] = 0.1
                # also clamp top to fs/2 - 1
                return arr

            pha_bands = build_bands(pha_start_hz, pha_end_hz, pha_n_bands, "phase")
            amp_bands = build_bands(amp_start_hz, amp_end_hz, amp_n_bands, "amp")
            self.bandpass = BandPassFilter(
                np.vstack([pha_bands, amp_bands]), fs, seq_len, fp16=fp16
            )
            self.PHA_MIDS_HZ = torch.tensor(pha_bands.mean(axis=1), dtype=torch.float)
            self.AMP_MIDS_HZ = torch.tensor(amp_bands.mean(axis=1), dtype=torch.float)

        # Hilbert transform => get phase, amplitude
        self.hilbert = Hilbert(seq_len, dim=-1, fp16=fp16)

        # Compute MI
        self.mod_index = ModulationIndex(n_bins=18, fp16=fp16, amp_prob=amp_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape => (B, ch, seq_len) or (B, ch, n_seg, seq_len).

        Return => shape (B, ch, [pha_n_bands, amp_n_bands]) or
                  amplitude distributions => depends on amp_prob
                  If n_perm is not None => returns surrogate z‐scores
        """
        # Ensure shape => (B, ch, n_seg, T)
        if x.ndim == 3:
            x = x.unsqueeze(2)  # => (B, ch, 1, T)

        bsz, ch, n_seg, T = x.shape

        # 1) bandpass => shape => (B, ch, n_seg, n_filt, T)
        # We'll merge B & ch => treat them as a bigger batch
        x_merged = x.reshape(bsz * ch, n_seg, T)
        xfilt = self.bandpass(x_merged)  # => (bsz*ch, n_seg, n_filt, T)
        # reshape => (B, ch, n_seg, n_filt, T)
        xfilt = xfilt.view(bsz, ch, n_seg, -1, T)

        # 2) Hilbert => separate into [phase, amplitude]
        # xfilt => we have n_filters along "dim=3"
        # shape => (B, ch, n_seg, n_filters, T)
        # apply Hilbert to each filter => produce => (B, ch, n_seg, n_filters, T, 2)
        # We'll reorder => (B,ch,n_filters,n_seg,T,2) for convenience:
        xfilt = xfilt.permute(0, 1, 3, 2, 4)
        B, C, NF, S, L = xfilt.shape

        # Flatten => (B*C*NF*S, L)
        xflat = xfilt.reshape(B * C * NF * S, L)
        xh = self.hilbert(xflat)  # => shape => (B*C*NF*S, L, 2)
        # return shape => (B*C, NF, S, L, 2) => we want to unflatten
        xh = xh.view(B, C, NF, S, L, 2)

        # The first portion of filters => phase filters
        n_pha = len(self.PHA_MIDS_HZ)
        # The second portion => amplitude
        n_amp = len(self.AMP_MIDS_HZ)

        # xfilt => [0 : n_pha] => phase, [n_pha : n_pha + n_amp] => amplitude
        # => shape => (B, ch, n_pha + n_amp, S, L, 2)
        # separate them:
        # phase => (B, ch, n_pha, S, L, 2)
        pha_part = xh[:, :, :n_pha, ...]
        # amplitude => (B, ch, n_amp, S, L, 2)
        amp_part = xh[:, :, n_pha:n_pha + n_amp, ...]

        # Extract [phase, amplitude]
        # phase => index=0, amplitude => index=1
        # shape => (B, ch, n_pha, S, L)
        pha_vals = pha_part[..., 0]
        # shape => (B, ch, n_pha, S, L)
        # amplitude => index=1 => (B, ch, n_pha, S, L)
        # we want amplitude => from amp_part => index=1
        amp_vals = amp_part[..., 1]

        # 3) pass to ModulationIndex => shape => (B, ch, n_pha, n_amp) or amplitude distribution
        # We have phase => shape => (B, ch, n_pha, S, L)
        # amplitude => shape => (B, ch, n_amp, S, L)
        # => but we want => (B, ch, n_freqs_pha, S, L) & (B, ch, n_freqs_amp, S, L)
        # We'll rename => (B, ch, freq_pha, S, T), etc
        # So we must unify the freq dimension => we call mod_index(pha_vals, amp_vals)
        # but we must feed => (B,ch,freq_pha,S,L), (B,ch,freq_amp,S,L).
        # Our mod_index expects => (B, ch, freq_pha, S, L) & (B, ch, freq_amp, S, L)

        pac_observed = self.mod_index(pha_vals, amp_vals)

        # Surrogates => optional
        if self.n_perm is None:
            return pac_observed

        # If we do surrogates => zscore
        # “shift in time dimension” approach => random roll on phase
        surrogates = self._generate_surrogates(pha_vals, amp_vals)
        # shape => (B, ch, freq_pha, freq_amp, self.n_perm)
        # mean & std across surrogates => shape => (B, ch, freq_pha, freq_amp)
        mm = surrogates.mean(dim=-1)
        ss = surrogates.std(dim=-1) + 1e-9
        zscore = (pac_observed - mm) / ss
        return zscore

    def _generate_surrogates(self, pha, amp):
        """
        Make surrogates by circularly shifting the phase in time dimension.
        Returns MI for each surrogate => shape (B, ch, freq_pha, freq_amp, n_perm).
        """
        B, C, NP, S, T = pha.shape
        NA = amp.shape[2]

        # We'll collect each surrogate's MI in a list
        out = []
        for _ in range(self.n_perm):
            # shift in time => random offset
            shift = torch.randint(0, T, (1,), device=pha.device).item()
            # roll the entire phase => shape => same as pha
            pha_roll = torch.roll(pha, shifts=shift, dims=-1)
            # compute MI => shape => (B, ch, NP, NA)
            val = self.mod_index(pha_roll, amp)
            out.append(val.unsqueeze(-1))  # shape => (B,ch,NP,NA,1)

        return torch.cat(out, dim=-1)  # => (B,ch,NP,NA,n_perm)


# ------------------------------------------------------------------------
# 5) Finally, the user‐facing pac(...) function
# ------------------------------------------------------------------------
def pac(
    x: torch.Tensor,
    fs: float,
    pha_start_hz: float = 2,
    pha_end_hz: float = 20,
    pha_n_bands: int = 100,
    amp_start_hz: float = 60,
    amp_end_hz: float = 160,
    amp_n_bands: int = 100,
    fp16: bool = False,
    trainable: bool = False,
    n_perm: int = None,
    amp_prob: bool = False,
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    A convenient wrapper function to compute Phase‐Amplitude Coupling (PAC)
    from an input signal. Internally uses the PAC(nn.Module) class above.

    Arguments
    ---------
    x : (torch.Tensor)
        Input data: shape (batch, channels, [optional: segments], time).
        If only 3D, we treat segments=1 automatically.
    fs : float
        Sampling frequency in Hz.
    pha_start_hz, pha_end_hz, pha_n_bands
        Phase frequency range & how many phase band filters.
    amp_start_hz, amp_end_hz, amp_n_bands
        Amplitude frequency range & how many amplitude band filters.
    fp16 : bool
        Use half precision if True. (Experimental)
    trainable : bool
        If True, uses a differentiable bandpass filter so frequency boundaries can be learned.
        If False, uses static bandpass filters.
    n_perm : Optional[int]
        Number of surrogate permutations to compute & produce z‐scores. If None => no surrogates.
    amp_prob : bool
        If True => returns the amplitude distribution across phase bins, rather than the numeric MI.

    Returns
    -------
    pac_or_dist : torch.Tensor
        Shape => (batch, channels, n_pha, n_amp) if amp_prob=False => the PAC or z‐score PAC
                  or if amp_prob=True => (B, ch, n_pha, n_amp, n_seg, n_bins).
    pha_mids : torch.Tensor
        The center frequencies used for the “phase” band filters.
    amp_mids : torch.Tensor
        The center frequencies used for the “amplitude” band filters.

    Example
    -------
        import torch
        import numpy as np

        # Suppose we have an EEG of shape (batch=2, ch=4, time=2000)
        x = torch.randn(2, 4, 2000)
        fs = 512
        pac_values, pha_mids, amp_mids = pac(x, fs, pha_start_hz=2, pha_end_hz=30)
        print(pac_values.shape)  # e.g. => (2, 4, 100, 100)
    """
    # We create an internal PAC object
    model = PAC(
        seq_len=x.shape[-1],
        fs=fs,
        pha_start_hz=pha_start_hz,
        pha_end_hz=pha_end_hz,
        pha_n_bands=pha_n_bands,
        amp_start_hz=amp_start_hz,
        amp_end_hz=amp_end_hz,
        amp_n_bands=amp_n_bands,
        n_perm=n_perm,
        trainable=trainable,
        fp16=fp16,
        amp_prob=amp_prob,
    )

    # Run forward => shape => (B, ch, n_pha, n_amp)
    pac_out = model(x)
    # Return PAC & the freq midpoints
    return pac_out, model.PHA_MIDS_HZ.detach().cpu(), model.AMP_MIDS_HZ.detach().cpu()

--------------------------------------------------------------------------------

HOW TO USE:

1. Save the above code to a file named “gPAC.py”.  
2. In your script or notebook, install PyTorch (and NumPy):  
   pip install torch numpy  
3. Then import and call:

   import torch
   import numpy as np
   import gPAC

   # Example signal: 2 batches, 4 channels, length=1024
   x = torch.randn(2, 4, 1024)
   fs = 512.0

   pac_values, pha_mids, amp_mids = gPAC.pac(
       x,
       fs,
       pha_start_hz=2, pha_end_hz=20, pha_n_bands=50,
       amp_start_hz=60, amp_end_hz=160, amp_n_bands=50,
       n_perm=10,  # for surrogate-based z-scores
       trainable=False,
       fp16=False,
       amp_prob=False
   )
   print("PAC shape:", pac_values.shape)
   print("Centers for phase freq (Hz):", pha_mids)
   print("Centers for amp freq (Hz):", amp_mids)

This completes the minimal isolation of mngs.dsp.pac() into a single “gPAC” module  
(with support code for bandpass filtering, Hilbert transform, and ModulationIndex).

<!-- EOF -->