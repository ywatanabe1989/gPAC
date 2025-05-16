#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 13:45:10 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/examples/05_advanced_statistical_analysis.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/05_advanced_statistical_analysis.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Advanced PAC analysis using surrogate distributions.

This example demonstrates:
1. Retrieving the full surrogate distribution from PAC calculations
2. Performing custom statistical analyses on the distribution
3. Calculating p-values and correcting for multiple comparisons
4. Visualizing surrogate distributions for different frequency pairs
5. Creating advanced statistical visualizations

Requirements:
- gpac
- numpy
- torch
- matplotlib
- seaborn
- statsmodels (for multiple comparison correction)
"""

import gc
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.stats.multitest import multipletests
import gpac

matplotlib.use("Agg")  # Non-interactive backend

# Set plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.2)

# Create output directory for figures
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__FILE__)), "examples/figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Parameters
FS = 1000  # Sampling frequency in Hz
DURATION = 10  # Signal duration in seconds
N_CHANNELS = 3  # Number of channels to generate
SEED = 42  # For reproducibility


def create_test_signals(fs, duration, n_channels):
    """
    Create multiple synthetic test signals with different PAC properties.
    
    Returns:
        torch.Tensor: Signals with shape (1, n_channels, 1, time)
        list: Ground truth PAC pairs (phase_hz, amplitude_hz) for each channel
    """
    # Set random seed for reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Time vector
    t = np.arange(0, duration, 1 / fs)
    time_points = len(t)
    
    # Initialize tensor for all signals: (1, n_channels, 1, time)
    signals = torch.zeros((1, n_channels, 1, time_points), dtype=torch.float32)
    
    # Define PAC configurations for each channel
    pac_config = []
    
    # Channel 0: Strong coupling between theta (6 Hz) and gamma (80 Hz)
    signals[0, 0, 0, :] = create_pac_signal(t, 6.0, 80.0, coupling_strength=0.8)
    pac_config.append((6.0, 80.0, 'Strong'))
    
    # Channel 1: Moderate coupling between alpha (10 Hz) and high gamma (110 Hz)
    signals[0, 1, 0, :] = create_pac_signal(t, 10.0, 110.0, coupling_strength=0.5)
    pac_config.append((10.0, 110.0, 'Moderate'))
    
    # Channel 2: Weak/No coupling (control)
    signals[0, 2, 0, :] = create_pac_signal(t, 4.0, 70.0, coupling_strength=0.1)
    pac_config.append((4.0, 70.0, 'Weak/None'))
    
    return signals, pac_config


def create_pac_signal(t, phase_freq, amp_freq, coupling_strength=0.8):
    """Create a single synthetic signal with PAC between specified frequencies."""
    # Phase signal (slow oscillation)
    phase_signal = np.sin(2 * np.pi * phase_freq * t)
    
    # Amplitude signal (fast oscillation)
    amp_carrier = np.sin(2 * np.pi * amp_freq * t)
    
    # Modulate amplitude by phase
    amplitude_modulation = (1 + coupling_strength * phase_signal) / 2
    pac_signal = amplitude_modulation * amp_carrier
    
    # Add phase component and noise
    noise = np.random.normal(0, 0.1, len(t))
    signal = phase_signal * 0.5 + pac_signal + noise
    
    return torch.from_numpy(signal.astype(np.float32))


def calculate_custom_statistics(pac_values, surrogate_dist):
    """
    Calculate custom statistics from the surrogate distribution.
    
    Args:
        pac_values: The observed PAC values tensor
        surrogate_dist: The surrogate distribution tensor
        
    Returns:
        dict: Dictionary of statistical metrics
    """
    # Convert tensors to NumPy for statistical analysis
    pac_np = pac_values.cpu().numpy()
    surr_np = surrogate_dist.cpu().numpy()
    
    # Calculate p-values (fraction of surrogates >= observed)
    p_values = np.mean(surr_np >= pac_np, axis=0)
    
    # Apply false discovery rate correction for multiple comparisons
    p_values_flat = p_values.reshape(-1)
    reject, p_corrected, _, _ = multipletests(p_values_flat, method='fdr_bh', alpha=0.05)
    p_corrected = p_corrected.reshape(p_values.shape)
    significance_mask = p_corrected < 0.05
    
    # Calculate standardized effect sizes: z-scores and Cohen's d
    mean_surr = np.mean(surr_np, axis=0)
    std_surr = np.std(surr_np, axis=0)
    z_scores = (pac_np - mean_surr) / (std_surr + 1e-9)  # Add small epsilon to avoid division by zero
    
    # Cohen's d = (observed - mean_surrogate) / pooled_std
    # Since n_observed = 1, this simplifies to (observed - mean_surrogate) / std_surrogate
    cohens_d = (pac_np - mean_surr) / (std_surr + 1e-9)
    
    # Calculate the 95% confidence interval from surrogate distribution
    ci_lower = np.percentile(surr_np, 2.5, axis=0)
    ci_upper = np.percentile(surr_np, 97.5, axis=0)
    
    return {
        'p_values': p_values,
        'p_corrected': p_corrected,
        'significance_mask': significance_mask,
        'z_scores': z_scores,
        'cohens_d': cohens_d,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mean_surr': mean_surr,
        'std_surr': std_surr
    }


def plot_significance_maps(stats, pha_freqs, amp_freqs, pac_config):
    """
    Plot PAC significance maps with different statistical metrics.
    
    Args:
        stats: Dictionary of statistical metrics
        pha_freqs: Array of phase frequencies
        amp_freqs: Array of amplitude frequencies
        pac_config: List of ground truth PAC configurations
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Map of p-values
    for ch in range(3):
        # Get metrics for this channel
        p_vals = stats['p_values'][0, ch]
        p_corr = stats['p_corrected'][0, ch]
        sig_mask = stats['significance_mask'][0, ch]
        z_scores = stats['z_scores'][0, ch]
        
        # Get ground truth info for this channel
        true_pha, true_amp, coupling = pac_config[ch]
        
        # Plot p-value map
        im = axes[ch, 0].imshow(
            -np.log10(p_vals),  # -log10 transformation for better visualization
            aspect='auto',
            origin='lower',
            cmap='viridis',
            vmin=0,
            vmax=4  # Corresponds to p = 0.0001
        )
        
        # Add significance contour
        axes[ch, 0].contour(sig_mask, levels=[0.5], colors='r', linewidths=1.5)
        
        # Set titles and labels
        axes[ch, 0].set_title(f'Channel {ch}: {coupling} Coupling\n-log10(p-value) Map')
        
        # Plot z-scores
        vmax = 5.0  # Maximum z-score for visualization
        im = axes[ch, 1].imshow(
            z_scores,
            aspect='auto',
            origin='lower',
            cmap='RdBu_r',
            vmin=-vmax,
            vmax=vmax
        )
        
        # Mark significant areas
        axes[ch, 1].contour(sig_mask, levels=[0.5], colors='k', linewidths=1.0)
        axes[ch, 1].set_title(f'Channel {ch}: Z-scores\n(with significance contour)')
        
        # Plot significant points only
        masked_z = np.zeros_like(z_scores)
        masked_z[sig_mask] = z_scores[sig_mask]
        
        im = axes[ch, 2].imshow(
            masked_z,
            aspect='auto',
            origin='lower',
            cmap='hot',
            vmin=0,
            vmax=vmax
        )
        axes[ch, 2].set_title(f'Channel {ch}: Significant Z-scores only')
        
        # Mark ground truth location
        true_pha_idx = np.argmin(np.abs(pha_freqs - true_pha))
        true_amp_idx = np.argmin(np.abs(amp_freqs - true_amp))
        
        for ax in axes[ch, :]:
            ax.plot(true_amp_idx, true_pha_idx, 'o', 
                    markerfacecolor='none', markeredgecolor='green', 
                    markersize=10, markeredgewidth=2)
            
            # Add frequency axis labels
            if ch == 2:  # Only for bottom row
                ax.set_xlabel('Amplitude Frequency (Hz)')
            
            # Add ticks at every 3rd frequency
            ax.set_xticks(np.arange(0, len(amp_freqs), 3))
            ax.set_xticklabels([f"{f:.1f}" for f in amp_freqs[::3]], rotation=45)
            
            ax.set_yticks(np.arange(0, len(pha_freqs), 2))
            ax.set_yticklabels([f"{f:.1f}" for f in pha_freqs[::2]])
            
        axes[ch, 0].set_ylabel('Phase Frequency (Hz)')
    
    # Add colorbar for each column
    cbar_ax1 = fig.add_axes([0.92, 0.66, 0.02, 0.2])
    cbar1 = fig.colorbar(im, cax=cbar_ax1)
    cbar1.set_label('-log10(p-value)')
    
    cbar_ax2 = fig.add_axes([0.92, 0.36, 0.02, 0.2])
    cbar2 = fig.colorbar(im, cax=cbar_ax2)
    cbar2.set_label('Z-score')
    
    cbar_ax3 = fig.add_axes([0.92, 0.06, 0.02, 0.2])
    cbar3 = fig.colorbar(im, cax=cbar_ax3)
    cbar3.set_label('Significant Z-score')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    save_path = os.path.join(FIGURE_DIR, '05_pac_significance_maps.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {os.path.abspath(save_path)}")
    plt.close()


def plot_distribution_comparisons(pac_values, surrogate_dist, stats, pha_freqs, amp_freqs, pac_config):
    """
    Plot distribution comparisons for different channels and frequency pairs.
    
    Args:
        pac_values: Observed PAC values
        surrogate_dist: Surrogate distributions 
        stats: Dictionary of statistical metrics
        pha_freqs: Array of phase frequencies
        amp_freqs: Array of amplitude frequencies
        pac_config: List of ground truth PAC configurations
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    for ch in range(3):
        # Get ground truth info for this channel
        true_pha, true_amp, coupling = pac_config[ch]
        
        # Find index of ground truth frequencies
        true_pha_idx = np.argmin(np.abs(pha_freqs - true_pha))
        true_amp_idx = np.argmin(np.abs(amp_freqs - true_amp))
        
        # Get observed values and distributions at ground truth frequencies
        observed = pac_values[0, ch, true_pha_idx, true_amp_idx].cpu().item()
        surrogates = surrogate_dist[:, 0, ch, true_pha_idx, true_amp_idx].cpu().numpy()
        
        # Get statistics
        p_value = stats['p_values'][0, ch, true_pha_idx, true_amp_idx]
        p_corr = stats['p_corrected'][0, ch, true_pha_idx, true_amp_idx]
        z_score = stats['z_scores'][0, ch, true_pha_idx, true_amp_idx]
        ci_lower = stats['ci_lower'][0, ch, true_pha_idx, true_amp_idx]
        ci_upper = stats['ci_upper'][0, ch, true_pha_idx, true_amp_idx]
        
        # Left column: histogram with distribution
        ax = axes[ch, 0]
        sns.histplot(surrogates, bins=20, kde=True, ax=ax, color='skyblue', alpha=0.7)
        
        # Plot observed value
        ax.axvline(observed, color='red', linestyle='--', linewidth=2, 
                   label=f'Observed: {observed:.3f}')
        
        # Plot 95% confidence interval
        ax.axvline(ci_lower, color='black', linestyle=':', linewidth=1, 
                   label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
        ax.axvline(ci_upper, color='black', linestyle=':', linewidth=1)
        
        # Add title and annotations
        is_significant = p_corr < 0.05
        sig_marker = "✓" if is_significant else "✗"
        sig_color = "green" if is_significant else "red"
        
        ax.set_title(f'Channel {ch}: {coupling} Coupling\n' 
                     f'Phase: {true_pha} Hz, Amp: {true_amp} Hz\n'
                     f'Significant: {sig_marker} (p-corr = {p_corr:.4f})',
                     color=sig_color)
        
        ax.set_xlabel('PAC Value')
        ax.set_ylabel('Count')
        ax.legend(loc='upper right')
        
        # Highlight if significant
        if is_significant:
            ax.patch.set_facecolor((0.9, 1.0, 0.9))
            ax.patch.set_alpha(0.3)
        
        # Right column: QQ plot of surrogates vs normal distribution
        ax = axes[ch, 1]
        
        # Create QQ plot
        from scipy import stats as scipy_stats
        
        # Standardize the surrogates for QQ plot
        surr_std = (surrogates - np.mean(surrogates)) / np.std(surrogates)
        qq = scipy_stats.probplot(surr_std, dist='norm', plot=ax)
        
        # Add reference line
        x = np.linspace(-3, 3, 100)
        ax.plot(x, x, 'r--', linewidth=1)
        
        # Add title and labels
        ax.set_title(f'Channel {ch}: Normal QQ Plot\nZ-score: {z_score:.2f}')
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        
        # Add the standardized observed value to the plot
        observed_std = (observed - np.mean(surrogates)) / np.std(surrogates)
        ax.scatter([scipy_stats.norm.ppf(0.5)], [observed_std], color='red', s=80, 
                   label=f'Observed (z={observed_std:.2f})')
        ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, '05_surrogate_distributions_comparison.png'), dpi=300, 
                bbox_inches='tight')
    plt.close()


def plot_effect_size_comparison(stats, pha_freqs, amp_freqs, pac_config):
    """
    Plot effect size comparison across different channels.
    
    Args:
        stats: Dictionary of statistical metrics
        pha_freqs: Array of phase frequencies
        amp_freqs: Array of amplitude frequencies
        pac_config: List of ground truth PAC configurations  
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot settings
    effect_size_cmap = 'RdYlBu_r'
    vmax = max(1.5, np.max(stats['cohens_d']) * 0.8)  # Scale based on data
    
    # Channel comparison plots
    for ch in range(3):
        # Get ground truth info
        true_pha, true_amp, coupling = pac_config[ch]
        true_pha_idx = np.argmin(np.abs(pha_freqs - true_pha))
        true_amp_idx = np.argmin(np.abs(amp_freqs - true_amp))
        
        # Extract effect sizes (Cohen's d)
        effect_size = stats['cohens_d'][0, ch]
        
        # Mask non-significant regions
        sig_mask = stats['significance_mask'][0, ch]
        masked_effect = np.ma.masked_array(effect_size, mask=~sig_mask)
        
        # Plot effect size map with significance masking
        im = axes[ch].imshow(
            masked_effect,
            aspect='auto',
            origin='lower',
            cmap=effect_size_cmap,
            vmin=-vmax,
            vmax=vmax,
            interpolation='none'
        )
        
        # Add title and labels
        axes[ch].set_title(f'Channel {ch}: {coupling} Coupling\nEffect Size (Cohen\'s d)')
        axes[ch].set_xlabel('Amplitude Frequency (Hz)')
        if ch == 0:
            axes[ch].set_ylabel('Phase Frequency (Hz)')
        
        # Add ticks
        axes[ch].set_xticks(np.arange(0, len(amp_freqs), 3))
        axes[ch].set_xticklabels([f"{f:.1f}" for f in amp_freqs[::3]], rotation=45)
        axes[ch].set_yticks(np.arange(0, len(pha_freqs), 2))
        axes[ch].set_yticklabels([f"{f:.1f}" for f in pha_freqs[::2]])
        
        # Mark ground truth location
        axes[ch].plot(true_amp_idx, true_pha_idx, 'o', 
                markerfacecolor='none', markeredgecolor='green', 
                markersize=10, markeredgewidth=2)
        
        # Annotate effect size at ground truth
        gt_effect = stats['cohens_d'][0, ch, true_pha_idx, true_amp_idx]
        gt_p = stats['p_corrected'][0, ch, true_pha_idx, true_amp_idx]
        
        # Position the annotation to avoid overlap
        xy_offset = (-20, -20) if true_pha_idx < len(pha_freqs) // 2 else (20, -20)
        
        axes[ch].annotate(
            f'd = {gt_effect:.2f}\np = {gt_p:.4f}',
            xy=(true_amp_idx, true_pha_idx),
            xytext=xy_offset,
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8)
        )
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Effect Size (Cohen\'s d)')
    
    # Add interpretation guide for effect sizes
    effect_guide = {
        0.2: 'Small',
        0.5: 'Medium',
        0.8: 'Large'
    }
    
    for effect, label in effect_guide.items():
        if effect <= vmax:
            cbar.ax.axhline(y=effect, color='black', linestyle='--', linewidth=0.8)
            cbar.ax.text(3.5, effect, f' {label} ({effect})', va='center', ha='left', fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(FIGURE_DIR, '05_effect_size_comparison.png'), dpi=300, 
                bbox_inches='tight')
    plt.close()


def plot_surrogates_3d(surrogate_dist, pac_values, pha_freqs, amp_freqs, pac_config):
    """
    Create a 3D visualization of the surrogate distributions.
    
    Args:
        surrogate_dist: Surrogate distribution tensor
        pac_values: Observed PAC values tensor
        pha_freqs: Array of phase frequencies
        amp_freqs: Array of amplitude frequencies
        pac_config: List of ground truth PAC configurations
    """
    # Only create this visualization if we have a reasonable number of surrogates
    if surrogate_dist.shape[0] < 30:
        return
        
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Select a single channel for visualization (strong coupling)
    ch = 0  # Channel with strong coupling
    true_pha, true_amp, _ = pac_config[ch]
    
    # Find frequencies near the ground truth
    true_pha_idx = np.argmin(np.abs(pha_freqs - true_pha))
    true_amp_idx = np.argmin(np.abs(amp_freqs - true_amp))
    
    # Define a window around the ground truth
    window_size = 2
    pha_slice = slice(max(0, true_pha_idx - window_size), 
                      min(len(pha_freqs), true_pha_idx + window_size + 1))
    amp_slice = slice(max(0, true_amp_idx - window_size), 
                      min(len(amp_freqs), true_amp_idx + window_size + 1))
    
    # Extract the subset of frequencies
    pha_subset = pha_freqs[pha_slice]
    amp_subset = amp_freqs[amp_slice]
    
    # Create meshgrid for 3D plot
    X, Y = np.meshgrid(range(len(amp_subset)), range(len(pha_subset)))
    
    # Plot surrogate distributions as vertical lines
    for p_idx, pha_idx in enumerate(range(pha_slice.start, pha_slice.stop)):
        for a_idx, amp_idx in enumerate(range(amp_slice.start, amp_slice.stop)):
            # Get surrogate values for this frequency pair
            surr_values = surrogate_dist[:, 0, ch, pha_idx, amp_idx].cpu().numpy()
            
            # Plot vertical bars for each surrogate (limit to 50 for clarity)
            max_surr_to_plot = min(50, len(surr_values))
            z_base = np.zeros(max_surr_to_plot)
            
            # Calculate frequencies for labeling
            pha_freq = pha_freqs[pha_idx]
            amp_freq = amp_freqs[amp_idx]
            
            # Set color based on distance from ground truth
            dist_from_gt = np.sqrt(
                ((pha_idx - true_pha_idx) / len(pha_freqs))**2 + 
                ((amp_idx - true_amp_idx) / len(amp_freqs))**2
            )
            color = plt.cm.plasma(1 - dist_from_gt)
            
            # Plot surrogates as stems
            ax.scatter([a_idx] * max_surr_to_plot, 
                      [p_idx] * max_surr_to_plot, 
                      surr_values[:max_surr_to_plot], 
                      marker='o', s=5, color=color, alpha=0.2)
            
            # Plot observed value as a larger marker
            observed = pac_values[0, ch, pha_idx, amp_idx].cpu().item()
            ax.scatter([a_idx], [p_idx], [observed], color='red', s=50, 
                      marker='o', label=f'Observed' if pha_idx == true_pha_idx and 
                                                       amp_idx == true_amp_idx else "")
            
            # For the ground truth frequencies, add mean and CI
            if pha_idx == true_pha_idx and amp_idx == true_amp_idx:
                mean_surr = np.mean(surr_values)
                ci_lower = np.percentile(surr_values, 2.5)
                ci_upper = np.percentile(surr_values, 97.5)
                
                # Plot mean of surrogates
                ax.scatter([a_idx], [p_idx], [mean_surr], color='blue', s=50, 
                          marker='s', label='Surrogate Mean')
                
                # Plot CI as vertical line
                ax.plot([a_idx, a_idx], [p_idx, p_idx], [ci_lower, ci_upper], 
                       'k-', linewidth=2, label='95% CI')
    
    # Setting labels and title
    ax.set_xlabel('Amplitude Frequency Index')
    ax.set_ylabel('Phase Frequency Index')
    ax.set_zlabel('PAC Value')
    
    # Set custom tick labels
    ax.set_xticks(range(len(amp_subset)))
    ax.set_xticklabels([f'{f:.1f}' for f in amp_subset])
    ax.set_yticks(range(len(pha_subset)))
    ax.set_yticklabels([f'{f:.1f}' for f in pha_subset])
    
    ax.set_title(f'3D Visualization of Surrogate Distributions Around Ground Truth\n'
                f'Channel {ch}: Phase {true_pha} Hz, Amplitude {true_amp} Hz')
    
    # Add legend (only for unique entries)
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc='upper center', bbox_to_anchor=(0.5, -0.05))
    
    # Set viewpoint
    ax.view_init(elev=25, azim=30)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, '05_surrogate_distributions_3d.png'), dpi=300, 
                bbox_inches='tight')
    plt.close()


def main():
    """Main function to run the example."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create test signals with different PAC configurations
    print("Creating test signals with different PAC strengths...")
    signals, pac_config = create_test_signals(FS, DURATION, N_CHANNELS)
    signals = signals.to(device) if device == "cuda" else signals

    # Calculate PAC with surrogate distribution
    print(f"Calculating PAC with surrogate distributions (on {device})...")
    
    # Use more permutations for a smoother distribution
    n_permutations = 200
    
    # Define frequency ranges for analysis
    pac_values, surrogate_dist, pha_freqs, amp_freqs = gpac.calculate_pac(
        signal=signals,
        fs=FS,
        pha_start_hz=2.0,
        pha_end_hz=15.0,
        pha_n_bands=14,  # 1 Hz resolution
        amp_start_hz=40.0,
        amp_end_hz=160.0,
        amp_n_bands=13,  # ~10 Hz resolution
        n_perm=n_permutations,
        return_dist=True,
        device=device,
        fp16=False  # Use fp32 for better precision in statistical analysis
    )
    
    # Calculate custom statistics from the surrogate distribution
    print("Performing custom statistical analysis...")
    stats = calculate_custom_statistics(pac_values, surrogate_dist)
    
    # Create visualizations
    print("Creating statistical visualizations...")
    
    # Significance maps with different metrics
    plot_significance_maps(stats, pha_freqs, amp_freqs, pac_config)
    
    # Distribution comparisons
    plot_distribution_comparisons(pac_values, surrogate_dist, stats, pha_freqs, amp_freqs, pac_config)
    
    # Effect size comparison
    plot_effect_size_comparison(stats, pha_freqs, amp_freqs, pac_config)
    
    # 3D visualization of surrogate distributions (if we have a reasonable number)
    try:
        if n_permutations <= 50:
            plot_surrogates_3d(surrogate_dist, pac_values, pha_freqs, amp_freqs, pac_config)
    except Exception as e:
        print(f"Could not create 3D visualization: {e}")
    
    # Clean up GPU memory if needed
    if device == "cuda":
        del signals, pac_values, surrogate_dist
        torch.cuda.empty_cache()
        gc.collect()
    
    print("Done! Figures saved in the 'figures' directory:")
    print("  - figures/05_pac_significance_maps.png")
    print("  - figures/05_surrogate_distributions_comparison.png")
    print("  - figures/05_effect_size_comparison.png")
    if n_permutations <= 50:
        print("  - figures/05_surrogate_distributions_3d.png")


if __name__ == "__main__":
    main()

# EOF