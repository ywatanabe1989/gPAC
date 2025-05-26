#!/usr/bin/env python3
"""
Test different padding strategies to match TensorPAC's filtfilt behavior.
"""

import numpy as np
import torch
from scipy.signal import filtfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def test_padding_strategies():
    """Test various padding approaches for conv1d to match filtfilt."""
    
    print("="*60)
    print("TESTING PADDING STRATEGIES")
    print("="*60)
    
    # Create test signal
    n_samples = 1000
    t = np.linspace(0, 1, n_samples)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
    
    # Create a simple lowpass filter
    filter_len = 101
    cutoff = 0.2  # Normalized frequency
    n = np.arange(filter_len)
    h = np.sinc(2 * cutoff * (n - (filter_len - 1) / 2))
    h *= np.hamming(filter_len)
    h /= np.sum(h)
    
    # Reference: scipy filtfilt
    ref_output = filtfilt(h, 1.0, signal, padlen=3*filter_len)
    
    # Convert to torch
    signal_t = torch.from_numpy(signal.astype(np.float32))
    kernel_t = torch.from_numpy(h.astype(np.float32))
    
    results = {}
    
    # Strategy 1: Simple conv1d with 'same' padding
    print("\n1. Simple conv1d with 'same' padding:")
    sig_1 = signal_t.unsqueeze(0).unsqueeze(0)
    ker_1 = kernel_t.unsqueeze(0).unsqueeze(0)
    out_1 = torch.nn.functional.conv1d(sig_1, ker_1, padding='same')
    out_1 = torch.nn.functional.conv1d(out_1.flip(-1), ker_1, padding='same').flip(-1)
    out_1 = out_1.squeeze().numpy()
    corr_1 = np.corrcoef(ref_output, out_1)[0, 1]
    print(f"   Correlation with filtfilt: {corr_1:.6f}")
    results['same'] = (out_1, corr_1)
    
    # Strategy 2: Manual padding with reflection
    print("\n2. Manual reflection padding:")
    padlen = 3 * filter_len
    sig_2d = signal_t.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    sig_padded = torch.nn.functional.pad(sig_2d, (padlen, padlen), mode='reflect')
    sig_2 = sig_padded
    out_2 = torch.nn.functional.conv1d(sig_2, ker_1, padding='same')
    out_2 = torch.nn.functional.conv1d(out_2.flip(-1), ker_1, padding='same').flip(-1)
    out_2 = out_2.squeeze()[padlen:-padlen].numpy()
    corr_2 = np.corrcoef(ref_output, out_2)[0, 1]
    print(f"   Correlation with filtfilt: {corr_2:.6f}")
    results['reflect'] = (out_2, corr_2)
    
    # Strategy 3: Replicate padding (edge values)
    print("\n3. Replicate padding:")
    sig_3d = signal_t.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    sig_padded = torch.nn.functional.pad(sig_3d, (padlen, padlen), mode='replicate')
    sig_3 = sig_padded
    out_3 = torch.nn.functional.conv1d(sig_3, ker_1, padding='same')
    out_3 = torch.nn.functional.conv1d(out_3.flip(-1), ker_1, padding='same').flip(-1)
    out_3 = out_3.squeeze()[padlen:-padlen].numpy()
    corr_3 = np.corrcoef(ref_output, out_3)[0, 1]
    print(f"   Correlation with filtfilt: {corr_3:.6f}")
    results['replicate'] = (out_3, corr_3)
    
    # Strategy 4: Scipy's default (odd extension)
    print("\n4. Custom odd extension (mimicking scipy):")
    # Implement scipy's odd extension manually
    sig_np = signal_t.numpy()
    # Create odd extension: [-flipped_signal, signal, -flipped_signal]
    left_ext = -sig_np[1:padlen+1][::-1]
    right_ext = -sig_np[-padlen-1:-1][::-1]
    sig_extended = np.concatenate([left_ext, sig_np, right_ext])
    sig_4 = torch.from_numpy(sig_extended.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    out_4 = torch.nn.functional.conv1d(sig_4, ker_1, padding='same')
    out_4 = torch.nn.functional.conv1d(out_4.flip(-1), ker_1, padding='same').flip(-1)
    out_4 = out_4.squeeze()[padlen:-padlen].numpy()
    corr_4 = np.corrcoef(ref_output, out_4)[0, 1]
    print(f"   Correlation with filtfilt: {corr_4:.6f}")
    results['odd_extension'] = (out_4, corr_4)
    
    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1][1])
    print(f"\n✅ Best strategy: {best_strategy[0]} with correlation {best_strategy[1][1]:.6f}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot signals
    ax = axes[0]
    ax.plot(ref_output[:200], 'k-', label='scipy filtfilt', linewidth=2)
    for name, (output, corr) in results.items():
        ax.plot(output[:200], '--', label=f'{name} (r={corr:.3f})', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.set_title('Filtered Signals Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot differences
    ax = axes[1]
    for name, (output, corr) in results.items():
        diff = output - ref_output
        ax.plot(diff[:200], label=f'{name} diff', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Difference from filtfilt')
    ax.set_title('Differences from scipy filtfilt')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./scripts/vs_tensorpac/results/figures/padding_strategies.png', dpi=150)
    plt.close()
    
    print("\n✅ Plot saved to: ./scripts/vs_tensorpac/results/figures/padding_strategies.png")
    
    return best_strategy[0], best_strategy[1][1]


def create_scipy_compatible_conv1d():
    """Create a conv1d implementation that matches scipy's filtfilt."""
    
    print("\n" + "="*60)
    print("CREATING SCIPY-COMPATIBLE CONV1D")
    print("="*60)
    
    def scipy_compatible_filtfilt(signal, kernel, padlen=None):
        """
        PyTorch implementation that matches scipy's filtfilt behavior.
        """
        if padlen is None:
            padlen = 3 * len(kernel)
        
        # Convert to tensors if needed
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal.astype(np.float32))
        if isinstance(kernel, np.ndarray):
            kernel = torch.from_numpy(kernel.astype(np.float32))
        
        # Get signal as numpy for odd extension
        sig_np = signal.numpy() if isinstance(signal, torch.Tensor) else signal
        
        # Create odd extension (scipy's default)
        padlen = min(padlen, len(sig_np) - 1)
        left_ext = -sig_np[1:padlen+1][::-1]
        right_ext = -sig_np[-padlen-1:-1][::-1]
        sig_extended = np.concatenate([left_ext, sig_np, right_ext])
        
        # Convert back to tensor
        sig_tensor = torch.from_numpy(sig_extended.astype(np.float32))
        sig_tensor = sig_tensor.unsqueeze(0).unsqueeze(0)
        kernel_tensor = kernel.unsqueeze(0).unsqueeze(0)
        
        # Forward pass
        filtered = torch.nn.functional.conv1d(sig_tensor, kernel_tensor, padding='same')
        # Backward pass
        filtered = torch.nn.functional.conv1d(
            filtered.flip(-1), kernel_tensor, padding='same'
        ).flip(-1)
        
        # Remove padding
        filtered = filtered.squeeze()[padlen:-padlen]
        
        return filtered
    
    # Test the implementation
    print("\nTesting scipy-compatible implementation...")
    
    # Test signal
    n_samples = 500
    signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, n_samples))
    
    # Test filter
    filter_len = 51
    h = np.hamming(filter_len)
    h /= np.sum(h)
    
    # Compare
    ref = filtfilt(h, 1.0, signal)
    test = scipy_compatible_filtfilt(signal, h, padlen=3*filter_len).numpy()
    
    corr = np.corrcoef(ref, test)[0, 1]
    rms_diff = np.sqrt(np.mean((ref - test)**2))
    
    print(f"Correlation: {corr:.6f}")
    print(f"RMS difference: {rms_diff:.9f}")
    
    if corr > 0.9999:
        print("✅ Successfully created scipy-compatible conv1d!")
    else:
        print("⚠️ Implementation needs refinement")
    
    return scipy_compatible_filtfilt


def main():
    """Test padding strategies and create compatible implementation."""
    
    # Test different strategies
    best_strategy, best_corr = test_padding_strategies()
    
    # Create scipy-compatible implementation
    scipy_compatible_filtfilt = create_scipy_compatible_conv1d()
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print(f"1. Best padding strategy: {best_strategy}")
    print(f"2. Achieved correlation: {best_corr:.6f}")
    print("3. For exact scipy compatibility, use odd extension padding")
    print("4. Consider updating BandPassFilter to use the scipy-compatible method")


if __name__ == "__main__":
    main()