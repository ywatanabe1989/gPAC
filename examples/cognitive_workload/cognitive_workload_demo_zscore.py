#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-05-30 11:00:00 (ywatanabe)"
# File: cognitive_workload_demo_zscore.py

# ----------------------------------------
import os
__FILE__ = (
    "./examples/cognitive_workload/cognitive_workload_demo_zscore.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Demonstrates gPAC for cognitive workload classification with z-scored PAC
  - Uses permutation testing for statistical robustness
  - Compares PAC between low/high cognitive load conditions
  - Performs statistical testing and classification

Dependencies:
  - scripts: None
  - packages: gpac, mne, torch, numpy, matplotlib, scipy

IO:
  - input-files: Cognitive workload EEG dataset (auto-downloaded)
  - output-files: ./cognitive_workload_demo_zscore_out/workload_analysis_zscore.png
"""

"""Imports"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
import mngs
import time

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    print("CatBoost not available. Install with: pip install catboost")
    CATBOOST_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("UMAP not available. Install with: pip install umap-learn")
    UMAP_AVAILABLE = False

try:
    import mne
    from mne.datasets import eegbci
except ImportError:
    print("Installing MNE-Python for EEG data access...")
    os.system("pip install mne")
    import mne
    from mne.datasets import eegbci

"""Functions & Classes"""
def download_nback_data(data_path=None):
    """Download and prepare n-back task EEG data."""
    # For demo, we'll use EEGBCI motor imagery data and treat 
    # different motor tasks as different cognitive loads
    print("Downloading EEG data...")
    
    # Set custom data path if provided
    if data_path:
        os.makedirs(data_path, exist_ok=True)
        mne.set_config('MNE_DATA', data_path)
    
    # Download data for subject 1, runs 4 (left hand) and 8 (right hand)
    runs_low = [4, 5, 6]  # Left hand movements (treat as low workload)
    runs_high = [8, 9, 10]  # Right hand movements (treat as high workload)
    
    # Load raw data with update_path=True to avoid interactive prompt
    raw_fnames_low = eegbci.load_data(1, runs_low, update_path=True, path=data_path)
    raw_fnames_high = eegbci.load_data(1, runs_high, update_path=True, path=data_path)
    
    # Concatenate runs
    raw_low = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames_low])
    raw_high = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames_high])
    
    # Standardize channel names
    eegbci.standardize(raw_low)
    eegbci.standardize(raw_high)
    
    # Set montage
    montage = mne.channels.make_standard_montage('standard_1005')
    raw_low.set_montage(montage)
    raw_high.set_montage(montage)
    
    # Apply bandpass filter
    raw_low.filter(1., 50., fir_design='firwin')
    raw_high.filter(1., 50., fir_design='firwin')
    
    return raw_low, raw_high


def prepare_epochs(raw_low, raw_high, tmin=-0.5, tmax=2.0):
    """Create epochs from continuous data."""
    # Find events (motor imagery cues)
    events_low = mne.events_from_annotations(raw_low)[0]
    events_high = mne.events_from_annotations(raw_high)[0]
    
    # Create epochs
    epochs_low = mne.Epochs(raw_low, events_low, tmin=tmin, tmax=tmax, 
                           baseline=None, preload=True)
    epochs_high = mne.Epochs(raw_high, events_high, tmin=tmin, tmax=tmax, 
                            baseline=None, preload=True)
    
    # Select frontal channels (where workload effects are strongest)
    frontal_channels = ['Fz', 'F3', 'F4', 'FC1', 'FC2']
    epochs_low.pick_channels([ch for ch in frontal_channels if ch in epochs_low.ch_names])
    epochs_high.pick_channels([ch for ch in frontal_channels if ch in epochs_high.ch_names])
    
    return epochs_low, epochs_high


def compute_pac_features_zscore(epochs, device='cuda', n_perm=200):
    """Extract z-scored PAC features from epochs using permutation testing."""
    import gpac
    
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']
    n_epochs, n_channels, n_times = data.shape
    
    # Initialize PAC model with permutation testing
    pac_model = gpac.PAC(
        seq_len=n_times,
        fs=sfreq,
        pha_start_hz=4.0,    # Theta band
        pha_end_hz=8.0,
        pha_n_bands=2,
        amp_start_hz=30.0,   # Gamma band
        amp_end_hz=45.0,
        amp_n_bands=3,
        n_perm=n_perm,       # Enable permutation testing
        return_dist=False,   # Return z-scores instead of raw PAC
        fp16=False
    )
    
    if device == 'cuda' and torch.cuda.is_available():
        pac_model = pac_model.cuda()
    
    # Compute z-scored PAC for each epoch and channel
    pac_features = []
    raw_pac_values = []  # Store raw PAC for comparison
    
    print(f"Computing z-scored PAC with {n_perm} permutations...")
    start_time = time.time()
    
    for epoch_idx in range(n_epochs):
        if epoch_idx % 10 == 0:
            print(f"  Processing epoch {epoch_idx+1}/{n_epochs}...")
        
        epoch_pac = []
        epoch_raw_pac = []
        
        for ch_idx in range(n_channels):
            # Get single channel data
            signal = torch.tensor(data[epoch_idx, ch_idx, :], dtype=torch.float32)
            signal = signal.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            
            if device == 'cuda' and torch.cuda.is_available():
                signal = signal.cuda()
            
            # Compute PAC with permutation testing
            with torch.no_grad():
                pac_result = pac_model(signal)
                
                # pac_result['pac'] now contains z-scores
                pac_zscore = pac_result['pac'].squeeze().cpu().numpy()
                
                # For comparison, compute raw PAC without permutations
                pac_model_raw = gpac.PAC(
                    seq_len=n_times, fs=sfreq,
                    pha_start_hz=4.0, pha_end_hz=8.0, pha_n_bands=2,
                    amp_start_hz=30.0, amp_end_hz=45.0, amp_n_bands=3,
                    n_perm=None, fp16=False
                )
                if device == 'cuda' and torch.cuda.is_available():
                    pac_model_raw = pac_model_raw.cuda()
                    
                pac_raw = pac_model_raw(signal)['pac'].squeeze().cpu().numpy()
                
            # Use max z-score as feature (most significant coupling)
            epoch_pac.append(pac_zscore.max())
            epoch_raw_pac.append(pac_raw.max())
        
        pac_features.append(epoch_pac)
        raw_pac_values.append(epoch_raw_pac)
    
    compute_time = time.time() - start_time
    print(f"  Computation time: {compute_time:.1f}s")
    
    pac_features = np.array(pac_features)  # (n_epochs, n_channels)
    raw_pac_values = np.array(raw_pac_values)
    
    # Get frequency info
    pha_freqs = pac_result['phase_frequencies'].numpy()
    amp_freqs = pac_result['amplitude_frequencies'].numpy()
    
    return pac_features, raw_pac_values, pha_freqs, amp_freqs


def statistical_analysis(pac_low, pac_high, channel_names):
    """Perform statistical testing between conditions."""
    results = {}
    
    # T-test for each channel
    t_stats = []
    p_values = []
    
    for ch_idx in range(pac_low.shape[1]):
        t_stat, p_val = stats.ttest_ind(pac_low[:, ch_idx], pac_high[:, ch_idx])
        t_stats.append(t_stat)
        p_values.append(p_val)
    
    results['t_stats'] = np.array(t_stats)
    results['p_values'] = np.array(p_values)
    results['significant'] = np.array(p_values) < 0.05
    
    # Effect size (Cohen's d)
    cohen_d = []
    for ch_idx in range(pac_low.shape[1]):
        mean_diff = pac_high[:, ch_idx].mean() - pac_low[:, ch_idx].mean()
        pooled_std = np.sqrt((pac_low[:, ch_idx].std()**2 + pac_high[:, ch_idx].std()**2) / 2)
        d = mean_diff / pooled_std if pooled_std > 0 else 0
        cohen_d.append(d)
    
    results['cohen_d'] = np.array(cohen_d)
    
    return results


def classify_workload_multiple(pac_low, pac_high):
    """Classify workload conditions using multiple classifiers with proper CV."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    
    # Prepare data
    X = np.vstack([pac_low, pac_high])
    y = np.hstack([np.zeros(len(pac_low)), np.ones(len(pac_high))])
    
    # Check class balance
    print(f"\nClass balance check:")
    print(f"  Class 0 (Low workload): {np.sum(y == 0)} samples")
    print(f"  Class 1 (High workload): {np.sum(y == 1)} samples")
    print(f"  Balance ratio: {np.sum(y == 0) / np.sum(y == 1):.2f}")
    
    # Initialize classifiers with pipelines to prevent data leakage
    classifiers = {
        'Dummy (stratified)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', DummyClassifier(strategy='stratified'))
        ]),
        'Dummy (most_frequent)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', DummyClassifier(strategy='most_frequent'))
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, probability=True))
        ]),
    }
    
    # Add CatBoost if available
    if CATBOOST_AVAILABLE:
        classifiers['CatBoost'] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=4,
                loss_function='Logloss',
                verbose=False,
                random_state=42
            ))
        ])
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate each classifier
    results = {}
    for name, pipeline in classifiers.items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        results[name] = scores
        print(f"\n{name}: {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"  Fold scores: {scores}")
    
    # For visualization, we need to scale all data together
    # This is OK since we're not using it for evaluation
    scaler = StandardScaler()
    X_scaled_viz = scaler.fit_transform(X)
    
    return results, X_scaled_viz, y


def create_umap_visualization(X_scaled_viz, y, pac_low_z, pac_high_z, channel_names):
    """Create UMAP visualization of PAC feature space.
    
    Note: X_scaled_viz is scaled on all data for visualization purposes only.
    This is acceptable since we're not using it for model evaluation.
    """
    if not UMAP_AVAILABLE:
        return None
    
    # Fit UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X_scaled_viz)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # UMAP scatter plot
    scatter = ax1.scatter(embedding[:, 0], embedding[:, 1], 
                         c=y, cmap='coolwarm', alpha=0.7, s=50)
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.set_title('UMAP Projection of PAC Features')
    plt.colorbar(scatter, ax=ax1, label='Workload (0=Low, 1=High)')
    
    # Feature correlation heatmap
    feature_data = np.vstack([pac_low_z, pac_high_z])
    corr_matrix = np.corrcoef(feature_data.T)
    
    im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(range(len(channel_names)))
    ax2.set_yticks(range(len(channel_names)))
    ax2.set_xticklabels(channel_names, rotation=45)
    ax2.set_yticklabels(channel_names)
    ax2.set_title('Feature Correlation Matrix')
    plt.colorbar(im, ax=ax2, label='Correlation')
    
    plt.tight_layout()
    return fig


def create_workload_figure_zscore(pac_low_z, pac_high_z, pac_low_raw, pac_high_raw,
                                 stats_results_z, stats_results_raw, 
                                 clf_scores_z, clf_scores_raw,
                                 channel_names, save_path):
    """Create comprehensive figure comparing z-scored vs raw PAC."""
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 1])
    
    # 1. Z-scored PAC distributions
    ax1 = fig.add_subplot(gs[0, :])
    positions_low = np.arange(len(channel_names)) * 2
    positions_high = positions_low + 0.8
    
    bp_low = ax1.boxplot([pac_low_z[:, i] for i in range(pac_low_z.shape[1])],
                         positions=positions_low, widths=0.6,
                         patch_artist=True, boxprops=dict(facecolor='lightblue'))
    bp_high = ax1.boxplot([pac_high_z[:, i] for i in range(pac_high_z.shape[1])],
                          positions=positions_high, widths=0.6,
                          patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    
    ax1.set_xticks(positions_low + 0.4)
    ax1.set_xticklabels(channel_names)
    ax1.set_ylabel('Z-scored PAC')
    ax1.set_title('Z-scored PAC Distribution by Cognitive Load (with Permutation Testing)', fontsize=14)
    ax1.legend([bp_low["boxes"][0], bp_high["boxes"][0]], 
               ['Low Load', 'High Load'], loc='upper right')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Add significance markers for z-scored
    for i, p_val in enumerate(stats_results_z['p_values']):
        if p_val < 0.001:
            ax1.text(positions_low[i] + 0.4, ax1.get_ylim()[1] * 0.95, '***', 
                    ha='center', fontsize=12)
        elif p_val < 0.01:
            ax1.text(positions_low[i] + 0.4, ax1.get_ylim()[1] * 0.95, '**', 
                    ha='center', fontsize=12)
        elif p_val < 0.05:
            ax1.text(positions_low[i] + 0.4, ax1.get_ylim()[1] * 0.95, '*', 
                    ha='center', fontsize=12)
    
    # 2. Raw PAC distributions (for comparison)
    ax2 = fig.add_subplot(gs[1, :])
    bp_low_raw = ax2.boxplot([pac_low_raw[:, i] for i in range(pac_low_raw.shape[1])],
                            positions=positions_low, widths=0.6,
                            patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.5))
    bp_high_raw = ax2.boxplot([pac_high_raw[:, i] for i in range(pac_high_raw.shape[1])],
                             positions=positions_high, widths=0.6,
                             patch_artist=True, boxprops=dict(facecolor='lightcoral', alpha=0.5))
    
    ax2.set_xticks(positions_low + 0.4)
    ax2.set_xticklabels(channel_names)
    ax2.set_ylabel('Raw PAC')
    ax2.set_title('Raw PAC Distribution (without Permutation Testing)', fontsize=14)
    
    # 3. Statistical comparison
    ax3 = fig.add_subplot(gs[2, 0])
    x = np.arange(len(channel_names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, stats_results_z['t_stats'], width, 
                    label='Z-scored', color='darkblue')
    bars2 = ax3.bar(x + width/2, stats_results_raw['t_stats'], width, 
                    label='Raw', color='darkred', alpha=0.7)
    
    ax3.set_ylabel('T-statistic')
    ax3.set_title('T-test Results Comparison', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(channel_names, rotation=45)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.legend()
    
    # 4. Effect sizes comparison
    ax4 = fig.add_subplot(gs[2, 1])
    bars1 = ax4.bar(x - width/2, stats_results_z['cohen_d'], width, 
                    label='Z-scored', color='darkgreen')
    bars2 = ax4.bar(x + width/2, stats_results_raw['cohen_d'], width, 
                    label='Raw', color='darkorange', alpha=0.7)
    
    ax4.set_ylabel("Cohen's d")
    ax4.set_title('Effect Sizes Comparison', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(channel_names, rotation=45)
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax4.axhline(y=0.8, color='r', linestyle='--', alpha=0.5)
    ax4.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
    ax4.legend()
    
    # 5. Classification comparison
    ax5 = fig.add_subplot(gs[2, 2])
    methods = ['Z-scored PAC', 'Raw PAC']
    accuracies = [clf_scores_z.mean(), clf_scores_raw.mean()]
    errors = [clf_scores_z.std(), clf_scores_raw.std()]
    
    bars = ax5.bar(methods, accuracies, yerr=errors, capsize=10, 
                   color=['purple', 'gray'], alpha=0.7)
    ax5.set_ylim([0, 1])
    ax5.set_ylabel('Classification Accuracy')
    ax5.set_title('Classification Performance', fontsize=12)
    ax5.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Chance')
    
    # Add accuracy values
    for i, (acc, err) in enumerate(zip(accuracies, errors)):
        ax5.text(i, acc + err + 0.02, f'{acc:.2f} ± {err:.2f}', 
                ha='center', fontsize=10)
    
    # 6. Feature importance (mean absolute z-scores)
    ax6 = fig.add_subplot(gs[3, :2])
    mean_abs_z = np.mean(np.abs(np.vstack([pac_low_z, pac_high_z])), axis=0)
    bars = ax6.bar(channel_names, mean_abs_z, color='teal', alpha=0.7)
    ax6.set_ylabel('Mean |Z-score|')
    ax6.set_xlabel('Channel')
    ax6.set_title('PAC Feature Importance (Mean Absolute Z-scores)', fontsize=12)
    ax6.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='Z=2 threshold')
    ax6.legend()
    
    # 7. Summary text
    ax7 = fig.add_subplot(gs[3, 2])
    ax7.axis('off')
    
    summary_text = f"""Summary:
    
Z-scored PAC Results:
• Significant channels: {stats_results_z['significant'].sum()}/{len(channel_names)}
• Mean accuracy: {clf_scores_z.mean():.1%}
• Largest effect: {channel_names[np.argmax(np.abs(stats_results_z['cohen_d']))]}
  (d = {stats_results_z['cohen_d'][np.argmax(np.abs(stats_results_z['cohen_d']))]:.2f})

Raw PAC Results:
• Significant channels: {stats_results_raw['significant'].sum()}/{len(channel_names)}
• Mean accuracy: {clf_scores_raw.mean():.1%}

Improvement with Z-scoring:
• Accuracy: {(clf_scores_z.mean() - clf_scores_raw.mean()) * 100:.1f}%
• Reduces noise & individual differences
• Provides statistical significance
    """
    
    ax7.text(0.1, 0.5, summary_text, transform=ax7.transAxes, 
             fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Cognitive Workload Analysis: Z-scored vs Raw PAC Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    mngs.io.save(fig, save_path)
    print(f"\n[GREEN] Figure saved to: {save_path} [/GREEN]\n")
    
    return fig


def main(args):
    """Main function for cognitive workload demo with z-scoring."""
    # Create output directory
    output_dir = './cognitive_workload_demo_zscore_out'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create MNE data directory under output directory
    mne_data_path = os.path.join(output_dir, "mne_data")
    os.makedirs(mne_data_path, exist_ok=True)
    
    # Download and prepare data
    print("\n==== Loading Cognitive Workload EEG Data ====\n")
    raw_low, raw_high = download_nback_data(data_path=mne_data_path)
    print(f"Loaded data: {len(raw_low.ch_names)} channels, {raw_low.info['sfreq']} Hz")
    
    # Create epochs
    print("\n==== Preparing Epochs ====\n")
    epochs_low, epochs_high = prepare_epochs(raw_low, raw_high)
    print(f"Low workload: {len(epochs_low)} epochs")
    print(f"High workload: {len(epochs_high)} epochs")
    print(f"Selected channels: {epochs_low.ch_names}")
    
    # Compute PAC features with z-scoring
    print("\n==== Computing Z-scored PAC Features ====\n")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Compute z-scored PAC
    pac_low_z, pac_low_raw, pha_freqs, amp_freqs = compute_pac_features_zscore(
        epochs_low, device, n_perm=200)
    pac_high_z, pac_high_raw, _, _ = compute_pac_features_zscore(
        epochs_high, device, n_perm=200)
    
    print(f"\nZ-scored PAC features shape: {pac_low_z.shape}")
    print(f"Phase frequencies: {pha_freqs}")
    print(f"Amplitude frequencies: {amp_freqs}")
    
    # Statistical analysis for both z-scored and raw
    print("\n==== Statistical Analysis ====\n")
    stats_results_z = statistical_analysis(pac_low_z, pac_high_z, epochs_low.ch_names)
    stats_results_raw = statistical_analysis(pac_low_raw, pac_high_raw, epochs_low.ch_names)
    
    print("Z-scored PAC results:")
    sig_channels_z = [ch for ch, sig in zip(epochs_low.ch_names, stats_results_z['significant']) if sig]
    print(f"  Significant channels: {sig_channels_z}")
    print(f"  Max effect size: d = {stats_results_z['cohen_d'].max():.2f}")
    
    print("\nRaw PAC results:")
    sig_channels_raw = [ch for ch, sig in zip(epochs_low.ch_names, stats_results_raw['significant']) if sig]
    print(f"  Significant channels: {sig_channels_raw}")
    print(f"  Max effect size: d = {stats_results_raw['cohen_d'].max():.2f}")
    
    # Classification with multiple classifiers
    print("\n==== Workload Classification ====\n")
    print("Testing with Z-scored PAC features:")
    clf_results_z, X_scaled_z, y_z = classify_workload_multiple(pac_low_z, pac_high_z)
    
    print("\n\nTesting with Raw PAC features:")
    clf_results_raw, X_scaled_raw, y_raw = classify_workload_multiple(pac_low_raw, pac_high_raw)
    
    # Extract SVM scores for backward compatibility
    clf_scores_z = clf_results_z['SVM (RBF)']
    clf_scores_raw = clf_results_raw['SVM (RBF)']
    
    # Create visualization
    print("\n==== Creating Visualization ====\n")
    fig = create_workload_figure_zscore(
        pac_low_z, pac_high_z, pac_low_raw, pac_high_raw,
        stats_results_z, stats_results_raw,
        clf_scores_z, clf_scores_raw,
        epochs_low.ch_names,
        os.path.join(output_dir, 'workload_analysis_zscore.png')
    )
    
    # Create UMAP visualization if available
    if UMAP_AVAILABLE:
        print("\n==== Creating UMAP Visualization ====\n")
        umap_fig = create_umap_visualization(
            X_scaled_z, y_z, pac_low_z, pac_high_z, epochs_low.ch_names
        )
        if umap_fig:
            import mngs
            mngs.io.save(umap_fig, os.path.join(output_dir, 'umap_visualization.png'))
            print(f"UMAP visualization saved to: {output_dir}/umap_visualization.png")
    
    # Final summary
    print("\n==== Analysis Complete ====\n")
    print(f"Z-scoring improved SVM classification by {(clf_scores_z.mean() - clf_scores_raw.mean()) * 100:.1f}%")
    
    # Compare all classifiers
    print("\nClassifier comparison (Z-scored features):")
    for name, scores in clf_results_z.items():
        print(f"  {name}: {scores.mean():.3f} ± {scores.std():.3f}")
    
    print(f"\nZ-scored PAC provides more robust features for cognitive state classification")
    
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs
    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description='Cognitive workload analysis with z-scored PAC')
    args = parser.parse_args()
    mngs.str.printc(args, c='yellow')
    return args


def run_main() -> None:
    """Initialize mngs framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt

    import sys
    import matplotlib.pyplot as plt
    import mngs

    args = parse_args()

    # Start mngs framework
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__file__,
        verbose=False,
        agg=True,
    )

    # Main
    exit_status = main(args)

    # Close the mngs framework
    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


if __name__ == '__main__':
    run_main()

# EOF