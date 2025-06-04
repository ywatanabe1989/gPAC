#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-30 10:00:00 (ywatanabe)"
# File: hand_grasping_demo.py

# ----------------------------------------
import os
__FILE__ = (
    "./examples/handgrasping/hand_grasping_demo.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Hand grasping classification using Phase-Amplitude Coupling (PAC)
  - Simulates EEG data based on WAY-EEG-GAL dataset structure
  - Shows motor cortex mu-gamma coupling for different grasp types
  - Demonstrates gPAC's utility for motor BCI applications

Dependencies:
  - scripts: None
  - packages: gpac, torch, numpy, matplotlib, sklearn, scipy

IO:
  - input-files: None
  - output-files: ./hand_grasping_demo_out/hand_grasping_pac_results.png
"""

"""Imports"""
import os
import sys
import argparse
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from umap import UMAP

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

"""Parameters"""
# from mngs.io import load_configs
# CONFIG = load_configs()

"""Functions & Classes"""


def generate_grasp_eeg(grasp_type, n_channels, n_times, sfreq):
    """Generate synthetic EEG with grasp-specific PAC patterns"""
    t = np.linspace(0, n_times / sfreq, n_times)
    eeg = np.zeros((n_channels, n_times))

    # Base neural activity
    for ch in range(n_channels):
        # Background noise
        eeg[ch] = 0.5 * np.random.randn(n_times)

        # Add mu rhythm (8-13 Hz) - stronger in motor channels
        mu_freq = 10 + np.random.randn()
        if ch in [0, 1, 2]:  # C3, Cz, C4
            mu_amp = 2.0 + 0.5 * np.random.randn()
        else:
            mu_amp = 1.0 + 0.3 * np.random.randn()
        mu_phase = 2 * np.pi * np.random.rand()
        eeg[ch] += mu_amp * np.sin(2 * np.pi * mu_freq * t + mu_phase)

        # Add beta rhythm (15-30 Hz)
        beta_freq = 20 + 3 * np.random.randn()
        beta_amp = 0.8 + 0.2 * np.random.randn()
        beta_phase = 2 * np.pi * np.random.rand()
        eeg[ch] += beta_amp * np.sin(2 * np.pi * beta_freq * t + beta_phase)

    # Add grasp-specific high gamma modulated by mu phase
    if grasp_type == 0:  # Lateral grasp - strong C3 activation
        pac_strength = 3.0
        primary_channel = 0  # C3
    elif grasp_type == 1:  # Palmar grasp - bilateral activation
        pac_strength = 2.5
        primary_channel = 1  # Cz
    elif grasp_type == 2:  # Tip grasp - precise, C4 dominant
        pac_strength = 3.5
        primary_channel = 2  # C4
    elif grasp_type == 3:  # Spherical grasp - distributed
        pac_strength = 2.0
        primary_channel = 1  # Cz
    else:  # Cylindrical grasp - strong bilateral
        pac_strength = 2.8
        primary_channel = 1  # Cz

    # Movement preparation phase (0.5-1s) - increased PAC
    prep_start = int(0.5 * sfreq)
    prep_end = int(1.0 * sfreq)

    # Movement execution phase (1-1.8s) - sustained PAC
    exec_start = int(1.0 * sfreq)
    exec_end = int(1.8 * sfreq)

    for phase_idx, (start, end) in enumerate(
        [(prep_start, prep_end), (exec_start, exec_end)]
    ):
        # Extract mu phase from primary channel
        from scipy.signal import hilbert, butter, filtfilt

        b, a = butter(4, [8, 13], btype="band", fs=sfreq)
        mu_filtered = filtfilt(b, a, eeg[primary_channel])
        mu_phase = np.angle(hilbert(mu_filtered))

        # Generate high gamma modulated by mu phase
        for ch in range(n_channels):
            # Channel-specific modulation
            if ch == primary_channel:
                ch_pac_strength = pac_strength
            elif abs(ch - primary_channel) == 1:  # Adjacent channels
                ch_pac_strength = pac_strength * 0.7
            else:
                ch_pac_strength = pac_strength * 0.3

            # Different gamma frequencies for different grasps
            gamma_center = 80 + grasp_type * 10
            gamma_freq = gamma_center + 10 * np.random.randn()

            # Phase-amplitude coupling
            gamma_amp = ch_pac_strength * (1 + 0.8 * np.cos(mu_phase[start:end]))
            gamma_signal = gamma_amp * np.sin(2 * np.pi * gamma_freq * t[start:end])

            # Add to EEG
            eeg[ch, start:end] += gamma_signal

    return eeg


def extract_pac_features(all_eeg, pac_model, device, batch_size=32):
    """Extract PAC features from EEG data"""
    features = []

    for i in range(0, len(all_eeg), batch_size):
        batch = all_eeg[i : i + batch_size]
        batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)

        with torch.no_grad():
            pac_output = pac_model(batch_tensor)
            # When n_perm is specified, output is a dict with 'pac' key
            if isinstance(pac_output, dict):
                pac_values = pac_output["pac"].cpu().numpy()
            else:
                pac_values = pac_output.cpu().numpy()

        # Extract motor-specific features
        for pac_matrix in pac_values:
            # Focus on motor channels (first 3 channels: C3, Cz, C4)
            motor_pac = pac_matrix[:3, :, :]

            feature_vector = [
                # Channel-specific max PAC
                motor_pac[0].max(),  # C3 max
                motor_pac[1].max(),  # Cz max
                motor_pac[2].max(),  # C4 max
                # Lateralization index
                (motor_pac[2].max() - motor_pac[0].max())
                / (motor_pac[2].max() + motor_pac[0].max() + 1e-6),
                # Mean PAC per channel
                motor_pac[0].mean(),
                motor_pac[1].mean(),
                motor_pac[2].mean(),
                # Preferred frequency bands
                np.unravel_index(motor_pac[0].argmax(), motor_pac[0].shape)[
                    0
                ],  # C3 phase band
                np.unravel_index(motor_pac[2].argmax(), motor_pac[2].shape)[
                    0
                ],  # C4 phase band
                # PAC spatial distribution
                motor_pac.std(),  # Spatial variability
                # Significant couplings
                (motor_pac > 2.5).sum(),  # Strong PAC count
                # Bilateral coordination
                np.corrcoef(motor_pac[0].flatten(), motor_pac[2].flatten())[0, 1],
            ]

            features.append(feature_vector)

    return np.array(features)


def create_visualizations(
    models,
    results,
    all_labels,
    y_pred,
    features_scaled,
    feature_names,
    feature_importance,
    grasp_names,
):
    """Create comprehensive visualization of results"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Cross-validation results
    ax = axes[0, 0]
    positions = np.arange(len(models))
    for i, (name, scores) in enumerate(results.items()):
        ax.bar(
            positions[i],
            scores.mean(),
            yerr=scores.std(),
            capsize=5,
            alpha=0.7,
            label=name,
        )
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification Performance (5-fold CV)")
    ax.set_xticks(positions)
    ax.set_xticklabels(models.keys(), rotation=45, ha="right")
    ax.axhline(y=0.2, color="r", linestyle="--", label="Chance (5 classes)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Confusion Matrix
    ax = axes[0, 1]
    cm = confusion_matrix(all_labels, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=grasp_names,
        yticklabels=grasp_names,
    )
    ax.set_title("Confusion Matrix - Hand Grasp Classification")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # 3. Feature Importance
    ax = axes[1, 0]
    importance_idx = np.argsort(feature_importance)[::-1]
    top_features = 8
    ax.barh(
        range(top_features), feature_importance[importance_idx[:top_features]][::-1]
    )
    ax.set_yticks(range(top_features))
    ax.set_yticklabels([feature_names[i] for i in importance_idx[:top_features]][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top PAC Features for Grasp Classification")
    ax.grid(True, alpha=0.3)

    # 4. UMAP/PCA visualization
    ax = axes[1, 1]
    if UMAP_AVAILABLE:
        umap_model = UMAP(n_components=2, random_state=42)
        features_2d = umap_model.fit_transform(features_scaled)
    else:
        # Fallback to PCA if UMAP not available
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(features_scaled)

    for i, grasp in enumerate(grasp_names):
        mask = all_labels == i
        ax.scatter(
            features_2d[mask, 0], features_2d[mask, 1], label=grasp, alpha=0.6, s=30
        )
    ax.set_xlabel("UMAP 1" if UMAP_AVAILABLE else "PC 1")
    ax.set_ylabel("UMAP 2" if UMAP_AVAILABLE else "PC 2")
    ax.set_title(
        "PAC Feature Space (UMAP)" if UMAP_AVAILABLE else "PAC Feature Space (PCA)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main(args):
    """Main function for hand grasping demo"""
    import mngs
    from gpac import PAC

    # Simulation parameters based on WAY-EEG-GAL dataset
    n_subjects = 12
    n_trials_per_grasp = 60  # Reduced for faster demo
    n_grasps = 5  # lateral, palmar, tip, spherical, cylindrical
    sfreq = 500  # Hz
    duration = 2.0  # seconds (reduced for faster processing)
    n_times = int(sfreq * duration)

    # Motor cortex channels
    motor_channels = ["C3", "Cz", "C4", "FC1", "FC2", "CP1", "CP2"]
    n_channels = len(motor_channels)
    grasp_names = ["Lateral", "Palmar", "Tip", "Spherical", "Cylindrical"]

    print("Hand Grasping Classification using gPAC")
    print("=" * 50)

    # Generate synthetic dataset
    print("\n1. Generating synthetic hand grasping EEG data...")
    all_eeg = []
    all_labels = []

    for subject in range(n_subjects):
        for grasp in range(n_grasps):
            for trial in range(n_trials_per_grasp // n_subjects):
                eeg = generate_grasp_eeg(grasp, n_channels, n_times, sfreq)
                all_eeg.append(eeg)
                all_labels.append(grasp)

    all_eeg = np.array(all_eeg)
    all_labels = np.array(all_labels)
    print(f"Generated data shape: {all_eeg.shape}")
    print(f"Grasp types: {grasp_names}")

    # Initialize gPAC model for motor-specific frequencies
    print("\n2. Initializing gPAC model...")
    print("   - Phase: Mu rhythm (8-13 Hz)")
    print("   - Amplitude: High gamma (60-150 Hz)")

    pac_model = PAC(
        seq_len=n_times,
        fs=sfreq,
        pha_start_hz=8.0,  # Mu rhythm start
        pha_end_hz=13.0,  # Mu rhythm end
        pha_n_bands=3,
        amp_start_hz=60.0,  # High gamma start
        amp_end_hz=150.0,  # High gamma end
        amp_n_bands=5,
        n_perm=50,  # Reduced permutations for faster demo
        fp16=False,
    )

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pac_model = pac_model.to(device)
    print(f"Using device: {device}")

    # Extract PAC features
    print("\n3. Extracting PAC features...")
    features = extract_pac_features(all_eeg, pac_model, device)
    print(f"Extracted features shape: {features.shape}")

    # Feature names for interpretation
    feature_names = [
        "C3_max_PAC",
        "Cz_max_PAC",
        "C4_max_PAC",
        "Lateralization_index",
        "C3_mean_PAC",
        "Cz_mean_PAC",
        "C4_mean_PAC",
        "C3_phase_band",
        "C4_phase_band",
        "Spatial_variability",
        "Strong_PAC_count",
        "Bilateral_coordination",
    ]

    # Classification with multiple models
    print("\n4. Training classifiers...")
    models = {
        "Dummy (Stratified)": DummyClassifier(strategy="stratified"),
        "Dummy (Uniform)": DummyClassifier(strategy="uniform"),
        "Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", gamma="scale", random_state=42),
    }

    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            loss_function="MultiClass",
            verbose=False,
            random_state=42,
        )

    # Cross-validation with pipelines to prevent data leakage
    from sklearn.pipeline import Pipeline
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # Create pipelines for each model
    pipelines = {}
    for name, model in models.items():
        pipelines[name] = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
    
    for name, pipeline in pipelines.items():
        scores = cross_val_score(
            pipeline, features, all_labels, cv=cv, scoring="accuracy"
        )
        results[name] = scores
        print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")

    # Train final model for analysis
    print("\n5. Training final Random Forest model...")
    # For final model and visualization, we can scale all data together
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(features_scaled, all_labels)

    # Feature importance
    feature_importance = rf_model.feature_importances_

    # Predictions for confusion matrix
    y_pred = rf_model.predict(features_scaled)

    # Create visualizations
    print("\n6. Creating visualizations...")
    fig = create_visualizations(
        models,
        results,
        all_labels,
        y_pred,
        features_scaled,
        feature_names,
        feature_importance,
        grasp_names,
    )

    # Save results
    sdir = mngs.io.mk_spath(__file__)
    spath = sdir / "hand_grasping_pac_results.png"
    mngs.io.save(fig, spath)
    print(f"Saved visualization to: {spath}")

    # Print detailed results
    print("\n7. Detailed Classification Report:")
    print("=" * 50)
    print(classification_report(all_labels, y_pred, target_names=grasp_names))

    # Grasp-specific PAC patterns
    print("\n8. Grasp-specific PAC patterns:")
    print("=" * 50)
    for i, grasp in enumerate(grasp_names):
        mask = all_labels == i
        grasp_features = features[mask]
        print(f"\n{grasp} Grasp:")
        print(
            f"  - C3 max PAC: {grasp_features[:, 0].mean():.2f} ± {grasp_features[:, 0].std():.2f}"
        )
        print(
            f"  - C4 max PAC: {grasp_features[:, 2].mean():.2f} ± {grasp_features[:, 2].std():.2f}"
        )
        print(
            f"  - Lateralization: {grasp_features[:, 3].mean():.3f} ± {grasp_features[:, 3].std():.3f}"
        )
        print(
            f"  - Bilateral coord: {grasp_features[:, 11].mean():.3f} ± {grasp_features[:, 11].std():.3f}"
        )

    print("\n9. Summary:")
    print("=" * 50)
    print("- Successfully classified 5 different hand grasp types using PAC features")
    print("- Z-scored PAC with permutation testing provides robust features")
    print("- Motor cortex mu-gamma coupling provides discriminative information")
    print("- Lateralization index and channel-specific PAC are key features")
    print("- Different grasps show distinct spatial PAC patterns")
    best_acc = max(results[name].mean() for name in models if "Dummy" not in name)
    print(f"- Best accuracy: {best_acc:.1%} (vs {1/5:.1%} chance)")
    if CATBOOST_AVAILABLE and "CatBoost" in results:
        print(
            f"- CatBoost accuracy: {results['CatBoost'].mean():.1%} ± {results['CatBoost'].std():.1%}"
        )
    print("\nThis demonstrates gPAC's utility for motor BCI applications!")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(
        description="Hand grasping classification using gPAC"
    )
    args = parser.parse_args()
    mngs.str.printc(args, c="yellow")
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


if __name__ == "__main__":
    run_main()

# EOF
