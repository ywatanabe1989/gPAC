#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-08 14:53:27 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/gpac/example__PAC.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/gpac/example__PAC.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Demonstrates PAC class functionality for phase-amplitude coupling analysis
  - Shows both static and trainable PAC computation
  - Visualizes PAC comodulograms and surrogate statistics
  - Tests gradient flow for trainable components
  - Compares performance and accuracy between modes

Dependencies:
  - scripts: None
  - packages: numpy, torch, matplotlib, gpac, mngs

IO:
  - input-files: None (generates synthetic PAC signals)
  - output-files:
    - 01_static_pac_analysis.gif
    - 02_trainable_pac_analysis.gif
    - 03_pac_comparison.gif
"""

"""Imports"""
import argparse

import matplotlib

matplotlib.use("Agg")
import gpac
import matplotlib.pyplot as plt
import numpy as np
import torch

"""Functions & Classes"""
def demo_static_pac(args):
    """Demonstrate static PAC analysis."""
    import mngs

    mngs.str.printc("=== Demo Static PAC ===", c=CC["orange"])

    # Generate synthetic PAC data
    pac_config = gpac.dataset.single_class_multi_pac_config
    batch = gpac.dataset.generate_pac_batch(
        batch_size=2,
        n_channels=8,
        n_segments=4,
        duration_sec=10,
        fs=512,
        pac_config=pac_config,
    )
    signal, label, metadata = batch

    # Create static PAC
    pac_static = gpac.PAC(
        seq_len=signal.shape[-1],
        fs=metadata["fs"][0],
        pha_start_hz=2.0,
        pha_end_hz=30.0,
        pha_n_bands=50,
        amp_start_hz=30.0,
        amp_end_hz=230.0,
        amp_n_bands=50,
        trainable=False,
        n_perm=50,
        fp16=False,
    )

    if torch.cuda.is_available():
        pac_static = pac_static.cuda()
        signal = signal.cuda()

    # Compute PAC
    pac_result = pac_static(signal)

    # Create visualization with ground truth
    fig, axes = mngs.plt.subplots(2, 2, figsize=(12, 10))

    sample_idx, channel_idx = 0, 0

    # Detected PAC comodulogram
    ax_pac = axes[0, 0]
    pac_data = pac_result["pac"][sample_idx, channel_idx].cpu().numpy()
    im1 = ax_pac.imshow(
        pac_data,
        aspect="auto",
        cmap="viridis",
    )
    ax_pac.set_xyt(
        "Amplitude Frequency [Hz]", "Phase Frequency [Hz]", "Detected PAC"
    )
    ax_pac.set_xticks(range(0, len(pac_static.amp_mids), 20))
    ax_pac.set_xticklabels([f"{f:.0f}" for f in pac_static.amp_mids[::20]])
    ax_pac.set_yticks(range(0, len(pac_static.pha_mids), 20))
    ax_pac.set_yticklabels([f"{f:.0f}" for f in pac_static.pha_mids[::20]])
    plt.colorbar(im1, ax=ax_pac)

    # Add ground truth markers
    if (
        "pac_components" in metadata
        and len(metadata["pac_components"][sample_idx]) > 0
    ):
        pha_freqs = pac_static.pha_mids.cpu().numpy()
        amp_freqs = pac_static.amp_mids.cpu().numpy()
        for coupling in metadata["pac_components"][sample_idx]:
            pha_hz = coupling["phase_hz"]
            amp_hz = coupling["amp_hz"]
            pha_idx = np.argmin(np.abs(pha_freqs - pha_hz))
            amp_idx = np.argmin(np.abs(amp_freqs - amp_hz))
            ax_pac.scatter(
                amp_idx,
                pha_idx,
                s=200,
                marker="x",
                c=CC["orange"],
                linewidth=3,
            )

    # Z-scores
    ax_z = axes[0, 1]
    if pac_result["pac_z"] is not None:
        z_data = pac_result["pac_z"][sample_idx, channel_idx].cpu().numpy()
        im2 = ax_z.imshow(
            z_data,
            aspect="auto",
            cmap="RdBu_r",
            # vmin=-3,
            # vmax=3,
        )
        ax_z.set_xyt(
            "Amplitude Frequency [Hz]", "Phase Frequency [Hz]", "Z-scores"
        )
        ax_z.set_xticks(range(0, len(pac_static.amp_mids), 20))
        ax_z.set_xticklabels([f"{f:.0f}" for f in pac_static.amp_mids[::20]])
        ax_z.set_yticks(range(0, len(pac_static.pha_mids), 20))
        ax_z.set_yticklabels([f"{f:.0f}" for f in pac_static.pha_mids[::20]])
        plt.colorbar(im2, ax=ax_z)

        # Add ground truth markers
        if (
            "pac_components" in metadata
            and len(metadata["pac_components"][sample_idx]) > 0
        ):
            for coupling in metadata["pac_components"][sample_idx]:
                pha_hz = coupling["phase_hz"]
                amp_hz = coupling["amp_hz"]
                pha_idx = np.argmin(np.abs(pha_freqs - pha_hz))
                amp_idx = np.argmin(np.abs(amp_freqs - amp_hz))
                ax_z.scatter(
                    amp_idx,
                    pha_idx,
                    s=200,
                    marker="x",
                    c=CC["orange"],
                    linewidth=3,
                )

    # Frequency profiles
    ax_pha = axes[1, 0]
    pac_mean_amp = pac_data.mean(axis=1)
    ax_pha.plot(
        pac_static.pha_mids.cpu(), pac_mean_amp, "o-", label="Detected"
    )
    if (
        "pac_components" in metadata
        and len(metadata["pac_components"][sample_idx]) > 0
    ):
        gt_pha_freqs = [
            c["phase_hz"] for c in metadata["pac_components"][sample_idx]
        ]
        for freq in gt_pha_freqs:
            ax_pha.axvline(
                x=freq, color=CC["orange"], linestyle="--", linewidth=1
            )

        # gt_strengths = [
        #     c.get("strength", 1.0)
        #     for c in metadata["pac_components"][sample_idx]
        # ]
        # ax_pha.scatter(
        #     gt_pha_freqs,
        #     gt_strengths,
        #     color=CC["orange"],
        #     s=50,
        #     marker="x",
        #     label="Ground Truth",
        # )
    ax_pha.set_xyt(
        "Phase Frequency [Hz]", "PAC Strength", "Phase Frequency Profile"
    )
    ax_pha.legend()
    ax_pha.grid(True, alpha=0.3)

    ax_amp = axes[1, 1]
    pac_mean_pha = pac_data.mean(axis=0)
    ax_amp.plot(
        pac_static.amp_mids.cpu(), pac_mean_pha, "s-", label="Detected"
    )
    if (
        "pac_components" in metadata
        and len(metadata["pac_components"][sample_idx]) > 0
    ):
        gt_amp_freqs = [
            c["amp_hz"] for c in metadata["pac_components"][sample_idx]
        ]
        for freq in gt_amp_freqs:
            ax_amp.axvline(
                x=freq, color=CC["orange"], linestyle="--", linewidth=1
            )

    ax_amp.set_xyt(
        "Amplitude Frequency [Hz]",
        "PAC Strength",
        "Amplitude Frequency Profile",
    )
    ax_amp.legend()
    ax_amp.grid(True, alpha=0.3)

    plt.tight_layout()
    mngs.io.save(fig, "01_static_pac_analysis.gif")
    plt.close()

    # Print statistics with ground truth comparison
    print(f"Input shape: {signal.shape}")
    print(f"PAC shape: {pac_result['pac'].shape}")
    print(f"Mean PAC: {pac_result['pac'].mean().item():.4f}")
    print(f"Max PAC: {pac_result['pac'].max().item():.4f}")
    if pac_result["pac_z"] is not None:
        significant_pairs = (torch.abs(pac_result["pac_z"]) > 2).sum().item()
        total_pairs = pac_result["pac_z"].numel()
        print(
            f"Significant pairs (|z| > 2): {significant_pairs}/{total_pairs}"
        )

    if (
        "pac_components" in metadata
        and len(metadata["pac_components"][sample_idx]) > 0
    ):
        print(f"Ground truth couplings for sample {sample_idx}:")
        for coupling in metadata["pac_components"][sample_idx]:
            print(
                f"  {coupling['phase_hz']:.1f} Hz -> {coupling['amp_hz']:.1f} Hz (strength: {coupling.get('strength', 1.0):.3f})"
            )


def demo_trainable_pac(args):
    """Demonstrate trainable PAC with 5-fold cross-validation classification."""
    import mngs
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    mngs.str.printc(
        "=== Demo Trainable PAC Classification ===", c=CC["orange"]
    )

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate multi-class dataset
    n_samples = 1024
    n_classes = 3
    signals = []
    labels = []

    # Define distinct class configurations
    class_configs = []
    for class_idx in range(n_classes):
        config = {
            f"class_{class_idx}": {
                "components": [
                    {
                        "phase_hz": 2.0 + class_idx * 4.0,
                        "amp_hz": 50.0 + class_idx * 25.0,
                        "strength": 0.4 + class_idx * 0.1,
                    }
                ],
                "noise_levels": [0.15],
            }
        }
        class_configs.append(config)

    for class_idx in range(n_classes):
        for sample_idx in range(n_samples // n_classes):
            batch = gpac.dataset.generate_pac_batch(
                batch_size=1,
                n_channels=2,
                n_segments=1,
                duration_sec=4,
                fs=512,
                pac_config=class_configs[class_idx],
            )
            signal, _, _ = batch
            signals.append(signal.squeeze())
            labels.append(class_idx)

    X = torch.stack(signals)
    y = torch.tensor(labels)

    if torch.cuda.is_available():
        X = X.cuda()

    # 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    learning_curves = []
    all_predictions = []
    all_true_labels = []
    class_comodulograms = []
    weights_data = None

    fig, axes = mngs.plt.subplots(2, 3, figsize=(18, 12))

    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(X.cpu(), y.cpu())
    ):
        print(f"Fold {fold_idx + 1}/5")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Reduced filters for 3-class problem
        pac_trainable = gpac.PAC(
            seq_len=X.shape[-1],
            fs=512,
            pha_start_hz=2.0,
            pha_end_hz=20.0,
            pha_n_bands=10,
            amp_start_hz=30.0,
            amp_end_hz=120.0,
            amp_n_bands=10,
            trainable=True,
            pha_n_pool_ratio=1.5,
            amp_n_pool_ratio=1.5,
            n_perm=None,
            fp16=False,
        )

        if torch.cuda.is_available():
            pac_trainable = pac_trainable.cuda()

        # Get feature dimension from a sample forward pass
        with torch.no_grad():
            sample_result = pac_trainable(X_train[:1])
            sample_tensor = sample_result["pac"]
            sample_features = torch.cat(
                [
                    torch.median(sample_tensor, dim=-1)[0]
                    .median(dim=-1)[0]
                    .flatten(1),
                    torch.quantile(sample_tensor, 0.75, dim=-1)
                    .median(dim=-1)[0]
                    .flatten(1),
                    torch.quantile(sample_tensor, 0.25, dim=-1)
                    .median(dim=-1)[0]
                    .flatten(1),
                ],
                dim=1,
            )
            feature_dim = sample_features.shape[1]

        classifier = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, n_classes),
        )

        if torch.cuda.is_available():
            classifier = classifier.cuda()

        optimizer = torch.optim.Adam(
            list(pac_trainable.parameters()) + list(classifier.parameters()),
            lr=1e-3,
        )

        fold_losses = []

        # Training loop
        pac_trainable.train()
        classifier.train()
        for epoch_idx in range(10):
            optimizer.zero_grad()
            __import__("ipdb").set_trace()
            pac_result = pac_trainable(X_train)
            pac_tensor = pac_result["pac"]

            pac_q75 = (
                torch.quantile(pac_tensor, 0.75, dim=-1)
                .median(dim=-1)[0]
                .flatten(1)
            )
            pac_median = (
                torch.median(pac_tensor, dim=-1)[0]
                .median(dim=-1)[0]
                .flatten(1)
            )
            pac_q25 = (
                torch.quantile(pac_tensor, 0.25, dim=-1)
                .median(dim=-1)[0]
                .flatten(1)
            )
            pac_features = torch.cat([pac_median, pac_q75, pac_q25], dim=1)

            noise = torch.randn_like(pac_features) * 0.01
            pac_features_noisy = pac_features + noise

            logits = classifier(pac_features_noisy)

            # Move y_train to same device as logits
            y_train_gpu = y_train.to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits, y_train_gpu)

            reg_loss = pac_trainable.get_filter_regularization_loss()
            total_loss = loss + 0.01 * reg_loss

            total_loss.backward(retain_graph=True)
            optimizer.step()

            fold_losses.append(total_loss.item())

        # Store class-specific comodulograms from first fold
        if fold_idx == 0:
            pac_trainable.eval()
            with torch.no_grad():
                for class_idx in range(n_classes):
                    class_mask = y_test == class_idx
                    if class_mask.sum() > 0:
                        class_samples = X_test[class_mask][:3]
                        class_pac = pac_trainable(class_samples)
                        # Average across samples and channels
                        avg_comod = (
                            class_pac["pac"].mean(dim=(0, 1)).cpu().numpy()
                        )
                        class_comodulograms.append(
                            {
                                "class": class_idx,
                                "comodulogram": avg_comod,
                                "expected_phase": 6.0 + class_idx * 4.0,
                                "expected_amp": 50.0 + class_idx * 25.0,
                            }
                        )

            # Store weights
            learned_info = pac_trainable.get_selected_frequencies()
            pha_selected, amp_selected = learned_info
            filter_info = pac_trainable.bandpass.info
            weights_data = {
                "pha_weights": filter_info["pha_weights"].cpu().numpy(),
                "amp_weights": filter_info["amp_weights"].cpu().numpy(),
                "pha_center_freqs": filter_info["pha_center_freqs"],
                "pha_selected": pha_selected,
                "amp_selected": amp_selected,
            }

        learning_curves.append(fold_losses)

        # Feature extraction and classification
        pac_trainable.eval()
        with torch.no_grad():
            train_pac = pac_trainable(X_train)
            test_pac = pac_trainable(X_test)
            train_features = train_pac["pac"].mean(dim=(1, 2)).cpu().numpy()
            test_features = test_pac["pac"].mean(dim=(1, 2)).cpu().numpy()

        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)

        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(train_features_scaled, y_train.cpu())

        train_pred = clf.predict(train_features_scaled)
        test_pred = clf.predict(test_features_scaled)

        train_acc = (train_pred == y_train.cpu().numpy()).mean()
        test_acc = (test_pred == y_test.cpu().numpy()).mean()

        fold_results.append({"train_acc": train_acc, "test_acc": test_acc})
        all_predictions.extend(test_pred)
        all_true_labels.extend(y_test.cpu().numpy())

    # Visualization
    # 1. Learning curves with symlog scale
    ax_learning = axes[0, 0]
    for fold_idx, losses in enumerate(learning_curves):
        ax_learning.plot(
            losses, alpha=0.7, label=f"Fold {fold_idx+1}", linewidth=1.0
        )
    # ax_learning.set_yscale("symlog", linthresh=1e-3)
    ax_learning.set_xyt("Epoch", "Loss", "Learning Curves")
    ax_learning.legend()
    ax_learning.grid(True, alpha=0.3)

    # 2. Cross-validation accuracy
    ax_cv = axes[0, 1]
    train_accs = [r["train_acc"] for r in fold_results]
    test_accs = [r["test_acc"] for r in fold_results]
    x_pos = range(len(train_accs))
    ax_cv.bar(
        [x - 0.2 for x in x_pos], train_accs, 0.4, label="Train", alpha=0.7
    )
    ax_cv.bar(
        [x + 0.2 for x in x_pos], test_accs, 0.4, label="Test", alpha=0.7
    )
    ax_cv.set_xyt(
        "Fold",
        "Accuracy",
        f"CV Results (Test: {np.mean(test_accs):.3f}±{np.std(test_accs):.3f})",
    )
    ax_cv.set_xticks(x_pos)
    ax_cv.set_xticklabels([f"F{ii+1}" for ii in x_pos])
    ax_cv.legend()
    ax_cv.grid(True, alpha=0.3)

    # 3. Confusion matrix
    ax_cm = axes[0, 2]
    cm = confusion_matrix(all_true_labels, all_predictions)
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xyt("Predicted", "True", "Confusion Matrix")
    ax_cm.set_xticks(range(n_classes))
    ax_cm.set_yticks(range(n_classes))
    for ii in range(n_classes):
        for jj in range(n_classes):
            ax_cm.text(jj, ii, str(cm[ii, jj]), ha="center", va="center")
    plt.colorbar(im, ax=ax_cm)

    # 4. Class-specific comodulograms (most informative)
    ax_pac = axes[1, 0]
    if class_comodulograms:
        # Show difference between class comodulograms
        comod_0 = class_comodulograms[0]["comodulogram"]
        comod_1 = (
            class_comodulograms[1]["comodulogram"]
            if len(class_comodulograms) > 1
            else comod_0
        )
        diff_comod = comod_1 - comod_0
        im_pac = ax_pac.imshow(diff_comod, cmap="RdBu_r", aspect="auto")
        ax_pac.set_xyt(
            "Amplitude Freq",
            "Phase Freq",
            f"Class 1 - Class 0\nPAC Difference",
        )
        plt.colorbar(im_pac, ax=ax_pac)

        # Mark expected locations
        if len(class_comodulograms) > 1:
            # Approximate frequency indices (simplified)
            pha_idx_0 = int(len(diff_comod) * 0.3)
            amp_idx_0 = int(len(diff_comod[0]) * 0.2)
            pha_idx_1 = int(len(diff_comod) * 0.5)
            amp_idx_1 = int(len(diff_comod[0]) * 0.5)
            ax_pac.scatter(
                [amp_idx_0, amp_idx_1],
                [pha_idx_0, pha_idx_1],
                s=100,
                marker="x",
                c="yellow",
                linewidth=2,
            )

    # 5. Feature distribution
    ax_feat = axes[1, 1]
    for class_idx in range(n_classes):
        class_mask = np.array(all_true_labels) == class_idx
        class_preds = np.array(all_predictions)[class_mask]
        ax_feat.hist(
            class_preds,
            alpha=0.7,
            label=f"True Class {class_idx}",
            bins=range(n_classes + 1),
        )
    ax_feat.set_xyt("Predicted Class", "Count", "Prediction Distribution")
    ax_feat.legend()

    # 6. Learned filter weights
    ax_weights = axes[1, 2]
    if weights_data is not None:
        pha_weights = weights_data["pha_weights"]
        amp_weights = weights_data["amp_weights"]
        n_pha = len(pha_weights)
        n_amp = len(amp_weights)

        ax_weights.bar(
            range(n_pha),
            pha_weights,
            alpha=0.7,
            label=f"Phase ({n_pha})",
            color="blue",
            width=0.8,
        )
        ax_weights.bar(
            range(n_pha, n_pha + n_amp),
            amp_weights,
            alpha=0.7,
            label=f"Amplitude ({n_amp})",
            color="red",
            width=0.8,
        )
        ax_weights.set_xyt(
            "Filter Index", "Selection Weight", "Learned Filter Weights"
        )
        ax_weights.legend()
        ax_weights.grid(True, alpha=0.3)

    plt.tight_layout()
    mngs.io.save(fig, "02_trainable_pac_classification.gif")

    # Print results
    report = classification_report(
        all_true_labels, all_predictions, output_dict=True
    )
    print(f"5-fold CV Results:")
    print(
        f"  Mean test accuracy: {np.mean(test_accs):.3f} ± {np.std(test_accs):.3f}"
    )
    print(f"Ground truth frequencies:")
    for class_idx in range(n_classes):
        expected_pha = 6.0 + class_idx * 4.0
        expected_amp = 50.0 + class_idx * 25.0
        print(f"  Class {class_idx}: {expected_pha}Hz -> {expected_amp}Hz")

    plt.close()


def demo_pac_comparison(args):
    """Compare static vs trainable PAC performance."""
    import mngs

    mngs.str.printc("=== Demo PAC Comparison ===", c=CC["orange"])

    # Generate test data with consistent PAC
    consistent_config = {
        "consistent_pac": {
            "components": [
                {"phase_hz": 8.0, "amp_hz": 80.0, "strength": 0.5},
                {"phase_hz": 12.0, "amp_hz": 120.0, "strength": 0.4},
            ],
            "noise_levels": [0.1],
        }
    }

    batch = gpac.dataset.generate_pac_batch(
        batch_size=2,
        n_channels=4,
        n_segments=2,
        duration_sec=10,
        fs=512,
        pac_config=consistent_config,
    )
    signal, label, metadata = batch

    if torch.cuda.is_available():
        signal = signal.cuda()

    # Static PAC
    static_pac = gpac.PAC(
        seq_len=signal.shape[-1],
        fs=metadata["fs"][0],
        pha_start_hz=2.0,
        pha_end_hz=30.0,
        pha_n_bands=50,
        amp_start_hz=30.0,
        amp_end_hz=230.0,
        amp_n_bands=50,
        trainable=False,
        n_perm=None,
        fp16=False,
    )

    # Trainable PAC
    trainable_pac = gpac.PAC(
        seq_len=signal.shape[-1],
        fs=metadata["fs"][0],
        pha_start_hz=2.0,
        pha_end_hz=30.0,
        pha_n_bands=50,
        amp_start_hz=30.0,
        amp_end_hz=230.0,
        amp_n_bands=50,
        trainable=True,
        pha_n_pool_ratio=1.5,
        amp_n_pool_ratio=1.5,
        n_perm=None,
        fp16=False,
    )

    if torch.cuda.is_available():
        static_pac = static_pac.cuda()
        trainable_pac = trainable_pac.cuda()

    # Compute PAC with timing
    import time

    # Static PAC timing
    start_time = time.time()
    static_result = static_pac(signal)
    static_time = time.time() - start_time

    # Trainable PAC timing
    start_time = time.time()
    trainable_result = trainable_pac(signal)
    trainable_time = time.time() - start_time

    # Create comparison visualization
    fig, axes = mngs.plt.subplots(2, 3, figsize=(15, 10))

    sample_idx, channel_idx = 0, 0

    # Static PAC
    ax_static = axes[0, 0]
    static_data = (
        static_result["pac"][sample_idx, channel_idx].detach().cpu().numpy()
    )
    im1 = ax_static.imshow(static_data, aspect="auto", cmap="viridis")
    ax_static.set_xyt("Amplitude [Hz]", "Phase [Hz]", "Static PAC")
    plt.colorbar(im1, ax=ax_static)

    # Add consistent ground truth markers to static PAC
    expected_couplings = [
        {"phase_hz": 8.0, "amp_hz": 80.0},
        {"phase_hz": 12.0, "amp_hz": 120.0},
    ]
    pha_freqs = static_pac.pha_mids.cpu().numpy()
    amp_freqs = static_pac.amp_mids.cpu().numpy()
    for coupling in expected_couplings:
        pha_hz = coupling["phase_hz"]
        amp_hz = coupling["amp_hz"]
        pha_idx = np.argmin(np.abs(pha_freqs - pha_hz))
        amp_idx = np.argmin(np.abs(amp_freqs - amp_hz))
        ax_static.scatter(
            amp_idx,
            pha_idx,
            s=200,
            marker="x",
            c=CC["orange"],
            linewidth=3,
        )

    # It is difficult for me to understand what this means
    # Trainable PAC
    # ax_trainable = axes[0, 1]
    # trainable_data = (
    #     trainable_result["pac"][sample_idx, channel_idx].cpu().numpy()
    # )
    # im2 = ax_trainable.imshow(trainable_data, aspect="auto", cmap="viridis")
    # ax_trainable.set_xyt("Amplitude [Hz]", "Phase [Hz]", "Trainable PAC")
    # plt.colorbar(im2, ax=ax_trainable)

    # Replace confusing trainable PAC section with clearer explanation
    ax_trainable = axes[0, 1]
    trainable_data = (
        trainable_result["pac"][sample_idx, channel_idx].detach().cpu().numpy()
    )
    im2 = ax_trainable.imshow(trainable_data, aspect="auto", cmap="viridis")
    ax_trainable.set_xyt(
        "Amplitude [Hz]",
        "Phase [Hz]",
        "Trainable PAC\n(Adaptive Frequency Selection)",
    )
    plt.colorbar(im2, ax=ax_trainable)

    # Show selected frequencies for trainable
    selected_pha, selected_amp = trainable_pac.get_selected_frequencies()
    print(f"Trainable PAC selected frequencies:")
    print(f"  Phase bands: {[f'{f:.1f}' for f in selected_pha[:5]]}...")
    print(f"  Amp bands: {[f'{f:.1f}' for f in selected_amp[:5]]}...")

    # Add consistent ground truth markers to trainable PAC
    pha_freqs_trainable = trainable_pac.pha_mids.cpu().numpy()
    amp_freqs_trainable = trainable_pac.amp_mids.cpu().numpy()
    for coupling in expected_couplings:
        pha_hz = coupling["phase_hz"]
        amp_hz = coupling["amp_hz"]
        pha_idx = np.argmin(np.abs(pha_freqs_trainable - pha_hz))
        amp_idx = np.argmin(np.abs(amp_freqs_trainable - amp_hz))
        ax_trainable.scatter(
            amp_idx,
            pha_idx,
            s=200,
            marker="x",
            c=CC["orange"],
            linewidth=3,
        )

    # Difference
    ax_diff = axes[0, 2]
    diff_data = trainable_data - static_data
    im3 = ax_diff.imshow(diff_data, aspect="auto", cmap="RdBu_r")
    ax_diff.set_xyt("Amplitude [Hz]", "Phase [Hz]", "Difference")
    plt.colorbar(im3, ax=ax_diff)

    # Statistics comparison
    ax_stats = axes[1, 0]
    methods = ["Static", "Trainable"]
    means = [static_data.mean(), trainable_data.mean()]
    maxs = [static_data.max(), trainable_data.max()]
    stds = [static_data.std(), trainable_data.std()]

    xx_pos = np.arange(len(methods))
    width = 0.25

    ax_stats.bar(xx_pos - width, means, width, label="Mean", alpha=0.7)
    ax_stats.bar(xx_pos, maxs, width, label="Max", alpha=0.7)
    ax_stats.bar(xx_pos + width, stds, width, label="Std", alpha=0.7)
    ax_stats.set_xyt("Method", "PAC Value", "Statistics Comparison")
    ax_stats.set_xticks(xx_pos)
    ax_stats.set_xticklabels(methods)
    ax_stats.legend()
    ax_stats.grid(True, alpha=0.3)

    # Timing comparison
    ax_time = axes[1, 1]
    times = [static_time * 1000, trainable_time * 1000]
    ax_time.bar(methods, times, alpha=0.7, color=[CC["blue"], CC["red"]])
    ax_time.set_xyt("Method", "Time [ms]", "Computation Time")
    ax_time.grid(True, alpha=0.3)

    # Memory info
    ax_memory = axes[1, 2]
    static_memory = static_pac.get_memory_info()
    trainable_memory = trainable_pac.get_memory_info()

    memory_data = [
        static_memory["device_details"][0]["allocated_gb"],
        trainable_memory["device_details"][0]["allocated_gb"],
    ]
    ax_memory.bar(
        methods, memory_data, alpha=0.7, color=[CC["green"], CC["orange"]]
    )
    ax_memory.set_xyt("Method", "Memory [GB]", "GPU Memory Usage")
    ax_memory.grid(True, alpha=0.3)

    plt.tight_layout()
    mngs.io.save(fig, "03_pac_comparison.gif")
    plt.close()

    # Print comparison results with ground truth info
    print(f"=== Comparison Results ===")
    print(f"Ground truth PAC: 8Hz->80Hz, 12Hz->120Hz")
    print(f"Static PAC:")
    print(f"  - Computation time: {static_time*1000:.2f} ms")
    print(f"  - Mean PAC: {static_data.mean():.4f}")
    print(f"  - Max PAC: {static_data.max():.4f}")
    print(f"Trainable PAC:")
    print(f"  - Computation time: {trainable_time*1000:.2f} ms")
    print(f"  - Mean PAC: {trainable_data.mean():.4f}")
    print(f"  - Max PAC: {trainable_data.max():.4f}")
    print(f"Performance ratio: {trainable_time/static_time:.2f}x")


def main(args):
    """Main function to demonstrate PAC functionality."""
    # demo_static_pac(args)
    demo_trainable_pac(args)
    # demo_pac_comparison(args)
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    parser = argparse.ArgumentParser(
        description="Demonstrate PAC functionality"
    )
    parser.add_argument(
        "--n_perm",
        type=int,
        default=50,
        help="Number of permutations for surrogate testing (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for processing (default: %(default)s)",
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

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__FILE__,
        verbose=False,
        agg=True,
    )

    exit_status = main(args)

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
