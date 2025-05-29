#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-28 22:58:37 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/examples/gpac/example_visualization.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/gpac/example_visualization.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Set non-interactive backend
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def create_parallelization_diagram():
    """Create a diagram showing why GPU parallelization speeds up PAC calculation."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # CPU Sequential Processing
    ax1.set_title("CPU: Sequential Processing", fontsize=16, fontweight="bold")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)

    # Timeline
    ax1.plot([1, 9], [11, 11], "k-", linewidth=2)
    ax1.text(5, 11.5, "Time →", ha="center", fontsize=12, fontweight="bold")

    # Sequential band calculations
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
    band_names = ["Band 1", "Band 2", "Band 3", "Band 4", "Band 5", "Band 6"]

    y_pos = 9.5
    x_start = 1.5
    band_width = 1.2

    for i, (color, name) in enumerate(zip(colors, band_names)):
        x = x_start + i * band_width
        rect = patches.Rectangle(
            (x, y_pos - 0.4),
            band_width - 0.1,
            0.8,
            facecolor=color,
            edgecolor="black",
            linewidth=1,
        )
        ax1.add_patch(rect)
        ax1.text(
            x + (band_width - 0.1) / 2,
            y_pos,
            name,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Total time arrow
    ax1.annotate(
        "",
        xy=(x_start + len(colors) * band_width - 0.1, y_pos - 1.5),
        xytext=(x_start, y_pos - 1.5),
        arrowprops=dict(arrowstyle="<->", color="red", lw=2),
    )
    ax1.text(
        5,
        y_pos - 2,
        "Total Time = Sum of all bands",
        ha="center",
        fontsize=12,
        color="red",
        fontweight="bold",
    )

    # Bottleneck explanation
    ax1.text(
        5,
        6,
        "Each band waits for\nprevious band to complete",
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
    )

    ax1.text(
        5,
        3.5,
        "1 CPU Core\n↓\nSequential Processing\n↓\nSlow",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)

    # GPU Parallel Processing
    ax2.set_title("GPU: Parallel Processing", fontsize=16, fontweight="bold")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)

    # Timeline
    ax2.plot([1, 9], [11, 11], "k-", linewidth=2)
    ax2.text(5, 11.5, "Time →", ha="center", fontsize=12, fontweight="bold")

    # Parallel band calculations - all at same time
    y_positions = [9.5, 8.5, 7.5, 6.5, 5.5, 4.5]
    x_pos = 3
    band_width = 3

    for i, (color, name, y) in enumerate(zip(colors, band_names, y_positions)):
        rect = patches.Rectangle(
            (x_pos, y - 0.3),
            band_width,
            0.6,
            facecolor=color,
            edgecolor="black",
            linewidth=1,
        )
        ax2.add_patch(rect)
        ax2.text(
            x_pos + band_width / 2,
            y,
            name,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Short time arrow
    ax2.annotate(
        "",
        xy=(x_pos + band_width, 3.5),
        xytext=(x_pos, 3.5),
        arrowprops=dict(arrowstyle="<->", color="green", lw=2),
    )
    ax2.text(
        5,
        3,
        "Total Time = Single band time",
        ha="center",
        fontsize=12,
        color="green",
        fontweight="bold",
    )

    # Parallel explanation
    ax2.text(
        8.5,
        7,
        "All bands\ncompute\nsimultaneously",
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
    )

    ax2.text(
        1.5,
        1.5,
        "Thousands of GPU Cores\n↓\nParallel Processing\n↓\n5-100x Faster",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
    )

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        "/home/ywatanabe/proj/gPAC/docs/parallelization_diagram.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    print(
        "Parallelization diagram saved to: /home/ywatanabe/proj/gPAC/docs/parallelization_diagram.png"
    )


def create_pac_computation_flow():
    """Create a detailed diagram showing PAC computation flow."""

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    ax.set_title(
        "PAC (Modulation Index) Computation: Independent Frequency Bands",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)

    # Input signal
    ax.text(
        6,
        13,
        "Input EEG Signal",
        ha="center",
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
    )

    # Phase and amplitude extraction
    ax.text(
        3,
        11.5,
        "Phase Extraction\n(Low Frequencies)",
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFE5B4"),
    )

    ax.text(
        9,
        11.5,
        "Amplitude Extraction\n(High Frequencies)",
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E5FFE5"),
    )

    # Arrows
    ax.annotate(
        "",
        xy=(3, 11),
        xytext=(5, 12.5),
        arrowprops=dict(arrowstyle="->", lw=2, color="orange"),
    )
    ax.annotate(
        "",
        xy=(9, 11),
        xytext=(7, 12.5),
        arrowprops=dict(arrowstyle="->", lw=2, color="green"),
    )

    # Frequency bands
    phase_bands = ["4-6 Hz", "6-8 Hz", "8-10 Hz", "10-12 Hz"]
    amp_bands = ["30-50 Hz", "50-70 Hz", "70-90 Hz", "90-110 Hz"]

    # Phase bands
    for i, band in enumerate(phase_bands):
        y = 9.5 - i * 0.8
        ax.text(
            1.5,
            y,
            band,
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFE5B4"),
        )

    # Amplitude bands
    for i, band in enumerate(amp_bands):
        y = 9.5 - i * 0.8
        ax.text(
            10.5,
            y,
            band,
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#E5FFE5"),
        )

    # Modulation Index calculations (grid)
    ax.text(
        6,
        5.5,
        "Modulation Index Calculations\n(Each combination is independent)",
        ha="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0"),
    )

    # Grid of MI calculations
    grid_colors = ["#FF9999", "#99FF99", "#9999FF", "#FFFF99"]
    for i, p_band in enumerate(phase_bands):
        for j, a_band in enumerate(amp_bands):
            x = 4 + j * 1
            y = 4 - i * 0.7
            color = grid_colors[i % len(grid_colors)]

            rect = patches.Rectangle(
                (x - 0.4, y - 0.25),
                0.8,
                0.5,
                facecolor=color,
                edgecolor="black",
                linewidth=1,
                alpha=0.7,
            )
            ax.add_patch(rect)
            ax.text(
                x,
                y,
                f"MI\n{p_band}\n{a_band}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

    # Independence note
    ax.text(
        6,
        0.5,
        "Key Insight: Each MI calculation is INDEPENDENT\n"
        + "→ Perfect for GPU parallel processing\n"
        + "→ 1000s of GPU cores can compute simultaneously",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        "/home/ywatanabe/proj/gPAC/docs/pac_computation_flow.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    print(
        "PAC computation flow diagram saved to: /home/ywatanabe/proj/gPAC/docs/pac_computation_flow.png"
    )


if __name__ == "__main__":
    print("Creating parallelization diagrams...")
    create_parallelization_diagram()
    create_pac_computation_flow()
    print("Done!")

# EOF
