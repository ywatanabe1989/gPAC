#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-25 14:40:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/visualize_filtfilt_math.py
# ----------------------------------------
"""
Visualize the mathematical difference between filtfilt implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib
matplotlib.use('Agg')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# TensorPAC/scipy.filtfilt method
ax1.set_title('TensorPAC: scipy.signal.filtfilt\n(Sequential filtering)', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')

# Draw the flow
# Input signal
ax1.text(5, 9, 'Input Signal x(t)', ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

# Forward filter
ax1.arrow(5, 8.5, 0, -1, head_width=0.3, head_length=0.1, fc='black', ec='black')
ax1.text(5, 7, 'Filter H(z)\n(forward)', ha='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

# Intermediate result
ax1.arrow(5, 6.5, 0, -1, head_width=0.3, head_length=0.1, fc='black', ec='black')
ax1.text(5, 5, 'y₁(t) = H * x', ha='center', fontsize=10)

# Time reverse
ax1.arrow(5, 4.5, 0, -1, head_width=0.3, head_length=0.1, fc='black', ec='black')
ax1.text(5, 3.5, 'Time Reverse', ha='center', fontsize=10, style='italic')

# Backward filter
ax1.arrow(5, 3, 0, -1, head_width=0.3, head_length=0.1, fc='black', ec='black')
ax1.text(5, 2, 'Filter H(z)\n(on reversed)', ha='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

# Final result
ax1.arrow(5, 1.5, 0, -1, head_width=0.3, head_length=0.1, fc='black', ec='black')
ax1.text(5, 0.5, 'Time Reverse\ny(t) = H * H * x', ha='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))

# Add frequency response note
ax1.text(1, 5, 'Frequency\nResponse:\n|H(f)|²', ha='center', fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))

# gPAC method
ax2.set_title('gPAC: Averaging approximation\n(Parallel filtering)', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Input signal
ax2.text(5, 9, 'Input Signal x(t)', ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

# Split into two paths
ax2.arrow(3.5, 8.5, -1, -1, head_width=0.2, head_length=0.1, fc='black', ec='black')
ax2.arrow(6.5, 8.5, 1, -1, head_width=0.2, head_length=0.1, fc='black', ec='black')

# Forward path
ax2.text(2, 6.5, 'Conv1D H\n(forward)', ha='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
ax2.arrow(2, 6, 0, -1, head_width=0.3, head_length=0.1, fc='black', ec='black')
ax2.text(2, 4.5, 'y₁(t) = H * x', ha='center', fontsize=10)

# Backward path
ax2.text(8, 6.5, 'Time Reverse', ha='center', fontsize=10, style='italic')
ax2.arrow(8, 6.2, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
ax2.text(8, 5.3, 'Conv1D H\n(on reversed)', ha='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
ax2.arrow(8, 4.8, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
ax2.text(8, 4, 'Time Reverse\ny₂(t)', ha='center', fontsize=10)

# Merge
ax2.arrow(2, 4, 1.5, -1, head_width=0.2, head_length=0.1, fc='black', ec='black')
ax2.arrow(8, 3.5, -1.5, -1, head_width=0.2, head_length=0.1, fc='black', ec='black')

# Average
ax2.text(5, 2, 'Average\n(y₁ + y₂) / 2', ha='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))

# Final result
ax2.arrow(5, 1.5, 0, -0.7, head_width=0.3, head_length=0.1, fc='black', ec='black')
ax2.text(5, 0.5, 'y(t) ≈ H * x\n(zero-phase)', ha='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))

# Add frequency response note
ax2.text(1, 5, 'Frequency\nResponse:\n≈|H(f)|\n(not squared)', ha='center', fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))

plt.suptitle('Mathematical Difference: Sequential vs Averaged Filtering', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('filtfilt_math_comparison.png', dpi=200, bbox_inches='tight')
print("Saved mathematical comparison to: filtfilt_math_comparison.png")