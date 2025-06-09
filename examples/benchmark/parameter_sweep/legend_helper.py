#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-09"
# Author: ywatanabe
# File: legend_helper.py
# ----------------------------------------
"""
Helper function for saving legends separately from main figures.

This module provides a utility function to handle the common pattern of:
1. Saving a legend as a separate file
2. Removing it from the main figure to avoid overlap
3. Automatically generating appropriate filenames

Usage:
    from legend_helper import save_legend_separately
    
    fig, ax = plt.subplots()
    # ... plotting code ...
    save_legend_separately(ax, "my_plot.png")
    plt.savefig("my_plot.png")
"""

import os
from pathlib import Path


def save_legend_separately(ax, main_filename, legend_suffix="_legend"):
    """
    Save legend as a separate file and remove it from the main figure.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes containing the legend to save
    main_filename : str
        The filename of the main figure (e.g., "plot.png")
    legend_suffix : str, optional
        Suffix to add to the legend filename (default: "_legend")
    
    Returns
    -------
    str
        The filename where the legend was saved
    
    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], label="data")
    >>> legend_filename = save_legend_separately(ax, "my_plot.png")
    >>> print(legend_filename)  # "my_plot_legend.png"
    >>> plt.savefig("my_plot.png")  # Saves without legend
    """
    # Parse the main filename to create legend filename
    path = Path(main_filename)
    legend_filename = str(path.parent / f"{path.stem}{legend_suffix}{path.suffix}")
    
    # Save legend separately using mngs
    try:
        # Try to use mngs if available
        ax.legend("separate", filename=legend_filename)
    except:
        # Fallback to manual legend extraction if mngs not available
        import matplotlib.pyplot as plt
        
        # Create legend if it doesn't exist
        if not ax.get_legend():
            ax.legend()
        
        # Get the legend
        legend = ax.get_legend()
        if legend:
            # Create a new figure for the legend
            legend_fig = plt.figure(figsize=(3, 2))
            legend_fig.legend(
                handles=legend.legendHandles,
                labels=[t.get_text() for t in legend.get_texts()],
                loc='center'
            )
            legend_fig.savefig(legend_filename, bbox_inches='tight', dpi=300)
            plt.close(legend_fig)
    
    # Remove legend from main figure
    legend = ax.get_legend()
    if legend:
        legend.remove()
    
    return legend_filename


def add_legend_back(ax, legend_data):
    """
    Add a legend back to an axes after it was removed.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the legend to
    legend_data : dict
        Dictionary containing handles and labels
    """
    if legend_data:
        ax.legend(
            handles=legend_data['handles'],
            labels=legend_data['labels']
        )


def get_legend_data(ax):
    """
    Extract legend data from an axes for later recreation.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes containing the legend
        
    Returns
    -------
    dict or None
        Dictionary with 'handles' and 'labels' keys, or None if no legend
    """
    legend = ax.get_legend()
    if legend:
        return {
            'handles': legend.legendHandles,
            'labels': [t.get_text() for t in legend.get_texts()]
        }
    return None