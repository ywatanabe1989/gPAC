#!/usr/bin/env python3
"""Convert GIF files to PNG for LaTeX compatibility"""

import os
from PIL import Image
import glob

def convert_gif_to_png(gif_path):
    """Convert a GIF file to PNG, extracting the first frame"""
    png_path = gif_path.replace('.gif', '.png')
    
    # Skip if PNG already exists
    if os.path.exists(png_path):
        print(f"Skipping {gif_path} - PNG already exists")
        return png_path
    
    try:
        # Open GIF and save first frame as PNG
        with Image.open(gif_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(png_path, 'PNG')
        print(f"Converted {gif_path} -> {png_path}")
        return png_path
    except Exception as e:
        print(f"Error converting {gif_path}: {e}")
        return None

# Convert all example GIFs
example_dirs = [
    "examples/gpac/example__PAC_simple_out",
    "examples/gpac/example__BandPassFilter_out", 
    "examples/gpac/example__Hilbert_out",
    "examples/gpac/example__ModulationIndex_out",
    "examples/gpac/example__simple_trainable_PAC_out",
    "examples/gpac/example__trainable_PAC_out"
]

for dir_path in example_dirs:
    if os.path.exists(dir_path):
        gif_files = glob.glob(os.path.join(dir_path, "*.gif"))
        for gif_file in gif_files:
            convert_gif_to_png(gif_file)

print("\nConversion complete!")