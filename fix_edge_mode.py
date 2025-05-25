#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-25 14:10:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/fix_edge_mode.py
# ----------------------------------------
"""
Script to fix the edge_mode parameter passing in PAC class.
"""

import fileinput
import sys

# Read the file
with open('src/gpac/_PAC.py', 'r') as f:
    content = f.read()

# Fix 1: Add edge_mode to CombinedBandPassFilter call
old_filter_call = """            filtfilt_mode=self.filtfilt_mode,  # Pass filtfilt mode
        )"""

new_filter_call = """            filtfilt_mode=self.filtfilt_mode,  # Pass filtfilt mode
            edge_mode=self.edge_mode,  # Pass edge mode
        )"""

content = content.replace(old_filter_call, new_filter_call)

# Write back
with open('src/gpac/_PAC.py', 'w') as f:
    f.write(content)

print("✅ Fixed edge_mode passing in _PAC.py")

# Also need to ensure edge_mode is passed in calculate_pac
with open('src/gpac/_calculate_gpac.py', 'r') as f:
    calc_content = f.read()

# Check if edge_mode is in the PAC instantiation
if "edge_mode=edge_mode," not in calc_content:
    # Find the PAC instantiation and add edge_mode
    old_pac_call = """        filtfilt_mode=filtfilt_mode,
    ).to(resolved_device)"""
    
    new_pac_call = """        filtfilt_mode=filtfilt_mode,
        edge_mode=edge_mode,
    ).to(resolved_device)"""
    
    calc_content = calc_content.replace(old_pac_call, new_pac_call)
    
    with open('src/gpac/_calculate_gpac.py', 'w') as f:
        f.write(calc_content)
    
    print("✅ Fixed edge_mode passing in _calculate_gpac.py")
else:
    print("✅ edge_mode already passed in _calculate_gpac.py")

print("\nNow let's test if the fix works...")