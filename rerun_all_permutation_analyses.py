#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-10 20:49:00 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/gPAC/rerun_all_permutation_analyses.py

"""
Re-run all analyses that use permutation testing after fixing the biased surrogate generation.

The surrogate generation was changed from restricted 25-75% shifts to full range (1 to time-1),
which affects all z-score calculations. This script re-runs:

1. Comparison with TensorPAC (16 pairs)
2. Parameter sweep benchmark 
3. Multiple GPU benchmarks
4. All example scripts with n_perm
"""

import os
import subprocess
import sys
from datetime import datetime

def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"Time: {datetime.now()}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ SUCCESS: {description}")
            if result.stdout:
                print("Output:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        else:
            print(f"✗ FAILED: {description}")
            print("Error:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"✗ EXCEPTION: {description}")
        print(f"Error: {e}")
        return False

def main():
    """Re-run all permutation-based analyses."""
    
    print("="*80)
    print("RE-RUNNING ALL PERMUTATION ANALYSES WITH FIXED SURROGATE GENERATION")
    print("Changed from biased 25-75% shifts to unbiased full range (1 to time-1)")
    print(f"Started at: {datetime.now()}")
    print("="*80)
    
    results = []
    
    # 1. TensorPAC comparison with 16 pairs
    results.append(run_command(
        "cd /data/gpfs/projects/punim2354/ywatanabe/gPAC/benchmark/pac_values_comparison_with_tensorpac && python generate_16_comparison_pairs.py",
        "TensorPAC comparison (16 pairs with z-scores)"
    ))
    
    # 2. Run comparison script
    results.append(run_command(
        "cd /data/gpfs/projects/punim2354/ywatanabe/gPAC/benchmark/pac_values_comparison_with_tensorpac && python run_all_comparisons.py",
        "Run all TensorPAC comparisons"
    ))
    
    # 3. Parameter sweep benchmark (includes n_perm tests)
    results.append(run_command(
        "cd /data/gpfs/projects/punim2354/ywatanabe/gPAC/benchmark/parameter_sweep && python parameter_sweep_benchmark.py",
        "Parameter sweep benchmark"
    ))
    
    # 4. Multiple GPU comodulogram benchmark
    results.append(run_command(
        "cd /data/gpfs/projects/punim2354/ywatanabe/gPAC/benchmark/_multiple_gpus && python comodulogram.py",
        "Multiple GPU comodulogram benchmark"
    ))
    
    # 5. Examples with PAC (includes n_perm)
    results.append(run_command(
        "cd /data/gpfs/projects/punim2354/ywatanabe/gPAC/examples/gpac && python example__PAC.py",
        "PAC example with permutations"
    ))
    
    # 6. Run tests that use permutations
    results.append(run_command(
        "cd /data/gpfs/projects/punim2354/ywatanabe/gPAC && pytest tests/comparison_with_tensorpac/test_pac.py -v",
        "PAC tests with TensorPAC"
    ))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF RE-RUN RESULTS")
    print("="*80)
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"Successful: {success_count}/{total_count}")
    print(f"Failed: {total_count - success_count}/{total_count}")
    
    if success_count < total_count:
        print("\n⚠️  Some analyses failed. Please check the output above.")
        sys.exit(1)
    else:
        print("\n✓ All analyses completed successfully with the new unbiased surrogate generation!")
        print("\nIMPORTANT: The z-scores calculated with the new method may differ from previous results.")
        print("This is expected as the previous 25-75% shift restriction created a biased null distribution.")
        print("The new full-range shifts provide more accurate statistical testing.")
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()

# EOF