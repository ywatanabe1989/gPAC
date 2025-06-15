#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-08 00:17:00 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/gPAC/tests/comparison_with_tensorpac/run_comparison.py
# ----------------------------------------

"""
Run comprehensive comparison between gPAC and TensorPAC.
Generates detailed comparison report with performance metrics.
"""

import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from test_comprehensive_comparison import run_all_comparisons


def main():
    parser = argparse.ArgumentParser(
        description="Compare gPAC and TensorPAC implementations"
    )
    parser.add_argument(
        "--save-report", action="store_true", help="Save comparison report to file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_results",
        help="Directory to save reports (default: comparison_results)",
    )
    args = parser.parse_args()

    # Create output directory if saving report
    if args.save_report:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(
            args.output_dir, f"gpac_tensorpac_comparison_{timestamp}.txt"
        )

        # Redirect stdout to file
        import sys

        original_stdout = sys.stdout
        with open(report_file, "w") as f:
            sys.stdout = f
            print(f"gPAC vs TensorPAC Comparison Report")
            print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            print()

            # Run comparisons
            run_all_comparisons()

        sys.stdout = original_stdout
        print(f"\nReport saved to: {report_file}")
    else:
        # Run comparisons normally
        run_all_comparisons()


if __name__ == "__main__":
    main()
