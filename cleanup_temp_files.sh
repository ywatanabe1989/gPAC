#!/bin/bash
# Safe cleanup of temporary test files

# Create directories for organization
mkdir -p benchmarks/comparison_scripts
mkdir -p benchmarks/analysis_scripts
mkdir -p benchmarks/figures

# Move important comparison scripts to benchmarks
mv test_hres_mres_comparison_improved.py benchmarks/comparison_scripts/
mv test_mi_comparison.py benchmarks/comparison_scripts/
mv benchmark_sequential_simple.py benchmarks/comparison_scripts/
mv fair_comparison.py benchmarks/comparison_scripts/

# Move analysis scripts
mv analyze_filter_differences.py benchmarks/analysis_scripts/
mv deep_investigation_differences.py benchmarks/analysis_scripts/
mv investigate_filtfilt_detail.py benchmarks/analysis_scripts/
mv verify_hres_mres_bands.py benchmarks/analysis_scripts/

# Move important figures to benchmarks/figures
mv hres_mres_comparison_improved.png benchmarks/figures/
mv mi_comparison.png benchmarks/figures/
mv readme_demo_output.png benchmarks/figures/
mv pac_model_comparison.png benchmarks/figures/

# Use safe_rm.sh to remove temporary files
./safe_rm.sh _debug_performance.py
./safe_rm.sh _fair_benchmark.py
./safe_rm.sh _scaling_analysis.py
./safe_rm.sh check_edge_mode_differences.py
./safe_rm.sh check_filtfilt_difference.py
./safe_rm.sh explain_differences.py
./safe_rm.sh filter_length_comparison.py
./safe_rm.sh filter_resolution_comparison.py
./safe_rm.sh fix_edge_mode.py
./safe_rm.sh resolution_demo.py
./safe_rm.sh sequential_filtfilt_fixed.py
./safe_rm.sh test_differences.py
./safe_rm.sh test_edge_handling_overhead.py
./safe_rm.sh test_edge_handling_pac_context.py
./safe_rm.sh test_edge_mode_pac.py
./safe_rm.sh test_filtfilt_performance.py
./safe_rm.sh test_filtfilt_simple.py
./safe_rm.sh test_tensorpac_compatibility.py
./safe_rm.sh test_tensorpac_float32.py
./safe_rm.sh test_torchaudio_filtfilt.py
./safe_rm.sh visualize_filtfilt_math.py
./safe_rm.sh final_comparison_sequential.py
./safe_rm.sh test_hres_mres_comparison.py
./safe_rm.sh test_sequential_filtfilt.py

# Remove other temporary files
./safe_rm.sh _performance_comparison.png
./safe_rm.sh edge_handling_comparison.png
./safe_rm.sh edge_mode_differences.png
./safe_rm.sh filter_response_comparison.png
./safe_rm.sh filtfilt_math_comparison.png
./safe_rm.sh gpac_edge_mode_test.png
./safe_rm.sh pac_computation_comparison.png
./safe_rm.sh tensorpac_filter_comparison.png
./safe_rm.sh readme_demo_edge_mode_output.png
./safe_rm.sh readme_demo_filtfilt_comparison.png
./safe_rm.sh readme_demo_with_edge_mode_output.png
./safe_rm.sh hres_mres_comparison.png

# Keep the summary document
echo "Keeping SEQUENTIAL_FILTFILT_SUMMARY.md as important documentation"

echo "Cleanup complete! Important files moved to:"
echo "  - benchmarks/comparison_scripts/"
echo "  - benchmarks/analysis_scripts/"
echo "  - benchmarks/figures/"