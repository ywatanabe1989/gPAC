<!-- ---
!-- Timestamp: 2025-06-03 14:43:21
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/.claude-worktree/gPAC/examples/performance/multiple_gpus/README.md
!-- --- -->

# Multi-GPU Performance Testing Suite

A modular collection of performance tests for multi-GPU PAC analysis, addressing different aspects of parallel computing benefits.

## 🎯 Visual Summary

<div align="center">
  <img src="throughput_out/throughput_scaling.gif" alt="Multi-GPU Throughput Scaling" width="100%">
  
  **Key Result: Multi-GPU enables processing 4x larger datasets with high efficiency**
</div>

## 📊 Test Modules

### `speed.py` - Computational Speed Test
**Purpose**: Measure speedup for identical workloads  
**Question**: "How much faster is multi-GPU for the same computation?"
```bash
python speed.py --n_perm 0
```
- Tests same data size on single vs multi-GPU
- Measures computational acceleration 
- Typical result: 1.2-1.8x speedup

### `vram.py` - VRAM Scaling Test  
**Purpose**: Test memory capacity scaling  
**Question**: "Can we process datasets that don't fit on single GPU?"
```bash
python vram.py --n_perm 0
```
- Finds maximum single GPU batch size
- Tests proportionally larger datasets on multi-GPU
- Demonstrates memory capacity benefits (measured in GB)

### `throughput.py` - Throughput Scaling Test ⭐
**Purpose**: Process 4x more data efficiently (main value proposition)  
**Question**: "Can we process 4x larger datasets with reasonable throughput?"
```bash
python throughput.py --n_perm 0
```
- **Key insight**: Scale from single GPU to multiple GPUs processing 4x larger datasets
- Measures samples/second for larger datasets
- Shows real-world benefit: handle datasets impossible on single GPU

### `comodulogram.py` - Consistency Verification Test
**Purpose**: Verify computational consistency between single and multi-GPU  
**Question**: "Do we get identical results with multi-GPU processing?"
```bash
python comodulogram.py --batch_size 4 --pha_bands 20 --amp_bands 15 --n_perm 0
```
- Computes PAC comodulogram on identical data using both configurations
- Visualizes single GPU, multi-GPU, and difference plots
- Verifies correlation and numerical consistency (typical: r > 0.999)

## 🛠️ Shared Components

### `utils.py` - Common Utilities
- GPU memory monitoring (in GB)
- Data generation and configuration
- Performance measurement functions
- Visualization and reporting (DRY principle)

## 📈 Key Insights

1. **Speed Test**: Measures computational efficiency (1.2-1.8x typical)
2. **VRAM Test**: Enables larger datasets than single GPU memory  
3. **Throughput Test**: ⭐ **Most meaningful** - process 4x more data efficiently
4. **Comodulogram Test**: Real-world research workflow demonstration

## 🎯 Main Value Proposition

The **throughput test** demonstrates the key multi-GPU benefit:
- Single GPU: Limited by VRAM capacity
- Multi-GPU: Process 4x larger datasets with reasonable throughput
- **Conclusion**: Scale dataset size, not just computation speed

## 📊 Example Results

### Speed Test Results
![Speed Comparison](speed_out/speed_comparison.gif)
*Computational speed comparison showing calculation time vs number of samples for different GPU configurations*

### Throughput Scaling Results
![Throughput Scaling](throughput_out/throughput_scaling.gif)
*Throughput scaling showing samples per second vs batch size for single and multi-GPU configurations*

### Comodulogram Consistency Test
![Comodulogram Comparison](comodulogram_out/comodulogram_comparison.gif)
*PAC comodulogram comparison demonstrating identical results between single and multi-GPU computations*

### Performance Metrics
```
Baseline Single GPU:    32 samples in 10s = 3.2 samples/sec
4x Multi-GPU Scaling:   128 samples in 12s = 10.7 samples/sec  
Throughput Gain:        3.34x
Scaling Efficiency:     83%
```

## 🚀 Quick Start

```bash
# Test computational speedup
python speed.py

# Test the main value proposition (recommended)
python throughput.py

# Test memory scaling capabilities  
python vram.py

# Test computational consistency
python comodulogram.py
```

## 📁 Output Structure

Each test creates its own output directory:
- `speed_out/` - Speed test results and plots
- `vram_out/` - VRAM scaling results  
- `throughput_out/` - Throughput scaling results
- `comodulogram_out/` - Comodulogram analysis results

## 🔧 Requirements

- CUDA-capable GPUs (ideally 4+ for full testing)
- Set `CUDA_VISIBLE_DEVICES=0,1,2,3` for multi-GPU testing
- PyTorch with CUDA support
- gPAC library with multi-GPU enabled

<!-- EOF -->