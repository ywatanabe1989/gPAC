<!-- ---
!-- Timestamp: 2025-06-03 18:48:03
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/.claude-worktree/gPAC/examples/performance/multiple_gpus/TODO.md
!-- --- -->

# speed.py ✅ IMPLEMENTED
- Panel 1 ✅ COMPLETED
  - [x] x: actual calculation time (implemented in create_speed_comparison_plot)
  - [x] y: n samples (implemented in create_speed_comparison_plot)
  - [x] hue: 1 gpu, 2 gpus, 3 gpus, 4 gpus (implemented in create_speed_comparison_plot)
  - Note: CPU comparison not included as gPAC is GPU-only

# throughput.py ✅ IMPLEMENTED
- Panel 1 ✅ COMPLETED
- [x] x: batch size (implemented in create_throughput_comparison_plot)
- [x] y: samples/sec (implemented in create_throughput_comparison_plot)
- [x] hue: 1 gpu, 2 gpus, 3 gpus, 4 gpus (implemented in create_throughput_comparison_plot)
- Note: CPU comparison not included as gPAC is GPU-only

# vram.py
- [x] Fixed memory measurement - was using wrong PAC initialization parameters (multi_gpu vs device_ids)
- [x] Changed units from GB to GiB throughout
- [x] Added debug logging to diagnose memory measurement issues
- [x] Fixed model device placement for proper GPU memory tracking

# README.md ✅ COMPLETED
- [x] Add figures created by files to show on github
  - [x] speed_comparison.gif (line 86 in README.md)
  - [x] throughput_scaling.gif (line 90 in README.md)  
  - [x] comodulogram_comparison.gif (line 94 in README.md)

<!-- EOF -->