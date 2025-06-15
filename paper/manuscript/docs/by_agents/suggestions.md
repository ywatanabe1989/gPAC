<!-- ---
!-- Timestamp: 2025-04-10 09:41:48
!-- Author: ywatanabe
!-- File: /ssh:sp:/home/ywatanabe/proj/gPAC/paper/manuscript/docs/suggestions.md
!-- --- -->

**Paper Title:** gPAC: A High-Performance PyTorch Framework for GPU-Accelerated Phase-Amplitude Coupling Analysis

**Authors:** [Your Name(s)/Affiliation(s)]

**Abstract:**

1.  Phase-Amplitude Coupling (PAC) is a crucial measure in neuroscience but computationally intensive, especially for large datasets or high-resolution analyses.
2.  Existing CPU-based tools like Tensorpac can be bottlenecks.
3.  We introduce gPAC, a PAC analysis framework implemented in PyTorch using the `scitex` library, designed for significant acceleration via GPU parallelization.
4.  We systematically benchmark gPAC (GPU and CPU modes) against Tensorpac (CPU) across a wide parameter space, varying signal dimensions (batch size, channels, segments, time length, sampling rate) and PAC calculation parameters (frequency band resolution, permutations).
5.  Performance is evaluated based on calculation time and system resource utilization (CPU, RAM, GPU, VRAM). Numerical accuracy against Tensorpac is validated.
6.  Results demonstrate substantial speedups (mention factor range, e.g., X-Y times faster) for gPAC on GPU compared to Tensorpac, particularly for larger data sizes and higher PAC resolutions. gPAC's CPU performance is also characterized. Accuracy validation confirms gPAC produces results highly consistent with Tensorpac.
7.  gPAC offers a validated, high-performance alternative for PAC analysis, enabling large-scale studies and facilitating integration with deep learning workflows due to its PyTorch backend and optional trainable components.

**1. Introduction:**

1.  **Background:**
    -   Define Phase-Amplitude Coupling (PAC).
    -   Explain its significance in neuroscience and other signal processing domains (e.g., identifying cross-frequency interactions).
    -   Briefly mention common PAC calculation methods (e.g., Modulation Index (MI), PLV, GLM).
2.  **Problem Statement:**
    -   Computational cost of PAC analysis, especially using methods like MI with filtering, Hilbert transform, and potentially permutation testing.
    -   Limitations of existing CPU-bound tools (e.g., `tensorpac`) for large datasets (scalp EEG, MEG, large iEEG arrays) or high-resolution parameter sweeps (many frequency bands).
3.  **Proposed Solution: gPAC**
    -   Introduce gPAC as a solution leveraging GPU acceleration via PyTorch and the `scitex` library.
    -   Highlight key features:
        -   PyTorch backend (`scitex.nn.PAC`, `scitex.nn.Filters`, `scitex.nn.Hilbert`, `scitex.nn.ModulationIndex`).
        -   GPU/CPU execution flexibility.
        -   Optional mixed-precision (FP16) support (`fp16` parameter).
        -   Potential for trainable components (`trainable` parameter in `scitex.nn.PAC` using `DifferentiableBandPassFilter`).
4.  **Paper Objectives & Outline:**
    -   Quantify the performance advantage of gPAC (GPU) over `tensorpac` (CPU).
    -   Compare gPAC (CPU) performance.
    -   Validate the numerical accuracy of gPAC against `tensorpac`.
    -   Outline the subsequent sections (Methods, Results, Discussion, Conclusion).

**2. Methods:**

1.  **PAC Implementations:**
    -   **gPAC (`scitex` library):**
        -   Describe the pipeline: Bandpass filtering (`scitex.nn.BandPassFilter` or `scitex.nn.DifferentiableBandPassFilter`) -> Hilbert Transform (`scitex.nn.Hilbert`) -> Modulation Index (`scitex.nn.ModulationIndex`).
        -   Mention implementation details: PyTorch backend, use of `torch.fft`, convolution for filtering.
        -   Specify options tested: `device` (cpu/cuda), `fp16`, `trainable`, `n_perm`, `in_place`.
    -   **Tensorpac:**
        -   Describe as the baseline CPU implementation.
        -   Mention its reliance on NumPy/SciPy for filtering (Wavelet) and MI calculation.
        -   Specify options tested: `use_threads`, `n_perm`.
2.  **Benchmarking Setup:**
    -   **Hardware:** Specify CPU model, RAM size, GPU model, VRAM size (refer perhaps to `resource_info.py` output).
    -   **Software:** OS, Python version, `scitex` version, `tensorpac` version, PyTorch version, CUDA version (refer to `requirements.txt`).
    -   **Synthetic Data Generation:**
        -   Method: Use `scitex.dsp.demo_sig` with `sig_type="tensorpac"` or `"pac"`.
        -   Rationale: Provides controllable, reproducible signals with known PAC characteristics for benchmarking.
    -   **Parameter Space:**
        -   Detail the parameters varied based on `config/PARAMS.yaml` (`VARIATIONS` or `ALL` section).
        -   List baseline parameters (`BASELINE` section).
        -   Explain the experimental design (varying one parameter group at a time from baseline, or full grid search if `ALL` was used). Reference `scripts/utils/define_parameter_space.py` logic if applicable.
    -   **Performance Metrics:**
        -   Calculation Time: How initialization time and PAC calculation time were measured (using `scitex.gen.TimeStamper` within `BaseHandler`). Mention averaging over `n_calc` runs.
        -   Resource Usage: How CPU %, RAM (GiB), GPU %, and VRAM (GiB) were logged (using `scitex.resource.log_processor_usages` via `record_processor_usages.py`). Explain linking to calculation times (nearest timestamp matching in `plot_linked_data.py`).
3.  **Validation:**
    -   Method: Compare PAC matrices generated by gPAC (non-trainable, fp32, CPU/GPU) and Tensorpac using identical input data and frequency band definitions.
    -   Metrics: Pearson correlation, Spearman correlation, Kendall's Tau, RMS of absolute difference, RMS of relative difference (reference `validate_precisions.py`).

**3. Results:**

1.  **Calculation Time Comparison:**
    -   Present plots (bar plots/line plots, generated by `plot_linked_data.py`) showing mean calculation time vs. varied parameters (batch size, n_chs, seq_len, fs, pha_n_bands, amp_n_bands, n_perm).
    -   Compare gPAC (GPU) vs. gPAC (CPU) vs. Tensorpac (CPU).
    -   Highlight speedup factors, especially for GPU vs. Tensorpac.
    -   Show the effect of `fp16` on gPAC (GPU) speed.
    -   Discuss the negligible time impact of `trainable`, `no_grad`, `in_place` flags (if applicable).
    -   Compare Tensorpac `use_threads=True` vs `False`.
2.  **Resource Usage Comparison:**
    -   Present plots (similar format to time plots) showing resource utilization (CPU%, RAM, GPU%, VRAM) vs. varied parameters.
    -   Compare resource profiles of gPAC (GPU/CPU) and Tensorpac (CPU).
    -   Show VRAM usage scaling for gPAC (GPU).
3.  **Validation Results:**
    -   Present a table summarizing the correlation and difference metrics between gPAC and Tensorpac outputs under baseline conditions.
    -   Include a figure showing example PAC matrices from both packages and their difference plot (reference `validate_precisions.py` output figure).

**4. Discussion:**

1.  **Summary of Findings:**
    -   Reiterate the significant performance gains of gPAC on GPU.
    -   Summarize the performance characteristics of gPAC on CPU compared to Tensorpac.
    -   Confirm the numerical consistency between gPAC and Tensorpac.
2.  **Interpretation & Implications:**
    -   Discuss the practical implications of the speedup (enabling larger analyses, faster iterations, feasibility of high-resolution PAC).
    -   Explain the source of the speedup (GPU parallelization for filtering, Hilbert, MI binning).
    -   Discuss resource usage trade-offs (higher VRAM for GPU vs. lower RAM).
3.  **Limitations:**
    -   Benchmarking performed on synthetic data.
    -   Results specific to the hardware used.
    -   Comparison focused on a specific PAC method (MI via filter-Hilbert).
    -   Overhead: Mention potential overhead for very small computations where GPU transfer time might dominate.
4.  **Future Directions:**
    -   Benchmarking on real-world electrophysiological data.
    -   Exploring the `trainable` filter capabilities in data-driven analysis.
    -   Integration into larger deep learning models for end-to-end analysis.
    -   Extending `scitex.dsp` to include other PAC metrics accelerated on GPU.

**5. Conclusion:**

1.  gPAC, leveraging PyTorch and the `scitex` library, provides a validated and substantially faster alternative to traditional CPU-based PAC analysis tools like Tensorpac, especially when utilizing GPUs.
2.  Its performance advantages and PyTorch integration open new avenues for large-scale PAC studies and novel deep learning applications in signal processing.

**References:**

-   Cite relevant papers on PAC theory and applications.
-   Cite the Tensorpac paper/documentation.
-   Cite PyTorch.
-   Cite NumPy, SciPy.
-   Cite the `scitex` library (if available, otherwise point to the repository).

**Supplementary Material (Optional):**

-   Link to the code repository (`README.org`).
-   Detailed hardware/software specifications.
-   All generated performance plots.
-   Full table of validation metrics.

<!-- EOF -->