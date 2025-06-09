<!-- ---
!-- Timestamp: 2025-06-02 06:14:36
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/.claude-worktree/gPAC/project_management/feature_requests/feature-request-smart-memory-allocation.md
!-- --- -->

gPAC Memory Management Integration Guide
This guide shows how to integrate the smart memory allocation system into the main PAC class for optimal GPU memory utilization.
Current Architecture Analysis
python# Current PAC class structure
class PAC(nn.Module):
    def __init__(self, ...):
        # Basic initialization
        self.bandpass = BandPassFilter(...)
        self.hilbert = Hilbert(...)
        self.modulation_index = ModulationIndex(...)
        
    def forward(self, x):
        # Simple forward pass
        # No memory management optimization
Target Architecture: Smart Memory-Aware PAC
pythonclass PAC(nn.Module):
    """
    GPU-accelerated PAC with intelligent memory management.
    
    Automatically selects optimal processing strategy based on:
    - Available GPU memory
    - Problem size (batch, channels, time, permutations)
    - Hardware capabilities
    """
    
    def __init__(
        self,
        seq_len: int,
        fs: float,
        # ... existing parameters ...
        memory_strategy: str = "auto",  # "auto", "conservative", "aggressive"
        max_memory_usage: float = 0.8,  # Use 80% of available VRAM
        enable_memory_profiling: bool = False,
    ):
        super().__init__()
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            strategy=memory_strategy,
            max_usage=max_memory_usage,
            enable_profiling=enable_memory_profiling
        )
        
        # Existing components
        self.bandpass = BandPassFilter(...)
        self.hilbert = Hilbert(...)
        self.modulation_index = ModulationIndex(...)
        
        # Memory-aware configuration
        self._configure_memory_strategy()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Memory-aware forward pass with automatic strategy selection."""
        
        # Analyze input and select strategy
        strategy = self.memory_manager.select_strategy(x, self.n_perm)
        
        # Execute based on selected strategy
        if strategy == "vectorized":
            return self._forward_vectorized(x)
        elif strategy == "chunked":
            return self._forward_chunked(x)
        else:  # sequential
            return self._forward_sequential(x)
Step 1: Create Memory Manager Class
python# File: src/gpac/_MemoryManager.py

import torch
import psutil
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from .memory_estimator import estimate_vram_usage

@dataclass
class MemoryStrategy:
    """Configuration for a specific memory strategy."""
    name: str
    max_batch_size: int
    max_permutations: int
    chunk_size: Optional[int] = None
    description: str = ""

class MemoryManager:
    """Intelligent memory management for PAC analysis."""
    
    def __init__(
        self,
        strategy: str = "auto",
        max_usage: float = 0.8,
        enable_profiling: bool = False
    ):
        self.strategy = strategy
        self.max_usage = max_usage
        self.enable_profiling = enable_profiling
        
        # Get system info
        self.device_info = self._get_device_info()
        self.available_memory = self._get_available_memory()
        
        # Define strategies
        self.strategies = self._define_strategies()
        
    def _get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        if not torch.cuda.is_available():
            return {"device": "cpu", "memory_gb": psutil.virtual_memory().total / (1024**3)}
        
        device_props = torch.cuda.get_device_properties(0)
        return {
            "device": "cuda",
            "name": device_props.name,
            "memory_gb": device_props.total_memory / (1024**3),
            "compute_capability": f"{device_props.major}.{device_props.minor}"
        }
    
    def _get_available_memory(self) -> float:
        """Get currently available memory in GB."""
        if self.device_info["device"] == "cpu":
            return psutil.virtual_memory().available / (1024**3)
        
        # GPU memory
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        total = self.device_info["memory_gb"]
        return (total - allocated) * self.max_usage
    
    def _define_strategies(self) -> Dict[str, MemoryStrategy]:
        """Define memory strategies based on available resources."""
        available_gb = self.available_memory
        
        return {
            "vectorized": MemoryStrategy(
                name="vectorized",
                max_batch_size=64,
                max_permutations=50,
                description="Process all permutations in parallel"
            ),
            "chunked": MemoryStrategy(
                name="chunked", 
                max_batch_size=32,
                max_permutations=1000,
                chunk_size=min(50, max(10, int(available_gb * 5))),
                description="Process permutations in chunks"
            ),
            "sequential": MemoryStrategy(
                name="sequential",
                max_batch_size=16,
                max_permutations=10000,
                description="Process permutations one by one"
            )
        }
    
    def select_strategy(
        self, 
        x: torch.Tensor, 
        n_perm: Optional[int],
        **pac_config
    ) -> str:
        """Select optimal strategy based on input characteristics."""
        
        if self.strategy != "auto":
            return self.strategy
        
        # Extract dimensions
        if x.ndim == 3:
            batch_size, n_chs, seq_len = x.shape
            n_segments = 1
        else:
            batch_size, n_chs, n_segments, seq_len = x.shape
        
        # Estimate memory requirements
        memory_estimate = estimate_vram_usage(
            batch_size=batch_size,
            n_chs=n_chs,
            seq_len=seq_len,
            fs=pac_config.get('fs', 256),
            pha_n_bands=pac_config.get('pha_n_bands', 30),
            amp_n_bands=pac_config.get('amp_n_bands', 30),
            n_perm=n_perm or 0,
            fp16=pac_config.get('fp16', False),
            device=self.device_info["device"]
        )
        
        estimated_gb = memory_estimate['total_mb'] / 1024
        
        # Strategy selection logic
        if n_perm is None or n_perm == 0:
            return "vectorized"  # No permutations, use fastest
        
        if estimated_gb < self.available_memory * 0.5 and n_perm <= 50:
            return "vectorized"
        elif estimated_gb < self.available_memory and n_perm <= 500:
            return "chunked"
        else:
            return "sequential"
    
    def get_optimal_chunk_size(self, total_permutations: int) -> int:
        """Calculate optimal chunk size for chunked processing."""
        if total_permutations <= 50:
            return total_permutations
        
        # Base chunk size on available memory
        base_chunk = max(10, int(self.available_memory * 5))
        return min(base_chunk, total_permutations // 4)
Step 2: Integrate into Main PAC Class
python# File: src/gpac/_PAC.py (updated)

from ._MemoryManager import MemoryManager
from ._Profiler import create_profiler

class PAC(nn.Module):
    def __init__(
        self,
        seq_len: int,
        fs: float,
        # ... existing parameters ...
        memory_strategy: str = "auto",
        max_memory_usage: float = 0.8,
        enable_memory_profiling: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # Store configuration
        self.seq_len = seq_len
        self.fs = fs
        self.n_perm = kwargs.get('n_perm', None)
        self.fp16 = kwargs.get('fp16', False)
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            strategy=memory_strategy,
            max_usage=max_memory_usage,
            enable_profiling=enable_memory_profiling
        )
        
        # Initialize profiler if enabled
        self.profiler = create_profiler(enable_gpu=True) if enable_memory_profiling else None
        
        # Existing components (unchanged)
        self.bandpass = BandPassFilter(...)
        self.hilbert = Hilbert(...)
        self.modulation_index = ModulationIndex(...)
        
        # Cache for reused computations
        self._strategy_cache = {}
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Memory-aware PAC computation with automatic strategy selection."""
        
        if self.profiler:
            self.profiler.profile("Total PAC Analysis").__enter__()
        
        try:
            # Prepare configuration for memory manager
            pac_config = {
                'fs': self.fs,
                'pha_n_bands': len(self.PHA_MIDS_HZ),
                'amp_n_bands': len(self.AMP_MIDS_HZ),
                'fp16': self.fp16
            }
            
            # Select optimal strategy
            strategy = self.memory_manager.select_strategy(x, self.n_perm, **pac_config)
            
            # Log strategy selection
            if hasattr(self, '_last_strategy') and self._last_strategy != strategy:
                print(f"Memory strategy switched: {self._last_strategy} → {strategy}")
            self._last_strategy = strategy
            
            # Execute based on strategy
            if strategy == "vectorized":
                return self._forward_vectorized(x)
            elif strategy == "chunked":
                return self._forward_chunked(x)
            else:
                return self._forward_sequential(x)
                
        finally:
            if self.profiler:
                self.profiler.profile("Total PAC Analysis").__exit__(None, None, None)
    
    def _forward_vectorized(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Vectorized processing - all permutations in parallel."""
        if self.profiler:
            with self.profiler.profile("Vectorized Forward"):
                return self._compute_pac_core(x, use_vectorized_surrogates=True)
        else:
            return self._compute_pac_core(x, use_vectorized_surrogates=True)
    
    def _forward_chunked(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Chunked processing - permutations in batches."""
        if self.profiler:
            with self.profiler.profile("Chunked Forward"):
                chunk_size = self.memory_manager.get_optimal_chunk_size(self.n_perm)
                return self._compute_pac_core(x, use_chunked_surrogates=True, chunk_size=chunk_size)
        else:
            chunk_size = self.memory_manager.get_optimal_chunk_size(self.n_perm)
            return self._compute_pac_core(x, use_chunked_surrogates=True, chunk_size=chunk_size)
    
    def _forward_sequential(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Sequential processing - one permutation at a time."""
        if self.profiler:
            with self.profiler.profile("Sequential Forward"):
                return self._compute_pac_core(x, use_sequential_surrogates=True)
        else:
            return self._compute_pac_core(x, use_sequential_surrogates=True)
    
    def _compute_pac_core(self, x: torch.Tensor, **surrogate_options) -> Dict[str, torch.Tensor]:
        """Core PAC computation with configurable surrogate strategy."""
        
        # Input validation and preparation (unchanged)
        x = self._ensure_4d_input(x)
        
        # Forward pass through existing pipeline
        with torch.set_grad_enabled(bool(self.trainable)):
            # Process signal (existing code)
            x = x.reshape(...)  # Reshape for bandpass
            x = self.bandpass(x, edge_len=0)
            x = self.hilbert(x)
            
            # Extract phase and amplitude
            pha = x[..., :n_pha_bands, :, 0]  # Phase
            amp = x[..., n_pha_bands:, :, 1]  # Amplitude
            
            # Rearrange dimensions
            pha = pha.permute(0, 1, 3, 2, 4)
            amp = amp.permute(0, 1, 3, 2, 4)
            
            # Remove edges
            edge_len = self._calculate_edge_length(seq_len)
            if edge_len > 0:
                pha = pha[..., edge_len:-edge_len]
                amp = amp[..., edge_len:-edge_len]
            
            # Convert precision
            if self.fp16:
                pha, amp = pha.half(), amp.half()
            
            # Calculate modulation index
            mi_results = self.modulation_index(pha, amp)
            pac_values = mi_results["mi"]
            
            # Prepare base output
            output = {
                "pac": pac_values,
                "phase_frequencies": self.PHA_MIDS_HZ.detach().cpu(),
                "amplitude_frequencies": self.AMP_MIDS_HZ.detach().cpu(),
                "mi_per_segment": None,
                "amplitude_distributions": None,
                "phase_bin_centers": None,
                "phase_bin_edges": None,
            }
            
            # Add surrogate statistics if requested
            if self.n_perm is not None:
                z_scores, surrogates = self._compute_surrogates_adaptive(
                    pha, amp, pac_values, **surrogate_options
                )
                output.update({
                    "pac_z": z_scores,
                    "surrogates": surrogates,
                    "surrogate_mean": surrogates.mean(dim=2),
                    "surrogate_std": surrogates.std(dim=2),
                })
            
            return output
    
    def _compute_surrogates_adaptive(
        self, 
        pha: torch.Tensor, 
        amp: torch.Tensor, 
        observed: torch.Tensor,
        use_vectorized_surrogates: bool = False,
        use_chunked_surrogates: bool = False,
        use_sequential_surrogates: bool = False,
        chunk_size: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute surrogates using the specified strategy."""
        
        if use_vectorized_surrogates:
            surrogates = self._generate_surrogates_vectorized(pha, amp)
        elif use_chunked_surrogates:
            surrogates = self._generate_surrogates_batched(pha, amp, chunk_size)
        else:  # sequential
            surrogates = self._generate_surrogates_sequential(pha, amp)
        
        # Calculate z-scores
        mm = surrogates.mean(dim=2).to(observed.device)
        ss = surrogates.std(dim=2).to(observed.device)
        z_scores = (observed - mm) / (ss + 1e-5)
        
        return z_scores, surrogates
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage and strategy information."""
        return {
            "device_info": self.memory_manager.device_info,
            "available_memory_gb": self.memory_manager.available_memory,
            "current_strategy": getattr(self, '_last_strategy', 'not_set'),
            "profiler_results": self.profiler.get_summary_dict() if self.profiler else None
        }
    
    def estimate_memory_for_config(self, **config) -> Dict[str, float]:
        """Estimate memory usage for a given configuration."""
        return self.memory_manager.estimate_memory(**config)
Step 3: Update Surrogate Generation Methods
python# Add to PAC class - optimized surrogate methods

def _generate_surrogates_vectorized(self, pha: torch.Tensor, amp: torch.Tensor) -> torch.Tensor:
    """Fully vectorized surrogate generation - fastest but most memory intensive."""
    
    batch, n_chs, n_freqs_pha, n_segments, seq_len = pha.shape
    n_freqs_amp = amp.shape[2]
    
    # Pre-allocate shift points
    shift_points = torch.randint(seq_len, (self.n_perm,), device=pha.device)
    
    with torch.no_grad():
        # Vectorized circular shift for all permutations
        pha_shifted = self._batch_circular_shift_optimized(pha, shift_points)
        
        # Reshape for batch processing
        pha_shifted = pha_shifted.reshape(
            self.n_perm * batch, n_chs, n_freqs_pha, n_segments, seq_len
        )
        amp_expanded = amp.repeat(self.n_perm, 1, 1, 1, 1)
        
        # Single forward pass
        mi_results = self.modulation_index(pha_shifted, amp_expanded)
        
        # Reshape to final format
        surrogates = mi_results["mi"].reshape(
            self.n_perm, batch, n_chs, n_freqs_pha, n_freqs_amp
        ).permute(1, 2, 0, 3, 4)
        
        return surrogates

def _generate_surrogates_batched(
    self, pha: torch.Tensor, amp: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """Chunked surrogate generation - balanced memory/speed."""
    
    batch, n_chs, n_freqs_pha, n_segments, seq_len = pha.shape
    n_freqs_amp = amp.shape[2]
    
    # Pre-allocate output on CPU to save GPU memory
    surrogates = torch.zeros(
        (batch, n_chs, self.n_perm, n_freqs_pha, n_freqs_amp),
        dtype=pha.dtype,
        device='cpu'
    )
    
    with torch.no_grad():
        for start_idx in range(0, self.n_perm, chunk_size):
            end_idx = min(start_idx + chunk_size, self.n_perm)
            current_chunk_size = end_idx - start_idx
            
            # Generate shifts for this chunk
            shift_points = torch.randint(
                seq_len, (current_chunk_size,), device=pha.device
            )
            
            # Process chunk
            pha_chunk = self._batch_circular_shift_optimized(pha, shift_points)
            pha_chunk = pha_chunk.reshape(
                current_chunk_size * batch, n_chs, n_freqs_pha, n_segments, seq_len
            )
            amp_chunk = amp.repeat(current_chunk_size, 1, 1, 1, 1)
            
            # Compute MI for chunk
            mi_chunk = self.modulation_index(pha_chunk, amp_chunk)["mi"]
            mi_chunk = mi_chunk.reshape(
                current_chunk_size, batch, n_chs, n_freqs_pha, n_freqs_amp
            ).permute(1, 2, 0, 3, 4)
            
            # Store on CPU
            surrogates[:, :, start_idx:end_idx] = mi_chunk.cpu()
            
            # Clear GPU cache
            if pha.is_cuda:
                torch.cuda.empty_cache()
    
    return surrogates.to(pha.device)

def _generate_surrogates_sequential(self, pha: torch.Tensor, amp: torch.Tensor) -> torch.Tensor:
    """Sequential surrogate generation - most memory efficient."""
    
    batch, n_chs, n_freqs_pha, n_segments, seq_len = pha.shape
    n_freqs_amp = amp.shape[2]
    
    # Use list append strategy (proven fastest for sequential)
    surrogate_list = []
    
    with torch.no_grad():
        for perm_idx in range(self.n_perm):
            # Single permutation shift
            shift = torch.randint(seq_len, (1,), device=pha.device).item()
            pha_shifted = torch.roll(pha, shifts=-shift, dims=-1)
            
            # Compute MI
            mi_result = self.modulation_index(pha_shifted, amp)
            surrogate_list.append(mi_result["mi"])
    
    # Stack all results
    surrogates = torch.stack(surrogate_list, dim=2)
    return surrogates

def _batch_circular_shift_optimized(self, tensor: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Optimized batch circular shift operation."""
    
    batch, n_chs, n_freqs, n_segments, seq_len = tensor.shape
    n_shifts = len(shifts)
    
    # Use advanced indexing for efficient shifting
    time_indices = torch.arange(seq_len, device=tensor.device)
    shifted_indices = (time_indices.unsqueeze(0) - shifts.unsqueeze(1)) % seq_len
    
    # Expand tensor and apply shifts
    tensor_expanded = tensor.unsqueeze(0).expand(n_shifts, -1, -1, -1, -1, -1)
    
    # Efficient gather operation
    shifted = torch.gather(
        tensor_expanded,
        dim=-1,
        index=shifted_indices.view(n_shifts, 1, 1, 1, 1, seq_len).expand(
            n_shifts, batch, n_chs, n_freqs, n_segments, seq_len
        )
    )
    
    return shifted
Step 4: Usage Examples
python# Example 1: Automatic memory management
pac = PAC(
    seq_len=2048,
    fs=256,
    n_perm=1000,
    memory_strategy="auto",  # Automatically select best strategy
    enable_memory_profiling=True
)

result = pac(signal)  # Automatically uses optimal strategy

# Check what strategy was used
memory_info = pac.get_memory_info()
print(f"Used strategy: {memory_info['current_strategy']}")

# Example 2: Conservative memory usage
pac_conservative = PAC(
    seq_len=2048,
    fs=256,
    n_perm=5000,
    memory_strategy="sequential",  # Force sequential for large n_perm
    max_memory_usage=0.6  # Use only 60% of available VRAM
)

# Example 3: Aggressive memory usage (for high-end GPUs)
pac_aggressive = PAC(
    seq_len=2048,
    fs=256,
    n_perm=100,
    memory_strategy="vectorized",  # Force vectorized
    max_memory_usage=0.95  # Use 95% of available VRAM
)

# Example 4: Memory estimation before processing
memory_estimate = pac.estimate_memory_for_config(
    batch_size=32,
    n_chs=64,
    seq_len=10000,
    n_perm=1000
)
print(f"Estimated memory usage: {memory_estimate['total_mb']:.1f} MB")
Step 5: Integration Testing
python# File: tests/test_memory_integration.py

import pytest
import torch
from gpac import PAC

class TestMemoryIntegration:
    
    def test_automatic_strategy_selection(self):
        """Test that memory manager selects appropriate strategies."""
        
        # Small problem - should use vectorized
        pac_small = PAC(seq_len=512, fs=256, n_perm=10)
        signal_small = torch.randn(2, 8, 512)
        
        result = pac_small(signal_small)
        assert pac_small._last_strategy == "vectorized"
        
        # Large problem - should use chunked or sequential
        pac_large = PAC(seq_len=8192, fs=256, n_perm=1000)
        signal_large = torch.randn(16, 64, 8192)
        
        result = pac_large(signal_large)
        assert pac_large._last_strategy in ["chunked", "sequential"]
    
    def test_memory_strategy_consistency(self):
        """Test that different strategies produce consistent results."""
        
        signal = torch.randn(4, 16, 1024)
        
        # Test with different strategies
        strategies = ["vectorized", "chunked", "sequential"]
        results = {}
        
        for strategy in strategies:
            pac = PAC(
                seq_len=1024, 
                fs=256, 
                n_perm=50,
                memory_strategy=strategy
            )
            results[strategy] = pac(signal)
        
        # Results should be very similar (allowing for random differences)
        for s1 in strategies:
            for s2 in strategies:
                if s1 != s2:
                    diff = torch.abs(
                        results[s1]["pac"] - results[s2]["pac"]
                    ).mean()
                    assert diff < 0.01, f"Strategies {s1} and {s2} differ too much"
    
    def test_memory_profiling(self):
        """Test memory profiling functionality."""
        
        pac = PAC(
            seq_len=1024,
            fs=256,
            n_perm=100,
            enable_memory_profiling=True
        )
        
        signal = torch.randn(4, 16, 1024)
        result = pac(signal)
        
        memory_info = pac.get_memory_info()
        assert "profiler_results" in memory_info
        assert memory_info["profiler_results"] is not None
This integration provides:

Automatic strategy selection based on available memory
Flexible memory management with user control
Performance profiling for optimization
Consistent results across different strategies
Production-ready error handling and logging

The system automatically adapts to your 4×80GB setup and scales appropriately for smaller systems too!

<!-- EOF -->