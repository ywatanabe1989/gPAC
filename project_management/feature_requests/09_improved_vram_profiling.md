# Feature Request: Improved VRAM Profiling with PyTorch Memory Tracking

## Problem
The current profiler uses GPUtil to track GPU memory, which:
- Reports system-wide memory usage (includes other processes)
- Doesn't account for PyTorch's memory caching behavior
- Shows cumulative memory usage across profiling blocks rather than per-block usage
- Can give misleading VRAM measurements when profiling sequential operations

## Proposed Solution
Replace GPUtil memory tracking with PyTorch's native memory tracking APIs for more accurate profiling.

## Implementation Details

### 1. Use PyTorch Memory APIs
```python
def _get_pytorch_memory_stats(self):
    """Get PyTorch-specific GPU memory statistics."""
    if not (self.enable_gpu and torch.cuda.is_available()):
        return {}
    
    return {
        'vram_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
        'vram_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
        'vram_peak_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
        'vram_peak_reserved_gb': torch.cuda.max_memory_reserved() / 1024**3,
    }
```

### 2. Clear Cache Before Each Block (Optional)
```python
@contextmanager
def profile(self, name: str, clear_cache: bool = None):
    """Profile a code block with optional cache clearing."""
    clear_cache = clear_cache if clear_cache is not None else self.strict_vram
    
    if clear_cache and self.enable_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # ... rest of profiling logic
```

### 3. Add Strict Mode Option
```python
def create_profiler(enable_gpu: bool = True, strict_vram: bool = False) -> Profiler:
    """Create profiler with optional strict VRAM tracking."""
    return Profiler(enable_gpu=enable_gpu, strict_vram=strict_vram)
```

## Benefits
1. **Accurate per-block measurements** - Shows actual memory used by each operation
2. **PyTorch-aware** - Understands memory pooling and caching
3. **Detailed metrics** - Distinguishes allocated vs reserved memory
4. **Peak tracking** - Captures maximum memory usage within blocks
5. **Optional cache clearing** - Can enforce strict isolation between profiled blocks

## Example Usage
```python
# Standard profiling (with caching)
profiler = create_profiler(enable_gpu=True)

# Strict profiling (clears cache between blocks)
profiler = create_profiler(enable_gpu=True, strict_vram=True)

with profiler.profile("Operation 1"):
    result1 = model1(input1)
    # Reports actual VRAM used by Operation 1

with profiler.profile("Operation 2"):
    result2 = model2(input2)
    # Reports actual VRAM used by Operation 2 (not cumulative)
```

## Backwards Compatibility
- Keep GPUtil for general GPU metrics (utilization, temperature)
- Add PyTorch memory stats as additional fields
- Make strict mode opt-in to preserve existing behavior

## Priority
High - Current VRAM measurements can be misleading for performance optimization

## Dependencies
- PyTorch (already required)
- No new dependencies needed (can remove GPUtil dependency for memory tracking)