<!-- ---
!-- Timestamp: 2025-06-02 08:20:00
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/.claude-worktree/gPAC/project_management/feature_requests/feature-request-simple-multi-gpu-batch.md
!-- --- -->

# Feature Request: Simple Multi-GPU Batch Parallelism

## Overview
Implement straightforward multi-GPU support by distributing batches across available GPUs. This is the simplest and most effective approach for immediate performance gains.

## Core Concept
Split the batch dimension across GPUs, process in parallel, then gather results. No complex synchronization needed.

## Implementation Design

### 1. Simple Multi-GPU PAC
```python
class PAC(nn.Module):
    def __init__(
        self,
        # ... existing parameters ...
        multi_gpu: bool = False,
        device_ids: Optional[List[int]] = None,  # e.g., [0, 1, 2, 3]
    ):
        super().__init__()
        
        # ... existing init ...
        
        # Multi-GPU setup
        self.multi_gpu = multi_gpu and torch.cuda.device_count() > 1
        
        if self.multi_gpu:
            if device_ids is None:
                # Use all visible GPUs
                self.device_ids = list(range(torch.cuda.device_count()))
            else:
                self.device_ids = device_ids
            
            print(f"Multi-GPU enabled: Using GPUs {self.device_ids}")
            
            # Replicate model components to each GPU
            self._setup_multi_gpu()
        else:
            self.device_ids = [0]  # Default to first GPU
```

### 2. Batch Distribution Forward Pass
```python
def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Forward pass with optional multi-GPU batch distribution."""
    
    if not self.multi_gpu or len(self.device_ids) == 1:
        # Single GPU path (existing code)
        return self._forward_single_gpu(x)
    
    # Multi-GPU batch parallel
    return self._forward_multi_gpu_batch_parallel(x)

def _forward_multi_gpu_batch_parallel(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Distribute batches across GPUs and process in parallel."""
    
    batch_size = x.shape[0]
    n_gpus = len(self.device_ids)
    
    # Split batch across GPUs
    if batch_size < n_gpus:
        # If batch smaller than GPUs, use subset
        active_gpus = self.device_ids[:batch_size]
        splits = [1] * batch_size
    else:
        # Distribute evenly
        active_gpus = self.device_ids
        base_size = batch_size // n_gpus
        remainder = batch_size % n_gpus
        splits = [base_size + (1 if i < remainder else 0) for i in range(n_gpus)]
    
    # Split input
    x_chunks = torch.split(x, splits, dim=0)
    
    # Process each chunk on its GPU
    futures = []
    for gpu_id, x_chunk in zip(active_gpus, x_chunks):
        # Async processing on each GPU
        future = torch.jit.fork(self._process_on_gpu, x_chunk, gpu_id)
        futures.append(future)
    
    # Wait and gather results
    results = [torch.jit.wait(f) for f in futures]
    
    # Merge results back
    return self._merge_gpu_results(results)
```

### 3. GPU-Specific Processing
```python
def _process_on_gpu(self, x_chunk: torch.Tensor, gpu_id: int) -> Dict[str, torch.Tensor]:
    """Process a batch chunk on specific GPU."""
    
    # Move to target GPU
    device = torch.device(f'cuda:{gpu_id}')
    x_chunk = x_chunk.to(device)
    
    # Use existing forward logic
    with torch.cuda.device(device):
        # This respects current memory strategy settings
        return self._forward_single_gpu(x_chunk)

def _merge_gpu_results(self, results: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Merge results from multiple GPUs back to primary GPU."""
    
    if not results:
        return {}
    
    # Move all results to GPU 0
    merged = {}
    
    # Merge tensor outputs
    for key in ['pac', 'pac_z', 'surrogates', 'surrogate_mean', 'surrogate_std']:
        if key in results[0]:
            # Gather all chunks to GPU 0
            chunks = [r[key].to('cuda:0') for r in results]
            merged[key] = torch.cat(chunks, dim=0)
    
    # Copy non-tensor outputs
    for key in ['phase_frequencies', 'amplitude_frequencies', 
                'mi_per_segment', 'amplitude_distributions',
                'phase_bin_centers', 'phase_bin_edges']:
        if key in results[0]:
            merged[key] = results[0][key]
    
    return merged
```

### 4. Simple Setup Helper
```python
def _setup_multi_gpu(self):
    """Replicate necessary components across GPUs."""
    
    # Components that need replication
    components_to_replicate = [
        'bandpass',
        'hilbert', 
        'modulation_index'
    ]
    
    for gpu_id in self.device_ids[1:]:  # Skip first GPU (original)
        for comp_name in components_to_replicate:
            if hasattr(self, comp_name):
                original = getattr(self, comp_name)
                # Create a copy on the target GPU
                setattr(self, f'{comp_name}_gpu{gpu_id}', 
                       copy.deepcopy(original).to(f'cuda:{gpu_id}'))
```

### 5. Usage Examples

#### Basic Multi-GPU
```python
# Automatically use all visible GPUs
pac = PAC(
    seq_len=2048,
    fs=256,
    n_perm=1000,
    multi_gpu=True  # That's it!
)

# Process large batch - automatically distributed
x = torch.randn(256, 64, 2048)  # 256 samples split across GPUs
result = pac(x)
```

#### Specific GPUs
```python
# Use specific GPUs
pac = PAC(
    seq_len=2048,
    fs=256,
    n_perm=1000,
    multi_gpu=True,
    device_ids=[0, 2, 3]  # Use GPU 0, 2, and 3
)
```

#### With Memory Strategy
```python
# Multi-GPU + Smart memory management
pac = PAC(
    seq_len=2048,
    fs=256,
    n_perm=1000,
    multi_gpu=True,
    memory_strategy="auto",  # Each GPU uses smart memory management
    vram_gb=40.0  # Per-GPU memory limit
)
```

### 6. Environment Variables
```bash
# Use specific GPUs via CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1,2,3 python script.py

# Or programmatically
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
```

### 7. Performance Expectations

For batch parallel on 4x A100 GPUs:
- **Theoretical**: Up to 4x speedup
- **Practical**: 3.5-3.8x speedup (accounting for overhead)
- **Memory**: Each GPU handles 1/4 of the batch

Example performance:
```
Single GPU: 100 samples/sec
4 GPUs: ~350-380 samples/sec
```

### 8. Integration with Current Features

The multi-GPU system works seamlessly with:
- **Memory strategies**: Each GPU respects the memory strategy
- **Dimensional control**: Applied independently on each GPU
- **Profiling**: Aggregate profiling across GPUs

### 9. Future Enhancements

Start simple, then add:
1. **Dynamic load balancing**: Adjust splits based on GPU utilization
2. **Pipeline parallelism**: Different stages on different GPUs
3. **Gradient accumulation**: For even larger effective batch sizes

### 10. Why This Approach First?

1. **Simplest to implement**: Batch splitting is straightforward
2. **Most common use case**: Users typically want to process more samples
3. **No communication overhead**: Each batch is independent
4. **Works with existing code**: Minimal changes to current architecture
5. **Immediate benefit**: Linear speedup with number of GPUs

## Implementation Priority

This should be the first multi-GPU feature because:
- Easiest to understand and use
- Provides immediate performance gains
- Doesn't complicate the codebase
- Natural extension of current single-GPU design

## Testing

```python
def test_multi_gpu_consistency():
    """Ensure multi-GPU produces same results as single GPU."""
    
    x = torch.randn(64, 32, 1024)
    
    # Single GPU
    pac_single = PAC(seq_len=1024, fs=256, multi_gpu=False)
    result_single = pac_single(x)
    
    # Multi GPU
    pac_multi = PAC(seq_len=1024, fs=256, multi_gpu=True)
    result_multi = pac_multi(x)
    
    # Results should be identical
    assert torch.allclose(result_single['pac'], result_multi['pac'])
```

<!-- EOF -->