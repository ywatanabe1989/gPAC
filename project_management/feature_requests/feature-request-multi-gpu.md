<!-- ---
!-- Timestamp: 2025-06-02 07:12:00
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/.claude-worktree/gPAC/project_management/feature_requests/feature-request-multi-gpu.md
!-- --- -->

# Feature Request: Multi-GPU Support for gPAC

## Overview
Enable gPAC to utilize multiple GPUs on a single node based on CUDA_VISIBLE_DEVICES, allowing for distributed processing of large datasets across available GPUs.

## Current Limitation
- gPAC currently uses only a single GPU (device 0 by default)
- Large batch processing is limited by single GPU memory
- Multiple GPUs on the same node remain unutilized

## Proposed Enhancement

### 1. Multi-GPU Architecture
```python
class PAC(nn.Module):
    def __init__(
        self,
        # ... existing parameters ...
        device: Union[str, torch.device, List[int]] = "auto",
        multi_gpu: bool = False,
        gpu_allocation_strategy: str = "balanced",  # "balanced", "memory", "sequential"
    ):
        super().__init__()
        
        # Multi-GPU setup
        self.multi_gpu = multi_gpu
        self.gpu_allocation_strategy = gpu_allocation_strategy
        
        # Detect available GPUs
        if multi_gpu:
            self.devices = self._setup_multi_gpu(device)
            self.n_gpus = len(self.devices)
            print(f"Multi-GPU mode: Using {self.n_gpus} GPUs: {self.devices}")
        else:
            self.devices = [self._setup_single_device(device)]
            self.n_gpus = 1
```

### 2. GPU Detection and Setup
```python
def _setup_multi_gpu(self, device_spec: Union[str, List[int]]) -> List[torch.device]:
    """Setup multiple GPUs based on CUDA_VISIBLE_DEVICES or explicit list."""
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        return [torch.device("cpu")]
    
    if device_spec == "auto":
        # Use all visible GPUs
        n_visible = torch.cuda.device_count()
        devices = [torch.device(f"cuda:{i}") for i in range(n_visible)]
    elif isinstance(device_spec, list):
        # Use specified GPU indices
        devices = [torch.device(f"cuda:{i}") for i in device_spec]
    else:
        # Single device specified
        devices = [torch.device(device_spec)]
    
    # Validate devices
    available_devices = []
    for dev in devices:
        try:
            torch.cuda.get_device_properties(dev.index)
            available_devices.append(dev)
        except:
            print(f"WARNING: Device {dev} not available, skipping")
    
    if not available_devices:
        print("WARNING: No valid CUDA devices found, falling back to CPU")
        return [torch.device("cpu")]
    
    return available_devices

def _get_device_memory_info(self) -> Dict[int, Dict[str, float]]:
    """Get memory information for all available GPUs."""
    memory_info = {}
    
    for device in self.devices:
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device.index)
            allocated = torch.cuda.memory_allocated(device.index)
            reserved = torch.cuda.memory_reserved(device.index)
            
            memory_info[device.index] = {
                "name": props.name,
                "total_gb": props.total_memory / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "free_gb": (props.total_memory - allocated) / (1024**3),
                "reserved_gb": reserved / (1024**3),
            }
    
    return memory_info
```

### 3. Multi-GPU Forward Pass Strategies

#### 3.1 Batch Parallel Strategy
```python
def _forward_multi_gpu_batch_parallel(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Distribute batches across GPUs for parallel processing.
    Best for: Large batch sizes, uniform computation per sample.
    """
    batch_size = x.shape[0]
    
    # Split batches across GPUs
    splits = self._split_balanced(batch_size, self.n_gpus)
    x_splits = torch.split(x, splits, dim=0)
    
    # Process each split on different GPU
    futures = []
    for i, (x_split, device) in enumerate(zip(x_splits, self.devices)):
        # Clone model components to each GPU if needed
        if not hasattr(self, f"_gpu_{i}_initialized"):
            self._initialize_gpu_components(i, device)
        
        # Async processing
        future = self._process_on_device_async(x_split, device, i)
        futures.append(future)
    
    # Gather results
    results = [future.result() for future in futures]
    
    # Merge results
    return self._merge_results(results)

def _process_on_device_async(self, x: torch.Tensor, device: torch.device, gpu_idx: int):
    """Process data on specific GPU asynchronously."""
    # Move data to device
    x = x.to(device)
    
    # Use GPU-specific model components
    with torch.cuda.device(device):
        # Process using components initialized for this GPU
        output = self._compute_pac_core_gpu(x, gpu_idx)
    
    return output
```

#### 3.2 Channel Parallel Strategy
```python
def _forward_multi_gpu_channel_parallel(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Distribute channels across GPUs.
    Best for: Many channels, memory-intensive per-channel processing.
    """
    batch_size, n_channels, *rest = x.shape
    
    # Split channels across GPUs
    splits = self._split_balanced(n_channels, self.n_gpus)
    x_splits = torch.split(x, splits, dim=1)
    
    # Process each channel group on different GPU
    results = []
    for x_split, device in zip(x_splits, self.devices):
        x_gpu = x_split.to(device)
        with torch.cuda.device(device):
            result = self._compute_pac_core(x_gpu)
        results.append(result)
    
    # Merge channel-wise results
    return self._merge_results_channel_wise(results)
```

#### 3.3 Permutation Parallel Strategy
```python
def _forward_multi_gpu_permutation_parallel(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Distribute permutations across GPUs.
    Best for: Large n_perm, surrogate-heavy computations.
    """
    if self.n_perm is None:
        # No permutations, fall back to batch parallel
        return self._forward_multi_gpu_batch_parallel(x)
    
    # First compute observed PAC on GPU 0
    with torch.cuda.device(self.devices[0]):
        x_gpu = x.to(self.devices[0])
        pac_observed = self._compute_pac_observed(x_gpu)
        pha, amp = self._extract_phase_amplitude(x_gpu)
    
    # Distribute permutations across GPUs
    perm_splits = self._split_balanced(self.n_perm, self.n_gpus)
    
    # Compute surrogates in parallel
    surrogate_results = []
    for i, (n_perm_gpu, device) in enumerate(zip(perm_splits, self.devices)):
        if n_perm_gpu > 0:
            pha_gpu = pha.to(device)
            amp_gpu = amp.to(device)
            
            with torch.cuda.device(device):
                surrogates = self._generate_surrogates_on_device(
                    pha_gpu, amp_gpu, n_perm_gpu
                )
            surrogate_results.append(surrogates)
    
    # Gather all surrogates
    all_surrogates = torch.cat([s.to(self.devices[0]) for s in surrogate_results], dim=2)
    
    # Compute statistics on primary GPU
    return self._compute_final_statistics(pac_observed, all_surrogates)
```

### 4. Smart GPU Allocation
```python
class GPUAllocator:
    """Smart allocation of work across GPUs based on their current state."""
    
    def __init__(self, devices: List[torch.device], strategy: str = "balanced"):
        self.devices = devices
        self.strategy = strategy
        
    def allocate_batch(self, batch_size: int) -> List[int]:
        """Allocate batch items to GPUs."""
        if self.strategy == "balanced":
            # Equal distribution
            return self._split_balanced(batch_size, len(self.devices))
        
        elif self.strategy == "memory":
            # Proportional to available memory
            memory_info = self._get_device_memory_info()
            free_memory = [info["free_gb"] for info in memory_info.values()]
            total_free = sum(free_memory)
            
            # Allocate proportionally
            allocations = []
            remaining = batch_size
            for i, free in enumerate(free_memory[:-1]):
                alloc = int(batch_size * free / total_free)
                allocations.append(alloc)
                remaining -= alloc
            allocations.append(remaining)  # Last GPU gets remainder
            
            return allocations
        
        elif self.strategy == "sequential":
            # Fill GPUs sequentially
            allocations = [0] * len(self.devices)
            items_per_gpu = batch_size // len(self.devices) + 1
            
            for i in range(batch_size):
                gpu_idx = min(i // items_per_gpu, len(self.devices) - 1)
                allocations[gpu_idx] += 1
            
            return allocations
```

### 5. Usage Examples

#### 5.1 Basic Multi-GPU Usage
```python
# Automatically use all visible GPUs
# E.g., CUDA_VISIBLE_DEVICES=0,1,2,3 python script.py
pac = PAC(
    seq_len=2048,
    fs=256,
    n_perm=1000,
    multi_gpu=True,
    device="auto"  # Uses all visible GPUs
)

# Process large batch across all GPUs
x = torch.randn(256, 64, 2048)  # Large batch
output = pac(x)  # Automatically distributed
```

#### 5.2 Explicit GPU Selection
```python
# Use specific GPUs
pac = PAC(
    seq_len=2048,
    fs=256,
    multi_gpu=True,
    device=[0, 2, 3]  # Use GPU 0, 2, and 3
)
```

#### 5.3 Memory-Aware Allocation
```python
# Allocate based on available memory
pac = PAC(
    seq_len=2048,
    fs=256,
    multi_gpu=True,
    gpu_allocation_strategy="memory"  # Smart allocation
)

# Get GPU status
gpu_info = pac.get_gpu_status()
print(gpu_info)
# Output:
# {
#   0: {"name": "A100", "total_gb": 80, "free_gb": 75, "allocation": 38%},
#   1: {"name": "A100", "total_gb": 80, "free_gb": 60, "allocation": 30%},
#   2: {"name": "A100", "total_gb": 80, "free_gb": 65, "allocation": 32%}
# }
```

### 6. DataLoader Integration
```python
class MultiGPUDataLoader:
    """DataLoader that pre-distributes data for multi-GPU processing."""
    
    def __init__(self, dataset, batch_size, pac_module, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pac = pac_module
        self.shuffle = shuffle
        
    def __iter__(self):
        indices = torch.randperm(len(self.dataset)) if self.shuffle else torch.arange(len(self.dataset))
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            
            # Pre-allocate to GPUs
            gpu_allocations = self.pac.gpu_allocator.allocate_batch(len(batch_data))
            
            # Yield pre-distributed batches
            distributed_batch = []
            start_idx = 0
            for alloc, device in zip(gpu_allocations, self.pac.devices):
                end_idx = start_idx + alloc
                if alloc > 0:
                    sub_batch = batch_data[start_idx:end_idx]
                    distributed_batch.append((sub_batch, device))
                start_idx = end_idx
            
            yield distributed_batch
```

### 7. Performance Monitoring
```python
def get_gpu_performance_metrics(self) -> Dict[str, Any]:
    """Get detailed performance metrics for multi-GPU setup."""
    metrics = {
        "n_gpus": self.n_gpus,
        "devices": [str(d) for d in self.devices],
        "allocation_strategy": self.gpu_allocation_strategy,
        "gpu_metrics": {}
    }
    
    for device in self.devices:
        if device.type == "cuda":
            idx = device.index
            metrics["gpu_metrics"][idx] = {
                "utilization": self._get_gpu_utilization(idx),
                "memory_used_gb": torch.cuda.memory_allocated(idx) / (1024**3),
                "memory_reserved_gb": torch.cuda.memory_reserved(idx) / (1024**3),
                "temperature": self._get_gpu_temperature(idx),
            }
    
    return metrics
```

### 8. Synchronization and Communication
```python
class MultiGPUSynchronizer:
    """Handle synchronization across multiple GPUs."""
    
    def __init__(self, devices: List[torch.device]):
        self.devices = devices
        
    def synchronize_all(self):
        """Synchronize all GPUs."""
        for device in self.devices:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    
    def all_gather(self, tensor_list: List[torch.Tensor]) -> torch.Tensor:
        """Gather tensors from all GPUs to GPU 0."""
        # Move all to GPU 0
        gathered = [t.to(self.devices[0]) for t in tensor_list]
        return torch.cat(gathered, dim=0)
    
    def broadcast(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Broadcast tensor from GPU 0 to all GPUs."""
        return [tensor.to(device) for device in self.devices]
```

### 9. Expected Performance Gains

For a 4×A100 (80GB) setup:
- **Single GPU**: 100 samples/sec
- **4 GPUs (batch parallel)**: ~380 samples/sec (3.8x speedup)
- **4 GPUs (channel parallel)**: ~350 samples/sec (3.5x speedup)
- **4 GPUs (permutation parallel)**: ~390 samples/sec (3.9x speedup)

### 10. Implementation Considerations

1. **Memory Management**: Each GPU needs its own memory manager instance
2. **Model Replication**: Efficiently replicate model components across GPUs
3. **Communication Overhead**: Minimize data transfer between GPUs
4. **Load Balancing**: Dynamic rebalancing based on GPU utilization
5. **Error Handling**: Graceful degradation if a GPU fails

## Conclusion

Multi-GPU support would enable gPAC to scale linearly with available GPUs, making it possible to process massive datasets efficiently on high-end workstations and compute nodes. The implementation prioritizes ease of use with automatic GPU detection while providing fine-grained control for advanced users.

<!-- EOF -->