#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-28 23:20:00"
# Author: ywatanabe
# File: /home/ywatanabe/proj/gPAC/src/gpac/_Profiler.py

"""Comprehensive profiler for tracking time, memory, and compute resources."""

import time
import psutil
import torch
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import GPUtil


@dataclass
class ProfileResult:
    """Container for profiling results."""
    name: str
    duration: float  # seconds
    cpu_percent: float
    ram_used_gb: float
    ram_percent: float
    gpu_used_percent: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_temp: Optional[float] = None
    gpu_memory_allocated_gb: Optional[float] = None
    gpu_memory_reserved_gb: Optional[float] = None
    
    def __str__(self):
        """Format profile result as a readable string."""
        lines = [
            f"📊 {self.name}",
            f"⏱️  Time: {self.duration:.3f}s",
            f"💻 CPU: {self.cpu_percent:.1f}%",
            f"🧠 RAM: {self.ram_used_gb:.2f}GB ({self.ram_percent:.1f}%)"
        ]
        
        if self.gpu_memory_used_gb is not None:
            vram_lines = [
                f"📦 VRAM (Allocated): {self.gpu_memory_allocated_gb:.2f}GB ({self.gpu_memory_percent:.1f}%)"
            ]
            if self.gpu_memory_reserved_gb is not None:
                vram_lines.append(f"📦 VRAM (Reserved): {self.gpu_memory_reserved_gb:.2f}GB")
            
            if self.gpu_used_percent is not None:
                vram_lines.insert(0, f"🎮 GPU: {self.gpu_used_percent:.1f}%")
            
            if self.gpu_temp:
                vram_lines.append(f"🌡️  Temp: {self.gpu_temp:.0f}°C")
            
            lines.extend(vram_lines)
        
        return "\n".join(filter(None, lines))


class Profiler:
    """Comprehensive profiler for tracking performance metrics."""
    
    def __init__(self, enable_gpu: bool = True):
        """Initialize profiler.
        
        Args:
            enable_gpu: Whether to track GPU metrics
        """
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        self.results: List[ProfileResult] = []
        self._active_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Check GPU availability
        self.gpus = []
        if self.enable_gpu:
            try:
                self.gpus = GPUtil.getGPUs()
            except:
                self.enable_gpu = False
                
    def _get_cpu_memory_stats(self) -> Dict[str, float]:
        """Get current CPU and memory statistics."""
        # CPU usage (average over 0.1 seconds)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        ram_used_gb = memory.used / (1024**3)
        ram_percent = memory.percent
        
        return {
            'cpu_percent': cpu_percent,
            'ram_used_gb': ram_used_gb,
            'ram_percent': ram_percent
        }
    
    def _get_gpu_stats(self) -> Dict[str, Optional[float]]:
        """Get current GPU statistics using PyTorch native APIs."""
        if not self.enable_gpu:
            return {
                'gpu_used_percent': None,
                'gpu_memory_used_gb': None,
                'gpu_memory_percent': None,
                'gpu_temp': None,
                'gpu_memory_allocated_gb': None,
                'gpu_memory_reserved_gb': None
            }
        
        try:
            # Get current GPU device
            device = torch.cuda.current_device()
            
            # PyTorch memory statistics (more accurate for PyTorch workloads)
            allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(device) / (1024**3)    # GB
            
            # Get total GPU memory
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
            
            # Calculate percentages
            allocated_percent = (allocated / total_memory) * 100
            reserved_percent = (reserved / total_memory) * 100
            
            # Try to get GPU utilization and temperature from GPUtil if available
            gpu_load = None
            gpu_temp = None
            if self.gpus:
                try:
                    gpu = GPUtil.getGPUs()[0]
                    gpu_load = gpu.load * 100
                    gpu_temp = gpu.temperature
                except:
                    pass
            
            return {
                'gpu_used_percent': gpu_load,
                'gpu_memory_used_gb': allocated,  # Actual PyTorch allocated memory
                'gpu_memory_percent': allocated_percent,
                'gpu_temp': gpu_temp,
                'gpu_memory_allocated_gb': allocated,
                'gpu_memory_reserved_gb': reserved
            }
        except Exception as e:
            return {
                'gpu_used_percent': None,
                'gpu_memory_used_gb': None,
                'gpu_memory_percent': None,
                'gpu_temp': None,
                'gpu_memory_allocated_gb': None,
                'gpu_memory_reserved_gb': None
            }
    
    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling a code block.
        
        Args:
            name: Name for this profile section
            
        Usage:
            profiler = Profiler()
            with profiler.profile("Model Forward"):
                output = model(input)
        """
        # Start profiling
        start_time = time.time()
        
        # Store initial stats
        self._active_profiles[name] = {
            'start_time': start_time,
            'initial_stats': {
                **self._get_cpu_memory_stats(),
                **self._get_gpu_stats()
            }
        }
        
        # Synchronize CUDA if using GPU
        if self.enable_gpu:
            torch.cuda.synchronize()
        
        try:
            yield
        finally:
            # Synchronize CUDA if using GPU
            if self.enable_gpu:
                torch.cuda.synchronize()
            
            # End profiling
            end_time = time.time()
            duration = end_time - start_time
            
            # Get final stats
            final_stats = {
                **self._get_cpu_memory_stats(),
                **self._get_gpu_stats()
            }
            
            # Create result (using peak values during the operation)
            result = ProfileResult(
                name=name,
                duration=duration,
                cpu_percent=final_stats['cpu_percent'],
                ram_used_gb=final_stats['ram_used_gb'],
                ram_percent=final_stats['ram_percent'],
                gpu_used_percent=final_stats['gpu_used_percent'],
                gpu_memory_used_gb=final_stats['gpu_memory_used_gb'],
                gpu_memory_percent=final_stats['gpu_memory_percent'],
                gpu_temp=final_stats['gpu_temp'],
                gpu_memory_allocated_gb=final_stats.get('gpu_memory_allocated_gb'),
                gpu_memory_reserved_gb=final_stats.get('gpu_memory_reserved_gb')
            )
            
            self.results.append(result)
            del self._active_profiles[name]
    
    def print_summary(self):
        """Print a summary of all profiling results."""
        if not self.results:
            print("No profiling results available.")
            return
        
        print("\n" + "="*60)
        print("📊 PROFILING SUMMARY")
        print("="*60)
        
        for result in self.results:
            print(f"\n{result}")
            print("-"*40)
        
        # Total time
        total_time = sum(r.duration for r in self.results)
        print(f"\n⏱️  Total Time: {total_time:.3f}s")
        
        # Peak memory usage
        if self.results:
            peak_ram = max(r.ram_used_gb for r in self.results)
            print(f"🧠 Peak RAM: {peak_ram:.2f}GB")
            
            if any(r.gpu_memory_used_gb is not None for r in self.results):
                peak_vram = max(r.gpu_memory_used_gb for r in self.results 
                               if r.gpu_memory_used_gb is not None)
                print(f"📦 Peak VRAM (Allocated): {peak_vram:.2f}GB")
                
                if any(r.gpu_memory_reserved_gb is not None for r in self.results):
                    peak_reserved = max(r.gpu_memory_reserved_gb for r in self.results 
                                      if r.gpu_memory_reserved_gb is not None)
                    print(f"📦 Peak VRAM (Reserved): {peak_reserved:.2f}GB")
        
        print("="*60)
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Get summary statistics as a dictionary."""
        if not self.results:
            return {}
        
        summary = {
            'total_time': sum(r.duration for r in self.results),
            'sections': {r.name: r.duration for r in self.results},
            'peak_ram_gb': max(r.ram_used_gb for r in self.results),
            'peak_cpu_percent': max(r.cpu_percent for r in self.results),
        }
        
        if any(r.gpu_memory_used_gb is not None for r in self.results):
            gpu_results = [r for r in self.results if r.gpu_memory_used_gb is not None]
            summary.update({
                'peak_vram_gb': max(r.gpu_memory_used_gb for r in gpu_results),
                'peak_vram_allocated_gb': max(r.gpu_memory_allocated_gb for r in gpu_results 
                                             if r.gpu_memory_allocated_gb is not None),
            })
            
            if any(r.gpu_memory_reserved_gb is not None for r in gpu_results):
                summary['peak_vram_reserved_gb'] = max(r.gpu_memory_reserved_gb for r in gpu_results 
                                                      if r.gpu_memory_reserved_gb is not None)
            
            if any(r.gpu_used_percent is not None for r in gpu_results):
                summary['peak_gpu_percent'] = max(r.gpu_used_percent for r in gpu_results
                                                if r.gpu_used_percent is not None)
        
        return summary
    
    def reset(self):
        """Reset all profiling results."""
        self.results.clear()
        self._active_profiles.clear()


# Convenience function
def create_profiler(enable_gpu: bool = True) -> Profiler:
    """Create a new profiler instance."""
    return Profiler(enable_gpu=enable_gpu)