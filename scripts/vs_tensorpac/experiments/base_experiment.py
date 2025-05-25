"""
Base Experiment Class for gPAC vs Tensorpac Comparisons

This module provides the abstract base class for all comparison experiments,
ensuring consistent methodology and rigorous scientific practices.
"""

import abc
import time
import psutil
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import yaml

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Container for individual benchmark results."""
    experiment_name: str
    library: str                    # 'gpac' or 'tensorpac'
    method: str                     # 'api' or 'class'
    device: str                     # 'cpu' or 'cuda'
    parameters: Dict[str, Any]
    timing_ms: float               # Wall-clock time in milliseconds
    memory_mb: float               # Peak memory usage in MB
    gpu_memory_mb: Optional[float] # Peak GPU memory usage in MB
    success: bool                  # Whether benchmark completed successfully
    error_message: Optional[str]   # Error details if failed
    metadata: Dict[str, Any]       # Additional experiment-specific data


@dataclass
class SystemInfo:
    """Container for system hardware/software information."""
    cpu_model: str
    cpu_cores: int
    ram_gb: float
    gpu_model: Optional[str]
    gpu_memory_gb: Optional[float]
    python_version: str
    torch_version: str
    cuda_version: Optional[str]
    gpac_version: str
    tensorpac_version: str


class BaseExperiment(abc.ABC):
    """
    Abstract base class for all gPAC vs Tensorpac comparison experiments.
    
    This class provides:
    - Consistent methodology across experiments
    - Hardware monitoring capabilities
    - Statistical rigor enforcement
    - Result logging and validation
    - Error handling and recovery
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize base experiment.
        
        Args:
            config: Experiment configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or self._setup_logger()
        self.system_info = self._collect_system_info()
        self.results: List[BenchmarkResult] = []
        
        # Initialize hardware monitoring
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_available = True
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.gpu_available = False
                self.gpu_handle = None
        else:
            self.gpu_available = False
            self.gpu_handle = None
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(getattr(logging, self.config.get('output', {}).get('log_level', 'INFO')))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _collect_system_info(self) -> SystemInfo:
        """Collect system hardware and software information."""
        import platform
        import sys
        
        # CPU information
        cpu_model = platform.processor() or "Unknown"
        cpu_cores = psutil.cpu_count()
        
        # Memory information  
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # GPU information
        gpu_model = None
        gpu_memory_gb = None
        cuda_version = None
        
        if self.gpu_available and self.gpu_handle:
            try:
                gpu_model = pynvml.nvmlDeviceGetName(self.gpu_handle).decode('utf-8')
                gpu_memory_bytes = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).total
                gpu_memory_gb = gpu_memory_bytes / (1024**3)
            except:
                pass
        
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
        
        # Software versions
        python_version = sys.version.split()[0]
        torch_version = torch.__version__
        
        try:
            import gpac
            gpac_version = gpac.__version__
        except:
            gpac_version = "Unknown"
        
        try:
            import tensorpac
            tensorpac_version = tensorpac.__version__
        except:
            tensorpac_version = "Unknown"
        
        return SystemInfo(
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            gpu_model=gpu_model,
            gpu_memory_gb=gpu_memory_gb,
            python_version=python_version,
            torch_version=torch_version,
            cuda_version=cuda_version,
            gpac_version=gpac_version,
            tensorpac_version=tensorpac_version
        )
    
    def _measure_memory_usage(self, device: str = "cpu") -> Tuple[float, Optional[float]]:
        """
        Measure current memory usage.
        
        Args:
            device: Device to measure ('cpu' or 'cuda')
            
        Returns:
            Tuple of (cpu_memory_mb, gpu_memory_mb)
        """
        # CPU memory
        process = psutil.Process()
        cpu_memory_mb = process.memory_info().rss / (1024**2)
        
        # GPU memory
        gpu_memory_mb = None
        if device == "cuda" and self.gpu_available and self.gpu_handle:
            try:
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_memory_mb = gpu_info.used / (1024**2)
            except:
                pass
        
        return cpu_memory_mb, gpu_memory_mb
    
    def _reset_memory_stats(self, device: str = "cpu"):
        """Reset memory statistics for clean measurement."""
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    def _run_single_benchmark(
        self,
        benchmark_func: callable,
        experiment_name: str,
        library: str,
        method: str,
        device: str,
        parameters: Dict[str, Any]
    ) -> BenchmarkResult:
        """
        Run a single benchmark with proper timing and monitoring.
        
        Args:
            benchmark_func: Function to benchmark
            experiment_name: Name of the experiment
            library: Library being tested ('gpac' or 'tensorpac')
            method: Method type ('api' or 'class')
            device: Device ('cpu' or 'cuda')
            parameters: Benchmark parameters
            
        Returns:
            BenchmarkResult containing timing and memory data
        """
        self.logger.debug(f"Running {experiment_name} - {library} ({method}) on {device}")
        
        # Reset memory statistics
        self._reset_memory_stats(device)
        
        # Measure baseline memory
        baseline_cpu_mem, baseline_gpu_mem = self._measure_memory_usage(device)
        
        # Run benchmark with timing
        start_time = time.perf_counter()
        success = True
        error_message = None
        metadata = {}
        
        try:
            # Execute benchmark function
            result = benchmark_func(device=device, **parameters)
            if isinstance(result, dict):
                metadata.update(result)
        except Exception as e:
            success = False
            error_message = str(e)
            self.logger.error(f"Benchmark failed: {e}")
            self.logger.debug(traceback.format_exc())
        
        end_time = time.perf_counter()
        timing_ms = (end_time - start_time) * 1000
        
        # Measure peak memory usage
        peak_cpu_mem, peak_gpu_mem = self._measure_memory_usage(device)
        memory_mb = max(0, peak_cpu_mem - baseline_cpu_mem)
        gpu_memory_mb = None
        
        if peak_gpu_mem is not None and baseline_gpu_mem is not None:
            gpu_memory_mb = max(0, peak_gpu_mem - baseline_gpu_mem)
        
        return BenchmarkResult(
            experiment_name=experiment_name,
            library=library,
            method=method,
            device=device,
            parameters=parameters.copy(),
            timing_ms=timing_ms,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            success=success,
            error_message=error_message,
            metadata=metadata
        )
    
    def _run_repeated_benchmark(
        self,
        benchmark_func: callable,
        experiment_name: str,
        library: str,
        method: str,
        device: str,
        parameters: Dict[str, Any],
        n_repetitions: Optional[int] = None
    ) -> List[BenchmarkResult]:
        """
        Run benchmark multiple times for statistical power.
        
        Args:
            benchmark_func: Function to benchmark
            experiment_name: Name of the experiment  
            library: Library being tested
            method: Method type
            device: Device
            parameters: Benchmark parameters
            n_repetitions: Number of repetitions (uses config default if None)
            
        Returns:
            List of BenchmarkResult instances
        """
        if n_repetitions is None:
            n_repetitions = self.config['experimental']['n_repetitions']
        
        results = []
        for i in range(n_repetitions):
            self.logger.debug(f"Repetition {i+1}/{n_repetitions}")
            result = self._run_single_benchmark(
                benchmark_func, experiment_name, library, method, device, parameters
            )
            results.append(result)
            
            # Brief pause between repetitions to allow system stabilization
            time.sleep(0.1)
        
        return results
    
    @abc.abstractmethod
    def run_experiments(self, devices: Optional[List[str]] = None) -> List[BenchmarkResult]:
        """
        Run all experiments in this category.
        
        Args:
            devices: List of devices to test on (uses config default if None)
            
        Returns:
            List of all benchmark results
        """
        pass
    
    @abc.abstractmethod
    def get_experiment_description(self) -> str:
        """
        Return a description of what this experiment tests.
        
        Returns:
            Human-readable description of the experiment
        """
        pass
    
    def validate_results(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """
        Validate experimental results for quality and completeness.
        
        Args:
            results: List of benchmark results
            
        Returns:
            Dictionary containing validation summary
        """
        total_results = len(results)
        successful_results = sum(1 for r in results if r.success)
        failed_results = total_results - successful_results
        
        # Calculate coefficient of variation for timing stability
        successful_timings = [r.timing_ms for r in results if r.success]
        cv_timing = np.std(successful_timings) / np.mean(successful_timings) if successful_timings else float('inf')
        
        validation_summary = {
            'total_results': total_results,
            'successful_results': successful_results,
            'failed_results': failed_results,
            'success_rate': successful_results / total_results if total_results > 0 else 0,
            'timing_cv': cv_timing,
            'timing_stable': cv_timing < 0.3,  # CV < 30% considered stable
            'min_timing_ms': min(successful_timings) if successful_timings else None,
            'max_timing_ms': max(successful_timings) if successful_timings else None,
            'mean_timing_ms': np.mean(successful_timings) if successful_timings else None
        }
        
        return validation_summary
    
    def save_results(self, results: List[BenchmarkResult], output_path: str):
        """
        Save experimental results to file.
        
        Args:
            results: List of benchmark results
            output_path: Path to save results
        """
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = {
                'experiment_name': result.experiment_name,
                'library': result.library,
                'method': result.method,
                'device': result.device,
                'parameters': result.parameters,
                'timing_ms': result.timing_ms,
                'memory_mb': result.memory_mb,
                'gpu_memory_mb': result.gpu_memory_mb,
                'success': result.success,
                'error_message': result.error_message,
                'metadata': result.metadata
            }
            serializable_results.append(result_dict)
        
        # Add system information
        output_data = {
            'system_info': {
                'cpu_model': self.system_info.cpu_model,
                'cpu_cores': self.system_info.cpu_cores,
                'ram_gb': self.system_info.ram_gb,
                'gpu_model': self.system_info.gpu_model,
                'gpu_memory_gb': self.system_info.gpu_memory_gb,
                'python_version': self.system_info.python_version,
                'torch_version': self.system_info.torch_version,
                'cuda_version': self.system_info.cuda_version,
                'gpac_version': self.system_info.gpac_version,
                'tensorpac_version': self.system_info.tensorpac_version
            },
            'config': self.config,
            'results': serializable_results,
            'validation': self.validate_results(results)
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            yaml.safe_dump(output_data, f, default_flow_style=False)
        
        self.logger.info(f"Results saved to {output_path}")