"""
gPAC vs Tensorpac Comparison Experiments

This package contains rigorous scientific benchmarking experiments for comparing
gPAC and Tensorpac libraries for Phase-Amplitude Coupling analysis.
"""

from .base_experiment import BaseExperiment
from .initialization_exp import InitializationExperiment  
from .computation_exp import ComputationExperiment
from .workflow_exp import WorkflowExperiment
from .utils import ExperimentUtils, DataGenerator, ResultLogger

__all__ = [
    'BaseExperiment',
    'InitializationExperiment', 
    'ComputationExperiment',
    'WorkflowExperiment',
    'ExperimentUtils',
    'DataGenerator', 
    'ResultLogger'
]

__version__ = "1.0.0"