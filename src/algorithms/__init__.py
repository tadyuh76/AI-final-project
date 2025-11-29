"""
Algorithms package for evacuation optimization.

Provides three optimization algorithms:
- GBFS: Greedy Best-First Search for pathfinding
- GWO: Grey Wolf Optimizer for flow distribution
- Hybrid: Combined GBFS + GWO for best results
"""

from .base import (
    AlgorithmType, AlgorithmConfig,
    BaseAlgorithm, EvacuationPlan, EvacuationRoute, AlgorithmMetrics,
    ProgressCallback
)
from .gbfs import GreedyBestFirstSearch
from .gwo import GreyWolfOptimizer
from .hybrid import HybridGBFSGWO
from .comparator import AlgorithmComparator, ComparisonResult, run_comparison

__all__ = [
    # Types and Config
    'AlgorithmType', 'AlgorithmConfig',
    # Base classes
    'BaseAlgorithm', 'EvacuationPlan', 'EvacuationRoute', 'AlgorithmMetrics',
    'ProgressCallback',
    # Algorithms
    'GreedyBestFirstSearch', 'GreyWolfOptimizer', 'HybridGBFSGWO',
    # Comparison
    'AlgorithmComparator', 'ComparisonResult', 'run_comparison'
]
