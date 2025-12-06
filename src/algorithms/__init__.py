"""
Gói thuật toán cho tối ưu hóa sơ tán.

Cung cấp hai thuật toán tối ưu hóa:
- GBFS: Tìm kiếm tốt nhất ưu tiên đầu tiên tham lam cho tìm đường
- GWO: Bộ tối ưu hóa bầy sói xám cho phân phối luồng
"""

from .base import (
    AlgorithmType, AlgorithmConfig,
    BaseAlgorithm, EvacuationPlan, EvacuationRoute, AlgorithmMetrics,
    ProgressCallback
)
from .gbfs import GreedyBestFirstSearch
from .gwo import GreyWolfOptimizer
from .comparator import AlgorithmComparator, ComparisonResult, run_comparison

__all__ = [
    # Các loại và cấu hình
    'AlgorithmType', 'AlgorithmConfig',
    # Các lớp cơ sở
    'BaseAlgorithm', 'EvacuationPlan', 'EvacuationRoute', 'AlgorithmMetrics',
    'ProgressCallback',
    # Các thuật toán
    'GreedyBestFirstSearch', 'GreyWolfOptimizer',
    # So sánh
    'AlgorithmComparator', 'ComparisonResult', 'run_comparison'
]
