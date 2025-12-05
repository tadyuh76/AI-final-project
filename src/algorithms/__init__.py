"""
Gói thuật toán cho tối ưu hóa sơ tán.

Cung cấp bốn thuật toán tối ưu hóa:
- A*: Tìm kiếm tối ưu với f(n) = g(n) + h(n) - baseline
- GBFS: Tìm kiếm tốt nhất ưu tiên đầu tiên tham lam cho tìm đường
- GWO: Bộ tối ưu hóa bầy sói xám cho phân phối luồng
- Hybrid: Kết hợp GBFS + GWO cho kết quả tốt nhất
"""

from .base import (
    AlgorithmType, AlgorithmConfig,
    BaseAlgorithm, EvacuationPlan, EvacuationRoute, AlgorithmMetrics,
    ProgressCallback
)
from .astar import AStarSearch
from .gbfs import GreedyBestFirstSearch
from .gwo import GreyWolfOptimizer
from .hybrid import HybridGBFSGWO
from .comparator import AlgorithmComparator, ComparisonResult, run_comparison

__all__ = [
    # Các loại và cấu hình
    'AlgorithmType', 'AlgorithmConfig',
    # Các lớp cơ sở
    'BaseAlgorithm', 'EvacuationPlan', 'EvacuationRoute', 'AlgorithmMetrics',
    'ProgressCallback',
    # Các thuật toán
    'AStarSearch', 'GreedyBestFirstSearch', 'GreyWolfOptimizer', 'HybridGBFSGWO',
    # So sánh
    'AlgorithmComparator', 'ComparisonResult', 'run_comparison'
]
