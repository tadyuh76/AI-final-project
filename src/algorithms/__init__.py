"""
Gói thuật toán cho tối ưu hóa sơ tán.

Cung cấp ba thuật toán tối ưu hóa:
- GBFS: Tìm kiếm tốt nhất ưu tiên đầu tiên tham lam cho tìm đường
- GWO: Bộ tối ưu hóa bầy sói xám cho phân phối luồng
- Hybrid: Kết hợp GBFS + GWO cho kết quả tốt nhất
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
    # Các loại và cấu hình
    'AlgorithmType', 'AlgorithmConfig',
    # Các lớp cơ sở
    'BaseAlgorithm', 'EvacuationPlan', 'EvacuationRoute', 'AlgorithmMetrics',
    'ProgressCallback',
    # Các thuật toán
    'GreedyBestFirstSearch', 'GreyWolfOptimizer', 'HybridGBFSGWO',
    # So sánh
    'AlgorithmComparator', 'ComparisonResult', 'run_comparison'
]
