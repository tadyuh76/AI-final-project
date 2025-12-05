"""
Các lớp cơ sở và giao diện cho các thuật toán tối ưu hóa sơ tán.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Any
from enum import Enum
import time


class AlgorithmType(Enum):
    """Các loại thuật toán khả dụng."""
    GBFS = "gbfs"
    GWO = "gwo"


@dataclass
class EvacuationRoute:
    """Đại diện cho một tuyến đường sơ tán đơn lẻ."""
    zone_id: str
    shelter_id: str
    path: List[str]  # Danh sách các ID nút từ khu vực đến nơi trú ẩn
    flow: int  # Số người được phân bổ cho tuyến đường này
    distance_km: float = 0.0
    estimated_time_hours: float = 0.0
    risk_score: float = 0.0

    @property
    def path_length(self) -> int:
        """Số lượng nút trong đường đi."""
        return len(self.path)


@dataclass
class EvacuationPlan:
    """Kế hoạch sơ tán hoàn chỉnh với tất cả các tuyến đường."""
    routes: List[EvacuationRoute] = field(default_factory=list)
    total_evacuees: int = 0
    total_time_hours: float = 0.0
    average_risk: float = 0.0
    algorithm_type: AlgorithmType = AlgorithmType.GBFS

    def add_route(self, route: EvacuationRoute) -> None:
        """Thêm một tuyến đường vào kế hoạch."""
        self.routes.append(route)
        self.total_evacuees += route.flow
        # Cập nhật các giá trị trung bình có trọng số
        if self.total_evacuees > 0:
            total_weighted_time = sum(r.estimated_time_hours * r.flow for r in self.routes)
            total_weighted_risk = sum(r.risk_score * r.flow for r in self.routes)
            self.total_time_hours = total_weighted_time / self.total_evacuees
            self.average_risk = total_weighted_risk / self.total_evacuees

    def get_routes_for_zone(self, zone_id: str) -> List[EvacuationRoute]:
        """Lấy tất cả các tuyến đường xuất phát từ một khu vực."""
        return [r for r in self.routes if r.zone_id == zone_id]

    def get_routes_to_shelter(self, shelter_id: str) -> List[EvacuationRoute]:
        """Lấy tất cả các tuyến đường đi đến một nơi trú ẩn."""
        return [r for r in self.routes if r.shelter_id == shelter_id]

    def get_shelter_loads(self) -> Dict[str, int]:
        """Lấy tổng lưu lượng đến từng nơi trú ẩn."""
        loads: Dict[str, int] = {}
        for route in self.routes:
            if route.shelter_id not in loads:
                loads[route.shelter_id] = 0
            loads[route.shelter_id] += route.flow
        return loads


@dataclass
class AlgorithmMetrics:
    """Các chỉ số hiệu suất cho việc thực thi thuật toán."""
    algorithm_type: AlgorithmType
    execution_time_seconds: float = 0.0
    iterations: int = 0
    convergence_history: List[float] = field(default_factory=list)
    final_cost: float = 0.0
    routes_found: int = 0
    evacuees_covered: int = 0
    average_path_length: float = 0.0
    coverage_rate: float = 0.0  # % dân số có tuyến đường

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang từ điển để tuần tự hóa."""
        return {
            'algorithm': self.algorithm_type.value,
            'execution_time_seconds': self.execution_time_seconds,
            'iterations': self.iterations,
            'final_cost': self.final_cost,
            'routes_found': self.routes_found,
            'evacuees_covered': self.evacuees_covered,
            'average_path_length': self.average_path_length,
            'coverage_rate': self.coverage_rate,
            'convergence_history': self.convergence_history
        }


# Bí danh kiểu dữ liệu cho callback tiến trình
ProgressCallback = Callable[[int, float, Optional[Any]], None]


class BaseAlgorithm(ABC):
    """Lớp cơ sở trừu tượng cho các thuật toán tối ưu hóa sơ tán."""

    def __init__(self, network: Any):
        """
        Khởi tạo thuật toán với mạng lưới.

        Args:
            network: Thể hiện EvacuationNetwork
        """
        self.network = network
        self._metrics = AlgorithmMetrics(algorithm_type=self.algorithm_type)
        self._progress_callback: Optional[ProgressCallback] = None
        self._is_running = False
        self._should_stop = False

    @property
    @abstractmethod
    def algorithm_type(self) -> AlgorithmType:
        """Trả về loại thuật toán."""
        pass

    @property
    def metrics(self) -> AlgorithmMetrics:
        """Lấy các chỉ số hiện tại."""
        return self._metrics

    def set_progress_callback(self, callback: ProgressCallback) -> None:
        """Đặt callback cho các cập nhật tiến trình."""
        self._progress_callback = callback

    def report_progress(self, iteration: int, cost: float, data: Any = None) -> None:
        """Báo cáo tiến trình cho callback nếu đã được đặt."""
        if self._progress_callback:
            self._progress_callback(iteration, cost, data)

    def stop(self) -> None:
        """Yêu cầu thuật toán dừng lại."""
        self._should_stop = True

    def reset(self) -> None:
        """Đặt lại trạng thái thuật toán."""
        self._metrics = AlgorithmMetrics(algorithm_type=self.algorithm_type)
        self._is_running = False
        self._should_stop = False

    @abstractmethod
    def optimize(self, **kwargs) -> Tuple[EvacuationPlan, AlgorithmMetrics]:
        """
        Chạy thuật toán tối ưu hóa.

        Returns:
            Tuple của (EvacuationPlan, AlgorithmMetrics)
        """
        pass

    def _start_timer(self) -> float:
        """Bắt đầu bộ đếm thời gian thực thi."""
        self._is_running = True
        self._should_stop = False
        return time.time()

    def _stop_timer(self, start_time: float) -> None:
        """Dừng bộ đếm thời gian thực thi và ghi lại thời gian."""
        self._metrics.execution_time_seconds = time.time() - start_time
        self._is_running = False


@dataclass
class AlgorithmConfig:
    """Cấu hình cho các tham số thuật toán."""
    # Trọng số GBFS (rebalanced for better shelter distribution)
    distance_weight: float = 0.35
    risk_weight: float = 0.25
    congestion_weight: float = 0.15
    capacity_weight: float = 0.25  # Increased from 0.1 to prioritize shelter capacity

    # Tham số GWO
    n_wolves: int = 30
    max_iterations: int = 100
    a_initial: float = 2.0  # Tham số khám phá

    # Chung
    min_flow_threshold: int = 20  # Lowered from 100 to avoid dropping small valid assignments
    min_zone_risk_for_evacuation: float = 0.1  # Zones with risk below this don't need evacuation

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang từ điển."""
        return {
            'distance_weight': self.distance_weight,
            'risk_weight': self.risk_weight,
            'congestion_weight': self.congestion_weight,
            'capacity_weight': self.capacity_weight,
            'n_wolves': self.n_wolves,
            'max_iterations': self.max_iterations,
            'a_initial': self.a_initial,
            'min_flow_threshold': self.min_flow_threshold,
            'min_zone_risk_for_evacuation': self.min_zone_risk_for_evacuation
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlgorithmConfig':
        """Tạo từ từ điển."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
