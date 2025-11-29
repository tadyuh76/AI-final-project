"""
Thuật toán Tối ưu hóa Bầy Sói Xám (GWO) cho phân phối luồng sơ tán.

GWO là thuật toán metaheuristic lấy cảm hứng từ hành vi săn mồi của sói xám.
Thuật toán tối ưu hóa phân phối luồng toàn cục trên mạng lưới.

Hệ thống cấp bậc sói:
- Alpha (α): Giải pháp tốt nhất
- Beta (β): Giải pháp tốt thứ hai
- Delta (δ): Giải pháp tốt thứ ba
- Omega (ω): Các cá thể còn lại trong quần thể
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import time

from .base import (
    BaseAlgorithm, AlgorithmType, AlgorithmConfig,
    EvacuationPlan, EvacuationRoute, AlgorithmMetrics
)
from ..models.network import EvacuationNetwork
from ..models.node import PopulationZone, Shelter, haversine_distance


@dataclass
class Wolf:
    """Đại diện cho một con sói (giải pháp) trong thuật toán GWO."""
    position: np.ndarray  # Ma trận phân phối luồng [n_zones x n_shelters]
    fitness: float = float('inf')

    def copy(self) -> 'Wolf':
        """Tạo một bản sao của con sói này."""
        return Wolf(position=self.position.copy(), fitness=self.fitness)


class GreyWolfOptimizer(BaseAlgorithm):
    """
    Thuật toán Tối ưu hóa Bầy Sói Xám cho phân phối luồng sơ tán.

    Tối ưu hóa cách phân phối dân số từ các khu vực đến các nơi trú ẩn
    để giảm thiểu tổng thời gian sơ tán trong khi vẫn đảm bảo các ràng buộc.
    """

    def __init__(self, network: EvacuationNetwork, config: Optional[AlgorithmConfig] = None):
        """
        Khởi tạo thuật toán GWO.

        Args:
            network: Mạng lưới sơ tán
            config: Cấu hình thuật toán (tùy chọn)
        """
        super().__init__(network)
        self.config = config or AlgorithmConfig()

        # Các tham số GWO
        self.n_wolves = self.config.n_wolves
        self.max_iterations = self.config.max_iterations
        self.a_initial = self.config.a_initial

        # Quần thể và các con sói
        self.wolves: List[Wolf] = []
        self.alpha: Optional[Wolf] = None
        self.beta: Optional[Wolf] = None
        self.delta: Optional[Wolf] = None

        # Các chiều của bài toán (được thiết lập trong quá trình tối ưu hóa)
        self.n_zones = 0
        self.n_shelters = 0
        self.zones: List[PopulationZone] = []
        self.shelters: List[Shelter] = []

        # Ma trận khoảng cách được tính trước để đánh giá fitness
        self._distance_matrix: Optional[np.ndarray] = None
        self._risk_matrix: Optional[np.ndarray] = None

    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.GWO

    def _initialize_problem(self) -> None:
        """Khởi tạo các chiều của bài toán và tính trước các ma trận."""
        self.zones = self.network.get_population_zones()
        self.shelters = self.network.get_shelters()
        self.n_zones = len(self.zones)
        self.n_shelters = len(self.shelters)

        # Tính trước ma trận khoảng cách
        self._distance_matrix = np.zeros((self.n_zones, self.n_shelters))
        for i, zone in enumerate(self.zones):
            for j, shelter in enumerate(self.shelters):
                self._distance_matrix[i, j] = haversine_distance(
                    zone.lat, zone.lon, shelter.lat, shelter.lon
                )

        # Tính trước ma trận rủi ro (dựa trên rủi ro tại điểm giữa đường đi)
        self._risk_matrix = np.zeros((self.n_zones, self.n_shelters))
        for i, zone in enumerate(self.zones):
            for j, shelter in enumerate(self.shelters):
                mid_lat = (zone.lat + shelter.lat) / 2
                mid_lon = (zone.lon + shelter.lon) / 2
                self._risk_matrix[i, j] = self.network.get_total_risk_at(mid_lat, mid_lon)

    def _initialize_population(self) -> None:
        """Khởi tạo quần thể sói với các giải pháp ngẫu nhiên."""
        self.wolves = []

        for _ in range(self.n_wolves):
            # Phân phối luồng ngẫu nhiên
            position = np.random.rand(self.n_zones, self.n_shelters)
            # Chuẩn hóa các hàng (luồng của mỗi khu vực tổng bằng 1)
            row_sums = position.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Tránh chia cho 0
            position = position / row_sums

            wolf = Wolf(position=position)
            wolf.fitness = self._calculate_fitness(wolf.position)
            self.wolves.append(wolf)

        # Sắp xếp và xác định alpha, beta, delta
        self._update_hierarchy()

    def _update_hierarchy(self) -> None:
        """Cập nhật các con sói alpha, beta, delta dựa trên fitness."""
        sorted_wolves = sorted(self.wolves, key=lambda w: w.fitness)
        self.alpha = sorted_wolves[0].copy()
        self.beta = sorted_wolves[1].copy() if len(sorted_wolves) > 1 else self.alpha.copy()
        self.delta = sorted_wolves[2].copy() if len(sorted_wolves) > 2 else self.beta.copy()

    def _calculate_fitness(self, position: np.ndarray) -> float:
        """
        Tính toán fitness (chi phí) của một giải pháp.

        Fitness thấp hơn là tốt hơn. Xem xét:
        - Tổng thời gian sơ tán (có trọng số theo luồng)
        - Vi phạm sức chứa của nơi trú ẩn
        - Phơi nhiễm rủi ro
        - Cân bằng luồng
        """
        # Lấy mảng dân số
        populations = np.array([z.population for z in self.zones])
        capacities = np.array([s.capacity for s in self.shelters])
        total_capacity = capacities.sum()

        # Giới hạn dân số theo tổng sức chứa của nơi trú ẩn để tối ưu hóa thực tế
        # Giảm tỷ lệ nếu dân số vượt quá sức chứa
        total_pop = populations.sum()
        if total_pop > total_capacity:
            scale_factor = total_capacity / total_pop
            effective_populations = populations * scale_factor
        else:
            effective_populations = populations

        # Tính toán luồng thực tế
        flows = position * effective_populations[:, np.newaxis]

        # 1. Chi phí thời gian (luồng * khoảng cách / tốc độ) - chuẩn hóa
        # Giả sử tốc độ trung bình là 30 km/h
        avg_speed = 30.0
        time_cost = np.sum(flows * self._distance_matrix / avg_speed) / 1000.0  # Chuẩn hóa

        # 2. Chi phí rủi ro - chuẩn hóa
        risk_cost = np.sum(flows * self._risk_matrix) / 1000.0

        # 3. Phạt vi phạm sức chứa - chuẩn hóa
        shelter_loads = flows.sum(axis=0)
        capacity_violations = np.maximum(0, shelter_loads - capacities)
        capacity_penalty = np.sum(capacity_violations / (capacities + 1)) * 10  # Vi phạm tương đối

        # 4. Phạt mất cân bằng luồng (ưu tiên phân phối đồng đều)
        if capacities.sum() > 0:
            utilization = shelter_loads / (capacities + 1)
            balance_penalty = np.std(utilization) * 5
        else:
            balance_penalty = 0

        # 5. Phạt độ bao phủ - sử dụng dân số hiệu quả
        total_flow = flows.sum()
        total_effective = effective_populations.sum()
        if total_effective > 0:
            coverage = total_flow / total_effective
            coverage_penalty = (1 - coverage) ** 2 * 10
        else:
            coverage_penalty = 0

        return time_cost + risk_cost + capacity_penalty + balance_penalty + coverage_penalty

    def _update_position(self, wolf: Wolf, a: float) -> None:
        """
        Cập nhật vị trí của con sói dựa trên alpha, beta, delta.

        Args:
            wolf: Con sói cần cập nhật
            a: Tham số khám phá (giảm dần theo các vòng lặp)
        """
        for i in range(self.n_zones):
            for j in range(self.n_shelters):
                # Tính toán cập nhật vị trí từ alpha, beta, delta
                # Ảnh hưởng của Alpha
                r1, r2 = np.random.rand(2)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * self.alpha.position[i, j] - wolf.position[i, j])
                X1 = self.alpha.position[i, j] - A1 * D_alpha

                # Ảnh hưởng của Beta
                r1, r2 = np.random.rand(2)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * self.beta.position[i, j] - wolf.position[i, j])
                X2 = self.beta.position[i, j] - A2 * D_beta

                # Ảnh hưởng của Delta
                r1, r2 = np.random.rand(2)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * self.delta.position[i, j] - wolf.position[i, j])
                X3 = self.delta.position[i, j] - A3 * D_delta

                # Trung bình của ba ảnh hưởng
                wolf.position[i, j] = (X1 + X2 + X3) / 3

        # Cắt bớt về phạm vi hợp lệ và chuẩn hóa
        wolf.position = np.clip(wolf.position, 0, 1)
        row_sums = wolf.position.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        wolf.position = wolf.position / row_sums

        # Tính lại fitness
        wolf.fitness = self._calculate_fitness(wolf.position)

    def optimize(self, **kwargs) -> Tuple[EvacuationPlan, AlgorithmMetrics]:
        """
        Chạy tối ưu hóa GWO.

        Returns:
            Tuple của (EvacuationPlan, AlgorithmMetrics)
        """
        start_time = self._start_timer()

        # Khởi tạo bài toán
        self._initialize_problem()

        if self.n_zones == 0 or self.n_shelters == 0:
            self._stop_timer(start_time)
            return EvacuationPlan(algorithm_type=AlgorithmType.GWO), self._metrics

        # Khởi tạo quần thể
        self._initialize_population()

        # Vòng lặp tối ưu hóa chính
        for iteration in range(self.max_iterations):
            if self._should_stop:
                break

            # Giảm tuyến tính a từ a_initial về 0
            a = self.a_initial - iteration * (self.a_initial / self.max_iterations)

            # Cập nhật từng con sói
            for wolf in self.wolves:
                self._update_position(wolf, a)

            # Cập nhật hệ thống cấp bậc
            self._update_hierarchy()

            # Ghi lại quá trình hội tụ
            self._metrics.convergence_history.append(self.alpha.fitness)

            # Báo cáo tiến trình
            self.report_progress(iteration + 1, self.alpha.fitness, self.alpha.position)

        # Chuyển đổi giải pháp tốt nhất thành kế hoạch sơ tán
        plan = self._convert_to_plan(self.alpha.position)

        # Hoàn thiện các chỉ số
        self._stop_timer(start_time)
        self._metrics.iterations = len(self._metrics.convergence_history)
        self._metrics.final_cost = self.alpha.fitness
        self._metrics.routes_found = len(plan.routes)
        self._metrics.evacuees_covered = plan.total_evacuees

        # Tỷ lệ bao phủ: số người được sơ tán / min(tổng dân số, tổng sức chứa)
        total_population = sum(z.population for z in self.zones)
        total_capacity = sum(s.capacity for s in self.shelters)
        max_possible = min(total_population, total_capacity)
        self._metrics.coverage_rate = (
            plan.total_evacuees / max_possible if max_possible > 0 else 0
        )

        return plan, self._metrics

    def _convert_to_plan(self, position: np.ndarray) -> EvacuationPlan:
        """
        Chuyển đổi giải pháp GWO (ma trận luồng) thành EvacuationPlan.

        Args:
            position: Ma trận phân phối luồng

        Returns:
            EvacuationPlan với các tuyến đường
        """
        plan = EvacuationPlan(algorithm_type=AlgorithmType.GWO)

        populations = np.array([z.population for z in self.zones])
        capacities = np.array([s.capacity for s in self.shelters])

        # Tính toán luồng thực tế
        flows = position * populations[:, np.newaxis]

        # Theo dõi mức độ sử dụng của nơi trú ẩn
        shelter_occupancy = np.zeros(self.n_shelters)

        for i, zone in enumerate(self.zones):
            for j, shelter in enumerate(self.shelters):
                flow = int(flows[i, j])

                # Áp dụng ràng buộc sức chứa
                available = capacities[j] - shelter_occupancy[j]
                actual_flow = min(flow, int(available))

                if actual_flow >= self.config.min_flow_threshold:
                    # Tạo đường đi trực tiếp đơn giản (GWO không thực hiện tìm đường)
                    # Đường đi thực tế sẽ được tinh chỉnh bởi thuật toán hybrid hoặc GBFS
                    path = [zone.id, shelter.id]

                    distance = self._distance_matrix[i, j]
                    time_hours = distance / 30.0  # Giả sử 30 km/h
                    risk = self._risk_matrix[i, j]

                    route = EvacuationRoute(
                        zone_id=zone.id,
                        shelter_id=shelter.id,
                        path=path,
                        flow=actual_flow,
                        distance_km=distance,
                        estimated_time_hours=time_hours,
                        risk_score=risk
                    )
                    plan.add_route(route)
                    shelter_occupancy[j] += actual_flow

        return plan

    def get_flow_matrix(self) -> Optional[np.ndarray]:
        """Lấy ma trận phân phối luồng đã được tối ưu hóa."""
        if self.alpha:
            return self.alpha.position.copy()
        return None

    def get_best_fitness(self) -> float:
        """Lấy giá trị fitness tốt nhất tìm được."""
        if self.alpha:
            return self.alpha.fitness
        return float('inf')
