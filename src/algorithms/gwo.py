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
        all_zones = self.network.get_population_zones()
        self.shelters = self.network.get_shelters()

        # Filter zones: only include zones that need evacuation (in hazard areas)
        self.zones = []
        self._zone_risks = []  # Store zone risks for sorting
        for z in all_zones:
            zone_risk = self.network.get_total_risk_at(z.lat, z.lon)
            if zone_risk >= self.config.min_zone_risk_for_evacuation:
                self.zones.append(z)
                self._zone_risks.append(zone_risk)

        # Sort zones by risk (highest first) for priority evacuation
        if self.zones:
            sorted_pairs = sorted(zip(self.zones, self._zone_risks),
                                  key=lambda x: (-x[1], -x[0].population))
            self.zones = [z for z, _ in sorted_pairs]
            self._zone_risks = [r for _, r in sorted_pairs]

        self.n_zones = len(self.zones)
        self.n_shelters = len(self.shelters)

        # Tính trước ma trận khoảng cách
        self._distance_matrix = np.zeros((self.n_zones, self.n_shelters))
        for i, zone in enumerate(self.zones):
            for j, shelter in enumerate(self.shelters):
                self._distance_matrix[i, j] = haversine_distance(
                    zone.lat, zone.lon, shelter.lat, shelter.lon
                )

        # Tính trước ma trận rủi ro (kiểm tra nhiều điểm dọc theo đường đi)
        self._risk_matrix = np.zeros((self.n_zones, self.n_shelters))
        for i, zone in enumerate(self.zones):
            for j, shelter in enumerate(self.shelters):
                # Check risk at multiple points along the route
                max_risk = 0.0
                for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    lat = zone.lat + t * (shelter.lat - zone.lat)
                    lon = zone.lon + t * (shelter.lon - zone.lon)
                    risk = self.network.get_total_risk_at(lat, lon)
                    max_risk = max(max_risk, risk)

                # Also check if shelter is in hazard zone
                shelter_risk = self.network.get_total_risk_at(shelter.lat, shelter.lon)
                max_risk = max(max_risk, shelter_risk)

                self._risk_matrix[i, j] = max_risk

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

        # 2. Chi phí rủi ro - HEAVILY penalize routes through hazard zones
        # Block routes where risk > 0.6 by adding massive penalty (standardized threshold)
        risk_penalty_matrix = np.where(self._risk_matrix > 0.6, 1000.0, self._risk_matrix * 10.0)
        risk_cost = np.sum(flows * risk_penalty_matrix) / 1000.0

        # 3. Phạt vi phạm sức chứa - stronger penalty to enforce hard constraints
        shelter_loads = flows.sum(axis=0)
        capacity_violations = np.maximum(0, shelter_loads - capacities)
        # Exponential penalty: small violations = small penalty, large violations = huge penalty
        relative_violations = capacity_violations / (capacities + 1)
        capacity_penalty = np.sum(relative_violations ** 2) * 100 + np.sum(capacity_violations > 0) * 50

        # 4. Phạt mất cân bằng luồng (ưu tiên phân phối đồng đều)
        # Increased multiplier from 5 to 50 to strongly penalize uneven distribution
        if capacities.sum() > 0:
            utilization = shelter_loads / (capacities + 1)
            balance_penalty = np.std(utilization) * 50  # Much stronger balance enforcement
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

        # Tính chi phí theo công thức chuẩn (để so sánh công bằng với GBFS và Hybrid)
        standardized_cost = self._calculate_plan_cost(plan)

        # Hoàn thiện các chỉ số
        self._stop_timer(start_time)
        self._metrics.iterations = len(self._metrics.convergence_history)
        self._metrics.final_cost = standardized_cost  # Dùng chi phí chuẩn, không phải fitness
        self._metrics.routes_found = len(plan.routes)
        self._metrics.evacuees_covered = plan.total_evacuees

        # Tính độ dài đường đi trung bình
        if plan.routes:
            total_path_length = sum(len(r.path) for r in plan.routes)
            self._metrics.average_path_length = total_path_length / len(plan.routes)
        else:
            self._metrics.average_path_length = 0.0

        # Tỷ lệ bao phủ: số người được sơ tán / min(dân số cần sơ tán, tổng sức chứa)
        # self.zones already filtered to only include zones needing evacuation
        population_needing_evacuation = sum(z.population for z in self.zones)
        total_capacity = sum(s.capacity for s in self.shelters)
        max_possible = min(population_needing_evacuation, total_capacity)
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

        # Track zones that have been assigned and remaining population
        zone_assigned = [False] * self.n_zones
        zone_remaining = [z.population for z in self.zones]  # Track remaining population per zone

        for i, zone in enumerate(self.zones):
            # Sort shelters by preference for this zone (lower risk, closer distance)
            shelter_scores = []
            for j, shelter in enumerate(self.shelters):
                risk = self._risk_matrix[i, j]
                dist = self._distance_matrix[i, j]
                flow = flows[i, j]
                available = capacities[j] - shelter_occupancy[j]

                # Skip if no capacity or high risk route (use consistent 0.6 threshold)
                if available < self.config.min_flow_threshold:
                    continue
                if risk > 0.6:  # Standardized risk threshold
                    continue

                # Score: prefer higher flow, lower risk, closer distance
                # Also consider capacity availability more strongly
                capacity_score = min(1.0, available / (zone_remaining[i] + 1)) * 50
                score = flow * 10 - risk * 100 - dist + capacity_score
                shelter_scores.append((j, score, flow, available))

            # Sort by score (higher is better)
            shelter_scores.sort(key=lambda x: -x[1])

            for j, score, flow, available in shelter_scores:
                if zone_remaining[i] < self.config.min_flow_threshold:
                    break  # Zone fully assigned

                shelter = self.shelters[j]
                # Use remaining population, not original flow
                actual_flow = min(int(zone_remaining[i]), int(available))
                # Ensure we don't exceed the GWO-suggested flow too much
                actual_flow = min(actual_flow, max(int(flow), int(available // 2)))

                if actual_flow < self.config.min_flow_threshold:
                    continue

                # Build path using network graph
                path = self._find_path(zone, shelter)

                distance = self._distance_matrix[i, j]
                time_hours = distance / 30.0
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
                zone_remaining[i] -= actual_flow
                zone_assigned[i] = True

        # Ensure all zones have at least one route - assign remaining population to nearest safe shelter
        for i, zone in enumerate(self.zones):
            if zone_remaining[i] < self.config.min_flow_threshold:
                continue  # Zone already fully assigned

            # Find nearest safe shelter with capacity
            best_shelter = None
            best_dist = float('inf')
            best_j = -1

            for j, shelter in enumerate(self.shelters):
                available = capacities[j] - shelter_occupancy[j]
                if available < self.config.min_flow_threshold:  # Use config threshold
                    continue

                risk = self._risk_matrix[i, j]
                if risk > 0.6:  # Consistent risk threshold
                    continue

                dist = self._distance_matrix[i, j]
                if dist < best_dist:
                    best_dist = dist
                    best_shelter = shelter
                    best_j = j

            # If no safe shelter found, try with higher risk tolerance for zones near hazard
            if best_shelter is None:
                for j, shelter in enumerate(self.shelters):
                    available = capacities[j] - shelter_occupancy[j]
                    if available < self.config.min_flow_threshold:
                        continue

                    dist = self._distance_matrix[i, j]
                    if dist < best_dist:
                        best_dist = dist
                        best_shelter = shelter
                        best_j = j

            if best_shelter is not None:
                # Assign remaining population
                available = capacities[best_j] - shelter_occupancy[best_j]
                actual_flow = min(zone_remaining[i], int(available))

                if actual_flow > 0:
                    path = self._find_path(zone, best_shelter)
                    risk = self._risk_matrix[i, best_j]

                    route = EvacuationRoute(
                        zone_id=zone.id,
                        shelter_id=best_shelter.id,
                        path=path,
                        flow=actual_flow,
                        distance_km=best_dist,
                        estimated_time_hours=best_dist / 30.0,
                        risk_score=risk
                    )
                    plan.add_route(route)
                    shelter_occupancy[best_j] += actual_flow

        return plan

    def _find_path(self, zone: PopulationZone, shelter: Shelter) -> List[str]:
        """Find path through network from zone to shelter using A* with hazard avoidance."""
        import heapq
        from ..models.node import haversine_distance

        # Thử lấy zone node trực tiếp từ mạng (nếu zone được kết nối)
        zone_in_network = self.network.get_node(zone.id)
        if zone_in_network:
            zone_node = zone_in_network
        else:
            # Dự phòng: tìm nút gần nhất
            zone_node = self.network.find_nearest_node(zone.lat, zone.lon)

        # Tương tự cho shelter
        shelter_in_network = self.network.get_node(shelter.id)
        shelter_nearest = self.network.find_nearest_node(shelter.lat, shelter.lon)

        if not zone_node:
            return [zone.id, shelter.id]

        # Xác định đích: có thể là shelter trực tiếp hoặc nút gần nhất với shelter
        target_node = shelter_in_network if shelter_in_network else shelter_nearest
        if not target_node:
            return [zone.id, shelter.id]

        # A* search with hazard avoidance
        # Priority queue: (f_cost, counter, node_id, path)
        counter = 0
        start_h = haversine_distance(zone_node.lat, zone_node.lon, target_node.lat, target_node.lon)

        # Bắt đầu từ zone node
        if zone_in_network:
            open_set = [(start_h, counter, zone_node.id, [zone_node.id])]
        else:
            open_set = [(start_h, counter, zone_node.id, [zone.id, zone_node.id])]

        visited = set()
        g_costs = {zone_node.id: 0.0}

        while open_set:
            f_cost, _, current_id, path = heapq.heappop(open_set)

            if current_id in visited:
                continue
            visited.add(current_id)

            # Kiểm tra nếu đã đến shelter trực tiếp
            if current_id == shelter.id:
                return path

            # Kiểm tra nếu đã đến nút gần shelter và có cạnh trực tiếp đến shelter
            if shelter_nearest and current_id == shelter_nearest.id:
                edge_to_shelter = self.network.get_edge_between(current_id, shelter.id)
                if edge_to_shelter and not edge_to_shelter.is_blocked:
                    return path + [shelter.id]

            current_node = self.network.get_node(current_id)
            if not current_node:
                continue

            for neighbor_id in self.network.get_neighbors(current_id):
                if neighbor_id in visited:
                    continue

                # Get edge and check for hazards
                edge = self.network.get_edge_between(current_id, neighbor_id)
                if not edge or edge.is_blocked:
                    continue

                # Skip high-risk edges completely (standardized 0.6 threshold)
                if edge.flood_risk > 0.6:
                    continue

                # Calculate cost with hazard penalty
                edge_cost = edge.length_km
                if edge.flood_risk > 0.3:
                    edge_cost *= (1 + edge.flood_risk * 10)  # Penalty for moderate risk

                new_g = g_costs[current_id] + edge_cost

                if neighbor_id not in g_costs or new_g < g_costs[neighbor_id]:
                    g_costs[neighbor_id] = new_g
                    neighbor_node = self.network.get_node(neighbor_id)
                    if neighbor_node:
                        h = haversine_distance(neighbor_node.lat, neighbor_node.lon,
                                              target_node.lat, target_node.lon)
                        f = new_g + h
                        counter += 1
                        heapq.heappush(open_set, (f, counter, neighbor_id, path + [neighbor_id]))

        # Fallback: try BFS without strict hazard avoidance
        from collections import deque
        visited = {zone_node.id}
        if zone_in_network:
            queue = deque([(zone_node.id, [zone_node.id])])
        else:
            queue = deque([(zone_node.id, [zone.id, zone_node.id])])

        while queue:
            current_id, path = queue.popleft()

            # Kiểm tra đã đến shelter
            if current_id == shelter.id:
                return path

            # Kiểm tra đã đến nút gần shelter
            if shelter_nearest and current_id == shelter_nearest.id:
                edge_to_shelter = self.network.get_edge_between(current_id, shelter.id)
                if edge_to_shelter and not edge_to_shelter.is_blocked:
                    return path + [shelter.id]

            for neighbor_id in self.network.get_neighbors(current_id):
                if neighbor_id in visited:
                    continue
                edge = self.network.get_edge_between(current_id, neighbor_id)
                if edge and edge.is_blocked:
                    continue
                visited.add(neighbor_id)
                queue.append((neighbor_id, path + [neighbor_id]))

        # Last resort: direct path
        return [zone.id, shelter.id]

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

    def _calculate_plan_cost(self, plan: EvacuationPlan) -> float:
        """
        Tính tổng chi phí của một kế hoạch theo công thức chuẩn.

        Args:
            plan: Kế hoạch sơ tán

        Returns:
            Giá trị tổng chi phí
        """
        if not plan.routes:
            return float('inf')

        total_cost = 0.0
        for route in plan.routes:
            # Công thức chuẩn: flow × (time + 0.3×risk + 0.001×distance)
            route_cost = route.flow * (
                route.estimated_time_hours +
                0.3 * route.risk_score +
                0.001 * route.distance_km
            )
            total_cost += route_cost
        return total_cost
