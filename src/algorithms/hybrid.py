"""
Thuật toán Hybrid GBFS + GWO cho tối ưu hóa sơ tán.

Kết hợp điểm mạnh của cả hai thuật toán:
- GWO: Tối ưu hóa toàn cục của phân phối luồng
- GBFS: Tìm đường cục bộ với heuristic đa mục tiêu

Phương pháp hai giai đoạn:
1. Giai đoạn 1 (GWO): Tối ưu hóa việc các khu vực gửi người đến nơi trú ẩn nào
2. Giai đoạn 2 (GBFS): Tìm đường đi thực tế cho mỗi phân công luồng
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import time

from .base import (
    BaseAlgorithm, AlgorithmType, AlgorithmConfig,
    EvacuationPlan, EvacuationRoute, AlgorithmMetrics
)
from .gbfs import GreedyBestFirstSearch
from .gwo import GreyWolfOptimizer
from ..models.network import EvacuationNetwork
from ..models.node import PopulationZone, Shelter


class HybridGBFSGWO(BaseAlgorithm):
    """
    Thuật toán Hybrid kết hợp tối ưu hóa toàn cục GWO với tìm đường GBFS.

    Giai đoạn 1: GWO xác định phân phối luồng tối ưu
    Giai đoạn 2: GBFS tìm đường đi thực tế cho mỗi phân công
    Tinh chỉnh: Cải thiện lặp lại dựa trên chi phí đường đi thực tế
    """

    def __init__(self, network: EvacuationNetwork, config: Optional[AlgorithmConfig] = None):
        """
        Khởi tạo thuật toán hybrid.

        Args:
            network: Mạng lưới sơ tán
            config: Cấu hình thuật toán (tùy chọn)
        """
        super().__init__(network)
        self.config = config or AlgorithmConfig()

        # Tạo các thuật toán con
        self.gwo = GreyWolfOptimizer(network, config)
        self.gbfs = GreedyBestFirstSearch(network, config)

        # Các tham số đặc thù cho Hybrid
        self.gwo_iterations = self.config.gwo_iterations
        self.refinement_iterations = self.config.refinement_iterations

    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.HYBRID

    def optimize(self, **kwargs) -> Tuple[EvacuationPlan, AlgorithmMetrics]:
        """
        Chạy tối ưu hóa hybrid.

        Returns:
            Tuple của (EvacuationPlan, AlgorithmMetrics)
        """
        start_time = self._start_timer()

        all_zones = self.network.get_population_zones()
        shelters = self.network.get_shelters()

        if not all_zones or not shelters:
            self._stop_timer(start_time)
            return EvacuationPlan(algorithm_type=AlgorithmType.HYBRID), self._metrics

        # Filter zones: only evacuate zones that are in hazard areas
        zones_with_risk = []
        for z in all_zones:
            zone_risk = self.network.get_total_risk_at(z.lat, z.lon)
            if zone_risk >= self.config.min_zone_risk_for_evacuation:
                zones_with_risk.append((z, zone_risk))

        # Sort by risk (highest first) then by population
        zones_with_risk.sort(key=lambda x: (-x[1], -x[0].population))
        zones = [z for z, _ in zones_with_risk]

        if not zones:
            self._stop_timer(start_time)
            return EvacuationPlan(algorithm_type=AlgorithmType.HYBRID), self._metrics

        # ============ Giai đoạn 1: Tối ưu hóa Toàn cục GWO ============
        self.report_progress(0, float('inf'), {'phase': 'gwo_start'})

        # Cấu hình GWO cho ít vòng lặp hơn trong chế độ hybrid
        gwo_config = AlgorithmConfig(
            n_wolves=self.config.n_wolves,
            max_iterations=self.gwo_iterations,
            distance_weight=self.config.distance_weight,
            risk_weight=self.config.risk_weight,
            min_zone_risk_for_evacuation=self.config.min_zone_risk_for_evacuation
        )
        self.gwo = GreyWolfOptimizer(self.network, gwo_config)

        # Thiết lập chuyển tiếp tiến trình
        def gwo_progress(iteration: int, cost: float, data: Any) -> None:
            self._metrics.convergence_history.append(cost)
            self.report_progress(iteration, cost, {'phase': 'gwo', 'data': data})

        self.gwo.set_progress_callback(gwo_progress)

        # Chạy GWO
        gwo_plan, gwo_metrics = self.gwo.optimize()
        flow_matrix = self.gwo.get_flow_matrix()

        if flow_matrix is None:
            self._stop_timer(start_time)
            return gwo_plan, self._metrics

        # ============ Giai đoạn 2: Tìm đường GBFS ============
        self.report_progress(self.gwo_iterations, gwo_metrics.final_cost, {'phase': 'gbfs_start'})

        # Use filtered zones from GWO (already sorted by risk)
        zones = self.gwo.zones  # GWO already filtered zones
        plan = self._apply_gbfs_pathfinding(flow_matrix, zones, shelters)

        # ============ Giai đoạn 3: Tinh chỉnh ============
        self.report_progress(
            self.gwo_iterations + 1,
            self._calculate_plan_cost(plan),
            {'phase': 'refinement_start'}
        )

        plan = self._refine_plan(plan, flow_matrix, zones, shelters)

        # ============ Hoàn thiện ============
        self._stop_timer(start_time)

        # Tính toán các chỉ số cuối cùng
        self._metrics.iterations = (
            self.gwo_iterations + len(zones) + self.refinement_iterations
        )
        self._metrics.final_cost = self._calculate_plan_cost(plan)
        self._metrics.routes_found = len(plan.routes)
        self._metrics.evacuees_covered = plan.total_evacuees

        # Tỷ lệ bao phủ: số người được sơ tán / min(dân số cần sơ tán, tổng sức chứa)
        # zones already filtered to only include zones needing evacuation
        population_needing_evacuation = sum(z.population for z in zones)
        total_capacity = sum(s.capacity for s in shelters)
        max_possible = min(population_needing_evacuation, total_capacity)
        self._metrics.coverage_rate = (
            plan.total_evacuees / max_possible if max_possible > 0 else 0
        )

        if plan.routes:
            self._metrics.average_path_length = (
                sum(len(r.path) for r in plan.routes) / len(plan.routes)
            )

        return plan, self._metrics

    def _apply_gbfs_pathfinding(self,
                                 flow_matrix: np.ndarray,
                                 zones: List[PopulationZone],
                                 shelters: List[Shelter]) -> EvacuationPlan:
        """
        Áp dụng tìm đường GBFS cho các phân công luồng GWO.

        Args:
            flow_matrix: Phân phối luồng được tối ưu hóa bởi GWO [n_zones x n_shelters]
            zones: Danh sách các khu vực dân cư
            shelters: Danh sách các nơi trú ẩn

        Returns:
            EvacuationPlan với các đường đi thực tế
        """
        plan = EvacuationPlan(algorithm_type=AlgorithmType.HYBRID)

        populations = np.array([z.population for z in zones])
        capacities = np.array([s.capacity for s in shelters])

        # Tính toán luồng thực tế
        flows = flow_matrix * populations[:, np.newaxis]

        # Theo dõi mức độ sử dụng nơi trú ẩn
        shelter_occupancy = {s.id: 0 for s in shelters}

        # Track which zones got routes and remaining population
        zones_with_routes = set()
        zone_remaining = {z.id: z.population for z in zones}

        # Xử lý từng khu vực
        for i, zone in enumerate(zones):
            if self._should_stop:
                break

            # Lấy các nơi trú ẩn mà khu vực này nên gửi người đến
            zone_flows = [(j, int(flows[i, j])) for j in range(len(shelters))]
            zone_flows = [(j, f) for j, f in zone_flows
                         if f >= self.config.min_flow_threshold]
            zone_flows.sort(key=lambda x: -x[1])  # Sắp xếp theo luồng giảm dần

            for j, target_flow in zone_flows:
                # Check if zone already fully assigned
                if zone_remaining[zone.id] < self.config.min_flow_threshold:
                    break

                shelter = shelters[j]

                # Kiểm tra sức chứa khả dụng
                available = shelter.capacity - shelter_occupancy[shelter.id]
                # Use remaining population, capped by available capacity and target flow
                actual_flow = min(zone_remaining[zone.id], int(available))
                actual_flow = min(actual_flow, max(target_flow, int(available // 2)))

                if actual_flow < self.config.min_flow_threshold:
                    continue

                # Tìm đường đi sử dụng GBFS
                path, found_shelter, cost = self.gbfs.find_path(zone, [shelter])

                if path and found_shelter:
                    # Tính toán các chỉ số tuyến đường
                    distance = self.gbfs._calculate_path_distance(path)
                    time_hours = self.gbfs._calculate_path_time(path)
                    risk = self.gbfs._calculate_path_risk(path)

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
                    shelter_occupancy[shelter.id] += actual_flow
                    zone_remaining[zone.id] -= actual_flow
                    zones_with_routes.add(zone.id)
                    # Continue to assign remaining population to other shelters

            # Báo cáo tiến trình
            iteration = self.gwo_iterations + i + 1
            self._metrics.convergence_history.append(self._calculate_plan_cost(plan))
            self.report_progress(iteration, self._calculate_plan_cost(plan),
                               {'phase': 'gbfs', 'zone': zone.id})

        # FALLBACK: Ensure all zones have routes - find path to ANY safe shelter
        for i, zone in enumerate(zones):
            # Check if zone still has remaining population to assign
            if zone_remaining.get(zone.id, 0) < self.config.min_flow_threshold:
                continue

            # Try to find path to any shelter with capacity (use consistent 0.6 risk threshold)
            safe_shelters = [s for s in shelters
                           if shelter_occupancy.get(s.id, 0) < s.capacity
                           and self.network.get_total_risk_at(s.lat, s.lon) < 0.6]

            if not safe_shelters:
                # Fallback to any shelter with capacity (for zones in hazard center)
                safe_shelters = [s for s in shelters
                               if shelter_occupancy.get(s.id, 0) < s.capacity]

            if safe_shelters:
                path, found_shelter, cost = self.gbfs.find_path(zone, safe_shelters)

                if path and found_shelter:
                    available = found_shelter.capacity - shelter_occupancy.get(found_shelter.id, 0)
                    actual_flow = min(zone_remaining.get(zone.id, 0), int(available))

                    if actual_flow > 0:
                        distance = self.gbfs._calculate_path_distance(path)
                        time_hours = self.gbfs._calculate_path_time(path)
                        risk = self.gbfs._calculate_path_risk(path)

                        route = EvacuationRoute(
                            zone_id=zone.id,
                            shelter_id=found_shelter.id,
                            path=path,
                            flow=actual_flow,
                            distance_km=distance,
                            estimated_time_hours=time_hours,
                            risk_score=risk
                        )
                        plan.add_route(route)
                        shelter_occupancy[found_shelter.id] = shelter_occupancy.get(found_shelter.id, 0) + actual_flow
                        zone_remaining[zone.id] = zone_remaining.get(zone.id, 0) - actual_flow

        return plan

    def _refine_plan(self,
                     plan: EvacuationPlan,
                     flow_matrix: np.ndarray,
                     zones: List[PopulationZone],
                     shelters: List[Shelter]) -> EvacuationPlan:
        """
        Tinh chỉnh kế hoạch sơ tán bằng cách phân phối lại luồng dựa trên chi phí đường đi thực tế.

        Args:
            plan: Kế hoạch sơ tán hiện tại
            flow_matrix: Ma trận luồng ban đầu
            zones: Các khu vực dân cư
            shelters: Các nơi trú ẩn

        Returns:
            EvacuationPlan đã được tinh chỉnh
        """
        if not plan.routes:
            return plan

        # Xây dựng ma trận chi phí từ các đường đi thực tế
        cost_matrix = np.full((len(zones), len(shelters)), float('inf'))
        zone_to_idx = {z.id: i for i, z in enumerate(zones)}
        shelter_to_idx = {s.id: j for j, s in enumerate(shelters)}

        for route in plan.routes:
            i = zone_to_idx.get(route.zone_id)
            j = shelter_to_idx.get(route.shelter_id)
            if i is not None and j is not None:
                # Chi phí = thời gian + phạt rủi ro
                cost_matrix[i, j] = route.estimated_time_hours + 0.5 * route.risk_score

        # Các vòng lặp tinh chỉnh
        current_plan = plan
        best_cost = self._calculate_plan_cost(current_plan)

        for ref_iter in range(self.refinement_iterations):
            if self._should_stop:
                break

            # Thử cải thiện bằng cách phân phối lại từ tuyến đường chi phí cao sang thấp
            improved_plan = self._try_redistribution(current_plan, cost_matrix,
                                                      zones, shelters)

            new_cost = self._calculate_plan_cost(improved_plan)

            if new_cost < best_cost:
                current_plan = improved_plan
                best_cost = new_cost

            # Báo cáo tiến trình
            iteration = self.gwo_iterations + len(zones) + ref_iter + 1
            self._metrics.convergence_history.append(best_cost)
            self.report_progress(iteration, best_cost, {'phase': 'refinement', 'iter': ref_iter})

        return current_plan

    def _try_redistribution(self,
                            plan: EvacuationPlan,
                            cost_matrix: np.ndarray,
                            zones: List[PopulationZone],
                            shelters: List[Shelter]) -> EvacuationPlan:
        """
        Thử phân phối lại luồng để cải thiện chi phí tổng thể.

        Args:
            plan: Kế hoạch hiện tại
            cost_matrix: Ma trận chi phí đường đi
            zones: Các khu vực
            shelters: Các nơi trú ẩn

        Returns:
            Kế hoạch có khả năng được cải thiện
        """
        # Tìm các tuyến đường chi phí cao có thể được hưởng lợi từ phân phối lại
        if not plan.routes:
            return plan

        # Tính chi phí trung bình
        route_costs = [r.estimated_time_hours + 0.5 * r.risk_score for r in plan.routes]
        mean_cost = np.mean(route_costs) if route_costs else 0

        # Tìm các tuyến đường trên chi phí trung bình
        high_cost_routes = [
            (i, r) for i, r in enumerate(plan.routes)
            if r.estimated_time_hours + 0.5 * r.risk_score > mean_cost
        ]

        if not high_cost_routes:
            return plan

        # Thử tìm các phương án tốt hơn cho các tuyến đường chi phí cao
        new_routes = list(plan.routes)
        shelter_loads = plan.get_shelter_loads()

        zone_to_idx = {z.id: i for i, z in enumerate(zones)}
        shelter_to_idx = {s.id: j for j, s in enumerate(shelters)}
        shelters_list = list(shelters)

        for route_idx, route in high_cost_routes:
            zone_idx = zone_to_idx.get(route.zone_id)
            if zone_idx is None:
                continue

            current_shelter_idx = shelter_to_idx.get(route.shelter_id)
            current_cost = cost_matrix[zone_idx, current_shelter_idx] if current_shelter_idx else float('inf')

            # Tìm các nơi trú ẩn thay thế với chi phí thấp hơn và sức chứa khả dụng
            for j, shelter in enumerate(shelters_list):
                if shelter.id == route.shelter_id:
                    continue

                alt_cost = cost_matrix[zone_idx, j]
                if alt_cost >= current_cost or alt_cost == float('inf'):
                    continue

                # Kiểm tra sức chứa
                current_load = shelter_loads.get(shelter.id, 0)
                if current_load + route.flow > shelter.capacity:
                    continue

                # Tìm thấy một phương án tốt hơn - tìm đường đi mới
                zone = zones[zone_idx]
                path, found_shelter, _ = self.gbfs.find_path(zone, [shelter])

                if path and found_shelter:
                    # Cập nhật tuyến đường
                    new_route = EvacuationRoute(
                        zone_id=route.zone_id,
                        shelter_id=shelter.id,
                        path=path,
                        flow=route.flow,
                        distance_km=self.gbfs._calculate_path_distance(path),
                        estimated_time_hours=self.gbfs._calculate_path_time(path),
                        risk_score=self.gbfs._calculate_path_risk(path)
                    )
                    new_routes[route_idx] = new_route

                    # Cập nhật tải của nơi trú ẩn
                    shelter_loads[route.shelter_id] = shelter_loads.get(route.shelter_id, 0) - route.flow
                    shelter_loads[shelter.id] = shelter_loads.get(shelter.id, 0) + route.flow

                    break  # Chuyển sang tuyến đường chi phí cao tiếp theo

        # Tạo kế hoạch mới
        improved_plan = EvacuationPlan(algorithm_type=AlgorithmType.HYBRID)
        for route in new_routes:
            improved_plan.add_route(route)

        return improved_plan

    def _calculate_plan_cost(self, plan: EvacuationPlan) -> float:
        """
        Tính tổng chi phí của một kế hoạch sơ tán.

        Args:
            plan: Kế hoạch sơ tán

        Returns:
            Giá trị tổng chi phí
        """
        if not plan.routes:
            return float('inf')

        total_cost = 0.0
        for route in plan.routes:
            # Tổng có trọng số của thời gian và rủi ro
            route_cost = route.flow * (
                route.estimated_time_hours +
                0.3 * route.risk_score +
                0.001 * route.distance_km  # Phạt khoảng cách nhỏ
            )
            total_cost += route_cost

        return total_cost

    def get_component_metrics(self) -> Dict[str, AlgorithmMetrics]:
        """Lấy các chỉ số từ các thành phần thuật toán riêng lẻ."""
        return {
            'gwo': self.gwo.metrics,
            'gbfs': self.gbfs.metrics
        }
