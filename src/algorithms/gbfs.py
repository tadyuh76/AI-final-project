"""
Thuật toán Tìm kiếm Tốt nhất Ưu tiên Tham lam (GBFS) cho tìm đường sơ tán.

Sử dụng heuristic đa mục tiêu kết hợp:
- Khoảng cách đến nơi trú ẩn
- Rủi ro lũ lụt/thiên tai
- Tắc nghẽn giao thông
- Sức chứa còn lại của nơi trú ẩn
"""

from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass
import heapq
import time

from .base import (
    BaseAlgorithm, AlgorithmType, AlgorithmConfig,
    EvacuationPlan, EvacuationRoute, AlgorithmMetrics
)
from ..models.network import EvacuationNetwork
from ..models.node import Node, PopulationZone, Shelter, haversine_distance


@dataclass
class SearchNode:
    """Nút trong cây tìm kiếm GBFS."""
    node_id: str
    g_cost: float  # Chi phí thực tế từ điểm bắt đầu
    h_cost: float  # Chi phí heuristic đến đích
    parent: Optional['SearchNode'] = None

    @property
    def f_cost(self) -> float:
        """Tổng chi phí ước tính (với GBFS, chủ yếu sử dụng h_cost)."""
        return self.h_cost

    def __lt__(self, other: 'SearchNode') -> bool:
        """So sánh cho hàng đợi ưu tiên (f_cost thấp hơn là tốt hơn)."""
        return self.f_cost < other.f_cost


class GreedyBestFirstSearch(BaseAlgorithm):
    """
    Tìm kiếm Tốt nhất Ưu tiên Tham lam cho tìm đường sơ tán.

    Tìm đường đi từ các khu vực dân cư đến các nơi trú ẩn sử dụng heuristic
    đa mục tiêu xem xét khoảng cách, rủi ro, tắc nghẽn và sức chứa.
    """

    def __init__(self, network: EvacuationNetwork, config: Optional[AlgorithmConfig] = None):
        """
        Khởi tạo thuật toán GBFS.

        Args:
            network: Mạng lưới sơ tán
            config: Cấu hình thuật toán (tùy chọn)
        """
        super().__init__(network)
        self.config = config or AlgorithmConfig()

        # Trọng số cho các thành phần heuristic
        self.w_dist = self.config.distance_weight
        self.w_risk = self.config.risk_weight
        self.w_congestion = self.config.congestion_weight
        self.w_capacity = self.config.capacity_weight

    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.GBFS

    def heuristic(self, node: Node, goal: Shelter, current_flow: Dict[str, int]) -> float:
        """
        Tính giá trị heuristic đa mục tiêu.

        Giá trị thấp hơn là tốt hơn.

        Args:
            node: Nút hiện tại
            goal: Nơi trú ẩn đích
            current_flow: Luồng hiện tại trên mỗi cạnh

        Returns:
            Ước tính chi phí heuristic
        """
        # Thành phần khoảng cách (chuẩn hóa theo khoảng cách tối đa điển hình ~30km)
        h_dist = haversine_distance(node.lat, node.lon, goal.lat, goal.lon) / 30.0

        # Thành phần rủi ro (từ các vùng nguy hiểm)
        h_risk = self.network.get_total_risk_at(node.lat, node.lon)

        # Thành phần tắc nghẽn (mức độ tắc nghẽn trung bình của các cạnh gần nút này)
        h_congestion = self._get_local_congestion(node.id)

        # Thành phần sức chứa (ưu tiên các nơi trú ẩn có nhiều sức chứa còn lại hơn)
        # Đảo ngược để giá trị thấp hơn là tốt hơn
        capacity_ratio = goal.occupancy_rate if goal.capacity > 0 else 1.0
        h_capacity = capacity_ratio

        # Kết hợp với trọng số
        return (self.w_dist * h_dist +
                self.w_risk * h_risk +
                self.w_congestion * h_congestion +
                self.w_capacity * h_capacity)

    def _get_local_congestion(self, node_id: str) -> float:
        """Lấy mức độ tắc nghẽn trung bình của các cạnh kết nối với một nút."""
        edges = self.network.get_outgoing_edges(node_id)
        if not edges:
            return 0.0
        return sum(e.congestion_level for e in edges) / len(edges)

    def find_path(self, source: PopulationZone,
                  shelters: List[Shelter]) -> Tuple[Optional[List[str]], Optional[Shelter], float]:
        """
        Tìm đường đi tốt nhất từ một khu vực dân cư đến bất kỳ nơi trú ẩn khả dụng nào.

        Args:
            source: Khu vực dân cư xuất phát
            shelters: Danh sách các nơi trú ẩn đích tiềm năng

        Returns:
            Tuple của (đường đi dưới dạng danh sách ID nút, nơi trú ẩn được chọn, tổng chi phí)
            Trả về (None, None, inf) nếu không tìm thấy đường đi
        """
        if not shelters:
            return None, None, float('inf')

        # Lọc các nơi trú ẩn có sức chứa và không nằm trong vùng nguy hiểm cao
        available_shelters = []
        for s in shelters:
            if not s.has_capacity():
                continue
            # Check if shelter is in high-risk hazard zone
            shelter_risk = self.network.get_total_risk_at(s.lat, s.lon)
            if shelter_risk > 0.6:  # Standardized risk threshold (0.6)
                continue
            available_shelters.append(s)

        if not available_shelters:
            # If no safe shelters, use any available shelters as fallback
            available_shelters = [s for s in shelters if s.has_capacity()]

        if not available_shelters:
            return None, None, float('inf')

        # Tạo tập hợp đích
        goal_ids = {s.id for s in available_shelters}
        shelter_map = {s.id: s for s in available_shelters}

        # Hàng đợi ưu tiên: (f_cost, counter, SearchNode)
        # Counter để phá vỡ sự ràng buộc tránh so sánh SearchNodes
        counter = 0
        open_set: List[Tuple[float, int, SearchNode]] = []

        # Tìm giao lộ gần nhất với nguồn
        source_node = self.network.find_nearest_node(source.lat, source.lon)
        if not source_node:
            return None, None, float('inf')

        # Khởi tạo với nút nguồn
        start = SearchNode(
            node_id=source_node.id,
            g_cost=0.0,
            h_cost=min(self.heuristic(source_node, s, {}) for s in available_shelters)
        )
        heapq.heappush(open_set, (start.f_cost, counter, start))
        counter += 1

        # Tập hợp đã thăm
        visited: Set[str] = set()
        current_flow: Dict[str, int] = {}

        while open_set:
            if self._should_stop:
                break

            _, _, current = heapq.heappop(open_set)

            # Bỏ qua nếu đã thăm
            if current.node_id in visited:
                continue
            visited.add(current.node_id)

            # Kiểm tra nếu đã đến đích (nơi trú ẩn)
            if current.node_id in goal_ids:
                # Tái tạo đường đi
                path = self._reconstruct_path(current)
                shelter = shelter_map[current.node_id]
                return path, shelter, current.g_cost

            # Mở rộng các láng giềng
            for neighbor_id in self.network.get_neighbors(current.node_id):
                if neighbor_id in visited:
                    continue

                # Lấy chi phí cạnh
                edge = self.network.get_edge_between(current.node_id, neighbor_id)
                if not edge or edge.is_blocked:
                    continue

                edge_cost = edge.get_cost(self.w_risk)
                new_g_cost = current.g_cost + edge_cost

                # Lấy nút láng giềng cho heuristic
                neighbor_node = self.network.get_node(neighbor_id)
                if not neighbor_node:
                    continue

                # Tính heuristic đến nơi trú ẩn gần nhất
                h_cost = min(
                    self.heuristic(neighbor_node, s, current_flow)
                    for s in available_shelters
                )

                neighbor = SearchNode(
                    node_id=neighbor_id,
                    g_cost=new_g_cost,
                    h_cost=h_cost,
                    parent=current
                )

                heapq.heappush(open_set, (neighbor.f_cost, counter, neighbor))
                counter += 1

        return None, None, float('inf')

    def _reconstruct_path(self, node: SearchNode) -> List[str]:
        """Tái tạo đường đi từ chuỗi SearchNode."""
        path = []
        current: Optional[SearchNode] = node
        while current:
            path.append(current.node_id)
            current = current.parent
        return list(reversed(path))

    def optimize(self, **kwargs) -> Tuple[EvacuationPlan, AlgorithmMetrics]:
        """
        Chạy tối ưu hóa GBFS cho tất cả các khu vực dân cư.

        Returns:
            Tuple của (EvacuationPlan, AlgorithmMetrics)
        """
        start_time = self._start_timer()

        plan = EvacuationPlan(algorithm_type=AlgorithmType.GBFS)
        zones = self.network.get_population_zones()
        shelters = self.network.get_shelters()

        if not zones or not shelters:
            self._stop_timer(start_time)
            return plan, self._metrics

        total_zones = len(zones)
        total_cost = 0.0
        paths_found = 0
        total_path_length = 0

        # Filter zones: only evacuate zones that are in hazard areas
        # Zones far from hazards (low risk) don't need evacuation
        zones_to_evacuate = []
        zones_skipped = []
        for z in zones:
            zone_risk = self.network.get_total_risk_at(z.lat, z.lon)
            if zone_risk >= self.config.min_zone_risk_for_evacuation:
                zones_to_evacuate.append((z, zone_risk))
            else:
                zones_skipped.append(z)

        # Sort by risk (highest first) then by population
        zones_to_evacuate.sort(key=lambda x: (-x[1], -x[0].population))
        zones = [z for z, _ in zones_to_evacuate]

        # Track remaining population per zone for multi-route support
        zone_remaining = {z.id: z.population for z in zones}

        for i, zone in enumerate(zones):
            if self._should_stop:
                break

            # Continue assigning until zone is fully evacuated or no more shelter capacity
            while zone_remaining[zone.id] >= self.config.min_flow_threshold:
                # Tìm đường đi cho khu vực này
                path, shelter, cost = self.find_path(zone, shelters)

                if not path or not shelter:
                    break

                # Tính toán các chỉ số tuyến đường
                route_distance = self._calculate_path_distance(path)
                route_time = self._calculate_path_time(path)
                route_risk = self._calculate_path_risk(path)

                # Use remaining population, capped by shelter capacity
                actual_flow = min(zone_remaining[zone.id], shelter.available_capacity)

                if actual_flow < self.config.min_flow_threshold:
                    break

                route = EvacuationRoute(
                    zone_id=zone.id,
                    shelter_id=shelter.id,
                    path=path,
                    flow=actual_flow,
                    distance_km=route_distance,
                    estimated_time_hours=route_time,
                    risk_score=route_risk
                )
                plan.add_route(route)

                # Cập nhật mức độ sử dụng nơi trú ẩn (để định tuyến nhận biết sức chứa)
                shelter.current_occupancy += actual_flow
                zone_remaining[zone.id] -= actual_flow

                total_cost += cost
                paths_found += 1
                total_path_length += len(path)

            # Báo cáo tiến trình
            self._metrics.convergence_history.append(total_cost)
            self.report_progress(i + 1, total_cost, plan)

        # Hoàn thiện các chỉ số
        self._stop_timer(start_time)
        self._metrics.iterations = len(zones)  # Only count zones that needed evacuation
        self._metrics.final_cost = total_cost
        self._metrics.routes_found = paths_found
        self._metrics.evacuees_covered = plan.total_evacuees
        self._metrics.average_path_length = (
            total_path_length / paths_found if paths_found > 0 else 0
        )

        # Tỷ lệ bao phủ: số người được sơ tán / min(dân số cần sơ tán, tổng sức chứa)
        # Only count population from zones that needed evacuation (in hazard areas)
        population_needing_evacuation = sum(z.population for z in zones)
        total_capacity = sum(s.capacity for s in shelters)
        max_possible = min(population_needing_evacuation, total_capacity)
        self._metrics.coverage_rate = (
            plan.total_evacuees / max_possible if max_possible > 0 else 0
        )

        return plan, self._metrics

    def _calculate_path_distance(self, path: List[str]) -> float:
        """Tính tổng khoảng cách của một đường đi tính bằng km."""
        if len(path) < 2:
            return 0.0

        total = 0.0
        for i in range(len(path) - 1):
            edge = self.network.get_edge_between(path[i], path[i + 1])
            if edge:
                total += edge.length_km
        return total

    def _calculate_path_time(self, path: List[str]) -> float:
        """Tính thời gian di chuyển ước tính tính bằng giờ."""
        if len(path) < 2:
            return 0.0

        total = 0.0
        for i in range(len(path) - 1):
            edge = self.network.get_edge_between(path[i], path[i + 1])
            if edge:
                total += edge.current_travel_time
        return total

    def _calculate_path_risk(self, path: List[str]) -> float:
        """Tính rủi ro trung bình dọc theo một đường đi."""
        if not path:
            return 0.0

        total_risk = 0.0
        for node_id in path:
            node = self.network.get_node(node_id)
            if node:
                total_risk += self.network.get_total_risk_at(node.lat, node.lon)

        return total_risk / len(path)

    def find_multiple_paths(self, source: PopulationZone,
                           shelters: List[Shelter],
                           k: int = 3) -> List[Tuple[List[str], Shelter, float]]:
        """
        Tìm k đường đi tốt nhất từ nguồn đến các nơi trú ẩn.

        Hữu ích cho phân phối luồng qua nhiều tuyến đường.

        Args:
            source: Khu vực xuất phát
            shelters: Các nơi trú ẩn khả dụng
            k: Số lượng đường đi cần tìm

        Returns:
            Danh sách các tuple (đường đi, nơi trú ẩn, chi phí)
        """
        paths = []
        used_shelters: Set[str] = set()

        for _ in range(k):
            # Loại trừ các nơi trú ẩn đã sử dụng
            available = [s for s in shelters
                        if s.id not in used_shelters and s.has_capacity()]
            if not available:
                break

            path, shelter, cost = self.find_path(source, available)
            if path and shelter:
                paths.append((path, shelter, cost))
                used_shelters.add(shelter.id)
            else:
                break

        return paths
