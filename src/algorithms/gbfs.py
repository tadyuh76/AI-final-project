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
    h_cost: float  # Chi phí heuristic đến đích (GBFS chỉ dùng h, không dùng g)
    parent: Optional['SearchNode'] = None

    def __lt__(self, other: 'SearchNode') -> bool:
        """So sánh cho hàng đợi ưu tiên (h_cost thấp hơn là tốt hơn)."""
        return self.h_cost < other.h_cost


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

    def heuristic(self, node: Node, goal: Shelter,
                  node_risk: float, node_congestion: float) -> float:
        """
        Tính giá trị heuristic đa mục tiêu cho GBFS.

        Kết hợp 4 yếu tố:
        - Khoảng cách đến shelter
        - Rủi ro tại vị trí hiện tại (pre-computed)
        - Tắc nghẽn giao thông (pre-computed)
        - Sức chứa còn lại của shelter

        Args:
            node: Nút hiện tại
            goal: Nơi trú ẩn đích
            node_risk: Rủi ro đã tính trước tại node
            node_congestion: Tắc nghẽn đã tính trước tại node

        Returns:
            Giá trị heuristic (thấp hơn là tốt hơn)
        """
        # 1. Khoảng cách đến shelter (km)
        h_dist = haversine_distance(node.lat, node.lon, goal.lat, goal.lon)

        # 2. Rủi ro tại vị trí hiện tại (0-1) - đã tính trước
        h_risk = node_risk

        # 3. Tắc nghẽn cục bộ (0-1) - đã tính trước
        h_congestion = node_congestion

        # 4. Penalty cho shelter gần đầy (0-10)
        if goal.capacity > 0:
            h_capacity = goal.occupancy_rate * 5.0  # 0-5 based on fullness
        else:
            h_capacity = 10.0  # No capacity = high penalty

        # Kết hợp với trọng số
        h = (self.w_dist * h_dist +
             self.w_risk * h_risk * 10.0 +  # Scale risk to be comparable
             self.w_congestion * h_congestion * 5.0 +
             self.w_capacity * h_capacity)

        return h

    def _get_local_congestion(self, node_id: str) -> float:
        """Lấy mức độ tắc nghẽn trung bình của các cạnh kết nối với một nút."""
        edges = self.network.get_outgoing_edges(node_id)
        if not edges:
            return 0.0
        return sum(e.congestion_level for e in edges) / len(edges)

    def find_path(self, source: PopulationZone,
                  shelters: List[Shelter],
                  allow_emergency: bool = False) -> Tuple[Optional[List[str]], Optional[Shelter], float]:
        """
        Tìm đường đi tốt nhất từ một khu vực dân cư đến bất kỳ nơi trú ẩn khả dụng nào.

        Args:
            source: Khu vực dân cư xuất phát
            shelters: Danh sách các nơi trú ẩn đích tiềm năng
            allow_emergency: Cho phép đi qua vùng nguy hiểm cao (cho zones bị kẹt)

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
            # In emergency mode, allow shelters in moderate risk areas
            risk_threshold = 0.8 if allow_emergency else 0.6
            if shelter_risk > risk_threshold:
                continue
            available_shelters.append(s)

        if not available_shelters:
            # If no safe shelters, use any available shelters as fallback
            available_shelters = [s for s in shelters if s.has_capacity()]

        if not available_shelters:
            return None, None, float('inf')

        # Lấy source node
        zone_node = self.network.get_node(source.id)
        if zone_node:
            source_node = zone_node
        else:
            source_node = self.network.find_nearest_node(source.lat, source.lon)

        if not source_node:
            return None, None, float('inf')

        # GBFS thuần túy: chọn MỘT shelter tốt nhất dựa trên heuristic từ source
        # Sắp xếp shelters theo heuristic và thử từng cái cho đến khi tìm được đường
        source_risk = self.network.get_total_risk_at(source_node.lat, source_node.lon)
        source_congestion = self._get_local_congestion(source_node.id)

        # Xếp hạng shelters theo heuristic
        shelter_rankings = []
        for s in available_shelters:
            h = self.heuristic(source_node, s, source_risk, source_congestion)
            shelter_rankings.append((h, s))
        shelter_rankings.sort(key=lambda x: x[0])

        # Thử từng shelter theo thứ tự ưu tiên (tối đa 5)
        for _, target_shelter in shelter_rankings[:5]:
            path, cost = self._search_to_shelter(source_node, target_shelter, allow_emergency)
            if path:
                return path, target_shelter, cost

        return None, None, float('inf')

    def _search_to_shelter(self, source_node: Node, target_shelter: Shelter,
                           allow_emergency: bool) -> Tuple[Optional[List[str]], float]:
        """
        Tìm đường đến MỘT shelter cụ thể - GBFS thuần túy.

        Args:
            source_node: Nút xuất phát
            target_shelter: Shelter đích duy nhất
            allow_emergency: Cho phép đi qua vùng nguy hiểm

        Returns:
            Tuple của (path, cost) hoặc (None, inf)
        """
        # Goal có thể là shelter trực tiếp hoặc nút gần nhất
        goal_id = target_shelter.id
        nearest_to_shelter = self.network.find_nearest_node(target_shelter.lat, target_shelter.lon)
        nearest_id = nearest_to_shelter.id if nearest_to_shelter and nearest_to_shelter.id != goal_id else None

        # Hàng đợi ưu tiên
        counter = 0
        open_set: List[Tuple[float, int, SearchNode]] = []

        # Khởi tạo
        start_h = self.heuristic(source_node, target_shelter,
                                 self.network.get_total_risk_at(source_node.lat, source_node.lon),
                                 self._get_local_congestion(source_node.id))
        start = SearchNode(node_id=source_node.id, h_cost=start_h)
        heapq.heappush(open_set, (start.h_cost, counter, start))
        counter += 1

        visited: Set[str] = set()

        while open_set:
            if self._should_stop:
                break

            _, _, current = heapq.heappop(open_set)

            if current.node_id in visited:
                continue
            visited.add(current.node_id)

            # Đã đến shelter trực tiếp
            if current.node_id == goal_id:
                return self._reconstruct_path(current), current.h_cost

            # Đã đến nút gần shelter - kiểm tra cạnh trực tiếp
            if current.node_id == nearest_id:
                edge_to_shelter = self.network.get_edge_between(current.node_id, goal_id)
                if edge_to_shelter and not edge_to_shelter.is_blocked:
                    path = self._reconstruct_path(current)
                    path.append(goal_id)
                    return path, current.h_cost

            # Mở rộng neighbors
            for neighbor_id in self.network.get_neighbors(current.node_id):
                if neighbor_id in visited:
                    continue

                edge = self.network.get_edge_between(current.node_id, neighbor_id)
                if not edge or edge.is_blocked:
                    continue

                if not allow_emergency and edge.flood_risk > 0.6:
                    continue

                neighbor_node = self.network.get_node(neighbor_id)
                if not neighbor_node:
                    continue

                # Heuristic đến MỘT shelter duy nhất
                neighbor_risk = self.network.get_total_risk_at(neighbor_node.lat, neighbor_node.lon)
                neighbor_congestion = self._get_local_congestion(neighbor_id)
                h_cost = self.heuristic(neighbor_node, target_shelter, neighbor_risk, neighbor_congestion)

                neighbor = SearchNode(node_id=neighbor_id, h_cost=h_cost, parent=current)
                heapq.heappush(open_set, (neighbor.h_cost, counter, neighbor))
                counter += 1

        return None, float('inf')

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
                # Tìm đường đi cho khu vực này (normal mode first)
                path, shelter, cost = self.find_path(zone, shelters, allow_emergency=False)

                # If no path found, retry with emergency mode (allows traversing high-risk areas)
                if not path or not shelter:
                    path, shelter, cost = self.find_path(zone, shelters, allow_emergency=True)

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

                # Tính chi phí theo công thức chuẩn (giống GWO)
                route_cost = actual_flow * (
                    route_time +
                    0.3 * route_risk +
                    0.001 * route_distance
                )
                total_cost += route_cost
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
                           k: int = 3,
                           allow_emergency: bool = False) -> List[Tuple[List[str], Shelter, float]]:
        """
        Tìm k đường đi tốt nhất từ nguồn đến các nơi trú ẩn.

        Hữu ích cho phân phối luồng qua nhiều tuyến đường.

        Args:
            source: Khu vực xuất phát
            shelters: Các nơi trú ẩn khả dụng
            k: Số lượng đường đi cần tìm
            allow_emergency: Cho phép đi qua vùng nguy hiểm cao

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

            path, shelter, cost = self.find_path(source, available, allow_emergency=allow_emergency)
            if path and shelter:
                paths.append((path, shelter, cost))
                used_shelters.add(shelter.id)
            else:
                break

        return paths
