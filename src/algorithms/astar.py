"""
Thuật toán A* cho tìm đường sơ tán tối ưu.

A* sử dụng công thức f(n) = g(n) + h(n):
- g(n): Chi phí thực tế từ điểm bắt đầu đến n
- h(n): Chi phí ước tính (heuristic) từ n đến đích

A* đảm bảo tìm đường đi tối ưu (optimal) nếu heuristic là admissible.
So sánh với GBFS chỉ dùng h(n), A* cân bằng giữa chi phí thực tế và ước tính.
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
class AStarNode:
    """Nút trong cây tìm kiếm A*."""
    node_id: str
    g_cost: float  # Chi phí thực tế từ điểm bắt đầu
    h_cost: float  # Chi phí heuristic đến đích
    parent: Optional['AStarNode'] = None

    @property
    def f_cost(self) -> float:
        """Tổng chi phí ước tính: f(n) = g(n) + h(n)."""
        return self.g_cost + self.h_cost

    def __lt__(self, other: 'AStarNode') -> bool:
        """So sánh cho hàng đợi ưu tiên (f_cost thấp hơn là tốt hơn)."""
        return self.f_cost < other.f_cost


class AStarSearch(BaseAlgorithm):
    """
    Thuật toán A* cho tìm đường sơ tán tối ưu.

    Tìm đường đi tối ưu từ các khu vực dân cư đến các nơi trú ẩn sử dụng
    công thức f(n) = g(n) + h(n).

    So với GBFS:
    - A* đảm bảo tìm đường đi tối ưu
    - Chậm hơn nhưng chất lượng tốt hơn
    - Phù hợp làm baseline để so sánh các thuật toán khác
    """

    def __init__(self, network: EvacuationNetwork, config: Optional[AlgorithmConfig] = None):
        """
        Khởi tạo thuật toán A*.

        Args:
            network: Mạng lưới sơ tán
            config: Cấu hình thuật toán (tùy chọn)
        """
        super().__init__(network)
        self.config = config or AlgorithmConfig()

        # Path cache: (zone_id, shelter_id) -> (path, distance, time, risk)
        self._path_cache: Dict[Tuple[str, str], Tuple[List[str], float, float, float]] = {}

        # Trọng số cho các thành phần heuristic
        self.w_dist = self.config.distance_weight
        self.w_risk = self.config.risk_weight
        self.w_congestion = self.config.congestion_weight
        self.w_capacity = self.config.capacity_weight

    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.ASTAR

    def heuristic(self, node: Node, goal: Shelter, current_flow: Dict[str, int]) -> float:
        """
        Tính giá trị heuristic đa mục tiêu (admissible).

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

    def _precompute_paths(self, zones: List[PopulationZone], shelters: List[Shelter]) -> None:
        """
        Tính trước tất cả đường đi từ zones đến shelters sử dụng A*.
        Kết quả được cache để sử dụng sau.
        """
        self._path_cache.clear()

        for zone in zones:
            for shelter in shelters:
                # Kiểm tra shelter có ở vùng rủi ro cao không
                shelter_risk = self.network.get_total_risk_at(shelter.lat, shelter.lon)
                if shelter_risk > 0.6:
                    continue  # Bỏ qua shelter không an toàn

                # Tìm đường đi bằng A* (optimal)
                path, distance, time_h, risk = self._find_path_astar(zone, shelter)

                if path:
                    self._path_cache[(zone.id, shelter.id)] = (path, distance, time_h, risk)

    def _find_path_astar(self, zone: PopulationZone, shelter: Shelter) -> Tuple[List[str], float, float, float]:
        """
        Tìm đường đi từ zone đến shelter sử dụng A* (f = g + h).

        Returns:
            Tuple của (path, distance_km, time_hours, max_risk)
        """
        # Lấy node zone và shelter từ mạng
        zone_node = self.network.get_node(zone.id)
        if not zone_node:
            zone_node = self.network.find_nearest_node(zone.lat, zone.lon)

        shelter_node = self.network.get_node(shelter.id)
        shelter_nearest = self.network.find_nearest_node(shelter.lat, shelter.lon)

        if not zone_node:
            return [], 0, 0, 1.0

        target_node = shelter_node if shelter_node else shelter_nearest
        if not target_node:
            return [], 0, 0, 1.0

        # A* search - f(n) = g(n) + h(n)
        counter = 0
        start_h = haversine_distance(zone_node.lat, zone_node.lon, target_node.lat, target_node.lon)

        # open_set: (f_cost, counter, node_id, path, g_cost, total_dist, total_time, max_risk)
        # A*: priority = f(n) = g(n) + h(n)
        if self.network.get_node(zone.id):
            open_set = [(start_h, counter, zone_node.id, [zone_node.id], 0.0, 0.0, 0.0, 0.0)]
        else:
            open_set = [(start_h, counter, zone_node.id, [zone.id, zone_node.id], 0.0, 0.0, 0.0, 0.0)]

        # Track best g_cost to each node (for optimality)
        best_g: Dict[str, float] = {zone_node.id: 0.0}

        while open_set:
            f, _, current_id, path, g_cost, total_dist, total_time, max_risk = heapq.heappop(open_set)

            # Skip if we've found a better path to this node
            if current_id in best_g and g_cost > best_g[current_id]:
                continue

            # Cập nhật max_risk
            current_node = self.network.get_node(current_id)
            if current_node:
                node_risk = self.network.get_total_risk_at(current_node.lat, current_node.lon)
                max_risk = max(max_risk, node_risk)

            # Kiểm tra đã đến shelter
            if current_id == shelter.id:
                return path, total_dist, total_time, max_risk

            # Kiểm tra đã đến node gần shelter
            if shelter_nearest and current_id == shelter_nearest.id:
                edge_to_shelter = self.network.get_edge_between(current_id, shelter.id)
                if edge_to_shelter and not edge_to_shelter.is_blocked:
                    final_path = path + [shelter.id]
                    final_dist = total_dist + edge_to_shelter.length_km
                    final_time = total_time + edge_to_shelter.current_travel_time
                    shelter_risk = self.network.get_total_risk_at(shelter.lat, shelter.lon)
                    final_risk = max(max_risk, edge_to_shelter.flood_risk, shelter_risk)
                    return final_path, final_dist, final_time, final_risk

            # Mở rộng neighbors
            for neighbor_id in self.network.get_neighbors(current_id):
                edge = self.network.get_edge_between(current_id, neighbor_id)
                if not edge or edge.is_blocked:
                    continue

                # Skip high-risk edges
                if edge.flood_risk > 0.6:
                    continue

                neighbor_node = self.network.get_node(neighbor_id)
                if neighbor_node:
                    # A*: g(n) = actual cost from start
                    # Use edge cost: travel_time + risk_penalty
                    edge_cost = edge.current_travel_time + edge.flood_risk * self.w_risk
                    new_g = g_cost + edge_cost

                    # Only expand if this is a better path
                    if neighbor_id in best_g and new_g >= best_g[neighbor_id]:
                        continue
                    best_g[neighbor_id] = new_g

                    # h(n) = heuristic estimate to goal
                    h_dist = haversine_distance(neighbor_node.lat, neighbor_node.lon,
                                               target_node.lat, target_node.lon)
                    h_risk = edge.flood_risk * self.w_risk
                    h = h_dist + h_risk

                    # f(n) = g(n) + h(n)
                    f_cost = new_g + h

                    counter += 1

                    new_dist = total_dist + edge.length_km
                    new_time = total_time + edge.current_travel_time
                    new_risk = max(max_risk, edge.flood_risk)

                    heapq.heappush(open_set, (
                        f_cost, counter, neighbor_id, path + [neighbor_id],
                        new_g, new_dist, new_time, new_risk
                    ))

        # Không tìm thấy đường
        return [], 0, 0, 1.0

    def find_path(self, source: PopulationZone,
                  shelters: List[Shelter],
                  allow_emergency: bool = False) -> Tuple[Optional[List[str]], Optional[Shelter], float]:
        """
        Tìm đường đi tối ưu từ một khu vực dân cư đến bất kỳ nơi trú ẩn khả dụng nào.

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
            shelter_risk = self.network.get_total_risk_at(s.lat, s.lon)
            risk_threshold = 0.8 if allow_emergency else 0.6
            if shelter_risk > risk_threshold:
                continue
            available_shelters.append(s)

        if not available_shelters:
            available_shelters = [s for s in shelters if s.has_capacity()]

        if not available_shelters:
            return None, None, float('inf')

        # ============ TRY CACHE FIRST ============
        best_cached = None
        best_cached_cost = float('inf')
        for s in available_shelters:
            cache_key = (source.id, s.id)
            if cache_key in self._path_cache:
                path, distance, time_h, risk = self._path_cache[cache_key]
                cost = time_h + 0.3 * risk + 0.001 * distance
                if cost < best_cached_cost:
                    best_cached_cost = cost
                    best_cached = (path, s, cost)

        if best_cached:
            return best_cached

        # ============ FALLBACK TO A* ============
        goal_ids = set()
        shelter_map = {}
        shelter_nearest_nodes = {}

        for s in available_shelters:
            goal_ids.add(s.id)
            shelter_map[s.id] = s
            nearest_to_shelter = self.network.find_nearest_node(s.lat, s.lon)
            if nearest_to_shelter and nearest_to_shelter.id != s.id:
                shelter_nearest_nodes[nearest_to_shelter.id] = s

        counter = 0
        open_set: List[Tuple[float, int, AStarNode]] = []

        zone_node = self.network.get_node(source.id)
        if zone_node:
            source_node = zone_node
        else:
            source_node = self.network.find_nearest_node(source.lat, source.lon)

        if not source_node:
            return None, None, float('inf')

        start = AStarNode(
            node_id=source_node.id,
            g_cost=0.0,
            h_cost=min(self.heuristic(source_node, s, {}) for s in available_shelters)
        )
        heapq.heappush(open_set, (start.f_cost, counter, start))
        counter += 1

        visited: Set[str] = set()
        best_g: Dict[str, float] = {source_node.id: 0.0}
        current_flow: Dict[str, int] = {}

        while open_set:
            if self._should_stop:
                break

            _, _, current = heapq.heappop(open_set)

            if current.node_id in visited:
                continue
            visited.add(current.node_id)

            if current.node_id in goal_ids:
                path = self._reconstruct_path(current)
                shelter = shelter_map[current.node_id]
                return path, shelter, current.g_cost

            if current.node_id in shelter_nearest_nodes:
                shelter = shelter_nearest_nodes[current.node_id]
                edge_to_shelter = self.network.get_edge_between(current.node_id, shelter.id)
                if edge_to_shelter and not edge_to_shelter.is_blocked:
                    path = self._reconstruct_path(current)
                    path.append(shelter.id)
                    final_cost = current.g_cost + edge_to_shelter.get_cost(self.w_risk, allow_emergency)
                    return path, shelter, final_cost

            for neighbor_id in self.network.get_neighbors(current.node_id):
                if neighbor_id in visited:
                    continue

                edge = self.network.get_edge_between(current.node_id, neighbor_id)
                if not edge or edge.is_blocked:
                    continue

                edge_cost = edge.get_cost(self.w_risk, allow_emergency=allow_emergency)
                new_g_cost = current.g_cost + edge_cost

                if new_g_cost == float('inf'):
                    continue

                # A* optimization: skip if we found a better path
                if neighbor_id in best_g and new_g_cost >= best_g[neighbor_id]:
                    continue
                best_g[neighbor_id] = new_g_cost

                neighbor_node = self.network.get_node(neighbor_id)
                if not neighbor_node:
                    continue

                h_cost = min(
                    self.heuristic(neighbor_node, s, current_flow)
                    for s in available_shelters
                )

                neighbor = AStarNode(
                    node_id=neighbor_id,
                    g_cost=new_g_cost,
                    h_cost=h_cost,
                    parent=current
                )

                heapq.heappush(open_set, (neighbor.f_cost, counter, neighbor))
                counter += 1

        return None, None, float('inf')

    def _reconstruct_path(self, node: AStarNode) -> List[str]:
        """Tái tạo đường đi từ chuỗi AStarNode."""
        path = []
        current: Optional[AStarNode] = node
        while current:
            path.append(current.node_id)
            current = current.parent
        return list(reversed(path))

    def optimize(self, **kwargs) -> Tuple[EvacuationPlan, AlgorithmMetrics]:
        """
        Chạy tối ưu hóa A* cho tất cả các khu vực dân cư.

        Returns:
            Tuple của (EvacuationPlan, AlgorithmMetrics)
        """
        start_time = self._start_timer()

        plan = EvacuationPlan(algorithm_type=AlgorithmType.ASTAR)
        zones = self.network.get_population_zones()
        shelters = self.network.get_shelters()

        if not zones or not shelters:
            self._stop_timer(start_time)
            return plan, self._metrics

        total_cost = 0.0
        paths_found = 0
        total_path_length = 0

        # Filter zones: only evacuate zones in hazard areas
        zones_to_evacuate = []
        for z in zones:
            zone_risk = self.network.get_total_risk_at(z.lat, z.lon)
            if zone_risk >= self.config.min_zone_risk_for_evacuation:
                zones_to_evacuate.append((z, zone_risk))

        # Sort by risk (highest first) then by population
        # Zones in the CENTER of hazard areas (highest risk) get PRIORITY
        zones_to_evacuate.sort(key=lambda x: (-x[1], -x[0].population))
        zones = [z for z, _ in zones_to_evacuate]

        # ============ GREEDY PRIORITY ALLOCATION ============
        # Ưu tiên sơ tán zones có rủi ro CAO NHẤT trước (greedy by risk)
        # Zones ở trung tâm vùng nguy hiểm được ưu tiên tối đa
        total_capacity = sum(s.capacity for s in shelters)
        remaining_capacity = total_capacity

        # Phân bổ theo thứ tự ưu tiên rủi ro (greedy)
        zone_allocation = {}
        for z in zones:
            # Zone có rủi ro cao nhất được phân bổ TOÀN BỘ dân số (nếu còn capacity)
            # Ưu tiên tuyệt đối cho zones nguy hiểm nhất
            needed = z.population
            allocation = min(needed, remaining_capacity)
            zone_allocation[z.id] = allocation
            remaining_capacity -= allocation

            if remaining_capacity <= 0:
                break

        # Zones còn lại (nếu hết capacity) không được phân bổ
        for z in zones:
            if z.id not in zone_allocation:
                zone_allocation[z.id] = 0

        # Track remaining allocation per zone
        zone_remaining = {z.id: zone_allocation[z.id] for z in zones}

        # ============ PRE-COMPUTE PATHS ============
        self._precompute_paths(zones, shelters)

        for i, zone in enumerate(zones):
            if self._should_stop:
                break

            while zone_remaining[zone.id] >= self.config.min_flow_threshold:
                path, shelter, cost = self.find_path(zone, shelters, allow_emergency=False)

                if not path or not shelter:
                    path, shelter, cost = self.find_path(zone, shelters, allow_emergency=True)

                if not path or not shelter:
                    break

                route_distance = self._calculate_path_distance(path)
                route_time = self._calculate_path_time(path)
                route_risk = self._calculate_path_risk(path)

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

                shelter.current_occupancy += actual_flow
                zone_remaining[zone.id] -= actual_flow

                route_cost = actual_flow * (
                    route_time +
                    0.3 * route_risk +
                    0.001 * route_distance
                )
                total_cost += route_cost
                paths_found += 1
                total_path_length += len(path)

            self._metrics.convergence_history.append(total_cost)
            self.report_progress(i + 1, total_cost, plan)

        self._stop_timer(start_time)
        self._metrics.iterations = len(zones)
        self._metrics.final_cost = total_cost
        self._metrics.routes_found = paths_found
        self._metrics.evacuees_covered = plan.total_evacuees
        self._metrics.average_path_length = (
            total_path_length / paths_found if paths_found > 0 else 0
        )

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
