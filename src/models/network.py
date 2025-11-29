"""
Mô hình đồ thị mạng lưới cho hệ thống sơ tán.
Bao bọc đồ thị NetworkX với chức năng chuyên biệt cho lĩnh vực.
"""

from typing import Dict, List, Optional, Tuple, Iterator, Set
from dataclasses import dataclass, field
import json

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

from .node import (
    Node, NodeType, PopulationZone, Shelter, HazardZone,
    haversine_distance, lat_lon_to_mercator
)
from .edge import Edge, RoadType


@dataclass
class NetworkStats:
    """Thống kê về mạng lưới."""
    total_nodes: int = 0
    total_edges: int = 0
    population_zones: int = 0
    shelters: int = 0
    total_population: int = 0
    total_shelter_capacity: int = 0
    total_road_length_km: float = 0.0
    blocked_edges: int = 0


class EvacuationNetwork:
    """
    Biểu diễn đồ thị của mạng lưới sơ tán.
    Chứa các nút (khu vực, nơi trú ẩn, giao lộ) và các cạnh (đường).
    """

    def __init__(self):
        """Khởi tạo mạng lưới trống."""
        # Cấu trúc đồ thị cốt lõi
        if HAS_NETWORKX:
            self._graph = nx.DiGraph()
        else:
            self._graph = None
            self._adjacency: Dict[str, Dict[str, str]] = {}  # node_id -> {neighbor_id: edge_id}

        # Lưu trữ nút theo loại
        self._nodes: Dict[str, Node] = {}
        self._population_zones: Dict[str, PopulationZone] = {}
        self._shelters: Dict[str, Shelter] = {}

        # Lưu trữ cạnh
        self._edges: Dict[str, Edge] = {}

        # Vùng nguy hiểm (không phải phần của đồ thị, nhưng ảnh hưởng chi phí cạnh)
        self._hazard_zones: List[HazardZone] = []

        # Giới hạn cho trực quan hóa
        self._min_lat: float = float('inf')
        self._max_lat: float = float('-inf')
        self._min_lon: float = float('inf')
        self._max_lon: float = float('-inf')

    # ==================== Thao tác Nút ====================

    def add_node(self, node: Node) -> None:
        """Thêm một nút vào mạng lưới."""
        self._nodes[node.id] = node
        self._update_bounds(node.lat, node.lon)

        if HAS_NETWORKX:
            self._graph.add_node(node.id, data=node)
        else:
            if node.id not in self._adjacency:
                self._adjacency[node.id] = {}

        # Theo dõi theo loại
        if isinstance(node, PopulationZone):
            self._population_zones[node.id] = node
        elif isinstance(node, Shelter):
            self._shelters[node.id] = node

    def get_node(self, node_id: str) -> Optional[Node]:
        """Lấy một nút theo ID."""
        return self._nodes.get(node_id)

    def get_nodes(self) -> Iterator[Node]:
        """Lặp qua tất cả các nút."""
        return iter(self._nodes.values())

    def get_population_zones(self) -> List[PopulationZone]:
        """Lấy tất cả các khu vực dân cư."""
        return list(self._population_zones.values())

    def get_shelters(self) -> List[Shelter]:
        """Lấy tất cả các nơi trú ẩn."""
        return list(self._shelters.values())

    def get_active_shelters(self) -> List[Shelter]:
        """Lấy các nơi trú ẩn đang hoạt động và có sức chứa."""
        return [s for s in self._shelters.values() if s.is_active and s.has_capacity()]

    # ==================== Thao tác Cạnh ====================

    def add_edge(self, edge: Edge) -> None:
        """Thêm một cạnh vào mạng lưới."""
        self._edges[edge.id] = edge

        if HAS_NETWORKX:
            self._graph.add_edge(edge.source_id, edge.target_id, edge_id=edge.id, data=edge)
            if not edge.is_oneway:
                self._graph.add_edge(edge.target_id, edge.source_id, edge_id=edge.id, data=edge)
        else:
            if edge.source_id not in self._adjacency:
                self._adjacency[edge.source_id] = {}
            self._adjacency[edge.source_id][edge.target_id] = edge.id

            if not edge.is_oneway:
                if edge.target_id not in self._adjacency:
                    self._adjacency[edge.target_id] = {}
                self._adjacency[edge.target_id][edge.source_id] = edge.id

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Lấy một cạnh theo ID."""
        return self._edges.get(edge_id)

    def get_edge_between(self, source_id: str, target_id: str) -> Optional[Edge]:
        """Lấy cạnh giữa hai nút."""
        if HAS_NETWORKX and self._graph.has_edge(source_id, target_id):
            edge_id = self._graph[source_id][target_id].get('edge_id')
            return self._edges.get(edge_id)
        elif not HAS_NETWORKX:
            edge_id = self._adjacency.get(source_id, {}).get(target_id)
            if edge_id:
                return self._edges.get(edge_id)
        return None

    def get_edges(self) -> Iterator[Edge]:
        """Lặp qua tất cả các cạnh."""
        return iter(self._edges.values())

    def get_neighbors(self, node_id: str) -> List[str]:
        """Lấy các ID nút láng giềng cho một nút."""
        if HAS_NETWORKX:
            return list(self._graph.successors(node_id))
        else:
            return list(self._adjacency.get(node_id, {}).keys())

    def get_outgoing_edges(self, node_id: str) -> List[Edge]:
        """Lấy tất cả các cạnh đi ra từ một nút."""
        edges = []
        for neighbor_id in self.get_neighbors(node_id):
            edge = self.get_edge_between(node_id, neighbor_id)
            if edge:
                edges.append(edge)
        return edges

    # ==================== Thao tác Vùng nguy hiểm ====================

    def add_hazard_zone(self, hazard: HazardZone) -> None:
        """Thêm một vùng nguy hiểm."""
        self._hazard_zones.append(hazard)
        self._update_edge_risks()

    def remove_hazard_zone(self, index: int) -> None:
        """Loại bỏ một vùng nguy hiểm theo chỉ số."""
        if 0 <= index < len(self._hazard_zones):
            self._hazard_zones.pop(index)
            self._update_edge_risks()

    def get_hazard_zones(self) -> List[HazardZone]:
        """Lấy tất cả các vùng nguy hiểm."""
        return self._hazard_zones

    def clear_hazard_zones(self) -> None:
        """Xóa tất cả các vùng nguy hiểm."""
        self._hazard_zones.clear()
        for edge in self._edges.values():
            edge.set_flood_risk(0.0)
            edge.is_blocked = False

    def _update_edge_risks(self) -> None:
        """Cập nhật rủi ro lũ lụt của cạnh dựa trên vùng nguy hiểm."""
        for edge in self._edges.values():
            source = self._nodes.get(edge.source_id)
            target = self._nodes.get(edge.target_id)
            if not source or not target:
                continue

            # Tính rủi ro tại điểm giữa của cạnh
            mid_lat = (source.lat + target.lat) / 2
            mid_lon = (source.lon + target.lon) / 2

            max_risk = 0.0
            for hazard in self._hazard_zones:
                risk = hazard.get_risk_at(mid_lat, mid_lon)
                max_risk = max(max_risk, risk)

            edge.set_flood_risk(max_risk)

    # ==================== Tính toán Chi phí ====================

    def get_edge_cost(self, source_id: str, target_id: str,
                      risk_weight: float = 0.3) -> float:
        """Lấy chi phí để đi qua một cạnh."""
        edge = self.get_edge_between(source_id, target_id)
        if not edge:
            return float('inf')
        return edge.get_cost(risk_weight)

    def get_total_risk_at(self, lat: float, lon: float) -> float:
        """Lấy mức độ rủi ro kết hợp tại một điểm từ tất cả các vùng nguy hiểm."""
        total_risk = 0.0
        for hazard in self._hazard_zones:
            total_risk = max(total_risk, hazard.get_risk_at(lat, lon))
        return min(1.0, total_risk)

    # ==================== Quản lý Trạng thái ====================

    def reset_simulation_state(self) -> None:
        """Đặt lại tất cả trạng thái động (luồng, lấp đầy, số lượng đã sơ tán)."""
        # Đặt lại luồng cạnh
        for edge in self._edges.values():
            edge.reset_flow()

        # Đặt lại lấp đầy nơi trú ẩn
        for shelter in self._shelters.values():
            shelter.current_occupancy = 0

        # Đặt lại số lượng đã sơ tán
        for zone in self._population_zones.values():
            zone.evacuated = 0

    def reset_hazards(self) -> None:
        """Đặt lại ảnh hưởng của nguy hiểm trên các cạnh."""
        for edge in self._edges.values():
            edge.flood_risk = 0.0
            edge.is_blocked = False

    # ==================== Thống kê ====================

    def get_stats(self) -> NetworkStats:
        """Lấy thống kê mạng lưới."""
        stats = NetworkStats()
        stats.total_nodes = len(self._nodes)
        stats.total_edges = len(self._edges)
        stats.population_zones = len(self._population_zones)
        stats.shelters = len(self._shelters)
        stats.total_population = sum(z.population for z in self._population_zones.values())
        stats.total_shelter_capacity = sum(s.capacity for s in self._shelters.values())
        stats.total_road_length_km = sum(e.length_km for e in self._edges.values())
        stats.blocked_edges = sum(1 for e in self._edges.values() if e.is_blocked)
        return stats

    # ==================== Giới hạn ====================

    def _update_bounds(self, lat: float, lon: float) -> None:
        """Cập nhật giới hạn địa lý."""
        self._min_lat = min(self._min_lat, lat)
        self._max_lat = max(self._max_lat, lat)
        self._min_lon = min(self._min_lon, lon)
        self._max_lon = max(self._max_lon, lon)

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Lấy giới hạn địa lý (min_lat, max_lat, min_lon, max_lon)."""
        return (self._min_lat, self._max_lat, self._min_lon, self._max_lon)

    def get_center(self) -> Tuple[float, float]:
        """Lấy tâm địa lý."""
        return (
            (self._min_lat + self._max_lat) / 2,
            (self._min_lon + self._max_lon) / 2
        )

    # ==================== Trợ giúp Tìm đường ====================

    def find_nearest_node(self, lat: float, lon: float,
                          node_type: Optional[NodeType] = None) -> Optional[Node]:
        """Tìm nút gần nhất đến một điểm cho trước."""
        min_dist = float('inf')
        nearest = None

        for node in self._nodes.values():
            if node_type and node.node_type != node_type:
                continue

            dist = haversine_distance(lat, lon, node.lat, node.lon)
            if dist < min_dist:
                min_dist = dist
                nearest = node

        return nearest

    def find_nearest_shelter(self, lat: float, lon: float,
                            min_capacity: int = 1) -> Optional[Shelter]:
        """Tìm nơi trú ẩn gần nhất có sức chứa khả dụng."""
        min_dist = float('inf')
        nearest = None

        for shelter in self._shelters.values():
            if not shelter.has_capacity(min_capacity):
                continue

            dist = haversine_distance(lat, lon, shelter.lat, shelter.lon)
            if dist < min_dist:
                min_dist = dist
                nearest = shelter

        return nearest

    # ==================== Tuần tự hóa ====================

    def to_dict(self) -> dict:
        """Chuyển đổi mạng lưới thành từ điển để tuần tự hóa."""
        return {
            'nodes': [
                {
                    'id': n.id,
                    'lat': n.lat,
                    'lon': n.lon,
                    'type': n.node_type.value,
                    'name': n.name,
                    **({'population': n.population, 'district': n.district_name}
                       if isinstance(n, PopulationZone) else {}),
                    **({'capacity': n.capacity, 'shelter_type': n.shelter_type}
                       if isinstance(n, Shelter) else {})
                }
                for n in self._nodes.values()
            ],
            'edges': [
                {
                    'id': e.id,
                    'source': e.source_id,
                    'target': e.target_id,
                    'length_km': e.length_km,
                    'road_type': e.road_type.value,
                    'lanes': e.lanes,
                    'max_speed': e.max_speed_kmh,
                    'name': e.name,
                    'oneway': e.is_oneway
                }
                for e in self._edges.values()
            ],
            'hazards': [
                {
                    'center_lat': h.center_lat,
                    'center_lon': h.center_lon,
                    'radius_km': h.radius_km,
                    'risk_level': h.risk_level,
                    'hazard_type': h.hazard_type
                }
                for h in self._hazard_zones
            ]
        }

    def save_to_file(self, filepath: str) -> None:
        """Lưu mạng lưới vào tệp JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'EvacuationNetwork':
        """Tải mạng lưới từ tệp JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        network = cls()

        # Tải các nút
        for node_data in data.get('nodes', []):
            node_type = NodeType(node_data['type'])
            if node_type == NodeType.POPULATION_ZONE:
                node = PopulationZone(
                    id=node_data['id'],
                    lat=node_data['lat'],
                    lon=node_data['lon'],
                    name=node_data.get('name'),
                    population=node_data.get('population', 0),
                    district_name=node_data.get('district', '')
                )
            elif node_type == NodeType.SHELTER:
                node = Shelter(
                    id=node_data['id'],
                    lat=node_data['lat'],
                    lon=node_data['lon'],
                    name=node_data.get('name'),
                    capacity=node_data.get('capacity', 1000),
                    shelter_type=node_data.get('shelter_type', 'general')
                )
            else:
                node = Node(
                    id=node_data['id'],
                    lat=node_data['lat'],
                    lon=node_data['lon'],
                    node_type=node_type,
                    name=node_data.get('name')
                )
            network.add_node(node)

        # Tải các cạnh
        for edge_data in data.get('edges', []):
            edge = Edge(
                id=edge_data['id'],
                source_id=edge_data['source'],
                target_id=edge_data['target'],
                length_km=edge_data['length_km'],
                road_type=RoadType(edge_data.get('road_type', 'unclassified')),
                lanes=edge_data.get('lanes', 1),
                max_speed_kmh=edge_data.get('max_speed', 30),
                name=edge_data.get('name'),
                is_oneway=edge_data.get('oneway', False)
            )
            network.add_edge(edge)

        # Tải các vùng nguy hiểm
        for hazard_data in data.get('hazards', []):
            hazard = HazardZone(
                center_lat=hazard_data['center_lat'],
                center_lon=hazard_data['center_lon'],
                radius_km=hazard_data['radius_km'],
                risk_level=hazard_data.get('risk_level', 0.8),
                hazard_type=hazard_data.get('hazard_type', 'flood')
            )
            network.add_hazard_zone(hazard)

        return network

    def __len__(self) -> int:
        """Trả về số lượng nút."""
        return len(self._nodes)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"EvacuationNetwork(nodes={stats.total_nodes}, "
                f"edges={stats.total_edges}, "
                f"zones={stats.population_zones}, "
                f"shelters={stats.shelters})")
