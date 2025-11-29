"""
Bộ tải dữ liệu OpenStreetMap cho Thành phố Hồ Chí Minh.
Tải xuống và xử lý dữ liệu mạng lưới đường thực tế sử dụng OSMnx.
Dự phòng dữ liệu được tạo nếu OSMnx không khả dụng.
"""

import os
import json
import hashlib
from typing import Optional, Tuple, List, Dict
from pathlib import Path

from ..models.network import EvacuationNetwork
from ..models.node import Node, NodeType, PopulationZone, Shelter, HazardZone
from ..models.edge import Edge, RoadType
from .hcm_data import (
    HCM_DISTRICTS, HCM_SHELTERS, FLOOD_PRONE_AREAS, HCM_BOUNDS,
    DistrictData, ShelterTemplate
)

# Thử import OSMnx
try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False
    ox = None

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None


class OSMDataLoader:
    """Tải dữ liệu mạng lưới đường từ OpenStreetMap."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Khởi tạo bộ tải.

        Args:
            cache_dir: Thư mục để lưu trữ dữ liệu đã tải xuống. Mặc định là assets/data/
        """
        if cache_dir is None:
            # Mặc định vào thư mục assets/data của dự án
            self.cache_dir = Path(__file__).parent.parent.parent / 'assets' / 'data'
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_hcm_network(self,
                         use_cache: bool = True,
                         simplify: bool = True,
                         network_type: str = 'drive') -> EvacuationNetwork:
        """
        Tải mạng lưới đường Thành phố Hồ Chí Minh.

        Args:
            use_cache: Có sử dụng dữ liệu đã lưu nếu có không
            simplify: Có đơn giản hóa đồ thị không (gộp các nút trung gian)
            network_type: Loại mạng OSMnx ('drive', 'walk', 'all')

        Returns:
            EvacuationNetwork với dữ liệu đường TP.HCM
        """
        cache_file = self.cache_dir / 'hcm_network.json'

        # Thử tải từ bộ nhớ đệm
        if use_cache and cache_file.exists():
            try:
                print("Đang tải mạng lưới TP.HCM đã lưu...")
                return EvacuationNetwork.load_from_file(str(cache_file))
            except Exception as e:
                print(f"Không thể tải bộ nhớ đệm: {e}")

        # Thử tải xuống từ OSM
        if HAS_OSMNX:
            try:
                print("Đang tải xuống mạng lưới đường TP.HCM từ OpenStreetMap...")
                network = self._download_osm_network(simplify, network_type)

                # Lưu vào bộ nhớ đệm
                if use_cache:
                    print("Đang lưu vào bộ nhớ đệm...")
                    network.save_to_file(str(cache_file))

                return network
            except Exception as e:
                print(f"Không thể tải xuống dữ liệu OSM: {e}")
                print("Đang chuyển sang mạng lưới được tạo...")

        # Dự phòng mạng lưới được tạo
        print("Đang tạo mạng lưới TP.HCM tổng hợp...")
        network = self._generate_synthetic_network()

        # Lưu vào bộ nhớ đệm
        if use_cache:
            network.save_to_file(str(cache_file))

        return network

    def _download_osm_network(self, simplify: bool, network_type: str) -> EvacuationNetwork:
        """Tải xuống mạng lưới đường từ OpenStreetMap sử dụng OSMnx."""
        if not HAS_OSMNX:
            raise RuntimeError("OSMnx chưa được cài đặt")

        # Cấu hình OSMnx
        ox.settings.use_cache = True
        ox.settings.log_console = True

        # Tải xuống đồ thị
        print(f"  Đang tải xuống mạng lưới {network_type} cho ranh giới TP.HCM...")
        G = ox.graph_from_bbox(
            bbox=(HCM_BOUNDS['north'], HCM_BOUNDS['south'],
                  HCM_BOUNDS['east'], HCM_BOUNDS['west']),
            network_type=network_type,
            simplify=simplify
        )

        print(f"  Đã tải xuống: {G.number_of_nodes()} nút, {G.number_of_edges()} cạnh")

        # Chuyển đổi sang EvacuationNetwork
        network = self._convert_osmnx_graph(G)

        # Thêm các khu vực dân cư
        self._add_population_zones(network)

        # Thêm điểm trú ẩn
        self._add_shelters(network)

        return network

    def _convert_osmnx_graph(self, G) -> EvacuationNetwork:
        """Chuyển đổi đồ thị OSMnx sang EvacuationNetwork."""
        network = EvacuationNetwork()

        # Thêm các nút
        for node_id, data in G.nodes(data=True):
            node = Node(
                id=str(node_id),
                lat=data.get('y', 0),
                lon=data.get('x', 0),
                node_type=NodeType.INTERSECTION
            )
            network.add_node(node)

        # Thêm các cạnh
        edge_count = 0
        for u, v, key, data in G.edges(keys=True, data=True):
            edge_id = f"e_{u}_{v}_{key}"
            edge = Edge.from_osm_data(edge_id, str(u), str(v), data)
            network.add_edge(edge)
            edge_count += 1

        print(f"  Đã chuyển đổi: {len(network._nodes)} nút, {edge_count} cạnh")
        return network

    def _generate_synthetic_network(self) -> EvacuationNetwork:
        """Tạo một mạng lưới đường tổng hợp cho TP.HCM."""
        network = EvacuationNetwork()

        # Tạo một mạng lưới dạng lưới bao phủ TP.HCM
        grid_size = 20  # Lưới 20x20
        lat_step = (HCM_BOUNDS['north'] - HCM_BOUNDS['south']) / grid_size
        lon_step = (HCM_BOUNDS['east'] - HCM_BOUNDS['west']) / grid_size

        # Tạo các nút lưới
        node_grid = {}
        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                lat = HCM_BOUNDS['south'] + i * lat_step
                lon = HCM_BOUNDS['west'] + j * lon_step

                node_id = f"n_{i}_{j}"
                node = Node(
                    id=node_id,
                    lat=lat,
                    lon=lon,
                    node_type=NodeType.INTERSECTION
                )
                network.add_node(node)
                node_grid[(i, j)] = node_id

        # Tạo các cạnh lưới (đường)
        edge_count = 0
        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                current = node_grid[(i, j)]

                # Kết nối với nút bên phải
                if j < grid_size:
                    neighbor = node_grid[(i, j + 1)]
                    edge = self._create_grid_edge(
                        f"e_{edge_count}",
                        current, neighbor,
                        network.get_node(current),
                        network.get_node(neighbor),
                        is_major=(i % 5 == 0)
                    )
                    network.add_edge(edge)
                    edge_count += 1

                # Kết nối với nút phía trên
                if i < grid_size:
                    neighbor = node_grid[(i + 1, j)]
                    edge = self._create_grid_edge(
                        f"e_{edge_count}",
                        current, neighbor,
                        network.get_node(current),
                        network.get_node(neighbor),
                        is_major=(j % 5 == 0)
                    )
                    network.add_edge(edge)
                    edge_count += 1

        # Thêm kết nối chéo cho các đường chính
        for i in range(0, grid_size, 5):
            for j in range(0, grid_size, 5):
                current = node_grid[(i, j)]
                if i + 5 <= grid_size and j + 5 <= grid_size:
                    neighbor = node_grid[(i + 5, j + 5)]
                    edge = self._create_grid_edge(
                        f"e_{edge_count}",
                        current, neighbor,
                        network.get_node(current),
                        network.get_node(neighbor),
                        is_major=True
                    )
                    network.add_edge(edge)
                    edge_count += 1

        print(f"  Đã tạo: {len(network._nodes)} nút, {edge_count} cạnh")

        # Thêm các khu vực dân cư
        self._add_population_zones(network)

        # Thêm điểm trú ẩn
        self._add_shelters(network)

        return network

    def _create_grid_edge(self, edge_id: str, source_id: str, target_id: str,
                          source: Node, target: Node, is_major: bool) -> Edge:
        """Tạo một cạnh lưới với các thuộc tính phù hợp."""
        from ..models.node import haversine_distance

        length = haversine_distance(source.lat, source.lon, target.lat, target.lon)

        if is_major:
            road_type = RoadType.PRIMARY
            lanes = 3
            speed = 50
        else:
            road_type = RoadType.RESIDENTIAL
            lanes = 1
            speed = 30

        return Edge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            length_km=length,
            road_type=road_type,
            lanes=lanes,
            max_speed_kmh=speed,
            is_oneway=False,
            geometry=[(source.lat, source.lon), (target.lat, target.lon)]
        )

    def _add_population_zones(self, network: EvacuationNetwork) -> None:
        """Thêm các khu vực dân cư từ dữ liệu quận TP.HCM."""
        for district_id, district in HCM_DISTRICTS.items():
            # Tìm nút mạng gần nhất với trung tâm quận
            nearest = network.find_nearest_node(district.center_lat, district.center_lon)

            if nearest:
                # Tạo khu vực dân cư tại trung tâm quận
                zone = PopulationZone(
                    id=f"zone_{district_id}",
                    lat=district.center_lat,
                    lon=district.center_lon,
                    population=district.population,
                    district_name=district.name,
                    name=district.name
                )
                network.add_node(zone)

                # Kết nối với giao lộ gần nhất
                from ..models.node import haversine_distance
                length = haversine_distance(zone.lat, zone.lon, nearest.lat, nearest.lon)
                edge = Edge(
                    id=f"e_zone_{district_id}",
                    source_id=zone.id,
                    target_id=nearest.id,
                    length_km=length,
                    road_type=RoadType.RESIDENTIAL,
                    lanes=2,
                    max_speed_kmh=30,
                    is_oneway=False
                )
                network.add_edge(edge)

        print(f"  Đã thêm {len(HCM_DISTRICTS)} khu vực dân cư")

    def _add_shelters(self, network: EvacuationNetwork) -> None:
        """Thêm điểm trú ẩn từ dữ liệu điểm trú ẩn TP.HCM."""
        for i, shelter_data in enumerate(HCM_SHELTERS):
            # Tìm nút mạng gần nhất
            nearest = network.find_nearest_node(shelter_data.lat, shelter_data.lon)

            if nearest:
                shelter = Shelter(
                    id=f"shelter_{i}",
                    lat=shelter_data.lat,
                    lon=shelter_data.lon,
                    name=shelter_data.name,
                    capacity=shelter_data.capacity,
                    shelter_type=shelter_data.shelter_type
                )
                network.add_node(shelter)

                # Kết nối với giao lộ gần nhất
                from ..models.node import haversine_distance
                length = haversine_distance(shelter.lat, shelter.lon, nearest.lat, nearest.lon)
                edge = Edge(
                    id=f"e_shelter_{i}",
                    source_id=nearest.id,
                    target_id=shelter.id,
                    length_km=length,
                    road_type=RoadType.SECONDARY,
                    lanes=2,
                    max_speed_kmh=40,
                    is_oneway=False
                )
                network.add_edge(edge)

        print(f"  Đã thêm {len(HCM_SHELTERS)} điểm trú ẩn")

    def add_default_hazards(self, network: EvacuationNetwork,
                           typhoon_intensity: float = 0.7) -> None:
        """
        Thêm các khu vực nguy hiểm ngập lụt mặc định dựa trên dữ liệu lịch sử.

        Args:
            network: Mạng lưới để thêm nguy hiểm vào
            typhoon_intensity: Hệ số nhân cho mức độ rủi ro (0.0 đến 1.0)
        """
        for area in FLOOD_PRONE_AREAS:
            hazard = HazardZone(
                center_lat=area['center_lat'],
                center_lon=area['center_lon'],
                radius_km=area['radius_km'],
                risk_level=min(1.0, area['risk'] * typhoon_intensity),
                hazard_type='flood'
            )
            network.add_hazard_zone(hazard)

        print(f"  Đã thêm {len(FLOOD_PRONE_AREAS)} khu vực nguy hiểm")


def load_network(use_cache: bool = True,
                use_osm: bool = True) -> EvacuationNetwork:
    """
    Hàm tiện ích để tải mạng lưới TP.HCM.

    Args:
        use_cache: Có sử dụng dữ liệu đã lưu không
        use_osm: Có thử tải xuống từ OSM không

    Returns:
        EvacuationNetwork đã tải
    """
    loader = OSMDataLoader()
    network = loader.load_hcm_network(use_cache=use_cache)
    return network


# Kiểm tra bộ tải
if __name__ == '__main__':
    print("Đang kiểm tra Bộ tải Dữ liệu OSM")
    print("=" * 50)
    print(f"OSMnx khả dụng: {HAS_OSMNX}")
    print(f"NetworkX khả dụng: {HAS_NETWORKX}")
    print()

    loader = OSMDataLoader()
    network = loader.load_hcm_network(use_cache=True)
    loader.add_default_hazards(network, typhoon_intensity=0.7)

    print()
    print("Thống kê Mạng lưới:")
    stats = network.get_stats()
    print(f"  Nút: {stats.total_nodes}")
    print(f"  Cạnh: {stats.total_edges}")
    print(f"  Khu vực Dân cư: {stats.population_zones}")
    print(f"  Điểm trú ẩn: {stats.shelters}")
    print(f"  Tổng Dân số: {stats.total_population:,}")
    print(f"  Tổng Sức chứa Điểm trú ẩn: {stats.total_shelter_capacity:,}")
    print(f"  Khu vực Nguy hiểm: {len(network.get_hazard_zones())}")
