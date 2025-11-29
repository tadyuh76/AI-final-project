"""
Kiểm thử đơn vị cho module models.
Kiểm thử các lớp Node, Edge và Network.
"""

import pytest
import sys
import os

# Thêm thư mục cha vào đường dẫn để import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.node import (
    Node, NodeType, PopulationZone, Shelter, HazardZone,
    haversine_distance, lat_lon_to_mercator, mercator_to_lat_lon
)
from src.models.edge import Edge, RoadType, ROAD_CAPACITY, DEFAULT_SPEEDS
from src.models.network import EvacuationNetwork, NetworkStats


class TestHaversineDistance:
    """Kiểm thử cho hàm tính khoảng cách haversine."""

    def test_same_point_returns_zero(self):
        """Khoảng cách từ một điểm đến chính nó phải bằng không."""
        lat, lon = 10.7769, 106.7009  # Trung tâm HCM
        assert haversine_distance(lat, lon, lat, lon) == 0.0

    def test_known_distance(self):
        """Kiểm tra khoảng cách giữa hai điểm đã biết ở HCM."""
        # Quận 1 đến Quận 7 (khoảng 4-5 km)
        lat1, lon1 = 10.7769, 106.7009  # Quận 1
        lat2, lon2 = 10.7365, 106.7218  # Quận 7
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        assert 4.0 < distance < 6.0  # Nên khoảng 4-5 km

    def test_symmetry(self):
        """Khoảng cách A->B phải bằng B->A."""
        lat1, lon1 = 10.7769, 106.7009
        lat2, lon2 = 10.8514, 106.7539
        assert haversine_distance(lat1, lon1, lat2, lon2) == \
               haversine_distance(lat2, lon2, lat1, lon1)


class TestMercatorConversion:
    """Kiểm thử cho các hàm chuyển đổi tọa độ."""

    def test_roundtrip_conversion(self):
        """Chuyển sang Mercator và ngược lại phải trả về tọa độ ban đầu."""
        lat, lon = 10.7769, 106.7009
        x, y = lat_lon_to_mercator(lat, lon)
        lat2, lon2 = mercator_to_lat_lon(x, y)
        assert abs(lat - lat2) < 0.0001
        assert abs(lon - lon2) < 0.0001


class TestNode:
    """Kiểm thử cho lớp Node cơ bản."""

    def test_node_creation(self):
        """Kiểm tra việc tạo node cơ bản."""
        node = Node(id="n1", lat=10.7769, lon=106.7009)
        assert node.id == "n1"
        assert node.lat == 10.7769
        assert node.lon == 106.7009
        assert node.node_type == NodeType.INTERSECTION

    def test_node_pos_property(self):
        """Kiểm tra thuộc tính pos trả về tuple đúng."""
        node = Node(id="n1", lat=10.5, lon=106.5)
        assert node.pos == (10.5, 106.5)

    def test_node_distance_to(self):
        """Kiểm tra phương thức distance_to."""
        node1 = Node(id="n1", lat=10.7769, lon=106.7009)
        node2 = Node(id="n2", lat=10.7365, lon=106.7218)
        distance = node1.distance_to(node2)
        assert distance > 0
        assert distance < 10  # Nên vài km


class TestPopulationZone:
    """Kiểm thử cho lớp PopulationZone."""

    def test_population_zone_creation(self):
        """Kiểm tra việc tạo khu dân cư với dân số."""
        zone = PopulationZone(
            id="zone1",
            lat=10.7769,
            lon=106.7009,
            population=100000,
            district_name="District 1"
        )
        assert zone.population == 100000
        assert zone.evacuated == 0
        assert zone.node_type == NodeType.POPULATION_ZONE

    def test_remaining_population(self):
        """Kiểm tra tính toán remaining_population."""
        zone = PopulationZone(
            id="zone1", lat=10.0, lon=106.0,
            population=10000, evacuated=3000
        )
        assert zone.remaining_population == 7000

    def test_remaining_population_never_negative(self):
        """Dân số còn lại không bao giờ âm."""
        zone = PopulationZone(
            id="zone1", lat=10.0, lon=106.0,
            population=10000, evacuated=15000  # Sơ tán quá mức
        )
        assert zone.remaining_population == 0

    def test_evacuation_progress(self):
        """Kiểm tra tính toán evacuation_progress."""
        zone = PopulationZone(
            id="zone1", lat=10.0, lon=106.0,
            population=10000, evacuated=2500
        )
        assert zone.evacuation_progress == 0.25

    def test_evacuation_progress_zero_population(self):
        """Tiến độ sơ tán phải là 1.0 cho dân số bằng không."""
        zone = PopulationZone(id="zone1", lat=10.0, lon=106.0, population=0)
        assert zone.evacuation_progress == 1.0


class TestShelter:
    """Kiểm thử cho lớp Shelter."""

    def test_shelter_creation(self):
        """Kiểm tra việc tạo nơi trú ẩn."""
        shelter = Shelter(
            id="shelter1",
            lat=10.7888,
            lon=106.6875,
            capacity=15000,
            shelter_type="stadium",
            name="Thong Nhat Stadium"
        )
        assert shelter.capacity == 15000
        assert shelter.current_occupancy == 0
        assert shelter.node_type == NodeType.SHELTER

    def test_available_capacity(self):
        """Kiểm tra tính toán available_capacity."""
        shelter = Shelter(
            id="s1", lat=10.0, lon=106.0,
            capacity=1000, current_occupancy=400
        )
        assert shelter.available_capacity == 600

    def test_occupancy_rate(self):
        """Kiểm tra tính toán occupancy_rate."""
        shelter = Shelter(
            id="s1", lat=10.0, lon=106.0,
            capacity=1000, current_occupancy=250
        )
        assert shelter.occupancy_rate == 0.25

    def test_has_capacity(self):
        """Kiểm tra phương thức has_capacity."""
        shelter = Shelter(
            id="s1", lat=10.0, lon=106.0,
            capacity=1000, current_occupancy=900
        )
        assert shelter.has_capacity(100) is True
        assert shelter.has_capacity(101) is False

    def test_has_capacity_inactive_shelter(self):
        """Nơi trú ẩn không hoạt động không bao giờ có sức chứa."""
        shelter = Shelter(
            id="s1", lat=10.0, lon=106.0,
            capacity=1000, is_active=False
        )
        assert shelter.has_capacity(1) is False

    def test_admit_evacuees(self):
        """Kiểm tra phương thức admit."""
        shelter = Shelter(id="s1", lat=10.0, lon=106.0, capacity=100)
        admitted = shelter.admit(50)
        assert admitted == 50
        assert shelter.current_occupancy == 50

    def test_admit_partial(self):
        """Admit chỉ nên chấp nhận đến sức chứa có sẵn."""
        shelter = Shelter(
            id="s1", lat=10.0, lon=106.0,
            capacity=100, current_occupancy=80
        )
        admitted = shelter.admit(50)
        assert admitted == 20
        assert shelter.current_occupancy == 100


class TestHazardZone:
    """Kiểm thử cho lớp HazardZone."""

    def test_hazard_zone_creation(self):
        """Kiểm tra việc tạo vùng nguy hiểm."""
        hazard = HazardZone(
            center_lat=10.7579,
            center_lon=106.7057,
            radius_km=2.0,
            risk_level=0.8,
            hazard_type="flood"
        )
        assert hazard.radius_km == 2.0
        assert hazard.risk_level == 0.8

    def test_risk_at_center(self):
        """Rủi ro tại tâm phải bằng risk_level."""
        hazard = HazardZone(
            center_lat=10.7579, center_lon=106.7057,
            radius_km=2.0, risk_level=0.8
        )
        risk = hazard.get_risk_at(10.7579, 106.7057)
        assert risk == 0.8

    def test_risk_outside_radius(self):
        """Rủi ro ngoài bán kính phải bằng không."""
        hazard = HazardZone(
            center_lat=10.7579, center_lon=106.7057,
            radius_km=1.0, risk_level=0.8
        )
        # Điểm xa
        risk = hazard.get_risk_at(10.9, 106.9)
        assert risk == 0.0

    def test_risk_inactive_hazard(self):
        """Vùng nguy hiểm không hoạt động phải trả về rủi ro bằng không."""
        hazard = HazardZone(
            center_lat=10.7579, center_lon=106.7057,
            radius_km=2.0, risk_level=0.8, is_active=False
        )
        risk = hazard.get_risk_at(10.7579, 106.7057)
        assert risk == 0.0


class TestEdge:
    """Kiểm thử cho lớp Edge."""

    def test_edge_creation(self):
        """Kiểm tra việc tạo edge cơ bản."""
        edge = Edge(
            id="e1",
            source_id="n1",
            target_id="n2",
            length_km=1.5,
            road_type=RoadType.PRIMARY,
            lanes=2,
            max_speed_kmh=50
        )
        assert edge.length_km == 1.5
        assert edge.lanes == 2
        assert edge.road_type == RoadType.PRIMARY

    def test_capacity_calculation(self):
        """Kiểm tra tính toán sức chứa dựa trên loại đường và số làn."""
        edge = Edge(
            id="e1", source_id="n1", target_id="n2",
            length_km=1.0, road_type=RoadType.PRIMARY, lanes=2
        )
        expected = ROAD_CAPACITY[RoadType.PRIMARY] * 2
        assert edge.capacity == expected

    def test_base_travel_time(self):
        """Kiểm tra tính toán thời gian di chuyển cơ bản."""
        edge = Edge(
            id="e1", source_id="n1", target_id="n2",
            length_km=30, max_speed_kmh=60
        )
        assert edge.base_travel_time == 0.5  # 30km / 60km/h = 0.5h

    def test_congestion_level(self):
        """Kiểm tra tính toán mức độ tắc nghẽn."""
        edge = Edge(
            id="e1", source_id="n1", target_id="n2",
            length_km=1.0, road_type=RoadType.PRIMARY, lanes=1
        )
        edge.current_flow = ROAD_CAPACITY[RoadType.PRIMARY] // 2
        assert 0.4 < edge.congestion_level < 0.6

    def test_blocked_edge_speed(self):
        """Edge bị chặn phải có tốc độ hiệu dụng bằng không."""
        edge = Edge(
            id="e1", source_id="n1", target_id="n2",
            length_km=1.0, is_blocked=True
        )
        assert edge.effective_speed == 0.0

    def test_blocked_edge_travel_time(self):
        """Edge bị chặn phải có thời gian di chuyển vô hạn."""
        edge = Edge(
            id="e1", source_id="n1", target_id="n2",
            length_km=1.0, is_blocked=True
        )
        assert edge.current_travel_time == float('inf')

    def test_get_cost(self):
        """Kiểm tra tính toán chi phí."""
        edge = Edge(
            id="e1", source_id="n1", target_id="n2",
            length_km=1.0, max_speed_kmh=30
        )
        cost = edge.get_cost(risk_weight=0.3)
        assert cost > 0

    def test_blocked_edge_cost(self):
        """Edge bị chặn phải có chi phí vô hạn."""
        edge = Edge(
            id="e1", source_id="n1", target_id="n2",
            length_km=1.0, is_blocked=True
        )
        assert edge.get_cost() == float('inf')

    def test_add_remove_flow(self):
        """Kiểm tra quản lý lưu lượng."""
        edge = Edge(id="e1", source_id="n1", target_id="n2", length_km=1.0)
        edge.add_flow(100)
        assert edge.current_flow == 100
        edge.remove_flow(30)
        assert edge.current_flow == 70
        edge.reset_flow()
        assert edge.current_flow == 0

    def test_remove_flow_never_negative(self):
        """Lưu lượng không bao giờ âm."""
        edge = Edge(id="e1", source_id="n1", target_id="n2", length_km=1.0)
        edge.current_flow = 50
        edge.remove_flow(100)
        assert edge.current_flow == 0

    def test_set_flood_risk_blocks_high_risk(self):
        """Rủi ro lũ lụt cao phải chặn đường."""
        edge = Edge(id="e1", source_id="n1", target_id="n2", length_km=1.0)
        edge.set_flood_risk(0.95)
        assert edge.is_blocked is True


class TestEvacuationNetwork:
    """Kiểm thử cho lớp EvacuationNetwork."""

    def test_network_creation(self):
        """Kiểm tra việc tạo mạng lưới trống."""
        network = EvacuationNetwork()
        assert len(network) == 0

    def test_add_and_get_node(self):
        """Kiểm tra thêm và lấy node."""
        network = EvacuationNetwork()
        node = Node(id="n1", lat=10.7769, lon=106.7009)
        network.add_node(node)

        retrieved = network.get_node("n1")
        assert retrieved is not None
        assert retrieved.id == "n1"

    def test_add_population_zone(self):
        """Kiểm tra thêm khu dân cư."""
        network = EvacuationNetwork()
        zone = PopulationZone(
            id="zone1", lat=10.0, lon=106.0, population=10000
        )
        network.add_node(zone)

        zones = network.get_population_zones()
        assert len(zones) == 1
        assert zones[0].population == 10000

    def test_add_shelter(self):
        """Kiểm tra thêm nơi trú ẩn."""
        network = EvacuationNetwork()
        shelter = Shelter(id="s1", lat=10.0, lon=106.0, capacity=1000)
        network.add_node(shelter)

        shelters = network.get_shelters()
        assert len(shelters) == 1
        assert shelters[0].capacity == 1000

    def test_add_and_get_edge(self):
        """Kiểm tra thêm và lấy edge."""
        network = EvacuationNetwork()
        n1 = Node(id="n1", lat=10.0, lon=106.0)
        n2 = Node(id="n2", lat=10.1, lon=106.1)
        network.add_node(n1)
        network.add_node(n2)

        edge = Edge(id="e1", source_id="n1", target_id="n2", length_km=1.0)
        network.add_edge(edge)

        retrieved = network.get_edge("e1")
        assert retrieved is not None
        assert retrieved.source_id == "n1"

    def test_get_edge_between(self):
        """Kiểm tra lấy edge giữa hai node."""
        network = EvacuationNetwork()
        n1 = Node(id="n1", lat=10.0, lon=106.0)
        n2 = Node(id="n2", lat=10.1, lon=106.1)
        network.add_node(n1)
        network.add_node(n2)

        edge = Edge(id="e1", source_id="n1", target_id="n2", length_km=1.0)
        network.add_edge(edge)

        found = network.get_edge_between("n1", "n2")
        assert found is not None
        assert found.id == "e1"

    def test_get_neighbors(self):
        """Kiểm tra lấy các node láng giềng."""
        network = EvacuationNetwork()
        n1 = Node(id="n1", lat=10.0, lon=106.0)
        n2 = Node(id="n2", lat=10.1, lon=106.1)
        n3 = Node(id="n3", lat=10.2, lon=106.2)
        network.add_node(n1)
        network.add_node(n2)
        network.add_node(n3)

        e1 = Edge(id="e1", source_id="n1", target_id="n2", length_km=1.0)
        e2 = Edge(id="e2", source_id="n1", target_id="n3", length_km=1.0)
        network.add_edge(e1)
        network.add_edge(e2)

        neighbors = network.get_neighbors("n1")
        assert len(neighbors) == 2
        assert "n2" in neighbors
        assert "n3" in neighbors

    def test_add_hazard_zone(self):
        """Kiểm tra thêm vùng nguy hiểm."""
        network = EvacuationNetwork()
        hazard = HazardZone(
            center_lat=10.7579, center_lon=106.7057,
            radius_km=2.0, risk_level=0.8
        )
        network.add_hazard_zone(hazard)

        hazards = network.get_hazard_zones()
        assert len(hazards) == 1

    def test_get_total_risk_at(self):
        """Kiểm tra tính toán tổng rủi ro tại một điểm."""
        network = EvacuationNetwork()
        hazard = HazardZone(
            center_lat=10.7579, center_lon=106.7057,
            radius_km=2.0, risk_level=0.8
        )
        network.add_hazard_zone(hazard)

        # Rủi ro tại tâm vùng nguy hiểm
        risk = network.get_total_risk_at(10.7579, 106.7057)
        assert risk == 0.8

        # Rủi ro xa
        risk_far = network.get_total_risk_at(10.9, 106.9)
        assert risk_far == 0.0

    def test_find_nearest_node(self):
        """Kiểm tra tìm node gần nhất."""
        network = EvacuationNetwork()
        n1 = Node(id="n1", lat=10.0, lon=106.0)
        n2 = Node(id="n2", lat=10.5, lon=106.5)
        network.add_node(n1)
        network.add_node(n2)

        nearest = network.find_nearest_node(10.01, 106.01)
        assert nearest.id == "n1"

    def test_find_nearest_shelter(self):
        """Kiểm tra tìm nơi trú ẩn gần nhất có sức chứa."""
        network = EvacuationNetwork()
        s1 = Shelter(id="s1", lat=10.0, lon=106.0, capacity=100, current_occupancy=100)
        s2 = Shelter(id="s2", lat=10.1, lon=106.1, capacity=100, current_occupancy=50)
        network.add_node(s1)
        network.add_node(s2)

        # s1 gần hơn nhưng đầy, nên trả về s2
        nearest = network.find_nearest_shelter(10.0, 106.0)
        assert nearest.id == "s2"

    def test_reset_simulation_state(self):
        """Kiểm tra đặt lại trạng thái mô phỏng."""
        network = EvacuationNetwork()
        zone = PopulationZone(id="z1", lat=10.0, lon=106.0, population=1000, evacuated=500)
        shelter = Shelter(id="s1", lat=10.1, lon=106.1, capacity=1000, current_occupancy=300)
        network.add_node(zone)
        network.add_node(shelter)

        edge = Edge(id="e1", source_id="z1", target_id="s1", length_km=1.0)
        edge.current_flow = 100
        network.add_edge(edge)

        network.reset_simulation_state()

        assert zone.evacuated == 0
        assert shelter.current_occupancy == 0
        assert edge.current_flow == 0

    def test_get_stats(self):
        """Kiểm tra thống kê mạng lưới."""
        network = EvacuationNetwork()
        zone = PopulationZone(id="z1", lat=10.0, lon=106.0, population=10000)
        shelter = Shelter(id="s1", lat=10.1, lon=106.1, capacity=5000)
        network.add_node(zone)
        network.add_node(shelter)

        edge = Edge(id="e1", source_id="z1", target_id="s1", length_km=2.5)
        network.add_edge(edge)

        stats = network.get_stats()
        assert stats.total_nodes == 2
        assert stats.total_edges == 1
        assert stats.population_zones == 1
        assert stats.shelters == 1
        assert stats.total_population == 10000
        assert stats.total_shelter_capacity == 5000
        assert stats.total_road_length_km == 2.5

    def test_get_bounds(self):
        """Kiểm tra tính toán ranh giới địa lý."""
        network = EvacuationNetwork()
        n1 = Node(id="n1", lat=10.0, lon=106.0)
        n2 = Node(id="n2", lat=11.0, lon=107.0)
        network.add_node(n1)
        network.add_node(n2)

        min_lat, max_lat, min_lon, max_lon = network.get_bounds()
        assert min_lat == 10.0
        assert max_lat == 11.0
        assert min_lon == 106.0
        assert max_lon == 107.0

    def test_get_center(self):
        """Kiểm tra tính toán tâm địa lý."""
        network = EvacuationNetwork()
        n1 = Node(id="n1", lat=10.0, lon=106.0)
        n2 = Node(id="n2", lat=12.0, lon=108.0)
        network.add_node(n1)
        network.add_node(n2)

        center_lat, center_lon = network.get_center()
        assert center_lat == 11.0
        assert center_lon == 107.0


class TestNetworkSerialization:
    """Kiểm thử cho việc tuần tự hóa mạng lưới."""

    def test_to_dict(self):
        """Kiểm tra chuyển đổi mạng lưới sang từ điển."""
        network = EvacuationNetwork()
        zone = PopulationZone(id="z1", lat=10.0, lon=106.0, population=1000)
        shelter = Shelter(id="s1", lat=10.1, lon=106.1, capacity=500)
        network.add_node(zone)
        network.add_node(shelter)

        edge = Edge(id="e1", source_id="z1", target_id="s1", length_km=1.0)
        network.add_edge(edge)

        data = network.to_dict()
        assert 'nodes' in data
        assert 'edges' in data
        assert len(data['nodes']) == 2
        assert len(data['edges']) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
