"""
Unit tests for the models module.
Tests Node, Edge, and Network classes.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.node import (
    Node, NodeType, PopulationZone, Shelter, HazardZone,
    haversine_distance, lat_lon_to_mercator, mercator_to_lat_lon
)
from src.models.edge import Edge, RoadType, ROAD_CAPACITY, DEFAULT_SPEEDS
from src.models.network import EvacuationNetwork, NetworkStats


class TestHaversineDistance:
    """Tests for haversine distance calculation."""

    def test_same_point_returns_zero(self):
        """Distance from a point to itself should be zero."""
        lat, lon = 10.7769, 106.7009  # HCM center
        assert haversine_distance(lat, lon, lat, lon) == 0.0

    def test_known_distance(self):
        """Test distance between two known points in HCM."""
        # District 1 to District 7 (approximately 4-5 km)
        lat1, lon1 = 10.7769, 106.7009  # District 1
        lat2, lon2 = 10.7365, 106.7218  # District 7
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        assert 4.0 < distance < 6.0  # Should be around 4-5 km

    def test_symmetry(self):
        """Distance A->B should equal B->A."""
        lat1, lon1 = 10.7769, 106.7009
        lat2, lon2 = 10.8514, 106.7539
        assert haversine_distance(lat1, lon1, lat2, lon2) == \
               haversine_distance(lat2, lon2, lat1, lon1)


class TestMercatorConversion:
    """Tests for coordinate conversion functions."""

    def test_roundtrip_conversion(self):
        """Converting to Mercator and back should return original coords."""
        lat, lon = 10.7769, 106.7009
        x, y = lat_lon_to_mercator(lat, lon)
        lat2, lon2 = mercator_to_lat_lon(x, y)
        assert abs(lat - lat2) < 0.0001
        assert abs(lon - lon2) < 0.0001


class TestNode:
    """Tests for the base Node class."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = Node(id="n1", lat=10.7769, lon=106.7009)
        assert node.id == "n1"
        assert node.lat == 10.7769
        assert node.lon == 106.7009
        assert node.node_type == NodeType.INTERSECTION

    def test_node_pos_property(self):
        """Test the pos property returns correct tuple."""
        node = Node(id="n1", lat=10.5, lon=106.5)
        assert node.pos == (10.5, 106.5)

    def test_node_distance_to(self):
        """Test distance_to method."""
        node1 = Node(id="n1", lat=10.7769, lon=106.7009)
        node2 = Node(id="n2", lat=10.7365, lon=106.7218)
        distance = node1.distance_to(node2)
        assert distance > 0
        assert distance < 10  # Should be a few km


class TestPopulationZone:
    """Tests for PopulationZone class."""

    def test_population_zone_creation(self):
        """Test population zone creation with population."""
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
        """Test remaining_population calculation."""
        zone = PopulationZone(
            id="zone1", lat=10.0, lon=106.0,
            population=10000, evacuated=3000
        )
        assert zone.remaining_population == 7000

    def test_remaining_population_never_negative(self):
        """Remaining population should never be negative."""
        zone = PopulationZone(
            id="zone1", lat=10.0, lon=106.0,
            population=10000, evacuated=15000  # Over-evacuated
        )
        assert zone.remaining_population == 0

    def test_evacuation_progress(self):
        """Test evacuation_progress calculation."""
        zone = PopulationZone(
            id="zone1", lat=10.0, lon=106.0,
            population=10000, evacuated=2500
        )
        assert zone.evacuation_progress == 0.25

    def test_evacuation_progress_zero_population(self):
        """Evacuation progress should be 1.0 for zero population."""
        zone = PopulationZone(id="zone1", lat=10.0, lon=106.0, population=0)
        assert zone.evacuation_progress == 1.0


class TestShelter:
    """Tests for Shelter class."""

    def test_shelter_creation(self):
        """Test shelter creation."""
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
        """Test available_capacity calculation."""
        shelter = Shelter(
            id="s1", lat=10.0, lon=106.0,
            capacity=1000, current_occupancy=400
        )
        assert shelter.available_capacity == 600

    def test_occupancy_rate(self):
        """Test occupancy_rate calculation."""
        shelter = Shelter(
            id="s1", lat=10.0, lon=106.0,
            capacity=1000, current_occupancy=250
        )
        assert shelter.occupancy_rate == 0.25

    def test_has_capacity(self):
        """Test has_capacity method."""
        shelter = Shelter(
            id="s1", lat=10.0, lon=106.0,
            capacity=1000, current_occupancy=900
        )
        assert shelter.has_capacity(100) is True
        assert shelter.has_capacity(101) is False

    def test_has_capacity_inactive_shelter(self):
        """Inactive shelter should never have capacity."""
        shelter = Shelter(
            id="s1", lat=10.0, lon=106.0,
            capacity=1000, is_active=False
        )
        assert shelter.has_capacity(1) is False

    def test_admit_evacuees(self):
        """Test admit method."""
        shelter = Shelter(id="s1", lat=10.0, lon=106.0, capacity=100)
        admitted = shelter.admit(50)
        assert admitted == 50
        assert shelter.current_occupancy == 50

    def test_admit_partial(self):
        """Admit should only accept up to available capacity."""
        shelter = Shelter(
            id="s1", lat=10.0, lon=106.0,
            capacity=100, current_occupancy=80
        )
        admitted = shelter.admit(50)
        assert admitted == 20
        assert shelter.current_occupancy == 100


class TestHazardZone:
    """Tests for HazardZone class."""

    def test_hazard_zone_creation(self):
        """Test hazard zone creation."""
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
        """Risk at center should equal risk_level."""
        hazard = HazardZone(
            center_lat=10.7579, center_lon=106.7057,
            radius_km=2.0, risk_level=0.8
        )
        risk = hazard.get_risk_at(10.7579, 106.7057)
        assert risk == 0.8

    def test_risk_outside_radius(self):
        """Risk outside radius should be zero."""
        hazard = HazardZone(
            center_lat=10.7579, center_lon=106.7057,
            radius_km=1.0, risk_level=0.8
        )
        # Point far away
        risk = hazard.get_risk_at(10.9, 106.9)
        assert risk == 0.0

    def test_risk_inactive_hazard(self):
        """Inactive hazard should return zero risk."""
        hazard = HazardZone(
            center_lat=10.7579, center_lon=106.7057,
            radius_km=2.0, risk_level=0.8, is_active=False
        )
        risk = hazard.get_risk_at(10.7579, 106.7057)
        assert risk == 0.0


class TestEdge:
    """Tests for Edge class."""

    def test_edge_creation(self):
        """Test basic edge creation."""
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
        """Test capacity based on road type and lanes."""
        edge = Edge(
            id="e1", source_id="n1", target_id="n2",
            length_km=1.0, road_type=RoadType.PRIMARY, lanes=2
        )
        expected = ROAD_CAPACITY[RoadType.PRIMARY] * 2
        assert edge.capacity == expected

    def test_base_travel_time(self):
        """Test base travel time calculation."""
        edge = Edge(
            id="e1", source_id="n1", target_id="n2",
            length_km=30, max_speed_kmh=60
        )
        assert edge.base_travel_time == 0.5  # 30km / 60km/h = 0.5h

    def test_congestion_level(self):
        """Test congestion level calculation."""
        edge = Edge(
            id="e1", source_id="n1", target_id="n2",
            length_km=1.0, road_type=RoadType.PRIMARY, lanes=1
        )
        edge.current_flow = ROAD_CAPACITY[RoadType.PRIMARY] // 2
        assert 0.4 < edge.congestion_level < 0.6

    def test_blocked_edge_speed(self):
        """Blocked edge should have zero effective speed."""
        edge = Edge(
            id="e1", source_id="n1", target_id="n2",
            length_km=1.0, is_blocked=True
        )
        assert edge.effective_speed == 0.0

    def test_blocked_edge_travel_time(self):
        """Blocked edge should have infinite travel time."""
        edge = Edge(
            id="e1", source_id="n1", target_id="n2",
            length_km=1.0, is_blocked=True
        )
        assert edge.current_travel_time == float('inf')

    def test_get_cost(self):
        """Test cost calculation."""
        edge = Edge(
            id="e1", source_id="n1", target_id="n2",
            length_km=1.0, max_speed_kmh=30
        )
        cost = edge.get_cost(risk_weight=0.3)
        assert cost > 0

    def test_blocked_edge_cost(self):
        """Blocked edge should have infinite cost."""
        edge = Edge(
            id="e1", source_id="n1", target_id="n2",
            length_km=1.0, is_blocked=True
        )
        assert edge.get_cost() == float('inf')

    def test_add_remove_flow(self):
        """Test flow management."""
        edge = Edge(id="e1", source_id="n1", target_id="n2", length_km=1.0)
        edge.add_flow(100)
        assert edge.current_flow == 100
        edge.remove_flow(30)
        assert edge.current_flow == 70
        edge.reset_flow()
        assert edge.current_flow == 0

    def test_remove_flow_never_negative(self):
        """Flow should never go negative."""
        edge = Edge(id="e1", source_id="n1", target_id="n2", length_km=1.0)
        edge.current_flow = 50
        edge.remove_flow(100)
        assert edge.current_flow == 0

    def test_set_flood_risk_blocks_high_risk(self):
        """High flood risk should block the road."""
        edge = Edge(id="e1", source_id="n1", target_id="n2", length_km=1.0)
        edge.set_flood_risk(0.95)
        assert edge.is_blocked is True


class TestEvacuationNetwork:
    """Tests for EvacuationNetwork class."""

    def test_network_creation(self):
        """Test empty network creation."""
        network = EvacuationNetwork()
        assert len(network) == 0

    def test_add_and_get_node(self):
        """Test adding and retrieving nodes."""
        network = EvacuationNetwork()
        node = Node(id="n1", lat=10.7769, lon=106.7009)
        network.add_node(node)

        retrieved = network.get_node("n1")
        assert retrieved is not None
        assert retrieved.id == "n1"

    def test_add_population_zone(self):
        """Test adding population zones."""
        network = EvacuationNetwork()
        zone = PopulationZone(
            id="zone1", lat=10.0, lon=106.0, population=10000
        )
        network.add_node(zone)

        zones = network.get_population_zones()
        assert len(zones) == 1
        assert zones[0].population == 10000

    def test_add_shelter(self):
        """Test adding shelters."""
        network = EvacuationNetwork()
        shelter = Shelter(id="s1", lat=10.0, lon=106.0, capacity=1000)
        network.add_node(shelter)

        shelters = network.get_shelters()
        assert len(shelters) == 1
        assert shelters[0].capacity == 1000

    def test_add_and_get_edge(self):
        """Test adding and retrieving edges."""
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
        """Test getting edge between two nodes."""
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
        """Test getting node neighbors."""
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
        """Test adding hazard zones."""
        network = EvacuationNetwork()
        hazard = HazardZone(
            center_lat=10.7579, center_lon=106.7057,
            radius_km=2.0, risk_level=0.8
        )
        network.add_hazard_zone(hazard)

        hazards = network.get_hazard_zones()
        assert len(hazards) == 1

    def test_get_total_risk_at(self):
        """Test total risk calculation at a point."""
        network = EvacuationNetwork()
        hazard = HazardZone(
            center_lat=10.7579, center_lon=106.7057,
            radius_km=2.0, risk_level=0.8
        )
        network.add_hazard_zone(hazard)

        # Risk at hazard center
        risk = network.get_total_risk_at(10.7579, 106.7057)
        assert risk == 0.8

        # Risk far away
        risk_far = network.get_total_risk_at(10.9, 106.9)
        assert risk_far == 0.0

    def test_find_nearest_node(self):
        """Test finding nearest node."""
        network = EvacuationNetwork()
        n1 = Node(id="n1", lat=10.0, lon=106.0)
        n2 = Node(id="n2", lat=10.5, lon=106.5)
        network.add_node(n1)
        network.add_node(n2)

        nearest = network.find_nearest_node(10.01, 106.01)
        assert nearest.id == "n1"

    def test_find_nearest_shelter(self):
        """Test finding nearest shelter with capacity."""
        network = EvacuationNetwork()
        s1 = Shelter(id="s1", lat=10.0, lon=106.0, capacity=100, current_occupancy=100)
        s2 = Shelter(id="s2", lat=10.1, lon=106.1, capacity=100, current_occupancy=50)
        network.add_node(s1)
        network.add_node(s2)

        # s1 is closer but full, should return s2
        nearest = network.find_nearest_shelter(10.0, 106.0)
        assert nearest.id == "s2"

    def test_reset_simulation_state(self):
        """Test resetting simulation state."""
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
        """Test network statistics."""
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
        """Test geographic bounds calculation."""
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
        """Test geographic center calculation."""
        network = EvacuationNetwork()
        n1 = Node(id="n1", lat=10.0, lon=106.0)
        n2 = Node(id="n2", lat=12.0, lon=108.0)
        network.add_node(n1)
        network.add_node(n2)

        center_lat, center_lon = network.get_center()
        assert center_lat == 11.0
        assert center_lon == 107.0


class TestNetworkSerialization:
    """Tests for network serialization."""

    def test_to_dict(self):
        """Test converting network to dictionary."""
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
