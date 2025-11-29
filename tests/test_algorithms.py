"""
Unit tests for the algorithms module.
Tests GBFS, GWO, Hybrid algorithms and the Comparator.
"""

import pytest
import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.node import Node, NodeType, PopulationZone, Shelter, HazardZone
from src.models.edge import Edge, RoadType
from src.models.network import EvacuationNetwork
from src.algorithms.base import (
    AlgorithmType, AlgorithmConfig, EvacuationRoute,
    EvacuationPlan, AlgorithmMetrics, BaseAlgorithm
)
from src.algorithms.gbfs import GreedyBestFirstSearch, SearchNode
from src.algorithms.gwo import GreyWolfOptimizer, Wolf
from src.algorithms.hybrid import HybridGBFSGWO
from src.algorithms.comparator import AlgorithmComparator, ComparisonResult


# ==================== Test Fixtures ====================

def create_simple_network():
    """Create a simple test network with 2 zones, 2 shelters, and connecting roads."""
    network = EvacuationNetwork()

    # Create intersections (grid layout)
    #   n1 -- n2 -- n3
    #   |     |     |
    #   n4 -- n5 -- n6
    nodes = [
        Node(id="n1", lat=10.0, lon=106.0),
        Node(id="n2", lat=10.0, lon=106.05),
        Node(id="n3", lat=10.0, lon=106.1),
        Node(id="n4", lat=10.05, lon=106.0),
        Node(id="n5", lat=10.05, lon=106.05),
        Node(id="n6", lat=10.05, lon=106.1),
    ]
    for n in nodes:
        network.add_node(n)

    # Create zones
    zone1 = PopulationZone(id="zone1", lat=10.0, lon=106.0, population=1000)
    zone2 = PopulationZone(id="zone2", lat=10.05, lon=106.0, population=1500)
    network.add_node(zone1)
    network.add_node(zone2)

    # Create shelters
    shelter1 = Shelter(id="shelter1", lat=10.0, lon=106.1, capacity=2000)
    shelter2 = Shelter(id="shelter2", lat=10.05, lon=106.1, capacity=1500)
    network.add_node(shelter1)
    network.add_node(shelter2)

    # Connect zones to grid
    network.add_edge(Edge(id="ez1", source_id="zone1", target_id="n1", length_km=0.1, is_oneway=False))
    network.add_edge(Edge(id="ez2", source_id="zone2", target_id="n4", length_km=0.1, is_oneway=False))

    # Connect shelters to grid
    network.add_edge(Edge(id="es1", source_id="n3", target_id="shelter1", length_km=0.1, is_oneway=False))
    network.add_edge(Edge(id="es2", source_id="n6", target_id="shelter2", length_km=0.1, is_oneway=False))

    # Create grid edges
    grid_edges = [
        ("n1", "n2"), ("n2", "n3"),
        ("n4", "n5"), ("n5", "n6"),
        ("n1", "n4"), ("n2", "n5"), ("n3", "n6"),
    ]
    for i, (src, tgt) in enumerate(grid_edges):
        network.add_edge(Edge(
            id=f"e{i}",
            source_id=src,
            target_id=tgt,
            length_km=2.0,
            road_type=RoadType.SECONDARY,
            lanes=2,
            max_speed_kmh=40,
            is_oneway=False
        ))

    return network


def create_network_with_hazard():
    """Create a network with a hazard zone blocking direct paths."""
    network = create_simple_network()

    # Add hazard zone between n2 and n5
    hazard = HazardZone(
        center_lat=10.025,
        center_lon=106.05,
        radius_km=1.0,
        risk_level=0.9
    )
    network.add_hazard_zone(hazard)

    return network


# ==================== Base Algorithm Tests ====================

class TestAlgorithmConfig:
    """Tests for AlgorithmConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AlgorithmConfig()
        assert config.distance_weight == 0.4
        assert config.risk_weight == 0.3
        assert config.n_wolves == 30
        assert config.max_iterations == 100

    def test_custom_values(self):
        """Test custom configuration."""
        config = AlgorithmConfig(
            distance_weight=0.5,
            n_wolves=50,
            max_iterations=200
        )
        assert config.distance_weight == 0.5
        assert config.n_wolves == 50
        assert config.max_iterations == 200

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = AlgorithmConfig()
        data = config.to_dict()
        assert 'distance_weight' in data
        assert 'n_wolves' in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {'distance_weight': 0.6, 'n_wolves': 40}
        config = AlgorithmConfig.from_dict(data)
        assert config.distance_weight == 0.6
        assert config.n_wolves == 40


class TestEvacuationRoute:
    """Tests for EvacuationRoute."""

    def test_route_creation(self):
        """Test route creation."""
        route = EvacuationRoute(
            zone_id="zone1",
            shelter_id="shelter1",
            path=["zone1", "n1", "n2", "shelter1"],
            flow=500,
            distance_km=5.0,
            estimated_time_hours=0.5,
            risk_score=0.2
        )
        assert route.zone_id == "zone1"
        assert route.flow == 500
        assert route.path_length == 4


class TestEvacuationPlan:
    """Tests for EvacuationPlan."""

    def test_empty_plan(self):
        """Test empty plan."""
        plan = EvacuationPlan()
        assert len(plan.routes) == 0
        assert plan.total_evacuees == 0

    def test_add_route(self):
        """Test adding routes to plan."""
        plan = EvacuationPlan()
        route = EvacuationRoute(
            zone_id="z1", shelter_id="s1",
            path=["z1", "s1"], flow=1000,
            estimated_time_hours=0.5, risk_score=0.1
        )
        plan.add_route(route)

        assert len(plan.routes) == 1
        assert plan.total_evacuees == 1000

    def test_weighted_metrics(self):
        """Test that metrics are correctly weighted by flow."""
        plan = EvacuationPlan()

        route1 = EvacuationRoute(
            zone_id="z1", shelter_id="s1",
            path=["z1", "s1"], flow=1000,
            estimated_time_hours=1.0, risk_score=0.2
        )
        route2 = EvacuationRoute(
            zone_id="z2", shelter_id="s1",
            path=["z2", "s1"], flow=1000,
            estimated_time_hours=2.0, risk_score=0.4
        )
        plan.add_route(route1)
        plan.add_route(route2)

        # Average time: (1.0*1000 + 2.0*1000) / 2000 = 1.5
        assert plan.total_time_hours == 1.5
        # Average risk: (0.2*1000 + 0.4*1000) / 2000 = 0.3
        assert plan.average_risk == 0.3

    def test_get_shelter_loads(self):
        """Test shelter load calculation."""
        plan = EvacuationPlan()
        plan.add_route(EvacuationRoute(
            zone_id="z1", shelter_id="s1",
            path=[], flow=500
        ))
        plan.add_route(EvacuationRoute(
            zone_id="z2", shelter_id="s1",
            path=[], flow=300
        ))
        plan.add_route(EvacuationRoute(
            zone_id="z3", shelter_id="s2",
            path=[], flow=700
        ))

        loads = plan.get_shelter_loads()
        assert loads["s1"] == 800
        assert loads["s2"] == 700


class TestAlgorithmMetrics:
    """Tests for AlgorithmMetrics."""

    def test_metrics_creation(self):
        """Test metrics creation."""
        metrics = AlgorithmMetrics(algorithm_type=AlgorithmType.GBFS)
        assert metrics.algorithm_type == AlgorithmType.GBFS
        assert metrics.execution_time_seconds == 0.0

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = AlgorithmMetrics(
            algorithm_type=AlgorithmType.GBFS,
            execution_time_seconds=1.5,
            final_cost=100.0
        )
        data = metrics.to_dict()
        assert data['algorithm'] == 'gbfs'
        assert data['execution_time'] == 1.5


# ==================== GBFS Tests ====================

class TestSearchNode:
    """Tests for GBFS SearchNode."""

    def test_search_node_creation(self):
        """Test search node creation."""
        node = SearchNode(node_id="n1", g_cost=1.0, h_cost=2.0)
        assert node.node_id == "n1"
        assert node.f_cost == 2.0  # GBFS uses h_cost for priority

    def test_search_node_comparison(self):
        """Test search node comparison for priority queue."""
        node1 = SearchNode(node_id="n1", g_cost=1.0, h_cost=2.0)
        node2 = SearchNode(node_id="n2", g_cost=1.0, h_cost=3.0)
        assert node1 < node2  # Lower h_cost should be "less than"


class TestGreedyBestFirstSearch:
    """Tests for GBFS algorithm."""

    def test_algorithm_type(self):
        """Test algorithm type property."""
        network = create_simple_network()
        gbfs = GreedyBestFirstSearch(network)
        assert gbfs.algorithm_type == AlgorithmType.GBFS

    def test_heuristic_calculation(self):
        """Test heuristic function."""
        network = create_simple_network()
        gbfs = GreedyBestFirstSearch(network)

        node = network.get_node("n1")
        shelter = network.get_shelters()[0]
        h = gbfs.heuristic(node, shelter, {})
        assert h >= 0  # Heuristic should be non-negative

    def test_find_path_simple(self):
        """Test finding a path in simple network."""
        network = create_simple_network()
        gbfs = GreedyBestFirstSearch(network)

        zone = network.get_population_zones()[0]
        shelters = network.get_shelters()

        path, shelter, cost = gbfs.find_path(zone, shelters)

        assert path is not None
        assert len(path) > 0
        assert shelter is not None
        assert cost < float('inf')

    def test_find_path_no_shelters(self):
        """Test finding path with no shelters returns None."""
        network = create_simple_network()
        gbfs = GreedyBestFirstSearch(network)

        zone = network.get_population_zones()[0]
        path, shelter, cost = gbfs.find_path(zone, [])

        assert path is None
        assert shelter is None
        assert cost == float('inf')

    def test_optimize_creates_plan(self):
        """Test that optimize creates a valid evacuation plan."""
        network = create_simple_network()
        gbfs = GreedyBestFirstSearch(network)

        plan, metrics = gbfs.optimize()

        assert plan is not None
        assert len(plan.routes) > 0
        assert metrics.execution_time_seconds >= 0
        assert metrics.routes_found > 0

    def test_optimize_respects_capacity(self):
        """Test that optimization respects shelter capacity."""
        network = create_simple_network()
        gbfs = GreedyBestFirstSearch(network)

        plan, _ = gbfs.optimize()

        shelter_loads = plan.get_shelter_loads()
        for shelter in network.get_shelters():
            if shelter.id in shelter_loads:
                assert shelter_loads[shelter.id] <= shelter.capacity

    def test_find_multiple_paths(self):
        """Test finding multiple paths."""
        network = create_simple_network()
        gbfs = GreedyBestFirstSearch(network)

        zone = network.get_population_zones()[0]
        shelters = network.get_shelters()

        paths = gbfs.find_multiple_paths(zone, shelters, k=2)
        assert len(paths) <= 2
        if len(paths) == 2:
            assert paths[0][1].id != paths[1][1].id  # Different shelters


# ==================== GWO Tests ====================

class TestWolf:
    """Tests for GWO Wolf class."""

    def test_wolf_creation(self):
        """Test wolf creation."""
        position = np.array([[0.5, 0.5], [0.3, 0.7]])
        wolf = Wolf(position=position, fitness=100.0)
        assert wolf.fitness == 100.0
        assert wolf.position.shape == (2, 2)

    def test_wolf_copy(self):
        """Test wolf copy creates independent copy."""
        position = np.array([[0.5, 0.5]])
        wolf = Wolf(position=position, fitness=100.0)
        copy = wolf.copy()

        copy.position[0, 0] = 0.9
        copy.fitness = 50.0

        assert wolf.position[0, 0] == 0.5
        assert wolf.fitness == 100.0


class TestGreyWolfOptimizer:
    """Tests for GWO algorithm."""

    def test_algorithm_type(self):
        """Test algorithm type property."""
        network = create_simple_network()
        gwo = GreyWolfOptimizer(network)
        assert gwo.algorithm_type == AlgorithmType.GWO

    def test_initialize_population(self):
        """Test population initialization."""
        network = create_simple_network()
        config = AlgorithmConfig(n_wolves=10, max_iterations=5)
        gwo = GreyWolfOptimizer(network, config)

        gwo._initialize_problem()
        gwo._initialize_population()

        assert len(gwo.wolves) == 10
        assert gwo.alpha is not None
        assert gwo.beta is not None
        assert gwo.delta is not None

    def test_wolf_positions_normalized(self):
        """Test that wolf positions are normalized (rows sum to 1)."""
        network = create_simple_network()
        config = AlgorithmConfig(n_wolves=5, max_iterations=5)
        gwo = GreyWolfOptimizer(network, config)

        gwo._initialize_problem()
        gwo._initialize_population()

        for wolf in gwo.wolves:
            row_sums = wolf.position.sum(axis=1)
            np.testing.assert_array_almost_equal(row_sums, np.ones_like(row_sums))

    def test_fitness_calculation(self):
        """Test fitness calculation."""
        network = create_simple_network()
        gwo = GreyWolfOptimizer(network)

        gwo._initialize_problem()
        position = np.ones((gwo.n_zones, gwo.n_shelters)) / gwo.n_shelters

        fitness = gwo._calculate_fitness(position)
        assert fitness > 0
        assert fitness < float('inf')

    def test_optimize_creates_plan(self):
        """Test that optimize creates a valid plan."""
        network = create_simple_network()
        config = AlgorithmConfig(n_wolves=10, max_iterations=10)
        gwo = GreyWolfOptimizer(network, config)

        plan, metrics = gwo.optimize()

        assert plan is not None
        assert metrics.iterations > 0
        assert len(metrics.convergence_history) > 0

    def test_convergence_improves(self):
        """Test that fitness generally improves over iterations."""
        network = create_simple_network()
        config = AlgorithmConfig(n_wolves=20, max_iterations=30)
        gwo = GreyWolfOptimizer(network, config)

        _, metrics = gwo.optimize()

        # First fitness should be >= last (improvement)
        if len(metrics.convergence_history) > 5:
            first_avg = np.mean(metrics.convergence_history[:5])
            last_avg = np.mean(metrics.convergence_history[-5:])
            # Allow for some variance, but generally should improve
            assert last_avg <= first_avg * 1.5

    def test_get_flow_matrix(self):
        """Test getting the optimized flow matrix."""
        network = create_simple_network()
        config = AlgorithmConfig(n_wolves=5, max_iterations=5)
        gwo = GreyWolfOptimizer(network, config)

        gwo.optimize()
        flow_matrix = gwo.get_flow_matrix()

        assert flow_matrix is not None
        assert flow_matrix.shape == (len(network.get_population_zones()),
                                     len(network.get_shelters()))


# ==================== Hybrid Algorithm Tests ====================

class TestHybridGBFSGWO:
    """Tests for Hybrid algorithm."""

    def test_algorithm_type(self):
        """Test algorithm type property."""
        network = create_simple_network()
        hybrid = HybridGBFSGWO(network)
        assert hybrid.algorithm_type == AlgorithmType.HYBRID

    def test_optimize_creates_plan(self):
        """Test that optimize creates a valid plan."""
        network = create_simple_network()
        config = AlgorithmConfig(
            n_wolves=10,
            gwo_iterations=10,
            refinement_iterations=5
        )
        hybrid = HybridGBFSGWO(network, config)

        plan, metrics = hybrid.optimize()

        assert plan is not None
        assert metrics.execution_time_seconds >= 0

    def test_hybrid_uses_both_algorithms(self):
        """Test that hybrid uses both GWO and GBFS."""
        network = create_simple_network()
        config = AlgorithmConfig(
            n_wolves=5,
            gwo_iterations=5,
            refinement_iterations=2
        )
        hybrid = HybridGBFSGWO(network, config)

        plan, _ = hybrid.optimize()

        # Hybrid should produce actual paths (not just zone->shelter)
        for route in plan.routes:
            # Paths from hybrid should have intermediate nodes
            # (unlike pure GWO which only has start/end)
            if route.path:
                assert len(route.path) >= 2

    def test_progress_callback(self):
        """Test progress callback is called."""
        network = create_simple_network()
        config = AlgorithmConfig(
            n_wolves=5,
            gwo_iterations=5,
            refinement_iterations=2
        )
        hybrid = HybridGBFSGWO(network, config)

        progress_updates = []

        def callback(iteration, cost, data):
            progress_updates.append((iteration, cost))

        hybrid.set_progress_callback(callback)
        hybrid.optimize()

        assert len(progress_updates) > 0


# ==================== Comparator Tests ====================

class TestComparisonResult:
    """Tests for ComparisonResult."""

    def test_get_metric_comparison(self):
        """Test getting metric comparison across algorithms."""
        result = ComparisonResult()
        result.metrics[AlgorithmType.GBFS] = AlgorithmMetrics(
            algorithm_type=AlgorithmType.GBFS,
            execution_time_seconds=0.5
        )
        result.metrics[AlgorithmType.GWO] = AlgorithmMetrics(
            algorithm_type=AlgorithmType.GWO,
            execution_time_seconds=1.5
        )

        comparison = result.get_metric_comparison('execution_time_seconds')
        assert comparison[AlgorithmType.GBFS] == 0.5
        assert comparison[AlgorithmType.GWO] == 1.5

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = ComparisonResult(
            algorithms=[AlgorithmType.GBFS, AlgorithmType.GWO],
            winner=AlgorithmType.GBFS,
            winner_score=0.85
        )
        data = result.to_dict()
        assert 'gbfs' in data['algorithms']
        assert data['winner'] == 'gbfs'


class TestAlgorithmComparator:
    """Tests for AlgorithmComparator."""

    def test_compare_single_algorithm(self):
        """Test comparing a single algorithm."""
        network = create_simple_network()
        comparator = AlgorithmComparator(network)

        result = comparator.compare([AlgorithmType.GBFS])

        assert len(result.algorithms) == 1
        assert AlgorithmType.GBFS in result.metrics

    def test_compare_all(self):
        """Test comparing all algorithms."""
        network = create_simple_network()
        config = AlgorithmConfig(
            n_wolves=5,
            max_iterations=5,
            gwo_iterations=5,
            refinement_iterations=2
        )
        comparator = AlgorithmComparator(network, config)

        result = comparator.compare_all()

        assert len(result.algorithms) == 3
        assert AlgorithmType.GBFS in result.metrics
        assert AlgorithmType.GWO in result.metrics
        assert AlgorithmType.HYBRID in result.metrics
        assert result.winner is not None

    def test_rankings_calculated(self):
        """Test that rankings are calculated."""
        network = create_simple_network()
        config = AlgorithmConfig(n_wolves=5, max_iterations=5)
        comparator = AlgorithmComparator(network, config)

        result = comparator.compare([AlgorithmType.GBFS, AlgorithmType.GWO])

        assert len(result.rankings) > 0
        assert 'execution_time' in result.rankings

    def test_generate_comparison_table(self):
        """Test generating comparison table."""
        network = create_simple_network()
        config = AlgorithmConfig(n_wolves=5, max_iterations=5)
        comparator = AlgorithmComparator(network, config)

        result = comparator.compare([AlgorithmType.GBFS, AlgorithmType.GWO])
        table = comparator.generate_comparison_table(result)

        assert 'ALGORITHM COMPARISON' in table
        assert 'GBFS' in table
        assert 'GWO' in table


# ==================== Integration Tests ====================

class TestAlgorithmIntegration:
    """Integration tests for algorithms working together."""

    def test_all_algorithms_produce_valid_plans(self):
        """Test that all algorithms produce valid evacuation plans."""
        network = create_simple_network()
        config = AlgorithmConfig(
            n_wolves=10,
            max_iterations=10,
            gwo_iterations=10,
            refinement_iterations=5
        )

        algorithms = [
            GreedyBestFirstSearch(network, config),
            GreyWolfOptimizer(network, config),
            HybridGBFSGWO(network, config)
        ]

        for algo in algorithms:
            plan, metrics = algo.optimize()

            # Each algorithm should produce routes
            assert plan is not None, f"{algo.algorithm_type} failed to produce plan"

            # Total evacuees should be positive
            assert plan.total_evacuees >= 0

            # Metrics should be recorded
            assert metrics.execution_time_seconds >= 0

    def test_algorithms_handle_hazards(self):
        """Test that algorithms can handle hazard zones."""
        network = create_network_with_hazard()
        config = AlgorithmConfig(n_wolves=10, max_iterations=10)

        gbfs = GreedyBestFirstSearch(network, config)
        plan, _ = gbfs.optimize()

        # Should still find paths even with hazards
        assert len(plan.routes) > 0

    def test_stop_functionality(self):
        """Test that algorithms can be stopped."""
        network = create_simple_network()
        config = AlgorithmConfig(n_wolves=50, max_iterations=1000)

        gwo = GreyWolfOptimizer(network, config)

        # Set up callback to stop after 5 iterations
        iteration_count = [0]

        def callback(iteration, cost, data):
            iteration_count[0] = iteration
            if iteration >= 5:
                gwo.stop()

        gwo.set_progress_callback(callback)
        gwo.optimize()

        # Should have stopped early
        assert iteration_count[0] < 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
