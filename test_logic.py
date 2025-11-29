#!/usr/bin/env python3
"""
Test script to verify all project logic is working correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("TEST 1: Import Modules")
    print("=" * 60)

    try:
        from src.models.node import (
            Node, NodeType, PopulationZone, Shelter, HazardZone,
            haversine_distance
        )
        print("  [OK] src.models.node")

        from src.models.edge import Edge, RoadType
        print("  [OK] src.models.edge")

        from src.models.network import EvacuationNetwork
        print("  [OK] src.models.network")

        from src.data.hcm_data import (
            HCM_DISTRICTS, HCM_SHELTERS, FLOOD_PRONE_AREAS,
            get_total_population, get_total_shelter_capacity
        )
        print("  [OK] src.data.hcm_data")

        from src.data.osm_loader import OSMDataLoader
        print("  [OK] src.data.osm_loader")

        from src.algorithms.base import (
            AlgorithmType, AlgorithmConfig, EvacuationPlan, EvacuationRoute
        )
        print("  [OK] src.algorithms.base")

        from src.algorithms.gbfs import GreedyBestFirstSearch
        print("  [OK] src.algorithms.gbfs")

        from src.algorithms.gwo import GreyWolfOptimizer
        print("  [OK] src.algorithms.gwo")

        from src.algorithms.hybrid import HybridGBFSGWO
        print("  [OK] src.algorithms.hybrid")

        from src.algorithms.comparator import AlgorithmComparator, run_comparison
        print("  [OK] src.algorithms.comparator")

        print("\n  All imports successful!")
        return True

    except ImportError as e:
        print(f"\n  [FAIL] Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hcm_data():
    """Test HCM city data."""
    print("\n" + "=" * 60)
    print("TEST 2: HCM City Data")
    print("=" * 60)

    from src.data.hcm_data import (
        HCM_DISTRICTS, HCM_SHELTERS, FLOOD_PRONE_AREAS,
        get_total_population, get_total_shelter_capacity,
        get_districts_by_flood_risk
    )

    print(f"  Districts: {len(HCM_DISTRICTS)}")
    print(f"  Shelters: {len(HCM_SHELTERS)}")
    print(f"  Flood zones: {len(FLOOD_PRONE_AREAS)}")
    print(f"  Total population: {get_total_population():,}")
    print(f"  Total shelter capacity: {get_total_shelter_capacity():,}")

    high_risk = get_districts_by_flood_risk(0.7)
    print(f"  High-risk districts (>70%): {len(high_risk)}")
    for d in high_risk[:3]:
        print(f"    - {d.name}: {d.flood_risk:.0%} risk")

    return True


def test_network_creation():
    """Test network creation and basic operations."""
    print("\n" + "=" * 60)
    print("TEST 3: Network Creation")
    print("=" * 60)

    from src.models.network import EvacuationNetwork
    from src.models.node import Node, NodeType, PopulationZone, Shelter, HazardZone
    from src.models.edge import Edge, RoadType

    # Create empty network
    network = EvacuationNetwork()
    print(f"  Created empty network: {network}")

    # Add nodes
    zone1 = PopulationZone(
        id="zone_1", lat=10.78, lon=106.70,
        population=50000, district_name="District 1"
    )
    network.add_node(zone1)

    shelter1 = Shelter(
        id="shelter_1", lat=10.79, lon=106.71,
        capacity=10000, shelter_type="stadium"
    )
    network.add_node(shelter1)

    intersection = Node(
        id="int_1", lat=10.785, lon=106.705,
        node_type=NodeType.INTERSECTION
    )
    network.add_node(intersection)

    print(f"  Added 3 nodes")

    # Add edges
    edge1 = Edge(
        id="e1", source_id="zone_1", target_id="int_1",
        length_km=1.5, road_type=RoadType.PRIMARY, lanes=2
    )
    network.add_edge(edge1)

    edge2 = Edge(
        id="e2", source_id="int_1", target_id="shelter_1",
        length_km=2.0, road_type=RoadType.PRIMARY, lanes=2
    )
    network.add_edge(edge2)

    print(f"  Added 2 edges")

    # Add hazard zone
    hazard = HazardZone(
        center_lat=10.785, center_lon=106.705,
        radius_km=0.5, risk_level=0.8
    )
    network.add_hazard_zone(hazard)
    print(f"  Added 1 hazard zone")

    # Get stats
    stats = network.get_stats()
    print(f"  Network stats:")
    print(f"    - Nodes: {stats.total_nodes}")
    print(f"    - Edges: {stats.total_edges}")
    print(f"    - Population zones: {stats.population_zones}")
    print(f"    - Shelters: {stats.shelters}")
    print(f"    - Total population: {stats.total_population:,}")

    # Test neighbors
    neighbors = network.get_neighbors("zone_1")
    print(f"  Neighbors of zone_1: {neighbors}")

    return True


def test_osm_loader():
    """Test OSM data loader (generates synthetic network)."""
    print("\n" + "=" * 60)
    print("TEST 4: OSM Data Loader")
    print("=" * 60)

    from src.data.osm_loader import OSMDataLoader

    loader = OSMDataLoader()
    print("  Loading HCM network (using cache or generating synthetic)...")

    # Force synthetic generation for testing
    network = loader.load_hcm_network(use_cache=False)

    stats = network.get_stats()
    print(f"  Network loaded:")
    print(f"    - Nodes: {stats.total_nodes}")
    print(f"    - Edges: {stats.total_edges}")
    print(f"    - Population zones: {stats.population_zones}")
    print(f"    - Shelters: {stats.shelters}")
    print(f"    - Total population: {stats.total_population:,}")
    print(f"    - Shelter capacity: {stats.total_shelter_capacity:,}")

    # Add hazards
    loader.add_default_hazards(network, typhoon_intensity=0.7)
    print(f"    - Hazard zones: {len(network.get_hazard_zones())}")

    return network


def test_gbfs(network):
    """Test GBFS algorithm."""
    print("\n" + "=" * 60)
    print("TEST 5: GBFS Algorithm")
    print("=" * 60)

    from src.algorithms.gbfs import GreedyBestFirstSearch
    from src.algorithms.base import AlgorithmConfig

    config = AlgorithmConfig(
        distance_weight=0.4,
        risk_weight=0.3,
        congestion_weight=0.2,
        capacity_weight=0.1
    )

    gbfs = GreedyBestFirstSearch(network, config)
    print("  Running GBFS optimization...")

    plan, metrics = gbfs.optimize()

    print(f"  Results:")
    print(f"    - Execution time: {metrics.execution_time_seconds:.3f}s")
    print(f"    - Routes found: {metrics.routes_found}")
    print(f"    - Evacuees covered: {metrics.evacuees_covered:,}")
    print(f"    - Coverage rate: {metrics.coverage_rate:.1%}")
    print(f"    - Final cost: {metrics.final_cost:.2f}")

    if plan.routes:
        print(f"    - Sample route: {plan.routes[0].zone_id} -> {plan.routes[0].shelter_id}")
        print(f"      Path length: {len(plan.routes[0].path)} nodes")
        print(f"      Flow: {plan.routes[0].flow:,} people")

    return plan, metrics


def test_gwo(network):
    """Test GWO algorithm."""
    print("\n" + "=" * 60)
    print("TEST 6: GWO Algorithm")
    print("=" * 60)

    from src.algorithms.gwo import GreyWolfOptimizer
    from src.algorithms.base import AlgorithmConfig

    # Reset network state
    network.reset_simulation_state()

    config = AlgorithmConfig(
        n_wolves=20,
        max_iterations=30  # Reduced for faster testing
    )

    gwo = GreyWolfOptimizer(network, config)
    print("  Running GWO optimization (30 iterations)...")

    plan, metrics = gwo.optimize()

    print(f"  Results:")
    print(f"    - Execution time: {metrics.execution_time_seconds:.3f}s")
    print(f"    - Iterations: {metrics.iterations}")
    print(f"    - Routes found: {metrics.routes_found}")
    print(f"    - Evacuees covered: {metrics.evacuees_covered:,}")
    print(f"    - Coverage rate: {metrics.coverage_rate:.1%}")
    print(f"    - Final cost: {metrics.final_cost:.2f}")

    # Check convergence
    if metrics.convergence_history:
        print(f"    - Initial cost: {metrics.convergence_history[0]:.2f}")
        print(f"    - Final cost: {metrics.convergence_history[-1]:.2f}")
        improvement = (metrics.convergence_history[0] - metrics.convergence_history[-1]) / metrics.convergence_history[0]
        print(f"    - Improvement: {improvement:.1%}")

    return plan, metrics


def test_hybrid(network):
    """Test Hybrid algorithm."""
    print("\n" + "=" * 60)
    print("TEST 7: Hybrid GBFS+GWO Algorithm")
    print("=" * 60)

    from src.algorithms.hybrid import HybridGBFSGWO
    from src.algorithms.base import AlgorithmConfig

    # Reset network state
    network.reset_simulation_state()

    config = AlgorithmConfig(
        n_wolves=20,
        gwo_iterations=20,
        refinement_iterations=10
    )

    hybrid = HybridGBFSGWO(network, config)
    print("  Running Hybrid optimization...")

    plan, metrics = hybrid.optimize()

    print(f"  Results:")
    print(f"    - Execution time: {metrics.execution_time_seconds:.3f}s")
    print(f"    - Total iterations: {metrics.iterations}")
    print(f"    - Routes found: {metrics.routes_found}")
    print(f"    - Evacuees covered: {metrics.evacuees_covered:,}")
    print(f"    - Coverage rate: {metrics.coverage_rate:.1%}")
    print(f"    - Final cost: {metrics.final_cost:.2f}")
    print(f"    - Avg path length: {metrics.average_path_length:.1f} nodes")

    return plan, metrics


def test_comparator(network):
    """Test algorithm comparator."""
    print("\n" + "=" * 60)
    print("TEST 8: Algorithm Comparator")
    print("=" * 60)

    from src.algorithms.comparator import AlgorithmComparator, run_comparison
    from src.algorithms.base import AlgorithmConfig, AlgorithmType

    config = AlgorithmConfig(
        n_wolves=15,
        max_iterations=20,
        gwo_iterations=15,
        refinement_iterations=5
    )

    comparator = AlgorithmComparator(network, config)
    print("  Running comparison of all algorithms...")

    result = comparator.compare_all()

    print("\n" + comparator.generate_comparison_table(result))

    print(f"\n  Rankings by metric:")
    for metric, ranking in result.rankings.items():
        print(f"    {metric}: {' > '.join(a.value for a in ranking)}")

    return result


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# SafeRoute HCM - Logic Test Suite")
    print("#" * 60)

    # Test 1: Imports
    if not test_imports():
        print("\n[FATAL] Import test failed. Cannot continue.")
        return 1

    # Test 2: HCM Data
    test_hcm_data()

    # Test 3: Network Creation
    test_network_creation()

    # Test 4: OSM Loader
    network = test_osm_loader()

    # Test 5: GBFS
    test_gbfs(network)

    # Test 6: GWO
    test_gwo(network)

    # Test 7: Hybrid
    test_hybrid(network)

    # Test 8: Comparator
    test_comparator(network)

    print("\n" + "#" * 60)
    print("# ALL TESTS COMPLETED SUCCESSFULLY!")
    print("#" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
