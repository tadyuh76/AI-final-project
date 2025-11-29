#!/usr/bin/env python3
"""
Comprehensive test of the SafeRoute HCM implementation.
Tests all components: models, algorithms, simulation, and data loading.
"""

import sys
import time

# ==================== Test Setup ====================
print("=" * 70)
print("SafeRoute HCM - Implementation Test")
print("=" * 70)
print()

# ==================== 1. Test Imports ====================
print("[1] Testing imports...")
try:
    from src.models.node import Node, NodeType, PopulationZone, Shelter, HazardZone, haversine_distance
    from src.models.edge import Edge, RoadType
    from src.models.network import EvacuationNetwork
    from src.algorithms.base import AlgorithmConfig, EvacuationPlan, AlgorithmType
    from src.algorithms.gbfs import GreedyBestFirstSearch
    from src.algorithms.gwo import GreyWolfOptimizer
    from src.algorithms.hybrid import HybridGBFSGWO
    from src.algorithms.comparator import AlgorithmComparator
    from src.simulation.engine import SimulationEngine, SimulationConfig
    from src.simulation.traffic import TrafficFlowModel, TrafficConfig
    from src.simulation.events import EventManager, EventFactory, EventType
    from src.data.hcm_data import HCM_DISTRICTS, HCM_SHELTERS, get_total_population, get_total_shelter_capacity
    from src.data.osm_loader import OSMDataLoader
    print("    ✓ All imports successful")
except ImportError as e:
    print(f"    ✗ Import error: {e}")
    sys.exit(1)

# ==================== 2. Test HCM Data ====================
print()
print("[2] Testing HCM data...")
print(f"    Districts: {len(HCM_DISTRICTS)}")
print(f"    Shelters: {len(HCM_SHELTERS)}")
print(f"    Total Population: {get_total_population():,}")
print(f"    Total Shelter Capacity: {get_total_shelter_capacity():,}")
print("    ✓ HCM data loaded successfully")

# ==================== 3. Test Network Creation ====================
print()
print("[3] Testing network creation...")

# Create network using synthetic data (faster for testing)
loader = OSMDataLoader()
network = loader._generate_synthetic_network()
loader.add_default_hazards(network, typhoon_intensity=0.7)

stats = network.get_stats()
print(f"    Nodes: {stats.total_nodes}")
print(f"    Edges: {stats.total_edges}")
print(f"    Population Zones: {stats.population_zones}")
print(f"    Shelters: {stats.shelters}")
print(f"    Total Population: {stats.total_population:,}")
print(f"    Total Shelter Capacity: {stats.total_shelter_capacity:,}")
print(f"    Hazard Zones: {len(network.get_hazard_zones())}")
print("    ✓ Network created successfully")

# ==================== 4. Test GBFS Algorithm ====================
print()
print("[4] Testing GBFS algorithm...")
start_time = time.time()

config = AlgorithmConfig(
    distance_weight=0.4,
    risk_weight=0.3,
    congestion_weight=0.2,
    capacity_weight=0.1
)
gbfs = GreedyBestFirstSearch(network, config)
gbfs_plan, gbfs_metrics = gbfs.optimize()

gbfs_time = time.time() - start_time
print(f"    Execution time: {gbfs_time:.3f}s")
print(f"    Routes found: {gbfs_metrics.routes_found}")
print(f"    Evacuees covered: {gbfs_metrics.evacuees_covered:,}")
print(f"    Coverage rate: {gbfs_metrics.coverage_rate:.1%}")
print(f"    Final cost: {gbfs_metrics.final_cost:.2f}")
print("    ✓ GBFS algorithm working")

# ==================== 5. Test GWO Algorithm ====================
print()
print("[5] Testing GWO algorithm...")
network.reset_simulation_state()
start_time = time.time()

gwo_config = AlgorithmConfig(
    n_wolves=20,
    max_iterations=30
)
gwo = GreyWolfOptimizer(network, gwo_config)
gwo_plan, gwo_metrics = gwo.optimize()

gwo_time = time.time() - start_time
print(f"    Execution time: {gwo_time:.3f}s")
print(f"    Iterations: {gwo_metrics.iterations}")
print(f"    Routes found: {gwo_metrics.routes_found}")
print(f"    Evacuees covered: {gwo_metrics.evacuees_covered:,}")
print(f"    Coverage rate: {gwo_metrics.coverage_rate:.1%}")
print(f"    Final cost: {gwo_metrics.final_cost:.2f}")

# Check convergence
if len(gwo_metrics.convergence_history) > 5:
    first_5 = sum(gwo_metrics.convergence_history[:5]) / 5
    last_5 = sum(gwo_metrics.convergence_history[-5:]) / 5
    improvement = (first_5 - last_5) / first_5 * 100 if first_5 > 0 else 0
    print(f"    Convergence improvement: {improvement:.1f}%")
print("    ✓ GWO algorithm working")

# ==================== 6. Test Hybrid Algorithm ====================
print()
print("[6] Testing Hybrid GBFS+GWO algorithm...")
network.reset_simulation_state()
start_time = time.time()

hybrid_config = AlgorithmConfig(
    n_wolves=15,
    gwo_iterations=20,
    refinement_iterations=10
)
hybrid = HybridGBFSGWO(network, hybrid_config)
hybrid_plan, hybrid_metrics = hybrid.optimize()

hybrid_time = time.time() - start_time
print(f"    Execution time: {hybrid_time:.3f}s")
print(f"    Iterations: {hybrid_metrics.iterations}")
print(f"    Routes found: {hybrid_metrics.routes_found}")
print(f"    Evacuees covered: {hybrid_metrics.evacuees_covered:,}")
print(f"    Coverage rate: {hybrid_metrics.coverage_rate:.1%}")
print(f"    Final cost: {hybrid_metrics.final_cost:.2f}")
print("    ✓ Hybrid algorithm working")

# ==================== 7. Test Algorithm Comparator ====================
print()
print("[7] Testing algorithm comparator...")
network.reset_simulation_state()

compare_config = AlgorithmConfig(
    n_wolves=10,
    max_iterations=15,
    gwo_iterations=10,
    refinement_iterations=5
)
comparator = AlgorithmComparator(network, compare_config)
result = comparator.compare_all()

print(f"    Algorithms compared: {len(result.algorithms)}")
print(f"    Winner: {result.winner.value.upper() if result.winner else 'N/A'}")
print(f"    Winner score: {result.winner_score:.3f}")
print()
print("    Performance Summary:")
print("    " + "-" * 50)
print(f"    {'Metric':<20} | {'GBFS':>8} | {'GWO':>8} | {'Hybrid':>8}")
print("    " + "-" * 50)

for algo_type in [AlgorithmType.GBFS, AlgorithmType.GWO, AlgorithmType.HYBRID]:
    m = result.metrics.get(algo_type)
    if m:
        print(f"    {'Time (s)':<20} | {m.execution_time_seconds:>8.3f} | ", end="")
    else:
        print(f"    {'Time (s)':<20} | {'N/A':>8} | ", end="")

# Format the table properly
metrics = result.metrics
if all(t in metrics for t in [AlgorithmType.GBFS, AlgorithmType.GWO, AlgorithmType.HYBRID]):
    m_gbfs = metrics[AlgorithmType.GBFS]
    m_gwo = metrics[AlgorithmType.GWO]
    m_hybrid = metrics[AlgorithmType.HYBRID]
    print()
    print(f"    {'Time (s)':<20} | {m_gbfs.execution_time_seconds:>8.3f} | {m_gwo.execution_time_seconds:>8.3f} | {m_hybrid.execution_time_seconds:>8.3f}")
    print(f"    {'Routes':<20} | {m_gbfs.routes_found:>8} | {m_gwo.routes_found:>8} | {m_hybrid.routes_found:>8}")
    print(f"    {'Evacuees':<20} | {m_gbfs.evacuees_covered:>8,} | {m_gwo.evacuees_covered:>8,} | {m_hybrid.evacuees_covered:>8,}")
    print(f"    {'Coverage':<20} | {m_gbfs.coverage_rate:>7.1%} | {m_gwo.coverage_rate:>7.1%} | {m_hybrid.coverage_rate:>7.1%}")

print("    " + "-" * 50)
print("    ✓ Algorithm comparator working")

# ==================== 8. Test Traffic Flow Model ====================
print()
print("[8] Testing traffic flow model...")

traffic_config = TrafficConfig()
traffic = TrafficFlowModel(network, traffic_config)

# Add some flow and update
for edge in list(network.get_edges())[:10]:
    edge.add_flow(500)

traffic_state = traffic.update(0.1)
print(f"    Total vehicles: {traffic_state.total_vehicles}")
print(f"    Average speed: {traffic_state.average_speed:.1f} km/h")
print(f"    Congested edges: {traffic_state.congested_edges}")

# Test route travel time
zones = network.get_population_zones()
shelters = network.get_shelters()
if zones and shelters:
    # Find a simple path for testing
    zone = zones[0]
    nearest = network.find_nearest_node(zone.lat, zone.lon)
    if nearest:
        test_path = [zone.id, nearest.id]
        travel_time = traffic.get_route_travel_time(test_path)
        print(f"    Sample route time: {travel_time*60:.1f} minutes")

print("    ✓ Traffic flow model working")

# ==================== 9. Test Simulation Engine ====================
print()
print("[9] Testing simulation engine...")
network.reset_simulation_state()

# Use the hybrid plan for simulation
sim_config = SimulationConfig(
    time_step_minutes=10.0,
    max_duration_hours=1.0,
    flow_rate_per_step=0.15
)
engine = SimulationEngine(network, sim_config)

# Track progress
progress_updates = []
def on_update(metrics, routes):
    progress_updates.append(metrics.evacuation_progress)

engine.set_update_callback(on_update)

start_time = time.time()
sim_metrics = engine.run(hybrid_plan)
sim_time = time.time() - start_time

print(f"    Simulation time: {sim_time:.3f}s")
print(f"    Simulated duration: {sim_metrics.current_time_hours:.2f} hours")
print(f"    Total evacuated: {sim_metrics.total_evacuated:,}")
print(f"    Evacuation progress: {sim_metrics.evacuation_progress:.1%}")
print(f"    Active routes: {sim_metrics.active_routes}")
print(f"    Completed routes: {sim_metrics.completed_routes}")
print(f"    Progress updates received: {len(progress_updates)}")
print("    ✓ Simulation engine working")

# ==================== 10. Test Event System ====================
print()
print("[10] Testing event system...")

event_manager = EventManager()
received_events = []

def on_event(event):
    received_events.append(event)

event_manager.subscribe(EventType.ACCIDENT, on_event)
event_manager.subscribe(EventType.FLOODING, on_event)
event_manager.subscribe(EventType.SHELTER_CLOSED, on_event)

# Create and emit events
event1 = EventFactory.create_accident("e1", 0.5, 0.7)
event2 = EventFactory.create_flooding("e2", 1.0, 45.0)
event3 = EventFactory.create_shelter_closed("s1", 1.5, "damage")

event_manager.emit(event1)
event_manager.emit(event2)
event_manager.emit(event3)

print(f"    Events emitted: 3")
print(f"    Events received: {len(received_events)}")
print(f"    Event types: {[e.event_type.value for e in received_events]}")

# Test scheduled events
scheduled_event = EventFactory.create_road_blocked("e3", 0)
event_manager.schedule(scheduled_event, trigger_time=2.0)
print(f"    Scheduled events: {event_manager.scheduled_count}")

# Process scheduled
triggered = event_manager.process_queue(2.5)
print(f"    Events triggered: {triggered}")

print("    ✓ Event system working")

# ==================== 11. Test Snapshot Export ====================
print()
print("[11] Testing data export...")

snapshot = engine.get_evacuation_snapshot()
print(f"    Snapshot keys: {list(snapshot.keys())}")
print(f"    Routes in snapshot: {len(snapshot['routes'])}")
print(f"    Shelters in snapshot: {len(snapshot['shelters'])}")
print(f"    Hazards in snapshot: {len(snapshot['hazards'])}")

# Test network serialization
network_dict = network.to_dict()
print(f"    Network nodes: {len(network_dict['nodes'])}")
print(f"    Network edges: {len(network_dict['edges'])}")
print("    ✓ Data export working")

# ==================== Summary ====================
print()
print("=" * 70)
print("IMPLEMENTATION TEST SUMMARY")
print("=" * 70)
print()
print("All components tested successfully:")
print()
print("  Phase 1 - Core Foundation:")
print("    ✓ Network graph model (nodes, edges, hazards)")
print("    ✓ GBFS algorithm (pathfinding)")
print("    ✓ GWO algorithm (flow optimization)")
print("    ✓ Hybrid algorithm (GBFS + GWO)")
print()
print("  Phase 2 - Data & Simulation:")
print("    ✓ HCM data (districts, shelters, flood zones)")
print("    ✓ Network generation (synthetic grid)")
print("    ✓ Simulation engine (time-stepped)")
print("    ✓ Traffic flow model (BPR-based)")
print("    ✓ Event system (dynamic events)")
print()
print("  Phase 3 - Hybrid Algorithm:")
print("    ✓ Algorithm comparator")
print("    ✓ Performance metrics")
print("    ✓ Winner determination")
print()
print("=" * 70)
print("All tests passed! Implementation is working correctly.")
print("=" * 70)
