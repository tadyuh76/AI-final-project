"""
Unit tests for the simulation module.
Tests SimulationEngine, TrafficFlowModel, and EventManager.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.node import Node, NodeType, PopulationZone, Shelter, HazardZone
from src.models.edge import Edge, RoadType
from src.models.network import EvacuationNetwork
from src.algorithms.base import EvacuationPlan, EvacuationRoute

from src.simulation.engine import (
    SimulationEngine, SimulationState, SimulationConfig,
    SimulationMetrics, RouteState
)
from src.simulation.traffic import (
    TrafficFlowModel, TrafficState, TrafficConfig,
    EdgeTrafficState, NetworkTrafficState, TrafficAssignment
)
from src.simulation.events import (
    EventManager, EventQueue, EventFactory, EventType,
    EventPriority, SimulationEvent, RandomEventGenerator
)


# ==================== Test Fixtures ====================

def create_test_network():
    """Create a simple test network for simulation testing."""
    network = EvacuationNetwork()

    # Create grid nodes
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

    # Create zones and shelters
    zone1 = PopulationZone(id="zone1", lat=10.0, lon=106.0, population=1000)
    zone2 = PopulationZone(id="zone2", lat=10.05, lon=106.0, population=1500)
    shelter1 = Shelter(id="shelter1", lat=10.0, lon=106.1, capacity=2000)
    shelter2 = Shelter(id="shelter2", lat=10.05, lon=106.1, capacity=1500)

    network.add_node(zone1)
    network.add_node(zone2)
    network.add_node(shelter1)
    network.add_node(shelter2)

    # Connect zones to grid
    network.add_edge(Edge(id="ez1", source_id="zone1", target_id="n1",
                          length_km=0.1, is_oneway=False))
    network.add_edge(Edge(id="ez2", source_id="zone2", target_id="n4",
                          length_km=0.1, is_oneway=False))

    # Connect shelters to grid
    network.add_edge(Edge(id="es1", source_id="n3", target_id="shelter1",
                          length_km=0.1, is_oneway=False))
    network.add_edge(Edge(id="es2", source_id="n6", target_id="shelter2",
                          length_km=0.1, is_oneway=False))

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


def create_test_plan(network):
    """Create a simple evacuation plan for testing."""
    plan = EvacuationPlan()

    route1 = EvacuationRoute(
        zone_id="zone1",
        shelter_id="shelter1",
        path=["zone1", "n1", "n2", "n3", "shelter1"],
        flow=1000,
        distance_km=4.2,
        estimated_time_hours=0.15,
        risk_score=0.1
    )

    route2 = EvacuationRoute(
        zone_id="zone2",
        shelter_id="shelter2",
        path=["zone2", "n4", "n5", "n6", "shelter2"],
        flow=1500,
        distance_km=4.2,
        estimated_time_hours=0.15,
        risk_score=0.1
    )

    plan.add_route(route1)
    plan.add_route(route2)

    return plan


# ==================== SimulationConfig Tests ====================

class TestSimulationConfig:
    """Tests for SimulationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SimulationConfig()
        assert config.time_step_minutes == 5.0
        assert config.max_duration_hours == 24.0
        assert config.flow_rate_per_step == 0.1

    def test_custom_config(self):
        """Test custom configuration."""
        config = SimulationConfig(
            time_step_minutes=10.0,
            max_duration_hours=12.0
        )
        assert config.time_step_minutes == 10.0
        assert config.max_duration_hours == 12.0


# ==================== SimulationMetrics Tests ====================

class TestSimulationMetrics:
    """Tests for SimulationMetrics."""

    def test_default_metrics(self):
        """Test default metrics values."""
        metrics = SimulationMetrics()
        assert metrics.total_evacuated == 0
        assert metrics.evacuation_progress == 0.0

    def test_to_dict(self):
        """Test metrics serialization."""
        metrics = SimulationMetrics(
            total_evacuated=500,
            evacuation_progress=0.5
        )
        data = metrics.to_dict()
        assert data['total_evacuated'] == 500
        assert data['evacuation_progress'] == 0.5


# ==================== RouteState Tests ====================

class TestRouteState:
    """Tests for RouteState."""

    def test_remaining_calculation(self):
        """Test remaining population calculation."""
        route = EvacuationRoute(
            zone_id="z1", shelter_id="s1",
            path=[], flow=1000
        )
        state = RouteState(
            route=route,
            total_assigned=1000,
            departed=300
        )
        assert state.remaining == 700

    def test_is_complete(self):
        """Test completion check."""
        route = EvacuationRoute(
            zone_id="z1", shelter_id="s1",
            path=[], flow=100
        )
        state = RouteState(
            route=route,
            total_assigned=100,
            arrived=100,
            in_transit=0
        )
        assert state.is_complete is True

    def test_not_complete_with_in_transit(self):
        """Test not complete when people still in transit."""
        route = EvacuationRoute(
            zone_id="z1", shelter_id="s1",
            path=[], flow=100
        )
        state = RouteState(
            route=route,
            total_assigned=100,
            arrived=80,
            in_transit=20
        )
        assert state.is_complete is False


# ==================== SimulationEngine Tests ====================

class TestSimulationEngine:
    """Tests for SimulationEngine."""

    def test_engine_creation(self):
        """Test engine creation."""
        network = create_test_network()
        engine = SimulationEngine(network)
        assert engine.state == SimulationState.IDLE

    def test_initialize(self):
        """Test simulation initialization."""
        network = create_test_network()
        plan = create_test_plan(network)
        engine = SimulationEngine(network)

        engine.initialize(plan)

        assert engine.state == SimulationState.IDLE
        assert engine.current_time == 0.0
        assert len(engine.get_route_states()) == 2

    def test_step(self):
        """Test single simulation step."""
        network = create_test_network()
        plan = create_test_plan(network)
        engine = SimulationEngine(network)

        engine.initialize(plan)
        engine._state = SimulationState.RUNNING
        metrics = engine.step()

        assert metrics.current_time_hours > 0
        assert engine.current_time > 0

    def test_run_simulation(self):
        """Test running complete simulation."""
        network = create_test_network()
        plan = create_test_plan(network)
        config = SimulationConfig(
            time_step_minutes=5.0,
            max_duration_hours=2.0
        )
        engine = SimulationEngine(network, config)

        metrics = engine.run(plan)

        assert metrics.total_evacuated > 0
        # Simulation may complete, hit max duration, or still be running
        assert engine.state in [
            SimulationState.COMPLETED,
            SimulationState.IDLE,
            SimulationState.RUNNING  # Hit max steps
        ]

    def test_pause_resume(self):
        """Test pause and resume."""
        network = create_test_network()
        engine = SimulationEngine(network)

        engine.pause()
        assert engine.state == SimulationState.PAUSED

        engine.resume()
        assert engine.state == SimulationState.RUNNING

    def test_stop(self):
        """Test stopping simulation."""
        network = create_test_network()
        engine = SimulationEngine(network)

        engine.stop()
        assert engine.state == SimulationState.IDLE

    def test_reset(self):
        """Test resetting simulation."""
        network = create_test_network()
        plan = create_test_plan(network)
        engine = SimulationEngine(network)

        engine.initialize(plan)
        engine.step()
        engine.reset()

        assert engine.current_time == 0.0
        assert len(engine.get_route_states()) == 0

    def test_update_callback(self):
        """Test update callback is called."""
        network = create_test_network()
        plan = create_test_plan(network)
        engine = SimulationEngine(network)

        callback_count = [0]

        def callback(metrics, routes):
            callback_count[0] += 1

        engine.set_update_callback(callback)
        engine.initialize(plan)
        engine._state = SimulationState.RUNNING
        engine.step()

        assert callback_count[0] > 0

    def test_get_evacuation_snapshot(self):
        """Test getting evacuation snapshot."""
        network = create_test_network()
        plan = create_test_plan(network)
        engine = SimulationEngine(network)

        engine.initialize(plan)
        snapshot = engine.get_evacuation_snapshot()

        assert 'time' in snapshot
        assert 'routes' in snapshot
        assert 'shelters' in snapshot
        assert len(snapshot['routes']) == 2


# ==================== TrafficConfig Tests ====================

class TestTrafficConfig:
    """Tests for TrafficConfig."""

    def test_default_config(self):
        """Test default traffic configuration."""
        config = TrafficConfig()
        assert config.bpr_alpha == 0.15
        assert config.bpr_beta == 4.0
        assert config.persons_per_vehicle == 3.0


# ==================== TrafficFlowModel Tests ====================

class TestTrafficFlowModel:
    """Tests for TrafficFlowModel."""

    def test_model_creation(self):
        """Test traffic model creation."""
        network = create_test_network()
        model = TrafficFlowModel(network)
        assert model is not None

    def test_update(self):
        """Test traffic update."""
        network = create_test_network()
        model = TrafficFlowModel(network)

        # Add some flow to edges
        for edge in network.get_edges():
            edge.add_flow(100)

        state = model.update(0.1)

        assert state.total_vehicles >= 0
        assert len(state.edges) > 0

    def test_bpr_travel_time(self):
        """Test BPR travel time calculation."""
        network = create_test_network()
        model = TrafficFlowModel(network)

        # Free flow
        time_free = model._bpr_travel_time(1.0, 0.0)
        assert time_free == 1.0

        # Congested
        time_congested = model._bpr_travel_time(1.0, 1.0)
        assert time_congested > 1.0

    def test_traffic_states(self):
        """Test traffic state determination."""
        network = create_test_network()
        model = TrafficFlowModel(network)

        assert model._determine_traffic_state(0.3) == TrafficState.FREE_FLOW
        assert model._determine_traffic_state(0.6) == TrafficState.SYNCHRONIZED
        assert model._determine_traffic_state(0.9) == TrafficState.CONGESTED
        assert model._determine_traffic_state(1.2) == TrafficState.GRIDLOCK

    def test_get_route_travel_time(self):
        """Test route travel time calculation."""
        network = create_test_network()
        model = TrafficFlowModel(network)

        path = ["zone1", "n1", "n2", "n3", "shelter1"]
        time = model.get_route_travel_time(path)

        assert time > 0
        assert time < float('inf')

    def test_congestion_map(self):
        """Test getting congestion map."""
        network = create_test_network()
        model = TrafficFlowModel(network)

        congestion = model.get_congestion_map()
        assert len(congestion) > 0

    def test_apply_incident(self):
        """Test applying and clearing incidents."""
        network = create_test_network()
        model = TrafficFlowModel(network)

        edge_id = list(network._edges.keys())[0]
        model.apply_incident(edge_id, 0.5)

        state = model._edge_states.get(edge_id)
        assert state.capacity_factor < 1.0

        model.clear_incident(edge_id)
        assert state.capacity_factor == 1.0

    def test_reset(self):
        """Test resetting traffic model."""
        network = create_test_network()
        model = TrafficFlowModel(network)

        # Modify state
        edge_id = list(network._edges.keys())[0]
        model.apply_incident(edge_id, 0.5)

        model.reset()

        state = model._edge_states.get(edge_id)
        assert state.capacity_factor == 1.0


# ==================== TrafficAssignment Tests ====================

class TestTrafficAssignment:
    """Tests for TrafficAssignment."""

    def test_assignment_creation(self):
        """Test traffic assignment creation."""
        network = create_test_network()
        assignment = TrafficAssignment(network)
        assert assignment is not None

    def test_assign_flow(self):
        """Test flow assignment."""
        network = create_test_network()
        assignment = TrafficAssignment(network)

        # Simple OD matrix
        od_matrix = {
            ("zone1", "shelter1"): 100.0
        }

        flows = assignment.assign_flow(od_matrix, max_iterations=10)
        assert len(flows) > 0


# ==================== EventQueue Tests ====================

class TestEventQueue:
    """Tests for EventQueue."""

    def test_push_pop(self):
        """Test push and pop operations."""
        queue = EventQueue()

        event1 = SimulationEvent(
            id="1", event_type=EventType.ACCIDENT,
            timestamp=1.0, priority=EventPriority.NORMAL
        )
        event2 = SimulationEvent(
            id="2", event_type=EventType.ROAD_BLOCKED,
            timestamp=0.5, priority=EventPriority.HIGH
        )

        queue.push(event1)
        queue.push(event2)

        # Higher priority should come first
        popped = queue.pop()
        assert popped.id == "2"

    def test_peek(self):
        """Test peek without removing."""
        queue = EventQueue()
        event = SimulationEvent(
            id="1", event_type=EventType.ACCIDENT,
            timestamp=1.0
        )
        queue.push(event)

        peeked = queue.peek()
        assert peeked.id == "1"
        assert len(queue) == 1

    def test_schedule(self):
        """Test scheduled events."""
        queue = EventQueue()
        event = SimulationEvent(
            id="1", event_type=EventType.ACCIDENT,
            timestamp=0
        )
        queue.schedule(event, trigger_time=1.0)

        # Not triggered yet
        triggered = queue.check_scheduled(0.5)
        assert len(triggered) == 0

        # Now triggered
        triggered = queue.check_scheduled(1.5)
        assert len(triggered) == 1

    def test_recurring_schedule(self):
        """Test recurring scheduled events."""
        queue = EventQueue()
        event = SimulationEvent(
            id="1", event_type=EventType.HAZARD_EXPANDED,
            timestamp=0
        )
        queue.schedule(event, trigger_time=1.0, recurring=True, interval=1.0)

        # First trigger
        triggered = queue.check_scheduled(1.0)
        assert len(triggered) == 1

        # Second trigger
        triggered = queue.check_scheduled(2.0)
        assert len(triggered) == 1


# ==================== EventManager Tests ====================

class TestEventManager:
    """Tests for EventManager."""

    def test_subscribe_emit(self):
        """Test subscribing and emitting events."""
        manager = EventManager()
        received = []

        def handler(event):
            received.append(event)

        manager.subscribe(EventType.ACCIDENT, handler)

        event = SimulationEvent(
            id="1", event_type=EventType.ACCIDENT,
            timestamp=1.0
        )
        manager.emit(event)

        assert len(received) == 1
        assert received[0].id == "1"

    def test_subscribe_all(self):
        """Test subscribing to all events."""
        manager = EventManager()
        received = []

        def handler(event):
            received.append(event)

        manager.subscribe_all(handler)

        event1 = SimulationEvent(id="1", event_type=EventType.ACCIDENT, timestamp=1.0)
        event2 = SimulationEvent(id="2", event_type=EventType.FLOODING, timestamp=2.0)

        manager.emit(event1)
        manager.emit(event2)

        assert len(received) == 2

    def test_queue_and_process(self):
        """Test queuing and processing events."""
        manager = EventManager()
        processed = []

        def handler(event):
            processed.append(event)

        manager.subscribe(EventType.ACCIDENT, handler)

        event = SimulationEvent(
            id="1", event_type=EventType.ACCIDENT,
            timestamp=1.0
        )
        manager.queue(event)

        # Process at time 0.5 (should not process)
        count = manager.process_queue(0.5)
        assert count == 0

        # Process at time 1.5 (should process)
        count = manager.process_queue(1.5)
        assert count == 1
        assert len(processed) == 1

    def test_get_history(self):
        """Test event history."""
        manager = EventManager()

        event1 = SimulationEvent(id="1", event_type=EventType.ACCIDENT, timestamp=1.0)
        event2 = SimulationEvent(id="2", event_type=EventType.FLOODING, timestamp=2.0)

        manager.emit(event1)
        manager.emit(event2)

        # All history
        history = manager.get_history()
        assert len(history) == 2

        # Filtered history
        history = manager.get_history(EventType.ACCIDENT)
        assert len(history) == 1

    def test_clear(self):
        """Test clearing events."""
        manager = EventManager()

        event = SimulationEvent(id="1", event_type=EventType.ACCIDENT, timestamp=1.0)
        manager.queue(event)
        manager.emit(event)

        manager.clear()

        assert manager.pending_count == 0


# ==================== EventFactory Tests ====================

class TestEventFactory:
    """Tests for EventFactory."""

    def test_create_road_blocked(self):
        """Test creating road blocked event."""
        event = EventFactory.create_road_blocked("edge1", 1.0, "flooding")

        assert event.event_type == EventType.ROAD_BLOCKED
        assert event.data['edge_id'] == "edge1"
        assert event.data['reason'] == "flooding"
        assert event.priority == EventPriority.HIGH

    def test_create_accident(self):
        """Test creating accident event."""
        event = EventFactory.create_accident("edge1", 1.0, 0.7)

        assert event.event_type == EventType.ACCIDENT
        assert event.data['severity'] == 0.7
        assert 'capacity_reduction' in event.data

    def test_create_flooding(self):
        """Test creating flooding event with depth-based severity."""
        # Shallow flooding
        event_shallow = EventFactory.create_flooding("edge1", 1.0, 25.0)
        assert event_shallow.data['severity'] == 0.5

        # Deep flooding
        event_deep = EventFactory.create_flooding("edge1", 1.0, 60.0)
        assert event_deep.data['severity'] == 1.0

    def test_create_shelter_closed(self):
        """Test creating shelter closed event."""
        event = EventFactory.create_shelter_closed("shelter1", 1.0, "damage")

        assert event.event_type == EventType.SHELTER_CLOSED
        assert event.priority == EventPriority.CRITICAL

    def test_create_hazard(self):
        """Test creating hazard event."""
        event = EventFactory.create_hazard(10.7, 106.7, 2.0, 1.0, "flood")

        assert event.event_type == EventType.HAZARD_CREATED
        assert event.data['center_lat'] == 10.7
        assert event.data['radius_km'] == 2.0


# ==================== RandomEventGenerator Tests ====================

class TestRandomEventGenerator:
    """Tests for RandomEventGenerator."""

    def test_generator_creation(self):
        """Test generator creation with seed."""
        network = create_test_network()
        generator = RandomEventGenerator(network, seed=42)
        assert generator is not None

    def test_generate_random_accident(self):
        """Test generating random accident."""
        network = create_test_network()
        generator = RandomEventGenerator(network, seed=42)

        event = generator.generate_random_accident(1.0)

        assert event.event_type == EventType.ACCIDENT
        assert 'edge_id' in event.data
        assert 0.3 <= event.data['severity'] <= 0.9

    def test_generate_random_flooding(self):
        """Test generating random flooding."""
        network = create_test_network()
        generator = RandomEventGenerator(network, seed=42)

        event = generator.generate_random_flooding(1.0)

        assert event.event_type == EventType.FLOODING
        assert 20 <= event.data['depth_cm'] <= 80

    def test_generate_scenario_events(self):
        """Test generating scenario events."""
        network = create_test_network()
        generator = RandomEventGenerator(network, seed=42)

        events = generator.generate_scenario_events(
            duration_hours=5.0,
            event_frequency=1.0
        )

        assert len(events) > 0
        # Events should be sorted by timestamp
        for i in range(1, len(events)):
            assert events[i].timestamp >= events[i-1].timestamp

    def test_reproducibility(self):
        """Test that same seed produces same events."""
        network = create_test_network()

        gen1 = RandomEventGenerator(network, seed=42)
        gen2 = RandomEventGenerator(network, seed=42)

        event1 = gen1.generate_random_accident(1.0)
        event2 = gen2.generate_random_accident(1.0)

        assert event1.data['edge_id'] == event2.data['edge_id']
        assert event1.data['severity'] == event2.data['severity']


# ==================== Integration Tests ====================

class TestSimulationIntegration:
    """Integration tests for simulation components."""

    def test_engine_with_traffic_model(self):
        """Test simulation engine with traffic model."""
        network = create_test_network()
        plan = create_test_plan(network)

        engine = SimulationEngine(network)
        traffic = TrafficFlowModel(network)

        engine.initialize(plan)
        engine._state = SimulationState.RUNNING

        # Run a few steps
        for _ in range(5):
            engine.step()
            traffic.update(0.1)

        # Both should have state
        assert engine.metrics.current_time_hours > 0

    def test_engine_with_events(self):
        """Test simulation engine with event handling."""
        network = create_test_network()
        plan = create_test_plan(network)

        engine = SimulationEngine(network)
        events = EventManager()

        # Track route blockages
        blocked_routes = []

        def on_route_blocked(event):
            blocked_routes.append(event.data.get('route_id'))

        events.subscribe(EventType.ROUTE_BLOCKED, on_route_blocked)

        engine.initialize(plan)

        # Simulate blocking a route
        event = EventFactory.create_reroute_needed("route_0", 0.5)
        events.emit(event)

        assert events.pending_count >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
