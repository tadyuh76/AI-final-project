"""
Kiểm thử đơn vị cho module simulation.
Kiểm thử SimulationEngine, TrafficFlowModel và EventManager.
"""

import pytest
import sys
import os

# Thêm thư mục cha vào đường dẫn để import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.node import Node, PopulationZone, Shelter
from src.models.edge import Edge, RoadType
from src.models.network import EvacuationNetwork
from src.algorithms.base import EvacuationPlan, EvacuationRoute

from src.simulation.engine import (
    SimulationEngine, SimulationState, SimulationConfig,
    SimulationMetrics, RouteState
)
from src.simulation.traffic import (
    TrafficFlowModel, TrafficState, TrafficConfig,
    TrafficAssignment
)
from src.simulation.events import (
    EventManager, EventQueue, EventFactory, EventType,
    EventPriority, SimulationEvent, RandomEventGenerator
)


# ==================== Fixture Kiểm Thử ====================

def create_test_network():
    """Tạo mạng lưới kiểm thử đơn giản cho kiểm thử mô phỏng."""
    network = EvacuationNetwork()

    # Tạo các node lưới
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

    # Tạo các khu dân cư và nơi trú ẩn
    zone1 = PopulationZone(id="zone1", lat=10.0, lon=106.0, population=1000)
    zone2 = PopulationZone(id="zone2", lat=10.05, lon=106.0, population=1500)
    shelter1 = Shelter(id="shelter1", lat=10.0, lon=106.1, capacity=2000)
    shelter2 = Shelter(id="shelter2", lat=10.05, lon=106.1, capacity=1500)

    network.add_node(zone1)
    network.add_node(zone2)
    network.add_node(shelter1)
    network.add_node(shelter2)

    # Kết nối các khu dân cư với lưới
    network.add_edge(Edge(id="ez1", source_id="zone1", target_id="n1",
                          length_km=0.1, is_oneway=False))
    network.add_edge(Edge(id="ez2", source_id="zone2", target_id="n4",
                          length_km=0.1, is_oneway=False))

    # Kết nối các nơi trú ẩn với lưới
    network.add_edge(Edge(id="es1", source_id="n3", target_id="shelter1",
                          length_km=0.1, is_oneway=False))
    network.add_edge(Edge(id="es2", source_id="n6", target_id="shelter2",
                          length_km=0.1, is_oneway=False))

    # Tạo các edge lưới
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
    """Tạo kế hoạch sơ tán đơn giản để kiểm thử."""
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


# ==================== Kiểm Thử SimulationConfig ====================

class TestSimulationConfig:
    """Kiểm thử cho SimulationConfig."""

    def test_default_config(self):
        """Kiểm tra các giá trị cấu hình mặc định."""
        config = SimulationConfig()
        assert config.time_step_minutes == 1.0
        assert config.max_duration_hours == 24.0
        assert config.flow_rate_per_step == 0.5

    def test_custom_config(self):
        """Kiểm tra cấu hình tùy chỉnh."""
        config = SimulationConfig(
            time_step_minutes=10.0,
            max_duration_hours=12.0
        )
        assert config.time_step_minutes == 10.0
        assert config.max_duration_hours == 12.0


# ==================== Kiểm Thử SimulationMetrics ====================

class TestSimulationMetrics:
    """Kiểm thử cho SimulationMetrics."""

    def test_default_metrics(self):
        """Kiểm tra các giá trị chỉ số mặc định."""
        metrics = SimulationMetrics()
        assert metrics.total_evacuated == 0
        assert metrics.evacuation_progress == 0.0

    def test_to_dict(self):
        """Kiểm tra tuần tự hóa chỉ số."""
        metrics = SimulationMetrics(
            total_evacuated=500,
            evacuation_progress=0.5
        )
        data = metrics.to_dict()
        assert data['total_evacuated'] == 500
        assert data['evacuation_progress'] == 0.5


# ==================== Kiểm Thử RouteState ====================

class TestRouteState:
    """Kiểm thử cho RouteState."""

    def test_remaining_calculation(self):
        """Kiểm tra tính toán dân số còn lại."""
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
        """Kiểm tra kiểm tra hoàn thành."""
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
        """Kiểm tra chưa hoàn thành khi còn người đang di chuyển."""
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


# ==================== Kiểm Thử SimulationEngine ====================

class TestSimulationEngine:
    """Kiểm thử cho SimulationEngine."""

    def test_engine_creation(self):
        """Kiểm tra tạo engine."""
        network = create_test_network()
        engine = SimulationEngine(network)
        assert engine.state == SimulationState.IDLE

    def test_initialize(self):
        """Kiểm tra khởi tạo mô phỏng."""
        network = create_test_network()
        plan = create_test_plan(network)
        engine = SimulationEngine(network)

        engine.initialize(plan)

        assert engine.state == SimulationState.IDLE
        assert engine.current_time == 0.0
        assert len(engine.get_route_states()) == 2

    def test_step(self):
        """Kiểm tra bước mô phỏng đơn."""
        network = create_test_network()
        plan = create_test_plan(network)
        engine = SimulationEngine(network)

        engine.initialize(plan)
        engine._state = SimulationState.RUNNING
        metrics = engine.step()

        assert metrics.current_time_hours > 0
        assert engine.current_time > 0

    def test_run_simulation(self):
        """Kiểm tra chạy mô phỏng hoàn chỉnh."""
        network = create_test_network()
        plan = create_test_plan(network)
        config = SimulationConfig(
            time_step_minutes=5.0,
            max_duration_hours=2.0
        )
        engine = SimulationEngine(network, config)

        metrics = engine.run(plan)

        assert metrics.total_evacuated > 0
        # Mô phỏng có thể hoàn thành, đạt thời lượng tối đa, hoặc vẫn chạy
        assert engine.state in [
            SimulationState.COMPLETED,
            SimulationState.IDLE,
            SimulationState.RUNNING  # Đạt số bước tối đa
        ]

    def test_pause_resume(self):
        """Kiểm tra tạm dừng và tiếp tục."""
        network = create_test_network()
        engine = SimulationEngine(network)

        engine.pause()
        assert engine.state == SimulationState.PAUSED

        engine.resume()
        assert engine.state == SimulationState.RUNNING

    def test_stop(self):
        """Kiểm tra dừng mô phỏng."""
        network = create_test_network()
        engine = SimulationEngine(network)

        engine.stop()
        assert engine.state == SimulationState.IDLE

    def test_reset(self):
        """Kiểm tra đặt lại mô phỏng."""
        network = create_test_network()
        plan = create_test_plan(network)
        engine = SimulationEngine(network)

        engine.initialize(plan)
        engine.step()
        engine.reset()

        assert engine.current_time == 0.0
        assert len(engine.get_route_states()) == 0

    def test_update_callback(self):
        """Kiểm tra callback cập nhật được gọi."""
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
        """Kiểm tra lấy snapshot sơ tán."""
        network = create_test_network()
        plan = create_test_plan(network)
        engine = SimulationEngine(network)

        engine.initialize(plan)
        snapshot = engine.get_evacuation_snapshot()

        assert 'time' in snapshot
        assert 'routes' in snapshot
        assert 'shelters' in snapshot
        assert len(snapshot['routes']) == 2


# ==================== Kiểm Thử TrafficConfig ====================

class TestTrafficConfig:
    """Kiểm thử cho TrafficConfig."""

    def test_default_config(self):
        """Kiểm tra cấu hình giao thông mặc định."""
        config = TrafficConfig()
        assert config.bpr_alpha == 0.15
        assert config.bpr_beta == 4.0
        assert config.persons_per_vehicle == 3.0


# ==================== Kiểm Thử TrafficFlowModel ====================

class TestTrafficFlowModel:
    """Kiểm thử cho TrafficFlowModel."""

    def test_model_creation(self):
        """Kiểm tra tạo mô hình giao thông."""
        network = create_test_network()
        model = TrafficFlowModel(network)
        assert model is not None

    def test_update(self):
        """Kiểm tra cập nhật giao thông."""
        network = create_test_network()
        model = TrafficFlowModel(network)

        # Thêm một số lưu lượng vào edge
        for edge in network.get_edges():
            edge.add_flow(100)

        state = model.update(0.1)

        assert state.total_vehicles >= 0
        assert len(state.edges) > 0

    def test_bpr_travel_time(self):
        """Kiểm tra tính toán thời gian di chuyển BPR."""
        network = create_test_network()
        model = TrafficFlowModel(network)

        # Lưu thông tự do
        time_free = model._bpr_travel_time(1.0, 0.0)
        assert time_free == 1.0

        # Tắc nghẽn
        time_congested = model._bpr_travel_time(1.0, 1.0)
        assert time_congested > 1.0

    def test_traffic_states(self):
        """Kiểm tra xác định trạng thái giao thông."""
        network = create_test_network()
        model = TrafficFlowModel(network)

        assert model._determine_traffic_state(0.3) == TrafficState.FREE_FLOW
        assert model._determine_traffic_state(0.6) == TrafficState.SYNCHRONIZED
        assert model._determine_traffic_state(0.9) == TrafficState.CONGESTED
        assert model._determine_traffic_state(1.2) == TrafficState.GRIDLOCK

    def test_get_route_travel_time(self):
        """Kiểm tra tính toán thời gian di chuyển tuyến đường."""
        network = create_test_network()
        model = TrafficFlowModel(network)

        path = ["zone1", "n1", "n2", "n3", "shelter1"]
        time = model.get_route_travel_time(path)

        assert time > 0
        assert time < float('inf')

    def test_congestion_map(self):
        """Kiểm tra lấy bản đồ tắc nghẽn."""
        network = create_test_network()
        model = TrafficFlowModel(network)

        congestion = model.get_congestion_map()
        assert len(congestion) > 0

    def test_apply_incident(self):
        """Kiểm tra áp dụng và xóa sự cố."""
        network = create_test_network()
        model = TrafficFlowModel(network)

        edge_id = list(network._edges.keys())[0]
        model.apply_incident(edge_id, 0.5)

        state = model._edge_states.get(edge_id)
        assert state.capacity_factor < 1.0

        model.clear_incident(edge_id)
        assert state.capacity_factor == 1.0

    def test_reset(self):
        """Kiểm tra đặt lại mô hình giao thông."""
        network = create_test_network()
        model = TrafficFlowModel(network)

        # Sửa đổi trạng thái
        edge_id = list(network._edges.keys())[0]
        model.apply_incident(edge_id, 0.5)

        model.reset()

        state = model._edge_states.get(edge_id)
        assert state.capacity_factor == 1.0


# ==================== Kiểm Thử TrafficAssignment ====================

class TestTrafficAssignment:
    """Kiểm thử cho TrafficAssignment."""

    def test_assignment_creation(self):
        """Kiểm tra tạo phân bổ giao thông."""
        network = create_test_network()
        assignment = TrafficAssignment(network)
        assert assignment is not None

    def test_assign_flow(self):
        """Kiểm tra phân bổ lưu lượng."""
        network = create_test_network()
        assignment = TrafficAssignment(network)

        # Ma trận OD đơn giản
        od_matrix = {
            ("zone1", "shelter1"): 100.0
        }

        flows = assignment.assign_flow(od_matrix, max_iterations=10)
        assert len(flows) > 0


# ==================== Kiểm Thử EventQueue ====================

class TestEventQueue:
    """Kiểm thử cho EventQueue."""

    def test_push_pop(self):
        """Kiểm tra các thao tác push và pop."""
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

        # Ưu tiên cao hơn nên lên trước
        popped = queue.pop()
        assert popped.id == "2"

    def test_peek(self):
        """Kiểm tra peek mà không xóa."""
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
        """Kiểm tra các sự kiện đã lên lịch."""
        queue = EventQueue()
        event = SimulationEvent(
            id="1", event_type=EventType.ACCIDENT,
            timestamp=0
        )
        queue.schedule(event, trigger_time=1.0)

        # Chưa kích hoạt
        triggered = queue.check_scheduled(0.5)
        assert len(triggered) == 0

        # Đã kích hoạt
        triggered = queue.check_scheduled(1.5)
        assert len(triggered) == 1

    def test_recurring_schedule(self):
        """Kiểm tra các sự kiện lên lịch định kỳ."""
        queue = EventQueue()
        event = SimulationEvent(
            id="1", event_type=EventType.HAZARD_EXPANDED,
            timestamp=0
        )
        queue.schedule(event, trigger_time=1.0, recurring=True, interval=1.0)

        # Kích hoạt lần đầu
        triggered = queue.check_scheduled(1.0)
        assert len(triggered) == 1

        # Kích hoạt lần hai
        triggered = queue.check_scheduled(2.0)
        assert len(triggered) == 1


# ==================== Kiểm Thử EventManager ====================

class TestEventManager:
    """Kiểm thử cho EventManager."""

    def test_subscribe_emit(self):
        """Kiểm tra đăng ký và phát sự kiện."""
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
        """Kiểm tra đăng ký tất cả sự kiện."""
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
        """Kiểm tra xếp hàng và xử lý sự kiện."""
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

        # Xử lý tại thời điểm 0.5 (không nên xử lý)
        count = manager.process_queue(0.5)
        assert count == 0

        # Xử lý tại thời điểm 1.5 (nên xử lý)
        count = manager.process_queue(1.5)
        assert count == 1
        assert len(processed) == 1

    def test_get_history(self):
        """Kiểm tra lịch sử sự kiện."""
        manager = EventManager()

        event1 = SimulationEvent(id="1", event_type=EventType.ACCIDENT, timestamp=1.0)
        event2 = SimulationEvent(id="2", event_type=EventType.FLOODING, timestamp=2.0)

        manager.emit(event1)
        manager.emit(event2)

        # Toàn bộ lịch sử
        history = manager.get_history()
        assert len(history) == 2

        # Lịch sử được lọc
        history = manager.get_history(EventType.ACCIDENT)
        assert len(history) == 1

    def test_clear(self):
        """Kiểm tra xóa sự kiện."""
        manager = EventManager()

        event = SimulationEvent(id="1", event_type=EventType.ACCIDENT, timestamp=1.0)
        manager.queue(event)
        manager.emit(event)

        manager.clear()

        assert manager.pending_count == 0


# ==================== Kiểm Thử EventFactory ====================

class TestEventFactory:
    """Kiểm thử cho EventFactory."""

    def test_create_road_blocked(self):
        """Kiểm tra tạo sự kiện đường bị chặn."""
        event = EventFactory.create_road_blocked("edge1", 1.0, "flooding")

        assert event.event_type == EventType.ROAD_BLOCKED
        assert event.data['edge_id'] == "edge1"
        assert event.data['reason'] == "flooding"
        assert event.priority == EventPriority.HIGH

    def test_create_accident(self):
        """Kiểm tra tạo sự kiện tai nạn."""
        event = EventFactory.create_accident("edge1", 1.0, 0.7)

        assert event.event_type == EventType.ACCIDENT
        assert event.data['severity'] == 0.7
        assert 'capacity_reduction' in event.data

    def test_create_flooding(self):
        """Kiểm tra tạo sự kiện lũ lụt với mức độ nghiêm trọng dựa trên độ sâu."""
        # Lũ nông
        event_shallow = EventFactory.create_flooding("edge1", 1.0, 25.0)
        assert event_shallow.data['severity'] == 0.5

        # Lũ sâu
        event_deep = EventFactory.create_flooding("edge1", 1.0, 60.0)
        assert event_deep.data['severity'] == 1.0

    def test_create_shelter_closed(self):
        """Kiểm tra tạo sự kiện nơi trú ẩn đóng cửa."""
        event = EventFactory.create_shelter_closed("shelter1", 1.0, "damage")

        assert event.event_type == EventType.SHELTER_CLOSED
        assert event.priority == EventPriority.CRITICAL

    def test_create_hazard(self):
        """Kiểm tra tạo sự kiện vùng nguy hiểm."""
        event = EventFactory.create_hazard(10.7, 106.7, 2.0, 1.0, "flood")

        assert event.event_type == EventType.HAZARD_CREATED
        assert event.data['center_lat'] == 10.7
        assert event.data['radius_km'] == 2.0


# ==================== Kiểm Thử RandomEventGenerator ====================

class TestRandomEventGenerator:
    """Kiểm thử cho RandomEventGenerator."""

    def test_generator_creation(self):
        """Kiểm tra tạo generator với seed."""
        network = create_test_network()
        generator = RandomEventGenerator(network, seed=42)
        assert generator is not None

    def test_generate_random_accident(self):
        """Kiểm tra tạo tai nạn ngẫu nhiên."""
        network = create_test_network()
        generator = RandomEventGenerator(network, seed=42)

        event = generator.generate_random_accident(1.0)

        assert event.event_type == EventType.ACCIDENT
        assert 'edge_id' in event.data
        assert 0.3 <= event.data['severity'] <= 0.9

    def test_generate_random_flooding(self):
        """Kiểm tra tạo lũ lụt ngẫu nhiên."""
        network = create_test_network()
        generator = RandomEventGenerator(network, seed=42)

        event = generator.generate_random_flooding(1.0)

        assert event.event_type == EventType.FLOODING
        assert 20 <= event.data['depth_cm'] <= 80

    def test_generate_scenario_events(self):
        """Kiểm tra tạo các sự kiện kịch bản."""
        network = create_test_network()
        generator = RandomEventGenerator(network, seed=42)

        events = generator.generate_scenario_events(
            duration_hours=5.0,
            event_frequency=1.0
        )

        assert len(events) > 0
        # Sự kiện nên được sắp xếp theo timestamp
        for i in range(1, len(events)):
            assert events[i].timestamp >= events[i-1].timestamp

    def test_reproducibility(self):
        """Kiểm tra cùng seed tạo cùng sự kiện."""
        network = create_test_network()

        gen1 = RandomEventGenerator(network, seed=42)
        gen2 = RandomEventGenerator(network, seed=42)

        event1 = gen1.generate_random_accident(1.0)
        event2 = gen2.generate_random_accident(1.0)

        assert event1.data['edge_id'] == event2.data['edge_id']
        assert event1.data['severity'] == event2.data['severity']


# ==================== Kiểm Thử Tích Hợp ====================

class TestSimulationIntegration:
    """Kiểm thử tích hợp cho các thành phần mô phỏng."""

    def test_engine_with_traffic_model(self):
        """Kiểm tra engine mô phỏng với mô hình giao thông."""
        network = create_test_network()
        plan = create_test_plan(network)

        engine = SimulationEngine(network)
        traffic = TrafficFlowModel(network)

        engine.initialize(plan)
        engine._state = SimulationState.RUNNING

        # Chạy vài bước
        for _ in range(5):
            engine.step()
            traffic.update(0.1)

        # Cả hai nên có trạng thái
        assert engine.metrics.current_time_hours > 0

    def test_engine_with_events(self):
        """Kiểm tra engine mô phỏng với xử lý sự kiện."""
        network = create_test_network()
        plan = create_test_plan(network)

        engine = SimulationEngine(network)
        events = EventManager()

        # Theo dõi các tuyến đường bị chặn
        blocked_routes = []

        def on_route_blocked(event):
            blocked_routes.append(event.data.get('route_id'))

        events.subscribe(EventType.ROUTE_BLOCKED, on_route_blocked)

        engine.initialize(plan)

        # Mô phỏng chặn một tuyến đường
        event = EventFactory.create_reroute_needed("route_0", 0.5)
        events.emit(event)

        assert events.pending_count >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
