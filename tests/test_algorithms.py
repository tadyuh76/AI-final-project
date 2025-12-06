"""
Kiểm thử đơn vị cho module algorithms.
Kiểm thử các thuật toán GBFS, GWO và Comparator.
"""

import pytest
import sys
import os
import numpy as np

# Thêm thư mục cha vào đường dẫn để import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.node import Node, PopulationZone, Shelter, HazardZone
from src.models.edge import Edge, RoadType
from src.models.network import EvacuationNetwork
from src.algorithms.base import (
    AlgorithmType, AlgorithmConfig, EvacuationRoute,
    EvacuationPlan, AlgorithmMetrics
)
from src.algorithms.gbfs import GreedyBestFirstSearch, SearchNode
from src.algorithms.gwo import GreyWolfOptimizer, Wolf
from src.algorithms.comparator import AlgorithmComparator, ComparisonResult


# ==================== Fixture Kiểm Thử ====================

def create_simple_network():
    """Tạo mạng lưới kiểm thử đơn giản với 2 khu dân cư, 2 nơi trú ẩn và đường kết nối."""
    network = EvacuationNetwork()

    # Tạo các giao lộ (bố trí lưới)
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

    # Tạo các khu dân cư
    zone1 = PopulationZone(id="zone1", lat=10.0, lon=106.0, population=1000)
    zone2 = PopulationZone(id="zone2", lat=10.05, lon=106.0, population=1500)
    network.add_node(zone1)
    network.add_node(zone2)

    # Tạo các nơi trú ẩn
    shelter1 = Shelter(id="shelter1", lat=10.0, lon=106.1, capacity=2000)
    shelter2 = Shelter(id="shelter2", lat=10.05, lon=106.1, capacity=1500)
    network.add_node(shelter1)
    network.add_node(shelter2)

    # Kết nối các khu dân cư với lưới
    network.add_edge(Edge(id="ez1", source_id="zone1", target_id="n1", length_km=0.1, is_oneway=False))
    network.add_edge(Edge(id="ez2", source_id="zone2", target_id="n4", length_km=0.1, is_oneway=False))

    # Kết nối các nơi trú ẩn với lưới
    network.add_edge(Edge(id="es1", source_id="n3", target_id="shelter1", length_km=0.1, is_oneway=False))
    network.add_edge(Edge(id="es2", source_id="n6", target_id="shelter2", length_km=0.1, is_oneway=False))

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

    # Add a small hazard near the zones so they need evacuation (risk >= 0.1)
    # This ensures zones are not filtered out by the min_zone_risk_for_evacuation threshold
    hazard = HazardZone(
        center_lat=10.025,  # Between zone1 and zone2
        center_lon=106.0,
        radius_km=5.0,      # Large enough to cover both zones
        risk_level=0.3      # Low risk, just enough to trigger evacuation
    )
    network.add_hazard_zone(hazard)

    return network


def create_network_with_hazard():
    """Tạo mạng lưới có vùng nguy hiểm chặn đường đi trực tiếp."""
    network = create_simple_network()

    # Thêm vùng nguy hiểm giữa n2 và n5
    hazard = HazardZone(
        center_lat=10.025,
        center_lon=106.05,
        radius_km=1.0,
        risk_level=0.9
    )
    network.add_hazard_zone(hazard)

    return network


# ==================== Kiểm Thử Thuật Toán Cơ Bản ====================

class TestAlgorithmConfig:
    """Kiểm thử cho AlgorithmConfig."""

    def test_default_values(self):
        """Kiểm tra các giá trị cấu hình mặc định."""
        config = AlgorithmConfig()
        # Weights matching UI control_panel.py
        assert config.distance_weight == 0.4
        assert config.risk_weight == 0.3
        assert config.congestion_weight == 0.2
        assert config.capacity_weight == 0.1
        assert config.min_flow_threshold == 20  # Lowered from 100
        assert config.n_wolves == 30
        assert config.max_iterations == 100

    def test_custom_values(self):
        """Kiểm tra cấu hình tùy chỉnh."""
        config = AlgorithmConfig(
            distance_weight=0.5,
            n_wolves=50,
            max_iterations=200
        )
        assert config.distance_weight == 0.5
        assert config.n_wolves == 50
        assert config.max_iterations == 200

    def test_to_dict(self):
        """Kiểm tra tuần tự hóa sang từ điển."""
        config = AlgorithmConfig()
        data = config.to_dict()
        assert 'distance_weight' in data
        assert 'n_wolves' in data

    def test_from_dict(self):
        """Kiểm tra giải tuần tự hóa từ từ điển."""
        data = {'distance_weight': 0.6, 'n_wolves': 40}
        config = AlgorithmConfig.from_dict(data)
        assert config.distance_weight == 0.6
        assert config.n_wolves == 40


class TestEvacuationRoute:
    """Kiểm thử cho EvacuationRoute."""

    def test_route_creation(self):
        """Kiểm tra tạo tuyến đường."""
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
    """Kiểm thử cho EvacuationPlan."""

    def test_empty_plan(self):
        """Kiểm tra kế hoạch trống."""
        plan = EvacuationPlan()
        assert len(plan.routes) == 0
        assert plan.total_evacuees == 0

    def test_add_route(self):
        """Kiểm tra thêm tuyến đường vào kế hoạch."""
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
        """Kiểm tra các chỉ số được trọng số hóa đúng theo lưu lượng."""
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

        # Thời gian trung bình: (1.0*1000 + 2.0*1000) / 2000 = 1.5
        assert plan.total_time_hours == 1.5
        # Rủi ro trung bình: (0.2*1000 + 0.4*1000) / 2000 = 0.3
        assert plan.average_risk == 0.3

    def test_get_shelter_loads(self):
        """Kiểm tra tính toán tải nơi trú ẩn."""
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
    """Kiểm thử cho AlgorithmMetrics."""

    def test_metrics_creation(self):
        """Kiểm tra tạo chỉ số."""
        metrics = AlgorithmMetrics(algorithm_type=AlgorithmType.GBFS)
        assert metrics.algorithm_type == AlgorithmType.GBFS
        assert metrics.execution_time_seconds == 0.0

    def test_metrics_to_dict(self):
        """Kiểm tra tuần tự hóa chỉ số."""
        metrics = AlgorithmMetrics(
            algorithm_type=AlgorithmType.GBFS,
            execution_time_seconds=1.5,
            final_cost=100.0
        )
        data = metrics.to_dict()
        assert data['algorithm'] == 'gbfs'
        assert data['execution_time_seconds'] == 1.5


# ==================== Kiểm Thử GBFS ====================

class TestSearchNode:
    """Kiểm thử cho GBFS SearchNode."""

    def test_search_node_creation(self):
        """Kiểm tra tạo node tìm kiếm."""
        node = SearchNode(node_id="n1", h_cost=2.0)
        assert node.node_id == "n1"
        assert node.h_cost == 2.0  # GBFS chỉ dùng h_cost

    def test_search_node_comparison(self):
        """Kiểm tra so sánh node tìm kiếm cho hàng đợi ưu tiên."""
        node1 = SearchNode(node_id="n1", h_cost=2.0)
        node2 = SearchNode(node_id="n2", h_cost=3.0)
        assert node1 < node2  # h_cost thấp hơn nên "nhỏ hơn"


class TestGreedyBestFirstSearch:
    """Kiểm thử cho thuật toán GBFS."""

    def test_algorithm_type(self):
        """Kiểm tra thuộc tính loại thuật toán."""
        network = create_simple_network()
        gbfs = GreedyBestFirstSearch(network)
        assert gbfs.algorithm_type == AlgorithmType.GBFS

    def test_heuristic_calculation(self):
        """Kiểm tra hàm heuristic."""
        network = create_simple_network()
        gbfs = GreedyBestFirstSearch(network)

        node = network.get_node("n1")
        shelter = network.get_shelters()[0]
        # heuristic(node, goal, node_risk, node_congestion)
        h = gbfs.heuristic(node, shelter, 0.0, 0.0)
        assert h >= 0  # Heuristic phải không âm

    def test_find_path_simple(self):
        """Kiểm tra tìm đường đi trong mạng lưới đơn giản."""
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
        """Kiểm tra tìm đường với không có nơi trú ẩn trả về None."""
        network = create_simple_network()
        gbfs = GreedyBestFirstSearch(network)

        zone = network.get_population_zones()[0]
        path, shelter, cost = gbfs.find_path(zone, [])

        assert path is None
        assert shelter is None
        assert cost == float('inf')

    def test_optimize_creates_plan(self):
        """Kiểm tra optimize tạo kế hoạch sơ tán hợp lệ."""
        network = create_simple_network()
        gbfs = GreedyBestFirstSearch(network)

        plan, metrics = gbfs.optimize()

        assert plan is not None
        assert len(plan.routes) > 0
        assert metrics.execution_time_seconds >= 0
        assert metrics.routes_found > 0

    def test_optimize_respects_capacity(self):
        """Kiểm tra tối ưu hóa tôn trọng sức chứa nơi trú ẩn."""
        network = create_simple_network()
        gbfs = GreedyBestFirstSearch(network)

        plan, _ = gbfs.optimize()

        shelter_loads = plan.get_shelter_loads()
        for shelter in network.get_shelters():
            if shelter.id in shelter_loads:
                assert shelter_loads[shelter.id] <= shelter.capacity

    def test_find_multiple_paths(self):
        """Kiểm tra tìm nhiều đường đi."""
        network = create_simple_network()
        gbfs = GreedyBestFirstSearch(network)

        zone = network.get_population_zones()[0]
        shelters = network.get_shelters()

        paths = gbfs.find_multiple_paths(zone, shelters, k=2)
        assert len(paths) <= 2
        if len(paths) == 2:
            assert paths[0][1].id != paths[1][1].id  # Các nơi trú ẩn khác nhau


# ==================== Kiểm Thử GWO ====================

class TestWolf:
    """Kiểm thử cho lớp Wolf của GWO."""

    def test_wolf_creation(self):
        """Kiểm tra tạo sói."""
        position = np.array([[0.5, 0.5], [0.3, 0.7]])
        wolf = Wolf(position=position, fitness=100.0)
        assert wolf.fitness == 100.0
        assert wolf.position.shape == (2, 2)

    def test_wolf_copy(self):
        """Kiểm tra sao chép sói tạo bản sao độc lập."""
        position = np.array([[0.5, 0.5]])
        wolf = Wolf(position=position, fitness=100.0)
        copy = wolf.copy()

        copy.position[0, 0] = 0.9
        copy.fitness = 50.0

        assert wolf.position[0, 0] == 0.5
        assert wolf.fitness == 100.0


class TestGreyWolfOptimizer:
    """Kiểm thử cho thuật toán GWO."""

    def test_algorithm_type(self):
        """Kiểm tra thuộc tính loại thuật toán."""
        network = create_simple_network()
        gwo = GreyWolfOptimizer(network)
        assert gwo.algorithm_type == AlgorithmType.GWO

    def test_initialize_population(self):
        """Kiểm tra khởi tạo quần thể."""
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
        """Kiểm tra vị trí sói được chuẩn hóa (tổng hàng bằng 1)."""
        network = create_simple_network()
        config = AlgorithmConfig(n_wolves=5, max_iterations=5)
        gwo = GreyWolfOptimizer(network, config)

        gwo._initialize_problem()
        gwo._initialize_population()

        for wolf in gwo.wolves:
            row_sums = wolf.position.sum(axis=1)
            np.testing.assert_array_almost_equal(row_sums, np.ones_like(row_sums))

    def test_fitness_calculation(self):
        """Kiểm tra tính toán độ thích nghi."""
        network = create_simple_network()
        gwo = GreyWolfOptimizer(network)

        gwo._initialize_problem()
        position = np.ones((gwo.n_zones, gwo.n_shelters)) / gwo.n_shelters

        fitness = gwo._calculate_fitness(position)
        assert fitness > 0
        assert fitness < float('inf')

    def test_optimize_creates_plan(self):
        """Kiểm tra optimize tạo kế hoạch hợp lệ."""
        network = create_simple_network()
        config = AlgorithmConfig(n_wolves=10, max_iterations=10)
        gwo = GreyWolfOptimizer(network, config)

        plan, metrics = gwo.optimize()

        assert plan is not None
        assert metrics.iterations > 0
        assert len(metrics.convergence_history) > 0

    def test_convergence_improves(self):
        """Kiểm tra độ thích nghi thường cải thiện qua các vòng lặp."""
        network = create_simple_network()
        config = AlgorithmConfig(n_wolves=20, max_iterations=30)
        gwo = GreyWolfOptimizer(network, config)

        _, metrics = gwo.optimize()

        # Độ thích nghi đầu nên >= cuối (cải thiện)
        if len(metrics.convergence_history) > 5:
            first_avg = np.mean(metrics.convergence_history[:5])
            last_avg = np.mean(metrics.convergence_history[-5:])
            # Cho phép một số biến động, nhưng nên cải thiện
            assert last_avg <= first_avg * 1.5

    def test_get_flow_matrix(self):
        """Kiểm tra lấy ma trận lưu lượng tối ưu."""
        network = create_simple_network()
        config = AlgorithmConfig(n_wolves=5, max_iterations=5)
        gwo = GreyWolfOptimizer(network, config)

        gwo.optimize()
        flow_matrix = gwo.get_flow_matrix()

        assert flow_matrix is not None
        assert flow_matrix.shape == (len(network.get_population_zones()),
                                     len(network.get_shelters()))


# ==================== Kiểm Thử Comparator ====================

class TestComparisonResult:
    """Kiểm thử cho ComparisonResult."""

    def test_get_metric_comparison(self):
        """Kiểm tra lấy so sánh chỉ số giữa các thuật toán."""
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
        """Kiểm tra tuần tự hóa sang từ điển."""
        result = ComparisonResult(
            algorithms=[AlgorithmType.GBFS, AlgorithmType.GWO],
            winner=AlgorithmType.GBFS,
            winner_score=0.85
        )
        data = result.to_dict()
        assert 'gbfs' in data['algorithms']
        assert data['winner'] == 'gbfs'


class TestAlgorithmComparator:
    """Kiểm thử cho AlgorithmComparator."""

    def test_compare_single_algorithm(self):
        """Kiểm tra so sánh một thuật toán đơn."""
        network = create_simple_network()
        comparator = AlgorithmComparator(network)

        result = comparator.compare([AlgorithmType.GBFS])

        assert len(result.algorithms) == 1
        assert AlgorithmType.GBFS in result.metrics

    def test_compare_all(self):
        """Kiểm tra so sánh tất cả thuật toán."""
        network = create_simple_network()
        config = AlgorithmConfig(
            n_wolves=5,
            max_iterations=5
        )
        comparator = AlgorithmComparator(network, config)

        result = comparator.compare_all()

        assert len(result.algorithms) == 2
        assert AlgorithmType.GBFS in result.metrics
        assert AlgorithmType.GWO in result.metrics
        assert result.winner is not None

    def test_rankings_calculated(self):
        """Kiểm tra xếp hạng được tính toán."""
        network = create_simple_network()
        config = AlgorithmConfig(n_wolves=5, max_iterations=5)
        comparator = AlgorithmComparator(network, config)

        result = comparator.compare([AlgorithmType.GBFS, AlgorithmType.GWO])

        assert len(result.rankings) > 0
        assert 'execution_time_seconds' in result.rankings

    def test_generate_comparison_table(self):
        """Kiểm tra tạo bảng so sánh."""
        network = create_simple_network()
        config = AlgorithmConfig(n_wolves=5, max_iterations=5)
        comparator = AlgorithmComparator(network, config)

        result = comparator.compare([AlgorithmType.GBFS, AlgorithmType.GWO])
        table = comparator.generate_comparison_table(result)

        assert 'KẾT QUẢ SO SÁNH THUẬT TOÁN' in table
        assert 'GBFS' in table
        assert 'GWO' in table


# ==================== Kiểm Thử Tích Hợp ====================

class TestAlgorithmIntegration:
    """Kiểm thử tích hợp cho các thuật toán hoạt động cùng nhau."""

    def test_all_algorithms_produce_valid_plans(self):
        """Kiểm tra tất cả thuật toán tạo kế hoạch sơ tán hợp lệ."""
        network = create_simple_network()
        config = AlgorithmConfig(
            n_wolves=10,
            max_iterations=10
        )

        algorithms = [
            GreedyBestFirstSearch(network, config),
            GreyWolfOptimizer(network, config)
        ]

        for algo in algorithms:
            plan, metrics = algo.optimize()

            # Mỗi thuật toán nên tạo tuyến đường
            assert plan is not None, f"{algo.algorithm_type} thất bại trong việc tạo kế hoạch"

            # Tổng số người sơ tán phải dương
            assert plan.total_evacuees >= 0

            # Chỉ số nên được ghi lại
            assert metrics.execution_time_seconds >= 0

    def test_algorithms_handle_hazards(self):
        """Kiểm tra thuật toán có thể xử lý vùng nguy hiểm."""
        network = create_network_with_hazard()
        config = AlgorithmConfig(n_wolves=10, max_iterations=10)

        gbfs = GreedyBestFirstSearch(network, config)
        plan, _ = gbfs.optimize()

        # Vẫn nên tìm được đường đi ngay cả có vùng nguy hiểm
        assert len(plan.routes) > 0

    def test_stop_functionality(self):
        """Kiểm tra thuật toán có thể bị dừng."""
        network = create_simple_network()
        config = AlgorithmConfig(n_wolves=50, max_iterations=1000)

        gwo = GreyWolfOptimizer(network, config)

        # Thiết lập callback để dừng sau 5 vòng lặp
        iteration_count = [0]

        def callback(iteration, cost, data):
            iteration_count[0] = iteration
            if iteration >= 5:
                gwo.stop()

        gwo.set_progress_callback(callback)
        gwo.optimize()

        # Nên dừng sớm
        assert iteration_count[0] < 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
