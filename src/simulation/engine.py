"""
Công cụ Mô phỏng cho các kịch bản sơ tán.

Cung cấp mô phỏng sơ tán theo bước thời gian với:
- Dòng chảy sơ tán tiến triển dọc theo các tuyến đường
- Tiến triển nguy hiểm động
- Theo dõi chỉ số thời gian thực
- Cập nhật theo sự kiện
"""

from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import copy

from ..models.network import EvacuationNetwork
from ..models.node import PopulationZone, Shelter, HazardZone
from ..models.edge import Edge
from ..algorithms.base import EvacuationPlan, EvacuationRoute


class SimulationState(Enum):
    """Các trạng thái có thể có của mô phỏng."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SimulationConfig:
    """Cấu hình cho các tham số mô phỏng."""
    time_step_minutes: float = 1.0  # Bước thời gian mô phỏng (giảm để chạy nhanh hơn)
    max_duration_hours: float = 24.0  # Thời lượng mô phỏng tối đa
    speed_multiplier: float = 1.0  # Hệ số nhân tốc độ thời gian thực

    # Tham số dòng chảy - tăng để mô phỏng nhanh hơn
    flow_rate_per_step: float = 0.5  # Phần trăm dân số còn lại mỗi bước (tăng từ 0.1)
    min_flow_per_route: int = 100  # Số người sơ tán tối thiểu mỗi tuyến đường mỗi bước (tăng từ 10)

    # Tiến triển nguy hiểm
    hazard_expansion_rate: float = 0.01  # km mỗi bước thời gian
    hazard_intensity_growth: float = 0.005  # Mức tăng rủi ro mỗi bước

    # Ảnh hưởng của tắc nghẽn
    congestion_threshold: float = 0.7  # Tỷ lệ dòng chảy/công suất cho tắc nghẽn
    congestion_speed_factor: float = 0.5  # Giảm tốc độ ở mức tắc nghẽn hoàn toàn


@dataclass
class SimulationMetrics:
    """Chỉ số thời gian thực trong quá trình mô phỏng."""
    current_time_hours: float = 0.0
    total_evacuated: int = 0
    total_remaining: int = 0
    active_routes: int = 0
    completed_routes: int = 0
    blocked_routes: int = 0

    # Chỉ số nơi trú ẩn
    shelter_utilization: Dict[str, float] = field(default_factory=dict)
    shelter_arrivals: Dict[str, int] = field(default_factory=dict)

    # Tỷ lệ sức chứa
    total_shelter_capacity: int = 0  # Tổng sức chứa của tất cả nơi trú ẩn
    total_population: int = 0  # Tổng dân số cần sơ tán
    remaining_shelter_capacity: int = 0  # Sức chứa còn lại

    # Chỉ số tuyến đường
    average_travel_time: float = 0.0
    average_risk_exposure: float = 0.0

    # Theo dõi tiến độ
    evacuation_progress: float = 0.0  # 0.0 đến 1.0
    estimated_completion_hours: float = 0.0

    # Lịch sử cho trực quan hóa
    evacuation_history: List[Tuple[float, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang từ điển để tuần tự hóa."""
        return {
            'current_time_hours': self.current_time_hours,
            'total_evacuated': self.total_evacuated,
            'total_remaining': self.total_remaining,
            'active_routes': self.active_routes,
            'completed_routes': self.completed_routes,
            'blocked_routes': self.blocked_routes,
            'evacuation_progress': self.evacuation_progress,
            'estimated_completion_hours': self.estimated_completion_hours,
            'average_travel_time': self.average_travel_time,
            'average_risk_exposure': self.average_risk_exposure,
            'shelter_utilization': self.shelter_utilization,
            'shelter_arrivals': self.shelter_arrivals,
            'total_shelter_capacity': self.total_shelter_capacity,
            'total_population': self.total_population,
            'remaining_shelter_capacity': self.remaining_shelter_capacity
        }


@dataclass
class RouteState:
    """Trạng thái của một tuyến đường sơ tán cá nhân trong quá trình mô phỏng."""
    route: EvacuationRoute
    total_assigned: int  # Tổng số người được phân công cho tuyến đường này
    departed: int = 0  # Số người đã bắt đầu sơ tán
    in_transit: int = 0  # Số người hiện đang trên đường
    arrived: int = 0  # Số người đã đến nơi trú ẩn
    blocked: bool = False  # Tuyến đường bị chặn bởi nguy hiểm

    # Thời gian
    start_time: float = 0.0
    estimated_arrival: float = 0.0

    # Theo dõi người sơ tán đang di chuyển với tiến độ của họ
    evacuee_groups: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        """Số người vẫn đang chờ khởi hành."""
        return max(0, self.total_assigned - self.departed)

    @property
    def is_complete(self) -> bool:
        """Kiểm tra xem tất cả người sơ tán được phân công đã đến nơi chưa."""
        return self.arrived >= self.total_assigned and self.in_transit == 0


# Bí danh kiểu cho callbacks
SimulationCallback = Callable[[SimulationMetrics, Dict[str, RouteState]], None]


class SimulationEngine:
    """
    Công cụ mô phỏng sơ tán theo bước thời gian.

    Mô phỏng quá trình sơ tán theo thời gian với:
    - Di chuyển dân số dần dần dọc theo các tuyến đường
    - Tắc nghẽn động dựa trên dòng chảy
    - Tiến triển vùng nguy hiểm
    - Cập nhật chỉ số thời gian thực
    """

    def __init__(self, network: EvacuationNetwork,
                 config: Optional[SimulationConfig] = None):
        """
        Khởi tạo công cụ mô phỏng.

        Args:
            network: Mạng lưới sơ tán
            config: Cấu hình mô phỏng
        """
        self.network = network
        self.config = config or SimulationConfig()

        # Trạng thái
        self._state = SimulationState.IDLE
        self._current_time = 0.0  # Thời gian mô phỏng tính bằng giờ
        self._metrics = SimulationMetrics()
        self._route_states: Dict[str, RouteState] = {}

        # Callbacks
        self._update_callback: Optional[SimulationCallback] = None
        self._completion_callback: Optional[Callable[[SimulationMetrics], None]] = None

        # Điều khiển
        self._should_stop = False
        self._is_paused = False

        # Ảnh chụp dân số ban đầu
        self._initial_population = 0

    @property
    def state(self) -> SimulationState:
        """Lấy trạng thái mô phỏng hiện tại."""
        return self._state

    @property
    def metrics(self) -> SimulationMetrics:
        """Lấy chỉ số hiện tại."""
        return self._metrics

    @property
    def current_time(self) -> float:
        """Lấy thời gian mô phỏng hiện tại tính bằng giờ."""
        return self._current_time

    def set_update_callback(self, callback: SimulationCallback) -> None:
        """Đặt callback cho cập nhật mô phỏng."""
        self._update_callback = callback

    def set_completion_callback(self,
                                callback: Callable[[SimulationMetrics], None]) -> None:
        """Đặt callback cho hoàn thành mô phỏng."""
        self._completion_callback = callback

    def initialize(self, plan: EvacuationPlan) -> None:
        """
        Khởi tạo mô phỏng với một kế hoạch sơ tán.

        Args:
            plan: Kế hoạch sơ tán để mô phỏng
        """
        self._state = SimulationState.IDLE
        self._current_time = 0.0
        self._should_stop = False
        self._is_paused = False

        # Đặt lại trạng thái mạng lưới
        self.network.reset_simulation_state()

        # Tính toán dân số ban đầu
        self._initial_population = sum(
            z.population for z in self.network.get_population_zones()
        )

        # Khởi tạo trạng thái tuyến đường
        self._route_states.clear()
        for i, route in enumerate(plan.routes):
            route_id = f"route_{i}_{route.zone_id}_{route.shelter_id}"
            self._route_states[route_id] = RouteState(
                route=route,
                total_assigned=route.flow,
                estimated_arrival=route.estimated_time_hours
            )

        # Tính tổng sức chứa nơi trú ẩn
        shelters = list(self.network.get_shelters())
        total_capacity = sum(s.capacity for s in shelters)

        # Khởi tạo chỉ số
        self._metrics = SimulationMetrics(
            total_remaining=self._initial_population,
            active_routes=len(self._route_states),
            total_shelter_capacity=total_capacity,
            total_population=self._initial_population,
            remaining_shelter_capacity=total_capacity
        )

        # Khởi tạo theo dõi mức sử dụng nơi trú ẩn
        for shelter in shelters:
            self._metrics.shelter_utilization[shelter.id] = 0.0
            self._metrics.shelter_arrivals[shelter.id] = 0

    def start(self) -> None:
        """Bắt đầu mô phỏng (chuyển trạng thái sang RUNNING)."""
        self._state = SimulationState.RUNNING

    def run(self, plan: EvacuationPlan,
            real_time: bool = False) -> SimulationMetrics:
        """
        Chạy mô phỏng đến khi hoàn thành.

        Args:
            plan: Kế hoạch sơ tán để mô phỏng
            real_time: Nếu True, chạy theo thời gian thực với độ trễ

        Returns:
            Chỉ số mô phỏng cuối cùng
        """
        self.initialize(plan)
        self._state = SimulationState.RUNNING

        time_step_hours = self.config.time_step_minutes / 60.0
        max_steps = int(self.config.max_duration_hours / time_step_hours)

        for step in range(max_steps):
            if self._should_stop:
                self._state = SimulationState.IDLE
                break

            while self._is_paused:
                time.sleep(0.1)
                if self._should_stop:
                    break

            # Thực hiện bước mô phỏng
            self._step(time_step_hours)

            # Kiểm tra hoàn thành
            if self._is_evacuation_complete():
                self._state = SimulationState.COMPLETED
                break

            # Độ trễ thời gian thực
            if real_time:
                delay = (self.config.time_step_minutes * 60) / self.config.speed_multiplier
                time.sleep(delay)

        # Callback cuối cùng
        if self._completion_callback:
            self._completion_callback(self._metrics)

        return self._metrics

    def step(self) -> SimulationMetrics:
        """
        Thực hiện một bước mô phỏng đơn.

        Returns:
            Chỉ số cập nhật sau bước
        """
        if self._state == SimulationState.IDLE:
            return self._metrics

        time_step_hours = self.config.time_step_minutes / 60.0
        self._step(time_step_hours)

        return self._metrics

    def _step(self, time_step_hours: float) -> None:
        """Thực thi bước nội bộ."""
        self._current_time += time_step_hours

        # 1. Cập nhật vùng nguy hiểm (tiến triển)
        self._update_hazards(time_step_hours)

        # 2. Kiểm tra tuyến đường bị chặn
        self._check_route_blockages()

        # 3. Xử lý khởi hành từ các khu vực
        self._process_departures(time_step_hours)

        # 4. Di chuyển người sơ tán đang trên đường
        self._process_transit(time_step_hours)

        # 5. Xử lý đến nơi tại các nơi trú ẩn
        self._process_arrivals()

        # 6. Cập nhật chỉ số
        self._update_metrics()

        # 7. Callback
        if self._update_callback:
            self._update_callback(self._metrics, self._route_states)

    def _update_hazards(self, time_step_hours: float) -> None:
        """Cập nhật vùng nguy hiểm (mở rộng, cường độ)."""
        for hazard in self.network.get_hazard_zones():
            # Mở rộng bán kính
            hazard.radius_km += self.config.hazard_expansion_rate * (time_step_hours * 60)

            # Tăng cường độ (giới hạn ở 1.0)
            hazard.risk_level = min(1.0,
                hazard.risk_level + self.config.hazard_intensity_growth * (time_step_hours * 60))

        # Cập nhật rủi ro cạnh dựa trên vị trí nguy hiểm mới
        for edge in self.network.get_edges():
            source = self.network.get_node(edge.source_id)
            target = self.network.get_node(edge.target_id)
            if source and target:
                mid_lat = (source.lat + target.lat) / 2
                mid_lon = (source.lon + target.lon) / 2
                risk = self.network.get_total_risk_at(mid_lat, mid_lon)
                edge.set_flood_risk(risk)

    def _check_route_blockages(self) -> None:
        """Kiểm tra xem có tuyến đường nào bị chặn bởi nguy hiểm không."""
        for route_id, state in self._route_states.items():
            if state.blocked:
                continue

            # Kiểm tra từng cạnh trong tuyến đường
            path = state.route.path
            for i in range(len(path) - 1):
                edge = self.network.get_edge_between(path[i], path[i + 1])
                if edge and edge.is_blocked:
                    state.blocked = True
                    break

    def _process_departures(self, time_step_hours: float) -> None:
        """Xử lý người sơ tán khởi hành từ các khu vực."""
        for route_id, state in self._route_states.items():
            if state.blocked or state.remaining <= 0:
                continue

            # Tính toán tỷ lệ khởi hành
            # Càng nhiều người rời đi khi thời gian trôi qua (tính cấp thiết)
            urgency_factor = 1.0 + (self._current_time / 2.0)  # Tăng theo thời gian
            base_rate = self.config.flow_rate_per_step * urgency_factor

            departing = max(
                self.config.min_flow_per_route,
                int(state.remaining * base_rate)
            )
            departing = min(departing, state.remaining)

            if departing > 0:
                # Thêm vào dòng chảy cạnh
                self._add_flow_to_path(state.route.path, departing)

                # Tạo nhóm người sơ tán
                state.evacuee_groups.append({
                    'count': departing,
                    'progress': 0.0,  # 0.0 = ở điểm bắt đầu, 1.0 = ở điểm đến
                    'departure_time': self._current_time
                })

                state.departed += departing
                state.in_transit += departing

    def _process_transit(self, time_step_hours: float) -> None:
        """Di chuyển người sơ tán dọc theo tuyến đường của họ."""
        for route_id, state in self._route_states.items():
            if not state.evacuee_groups:
                continue

            # Tính toán di chuyển dựa trên thời gian di chuyển tuyến đường
            base_travel_time = state.route.estimated_time_hours
            if base_travel_time <= 0:
                base_travel_time = 0.1  # Thời gian di chuyển tối thiểu

            # Áp dụng hệ số tắc nghẽn
            congestion_factor = self._get_route_congestion(state.route.path)
            effective_travel_time = base_travel_time * (1.0 + congestion_factor)

            # Tiến độ mỗi bước thời gian
            progress_per_step = time_step_hours / effective_travel_time

            # Cập nhật từng nhóm
            completed_groups = []
            for group in state.evacuee_groups:
                group['progress'] += progress_per_step
                if group['progress'] >= 1.0:
                    completed_groups.append(group)

            # Xử lý các nhóm đã hoàn thành
            for group in completed_groups:
                state.evacuee_groups.remove(group)
                state.in_transit -= group['count']
                state.arrived += group['count']

                # Xóa dòng chảy khỏi đường dẫn
                self._remove_flow_from_path(state.route.path, group['count'])

    def _process_arrivals(self) -> None:
        """Xử lý người sơ tán đến nơi tại các nơi trú ẩn."""
        for route_id, state in self._route_states.items():
            if state.arrived > self._metrics.shelter_arrivals.get(state.route.shelter_id, 0):
                new_arrivals = state.arrived - self._metrics.shelter_arrivals.get(
                    state.route.shelter_id, 0)

                # Cập nhật nơi trú ẩn
                shelter = self.network.get_node(state.route.shelter_id)
                if isinstance(shelter, Shelter):
                    admitted = shelter.admit(new_arrivals)
                    self._metrics.shelter_arrivals[shelter.id] = \
                        self._metrics.shelter_arrivals.get(shelter.id, 0) + admitted
                    self._metrics.shelter_utilization[shelter.id] = shelter.occupancy_rate

        # Cập nhật sức chứa còn lại
        total_arrivals = sum(self._metrics.shelter_arrivals.values())
        self._metrics.remaining_shelter_capacity = max(
            0, self._metrics.total_shelter_capacity - total_arrivals
        )

    def _add_flow_to_path(self, path: List[str], count: int) -> None:
        """Thêm dòng chảy cho tất cả các cạnh trong một đường dẫn."""
        for i in range(len(path) - 1):
            edge = self.network.get_edge_between(path[i], path[i + 1])
            if edge:
                edge.add_flow(count)

    def _remove_flow_from_path(self, path: List[str], count: int) -> None:
        """Xóa dòng chảy khỏi tất cả các cạnh trong một đường dẫn."""
        for i in range(len(path) - 1):
            edge = self.network.get_edge_between(path[i], path[i + 1])
            if edge:
                edge.remove_flow(count)

    def _get_route_congestion(self, path: List[str]) -> float:
        """Tính toán mức tắc nghẽn trung bình dọc theo tuyến đường."""
        if len(path) < 2:
            return 0.0

        total_congestion = 0.0
        edge_count = 0

        for i in range(len(path) - 1):
            edge = self.network.get_edge_between(path[i], path[i + 1])
            if edge:
                total_congestion += edge.congestion_level
                edge_count += 1

        if edge_count == 0:
            return 0.0

        avg_congestion = total_congestion / edge_count

        # Áp dụng hiệu ứng tắc nghẽn phi tuyến
        if avg_congestion > self.config.congestion_threshold:
            excess = avg_congestion - self.config.congestion_threshold
            return excess * (1.0 / (1.0 - self.config.congestion_threshold))

        return 0.0

    def _update_metrics(self) -> None:
        """Cập nhật chỉ số mô phỏng."""
        total_evacuated = 0
        total_in_transit = 0
        active_routes = 0
        completed_routes = 0
        blocked_routes = 0
        total_travel_time = 0.0
        total_risk = 0.0

        for route_id, state in self._route_states.items():
            total_evacuated += state.arrived
            total_in_transit += state.in_transit

            if state.blocked:
                blocked_routes += 1
            elif state.is_complete:
                completed_routes += 1
            else:
                active_routes += 1

            # Tích lũy chỉ số có trọng số
            if state.arrived > 0:
                total_travel_time += state.route.estimated_time_hours * state.arrived
                total_risk += state.route.risk_score * state.arrived

        self._metrics.current_time_hours = self._current_time
        self._metrics.total_evacuated = total_evacuated
        self._metrics.total_remaining = self._initial_population - total_evacuated - total_in_transit
        self._metrics.active_routes = active_routes
        self._metrics.completed_routes = completed_routes
        self._metrics.blocked_routes = blocked_routes

        # Tính toán trung bình
        if total_evacuated > 0:
            self._metrics.average_travel_time = total_travel_time / total_evacuated
            self._metrics.average_risk_exposure = total_risk / total_evacuated

        # Tiến độ
        if self._initial_population > 0:
            self._metrics.evacuation_progress = total_evacuated / self._initial_population

        # Ước tính thời gian hoàn thành
        # Tính dựa trên người đã khởi hành (không chỉ đã đến) để có ước tính chính xác hơn
        total_departed = sum(state.departed for state in self._route_states.values())

        if total_departed > 0 and self._current_time > 0:
            # Tính tốc độ khởi hành (người/giờ)
            departure_rate = total_departed / self._current_time

            # Tính thời gian di chuyển trung bình của các tuyến đang hoạt động
            avg_travel_time = 0.0
            active_count = 0
            for state in self._route_states.values():
                if not state.is_complete and not state.blocked:
                    avg_travel_time += state.route.estimated_time_hours
                    active_count += 1
            if active_count > 0:
                avg_travel_time /= active_count
            else:
                avg_travel_time = self._metrics.average_travel_time if self._metrics.average_travel_time > 0 else 0.5

            # Số người còn lại cần khởi hành
            remaining_to_depart = self._initial_population - total_departed

            if departure_rate > 0:
                # Thời gian để tất cả khởi hành + thời gian di chuyển trung bình
                time_to_complete_departures = remaining_to_depart / departure_rate
                self._metrics.estimated_completion_hours = \
                    self._current_time + time_to_complete_departures + avg_travel_time
            else:
                self._metrics.estimated_completion_hours = self._current_time + avg_travel_time
        elif self._current_time == 0:
            # Ước tính ban đầu dựa trên thời gian di chuyển trung bình của kế hoạch
            avg_time = 0.0
            count = 0
            for state in self._route_states.values():
                avg_time += state.route.estimated_time_hours
                count += 1
            if count > 0:
                self._metrics.estimated_completion_hours = (avg_time / count) * 2  # x2 for buffer

        # Ghi điểm lịch sử
        self._metrics.evacuation_history.append(
            (self._current_time, total_evacuated)
        )

    def _is_evacuation_complete(self) -> bool:
        """Kiểm tra xem việc sơ tán đã hoàn thành chưa."""
        # Tất cả các tuyến đường đã hoàn thành hoặc bị chặn
        for state in self._route_states.values():
            if not state.is_complete and not state.blocked:
                return False
        return True

    def pause(self) -> None:
        """Tạm dừng mô phỏng."""
        self._is_paused = True
        self._state = SimulationState.PAUSED

    def resume(self) -> None:
        """Tiếp tục mô phỏng."""
        self._is_paused = False
        self._state = SimulationState.RUNNING

    def stop(self) -> None:
        """Dừng mô phỏng."""
        self._should_stop = True
        self._state = SimulationState.IDLE

    def reset(self) -> None:
        """Đặt lại mô phỏng về trạng thái ban đầu."""
        self._state = SimulationState.IDLE
        self._current_time = 0.0
        self._should_stop = False
        self._is_paused = False
        self._route_states.clear()
        self._metrics = SimulationMetrics()
        self.network.reset_simulation_state()

    def get_route_states(self) -> Dict[str, RouteState]:
        """Lấy trạng thái tuyến đường hiện tại."""
        return self._route_states

    def get_evacuation_snapshot(self) -> Dict[str, Any]:
        """Lấy ảnh chụp trạng thái sơ tán hiện tại để trực quan hóa."""
        snapshot = {
            'time': self._current_time,
            'state': self._state.value,
            'metrics': self._metrics.to_dict(),
            'routes': [],
            'shelters': [],
            'hazards': []
        }

        # Dữ liệu tuyến đường
        for route_id, state in self._route_states.items():
            route_data = {
                'id': route_id,
                'zone_id': state.route.zone_id,
                'shelter_id': state.route.shelter_id,
                'path': state.route.path,
                'total': state.total_assigned,
                'departed': state.departed,
                'in_transit': state.in_transit,
                'arrived': state.arrived,
                'blocked': state.blocked,
                'progress': state.arrived / state.total_assigned if state.total_assigned > 0 else 0
            }
            snapshot['routes'].append(route_data)

        # Dữ liệu nơi trú ẩn
        for shelter in self.network.get_shelters():
            shelter_data = {
                'id': shelter.id,
                'name': shelter.name,
                'lat': shelter.lat,
                'lon': shelter.lon,
                'capacity': shelter.capacity,
                'occupancy': shelter.current_occupancy,
                'utilization': shelter.occupancy_rate
            }
            snapshot['shelters'].append(shelter_data)

        # Dữ liệu nguy hiểm
        for i, hazard in enumerate(self.network.get_hazard_zones()):
            hazard_data = {
                'id': f'hazard_{i}',
                'lat': hazard.center_lat,
                'lon': hazard.center_lon,
                'radius_km': hazard.radius_km,
                'risk_level': hazard.risk_level,
                'type': hazard.hazard_type
            }
            snapshot['hazards'].append(hazard_data)

        return snapshot
