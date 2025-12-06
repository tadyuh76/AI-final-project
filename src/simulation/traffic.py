"""
Mô hình Dòng chảy Giao thông cho mô phỏng sơ tán.

Triển khai động lực dòng chảy giao thông bao gồm:
- Mối quan hệ tốc độ-dòng chảy BPR (Bureau of Public Roads)
- Hình thành và tiêu tán hàng đợi
- Độ trễ giao lộ
- Giảm công suất động từ các mối nguy hiểm
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..models.network import EvacuationNetwork
from ..models.edge import Edge


class TrafficState(Enum):
    """Các trạng thái dòng chảy giao thông."""
    FREE_FLOW = "free_flow"
    SYNCHRONIZED = "synchronized"
    CONGESTED = "congested"
    GRIDLOCK = "gridlock"


@dataclass
class TrafficConfig:
    """Cấu hình cho các tham số mô hình giao thông."""
    # Tham số hàm BPR
    bpr_alpha: float = 0.15  # Tham số alpha BPR
    bpr_beta: float = 4.0  # Tham số beta BPR (lũy thừa)

    # Hệ số công suất
    hazard_capacity_reduction: float = 0.5  # Giảm công suất trong vùng nguy hiểm
    rain_capacity_reduction: float = 0.2  # Giảm công suất khi mưa
    night_capacity_reduction: float = 0.15  # Giảm công suất vào ban đêm

    # Tham số hàng đợi
    queue_discharge_rate: float = 1800  # Xe mỗi giờ mỗi làn
    max_queue_length: int = 500  # Độ dài hàng đợi tối đa trước khi tràn ngược

    # Độ trễ giao lộ (giờ)
    signalized_delay: float = 0.01  # ~36 giây trung bình
    unsignalized_delay: float = 0.005  # ~18 giây trung bình

    # Chuyển đổi người sang phương tiện
    persons_per_vehicle: float = 3.0  # Mức độ sử dụng trung bình trong quá trình sơ tán


@dataclass
class EdgeTrafficState:
    """Trạng thái giao thông cho một cạnh cá nhân."""
    edge_id: str
    current_flow: int = 0  # Dòng chảy hiện tại (người)
    queue_length: int = 0  # Hàng đợi chờ ở lối vào
    travel_time: float = 0.0  # Thời gian di chuyển hiện tại (giờ)
    speed: float = 0.0  # Tốc độ hiện tại (km/h)
    density: float = 0.0  # Xe mỗi km
    state: TrafficState = TrafficState.FREE_FLOW
    capacity_factor: float = 1.0  # Hệ số nhân công suất hiệu quả


@dataclass
class NetworkTrafficState:
    """Trạng thái giao thông tổng hợp cho mạng lưới."""
    total_vehicles: int = 0
    total_in_queue: int = 0
    average_speed: float = 0.0
    congested_edges: int = 0
    gridlocked_edges: int = 0
    edges: Dict[str, EdgeTrafficState] = field(default_factory=dict)


class TrafficFlowModel:
    """
    Mô hình mô phỏng dòng chảy giao thông.

    Triển khai dòng chảy giao thông vĩ mô sử dụng:
    - Hàm BPR cho mối quan hệ tốc độ-dòng chảy
    - Mô hình sóng động học bậc nhất cho lan truyền
    - Mô hình hóa giao lộ dựa trên hàng đợi
    """

    def __init__(self, network: EvacuationNetwork,
                 config: Optional[TrafficConfig] = None):
        """
        Khởi tạo mô hình giao thông.

        Args:
            network: Mạng lưới sơ tán
            config: Cấu hình giao thông
        """
        self.network = network
        self.config = config or TrafficConfig()

        # Trạng thái giao thông mỗi cạnh
        self._edge_states: Dict[str, EdgeTrafficState] = {}

        # Khởi tạo trạng thái cạnh
        self._initialize_edge_states()

    def _initialize_edge_states(self) -> None:
        """Khởi tạo trạng thái giao thông cho tất cả các cạnh."""
        for edge in self.network.get_edges():
            self._edge_states[edge.id] = EdgeTrafficState(
                edge_id=edge.id,
                speed=edge.max_speed_kmh,
                travel_time=edge.base_travel_time
            )

    def reset(self) -> None:
        """Đặt lại trạng thái giao thông."""
        self._initialize_edge_states()

    def update(self, time_step_hours: float) -> NetworkTrafficState:
        """
        Cập nhật trạng thái giao thông cho một bước thời gian.

        Args:
            time_step_hours: Thời lượng bước thời gian tính bằng giờ

        Returns:
            Trạng thái giao thông mạng lưới đã cập nhật
        """
        # Cập nhật từng cạnh
        for edge in self.network.get_edges():
            state = self._edge_states.get(edge.id)
            if state:
                self._update_edge_state(edge, state, time_step_hours)

        # Xử lý hàng đợi và tràn ngược
        self._process_queues(time_step_hours)

        # Tính toán chỉ số toàn mạng lưới
        return self._calculate_network_state()

    def _update_edge_state(self, edge: Edge, state: EdgeTrafficState,
                           time_step_hours: float) -> None:
        """Cập nhật trạng thái giao thông cho một cạnh đơn."""
        # Lấy dòng chảy hiện tại từ cạnh
        state.current_flow = edge.current_flow

        # Tính toán công suất hiệu quả
        base_capacity = edge.capacity
        capacity_factor = self._calculate_capacity_factor(edge)
        state.capacity_factor = capacity_factor
        effective_capacity = base_capacity * capacity_factor

        # Chuyển đổi người sang phương tiện
        vehicle_flow = state.current_flow / self.config.persons_per_vehicle

        # Tính toán tỷ lệ lưu lượng trên công suất
        if effective_capacity > 0:
            vc_ratio = vehicle_flow / effective_capacity
        else:
            vc_ratio = 1.0

        # Áp dụng hàm BPR cho thời gian di chuyển
        base_time = edge.base_travel_time
        state.travel_time = self._bpr_travel_time(base_time, vc_ratio)

        # Tính toán tốc độ
        if state.travel_time > 0:
            state.speed = edge.length_km / state.travel_time
        else:
            state.speed = edge.max_speed_kmh

        # Tính toán mật độ (xe mỗi km)
        if edge.length_km > 0:
            state.density = vehicle_flow / (edge.length_km * edge.lanes)
        else:
            state.density = 0

        # Xác định trạng thái giao thông
        state.state = self._determine_traffic_state(vc_ratio)

    def _bpr_travel_time(self, free_flow_time: float,
                         vc_ratio: float) -> float:
        """
        Tính toán thời gian di chuyển sử dụng hàm BPR.

        BPR: t = t0 * (1 + alpha * (v/c)^beta)

        Args:
            free_flow_time: Thời gian di chuyển dòng chảy tự do
            vc_ratio: Tỷ lệ lưu lượng trên công suất

        Returns:
            Thời gian di chuyển có tắc nghẽn
        """
        return free_flow_time * (
            1 + self.config.bpr_alpha * (vc_ratio ** self.config.bpr_beta)
        )

    def _calculate_capacity_factor(self, edge: Edge) -> float:
        """Tính toán hệ số công suất hiệu quả dựa trên điều kiện."""
        factor = 1.0

        # Giảm công suất trong vùng nguy hiểm
        if edge.flood_risk > 0:
            factor *= (1 - edge.flood_risk * self.config.hazard_capacity_reduction)

        # Các cạnh bị chặn có công suất bằng không
        if edge.is_blocked:
            return 0.0

        return max(0.1, factor)  # Công suất tối thiểu 10%

    def _determine_traffic_state(self, vc_ratio: float) -> TrafficState:
        """Xác định trạng thái giao thông dựa trên tỷ lệ V/C."""
        if vc_ratio < 0.5:
            return TrafficState.FREE_FLOW
        elif vc_ratio < 0.8:
            return TrafficState.SYNCHRONIZED
        elif vc_ratio < 1.0:
            return TrafficState.CONGESTED
        else:
            return TrafficState.GRIDLOCK

    def _process_queues(self, time_step_hours: float) -> None:
        """Xử lý hình thành và tiêu tán hàng đợi."""
        for edge_id, state in self._edge_states.items():
            edge = self.network.get_edge(edge_id)
            if not edge:
                continue

            # Hàng đợi hình thành khi nhu cầu vượt quá công suất
            effective_capacity = edge.capacity * state.capacity_factor
            demand = edge.current_flow / self.config.persons_per_vehicle

            if demand > effective_capacity:
                # Tăng trưởng hàng đợi
                excess = demand - effective_capacity
                queue_growth = int(excess * time_step_hours)
                state.queue_length = min(
                    state.queue_length + queue_growth,
                    self.config.max_queue_length
                )
            else:
                # Tiêu tán hàng đợi
                discharge = int(
                    self.config.queue_discharge_rate *
                    edge.lanes *
                    time_step_hours *
                    state.capacity_factor
                )
                state.queue_length = max(0, state.queue_length - discharge)

    def _calculate_network_state(self) -> NetworkTrafficState:
        """Tính toán trạng thái giao thông mạng lưới tổng hợp."""
        network_state = NetworkTrafficState()

        total_flow = 0
        total_queue = 0
        total_speed_weighted = 0.0
        total_length = 0.0
        congested = 0
        gridlocked = 0

        for edge_id, state in self._edge_states.items():
            edge = self.network.get_edge(edge_id)
            if not edge:
                continue

            total_flow += state.current_flow
            total_queue += state.queue_length

            # Trọng số tốc độ theo độ dài cạnh
            total_speed_weighted += state.speed * edge.length_km
            total_length += edge.length_km

            if state.state == TrafficState.CONGESTED:
                congested += 1
            elif state.state == TrafficState.GRIDLOCK:
                gridlocked += 1

            network_state.edges[edge_id] = state

        network_state.total_vehicles = int(
            total_flow / self.config.persons_per_vehicle
        )
        network_state.total_in_queue = total_queue
        network_state.average_speed = (
            total_speed_weighted / total_length if total_length > 0 else 0
        )
        network_state.congested_edges = congested
        network_state.gridlocked_edges = gridlocked

        return network_state

    def get_edge_travel_time(self, edge_id: str) -> float:
        """Lấy thời gian di chuyển hiện tại cho một cạnh."""
        state = self._edge_states.get(edge_id)
        if state:
            return state.travel_time
        return float('inf')

    def get_edge_speed(self, edge_id: str) -> float:
        """Lấy tốc độ hiện tại trên một cạnh."""
        state = self._edge_states.get(edge_id)
        if state:
            return state.speed
        return 0.0

    def get_route_travel_time(self, path: List[str]) -> float:
        """
        Tính toán tổng thời gian di chuyển dọc theo tuyến đường.

        Args:
            path: Danh sách các ID node

        Returns:
            Tổng thời gian di chuyển tính bằng giờ
        """
        if len(path) < 2:
            return 0.0

        total_time = 0.0
        for i in range(len(path) - 1):
            edge = self.network.get_edge_between(path[i], path[i + 1])
            if edge:
                state = self._edge_states.get(edge.id)
                if state:
                    total_time += state.travel_time
                    # Thêm độ trễ giao lộ
                    total_time += self.config.signalized_delay
                else:
                    total_time += edge.base_travel_time

        return total_time

    def get_congestion_map(self) -> Dict[str, float]:
        """
        Lấy mức độ tắc nghẽn cho tất cả các cạnh.

        Returns:
            Từ điển edge_id -> mức độ tắc nghẽn (0-1)
        """
        congestion_map = {}
        for edge_id, state in self._edge_states.items():
            edge = self.network.get_edge(edge_id)
            if edge:
                congestion_map[edge_id] = edge.congestion_level
        return congestion_map

    def apply_incident(self, edge_id: str, capacity_reduction: float) -> None:
        """
        Áp dụng sự cố (tai nạn, hỏng hóc) cho một cạnh.

        Args:
            edge_id: Cạnh bị ảnh hưởng
            capacity_reduction: Hệ số giảm công suất (0-1)
        """
        edge = self.network.get_edge(edge_id)
        if edge:
            state = self._edge_states.get(edge_id)
            if state:
                state.capacity_factor *= (1 - capacity_reduction)

    def clear_incident(self, edge_id: str) -> None:
        """Xóa sự cố khỏi một cạnh."""
        state = self._edge_states.get(edge_id)
        if state:
            state.capacity_factor = 1.0


@dataclass
class FlowAssignment:
    """Phân công dòng chảy cho cân bằng mạng lưới."""
    edge_id: str
    flow: float  # Xe mỗi giờ
    travel_time: float  # Thời gian di chuyển hiện tại


class TrafficAssignment:
    """
    Mô hình phân công giao thông cho phân tích cân bằng.

    Triển khai cân bằng người dùng cơ bản sử dụng Phương pháp Trung bình Liên tiếp (MSA).
    """

    def __init__(self, network: EvacuationNetwork,
                 config: Optional[TrafficConfig] = None):
        """
        Khởi tạo mô hình phân công giao thông.

        Args:
            network: Mạng lưới sơ tán
            config: Cấu hình giao thông
        """
        self.network = network
        self.config = config or TrafficConfig()

    def assign_flow(self, od_matrix: Dict[Tuple[str, str], float],
                    max_iterations: int = 50,
                    convergence_threshold: float = 0.01) -> Dict[str, float]:
        """
        Phân công dòng chảy giao thông sử dụng Phương pháp Trung bình Liên tiếp.

        Args:
            od_matrix: Ma trận nhu cầu xuất phát-đích {(origin_id, dest_id): flow}
            max_iterations: Số lần lặp tối đa để hội tụ
            convergence_threshold: Tiêu chí hội tụ

        Returns:
            Từ điển edge_id -> dòng chảy được phân công
        """
        # Khởi tạo với phân công tất cả hoặc không
        edge_flows: Dict[str, float] = {}

        for iteration in range(max_iterations):
            # Tính toán thời gian di chuyển hiện tại
            travel_times = self._calculate_travel_times(edge_flows)

            # Phân công tất cả hoặc không với thời gian hiện tại
            new_flows = self._all_or_nothing_assignment(od_matrix, travel_times)

            # Tính trung bình MSA
            alpha = 1.0 / (iteration + 1)
            for edge_id in set(edge_flows.keys()) | set(new_flows.keys()):
                old_flow = edge_flows.get(edge_id, 0.0)
                new_flow = new_flows.get(edge_id, 0.0)
                edge_flows[edge_id] = old_flow + alpha * (new_flow - old_flow)

            # Kiểm tra hội tụ
            if iteration > 0:
                gap = self._calculate_gap(edge_flows, new_flows)
                if gap < convergence_threshold:
                    break

        return edge_flows

    def _calculate_travel_times(self,
                                edge_flows: Dict[str, float]) -> Dict[str, float]:
        """Tính toán thời gian di chuyển cho các dòng chảy đã cho."""
        travel_times = {}
        for edge in self.network.get_edges():
            flow = edge_flows.get(edge.id, 0.0)
            capacity = edge.capacity

            if capacity > 0:
                vc_ratio = flow / capacity
            else:
                vc_ratio = 1.0

            # Hàm BPR
            travel_times[edge.id] = edge.base_travel_time * (
                1 + self.config.bpr_alpha * (vc_ratio ** self.config.bpr_beta)
            )

        return travel_times

    def _all_or_nothing_assignment(
            self,
            od_matrix: Dict[Tuple[str, str], float],
            travel_times: Dict[str, float]) -> Dict[str, float]:
        """Thực hiện phân công tất cả hoặc không sử dụng đường đi ngắn nhất."""
        edge_flows: Dict[str, float] = {}

        for (origin, destination), demand in od_matrix.items():
            # Tìm đường đi ngắn nhất sử dụng thời gian di chuyển hiện tại
            path = self._find_shortest_path(origin, destination, travel_times)

            if path:
                # Phân công tất cả nhu cầu cho đường đi này
                for i in range(len(path) - 1):
                    edge = self.network.get_edge_between(path[i], path[i + 1])
                    if edge:
                        edge_flows[edge.id] = edge_flows.get(edge.id, 0.0) + demand

        return edge_flows

    def _find_shortest_path(self, origin: str, destination: str,
                           travel_times: Dict[str, float]) -> Optional[List[str]]:
        """Tìm đường đi ngắn nhất sử dụng thuật toán Dijkstra."""
        import heapq

        distances = {origin: 0.0}
        predecessors: Dict[str, str] = {}
        pq = [(0.0, origin)]
        visited = set()

        while pq:
            dist, node = heapq.heappop(pq)

            if node in visited:
                continue
            visited.add(node)

            if node == destination:
                # Tái tạo đường đi
                path = [destination]
                current = destination
                while current in predecessors:
                    current = predecessors[current]
                    path.append(current)
                return list(reversed(path))

            for neighbor in self.network.get_neighbors(node):
                if neighbor in visited:
                    continue

                edge = self.network.get_edge_between(node, neighbor)
                if edge and not edge.is_blocked:
                    edge_time = travel_times.get(edge.id, edge.base_travel_time)
                    new_dist = dist + edge_time

                    if new_dist < distances.get(neighbor, float('inf')):
                        distances[neighbor] = new_dist
                        predecessors[neighbor] = node
                        heapq.heappush(pq, (new_dist, neighbor))

        return None

    def _calculate_gap(self, old_flows: Dict[str, float],
                       new_flows: Dict[str, float]) -> float:
        """Tính toán khoảng cách tương đối giữa các phân công dòng chảy."""
        total_diff = 0.0
        total_flow = 0.0

        for edge_id in set(old_flows.keys()) | set(new_flows.keys()):
            old = old_flows.get(edge_id, 0.0)
            new = new_flows.get(edge_id, 0.0)
            total_diff += abs(new - old)
            total_flow += old

        if total_flow > 0:
            return total_diff / total_flow
        return 0.0
