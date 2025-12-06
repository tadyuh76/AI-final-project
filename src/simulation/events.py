"""
Hệ thống Sự kiện Động cho mô phỏng sơ tán.

Xử lý các sự kiện động xảy ra trong quá trình sơ tán:
- Tắc nghẽn đường (tai nạn, mảnh vỡ, ngập lụt)
- Thay đổi trạng thái nơi trú ẩn (cập nhật công suất, đóng cửa)
- Mở rộng vùng nguy hiểm
- Kích hoạt định tuyến lại
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random
import uuid


class EventType(Enum):
    """Các loại sự kiện mô phỏng."""
    # Sự kiện đường
    ROAD_BLOCKED = "road_blocked"
    ROAD_CLEARED = "road_cleared"
    ROAD_CAPACITY_REDUCED = "road_capacity_reduced"
    ACCIDENT = "accident"
    FLOODING = "flooding"

    # Sự kiện nơi trú ẩn
    SHELTER_CLOSED = "shelter_closed"
    SHELTER_OPENED = "shelter_opened"
    SHELTER_CAPACITY_CHANGED = "shelter_capacity_changed"
    SHELTER_FULL = "shelter_full"

    # Sự kiện nguy hiểm
    HAZARD_CREATED = "hazard_created"
    HAZARD_EXPANDED = "hazard_expanded"
    HAZARD_CLEARED = "hazard_cleared"

    # Sự kiện sơ tán
    EVACUATION_STARTED = "evacuation_started"
    EVACUATION_COMPLETED = "evacuation_completed"
    ROUTE_BLOCKED = "route_blocked"
    REROUTE_NEEDED = "reroute_needed"

    # Sự kiện hệ thống
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_PAUSED = "simulation_paused"
    SIMULATION_COMPLETED = "simulation_completed"


class EventPriority(Enum):
    """Mức độ ưu tiên cho các sự kiện."""
    CRITICAL = 1  # Phải được xử lý ngay lập tức
    HIGH = 2  # Ưu tiên cao
    NORMAL = 3  # Ưu tiên thông thường
    LOW = 4  # Ưu tiên thấp, có thể trì hoãn


@dataclass
class SimulationEvent:
    """Đại diện cho một sự kiện động trong quá trình mô phỏng."""
    id: str
    event_type: EventType
    timestamp: float  # Thời gian mô phỏng tính bằng giờ
    priority: EventPriority = EventPriority.NORMAL
    data: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    # Callback tùy chọn cho xử lý tùy chỉnh
    callback: Optional[Callable[['SimulationEvent'], None]] = None

    def __lt__(self, other: 'SimulationEvent') -> bool:
        """So sánh cho hàng đợi ưu tiên (giá trị ưu tiên thấp hơn = ưu tiên cao hơn)."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.timestamp < other.timestamp


@dataclass
class ScheduledEvent:
    """Một sự kiện được lên lịch xảy ra vào một thời điểm cụ thể."""
    event: SimulationEvent
    trigger_time: float  # Thời điểm kích hoạt (thời gian mô phỏng tính bằng giờ)
    recurring: bool = False
    interval: float = 0.0  # Khoảng thời gian tái diễn tính bằng giờ


# Bí danh kiểu cho trình xử lý sự kiện
EventHandler = Callable[[SimulationEvent], None]


class EventQueue:
    """Hàng đợi ưu tiên cho các sự kiện mô phỏng."""

    def __init__(self):
        self._events: List[SimulationEvent] = []
        self._scheduled: List[ScheduledEvent] = []

    def push(self, event: SimulationEvent) -> None:
        """Thêm một sự kiện vào hàng đợi."""
        self._events.append(event)
        self._events.sort()

    def pop(self) -> Optional[SimulationEvent]:
        """Xóa và trả về sự kiện có ưu tiên cao nhất."""
        if self._events:
            return self._events.pop(0)
        return None

    def peek(self) -> Optional[SimulationEvent]:
        """Trả về sự kiện có ưu tiên cao nhất mà không xóa nó."""
        if self._events:
            return self._events[0]
        return None

    def schedule(self, event: SimulationEvent, trigger_time: float,
                 recurring: bool = False, interval: float = 0.0) -> None:
        """Lên lịch một sự kiện để thực thi trong tương lai."""
        scheduled = ScheduledEvent(
            event=event,
            trigger_time=trigger_time,
            recurring=recurring,
            interval=interval
        )
        self._scheduled.append(scheduled)
        self._scheduled.sort(key=lambda s: s.trigger_time)

    def check_scheduled(self, current_time: float) -> List[SimulationEvent]:
        """Kiểm tra các sự kiện đã lên lịch nên kích hoạt."""
        triggered = []
        remaining = []

        for scheduled in self._scheduled:
            if scheduled.trigger_time <= current_time:
                scheduled.event.timestamp = current_time
                triggered.append(scheduled.event)

                if scheduled.recurring:
                    # Lên lịch lại cho lần xuất hiện tiếp theo
                    scheduled.trigger_time += scheduled.interval
                    remaining.append(scheduled)
            else:
                remaining.append(scheduled)

        self._scheduled = remaining
        return triggered

    def clear(self) -> None:
        """Xóa tất cả các sự kiện."""
        self._events.clear()
        self._scheduled.clear()

    def __len__(self) -> int:
        return len(self._events)

    @property
    def pending_count(self) -> int:
        """Số sự kiện đang chờ xử lý."""
        return len(self._events)

    @property
    def scheduled_count(self) -> int:
        """Số sự kiện đã lên lịch."""
        return len(self._scheduled)


class EventManager:
    """
    Quản lý các sự kiện mô phỏng và trình xử lý của chúng.

    Cung cấp:
    - Đăng ký và phân phối sự kiện
    - Thực thi sự kiện đã lên lịch
    - Theo dõi lịch sử sự kiện
    - Đăng ký trình xử lý
    """

    def __init__(self):
        self._queue = EventQueue()
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []
        self._history: List[SimulationEvent] = []
        self._max_history = 1000

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Đăng ký một trình xử lý cho một loại sự kiện cụ thể.

        Args:
            event_type: Loại sự kiện cần xử lý
            handler: Hàm callback
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Đăng ký một trình xử lý cho tất cả các sự kiện."""
        self._global_handlers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Hủy đăng ký một trình xử lý khỏi một loại sự kiện."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h != handler
            ]

    def emit(self, event: SimulationEvent) -> None:
        """Phát ra một sự kiện ngay lập tức."""
        self._dispatch(event)

    def queue(self, event: SimulationEvent) -> None:
        """Thêm một sự kiện vào hàng đợi để xử lý sau."""
        self._queue.push(event)

    def schedule(self, event: SimulationEvent, trigger_time: float,
                 recurring: bool = False, interval: float = 0.0) -> None:
        """Lên lịch một sự kiện để thực thi trong tương lai."""
        self._queue.schedule(event, trigger_time, recurring, interval)

    def process_queue(self, current_time: float) -> int:
        """
        Xử lý tất cả các sự kiện đang chờ cho đến thời gian hiện tại.

        Args:
            current_time: Thời gian mô phỏng hiện tại tính bằng giờ

        Returns:
            Số sự kiện đã xử lý
        """
        processed = 0

        # Kiểm tra các sự kiện đã lên lịch nên kích hoạt
        triggered = self._queue.check_scheduled(current_time)
        for event in triggered:
            self._queue.push(event)

        # Xử lý các sự kiện trong hàng đợi
        while True:
            event = self._queue.peek()
            if event is None:
                break
            if event.timestamp > current_time:
                break

            event = self._queue.pop()
            if event:
                self._dispatch(event)
                processed += 1

        return processed

    def _dispatch(self, event: SimulationEvent) -> None:
        """Phân phối một sự kiện đến các trình xử lý của nó."""
        event.processed = True

        # Ghi vào lịch sử
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Gọi callback cụ thể cho sự kiện
        if event.callback:
            event.callback(event)

        # Gọi các trình xử lý cụ thể theo loại
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            handler(event)

        # Gọi các trình xử lý toàn cục
        for handler in self._global_handlers:
            handler(event)

    def get_history(self, event_type: Optional[EventType] = None,
                    limit: int = 100) -> List[SimulationEvent]:
        """
        Lấy lịch sử sự kiện.

        Args:
            event_type: Lọc theo loại sự kiện (tùy chọn)
            limit: Số lượng sự kiện tối đa để trả về

        Returns:
            Danh sách các sự kiện lịch sử
        """
        if event_type:
            filtered = [e for e in self._history if e.event_type == event_type]
        else:
            filtered = self._history

        return filtered[-limit:]

    def clear_history(self) -> None:
        """Xóa lịch sử sự kiện."""
        self._history.clear()

    def clear(self) -> None:
        """Xóa tất cả các sự kiện và lịch sử."""
        self._queue.clear()
        self._history.clear()

    @property
    def pending_count(self) -> int:
        """Số sự kiện đang chờ xử lý."""
        return self._queue.pending_count

    @property
    def scheduled_count(self) -> int:
        """Số sự kiện đã lên lịch."""
        return self._queue.scheduled_count


class EventFactory:
    """Nhà máy để tạo các sự kiện mô phỏng thông thường."""

    @staticmethod
    def create_road_blocked(edge_id: str, timestamp: float,
                           reason: str = "không rõ") -> SimulationEvent:
        """Tạo sự kiện đường bị chặn."""
        return SimulationEvent(
            id=str(uuid.uuid4()),
            event_type=EventType.ROAD_BLOCKED,
            timestamp=timestamp,
            priority=EventPriority.HIGH,
            data={
                'edge_id': edge_id,
                'reason': reason
            }
        )

    @staticmethod
    def create_road_cleared(edge_id: str, timestamp: float) -> SimulationEvent:
        """Tạo sự kiện đường được thông thoáng."""
        return SimulationEvent(
            id=str(uuid.uuid4()),
            event_type=EventType.ROAD_CLEARED,
            timestamp=timestamp,
            priority=EventPriority.NORMAL,
            data={'edge_id': edge_id}
        )

    @staticmethod
    def create_accident(edge_id: str, timestamp: float,
                       severity: float = 0.5) -> SimulationEvent:
        """Tạo sự kiện tai nạn."""
        return SimulationEvent(
            id=str(uuid.uuid4()),
            event_type=EventType.ACCIDENT,
            timestamp=timestamp,
            priority=EventPriority.HIGH,
            data={
                'edge_id': edge_id,
                'severity': severity,
                'capacity_reduction': severity * 0.8
            }
        )

    @staticmethod
    def create_flooding(edge_id: str, timestamp: float,
                        depth_cm: float = 30.0) -> SimulationEvent:
        """Tạo sự kiện ngập lụt."""
        # Xác định mức độ nghiêm trọng dựa trên độ sâu nước
        if depth_cm > 50:
            severity = 1.0  # Không thể đi qua
        elif depth_cm > 30:
            severity = 0.8
        else:
            severity = 0.5

        return SimulationEvent(
            id=str(uuid.uuid4()),
            event_type=EventType.FLOODING,
            timestamp=timestamp,
            priority=EventPriority.HIGH,
            data={
                'edge_id': edge_id,
                'depth_cm': depth_cm,
                'severity': severity
            }
        )

    @staticmethod
    def create_shelter_closed(shelter_id: str, timestamp: float,
                              reason: str = "không rõ") -> SimulationEvent:
        """Tạo sự kiện đóng cửa nơi trú ẩn."""
        return SimulationEvent(
            id=str(uuid.uuid4()),
            event_type=EventType.SHELTER_CLOSED,
            timestamp=timestamp,
            priority=EventPriority.CRITICAL,
            data={
                'shelter_id': shelter_id,
                'reason': reason
            }
        )

    @staticmethod
    def create_shelter_full(shelter_id: str, timestamp: float) -> SimulationEvent:
        """Tạo sự kiện nơi trú ẩn đầy."""
        return SimulationEvent(
            id=str(uuid.uuid4()),
            event_type=EventType.SHELTER_FULL,
            timestamp=timestamp,
            priority=EventPriority.HIGH,
            data={'shelter_id': shelter_id}
        )

    @staticmethod
    def create_hazard(center_lat: float, center_lon: float,
                      radius_km: float, timestamp: float,
                      hazard_type: str = "flood") -> SimulationEvent:
        """Tạo sự kiện nguy hiểm mới."""
        return SimulationEvent(
            id=str(uuid.uuid4()),
            event_type=EventType.HAZARD_CREATED,
            timestamp=timestamp,
            priority=EventPriority.CRITICAL,
            data={
                'center_lat': center_lat,
                'center_lon': center_lon,
                'radius_km': radius_km,
                'hazard_type': hazard_type,
                'risk_level': 0.8
            }
        )

    @staticmethod
    def create_hazard_expanded(hazard_index: int, new_radius: float,
                               timestamp: float) -> SimulationEvent:
        """Tạo sự kiện mở rộng nguy hiểm."""
        return SimulationEvent(
            id=str(uuid.uuid4()),
            event_type=EventType.HAZARD_EXPANDED,
            timestamp=timestamp,
            priority=EventPriority.HIGH,
            data={
                'hazard_index': hazard_index,
                'new_radius': new_radius
            }
        )

    @staticmethod
    def create_reroute_needed(route_id: str, timestamp: float,
                              reason: str = "bị chặn") -> SimulationEvent:
        """Tạo sự kiện cần định tuyến lại."""
        return SimulationEvent(
            id=str(uuid.uuid4()),
            event_type=EventType.REROUTE_NEEDED,
            timestamp=timestamp,
            priority=EventPriority.HIGH,
            data={
                'route_id': route_id,
                'reason': reason
            }
        )


class RandomEventGenerator:
    """
    Tạo các sự kiện ngẫu nhiên cho các kịch bản mô phỏng.

    Hữu ích cho việc kiểm tra và tạo các kịch bản động.
    """

    def __init__(self, network, seed: Optional[int] = None):
        """
        Khởi tạo trình tạo sự kiện.

        Args:
            network: Mạng lưới sơ tán
            seed: Hạt giống ngẫu nhiên để tái tạo
        """
        self.network = network
        self.rng = random.Random(seed)

    def generate_random_accident(self, timestamp: float) -> SimulationEvent:
        """Tạo sự kiện tai nạn ngẫu nhiên."""
        edges = list(self.network.get_edges())
        if not edges:
            raise ValueError("Mạng lưới không có cạnh nào")

        edge = self.rng.choice(edges)
        severity = self.rng.uniform(0.3, 0.9)

        return EventFactory.create_accident(edge.id, timestamp, severity)

    def generate_random_flooding(self, timestamp: float) -> SimulationEvent:
        """Tạo sự kiện ngập lụt ngẫu nhiên trên đường thấp."""
        edges = list(self.network.get_edges())
        if not edges:
            raise ValueError("Mạng lưới không có cạnh nào")

        # Ưu tiên các cạnh đã có rủi ro
        at_risk = [e for e in edges if e.flood_risk > 0.3]
        if at_risk:
            edge = self.rng.choice(at_risk)
        else:
            edge = self.rng.choice(edges)

        depth = self.rng.uniform(20, 80)  # 20-80 cm

        return EventFactory.create_flooding(edge.id, timestamp, depth)

    def generate_random_shelter_closure(self, timestamp: float) -> Optional[SimulationEvent]:
        """Tạo sự kiện đóng cửa nơi trú ẩn ngẫu nhiên."""
        shelters = [s for s in self.network.get_shelters() if s.is_active]
        if not shelters:
            return None

        shelter = self.rng.choice(shelters)
        reasons = ["hư hỏng kết cấu", "ngập lụt", "vượt quá công suất", "lối vào bị chặn"]
        reason = self.rng.choice(reasons)

        return EventFactory.create_shelter_closed(shelter.id, timestamp, reason)

    def generate_scenario_events(self,
                                  duration_hours: float,
                                  event_frequency: float = 0.5) -> List[SimulationEvent]:
        """
        Tạo một chuỗi sự kiện ngẫu nhiên cho một kịch bản.

        Args:
            duration_hours: Tổng thời lượng kịch bản
            event_frequency: Số sự kiện trung bình mỗi giờ

        Returns:
            Danh sách các sự kiện được sắp xếp theo thời gian
        """
        events = []
        current_time = 0.0

        while current_time < duration_hours:
            # Thời gian ngẫu nhiên đến sự kiện tiếp theo (phân phối mũ)
            time_to_next = self.rng.expovariate(event_frequency)
            current_time += time_to_next

            if current_time >= duration_hours:
                break

            # Chọn loại sự kiện
            event_type = self.rng.choices(
                ['accident', 'flooding', 'shelter_closure'],
                weights=[0.4, 0.4, 0.2]
            )[0]

            try:
                if event_type == 'accident':
                    event = self.generate_random_accident(current_time)
                elif event_type == 'flooding':
                    event = self.generate_random_flooding(current_time)
                else:
                    event = self.generate_random_shelter_closure(current_time)

                if event:
                    events.append(event)
            except ValueError:
                continue

        return sorted(events, key=lambda e: e.timestamp)
