"""
Dynamic Events System for evacuation simulation.

Handles dynamic events that occur during evacuation:
- Road blockages (accidents, debris, flooding)
- Shelter status changes (capacity updates, closures)
- Hazard zone expansion
- Route rerouting triggers
"""

from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random
import uuid


class EventType(Enum):
    """Types of simulation events."""
    # Road events
    ROAD_BLOCKED = "road_blocked"
    ROAD_CLEARED = "road_cleared"
    ROAD_CAPACITY_REDUCED = "road_capacity_reduced"
    ACCIDENT = "accident"
    FLOODING = "flooding"

    # Shelter events
    SHELTER_CLOSED = "shelter_closed"
    SHELTER_OPENED = "shelter_opened"
    SHELTER_CAPACITY_CHANGED = "shelter_capacity_changed"
    SHELTER_FULL = "shelter_full"

    # Hazard events
    HAZARD_CREATED = "hazard_created"
    HAZARD_EXPANDED = "hazard_expanded"
    HAZARD_CLEARED = "hazard_cleared"

    # Evacuation events
    EVACUATION_STARTED = "evacuation_started"
    EVACUATION_COMPLETED = "evacuation_completed"
    ROUTE_BLOCKED = "route_blocked"
    REROUTE_NEEDED = "reroute_needed"

    # System events
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_PAUSED = "simulation_paused"
    SIMULATION_COMPLETED = "simulation_completed"


class EventPriority(Enum):
    """Priority levels for events."""
    CRITICAL = 1  # Must be processed immediately
    HIGH = 2  # High priority
    NORMAL = 3  # Normal priority
    LOW = 4  # Low priority, can be delayed


@dataclass
class SimulationEvent:
    """Represents a dynamic event during simulation."""
    id: str
    event_type: EventType
    timestamp: float  # Simulation time in hours
    priority: EventPriority = EventPriority.NORMAL
    data: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    # Optional callback for custom handling
    callback: Optional[Callable[['SimulationEvent'], None]] = None

    def __lt__(self, other: 'SimulationEvent') -> bool:
        """Comparison for priority queue (lower priority value = higher priority)."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.timestamp < other.timestamp


@dataclass
class ScheduledEvent:
    """An event scheduled to occur at a specific time."""
    event: SimulationEvent
    trigger_time: float  # When to trigger (simulation time in hours)
    recurring: bool = False
    interval: float = 0.0  # Recurrence interval in hours


# Type alias for event handlers
EventHandler = Callable[[SimulationEvent], None]


class EventQueue:
    """Priority queue for simulation events."""

    def __init__(self):
        self._events: List[SimulationEvent] = []
        self._scheduled: List[ScheduledEvent] = []

    def push(self, event: SimulationEvent) -> None:
        """Add an event to the queue."""
        self._events.append(event)
        self._events.sort()

    def pop(self) -> Optional[SimulationEvent]:
        """Remove and return the highest priority event."""
        if self._events:
            return self._events.pop(0)
        return None

    def peek(self) -> Optional[SimulationEvent]:
        """Return the highest priority event without removing it."""
        if self._events:
            return self._events[0]
        return None

    def schedule(self, event: SimulationEvent, trigger_time: float,
                 recurring: bool = False, interval: float = 0.0) -> None:
        """Schedule an event for future execution."""
        scheduled = ScheduledEvent(
            event=event,
            trigger_time=trigger_time,
            recurring=recurring,
            interval=interval
        )
        self._scheduled.append(scheduled)
        self._scheduled.sort(key=lambda s: s.trigger_time)

    def check_scheduled(self, current_time: float) -> List[SimulationEvent]:
        """Check for scheduled events that should trigger."""
        triggered = []
        remaining = []

        for scheduled in self._scheduled:
            if scheduled.trigger_time <= current_time:
                scheduled.event.timestamp = current_time
                triggered.append(scheduled.event)

                if scheduled.recurring:
                    # Reschedule for next occurrence
                    scheduled.trigger_time += scheduled.interval
                    remaining.append(scheduled)
            else:
                remaining.append(scheduled)

        self._scheduled = remaining
        return triggered

    def clear(self) -> None:
        """Clear all events."""
        self._events.clear()
        self._scheduled.clear()

    def __len__(self) -> int:
        return len(self._events)

    @property
    def pending_count(self) -> int:
        """Number of pending events."""
        return len(self._events)

    @property
    def scheduled_count(self) -> int:
        """Number of scheduled events."""
        return len(self._scheduled)


class EventManager:
    """
    Manages simulation events and their handlers.

    Provides:
    - Event registration and dispatch
    - Scheduled event execution
    - Event history tracking
    - Handler subscription
    """

    def __init__(self):
        self._queue = EventQueue()
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []
        self._history: List[SimulationEvent] = []
        self._max_history = 1000

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Subscribe a handler to a specific event type.

        Args:
            event_type: Type of event to handle
            handler: Callback function
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe a handler to all events."""
        self._global_handlers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event type."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h != handler
            ]

    def emit(self, event: SimulationEvent) -> None:
        """Emit an event immediately."""
        self._dispatch(event)

    def queue(self, event: SimulationEvent) -> None:
        """Add an event to the queue for later processing."""
        self._queue.push(event)

    def schedule(self, event: SimulationEvent, trigger_time: float,
                 recurring: bool = False, interval: float = 0.0) -> None:
        """Schedule an event for future execution."""
        self._queue.schedule(event, trigger_time, recurring, interval)

    def process_queue(self, current_time: float) -> int:
        """
        Process all pending events up to current time.

        Args:
            current_time: Current simulation time in hours

        Returns:
            Number of events processed
        """
        processed = 0

        # Check for scheduled events that should trigger
        triggered = self._queue.check_scheduled(current_time)
        for event in triggered:
            self._queue.push(event)

        # Process queued events
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
        """Dispatch an event to its handlers."""
        event.processed = True

        # Record in history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Call event-specific callback
        if event.callback:
            event.callback(event)

        # Call type-specific handlers
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            handler(event)

        # Call global handlers
        for handler in self._global_handlers:
            handler(event)

    def get_history(self, event_type: Optional[EventType] = None,
                    limit: int = 100) -> List[SimulationEvent]:
        """
        Get event history.

        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return

        Returns:
            List of historical events
        """
        if event_type:
            filtered = [e for e in self._history if e.event_type == event_type]
        else:
            filtered = self._history

        return filtered[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()

    def clear(self) -> None:
        """Clear all events and history."""
        self._queue.clear()
        self._history.clear()

    @property
    def pending_count(self) -> int:
        """Number of pending events."""
        return self._queue.pending_count

    @property
    def scheduled_count(self) -> int:
        """Number of scheduled events."""
        return self._queue.scheduled_count


class EventFactory:
    """Factory for creating common simulation events."""

    @staticmethod
    def create_road_blocked(edge_id: str, timestamp: float,
                           reason: str = "unknown") -> SimulationEvent:
        """Create a road blocked event."""
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
        """Create a road cleared event."""
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
        """Create an accident event."""
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
        """Create a flooding event."""
        # Determine severity based on water depth
        if depth_cm > 50:
            severity = 1.0  # Impassable
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
                              reason: str = "unknown") -> SimulationEvent:
        """Create a shelter closed event."""
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
        """Create a shelter full event."""
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
        """Create a new hazard event."""
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
        """Create a hazard expansion event."""
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
                              reason: str = "blocked") -> SimulationEvent:
        """Create a reroute needed event."""
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
    Generates random events for simulation scenarios.

    Useful for testing and creating dynamic scenarios.
    """

    def __init__(self, network, seed: Optional[int] = None):
        """
        Initialize event generator.

        Args:
            network: The evacuation network
            seed: Random seed for reproducibility
        """
        self.network = network
        self.rng = random.Random(seed)

    def generate_random_accident(self, timestamp: float) -> SimulationEvent:
        """Generate a random accident event."""
        edges = list(self.network.get_edges())
        if not edges:
            raise ValueError("Network has no edges")

        edge = self.rng.choice(edges)
        severity = self.rng.uniform(0.3, 0.9)

        return EventFactory.create_accident(edge.id, timestamp, severity)

    def generate_random_flooding(self, timestamp: float) -> SimulationEvent:
        """Generate a random flooding event on a low-lying road."""
        edges = list(self.network.get_edges())
        if not edges:
            raise ValueError("Network has no edges")

        # Prefer edges already at risk
        at_risk = [e for e in edges if e.flood_risk > 0.3]
        if at_risk:
            edge = self.rng.choice(at_risk)
        else:
            edge = self.rng.choice(edges)

        depth = self.rng.uniform(20, 80)  # 20-80 cm

        return EventFactory.create_flooding(edge.id, timestamp, depth)

    def generate_random_shelter_closure(self, timestamp: float) -> Optional[SimulationEvent]:
        """Generate a random shelter closure event."""
        shelters = [s for s in self.network.get_shelters() if s.is_active]
        if not shelters:
            return None

        shelter = self.rng.choice(shelters)
        reasons = ["structural damage", "flooding", "capacity exceeded", "access blocked"]
        reason = self.rng.choice(reasons)

        return EventFactory.create_shelter_closed(shelter.id, timestamp, reason)

    def generate_scenario_events(self,
                                  duration_hours: float,
                                  event_frequency: float = 0.5) -> List[SimulationEvent]:
        """
        Generate a sequence of random events for a scenario.

        Args:
            duration_hours: Total scenario duration
            event_frequency: Average events per hour

        Returns:
            List of events sorted by timestamp
        """
        events = []
        current_time = 0.0

        while current_time < duration_hours:
            # Random time to next event (exponential distribution)
            time_to_next = self.rng.expovariate(event_frequency)
            current_time += time_to_next

            if current_time >= duration_hours:
                break

            # Choose event type
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
