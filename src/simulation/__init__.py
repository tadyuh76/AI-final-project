"""
Module Mô phỏng cho mô phỏng kịch bản sơ tán.

Cung cấp:
- SimulationEngine: Mô phỏng sơ tán theo bước thời gian
- TrafficFlowModel: Động lực dòng chảy giao thông dựa trên BPR
- EventManager: Hệ thống xử lý sự kiện động
"""

from .engine import (
    SimulationEngine,
    SimulationState,
    SimulationConfig,
    SimulationMetrics,
    RouteState
)

from .traffic import (
    TrafficFlowModel,
    TrafficState,
    TrafficConfig,
    EdgeTrafficState,
    NetworkTrafficState,
    TrafficAssignment
)

from .events import (
    EventManager,
    EventQueue,
    EventFactory,
    EventType,
    EventPriority,
    SimulationEvent,
    ScheduledEvent,
    RandomEventGenerator
)

__all__ = [
    # Engine
    'SimulationEngine',
    'SimulationState',
    'SimulationConfig',
    'SimulationMetrics',
    'RouteState',
    # Traffic
    'TrafficFlowModel',
    'TrafficState',
    'TrafficConfig',
    'EdgeTrafficState',
    'NetworkTrafficState',
    'TrafficAssignment',
    # Events
    'EventManager',
    'EventQueue',
    'EventFactory',
    'EventType',
    'EventPriority',
    'SimulationEvent',
    'ScheduledEvent',
    'RandomEventGenerator',
]
