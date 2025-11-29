"""
Simulation module for evacuation scenario simulation.

Provides:
- SimulationEngine: Time-stepped evacuation simulation
- TrafficFlowModel: BPR-based traffic flow dynamics
- EventManager: Dynamic event handling system
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
