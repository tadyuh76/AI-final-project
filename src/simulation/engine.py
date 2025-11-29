"""
Simulation Engine for evacuation scenarios.

Provides time-stepped simulation of evacuation with:
- Progressive evacuation flow along routes
- Dynamic hazard progression
- Real-time metrics tracking
- Event-driven updates
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
    """Possible states of the simulation."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    time_step_minutes: float = 5.0  # Simulation time step
    max_duration_hours: float = 24.0  # Maximum simulation duration
    speed_multiplier: float = 1.0  # Real-time speed multiplier

    # Flow parameters
    flow_rate_per_step: float = 0.1  # Fraction of remaining population per step
    min_flow_per_route: int = 10  # Minimum evacuees per route per step

    # Hazard progression
    hazard_expansion_rate: float = 0.01  # km per time step
    hazard_intensity_growth: float = 0.005  # Risk increase per step

    # Congestion effects
    congestion_threshold: float = 0.7  # Flow/capacity ratio for congestion
    congestion_speed_factor: float = 0.5  # Speed reduction at full congestion


@dataclass
class SimulationMetrics:
    """Real-time metrics during simulation."""
    current_time_hours: float = 0.0
    total_evacuated: int = 0
    total_remaining: int = 0
    active_routes: int = 0
    completed_routes: int = 0
    blocked_routes: int = 0

    # Shelter metrics
    shelter_utilization: Dict[str, float] = field(default_factory=dict)
    shelter_arrivals: Dict[str, int] = field(default_factory=dict)

    # Route metrics
    average_travel_time: float = 0.0
    average_risk_exposure: float = 0.0

    # Progress tracking
    evacuation_progress: float = 0.0  # 0.0 to 1.0
    estimated_completion_hours: float = 0.0

    # History for visualization
    evacuation_history: List[Tuple[float, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
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
            'shelter_utilization': self.shelter_utilization
        }


@dataclass
class RouteState:
    """State of an individual evacuation route during simulation."""
    route: EvacuationRoute
    total_assigned: int  # Total people assigned to this route
    departed: int = 0  # People who have started evacuation
    in_transit: int = 0  # People currently on the route
    arrived: int = 0  # People who have reached shelter
    blocked: bool = False  # Route is blocked by hazard

    # Timing
    start_time: float = 0.0
    estimated_arrival: float = 0.0

    # Tracking evacuees in transit with their progress
    evacuee_groups: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        """People still waiting to depart."""
        return max(0, self.total_assigned - self.departed)

    @property
    def is_complete(self) -> bool:
        """Check if all assigned evacuees have arrived."""
        return self.arrived >= self.total_assigned and self.in_transit == 0


# Type alias for callbacks
SimulationCallback = Callable[[SimulationMetrics, Dict[str, RouteState]], None]


class SimulationEngine:
    """
    Time-stepped evacuation simulation engine.

    Simulates the evacuation process over time with:
    - Gradual population movement along routes
    - Dynamic congestion based on flow
    - Hazard zone progression
    - Real-time metrics updates
    """

    def __init__(self, network: EvacuationNetwork,
                 config: Optional[SimulationConfig] = None):
        """
        Initialize simulation engine.

        Args:
            network: The evacuation network
            config: Simulation configuration
        """
        self.network = network
        self.config = config or SimulationConfig()

        # State
        self._state = SimulationState.IDLE
        self._current_time = 0.0  # Simulation time in hours
        self._metrics = SimulationMetrics()
        self._route_states: Dict[str, RouteState] = {}

        # Callbacks
        self._update_callback: Optional[SimulationCallback] = None
        self._completion_callback: Optional[Callable[[SimulationMetrics], None]] = None

        # Control
        self._should_stop = False
        self._is_paused = False

        # Initial population snapshot
        self._initial_population = 0

    @property
    def state(self) -> SimulationState:
        """Get current simulation state."""
        return self._state

    @property
    def metrics(self) -> SimulationMetrics:
        """Get current metrics."""
        return self._metrics

    @property
    def current_time(self) -> float:
        """Get current simulation time in hours."""
        return self._current_time

    def set_update_callback(self, callback: SimulationCallback) -> None:
        """Set callback for simulation updates."""
        self._update_callback = callback

    def set_completion_callback(self,
                                callback: Callable[[SimulationMetrics], None]) -> None:
        """Set callback for simulation completion."""
        self._completion_callback = callback

    def initialize(self, plan: EvacuationPlan) -> None:
        """
        Initialize simulation with an evacuation plan.

        Args:
            plan: The evacuation plan to simulate
        """
        self._state = SimulationState.IDLE
        self._current_time = 0.0
        self._should_stop = False
        self._is_paused = False

        # Reset network state
        self.network.reset_simulation_state()

        # Calculate initial population
        self._initial_population = sum(
            z.population for z in self.network.get_population_zones()
        )

        # Initialize route states
        self._route_states.clear()
        for i, route in enumerate(plan.routes):
            route_id = f"route_{i}_{route.zone_id}_{route.shelter_id}"
            self._route_states[route_id] = RouteState(
                route=route,
                total_assigned=route.flow,
                estimated_arrival=route.estimated_time_hours
            )

        # Initialize metrics
        self._metrics = SimulationMetrics(
            total_remaining=self._initial_population,
            active_routes=len(self._route_states)
        )

        # Initialize shelter utilization tracking
        for shelter in self.network.get_shelters():
            self._metrics.shelter_utilization[shelter.id] = 0.0
            self._metrics.shelter_arrivals[shelter.id] = 0

    def run(self, plan: EvacuationPlan,
            real_time: bool = False) -> SimulationMetrics:
        """
        Run the simulation to completion.

        Args:
            plan: Evacuation plan to simulate
            real_time: If True, run in real-time with delays

        Returns:
            Final simulation metrics
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

            # Perform simulation step
            self._step(time_step_hours)

            # Check completion
            if self._is_evacuation_complete():
                self._state = SimulationState.COMPLETED
                break

            # Real-time delay
            if real_time:
                delay = (self.config.time_step_minutes * 60) / self.config.speed_multiplier
                time.sleep(delay)

        # Final callback
        if self._completion_callback:
            self._completion_callback(self._metrics)

        return self._metrics

    def step(self) -> SimulationMetrics:
        """
        Execute a single simulation step.

        Returns:
            Updated metrics after step
        """
        if self._state == SimulationState.IDLE:
            return self._metrics

        time_step_hours = self.config.time_step_minutes / 60.0
        self._step(time_step_hours)

        return self._metrics

    def _step(self, time_step_hours: float) -> None:
        """Internal step execution."""
        self._current_time += time_step_hours

        # 1. Update hazard zones (progression)
        self._update_hazards(time_step_hours)

        # 2. Check for blocked routes
        self._check_route_blockages()

        # 3. Process departures from zones
        self._process_departures(time_step_hours)

        # 4. Move evacuees in transit
        self._process_transit(time_step_hours)

        # 5. Process arrivals at shelters
        self._process_arrivals()

        # 6. Update metrics
        self._update_metrics()

        # 7. Callback
        if self._update_callback:
            self._update_callback(self._metrics, self._route_states)

    def _update_hazards(self, time_step_hours: float) -> None:
        """Update hazard zones (expansion, intensity)."""
        for hazard in self.network.get_hazard_zones():
            # Expand radius
            hazard.radius_km += self.config.hazard_expansion_rate * (time_step_hours * 60)

            # Increase intensity (capped at 1.0)
            hazard.risk_level = min(1.0,
                hazard.risk_level + self.config.hazard_intensity_growth * (time_step_hours * 60))

        # Update edge risks based on new hazard positions
        for edge in self.network.get_edges():
            source = self.network.get_node(edge.source_id)
            target = self.network.get_node(edge.target_id)
            if source and target:
                mid_lat = (source.lat + target.lat) / 2
                mid_lon = (source.lon + target.lon) / 2
                risk = self.network.get_total_risk_at(mid_lat, mid_lon)
                edge.set_flood_risk(risk)

    def _check_route_blockages(self) -> None:
        """Check if any routes are blocked by hazards."""
        for route_id, state in self._route_states.items():
            if state.blocked:
                continue

            # Check each edge in the route
            path = state.route.path
            for i in range(len(path) - 1):
                edge = self.network.get_edge_between(path[i], path[i + 1])
                if edge and edge.is_blocked:
                    state.blocked = True
                    break

    def _process_departures(self, time_step_hours: float) -> None:
        """Process evacuees departing from zones."""
        for route_id, state in self._route_states.items():
            if state.blocked or state.remaining <= 0:
                continue

            # Calculate departure rate
            # More people leave as time goes on (urgency)
            urgency_factor = 1.0 + (self._current_time / 2.0)  # Increases over time
            base_rate = self.config.flow_rate_per_step * urgency_factor

            departing = max(
                self.config.min_flow_per_route,
                int(state.remaining * base_rate)
            )
            departing = min(departing, state.remaining)

            if departing > 0:
                # Add to edge flows
                self._add_flow_to_path(state.route.path, departing)

                # Create evacuee group
                state.evacuee_groups.append({
                    'count': departing,
                    'progress': 0.0,  # 0.0 = at start, 1.0 = at destination
                    'departure_time': self._current_time
                })

                state.departed += departing
                state.in_transit += departing

    def _process_transit(self, time_step_hours: float) -> None:
        """Move evacuees along their routes."""
        for route_id, state in self._route_states.items():
            if not state.evacuee_groups:
                continue

            # Calculate movement based on route travel time
            base_travel_time = state.route.estimated_time_hours
            if base_travel_time <= 0:
                base_travel_time = 0.1  # Minimum travel time

            # Apply congestion factor
            congestion_factor = self._get_route_congestion(state.route.path)
            effective_travel_time = base_travel_time * (1.0 + congestion_factor)

            # Progress per time step
            progress_per_step = time_step_hours / effective_travel_time

            # Update each group
            completed_groups = []
            for group in state.evacuee_groups:
                group['progress'] += progress_per_step
                if group['progress'] >= 1.0:
                    completed_groups.append(group)

            # Process completed groups
            for group in completed_groups:
                state.evacuee_groups.remove(group)
                state.in_transit -= group['count']
                state.arrived += group['count']

                # Remove flow from path
                self._remove_flow_from_path(state.route.path, group['count'])

    def _process_arrivals(self) -> None:
        """Process evacuees arriving at shelters."""
        for route_id, state in self._route_states.items():
            if state.arrived > self._metrics.shelter_arrivals.get(state.route.shelter_id, 0):
                new_arrivals = state.arrived - self._metrics.shelter_arrivals.get(
                    state.route.shelter_id, 0)

                # Update shelter
                shelter = self.network.get_node(state.route.shelter_id)
                if isinstance(shelter, Shelter):
                    admitted = shelter.admit(new_arrivals)
                    self._metrics.shelter_arrivals[shelter.id] = \
                        self._metrics.shelter_arrivals.get(shelter.id, 0) + admitted
                    self._metrics.shelter_utilization[shelter.id] = shelter.occupancy_rate

    def _add_flow_to_path(self, path: List[str], count: int) -> None:
        """Add flow to all edges in a path."""
        for i in range(len(path) - 1):
            edge = self.network.get_edge_between(path[i], path[i + 1])
            if edge:
                edge.add_flow(count)

    def _remove_flow_from_path(self, path: List[str], count: int) -> None:
        """Remove flow from all edges in a path."""
        for i in range(len(path) - 1):
            edge = self.network.get_edge_between(path[i], path[i + 1])
            if edge:
                edge.remove_flow(count)

    def _get_route_congestion(self, path: List[str]) -> float:
        """Calculate average congestion along a route."""
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

        # Apply non-linear congestion effect
        if avg_congestion > self.config.congestion_threshold:
            excess = avg_congestion - self.config.congestion_threshold
            return excess * (1.0 / (1.0 - self.config.congestion_threshold))

        return 0.0

    def _update_metrics(self) -> None:
        """Update simulation metrics."""
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

            # Accumulate weighted metrics
            if state.arrived > 0:
                total_travel_time += state.route.estimated_time_hours * state.arrived
                total_risk += state.route.risk_score * state.arrived

        self._metrics.current_time_hours = self._current_time
        self._metrics.total_evacuated = total_evacuated
        self._metrics.total_remaining = self._initial_population - total_evacuated - total_in_transit
        self._metrics.active_routes = active_routes
        self._metrics.completed_routes = completed_routes
        self._metrics.blocked_routes = blocked_routes

        # Calculate averages
        if total_evacuated > 0:
            self._metrics.average_travel_time = total_travel_time / total_evacuated
            self._metrics.average_risk_exposure = total_risk / total_evacuated

        # Progress
        if self._initial_population > 0:
            self._metrics.evacuation_progress = total_evacuated / self._initial_population

        # Estimate completion time
        if total_evacuated > 0 and self._current_time > 0:
            rate = total_evacuated / self._current_time
            remaining = self._initial_population - total_evacuated
            if rate > 0:
                self._metrics.estimated_completion_hours = \
                    self._current_time + (remaining / rate)

        # Record history point
        self._metrics.evacuation_history.append(
            (self._current_time, total_evacuated)
        )

    def _is_evacuation_complete(self) -> bool:
        """Check if evacuation is complete."""
        # All routes completed or blocked
        for state in self._route_states.values():
            if not state.is_complete and not state.blocked:
                return False
        return True

    def pause(self) -> None:
        """Pause the simulation."""
        self._is_paused = True
        self._state = SimulationState.PAUSED

    def resume(self) -> None:
        """Resume the simulation."""
        self._is_paused = False
        self._state = SimulationState.RUNNING

    def stop(self) -> None:
        """Stop the simulation."""
        self._should_stop = True
        self._state = SimulationState.IDLE

    def reset(self) -> None:
        """Reset the simulation to initial state."""
        self._state = SimulationState.IDLE
        self._current_time = 0.0
        self._should_stop = False
        self._is_paused = False
        self._route_states.clear()
        self._metrics = SimulationMetrics()
        self.network.reset_simulation_state()

    def get_route_states(self) -> Dict[str, RouteState]:
        """Get current route states."""
        return self._route_states

    def get_evacuation_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of current evacuation state for visualization."""
        snapshot = {
            'time': self._current_time,
            'state': self._state.value,
            'metrics': self._metrics.to_dict(),
            'routes': [],
            'shelters': [],
            'hazards': []
        }

        # Route data
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

        # Shelter data
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

        # Hazard data
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
