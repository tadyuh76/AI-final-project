"""
Traffic Flow Model for evacuation simulation.

Implements traffic flow dynamics including:
- BPR (Bureau of Public Roads) speed-flow relationship
- Queue formation and dissipation
- Intersection delays
- Dynamic capacity reduction from hazards
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

from ..models.network import EvacuationNetwork
from ..models.edge import Edge, RoadType


class TrafficState(Enum):
    """Traffic flow states."""
    FREE_FLOW = "free_flow"
    SYNCHRONIZED = "synchronized"
    CONGESTED = "congested"
    GRIDLOCK = "gridlock"


@dataclass
class TrafficConfig:
    """Configuration for traffic model parameters."""
    # BPR function parameters
    bpr_alpha: float = 0.15  # BPR alpha parameter
    bpr_beta: float = 4.0  # BPR beta parameter (power)

    # Capacity factors
    hazard_capacity_reduction: float = 0.5  # Capacity reduction in hazard zones
    rain_capacity_reduction: float = 0.2  # Capacity reduction during rain
    night_capacity_reduction: float = 0.15  # Capacity reduction at night

    # Queue parameters
    queue_discharge_rate: float = 1800  # Vehicles per hour per lane
    max_queue_length: int = 500  # Maximum queue before spillback

    # Intersection delays (hours)
    signalized_delay: float = 0.01  # ~36 seconds average
    unsignalized_delay: float = 0.005  # ~18 seconds average

    # Person-to-vehicle conversion
    persons_per_vehicle: float = 3.0  # Average occupancy during evacuation


@dataclass
class EdgeTrafficState:
    """Traffic state for an individual edge."""
    edge_id: str
    current_flow: int = 0  # Current flow (persons)
    queue_length: int = 0  # Waiting queue at entry
    travel_time: float = 0.0  # Current travel time (hours)
    speed: float = 0.0  # Current speed (km/h)
    density: float = 0.0  # Vehicles per km
    state: TrafficState = TrafficState.FREE_FLOW
    capacity_factor: float = 1.0  # Effective capacity multiplier


@dataclass
class NetworkTrafficState:
    """Aggregate traffic state for the network."""
    total_vehicles: int = 0
    total_in_queue: int = 0
    average_speed: float = 0.0
    congested_edges: int = 0
    gridlocked_edges: int = 0
    edges: Dict[str, EdgeTrafficState] = field(default_factory=dict)


class TrafficFlowModel:
    """
    Traffic flow simulation model.

    Implements macroscopic traffic flow using:
    - BPR function for speed-flow relationship
    - First-order kinematic wave model for propagation
    - Queue-based intersection modeling
    """

    def __init__(self, network: EvacuationNetwork,
                 config: Optional[TrafficConfig] = None):
        """
        Initialize traffic model.

        Args:
            network: The evacuation network
            config: Traffic configuration
        """
        self.network = network
        self.config = config or TrafficConfig()

        # Traffic state per edge
        self._edge_states: Dict[str, EdgeTrafficState] = {}

        # Initialize edge states
        self._initialize_edge_states()

    def _initialize_edge_states(self) -> None:
        """Initialize traffic state for all edges."""
        for edge in self.network.get_edges():
            self._edge_states[edge.id] = EdgeTrafficState(
                edge_id=edge.id,
                speed=edge.max_speed_kmh,
                travel_time=edge.base_travel_time
            )

    def reset(self) -> None:
        """Reset traffic state."""
        self._initialize_edge_states()

    def update(self, time_step_hours: float) -> NetworkTrafficState:
        """
        Update traffic state for one time step.

        Args:
            time_step_hours: Time step duration in hours

        Returns:
            Updated network traffic state
        """
        # Update each edge
        for edge in self.network.get_edges():
            state = self._edge_states.get(edge.id)
            if state:
                self._update_edge_state(edge, state, time_step_hours)

        # Process queues and spillback
        self._process_queues(time_step_hours)

        # Calculate network-wide metrics
        return self._calculate_network_state()

    def _update_edge_state(self, edge: Edge, state: EdgeTrafficState,
                           time_step_hours: float) -> None:
        """Update traffic state for a single edge."""
        # Get current flow from edge
        state.current_flow = edge.current_flow

        # Calculate effective capacity
        base_capacity = edge.capacity
        capacity_factor = self._calculate_capacity_factor(edge)
        state.capacity_factor = capacity_factor
        effective_capacity = base_capacity * capacity_factor

        # Convert persons to vehicles
        vehicle_flow = state.current_flow / self.config.persons_per_vehicle

        # Calculate volume-to-capacity ratio
        if effective_capacity > 0:
            vc_ratio = vehicle_flow / effective_capacity
        else:
            vc_ratio = 1.0

        # Apply BPR function for travel time
        base_time = edge.base_travel_time
        state.travel_time = self._bpr_travel_time(base_time, vc_ratio)

        # Calculate speed
        if state.travel_time > 0:
            state.speed = edge.length_km / state.travel_time
        else:
            state.speed = edge.max_speed_kmh

        # Calculate density (vehicles per km)
        if edge.length_km > 0:
            state.density = vehicle_flow / (edge.length_km * edge.lanes)
        else:
            state.density = 0

        # Determine traffic state
        state.state = self._determine_traffic_state(vc_ratio)

    def _bpr_travel_time(self, free_flow_time: float,
                         vc_ratio: float) -> float:
        """
        Calculate travel time using BPR function.

        BPR: t = t0 * (1 + alpha * (v/c)^beta)

        Args:
            free_flow_time: Free-flow travel time
            vc_ratio: Volume-to-capacity ratio

        Returns:
            Congested travel time
        """
        return free_flow_time * (
            1 + self.config.bpr_alpha * (vc_ratio ** self.config.bpr_beta)
        )

    def _calculate_capacity_factor(self, edge: Edge) -> float:
        """Calculate effective capacity factor based on conditions."""
        factor = 1.0

        # Reduce capacity in hazard zones
        if edge.flood_risk > 0:
            factor *= (1 - edge.flood_risk * self.config.hazard_capacity_reduction)

        # Blocked edges have zero capacity
        if edge.is_blocked:
            return 0.0

        return max(0.1, factor)  # Minimum 10% capacity

    def _determine_traffic_state(self, vc_ratio: float) -> TrafficState:
        """Determine traffic state based on V/C ratio."""
        if vc_ratio < 0.5:
            return TrafficState.FREE_FLOW
        elif vc_ratio < 0.8:
            return TrafficState.SYNCHRONIZED
        elif vc_ratio < 1.0:
            return TrafficState.CONGESTED
        else:
            return TrafficState.GRIDLOCK

    def _process_queues(self, time_step_hours: float) -> None:
        """Process queue formation and dissipation."""
        for edge_id, state in self._edge_states.items():
            edge = self.network.get_edge(edge_id)
            if not edge:
                continue

            # Queue forms when demand exceeds capacity
            effective_capacity = edge.capacity * state.capacity_factor
            demand = edge.current_flow / self.config.persons_per_vehicle

            if demand > effective_capacity:
                # Queue growth
                excess = demand - effective_capacity
                queue_growth = int(excess * time_step_hours)
                state.queue_length = min(
                    state.queue_length + queue_growth,
                    self.config.max_queue_length
                )
            else:
                # Queue dissipation
                discharge = int(
                    self.config.queue_discharge_rate *
                    edge.lanes *
                    time_step_hours *
                    state.capacity_factor
                )
                state.queue_length = max(0, state.queue_length - discharge)

    def _calculate_network_state(self) -> NetworkTrafficState:
        """Calculate aggregate network traffic state."""
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

            # Weight speed by edge length
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
        """Get current travel time for an edge."""
        state = self._edge_states.get(edge_id)
        if state:
            return state.travel_time
        return float('inf')

    def get_edge_speed(self, edge_id: str) -> float:
        """Get current speed on an edge."""
        state = self._edge_states.get(edge_id)
        if state:
            return state.speed
        return 0.0

    def get_route_travel_time(self, path: List[str]) -> float:
        """
        Calculate total travel time along a route.

        Args:
            path: List of node IDs

        Returns:
            Total travel time in hours
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
                    # Add intersection delay
                    total_time += self.config.signalized_delay
                else:
                    total_time += edge.base_travel_time

        return total_time

    def get_congestion_map(self) -> Dict[str, float]:
        """
        Get congestion levels for all edges.

        Returns:
            Dictionary of edge_id -> congestion level (0-1)
        """
        congestion_map = {}
        for edge_id, state in self._edge_states.items():
            edge = self.network.get_edge(edge_id)
            if edge:
                congestion_map[edge_id] = edge.congestion_level
        return congestion_map

    def apply_incident(self, edge_id: str, capacity_reduction: float) -> None:
        """
        Apply an incident (accident, breakdown) to an edge.

        Args:
            edge_id: Edge to affect
            capacity_reduction: Capacity reduction factor (0-1)
        """
        edge = self.network.get_edge(edge_id)
        if edge:
            state = self._edge_states.get(edge_id)
            if state:
                state.capacity_factor *= (1 - capacity_reduction)

    def clear_incident(self, edge_id: str) -> None:
        """Clear an incident from an edge."""
        state = self._edge_states.get(edge_id)
        if state:
            state.capacity_factor = 1.0


@dataclass
class FlowAssignment:
    """Flow assignment for network equilibrium."""
    edge_id: str
    flow: float  # Vehicles per hour
    travel_time: float  # Current travel time


class TrafficAssignment:
    """
    Traffic assignment model for equilibrium analysis.

    Implements basic user equilibrium using Method of Successive Averages (MSA).
    """

    def __init__(self, network: EvacuationNetwork,
                 config: Optional[TrafficConfig] = None):
        """
        Initialize traffic assignment model.

        Args:
            network: The evacuation network
            config: Traffic configuration
        """
        self.network = network
        self.config = config or TrafficConfig()

    def assign_flow(self, od_matrix: Dict[Tuple[str, str], float],
                    max_iterations: int = 50,
                    convergence_threshold: float = 0.01) -> Dict[str, float]:
        """
        Assign traffic flow using Method of Successive Averages.

        Args:
            od_matrix: Origin-destination demand matrix {(origin_id, dest_id): flow}
            max_iterations: Maximum iterations for convergence
            convergence_threshold: Convergence criterion

        Returns:
            Dictionary of edge_id -> assigned flow
        """
        # Initialize with all-or-nothing assignment
        edge_flows: Dict[str, float] = {}

        for iteration in range(max_iterations):
            # Calculate current travel times
            travel_times = self._calculate_travel_times(edge_flows)

            # All-or-nothing assignment with current times
            new_flows = self._all_or_nothing_assignment(od_matrix, travel_times)

            # MSA averaging
            alpha = 1.0 / (iteration + 1)
            for edge_id in set(edge_flows.keys()) | set(new_flows.keys()):
                old_flow = edge_flows.get(edge_id, 0.0)
                new_flow = new_flows.get(edge_id, 0.0)
                edge_flows[edge_id] = old_flow + alpha * (new_flow - old_flow)

            # Check convergence
            if iteration > 0:
                gap = self._calculate_gap(edge_flows, new_flows)
                if gap < convergence_threshold:
                    break

        return edge_flows

    def _calculate_travel_times(self,
                                edge_flows: Dict[str, float]) -> Dict[str, float]:
        """Calculate travel times for given flows."""
        travel_times = {}
        for edge in self.network.get_edges():
            flow = edge_flows.get(edge.id, 0.0)
            capacity = edge.capacity

            if capacity > 0:
                vc_ratio = flow / capacity
            else:
                vc_ratio = 1.0

            # BPR function
            travel_times[edge.id] = edge.base_travel_time * (
                1 + self.config.bpr_alpha * (vc_ratio ** self.config.bpr_beta)
            )

        return travel_times

    def _all_or_nothing_assignment(
            self,
            od_matrix: Dict[Tuple[str, str], float],
            travel_times: Dict[str, float]) -> Dict[str, float]:
        """Perform all-or-nothing assignment using shortest paths."""
        edge_flows: Dict[str, float] = {}

        for (origin, destination), demand in od_matrix.items():
            # Find shortest path using current travel times
            path = self._find_shortest_path(origin, destination, travel_times)

            if path:
                # Assign all demand to this path
                for i in range(len(path) - 1):
                    edge = self.network.get_edge_between(path[i], path[i + 1])
                    if edge:
                        edge_flows[edge.id] = edge_flows.get(edge.id, 0.0) + demand

        return edge_flows

    def _find_shortest_path(self, origin: str, destination: str,
                           travel_times: Dict[str, float]) -> Optional[List[str]]:
        """Find shortest path using Dijkstra's algorithm."""
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
                # Reconstruct path
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
        """Calculate relative gap between flow assignments."""
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
