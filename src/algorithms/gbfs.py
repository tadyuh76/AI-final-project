"""
Greedy Best-First Search (GBFS) Algorithm for evacuation pathfinding.

Uses a multi-objective heuristic combining:
- Distance to shelter
- Flood/hazard risk
- Road congestion
- Shelter capacity remaining
"""

from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass
import heapq
import time

from .base import (
    BaseAlgorithm, AlgorithmType, AlgorithmConfig,
    EvacuationPlan, EvacuationRoute, AlgorithmMetrics
)
from ..models.network import EvacuationNetwork
from ..models.node import Node, PopulationZone, Shelter, haversine_distance


@dataclass
class SearchNode:
    """Node in the GBFS search tree."""
    node_id: str
    g_cost: float  # Actual cost from start
    h_cost: float  # Heuristic cost to goal
    parent: Optional['SearchNode'] = None

    @property
    def f_cost(self) -> float:
        """Total estimated cost (for GBFS, we primarily use h_cost)."""
        return self.h_cost

    def __lt__(self, other: 'SearchNode') -> bool:
        """Comparison for priority queue (lower f_cost is better)."""
        return self.f_cost < other.f_cost


class GreedyBestFirstSearch(BaseAlgorithm):
    """
    Greedy Best-First Search for evacuation pathfinding.

    Finds paths from population zones to shelters using a multi-objective
    heuristic that considers distance, risk, congestion, and capacity.
    """

    def __init__(self, network: EvacuationNetwork, config: Optional[AlgorithmConfig] = None):
        """
        Initialize GBFS algorithm.

        Args:
            network: The evacuation network
            config: Algorithm configuration (optional)
        """
        super().__init__(network)
        self.config = config or AlgorithmConfig()

        # Weights for heuristic components
        self.w_dist = self.config.distance_weight
        self.w_risk = self.config.risk_weight
        self.w_congestion = self.config.congestion_weight
        self.w_capacity = self.config.capacity_weight

    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.GBFS

    def heuristic(self, node: Node, goal: Shelter, current_flow: Dict[str, int]) -> float:
        """
        Calculate multi-objective heuristic value.

        Lower values are better.

        Args:
            node: Current node
            goal: Target shelter
            current_flow: Current flow on each edge

        Returns:
            Heuristic cost estimate
        """
        # Distance component (normalized by typical max distance ~30km)
        h_dist = haversine_distance(node.lat, node.lon, goal.lat, goal.lon) / 30.0

        # Risk component (from hazard zones)
        h_risk = self.network.get_total_risk_at(node.lat, node.lon)

        # Congestion component (average edge congestion near this node)
        h_congestion = self._get_local_congestion(node.id)

        # Capacity component (prefer shelters with more remaining capacity)
        # Invert so lower is better
        capacity_ratio = goal.occupancy_rate if goal.capacity > 0 else 1.0
        h_capacity = capacity_ratio

        # Combine with weights
        return (self.w_dist * h_dist +
                self.w_risk * h_risk +
                self.w_congestion * h_congestion +
                self.w_capacity * h_capacity)

    def _get_local_congestion(self, node_id: str) -> float:
        """Get average congestion of edges connected to a node."""
        edges = self.network.get_outgoing_edges(node_id)
        if not edges:
            return 0.0
        return sum(e.congestion_level for e in edges) / len(edges)

    def find_path(self, source: PopulationZone,
                  shelters: List[Shelter]) -> Tuple[Optional[List[str]], Optional[Shelter], float]:
        """
        Find the best path from a population zone to any available shelter.

        Args:
            source: Starting population zone
            shelters: List of potential destination shelters

        Returns:
            Tuple of (path as list of node IDs, chosen shelter, total cost)
            Returns (None, None, inf) if no path found
        """
        if not shelters:
            return None, None, float('inf')

        # Filter to shelters with capacity
        available_shelters = [s for s in shelters if s.has_capacity()]
        if not available_shelters:
            return None, None, float('inf')

        # Create goal set
        goal_ids = {s.id for s in available_shelters}
        shelter_map = {s.id: s for s in available_shelters}

        # Priority queue: (f_cost, counter, SearchNode)
        # Counter for tie-breaking to avoid comparing SearchNodes
        counter = 0
        open_set: List[Tuple[float, int, SearchNode]] = []

        # Find nearest intersection to source
        source_node = self.network.find_nearest_node(source.lat, source.lon)
        if not source_node:
            return None, None, float('inf')

        # Initialize with source node
        start = SearchNode(
            node_id=source_node.id,
            g_cost=0.0,
            h_cost=min(self.heuristic(source_node, s, {}) for s in available_shelters)
        )
        heapq.heappush(open_set, (start.f_cost, counter, start))
        counter += 1

        # Visited set
        visited: Set[str] = set()
        current_flow: Dict[str, int] = {}

        while open_set:
            if self._should_stop:
                break

            _, _, current = heapq.heappop(open_set)

            # Skip if already visited
            if current.node_id in visited:
                continue
            visited.add(current.node_id)

            # Check if we reached a goal (shelter)
            if current.node_id in goal_ids:
                # Reconstruct path
                path = self._reconstruct_path(current)
                shelter = shelter_map[current.node_id]
                return path, shelter, current.g_cost

            # Expand neighbors
            for neighbor_id in self.network.get_neighbors(current.node_id):
                if neighbor_id in visited:
                    continue

                # Get edge cost
                edge = self.network.get_edge_between(current.node_id, neighbor_id)
                if not edge or edge.is_blocked:
                    continue

                edge_cost = edge.get_cost(self.w_risk)
                new_g_cost = current.g_cost + edge_cost

                # Get neighbor node for heuristic
                neighbor_node = self.network.get_node(neighbor_id)
                if not neighbor_node:
                    continue

                # Calculate heuristic to nearest shelter
                h_cost = min(
                    self.heuristic(neighbor_node, s, current_flow)
                    for s in available_shelters
                )

                neighbor = SearchNode(
                    node_id=neighbor_id,
                    g_cost=new_g_cost,
                    h_cost=h_cost,
                    parent=current
                )

                heapq.heappush(open_set, (neighbor.f_cost, counter, neighbor))
                counter += 1

        return None, None, float('inf')

    def _reconstruct_path(self, node: SearchNode) -> List[str]:
        """Reconstruct path from SearchNode chain."""
        path = []
        current: Optional[SearchNode] = node
        while current:
            path.append(current.node_id)
            current = current.parent
        return list(reversed(path))

    def optimize(self, **kwargs) -> Tuple[EvacuationPlan, AlgorithmMetrics]:
        """
        Run GBFS optimization for all population zones.

        Returns:
            Tuple of (EvacuationPlan, AlgorithmMetrics)
        """
        start_time = self._start_timer()

        plan = EvacuationPlan(algorithm_type=AlgorithmType.GBFS)
        zones = self.network.get_population_zones()
        shelters = self.network.get_shelters()

        if not zones or not shelters:
            self._stop_timer(start_time)
            return plan, self._metrics

        total_zones = len(zones)
        total_cost = 0.0
        paths_found = 0
        total_path_length = 0

        # Sort zones by population (prioritize larger zones)
        zones = sorted(zones, key=lambda z: z.population, reverse=True)

        for i, zone in enumerate(zones):
            if self._should_stop:
                break

            # Find path for this zone
            path, shelter, cost = self.find_path(zone, shelters)

            if path and shelter:
                # Calculate route metrics
                route_distance = self._calculate_path_distance(path)
                route_time = self._calculate_path_time(path)
                route_risk = self._calculate_path_risk(path)

                # Determine flow (all remaining population from zone)
                flow = zone.remaining_population

                # Check shelter capacity
                actual_flow = min(flow, shelter.available_capacity)

                if actual_flow > 0:
                    route = EvacuationRoute(
                        zone_id=zone.id,
                        shelter_id=shelter.id,
                        path=path,
                        flow=actual_flow,
                        distance_km=route_distance,
                        estimated_time_hours=route_time,
                        risk_score=route_risk
                    )
                    plan.add_route(route)

                    # Update shelter occupancy (for capacity-aware routing)
                    shelter.current_occupancy += actual_flow
                    zone.evacuated += actual_flow

                    total_cost += cost
                    paths_found += 1
                    total_path_length += len(path)

            # Report progress
            self._metrics.convergence_history.append(total_cost)
            self.report_progress(i + 1, total_cost, plan)

        # Finalize metrics
        self._stop_timer(start_time)
        self._metrics.iterations = total_zones
        self._metrics.final_cost = total_cost
        self._metrics.routes_found = paths_found
        self._metrics.evacuees_covered = plan.total_evacuees
        self._metrics.average_path_length = (
            total_path_length / paths_found if paths_found > 0 else 0
        )

        # Coverage rate: evacuees covered / min(total_population, total_capacity)
        total_population = sum(z.population for z in zones)
        total_capacity = sum(s.capacity for s in shelters)
        max_possible = min(total_population, total_capacity)
        self._metrics.coverage_rate = (
            plan.total_evacuees / max_possible if max_possible > 0 else 0
        )

        return plan, self._metrics

    def _calculate_path_distance(self, path: List[str]) -> float:
        """Calculate total distance of a path in km."""
        if len(path) < 2:
            return 0.0

        total = 0.0
        for i in range(len(path) - 1):
            edge = self.network.get_edge_between(path[i], path[i + 1])
            if edge:
                total += edge.length_km
        return total

    def _calculate_path_time(self, path: List[str]) -> float:
        """Calculate estimated travel time in hours."""
        if len(path) < 2:
            return 0.0

        total = 0.0
        for i in range(len(path) - 1):
            edge = self.network.get_edge_between(path[i], path[i + 1])
            if edge:
                total += edge.current_travel_time
        return total

    def _calculate_path_risk(self, path: List[str]) -> float:
        """Calculate average risk along a path."""
        if not path:
            return 0.0

        total_risk = 0.0
        for node_id in path:
            node = self.network.get_node(node_id)
            if node:
                total_risk += self.network.get_total_risk_at(node.lat, node.lon)

        return total_risk / len(path)

    def find_multiple_paths(self, source: PopulationZone,
                           shelters: List[Shelter],
                           k: int = 3) -> List[Tuple[List[str], Shelter, float]]:
        """
        Find k-best paths from source to shelters.

        Useful for flow distribution across multiple routes.

        Args:
            source: Starting zone
            shelters: Available shelters
            k: Number of paths to find

        Returns:
            List of (path, shelter, cost) tuples
        """
        paths = []
        used_shelters: Set[str] = set()

        for _ in range(k):
            # Exclude already-used shelters
            available = [s for s in shelters
                        if s.id not in used_shelters and s.has_capacity()]
            if not available:
                break

            path, shelter, cost = self.find_path(source, available)
            if path and shelter:
                paths.append((path, shelter, cost))
                used_shelters.add(shelter.id)
            else:
                break

        return paths
