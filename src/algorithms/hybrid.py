"""
Hybrid GBFS + GWO Algorithm for evacuation optimization.

Combines the strengths of both algorithms:
- GWO: Global optimization of flow distribution
- GBFS: Local pathfinding with multi-objective heuristic

Two-phase approach:
1. Phase 1 (GWO): Optimize which zones send people to which shelters
2. Phase 2 (GBFS): Find actual paths for each flow assignment
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import time

from .base import (
    BaseAlgorithm, AlgorithmType, AlgorithmConfig,
    EvacuationPlan, EvacuationRoute, AlgorithmMetrics
)
from .gbfs import GreedyBestFirstSearch
from .gwo import GreyWolfOptimizer
from ..models.network import EvacuationNetwork
from ..models.node import PopulationZone, Shelter


class HybridGBFSGWO(BaseAlgorithm):
    """
    Hybrid algorithm combining GWO global optimization with GBFS pathfinding.

    Phase 1: GWO determines optimal flow distribution
    Phase 2: GBFS finds actual paths for each assignment
    Refinement: Iteratively improve based on actual path costs
    """

    def __init__(self, network: EvacuationNetwork, config: Optional[AlgorithmConfig] = None):
        """
        Initialize hybrid algorithm.

        Args:
            network: The evacuation network
            config: Algorithm configuration (optional)
        """
        super().__init__(network)
        self.config = config or AlgorithmConfig()

        # Create sub-algorithms
        self.gwo = GreyWolfOptimizer(network, config)
        self.gbfs = GreedyBestFirstSearch(network, config)

        # Hybrid-specific parameters
        self.gwo_iterations = self.config.gwo_iterations
        self.refinement_iterations = self.config.refinement_iterations

    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.HYBRID

    def optimize(self, **kwargs) -> Tuple[EvacuationPlan, AlgorithmMetrics]:
        """
        Run hybrid optimization.

        Returns:
            Tuple of (EvacuationPlan, AlgorithmMetrics)
        """
        start_time = self._start_timer()

        zones = self.network.get_population_zones()
        shelters = self.network.get_shelters()

        if not zones or not shelters:
            self._stop_timer(start_time)
            return EvacuationPlan(algorithm_type=AlgorithmType.HYBRID), self._metrics

        # ============ Phase 1: GWO Global Optimization ============
        self.report_progress(0, float('inf'), {'phase': 'gwo_start'})

        # Configure GWO for fewer iterations in hybrid mode
        gwo_config = AlgorithmConfig(
            n_wolves=self.config.n_wolves,
            max_iterations=self.gwo_iterations,
            distance_weight=self.config.distance_weight,
            risk_weight=self.config.risk_weight
        )
        self.gwo = GreyWolfOptimizer(self.network, gwo_config)

        # Set up progress forwarding
        def gwo_progress(iteration: int, cost: float, data: Any) -> None:
            self._metrics.convergence_history.append(cost)
            self.report_progress(iteration, cost, {'phase': 'gwo', 'data': data})

        self.gwo.set_progress_callback(gwo_progress)

        # Run GWO
        gwo_plan, gwo_metrics = self.gwo.optimize()
        flow_matrix = self.gwo.get_flow_matrix()

        if flow_matrix is None:
            self._stop_timer(start_time)
            return gwo_plan, self._metrics

        # ============ Phase 2: GBFS Pathfinding ============
        self.report_progress(self.gwo_iterations, gwo_metrics.final_cost, {'phase': 'gbfs_start'})

        plan = self._apply_gbfs_pathfinding(flow_matrix, zones, shelters)

        # ============ Phase 3: Refinement ============
        self.report_progress(
            self.gwo_iterations + 1,
            self._calculate_plan_cost(plan),
            {'phase': 'refinement_start'}
        )

        plan = self._refine_plan(plan, flow_matrix, zones, shelters)

        # ============ Finalize ============
        self._stop_timer(start_time)

        # Calculate final metrics
        self._metrics.iterations = (
            self.gwo_iterations + len(zones) + self.refinement_iterations
        )
        self._metrics.final_cost = self._calculate_plan_cost(plan)
        self._metrics.routes_found = len(plan.routes)
        self._metrics.evacuees_covered = plan.total_evacuees

        total_population = sum(z.population for z in zones)
        self._metrics.coverage_rate = (
            plan.total_evacuees / total_population if total_population > 0 else 0
        )

        if plan.routes:
            self._metrics.average_path_length = (
                sum(len(r.path) for r in plan.routes) / len(plan.routes)
            )

        return plan, self._metrics

    def _apply_gbfs_pathfinding(self,
                                 flow_matrix: np.ndarray,
                                 zones: List[PopulationZone],
                                 shelters: List[Shelter]) -> EvacuationPlan:
        """
        Apply GBFS pathfinding to GWO flow assignments.

        Args:
            flow_matrix: GWO-optimized flow distribution [n_zones x n_shelters]
            zones: List of population zones
            shelters: List of shelters

        Returns:
            EvacuationPlan with actual paths
        """
        plan = EvacuationPlan(algorithm_type=AlgorithmType.HYBRID)

        populations = np.array([z.population for z in zones])
        capacities = np.array([s.capacity for s in shelters])

        # Calculate actual flows
        flows = flow_matrix * populations[:, np.newaxis]

        # Track shelter occupancy
        shelter_occupancy = {s.id: 0 for s in shelters}

        # Process each zone
        for i, zone in enumerate(zones):
            if self._should_stop:
                break

            # Get shelters this zone should send people to
            zone_flows = [(j, int(flows[i, j])) for j in range(len(shelters))]
            zone_flows = [(j, f) for j, f in zone_flows
                         if f >= self.config.min_flow_threshold]
            zone_flows.sort(key=lambda x: -x[1])  # Sort by flow descending

            for j, target_flow in zone_flows:
                shelter = shelters[j]

                # Check available capacity
                available = shelter.capacity - shelter_occupancy[shelter.id]
                actual_flow = min(target_flow, int(available))

                if actual_flow < self.config.min_flow_threshold:
                    continue

                # Find path using GBFS
                path, found_shelter, cost = self.gbfs.find_path(zone, [shelter])

                if path and found_shelter:
                    # Calculate route metrics
                    distance = self.gbfs._calculate_path_distance(path)
                    time_hours = self.gbfs._calculate_path_time(path)
                    risk = self.gbfs._calculate_path_risk(path)

                    route = EvacuationRoute(
                        zone_id=zone.id,
                        shelter_id=shelter.id,
                        path=path,
                        flow=actual_flow,
                        distance_km=distance,
                        estimated_time_hours=time_hours,
                        risk_score=risk
                    )
                    plan.add_route(route)
                    shelter_occupancy[shelter.id] += actual_flow

            # Report progress
            iteration = self.gwo_iterations + i + 1
            self._metrics.convergence_history.append(self._calculate_plan_cost(plan))
            self.report_progress(iteration, self._calculate_plan_cost(plan),
                               {'phase': 'gbfs', 'zone': zone.id})

        return plan

    def _refine_plan(self,
                     plan: EvacuationPlan,
                     flow_matrix: np.ndarray,
                     zones: List[PopulationZone],
                     shelters: List[Shelter]) -> EvacuationPlan:
        """
        Refine the evacuation plan by redistributing flows based on actual path costs.

        Args:
            plan: Current evacuation plan
            flow_matrix: Original flow matrix
            zones: Population zones
            shelters: Shelters

        Returns:
            Refined EvacuationPlan
        """
        if not plan.routes:
            return plan

        # Build cost matrix from actual paths
        cost_matrix = np.full((len(zones), len(shelters)), float('inf'))
        zone_to_idx = {z.id: i for i, z in enumerate(zones)}
        shelter_to_idx = {s.id: j for j, s in enumerate(shelters)}

        for route in plan.routes:
            i = zone_to_idx.get(route.zone_id)
            j = shelter_to_idx.get(route.shelter_id)
            if i is not None and j is not None:
                # Cost = time + risk penalty
                cost_matrix[i, j] = route.estimated_time_hours + 0.5 * route.risk_score

        # Refinement iterations
        current_plan = plan
        best_cost = self._calculate_plan_cost(current_plan)

        for ref_iter in range(self.refinement_iterations):
            if self._should_stop:
                break

            # Try to improve by redistributing from high-cost to low-cost routes
            improved_plan = self._try_redistribution(current_plan, cost_matrix,
                                                      zones, shelters)

            new_cost = self._calculate_plan_cost(improved_plan)

            if new_cost < best_cost:
                current_plan = improved_plan
                best_cost = new_cost

            # Report progress
            iteration = self.gwo_iterations + len(zones) + ref_iter + 1
            self._metrics.convergence_history.append(best_cost)
            self.report_progress(iteration, best_cost, {'phase': 'refinement', 'iter': ref_iter})

        return current_plan

    def _try_redistribution(self,
                            plan: EvacuationPlan,
                            cost_matrix: np.ndarray,
                            zones: List[PopulationZone],
                            shelters: List[Shelter]) -> EvacuationPlan:
        """
        Try to redistribute flows to improve overall cost.

        Args:
            plan: Current plan
            cost_matrix: Path cost matrix
            zones: Zones
            shelters: Shelters

        Returns:
            Potentially improved plan
        """
        # Find high-cost routes that might benefit from redistribution
        if not plan.routes:
            return plan

        # Calculate mean cost
        route_costs = [r.estimated_time_hours + 0.5 * r.risk_score for r in plan.routes]
        mean_cost = np.mean(route_costs) if route_costs else 0

        # Find routes above mean cost
        high_cost_routes = [
            (i, r) for i, r in enumerate(plan.routes)
            if r.estimated_time_hours + 0.5 * r.risk_score > mean_cost
        ]

        if not high_cost_routes:
            return plan

        # Try to find better alternatives for high-cost routes
        new_routes = list(plan.routes)
        shelter_loads = plan.get_shelter_loads()

        zone_to_idx = {z.id: i for i, z in enumerate(zones)}
        shelter_to_idx = {s.id: j for j, s in enumerate(shelters)}
        shelters_list = list(shelters)

        for route_idx, route in high_cost_routes:
            zone_idx = zone_to_idx.get(route.zone_id)
            if zone_idx is None:
                continue

            current_shelter_idx = shelter_to_idx.get(route.shelter_id)
            current_cost = cost_matrix[zone_idx, current_shelter_idx] if current_shelter_idx else float('inf')

            # Find alternative shelters with lower cost and available capacity
            for j, shelter in enumerate(shelters_list):
                if shelter.id == route.shelter_id:
                    continue

                alt_cost = cost_matrix[zone_idx, j]
                if alt_cost >= current_cost or alt_cost == float('inf'):
                    continue

                # Check capacity
                current_load = shelter_loads.get(shelter.id, 0)
                if current_load + route.flow > shelter.capacity:
                    continue

                # Found a better alternative - find new path
                zone = zones[zone_idx]
                path, found_shelter, _ = self.gbfs.find_path(zone, [shelter])

                if path and found_shelter:
                    # Update route
                    new_route = EvacuationRoute(
                        zone_id=route.zone_id,
                        shelter_id=shelter.id,
                        path=path,
                        flow=route.flow,
                        distance_km=self.gbfs._calculate_path_distance(path),
                        estimated_time_hours=self.gbfs._calculate_path_time(path),
                        risk_score=self.gbfs._calculate_path_risk(path)
                    )
                    new_routes[route_idx] = new_route

                    # Update shelter loads
                    shelter_loads[route.shelter_id] = shelter_loads.get(route.shelter_id, 0) - route.flow
                    shelter_loads[shelter.id] = shelter_loads.get(shelter.id, 0) + route.flow

                    break  # Move to next high-cost route

        # Create new plan
        improved_plan = EvacuationPlan(algorithm_type=AlgorithmType.HYBRID)
        for route in new_routes:
            improved_plan.add_route(route)

        return improved_plan

    def _calculate_plan_cost(self, plan: EvacuationPlan) -> float:
        """
        Calculate total cost of an evacuation plan.

        Args:
            plan: The evacuation plan

        Returns:
            Total cost value
        """
        if not plan.routes:
            return float('inf')

        total_cost = 0.0
        for route in plan.routes:
            # Weighted sum of time and risk
            route_cost = route.flow * (
                route.estimated_time_hours +
                0.3 * route.risk_score +
                0.001 * route.distance_km  # Small distance penalty
            )
            total_cost += route_cost

        return total_cost

    def get_component_metrics(self) -> Dict[str, AlgorithmMetrics]:
        """Get metrics from individual algorithm components."""
        return {
            'gwo': self.gwo.metrics,
            'gbfs': self.gbfs.metrics
        }
