"""
Grey Wolf Optimizer (GWO) Algorithm for evacuation flow distribution.

GWO is a metaheuristic inspired by grey wolf hunting behavior.
The algorithm optimizes the global flow distribution across the network.

Wolf hierarchy:
- Alpha (α): Best solution
- Beta (β): Second best solution
- Delta (δ): Third best solution
- Omega (ω): Rest of the population
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import time

from .base import (
    BaseAlgorithm, AlgorithmType, AlgorithmConfig,
    EvacuationPlan, EvacuationRoute, AlgorithmMetrics
)
from ..models.network import EvacuationNetwork
from ..models.node import PopulationZone, Shelter, haversine_distance


@dataclass
class Wolf:
    """Represents a wolf (solution) in the GWO algorithm."""
    position: np.ndarray  # Flow distribution matrix [n_zones x n_shelters]
    fitness: float = float('inf')

    def copy(self) -> 'Wolf':
        """Create a copy of this wolf."""
        return Wolf(position=self.position.copy(), fitness=self.fitness)


class GreyWolfOptimizer(BaseAlgorithm):
    """
    Grey Wolf Optimizer for evacuation flow distribution.

    Optimizes how population should be distributed from zones to shelters
    to minimize total evacuation time while respecting constraints.
    """

    def __init__(self, network: EvacuationNetwork, config: Optional[AlgorithmConfig] = None):
        """
        Initialize GWO algorithm.

        Args:
            network: The evacuation network
            config: Algorithm configuration (optional)
        """
        super().__init__(network)
        self.config = config or AlgorithmConfig()

        # GWO parameters
        self.n_wolves = self.config.n_wolves
        self.max_iterations = self.config.max_iterations
        self.a_initial = self.config.a_initial

        # Population and wolves
        self.wolves: List[Wolf] = []
        self.alpha: Optional[Wolf] = None
        self.beta: Optional[Wolf] = None
        self.delta: Optional[Wolf] = None

        # Problem dimensions (set during optimization)
        self.n_zones = 0
        self.n_shelters = 0
        self.zones: List[PopulationZone] = []
        self.shelters: List[Shelter] = []

        # Precomputed distances for fitness evaluation
        self._distance_matrix: Optional[np.ndarray] = None
        self._risk_matrix: Optional[np.ndarray] = None

    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.GWO

    def _initialize_problem(self) -> None:
        """Initialize problem dimensions and precompute matrices."""
        self.zones = self.network.get_population_zones()
        self.shelters = self.network.get_shelters()
        self.n_zones = len(self.zones)
        self.n_shelters = len(self.shelters)

        # Precompute distance matrix
        self._distance_matrix = np.zeros((self.n_zones, self.n_shelters))
        for i, zone in enumerate(self.zones):
            for j, shelter in enumerate(self.shelters):
                self._distance_matrix[i, j] = haversine_distance(
                    zone.lat, zone.lon, shelter.lat, shelter.lon
                )

        # Precompute risk matrix (based on path midpoint risk)
        self._risk_matrix = np.zeros((self.n_zones, self.n_shelters))
        for i, zone in enumerate(self.zones):
            for j, shelter in enumerate(self.shelters):
                mid_lat = (zone.lat + shelter.lat) / 2
                mid_lon = (zone.lon + shelter.lon) / 2
                self._risk_matrix[i, j] = self.network.get_total_risk_at(mid_lat, mid_lon)

    def _initialize_population(self) -> None:
        """Initialize wolf population with random solutions."""
        self.wolves = []

        for _ in range(self.n_wolves):
            # Random flow distribution
            position = np.random.rand(self.n_zones, self.n_shelters)
            # Normalize rows (each zone's flow sums to 1)
            row_sums = position.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            position = position / row_sums

            wolf = Wolf(position=position)
            wolf.fitness = self._calculate_fitness(wolf.position)
            self.wolves.append(wolf)

        # Sort and identify alpha, beta, delta
        self._update_hierarchy()

    def _update_hierarchy(self) -> None:
        """Update alpha, beta, delta wolves based on fitness."""
        sorted_wolves = sorted(self.wolves, key=lambda w: w.fitness)
        self.alpha = sorted_wolves[0].copy()
        self.beta = sorted_wolves[1].copy() if len(sorted_wolves) > 1 else self.alpha.copy()
        self.delta = sorted_wolves[2].copy() if len(sorted_wolves) > 2 else self.beta.copy()

    def _calculate_fitness(self, position: np.ndarray) -> float:
        """
        Calculate fitness (cost) of a solution.

        Lower fitness is better. Considers:
        - Total evacuation time (weighted by flow)
        - Shelter capacity violations
        - Risk exposure
        - Flow balance
        """
        # Get population array
        populations = np.array([z.population for z in self.zones])
        capacities = np.array([s.capacity for s in self.shelters])
        total_capacity = capacities.sum()

        # Limit populations to total shelter capacity for realistic optimization
        # Scale down proportionally if population exceeds capacity
        total_pop = populations.sum()
        if total_pop > total_capacity:
            scale_factor = total_capacity / total_pop
            effective_populations = populations * scale_factor
        else:
            effective_populations = populations

        # Calculate actual flows
        flows = position * effective_populations[:, np.newaxis]

        # 1. Time cost (flow * distance / speed) - normalized
        # Assume average speed of 30 km/h
        avg_speed = 30.0
        time_cost = np.sum(flows * self._distance_matrix / avg_speed) / 1000.0  # Normalize

        # 2. Risk cost - normalized
        risk_cost = np.sum(flows * self._risk_matrix) / 1000.0

        # 3. Capacity violation penalty - normalized
        shelter_loads = flows.sum(axis=0)
        capacity_violations = np.maximum(0, shelter_loads - capacities)
        capacity_penalty = np.sum(capacity_violations / (capacities + 1)) * 10  # Relative violation

        # 4. Flow balance penalty (prefer even distribution)
        if capacities.sum() > 0:
            utilization = shelter_loads / (capacities + 1)
            balance_penalty = np.std(utilization) * 5
        else:
            balance_penalty = 0

        # 5. Coverage penalty - use effective population
        total_flow = flows.sum()
        total_effective = effective_populations.sum()
        if total_effective > 0:
            coverage = total_flow / total_effective
            coverage_penalty = (1 - coverage) ** 2 * 10
        else:
            coverage_penalty = 0

        return time_cost + risk_cost + capacity_penalty + balance_penalty + coverage_penalty

    def _update_position(self, wolf: Wolf, a: float) -> None:
        """
        Update wolf position based on alpha, beta, delta.

        Args:
            wolf: Wolf to update
            a: Exploration parameter (decreases over iterations)
        """
        for i in range(self.n_zones):
            for j in range(self.n_shelters):
                # Calculate position updates from alpha, beta, delta
                # Alpha influence
                r1, r2 = np.random.rand(2)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * self.alpha.position[i, j] - wolf.position[i, j])
                X1 = self.alpha.position[i, j] - A1 * D_alpha

                # Beta influence
                r1, r2 = np.random.rand(2)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * self.beta.position[i, j] - wolf.position[i, j])
                X2 = self.beta.position[i, j] - A2 * D_beta

                # Delta influence
                r1, r2 = np.random.rand(2)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * self.delta.position[i, j] - wolf.position[i, j])
                X3 = self.delta.position[i, j] - A3 * D_delta

                # Average of three influences
                wolf.position[i, j] = (X1 + X2 + X3) / 3

        # Clip to valid range and normalize
        wolf.position = np.clip(wolf.position, 0, 1)
        row_sums = wolf.position.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        wolf.position = wolf.position / row_sums

        # Recalculate fitness
        wolf.fitness = self._calculate_fitness(wolf.position)

    def optimize(self, **kwargs) -> Tuple[EvacuationPlan, AlgorithmMetrics]:
        """
        Run GWO optimization.

        Returns:
            Tuple of (EvacuationPlan, AlgorithmMetrics)
        """
        start_time = self._start_timer()

        # Initialize problem
        self._initialize_problem()

        if self.n_zones == 0 or self.n_shelters == 0:
            self._stop_timer(start_time)
            return EvacuationPlan(algorithm_type=AlgorithmType.GWO), self._metrics

        # Initialize population
        self._initialize_population()

        # Main optimization loop
        for iteration in range(self.max_iterations):
            if self._should_stop:
                break

            # Linearly decrease a from a_initial to 0
            a = self.a_initial - iteration * (self.a_initial / self.max_iterations)

            # Update each wolf
            for wolf in self.wolves:
                self._update_position(wolf, a)

            # Update hierarchy
            self._update_hierarchy()

            # Record convergence
            self._metrics.convergence_history.append(self.alpha.fitness)

            # Report progress
            self.report_progress(iteration + 1, self.alpha.fitness, self.alpha.position)

        # Convert best solution to evacuation plan
        plan = self._convert_to_plan(self.alpha.position)

        # Finalize metrics
        self._stop_timer(start_time)
        self._metrics.iterations = len(self._metrics.convergence_history)
        self._metrics.final_cost = self.alpha.fitness
        self._metrics.routes_found = len(plan.routes)
        self._metrics.evacuees_covered = plan.total_evacuees

        # Coverage rate: evacuees covered / min(total_population, total_capacity)
        total_population = sum(z.population for z in self.zones)
        total_capacity = sum(s.capacity for s in self.shelters)
        max_possible = min(total_population, total_capacity)
        self._metrics.coverage_rate = (
            plan.total_evacuees / max_possible if max_possible > 0 else 0
        )

        return plan, self._metrics

    def _convert_to_plan(self, position: np.ndarray) -> EvacuationPlan:
        """
        Convert GWO solution (flow matrix) to EvacuationPlan.

        Args:
            position: Flow distribution matrix

        Returns:
            EvacuationPlan with routes
        """
        plan = EvacuationPlan(algorithm_type=AlgorithmType.GWO)

        populations = np.array([z.population for z in self.zones])
        capacities = np.array([s.capacity for s in self.shelters])

        # Calculate actual flows
        flows = position * populations[:, np.newaxis]

        # Track shelter occupancy
        shelter_occupancy = np.zeros(self.n_shelters)

        for i, zone in enumerate(self.zones):
            for j, shelter in enumerate(self.shelters):
                flow = int(flows[i, j])

                # Apply capacity constraint
                available = capacities[j] - shelter_occupancy[j]
                actual_flow = min(flow, int(available))

                if actual_flow >= self.config.min_flow_threshold:
                    # Create simple direct path (GWO doesn't do pathfinding)
                    # The actual path will be refined by hybrid algorithm or GBFS
                    path = [zone.id, shelter.id]

                    distance = self._distance_matrix[i, j]
                    time_hours = distance / 30.0  # Assume 30 km/h
                    risk = self._risk_matrix[i, j]

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
                    shelter_occupancy[j] += actual_flow

        return plan

    def get_flow_matrix(self) -> Optional[np.ndarray]:
        """Get the optimized flow distribution matrix."""
        if self.alpha:
            return self.alpha.position.copy()
        return None

    def get_best_fitness(self) -> float:
        """Get the best fitness value found."""
        if self.alpha:
            return self.alpha.fitness
        return float('inf')
