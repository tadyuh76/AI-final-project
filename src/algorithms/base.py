"""
Base classes and interfaces for evacuation optimization algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Any
from enum import Enum
import time


class AlgorithmType(Enum):
    """Types of algorithms available."""
    GBFS = "gbfs"
    GWO = "gwo"
    HYBRID = "hybrid"


@dataclass
class EvacuationRoute:
    """Represents a single evacuation route."""
    zone_id: str
    shelter_id: str
    path: List[str]  # List of node IDs from zone to shelter
    flow: int  # Number of people assigned to this route
    distance_km: float = 0.0
    estimated_time_hours: float = 0.0
    risk_score: float = 0.0

    @property
    def path_length(self) -> int:
        """Number of nodes in path."""
        return len(self.path)


@dataclass
class EvacuationPlan:
    """Complete evacuation plan with all routes."""
    routes: List[EvacuationRoute] = field(default_factory=list)
    total_evacuees: int = 0
    total_time_hours: float = 0.0
    average_risk: float = 0.0
    algorithm_type: AlgorithmType = AlgorithmType.GBFS

    def add_route(self, route: EvacuationRoute) -> None:
        """Add a route to the plan."""
        self.routes.append(route)
        self.total_evacuees += route.flow
        # Update weighted averages
        if self.total_evacuees > 0:
            total_weighted_time = sum(r.estimated_time_hours * r.flow for r in self.routes)
            total_weighted_risk = sum(r.risk_score * r.flow for r in self.routes)
            self.total_time_hours = total_weighted_time / self.total_evacuees
            self.average_risk = total_weighted_risk / self.total_evacuees

    def get_routes_for_zone(self, zone_id: str) -> List[EvacuationRoute]:
        """Get all routes originating from a zone."""
        return [r for r in self.routes if r.zone_id == zone_id]

    def get_routes_to_shelter(self, shelter_id: str) -> List[EvacuationRoute]:
        """Get all routes going to a shelter."""
        return [r for r in self.routes if r.shelter_id == shelter_id]

    def get_shelter_loads(self) -> Dict[str, int]:
        """Get total flow to each shelter."""
        loads: Dict[str, int] = {}
        for route in self.routes:
            if route.shelter_id not in loads:
                loads[route.shelter_id] = 0
            loads[route.shelter_id] += route.flow
        return loads


@dataclass
class AlgorithmMetrics:
    """Performance metrics for algorithm execution."""
    algorithm_type: AlgorithmType
    execution_time_seconds: float = 0.0
    iterations: int = 0
    convergence_history: List[float] = field(default_factory=list)
    final_cost: float = 0.0
    routes_found: int = 0
    evacuees_covered: int = 0
    average_path_length: float = 0.0
    coverage_rate: float = 0.0  # % of population with routes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'algorithm': self.algorithm_type.value,
            'execution_time': self.execution_time_seconds,
            'iterations': self.iterations,
            'final_cost': self.final_cost,
            'routes_found': self.routes_found,
            'evacuees_covered': self.evacuees_covered,
            'average_path_length': self.average_path_length,
            'coverage_rate': self.coverage_rate,
            'convergence_history': self.convergence_history
        }


# Type alias for progress callback
ProgressCallback = Callable[[int, float, Optional[Any]], None]


class BaseAlgorithm(ABC):
    """Abstract base class for evacuation optimization algorithms."""

    def __init__(self, network: Any):
        """
        Initialize algorithm with network.

        Args:
            network: EvacuationNetwork instance
        """
        self.network = network
        self._metrics = AlgorithmMetrics(algorithm_type=self.algorithm_type)
        self._progress_callback: Optional[ProgressCallback] = None
        self._is_running = False
        self._should_stop = False

    @property
    @abstractmethod
    def algorithm_type(self) -> AlgorithmType:
        """Return the algorithm type."""
        pass

    @property
    def metrics(self) -> AlgorithmMetrics:
        """Get current metrics."""
        return self._metrics

    def set_progress_callback(self, callback: ProgressCallback) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def report_progress(self, iteration: int, cost: float, data: Any = None) -> None:
        """Report progress to callback if set."""
        if self._progress_callback:
            self._progress_callback(iteration, cost, data)

    def stop(self) -> None:
        """Request algorithm to stop."""
        self._should_stop = True

    def reset(self) -> None:
        """Reset algorithm state."""
        self._metrics = AlgorithmMetrics(algorithm_type=self.algorithm_type)
        self._is_running = False
        self._should_stop = False

    @abstractmethod
    def optimize(self, **kwargs) -> Tuple[EvacuationPlan, AlgorithmMetrics]:
        """
        Run the optimization algorithm.

        Returns:
            Tuple of (EvacuationPlan, AlgorithmMetrics)
        """
        pass

    def _start_timer(self) -> float:
        """Start execution timer."""
        self._is_running = True
        self._should_stop = False
        return time.time()

    def _stop_timer(self, start_time: float) -> None:
        """Stop execution timer and record time."""
        self._metrics.execution_time_seconds = time.time() - start_time
        self._is_running = False


@dataclass
class AlgorithmConfig:
    """Configuration for algorithm parameters."""
    # GBFS weights
    distance_weight: float = 0.4
    risk_weight: float = 0.3
    congestion_weight: float = 0.2
    capacity_weight: float = 0.1

    # GWO parameters
    n_wolves: int = 30
    max_iterations: int = 100
    a_initial: float = 2.0  # Exploration parameter

    # Hybrid parameters
    gwo_iterations: int = 50
    refinement_iterations: int = 20

    # General
    min_flow_threshold: int = 100  # Minimum flow to create a route

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'distance_weight': self.distance_weight,
            'risk_weight': self.risk_weight,
            'congestion_weight': self.congestion_weight,
            'capacity_weight': self.capacity_weight,
            'n_wolves': self.n_wolves,
            'max_iterations': self.max_iterations,
            'a_initial': self.a_initial,
            'gwo_iterations': self.gwo_iterations,
            'refinement_iterations': self.refinement_iterations,
            'min_flow_threshold': self.min_flow_threshold
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlgorithmConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
