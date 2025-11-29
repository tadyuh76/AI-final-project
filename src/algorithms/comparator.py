"""
Algorithm Comparator for benchmarking and comparing evacuation algorithms.

Provides side-by-side comparison of GBFS, GWO, and Hybrid algorithms
with detailed metrics and visualization data.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import time
import json

from .base import (
    BaseAlgorithm, AlgorithmType, AlgorithmConfig,
    EvacuationPlan, AlgorithmMetrics, ProgressCallback
)
from .gbfs import GreedyBestFirstSearch
from .gwo import GreyWolfOptimizer
from .hybrid import HybridGBFSGWO
from ..models.network import EvacuationNetwork


@dataclass
class ComparisonResult:
    """Results of algorithm comparison."""
    algorithms: List[AlgorithmType] = field(default_factory=list)
    plans: Dict[AlgorithmType, EvacuationPlan] = field(default_factory=dict)
    metrics: Dict[AlgorithmType, AlgorithmMetrics] = field(default_factory=dict)
    rankings: Dict[str, List[AlgorithmType]] = field(default_factory=dict)
    winner: Optional[AlgorithmType] = None
    winner_score: float = 0.0

    def get_metric_comparison(self, metric_name: str) -> Dict[AlgorithmType, float]:
        """Get a specific metric across all algorithms."""
        result = {}
        for algo, m in self.metrics.items():
            value = getattr(m, metric_name, None)
            if value is not None:
                result[algo] = value
        return result

    def get_convergence_data(self) -> Dict[AlgorithmType, List[float]]:
        """Get convergence history for all algorithms."""
        return {
            algo: m.convergence_history
            for algo, m in self.metrics.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'algorithms': [a.value for a in self.algorithms],
            'metrics': {a.value: m.to_dict() for a, m in self.metrics.items()},
            'rankings': {k: [a.value for a in v] for k, v in self.rankings.items()},
            'winner': self.winner.value if self.winner else None,
            'winner_score': self.winner_score
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class AlgorithmComparator:
    """
    Compares multiple evacuation optimization algorithms.

    Runs each algorithm on the same network and compares results
    across multiple metrics.
    """

    # Metric weights for overall scoring (sum to 1.0)
    METRIC_WEIGHTS = {
        'execution_time': 0.15,     # Lower is better
        'final_cost': 0.25,         # Lower is better
        'coverage_rate': 0.25,      # Higher is better
        'average_path_length': 0.10, # Lower is better (efficiency)
        'routes_found': 0.10,       # Higher is better
        'evacuees_covered': 0.15    # Higher is better
    }

    def __init__(self, network: EvacuationNetwork, config: Optional[AlgorithmConfig] = None):
        """
        Initialize comparator.

        Args:
            network: The evacuation network to test on
            config: Shared algorithm configuration
        """
        self.network = network
        self.config = config or AlgorithmConfig()
        self._progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable[[str, int, float], None]) -> None:
        """
        Set callback for progress updates.

        Args:
            callback: Function(algorithm_name, iteration, cost)
        """
        self._progress_callback = callback

    def compare_all(self) -> ComparisonResult:
        """
        Run all three algorithms and compare results.

        Returns:
            ComparisonResult with all metrics and rankings
        """
        return self.compare([AlgorithmType.GBFS, AlgorithmType.GWO, AlgorithmType.HYBRID])

    def compare(self, algorithms: List[AlgorithmType]) -> ComparisonResult:
        """
        Compare specified algorithms.

        Args:
            algorithms: List of algorithm types to compare

        Returns:
            ComparisonResult with metrics and rankings
        """
        result = ComparisonResult(algorithms=algorithms)

        for algo_type in algorithms:
            # Reset network state before each run
            self.network.reset_simulation_state()

            # Create and run algorithm
            algorithm = self._create_algorithm(algo_type)

            if self._progress_callback:
                def make_callback(atype: AlgorithmType):
                    def cb(iteration: int, cost: float, data: Any) -> None:
                        self._progress_callback(atype.value, iteration, cost)
                    return cb
                algorithm.set_progress_callback(make_callback(algo_type))

            plan, metrics = algorithm.optimize()

            result.plans[algo_type] = plan
            result.metrics[algo_type] = metrics

        # Calculate rankings
        result.rankings = self._calculate_rankings(result.metrics)

        # Determine overall winner
        result.winner, result.winner_score = self._calculate_winner(result.metrics)

        return result

    def _create_algorithm(self, algo_type: AlgorithmType) -> BaseAlgorithm:
        """Create an algorithm instance of the specified type."""
        if algo_type == AlgorithmType.GBFS:
            return GreedyBestFirstSearch(self.network, self.config)
        elif algo_type == AlgorithmType.GWO:
            return GreyWolfOptimizer(self.network, self.config)
        elif algo_type == AlgorithmType.HYBRID:
            return HybridGBFSGWO(self.network, self.config)
        else:
            raise ValueError(f"Unknown algorithm type: {algo_type}")

    def _calculate_rankings(self,
                           metrics: Dict[AlgorithmType, AlgorithmMetrics]) -> Dict[str, List[AlgorithmType]]:
        """
        Calculate rankings for each metric.

        Args:
            metrics: Algorithm metrics

        Returns:
            Dictionary mapping metric names to ranked algorithm lists
        """
        rankings = {}

        # Metrics where lower is better
        lower_is_better = ['execution_time', 'final_cost', 'average_path_length']

        for metric_name in self.METRIC_WEIGHTS.keys():
            values = []
            for algo, m in metrics.items():
                value = getattr(m, metric_name, None)
                if value is not None:
                    values.append((algo, value))

            # Sort appropriately
            if metric_name in lower_is_better:
                values.sort(key=lambda x: x[1])  # Ascending
            else:
                values.sort(key=lambda x: -x[1])  # Descending

            rankings[metric_name] = [algo for algo, _ in values]

        return rankings

    def _calculate_winner(self,
                         metrics: Dict[AlgorithmType, AlgorithmMetrics]) -> Tuple[Optional[AlgorithmType], float]:
        """
        Calculate overall winner based on weighted scores.

        Args:
            metrics: Algorithm metrics

        Returns:
            Tuple of (winner algorithm type, score)
        """
        if not metrics:
            return None, 0.0

        # Normalize metrics for fair comparison
        normalized = self._normalize_metrics(metrics)

        # Calculate weighted scores
        scores = {}
        for algo in metrics.keys():
            score = 0.0
            for metric_name, weight in self.METRIC_WEIGHTS.items():
                if algo in normalized.get(metric_name, {}):
                    score += weight * normalized[metric_name][algo]
            scores[algo] = score

        # Find winner
        winner = max(scores.keys(), key=lambda a: scores[a])
        return winner, scores[winner]

    def _normalize_metrics(self,
                          metrics: Dict[AlgorithmType, AlgorithmMetrics]) -> Dict[str, Dict[AlgorithmType, float]]:
        """
        Normalize metrics to 0-1 range for fair comparison.

        Args:
            metrics: Raw algorithm metrics

        Returns:
            Normalized metrics dictionary
        """
        normalized = {}
        lower_is_better = ['execution_time', 'final_cost', 'average_path_length']

        for metric_name in self.METRIC_WEIGHTS.keys():
            values = {}
            for algo, m in metrics.items():
                value = getattr(m, metric_name, None)
                if value is not None:
                    values[algo] = value

            if not values:
                continue

            # Normalize to 0-1
            min_val = min(values.values())
            max_val = max(values.values())
            range_val = max_val - min_val if max_val != min_val else 1.0

            normalized[metric_name] = {}
            for algo, value in values.items():
                norm_value = (value - min_val) / range_val

                # Invert if lower is better
                if metric_name in lower_is_better:
                    norm_value = 1.0 - norm_value

                normalized[metric_name][algo] = norm_value

        return normalized

    def benchmark(self, n_runs: int = 5) -> Dict[AlgorithmType, Dict[str, float]]:
        """
        Run multiple comparisons and calculate average metrics.

        Args:
            n_runs: Number of runs to average

        Returns:
            Dictionary of algorithm -> averaged metrics
        """
        all_metrics: Dict[AlgorithmType, List[AlgorithmMetrics]] = {
            AlgorithmType.GBFS: [],
            AlgorithmType.GWO: [],
            AlgorithmType.HYBRID: []
        }

        for run in range(n_runs):
            result = self.compare_all()
            for algo, metrics in result.metrics.items():
                all_metrics[algo].append(metrics)

        # Calculate averages
        averages = {}
        for algo, metrics_list in all_metrics.items():
            if not metrics_list:
                continue

            averages[algo] = {
                'avg_execution_time': sum(m.execution_time_seconds for m in metrics_list) / len(metrics_list),
                'avg_final_cost': sum(m.final_cost for m in metrics_list) / len(metrics_list),
                'avg_coverage_rate': sum(m.coverage_rate for m in metrics_list) / len(metrics_list),
                'avg_routes_found': sum(m.routes_found for m in metrics_list) / len(metrics_list),
                'std_execution_time': self._std([m.execution_time_seconds for m in metrics_list]),
                'std_final_cost': self._std([m.final_cost for m in metrics_list])
            }

        return averages

    @staticmethod
    def _std(values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def generate_comparison_table(self, result: ComparisonResult) -> str:
        """
        Generate a text table comparing algorithm metrics.

        Args:
            result: Comparison result

        Returns:
            Formatted table string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("ALGORITHM COMPARISON RESULTS")
        lines.append("=" * 70)
        lines.append("")

        # Header
        header = f"{'Metric':<25} | {'GBFS':>12} | {'GWO':>12} | {'Hybrid':>12}"
        lines.append(header)
        lines.append("-" * 70)

        # Metrics
        metrics_display = [
            ('Execution Time (s)', 'execution_time_seconds', '{:.3f}'),
            ('Final Cost', 'final_cost', '{:.2f}'),
            ('Routes Found', 'routes_found', '{:d}'),
            ('Evacuees Covered', 'evacuees_covered', '{:,d}'),
            ('Coverage Rate (%)', 'coverage_rate', '{:.1%}'),
            ('Avg Path Length', 'average_path_length', '{:.1f}'),
            ('Iterations', 'iterations', '{:d}')
        ]

        for display_name, attr_name, fmt in metrics_display:
            values = []
            for algo in [AlgorithmType.GBFS, AlgorithmType.GWO, AlgorithmType.HYBRID]:
                m = result.metrics.get(algo)
                if m:
                    val = getattr(m, attr_name, 0)
                    if isinstance(val, float) and '%' in fmt:
                        values.append(fmt.format(val))
                    elif isinstance(val, int):
                        values.append(fmt.format(val))
                    else:
                        values.append(fmt.format(val))
                else:
                    values.append("N/A")

            line = f"{display_name:<25} | {values[0]:>12} | {values[1]:>12} | {values[2]:>12}"
            lines.append(line)

        lines.append("-" * 70)

        # Winner
        if result.winner:
            lines.append(f"WINNER: {result.winner.value.upper()} (Score: {result.winner_score:.3f})")
        else:
            lines.append("WINNER: N/A")

        lines.append("=" * 70)

        return "\n".join(lines)


def run_comparison(network: EvacuationNetwork,
                   config: Optional[AlgorithmConfig] = None,
                   verbose: bool = True) -> ComparisonResult:
    """
    Convenience function to run algorithm comparison.

    Args:
        network: Evacuation network
        config: Algorithm configuration
        verbose: Whether to print results

    Returns:
        ComparisonResult
    """
    comparator = AlgorithmComparator(network, config)
    result = comparator.compare_all()

    if verbose:
        print(comparator.generate_comparison_table(result))

    return result
