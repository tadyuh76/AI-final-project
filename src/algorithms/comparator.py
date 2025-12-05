"""
Bộ so sánh thuật toán để đánh giá và so sánh các thuật toán sơ tán.

Cung cấp so sánh song song của các thuật toán GBFS và GWO
với các chỉ số chi tiết và dữ liệu trực quan hóa.
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
from ..models.network import EvacuationNetwork


@dataclass
class ComparisonResult:
    """Kết quả so sánh thuật toán."""
    algorithms: List[AlgorithmType] = field(default_factory=list)
    plans: Dict[AlgorithmType, EvacuationPlan] = field(default_factory=dict)
    metrics: Dict[AlgorithmType, AlgorithmMetrics] = field(default_factory=dict)
    rankings: Dict[str, List[AlgorithmType]] = field(default_factory=dict)
    winner: Optional[AlgorithmType] = None
    winner_score: float = 0.0

    def get_metric_comparison(self, metric_name: str) -> Dict[AlgorithmType, float]:
        """Lấy một chỉ số cụ thể qua tất cả các thuật toán."""
        result = {}
        for algo, m in self.metrics.items():
            value = getattr(m, metric_name, None)
            if value is not None:
                result[algo] = value
        return result

    def get_convergence_data(self) -> Dict[AlgorithmType, List[float]]:
        """Lấy lịch sử hội tụ cho tất cả các thuật toán."""
        return {
            algo: m.convergence_history
            for algo, m in self.metrics.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang từ điển để tuần tự hóa."""
        return {
            'algorithms': [a.value for a in self.algorithms],
            'metrics': {a.value: m.to_dict() for a, m in self.metrics.items()},
            'rankings': {k: [a.value for a in v] for k, v in self.rankings.items()},
            'winner': self.winner.value if self.winner else None,
            'winner_score': self.winner_score
        }

    def to_json(self) -> str:
        """Chuyển đổi sang chuỗi JSON."""
        return json.dumps(self.to_dict(), indent=2)


class AlgorithmComparator:
    """
    So sánh nhiều thuật toán tối ưu hóa sơ tán.

    Chạy từng thuật toán trên cùng một mạng lưới và so sánh kết quả
    qua nhiều chỉ số.
    """

    # Trọng số chỉ số cho tính điểm tổng thể (tổng là 1.0)
    METRIC_WEIGHTS = {
        'execution_time_seconds': 0.15,  # Thấp hơn là tốt hơn
        'final_cost': 0.25,              # Thấp hơn là tốt hơn
        'coverage_rate': 0.25,           # Cao hơn là tốt hơn
        'average_path_length': 0.10,     # Thấp hơn là tốt hơn (hiệu quả)
        'routes_found': 0.10,            # Cao hơn là tốt hơn
        'evacuees_covered': 0.15         # Cao hơn là tốt hơn
    }

    def __init__(self, network: EvacuationNetwork, config: Optional[AlgorithmConfig] = None):
        """
        Khởi tạo bộ so sánh.

        Args:
            network: Mạng lưới sơ tán để kiểm tra
            config: Cấu hình thuật toán chung
        """
        self.network = network
        self.config = config or AlgorithmConfig()
        self._progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable[[str, int, float], None]) -> None:
        """
        Đặt callback cho các cập nhật tiến trình.

        Args:
            callback: Function(algorithm_name, iteration, cost)
        """
        self._progress_callback = callback

    def compare_all(self) -> ComparisonResult:
        """
        Chạy cả hai thuật toán và so sánh kết quả.

        Returns:
            ComparisonResult với tất cả các chỉ số và xếp hạng
        """
        return self.compare([AlgorithmType.GBFS, AlgorithmType.GWO])

    def compare(self, algorithms: List[AlgorithmType]) -> ComparisonResult:
        """
        So sánh các thuật toán được chỉ định.

        Args:
            algorithms: Danh sách các loại thuật toán để so sánh

        Returns:
            ComparisonResult với các chỉ số và xếp hạng
        """
        result = ComparisonResult(algorithms=algorithms)

        for algo_type in algorithms:
            # Đặt lại trạng thái mạng lưới trước mỗi lần chạy
            self.network.reset_simulation_state()

            # Tạo và chạy thuật toán
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

        # Tính toán xếp hạng
        result.rankings = self._calculate_rankings(result.metrics)

        # Xác định người chiến thắng tổng thể
        result.winner, result.winner_score = self._calculate_winner(result.metrics)

        return result

    def _create_algorithm(self, algo_type: AlgorithmType) -> BaseAlgorithm:
        """Tạo một thể hiện thuật toán của loại được chỉ định."""
        if algo_type == AlgorithmType.GBFS:
            return GreedyBestFirstSearch(self.network, self.config)
        elif algo_type == AlgorithmType.GWO:
            return GreyWolfOptimizer(self.network, self.config)
        else:
            raise ValueError(f"Loại thuật toán không xác định: {algo_type}")

    def _calculate_rankings(self,
                           metrics: Dict[AlgorithmType, AlgorithmMetrics]) -> Dict[str, List[AlgorithmType]]:
        """
        Tính toán xếp hạng cho từng chỉ số.

        Args:
            metrics: Các chỉ số thuật toán

        Returns:
            Từ điển ánh xạ tên chỉ số tới danh sách thuật toán được xếp hạng
        """
        rankings = {}

        # Các chỉ số mà thấp hơn là tốt hơn
        lower_is_better = ['execution_time_seconds', 'final_cost', 'average_path_length']

        for metric_name in self.METRIC_WEIGHTS.keys():
            values = []
            for algo, m in metrics.items():
                value = getattr(m, metric_name, None)
                if value is not None:
                    values.append((algo, value))

            # Sắp xếp phù hợp
            if metric_name in lower_is_better:
                values.sort(key=lambda x: x[1])  # Tăng dần
            else:
                values.sort(key=lambda x: -x[1])  # Giảm dần

            rankings[metric_name] = [algo for algo, _ in values]

        return rankings

    def _calculate_winner(self,
                         metrics: Dict[AlgorithmType, AlgorithmMetrics]) -> Tuple[Optional[AlgorithmType], float]:
        """
        Tính toán người chiến thắng tổng thể dựa trên điểm số có trọng số.

        Args:
            metrics: Các chỉ số thuật toán

        Returns:
            Tuple của (loại thuật toán chiến thắng, điểm số)
        """
        if not metrics:
            return None, 0.0

        # Chuẩn hóa các chỉ số để so sánh công bằng
        normalized = self._normalize_metrics(metrics)

        # Tính toán điểm số có trọng số
        scores = {}
        for algo in metrics.keys():
            score = 0.0
            for metric_name, weight in self.METRIC_WEIGHTS.items():
                if algo in normalized.get(metric_name, {}):
                    score += weight * normalized[metric_name][algo]
            scores[algo] = score

        # Tìm người chiến thắng
        winner = max(scores.keys(), key=lambda a: scores[a])
        return winner, scores[winner]

    def _normalize_metrics(self,
                          metrics: Dict[AlgorithmType, AlgorithmMetrics]) -> Dict[str, Dict[AlgorithmType, float]]:
        """
        Chuẩn hóa các chỉ số về phạm vi 0-1 để so sánh công bằng.

        Args:
            metrics: Các chỉ số thuật toán thô

        Returns:
            Từ điển các chỉ số đã chuẩn hóa
        """
        normalized = {}
        lower_is_better = ['execution_time_seconds', 'final_cost', 'average_path_length']

        for metric_name in self.METRIC_WEIGHTS.keys():
            values = {}
            for algo, m in metrics.items():
                value = getattr(m, metric_name, None)
                if value is not None:
                    values[algo] = value

            if not values:
                continue

            # Chuẩn hóa về 0-1
            min_val = min(values.values())
            max_val = max(values.values())
            range_val = max_val - min_val if max_val != min_val else 1.0

            normalized[metric_name] = {}
            for algo, value in values.items():
                norm_value = (value - min_val) / range_val

                # Đảo ngược nếu thấp hơn là tốt hơn
                if metric_name in lower_is_better:
                    norm_value = 1.0 - norm_value

                normalized[metric_name][algo] = norm_value

        return normalized

    def benchmark(self, n_runs: int = 5) -> Dict[AlgorithmType, Dict[str, float]]:
        """
        Chạy nhiều lần so sánh và tính toán các chỉ số trung bình.

        Args:
            n_runs: Số lần chạy để tính trung bình

        Returns:
            Từ điển của thuật toán -> các chỉ số trung bình
        """
        all_metrics: Dict[AlgorithmType, List[AlgorithmMetrics]] = {
            AlgorithmType.GBFS: [],
            AlgorithmType.GWO: []
        }

        for run in range(n_runs):
            result = self.compare_all()
            for algo, metrics in result.metrics.items():
                all_metrics[algo].append(metrics)

        # Tính toán trung bình
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
        """Tính độ lệch chuẩn."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def generate_comparison_table(self, result: ComparisonResult) -> str:
        """
        Tạo bảng văn bản so sánh các chỉ số thuật toán.

        Args:
            result: Kết quả so sánh

        Returns:
            Chuỗi bảng đã định dạng
        """
        lines = []
        lines.append("=" * 70)
        lines.append("KẾT QUẢ SO SÁNH THUẬT TOÁN")
        lines.append("=" * 70)
        lines.append("")

        # Tiêu đề
        header = f"{'Chỉ số':<25} | {'GBFS':>12} | {'GWO':>12}"
        lines.append(header)
        lines.append("-" * 55)

        # Các chỉ số
        metrics_display = [
            ('Thời gian thực thi (s)', 'execution_time_seconds', '{:.3f}'),
            ('Chi phí cuối cùng', 'final_cost', '{:.2f}'),
            ('Tuyến đường tìm thấy', 'routes_found', '{:d}'),
            ('Người được sơ tán', 'evacuees_covered', '{:,d}'),
            ('Tỷ lệ bao phủ (%)', 'coverage_rate', '{:.1%}'),
            ('Độ dài đường TB', 'average_path_length', '{:.1f}'),
            ('Số lần lặp', 'iterations', '{:d}')
        ]

        for display_name, attr_name, fmt in metrics_display:
            values = []
            for algo in [AlgorithmType.GBFS, AlgorithmType.GWO]:
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

            line = f"{display_name:<25} | {values[0]:>12} | {values[1]:>12}"
            lines.append(line)

        lines.append("-" * 70)

        # Người chiến thắng
        if result.winner:
            lines.append(f"CHIẾN THẮNG: {result.winner.value.upper()} (Điểm số: {result.winner_score:.3f})")
        else:
            lines.append("CHIẾN THẮNG: N/A")

        lines.append("=" * 70)

        return "\n".join(lines)


def run_comparison(network: EvacuationNetwork,
                   config: Optional[AlgorithmConfig] = None,
                   verbose: bool = True) -> ComparisonResult:
    """
    Hàm tiện ích để chạy so sánh thuật toán.

    Args:
        network: Mạng lưới sơ tán
        config: Cấu hình thuật toán
        verbose: Có in kết quả hay không

    Returns:
        ComparisonResult
    """
    comparator = AlgorithmComparator(network, config)
    result = comparator.compare_all()

    if verbose:
        print(comparator.generate_comparison_table(result))

    return result
