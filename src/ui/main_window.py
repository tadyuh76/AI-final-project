"""
Cửa sổ chính của ứng dụng SafeRoute HCM.
Tích hợp tất cả các thành phần UI và xử lý logic ứng dụng.
"""

import sys
from typing import Optional, Dict, Any
from threading import Thread

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QSplitter, QStatusBar, QLabel,
    QMessageBox, QApplication, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QFont, QCloseEvent

from .styles import COLORS, MAIN_STYLESHEET, Sizes, hex_to_rgb
from .map_widget import MapWidget
from .control_panel import ControlPanel
from .dashboard import Dashboard
from .comparison_view import ComparisonView

from ..models.network import EvacuationNetwork
from ..algorithms.base import AlgorithmConfig, AlgorithmType, EvacuationPlan
from ..algorithms.gbfs import GreedyBestFirstSearch
from ..algorithms.gwo import GreyWolfOptimizer
from ..algorithms.hybrid import HybridGBFSGWO
from ..algorithms.comparator import AlgorithmComparator, ComparisonResult
from ..simulation.engine import SimulationEngine, SimulationConfig, SimulationMetrics
from ..data.hcm_data import HCM_DISTRICTS, HCM_SHELTERS, FLOOD_PRONE_AREAS
from ..data.osm_loader import OSMDataLoader, load_network


def hex_to_qcolor(hex_color: str, alpha: int = 255):
    """Chuyển đổi hex sang QColor."""
    from PyQt6.QtGui import QColor
    r, g, b = hex_to_rgb(hex_color)
    return QColor(r, g, b, alpha)


class OptimizationWorker(QThread):
    """Worker thread cho việc chạy thuật toán tối ưu."""

    # Signals
    progress_updated = pyqtSignal(str, int, float)  # algo, iteration, cost
    optimization_completed = pyqtSignal(object, object)  # plan, metrics
    comparison_completed = pyqtSignal(object)  # ComparisonResult
    error_occurred = pyqtSignal(str)

    def __init__(self, network: EvacuationNetwork, config: AlgorithmConfig,
                 algorithm_type: str = 'hybrid', compare_all: bool = False):
        super().__init__()
        self.network = network
        self.config = config
        self.algorithm_type = algorithm_type
        self.compare_all = compare_all
        self._should_stop = False

    def run(self):
        """Chạy thuật toán trong thread riêng."""
        try:
            if self.compare_all:
                self._run_comparison()
            else:
                self._run_single_algorithm()
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _run_single_algorithm(self):
        """Chạy một thuật toán duy nhất."""
        # Create algorithm
        if self.algorithm_type == 'gbfs':
            algorithm = GreedyBestFirstSearch(self.network, self.config)
            algo_type = AlgorithmType.GBFS
        elif self.algorithm_type == 'gwo':
            algorithm = GreyWolfOptimizer(self.network, self.config)
            algo_type = AlgorithmType.GWO
        else:
            algorithm = HybridGBFSGWO(self.network, self.config)
            algo_type = AlgorithmType.HYBRID

        # Set progress callback
        def progress_callback(iteration: int, cost: float, data: Any):
            if not self._should_stop:
                self.progress_updated.emit(self.algorithm_type, iteration, cost)

        algorithm.set_progress_callback(progress_callback)

        # Run optimization
        plan, metrics = algorithm.optimize()

        if not self._should_stop:
            self.optimization_completed.emit(plan, metrics)

    def _run_comparison(self):
        """Chạy so sánh tất cả các thuật toán."""
        comparator = AlgorithmComparator(self.network, self.config)

        def progress_callback(algo_name: str, iteration: int, cost: float):
            if not self._should_stop:
                self.progress_updated.emit(algo_name, iteration, cost)

        comparator.set_progress_callback(progress_callback)

        result = comparator.compare_all()

        if not self._should_stop:
            self.comparison_completed.emit(result)

    def stop(self):
        """Dừng worker."""
        self._should_stop = True


class SimulationWorker(QThread):
    """Worker thread cho việc chạy mô phỏng."""

    # Signals
    step_completed = pyqtSignal(dict)  # metrics dict
    simulation_completed = pyqtSignal(object)  # final metrics
    error_occurred = pyqtSignal(str)

    def __init__(self, engine: SimulationEngine, plan: EvacuationPlan):
        super().__init__()
        self.engine = engine
        self.plan = plan
        self._should_stop = False
        self._is_paused = False

    def run(self):
        """Chạy mô phỏng trong thread riêng."""
        try:
            self.engine.initialize(self.plan)

            while not self._should_stop and not self._is_completed():
                while self._is_paused and not self._should_stop:
                    self.msleep(100)

                if self._should_stop:
                    break

                metrics = self.engine.step()
                self.step_completed.emit(metrics.to_dict())
                self.msleep(50)  # ~20 FPS update

            if not self._should_stop:
                self.simulation_completed.emit(self.engine.metrics)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _is_completed(self) -> bool:
        """Kiểm tra xem mô phỏng đã hoàn thành chưa."""
        from ..simulation.engine import SimulationState
        return self.engine.state == SimulationState.COMPLETED

    def pause(self):
        """Tạm dừng mô phỏng."""
        self._is_paused = True

    def resume(self):
        """Tiếp tục mô phỏng."""
        self._is_paused = False

    def stop(self):
        """Dừng mô phỏng."""
        self._should_stop = True


class MainWindow(QMainWindow):
    """
    Cửa sổ chính của ứng dụng SafeRoute HCM.
    """

    def __init__(self):
        super().__init__()

        # State
        self._network: Optional[EvacuationNetwork] = None
        self._current_plan: Optional[EvacuationPlan] = None
        self._optimization_worker: Optional[OptimizationWorker] = None
        self._simulation_worker: Optional[SimulationWorker] = None
        self._simulation_engine: Optional[SimulationEngine] = None

        # Setup UI
        self._setup_window()
        self._setup_ui()
        self._connect_signals()

        # Load initial network
        self._load_network()

    def _setup_window(self):
        """Thiết lập thuộc tính cửa sổ."""
        self.setWindowTitle("SafeRoute HCM - Toi uu Hoa So tan Bao")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Apply stylesheet
        self.setStyleSheet(MAIN_STYLESHEET)

    def _setup_ui(self):
        """Thiết lập giao diện người dùng."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left: Control panel
        self.control_panel = ControlPanel()
        main_layout.addWidget(self.control_panel)

        # Right: Main content
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(Sizes.PADDING_SM, Sizes.PADDING_SM,
                                         Sizes.PADDING_SM, Sizes.PADDING_SM)
        content_layout.setSpacing(Sizes.PADDING_SM)

        # Tab widget for Map View and Comparison
        self.tab_widget = QTabWidget()

        # Tab 1: Map View
        map_tab = QWidget()
        map_layout = QVBoxLayout(map_tab)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(Sizes.PADDING_SM)

        # Map widget
        self.map_widget = MapWidget()
        map_layout.addWidget(self.map_widget, 3)

        # Dashboard
        self.dashboard = Dashboard()
        map_layout.addWidget(self.dashboard, 1)

        self.tab_widget.addTab(map_tab, "Ban do")

        # Tab 2: Comparison View
        self.comparison_view = ComparisonView()
        self.tab_widget.addTab(self.comparison_view, "So sanh Thuat toan")

        content_layout.addWidget(self.tab_widget)

        main_layout.addWidget(content_widget, 1)

        # Status bar
        self._setup_status_bar()

    def _setup_status_bar(self):
        """Thiết lập status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Status label
        self.status_label = QLabel("San sang")
        self.status_bar.addWidget(self.status_label)

        # Spacer
        self.status_bar.addWidget(QLabel(""), 1)

        # FPS label
        self.fps_label = QLabel("60 FPS")
        self.status_bar.addPermanentWidget(self.fps_label)

        # Algorithm label
        self.algo_label = QLabel("Hybrid GBFS+GWO")
        self.status_bar.addPermanentWidget(self.algo_label)

        # Iteration label
        self.iter_label = QLabel("Vong lap: 0")
        self.status_bar.addPermanentWidget(self.iter_label)

    def _connect_signals(self):
        """Kết nối các signals."""
        # Control panel signals
        self.control_panel.run_clicked.connect(self._on_run_clicked)
        self.control_panel.pause_clicked.connect(self._on_pause_clicked)
        self.control_panel.reset_clicked.connect(self._on_reset_clicked)
        self.control_panel.stop_clicked.connect(self._on_stop_clicked)
        self.control_panel.algorithm_changed.connect(self._on_algorithm_changed)
        self.control_panel.config_changed.connect(self._on_config_changed)

    def _load_network(self):
        """Tải mạng lưới từ dữ liệu."""
        self.status_label.setText("Dang tai mang luoi...")
        QApplication.processEvents()

        try:
            loader = OSMDataLoader()
            self._network = loader.load_hcm_network(use_cache=True)
            loader.add_default_hazards(self._network, typhoon_intensity=0.7)
            self.map_widget.set_network(self._network)

            stats = self._network.get_stats()
            self.status_label.setText(
                f"Mang luoi: {stats.total_nodes} nut, "
                f"{stats.total_edges} canh, "
                f"{stats.population_zones} khu vuc, "
                f"{stats.shelters} noi tru an"
            )

            # Initialize dashboard with shelter info
            active_shelters = len(self._network.get_active_shelters())
            total_shelters = stats.shelters
            total_capacity = stats.total_shelter_capacity
            self.dashboard.update_shelter_status(active_shelters, total_shelters, total_capacity)

        except Exception as e:
            QMessageBox.warning(self, "Loi", f"Khong the tai mang luoi: {e}")
            self.status_label.setText("Loi khi tai mang luoi")

    @pyqtSlot()
    def _on_run_clicked(self):
        """Xử lý khi nhấn nút Run."""
        if not self._network:
            QMessageBox.warning(self, "Loi", "Chua tai mang luoi")
            return

        # Get config
        config = self.control_panel.get_config()

        # Create algorithm config
        algo_config = AlgorithmConfig(
            distance_weight=config['weights']['distance'],
            risk_weight=config['weights']['risk'],
            congestion_weight=config['weights']['congestion'],
            capacity_weight=config['weights']['capacity'],
            n_wolves=config['n_wolves'],
            max_iterations=config['max_iterations']
        )

        # Reset simulation state
        self._network.reset_simulation_state()
        self.map_widget.clear_routes()
        self.dashboard.reset()
        self.comparison_view.clear()

        # Update UI state
        self.control_panel.set_running_state(True)
        self.status_label.setText("Dang chay thuat toan...")

        # Check if we should compare all or run single
        compare_all = self.tab_widget.currentIndex() == 1

        # Create and start worker
        self._optimization_worker = OptimizationWorker(
            self._network, algo_config,
            algorithm_type=config['algorithm'],
            compare_all=compare_all
        )
        self._optimization_worker.progress_updated.connect(self._on_optimization_progress)
        self._optimization_worker.optimization_completed.connect(self._on_optimization_completed)
        self._optimization_worker.comparison_completed.connect(self._on_comparison_completed)
        self._optimization_worker.error_occurred.connect(self._on_error)
        self._optimization_worker.start()

    @pyqtSlot()
    def _on_pause_clicked(self):
        """Xử lý khi nhấn nút Pause/Resume."""
        if self._simulation_worker:
            if self._simulation_worker._is_paused:
                self._simulation_worker.resume()
                self.control_panel.set_paused_state(False)
                self.status_label.setText("Tiep tuc mo phong...")
            else:
                self._simulation_worker.pause()
                self.control_panel.set_paused_state(True)
                self.status_label.setText("Tam dung mo phong")

    @pyqtSlot()
    def _on_reset_clicked(self):
        """Xử lý khi nhấn nút Reset."""
        self._stop_all_workers()

        if self._network:
            self._network.reset_simulation_state()

        self.map_widget.clear_routes()
        self.map_widget.stop_animation()
        self.dashboard.reset()
        self.comparison_view.clear()

        self.control_panel.set_running_state(False)
        self.status_label.setText("Da dat lai")

    @pyqtSlot()
    def _on_stop_clicked(self):
        """Xử lý khi nhấn nút Stop."""
        self._stop_all_workers()
        self.map_widget.stop_animation()
        self.control_panel.set_running_state(False)
        self.status_label.setText("Da dung")

    def _stop_all_workers(self):
        """Dừng tất cả các worker threads."""
        if self._optimization_worker:
            self._optimization_worker.stop()
            self._optimization_worker.wait()
            self._optimization_worker = None

        if self._simulation_worker:
            self._simulation_worker.stop()
            self._simulation_worker.wait()
            self._simulation_worker = None

    @pyqtSlot(str)
    def _on_algorithm_changed(self, algorithm: str):
        """Xử lý khi thuật toán thay đổi."""
        algo_names = {
            'gbfs': 'GBFS',
            'gwo': 'GWO',
            'hybrid': 'Hybrid GBFS+GWO'
        }
        self.algo_label.setText(algo_names.get(algorithm, algorithm))

    @pyqtSlot(dict)
    def _on_config_changed(self, config: Dict[str, Any]):
        """Xử lý khi cấu hình thay đổi."""
        pass  # Could update preview or recalculate

    @pyqtSlot(str, int, float)
    def _on_optimization_progress(self, algo: str, iteration: int, cost: float):
        """Xử lý cập nhật tiến trình thuật toán."""
        self.iter_label.setText(f"Vong lap: {iteration}")
        self.status_label.setText(f"{algo.upper()}: Vong {iteration}, Chi phi: {cost:.2f}")

        # Update comparison view if on that tab
        if self.tab_widget.currentIndex() == 1:
            self.comparison_view.add_convergence_point(algo, iteration, cost)

    @pyqtSlot(object, object)
    def _on_optimization_completed(self, plan: EvacuationPlan, metrics):
        """Xử lý khi thuật toán hoàn thành."""
        self._current_plan = plan

        # Draw routes on map
        for i, route in enumerate(plan.routes):
            route_id = f"route_{i}"
            self.map_widget.add_route(route_id, route.path, route.flow, route.risk_score)

        # Start simulation
        self._start_simulation(plan)

        self.status_label.setText(
            f"Hoan thanh: {len(plan.routes)} tuyen, "
            f"{plan.total_evacuees:,} nguoi"
        )

    @pyqtSlot(object)
    def _on_comparison_completed(self, result: ComparisonResult):
        """Xử lý khi so sánh hoàn thành."""
        # Prepare data for comparison view
        comparison_data = {
            'metrics': {},
            'convergence': {},
            'radar_data': {},
            'winner': result.winner.value if result.winner else '',
            'winner_score': result.winner_score
        }

        for algo, metrics in result.metrics.items():
            comparison_data['metrics'][algo.value] = metrics.to_dict()
            comparison_data['convergence'][algo.value] = metrics.convergence_history

            # Calculate radar data (normalized 0-1)
            # [Speed, Safety, Coverage, Balance, Efficiency]
            max_time = max(m.execution_time_seconds for m in result.metrics.values()) or 1
            max_cost = max(m.final_cost for m in result.metrics.values()) or 1

            comparison_data['radar_data'][algo.value] = [
                1 - (metrics.execution_time_seconds / max_time),  # Speed
                1 - (metrics.final_cost / max_cost),  # Safety/Quality
                metrics.coverage_rate,  # Coverage
                0.8,  # Balance (placeholder)
                1 - (metrics.average_path_length / 20 if metrics.average_path_length else 0.5)  # Efficiency
            ]

        self.comparison_view.update_comparison(comparison_data)

        # Show best plan on map
        if result.winner:
            best_plan = result.plans.get(result.winner)
            if best_plan:
                self._current_plan = best_plan
                for i, route in enumerate(best_plan.routes):
                    route_id = f"route_{i}"
                    self.map_widget.add_route(route_id, route.path, route.flow, route.risk_score)

        self.control_panel.set_completed_state()
        self.status_label.setText(
            f"So sanh hoan thanh! Chien thang: {result.winner.value if result.winner else 'N/A'}"
        )

    def _start_simulation(self, plan: EvacuationPlan):
        """Bắt đầu mô phỏng sơ tán."""
        if not self._network:
            return

        # Create simulation engine
        sim_config = SimulationConfig(
            time_step_minutes=5.0,
            speed_multiplier=self.control_panel.get_config().get('simulation_speed', 1.0)
        )
        self._simulation_engine = SimulationEngine(self._network, sim_config)

        # Create and start worker
        self._simulation_worker = SimulationWorker(self._simulation_engine, plan)
        self._simulation_worker.step_completed.connect(self._on_simulation_step)
        self._simulation_worker.simulation_completed.connect(self._on_simulation_completed)
        self._simulation_worker.error_occurred.connect(self._on_error)
        self._simulation_worker.start()

        # Start map animation
        self.map_widget.start_animation()

    @pyqtSlot(dict)
    def _on_simulation_step(self, metrics: Dict[str, Any]):
        """Xử lý cập nhật từng bước mô phỏng."""
        self.dashboard.update_metrics(metrics)

        # Update map visualization
        # (In a full implementation, we'd update zone progress and shelter occupancy here)

    @pyqtSlot(object)
    def _on_simulation_completed(self, metrics: SimulationMetrics):
        """Xử lý khi mô phỏng hoàn thành."""
        self.map_widget.stop_animation()
        self.control_panel.set_completed_state()
        self.status_label.setText(
            f"Mo phong hoan thanh! Da so tan: {metrics.total_evacuated:,} nguoi"
        )

    @pyqtSlot(str)
    def _on_error(self, error_msg: str):
        """Xử lý lỗi."""
        self.control_panel.set_running_state(False)
        self.status_label.setText(f"Loi: {error_msg}")
        QMessageBox.critical(self, "Loi", f"Da xay ra loi:\n{error_msg}")

    def closeEvent(self, event: QCloseEvent):
        """Xử lý khi đóng cửa sổ."""
        self._stop_all_workers()
        event.accept()


def run_app():
    """Khởi chạy ứng dụng."""
    app = QApplication(sys.argv)

    # Set application info
    app.setApplicationName("SafeRoute HCM")
    app.setOrganizationName("UEH AI Team")

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run event loop
    sys.exit(app.exec())


if __name__ == '__main__':
    run_app()
