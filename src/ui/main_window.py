"""
Cửa sổ chính của ứng dụng SafeRoute HCM.
Tích hợp tất cả các thành phần UI và xử lý logic ứng dụng.
"""

import sys
import random
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QStatusBar, QLabel,
    QMessageBox, QApplication
)
from PyQt6.QtCore import pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QCloseEvent

from .styles import MAIN_STYLESHEET, Sizes, hex_to_rgb
from .map_widget import MapWidget
from .control_panel import ControlPanel
from .dashboard import Dashboard
from .comparison_view import ComparisonView

from ..models.network import EvacuationNetwork
from ..algorithms.base import AlgorithmConfig, AlgorithmType, EvacuationPlan
from ..algorithms.gbfs import GreedyBestFirstSearch
from ..algorithms.gwo import GreyWolfOptimizer
from ..algorithms.comparator import AlgorithmComparator, ComparisonResult
from ..simulation.engine import SimulationEngine, SimulationConfig, SimulationMetrics
from ..data.hcm_data import HCM_BOUNDS
from ..models.node import HazardZone
from ..data.osm_loader import OSMDataLoader


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
                 algorithm_type: str = 'gbfs', compare_all: bool = False):
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
        if self.algorithm_type == 'gwo':
            algorithm = GreyWolfOptimizer(self.network, self.config)
            algo_type = AlgorithmType.GWO
        else:
            # Default to GBFS
            algorithm = GreedyBestFirstSearch(self.network, self.config)
            algo_type = AlgorithmType.GBFS

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
            self.engine.start()  # Set state to RUNNING

            while not self._should_stop and not self._is_completed():
                while self._is_paused and not self._should_stop:
                    self.msleep(100)

                if self._should_stop:
                    break

                metrics = self.engine.step()
                metrics_dict = metrics.to_dict()

                # Add zone evacuation progress (aggregate departed per zone)
                zone_evacuated = {}
                for route_id, state in self.engine.get_route_states().items():
                    zone_id = state.route.zone_id
                    if zone_id not in zone_evacuated:
                        zone_evacuated[zone_id] = 0
                    zone_evacuated[zone_id] += state.departed
                metrics_dict['zone_evacuated'] = zone_evacuated

                self.step_completed.emit(metrics_dict)
                self.msleep(10)  # Faster updates for quicker simulation

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
        self._population_scale: float = 0.5  # Mặc định 50%

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

        self.tab_widget.addTab(map_tab, "Bản đồ")

        # Tab 2: Comparison View
        self.comparison_view = ComparisonView()
        self.tab_widget.addTab(self.comparison_view, "So sánh Thuật toán")

        content_layout.addWidget(self.tab_widget)

        main_layout.addWidget(content_widget, 1)

        # Status bar
        self._setup_status_bar()

    def _setup_status_bar(self):
        """Thiết lập status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Status label
        self.status_label = QLabel("Sẵn sàng")
        self.status_bar.addWidget(self.status_label)

        # Spacer
        self.status_bar.addWidget(QLabel(""), 1)

        # FPS label
        self.fps_label = QLabel("60 FPS")
        self.status_bar.addPermanentWidget(self.fps_label)

        # Algorithm label
        self.algo_label = QLabel("GBFS")
        self.status_bar.addPermanentWidget(self.algo_label)

        # Iteration label
        self.iter_label = QLabel("Vòng lặp: 0")
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

        # Hazard zone configuration signals
        self.control_panel.hazard_add_mode_changed.connect(self._on_hazard_add_mode_changed)
        self.control_panel.hazard_zone_delete_requested.connect(self._on_hazard_zone_delete)
        self.control_panel.hazard_zones_clear_requested.connect(self._on_hazard_zones_clear)
        self.control_panel.hazard_zones_randomize_requested.connect(self._on_hazard_zones_randomize)

        # Map click signal for hazard placement
        self.map_widget.canvas.map_clicked_for_hazard.connect(self._on_map_clicked_for_hazard)

    def _load_network(self):
        """Tải mạng lưới từ dữ liệu."""
        self.status_label.setText("Đang tải mạng lưới...")
        QApplication.processEvents()

        try:
            loader = OSMDataLoader()
            self._network = loader.load_hcm_network(use_cache=True)
            loader.add_default_hazards(self._network, typhoon_intensity=0.7)
            self.map_widget.set_network(self._network)

            stats = self._network.get_stats()
            self.status_label.setText(
                f"Mạng lưới: {stats.total_nodes} nút, "
                f"{stats.total_edges} cạnh, "
                f"{stats.population_zones} khu vực, "
                f"{stats.shelters} nơi trú ẩn"
            )

            # Initialize dashboard with shelter info
            active_shelters = len(self._network.get_active_shelters())
            total_shelters = stats.shelters
            total_capacity = stats.total_shelter_capacity
            self.dashboard.update_shelter_status(active_shelters, total_shelters, total_capacity)

            # Initialize hazard zone list in control panel
            self.control_panel.update_hazard_zone_list(self._network.get_hazard_zones())

        except Exception as e:
            QMessageBox.warning(self, "Lỗi", f"Không thể tải mạng lưới: {e}")
            self.status_label.setText("Lỗi khi tải mạng lưới")

    @pyqtSlot()
    def _on_run_clicked(self):
        """Xử lý khi nhấn nút Run."""
        if not self._network:
            QMessageBox.warning(self, "Lỗi", "Chưa tải mạng lưới")
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
        self.status_label.setText("Đang chạy thuật toán...")

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
                self.map_widget.canvas.start_animation()  # Resume animation
                self.control_panel.set_paused_state(False)
                self.status_label.setText("Tiếp tục mô phỏng...")
            else:
                self._simulation_worker.pause()
                self.map_widget.canvas.stop_animation()  # Stop animation when paused
                self.control_panel.set_paused_state(True)
                self.status_label.setText("Tạm dừng mô phỏng")

    @pyqtSlot()
    def _on_reset_clicked(self):
        """Xử lý khi nhấn nút Reset."""
        self._stop_all_workers()

        if self._network:
            self._network.reset_simulation_state()

        self.map_widget.clear_routes()
        self.map_widget.stop_animation()
        self.map_widget.reset_all_visual_states()  # Đặt lại màu sắc shelters và zones
        self.dashboard.reset()
        self.comparison_view.clear()

        self.control_panel.set_running_state(False)
        self.status_label.setText("Đã đặt lại")

    @pyqtSlot()
    def _on_stop_clicked(self):
        """Xử lý khi nhấn nút Stop."""
        self._stop_all_workers()
        self.map_widget.stop_animation()
        self.control_panel.set_running_state(False)
        self.status_label.setText("Đã dừng")

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
        """Xử lý khi thuật toán thay đổi - dừng và reset."""
        # Stop all running workers
        self._stop_all_workers()

        # Reset map and simulation
        if self._network:
            self._network.reset_simulation_state()

        self.map_widget.clear_routes()
        self.map_widget.stop_animation()
        self.map_widget.reset_all_visual_states()  # Đặt lại màu sắc shelters và zones
        self.dashboard.reset()

        # Update UI state
        self.control_panel.set_running_state(False)

        # Update algorithm label
        algo_names = {
            'gbfs': 'GBFS',
            'gwo': 'GWO'
        }
        self.algo_label.setText(algo_names.get(algorithm, algorithm))
        self.status_label.setText(f"Đã chọn thuật toán: {algo_names.get(algorithm, algorithm)}")

    @pyqtSlot(dict)
    def _on_config_changed(self, config: Dict[str, Any]):
        """Xử lý khi cấu hình thay đổi."""
        # Cập nhật population percentage
        population_percent = config.get('population_percent', 50)
        self._update_population_scale(population_percent)

        # Cập nhật visibility settings cho map
        self._update_map_visibility(config)

        # Cập nhật simulation speed nếu đang chạy
        simulation_speed = config.get('simulation_speed', 1.0)
        if self._simulation_engine:
            self._simulation_engine.config.speed_multiplier = simulation_speed

    def _update_population_scale(self, percent: float):
        """Cập nhật tỷ lệ dân số cần sơ tán."""
        if not self._network:
            return

        # Lưu tỷ lệ để sử dụng khi chạy algorithm
        self._population_scale = percent / 100.0

        # Cập nhật dân số hiển thị trong các zone
        for zone in self._network.get_population_zones():
            # base_population được thiết lập tự động trong __post_init__
            zone.population = int(zone.base_population * self._population_scale)

        # Cập nhật dashboard hiển thị tổng dân số
        stats = self._network.get_stats()
        self.status_label.setText(
            f"Dân số sơ tán: {stats.total_population:,} ({percent:.0f}%)"
        )

    def _update_map_visibility(self, config: Dict[str, Any]):
        """Cập nhật visibility của các thành phần trên map."""
        show_particles = config.get('show_particles', True)
        show_routes = config.get('show_routes', True)
        show_hazards = config.get('show_hazards', True)
        show_all_roads = config.get('show_all_roads', False)

        # Cập nhật particles visibility
        for particle in self.map_widget.canvas._particles:
            particle.setVisible(show_particles)

        # Cập nhật routes visibility
        for route_item in self.map_widget.canvas._route_items.values():
            route_item.setVisible(show_routes)

        # Cập nhật hazards visibility
        for hazard_item in self.map_widget.canvas._hazard_items.values():
            hazard_item.setVisible(show_hazards)

        # Cập nhật hiển thị tất cả đường (cần vẽ lại)
        # Chỉ vẽ lại nếu giá trị thay đổi
        if not hasattr(self, '_last_show_all_roads'):
            self._last_show_all_roads = False
        if show_all_roads != self._last_show_all_roads:
            self._last_show_all_roads = show_all_roads
            # Hiển thị thông báo đang xử lý
            if show_all_roads:
                self.status_label.setText("Đang vẽ tất cả đường (195k+)...")
            self.map_widget.canvas.redraw_edges(show_all_roads)
            # Cập nhật scene để hiển thị thay đổi
            self.map_widget.canvas.viewport().update()
            if show_all_roads:
                self.status_label.setText(f"Đã vẽ tất cả đường ({len(self.map_widget.canvas._edge_items)} layers)")
            else:
                self.status_label.setText("Chỉ hiển thị đường chính")

    # ===== Hazard Zone Configuration Handlers =====

    @pyqtSlot(bool)
    def _on_hazard_add_mode_changed(self, enabled: bool):
        """Xử lý khi bật/tắt chế độ đặt vùng nguy hiểm."""
        self.map_widget.canvas.set_hazard_add_mode(enabled)
        if enabled:
            self.status_label.setText("Nhấn vào bản đồ để đặt vùng nguy hiểm")
        else:
            self.status_label.setText("Đã tắt chế độ đặt vùng nguy hiểm")

    @pyqtSlot(float, float)
    def _on_map_clicked_for_hazard(self, lat: float, lon: float):
        """Xử lý khi click vào bản đồ để đặt vùng nguy hiểm."""
        if not self._network:
            return

        # Lấy thông số từ control panel
        params = self.control_panel.get_new_zone_params()

        # Tạo vùng nguy hiểm mới (luôn là flood type)
        hazard = HazardZone(
            center_lat=lat,
            center_lon=lon,
            radius_km=params['radius_km'],
            risk_level=params['risk_level'],
            hazard_type='flood'
        )

        # Thêm vào network
        self._network.add_hazard_zone(hazard)

        # Cập nhật visualization
        self.map_widget.canvas.refresh_hazard_zones()

        # Cập nhật danh sách trong control panel
        self.control_panel.update_hazard_zone_list(self._network.get_hazard_zones())

        self.status_label.setText(f"Đã thêm vùng nguy hiểm tại ({lat:.4f}, {lon:.4f})")

    @pyqtSlot(int)
    def _on_hazard_zone_delete(self, index: int):
        """Xử lý khi xóa một vùng nguy hiểm."""
        if not self._network:
            return

        zones = self._network.get_hazard_zones()
        if 0 <= index < len(zones):
            self._network.remove_hazard_zone(index)
            self.map_widget.canvas.refresh_hazard_zones()
            self.control_panel.update_hazard_zone_list(self._network.get_hazard_zones())
            self.status_label.setText("Đã xóa vùng nguy hiểm")

    @pyqtSlot()
    def _on_hazard_zones_clear(self):
        """Xử lý khi xóa tất cả vùng nguy hiểm."""
        if not self._network:
            return

        self._network.clear_hazard_zones()
        self.map_widget.canvas.refresh_hazard_zones()
        self.control_panel.update_hazard_zone_list([])
        self.status_label.setText("Đã xóa tất cả vùng nguy hiểm")

    @pyqtSlot(dict)
    def _on_hazard_zones_randomize(self, params: dict):
        """Xử lý khi tạo ngẫu nhiên các vùng nguy hiểm."""
        if not self._network:
            return

        # Xóa các vùng hiện có
        self._network.clear_hazard_zones()

        # Tạo các vùng ngẫu nhiên
        count = params['count']
        for _ in range(count):
            lat = random.uniform(HCM_BOUNDS['south'], HCM_BOUNDS['north'])
            lon = random.uniform(HCM_BOUNDS['west'], HCM_BOUNDS['east'])
            radius = random.uniform(params['min_radius'], params['max_radius'])
            severity = random.uniform(params['min_severity'], params['max_severity'])

            hazard = HazardZone(
                center_lat=lat,
                center_lon=lon,
                radius_km=radius,
                risk_level=severity,
                hazard_type='flood'
            )
            self._network.add_hazard_zone(hazard)

        # Cập nhật visualization
        self.map_widget.canvas.refresh_hazard_zones()
        self.control_panel.update_hazard_zone_list(self._network.get_hazard_zones())
        self.status_label.setText(f"Đã tạo {count} vùng nguy hiểm ngẫu nhiên")

    @pyqtSlot(str, int, float)
    def _on_optimization_progress(self, algo: str, iteration: int, cost: float):
        """Xử lý cập nhật tiến trình thuật toán."""
        self.iter_label.setText(f"Vòng lặp: {iteration}")
        self.status_label.setText(f"{algo.upper()}: Vòng {iteration}, Chi phí: {cost:.2f}")

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
            f"Hoàn thành: {len(plan.routes)} tuyến, "
            f"{plan.total_evacuees:,} người"
        )

    @pyqtSlot(object)
    def _on_comparison_completed(self, result: ComparisonResult):
        """Xử lý khi so sánh hoàn thành."""
        # Chuẩn bị dữ liệu cho comparison view
        comparison_data = {
            'metrics': {},
            'convergence': {},
            'radar_data': {},
            'plans': {},
            'winner': result.winner.value if result.winner else '',
            'winner_score': result.winner_score
        }

        for algo, metrics in result.metrics.items():
            comparison_data['metrics'][algo.value] = metrics.to_dict()
            comparison_data['convergence'][algo.value] = metrics.convergence_history

            # Tính toán dữ liệu radar (chuẩn hóa 0-1)
            # [Tốc độ, An toàn, Bao phủ, Cân bằng, Hiệu quả]
            max_time = max(m.execution_time_seconds for m in result.metrics.values()) or 1
            max_cost = max(m.final_cost for m in result.metrics.values()) or 1

            comparison_data['radar_data'][algo.value] = [
                1 - (metrics.execution_time_seconds / max_time),  # Tốc độ
                1 - (metrics.final_cost / max_cost),  # An toàn/Chất lượng
                metrics.coverage_rate,  # Bao phủ
                0.8,  # Cân bằng (placeholder)
                1 - (metrics.average_path_length / 20 if metrics.average_path_length else 0.5)  # Hiệu quả
            ]

        # Cập nhật comparison view
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
            f"So sánh hoàn thành! Chiến thắng: {result.winner.value if result.winner else 'N/A'}"
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
        # Calculate remaining shelter capacity from network
        if self._network:
            shelter_arrivals = metrics.get('shelter_arrivals', {})
            total_capacity = 0
            total_arrivals = 0
            for shelter in self._network.get_active_shelters():
                total_capacity += shelter.capacity
                arrivals = shelter_arrivals.get(shelter.id, 0)
                total_arrivals += arrivals
                # Update shelter color on map based on occupancy
                if shelter.id in self.map_widget.canvas._shelter_items:
                    self.map_widget.canvas._shelter_items[shelter.id].update_occupancy(arrivals)
            metrics['remaining_shelter_capacity'] = max(0, total_capacity - total_arrivals)

            # Update zone evacuation progress on map
            zone_evacuated = metrics.get('zone_evacuated', {})
            for zone_id, evacuated in zone_evacuated.items():
                self.map_widget.update_zone_progress(zone_id, evacuated)

        self.dashboard.update_metrics(metrics)

    @pyqtSlot(object)
    def _on_simulation_completed(self, metrics: SimulationMetrics):
        """Xử lý khi mô phỏng hoàn thành."""
        self.map_widget.stop_animation()
        self.control_panel.set_completed_state()
        self.status_label.setText(
            f"Mô phỏng hoàn thành! Đã sơ tán: {metrics.total_evacuated:,} người"
        )

    @pyqtSlot(str)
    def _on_error(self, error_msg: str):
        """Xử lý lỗi."""
        self.control_panel.set_running_state(False)
        self.status_label.setText(f"Lỗi: {error_msg}")
        QMessageBox.critical(self, "Lỗi", f"Đã xảy ra lỗi:\n{error_msg}")

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
