"""
View so sánh thuật toán với biểu đồ và bảng hiệu suất.
Sử dụng pyqtgraph cho biểu đồ thời gian thực.
Bao gồm các tính năng: biểu đồ hội tụ, radar, biểu đồ cột,
bản đồ so sánh tuyến đường, phân tích nơi trú ẩn, và xuất dữ liệu.
"""

import csv
from typing import Optional, Dict, List, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QSplitter, QTabWidget,
    QPushButton, QCheckBox, QFileDialog, QComboBox,
    QScrollArea, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QBrush

from .styles import COLORS, Sizes, hex_to_rgb

# Thử import pyqtgraph
try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    pg = None


def hex_to_qcolor(hex_color: str, alpha: int = 255) -> QColor:
    """Chuyển đổi hex sang QColor."""
    r, g, b = hex_to_rgb(hex_color)
    return QColor(r, g, b, alpha)


# =============================================================================
# BIỂU ĐỒ HỘI TỤ
# =============================================================================

class ConvergenceChart(QWidget):
    """Biểu đồ hiển thị quá trình hội tụ của các thuật toán."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(250)

        self._data: Dict[str, List[float]] = {}
        self._colors = {
            'gbfs': COLORS.success,
            'gwo': COLORS.purple,
            'hybrid': COLORS.cyan
        }

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if HAS_PYQTGRAPH:
            # Cấu hình pyqtgraph
            pg.setConfigOptions(
                background=hex_to_qcolor(COLORS.surface),
                foreground=hex_to_qcolor(COLORS.text),
                antialias=True
            )

            # Tạo plot widget
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setLabel('left', 'Chi phí')
            self.plot_widget.setLabel('bottom', 'Vòng lặp')
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_widget.setTitle('Biểu đồ Hội tụ')

            # Chú giải
            self.plot_widget.addLegend(offset=(60, 30))

            # Các item biểu đồ
            self._plot_items: Dict[str, Any] = {}
            self._annotation_items: List[Any] = []

            layout.addWidget(self.plot_widget)
        else:
            # Fallback: nhãn đơn giản
            label = QLabel("Cài đặt pyqtgraph để xem biểu đồ\npip install pyqtgraph")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(f"color: {COLORS.text_muted}; padding: 40px;")
            layout.addWidget(label)

    def set_data(self, algorithm: str, convergence_history: List[float]):
        """Thiết lập dữ liệu hội tụ cho một thuật toán."""
        self._data[algorithm] = convergence_history
        self._update_plot()

    def clear_data(self):
        """Xóa tất cả dữ liệu."""
        self._data.clear()
        if HAS_PYQTGRAPH:
            self.plot_widget.clear()
            self._plot_items.clear()
            self._annotation_items.clear()
            # Thêm lại legend sau khi clear
            self.plot_widget.addLegend(offset=(60, 30))

    def _update_plot(self):
        """Cập nhật biểu đồ với dữ liệu hiện tại."""
        if not HAS_PYQTGRAPH:
            return

        # Xóa các annotation cũ
        for item in self._annotation_items:
            self.plot_widget.removeItem(item)
        self._annotation_items.clear()

        for algo, data in self._data.items():
            if not data:
                continue

            color = self._colors.get(algo, COLORS.text)
            r, g, b = hex_to_rgb(color)
            pen = pg.mkPen(color=QColor(r, g, b), width=2)

            if algo in self._plot_items:
                # Cập nhật plot hiện có
                self._plot_items[algo].setData(range(len(data)), data)
            else:
                # Tạo plot mới
                self._plot_items[algo] = self.plot_widget.plot(
                    range(len(data)), data,
                    pen=pen, name=algo.upper()
                )

            # Thêm annotation cho điểm tốt nhất
            if data:
                min_idx = data.index(min(data))
                min_val = data[min_idx]

                # Thêm marker tại điểm tốt nhất
                scatter = pg.ScatterPlotItem(
                    [min_idx], [min_val],
                    pen=pg.mkPen(color=QColor(r, g, b), width=2),
                    brush=QColor(r, g, b),
                    size=10
                )
                self.plot_widget.addItem(scatter)
                self._annotation_items.append(scatter)

    def add_point(self, algorithm: str, iteration: int, cost: float):
        """Thêm điểm dữ liệu mới (cho cập nhật thời gian thực)."""
        if algorithm not in self._data:
            self._data[algorithm] = []

        self._data[algorithm].append(cost)
        self._update_plot()

    def export_image(self, filepath: str):
        """Xuất biểu đồ ra file hình ảnh."""
        if HAS_PYQTGRAPH and hasattr(self, 'plot_widget'):
            exporter = pg.exporters.ImageExporter(self.plot_widget.plotItem)
            exporter.export(filepath)


# =============================================================================
# BẢNG HIỆU SUẤT
# =============================================================================

class PerformanceTable(QTableWidget):
    """Bảng so sánh hiệu suất các thuật toán."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Thiết lập bảng
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(['Chỉ số', 'GBFS', 'GWO', 'Hybrid'])

        # Các hàng
        metrics = [
            'Thời gian (s)',
            'Chi phí',
            'Tuyến đường',
            'Người sơ tán',
            'Tỷ lệ bao phủ',
            'Độ dài TB',
            'Vòng lặp'
        ]
        self.setRowCount(len(metrics))
        for i, metric in enumerate(metrics):
            self.setItem(i, 0, QTableWidgetItem(metric))

        # Định dạng
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        # Lưu trữ dữ liệu metrics
        self._metrics_data: Dict[str, Dict[str, Any]] = {}

    def update_metrics(self, metrics: Dict[str, Dict[str, Any]]):
        """
        Cập nhật bảng với các chỉ số từ kết quả so sánh.

        Args:
            metrics: Dict với key là loại thuật toán ('gbfs', 'gwo', 'hybrid')
                     và value là dict chứa các chỉ số
        """
        self._metrics_data = metrics
        algo_columns = {'gbfs': 1, 'gwo': 2, 'hybrid': 3}
        metric_rows = {
            'execution_time_seconds': 0,
            'final_cost': 1,
            'routes_found': 2,
            'evacuees_covered': 3,
            'coverage_rate': 4,
            'average_path_length': 5,
            'iterations': 6
        }

        for algo, data in metrics.items():
            col = algo_columns.get(algo)
            if col is None:
                continue

            for metric_name, row in metric_rows.items():
                value = data.get(metric_name, 0)

                # Định dạng giá trị
                if metric_name == 'execution_time_seconds':
                    text = f"{value:.3f}"
                elif metric_name == 'final_cost':
                    text = f"{value:.2f}"
                elif metric_name == 'coverage_rate':
                    text = f"{value:.1%}"
                elif metric_name == 'average_path_length':
                    text = f"{value:.1f}"
                elif isinstance(value, int):
                    text = f"{value:,}"
                else:
                    text = str(value)

                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.setItem(row, col, item)

    def highlight_winner(self, winner: str):
        """Đánh dấu cột thuật toán chiến thắng."""
        algo_columns = {'gbfs': 1, 'gwo': 2, 'hybrid': 3}
        winner_col = algo_columns.get(winner)

        if winner_col is None:
            return

        highlight_color = hex_to_qcolor(COLORS.success, 50)

        for row in range(self.rowCount()):
            item = self.item(row, winner_col)
            if item:
                item.setBackground(QBrush(highlight_color))

    def get_metrics_data(self) -> Dict[str, Dict[str, Any]]:
        """Trả về dữ liệu metrics đã lưu."""
        return self._metrics_data


# =============================================================================
# BIỂU ĐỒ RADAR
# =============================================================================

class RadarChart(QWidget):
    """Biểu đồ radar cho so sánh đa mục tiêu."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)

        self._data: Dict[str, List[float]] = {}
        self._labels = ['Tốc độ', 'An toàn', 'Bao phủ', 'Cân bằng', 'Hiệu quả']
        self._colors = {
            'gbfs': COLORS.success,
            'gwo': COLORS.purple,
            'hybrid': COLORS.cyan
        }

    def set_data(self, algorithm: str, values: List[float]):
        """
        Thiết lập dữ liệu cho thuật toán.

        Args:
            algorithm: Loại thuật toán
            values: Danh sách 5 giá trị (0-1) cho mỗi chiều
        """
        self._data[algorithm] = values
        self.update()

    def clear_data(self):
        """Xóa tất cả dữ liệu."""
        self._data.clear()
        self.update()

    def paintEvent(self, event):
        """Vẽ biểu đồ radar."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Nền
        painter.fillRect(self.rect(), hex_to_qcolor(COLORS.surface))

        # Tính toán tâm và bán kính
        center_x = self.width() // 2
        center_y = self.height() // 2
        radius = min(center_x, center_y) - 40

        if radius < 50:
            return

        import math

        num_axes = len(self._labels)
        angle_step = 2 * math.pi / num_axes

        # Vẽ lưới
        painter.setPen(QPen(hex_to_qcolor(COLORS.border), 1))

        for level in [0.25, 0.5, 0.75, 1.0]:
            points = []
            for i in range(num_axes):
                angle = i * angle_step - math.pi / 2
                x = center_x + radius * level * math.cos(angle)
                y = center_y + radius * level * math.sin(angle)
                points.append((x, y))

            for i in range(num_axes):
                next_i = (i + 1) % num_axes
                painter.drawLine(
                    int(points[i][0]), int(points[i][1]),
                    int(points[next_i][0]), int(points[next_i][1])
                )

        # Vẽ các trục
        for i in range(num_axes):
            angle = i * angle_step - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            painter.drawLine(center_x, center_y, int(x), int(y))

            # Nhãn
            label_x = center_x + (radius + 20) * math.cos(angle)
            label_y = center_y + (radius + 20) * math.sin(angle)
            painter.setPen(QPen(hex_to_qcolor(COLORS.text)))
            painter.drawText(int(label_x - 30), int(label_y - 5), 60, 20,
                            Qt.AlignmentFlag.AlignCenter, self._labels[i])
            painter.setPen(QPen(hex_to_qcolor(COLORS.border), 1))

        # Vẽ các đa giác dữ liệu
        for algo, values in self._data.items():
            if len(values) != num_axes:
                continue

            color = self._colors.get(algo, COLORS.text)
            r, g, b = hex_to_rgb(color)

            # Tô màu
            fill_color = QColor(r, g, b, 50)
            painter.setBrush(QBrush(fill_color))
            painter.setPen(QPen(QColor(r, g, b), 2))

            points = []
            from PyQt6.QtGui import QPolygon
            from PyQt6.QtCore import QPoint

            for i in range(num_axes):
                angle = i * angle_step - math.pi / 2
                value = min(1.0, max(0.0, values[i]))
                x = center_x + radius * value * math.cos(angle)
                y = center_y + radius * value * math.sin(angle)
                points.append(QPoint(int(x), int(y)))

            polygon = QPolygon(points)
            painter.drawPolygon(polygon)

        painter.end()


# =============================================================================
# BIỂU ĐỒ CỘT SO SÁNH CHỈ SỐ
# =============================================================================

class MetricBarChart(QWidget):
    """Biểu đồ cột so sánh các chỉ số giữa các thuật toán."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self._metrics_data: Dict[str, Dict[str, Any]] = {}
        self._current_metric = 'final_cost'
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Sizes.PADDING_SM)

        # Bộ chọn chỉ số
        selector_layout = QHBoxLayout()
        selector_label = QLabel("Chỉ số:")
        selector_label.setProperty("muted", True)
        selector_layout.addWidget(selector_label)

        self.metric_selector = QComboBox()
        self.metric_selector.addItems([
            'Chi phí cuối',
            'Thời gian (s)',
            'Số tuyến đường',
            'Người sơ tán',
            'Tỷ lệ bao phủ',
            'Độ dài TB'
        ])
        self.metric_selector.currentIndexChanged.connect(self._on_metric_changed)
        selector_layout.addWidget(self.metric_selector)
        selector_layout.addStretch()
        layout.addLayout(selector_layout)

        if HAS_PYQTGRAPH:
            # Cấu hình pyqtgraph
            pg.setConfigOptions(
                background=hex_to_qcolor(COLORS.surface),
                foreground=hex_to_qcolor(COLORS.text),
                antialias=True
            )

            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setTitle("So sánh chỉ số")
            self.plot_widget.showGrid(x=False, y=True, alpha=0.3)
            layout.addWidget(self.plot_widget)
        else:
            label = QLabel("Cài đặt pyqtgraph để xem biểu đồ")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(f"color: {COLORS.text_muted}; padding: 40px;")
            layout.addWidget(label)

    def _on_metric_changed(self, index: int):
        """Xử lý khi thay đổi chỉ số được chọn."""
        metric_map = {
            0: 'final_cost',
            1: 'execution_time_seconds',
            2: 'routes_found',
            3: 'evacuees_covered',
            4: 'coverage_rate',
            5: 'average_path_length'
        }
        self._current_metric = metric_map.get(index, 'final_cost')
        self._update_chart()

    def set_data(self, metrics: Dict[str, Dict[str, Any]]):
        """Thiết lập dữ liệu metrics."""
        self._metrics_data = metrics
        self._update_chart()

    def _update_chart(self):
        """Cập nhật biểu đồ cột."""
        if not HAS_PYQTGRAPH or not hasattr(self, 'plot_widget'):
            return

        self.plot_widget.clear()

        if not self._metrics_data:
            return

        algorithms = ['gbfs', 'gwo', 'hybrid']
        colors = [
            hex_to_qcolor(COLORS.success),
            hex_to_qcolor(COLORS.purple),
            hex_to_qcolor(COLORS.cyan)
        ]

        values = []
        for algo in algorithms:
            data = self._metrics_data.get(algo, {})
            val = data.get(self._current_metric, 0)
            if isinstance(val, (int, float)):
                values.append(val)
            else:
                values.append(0)

        # Tạo biểu đồ cột
        x = list(range(len(algorithms)))
        for i, (xi, val) in enumerate(zip(x, values)):
            bar = pg.BarGraphItem(
                x=[xi], height=[val], width=0.6,
                brush=colors[i % len(colors)]
            )
            self.plot_widget.addItem(bar)

        # Thiết lập nhãn trục x
        axis = self.plot_widget.getAxis('bottom')
        axis.setTicks([[(i, algo.upper()) for i, algo in enumerate(algorithms)]])

    def clear_data(self):
        """Xóa tất cả dữ liệu."""
        self._metrics_data.clear()
        if HAS_PYQTGRAPH and hasattr(self, 'plot_widget'):
            self.plot_widget.clear()


# =============================================================================
# PANEL THỐNG KÊ BENCHMARK
# =============================================================================

class StatisticalSummaryPanel(QFrame):
    """Panel hiển thị thống kê từ benchmark runs."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("card", True)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Sizes.PADDING_MD, Sizes.PADDING_MD,
                                  Sizes.PADDING_MD, Sizes.PADDING_MD)

        # Tiêu đề
        title = QLabel("THỐNG KÊ SO SÁNH")
        title.setProperty("subheading", True)
        layout.addWidget(title)

        # Grid cho thống kê
        self.stats_grid = QGridLayout()
        self.stats_grid.setSpacing(Sizes.PADDING_SM)

        # Headers
        headers = ["Thuật toán", "Thời gian", "Chi phí", "Bao phủ", "Xếp hạng"]
        for col, header in enumerate(headers):
            label = QLabel(header)
            label.setProperty("muted", True)
            label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.stats_grid.addWidget(label, 0, col)

        # Thêm các hàng cho từng thuật toán
        self._algo_labels: Dict[str, Dict[str, QLabel]] = {}
        algos = ['gbfs', 'gwo', 'hybrid']
        for row, algo in enumerate(algos, 1):
            self._algo_labels[algo] = {}

            # Tên thuật toán
            name_label = QLabel(algo.upper())
            name_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            color = {
                'gbfs': COLORS.success,
                'gwo': COLORS.purple,
                'hybrid': COLORS.cyan
            }.get(algo, COLORS.text)
            name_label.setStyleSheet(f"color: {color};")
            self.stats_grid.addWidget(name_label, row, 0)

            # Các cột dữ liệu
            for col, key in enumerate(['time', 'cost', 'coverage', 'rank'], 1):
                label = QLabel("--")
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.stats_grid.addWidget(label, row, col)
                self._algo_labels[algo][key] = label

        layout.addLayout(self.stats_grid)

    def update_stats(self, metrics: Dict[str, Dict[str, Any]], winner: str = ""):
        """
        Cập nhật thống kê từ kết quả so sánh.

        Args:
            metrics: Dict các metrics theo thuật toán
            winner: Tên thuật toán chiến thắng
        """
        # Tính toán xếp hạng dựa trên chi phí (thấp hơn tốt hơn)
        costs = [(algo, data.get('final_cost', float('inf')))
                 for algo, data in metrics.items()]
        costs.sort(key=lambda x: x[1])
        rankings = {algo: rank + 1 for rank, (algo, _) in enumerate(costs)}

        for algo, data in metrics.items():
            if algo not in self._algo_labels:
                continue

            labels = self._algo_labels[algo]

            # Thời gian
            time_val = data.get('execution_time_seconds', 0)
            labels['time'].setText(f"{time_val:.3f}s")

            # Chi phí
            cost_val = data.get('final_cost', 0)
            labels['cost'].setText(f"{cost_val:.2f}")

            # Bao phủ
            coverage_val = data.get('coverage_rate', 0)
            labels['coverage'].setText(f"{coverage_val:.1%}")

            # Xếp hạng
            rank = rankings.get(algo, 0)
            rank_text = f"#{rank}"
            if algo == winner:
                rank_text = f"🏆 #{rank}"
            labels['rank'].setText(rank_text)


# =============================================================================
# BẢN ĐỒ SO SÁNH TUYẾN ĐƯỜNG
# =============================================================================

class RouteComparisonMap(QWidget):
    """Bản đồ so sánh tuyến đường giữa các thuật toán."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._network = None
        self._plans: Dict[str, Any] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Sizes.PADDING_SM)

        # Panel điều khiển
        controls = QHBoxLayout()
        controls.setSpacing(Sizes.PADDING_MD)

        controls_label = QLabel("Hiện/Ẩn tuyến đường:")
        controls_label.setProperty("muted", True)
        controls.addWidget(controls_label)

        # Checkboxes để toggle thuật toán
        self.gbfs_check = QCheckBox("GBFS")
        self.gbfs_check.setChecked(True)
        self.gbfs_check.setStyleSheet(f"color: {COLORS.success};")
        self.gbfs_check.stateChanged.connect(self._update_display)
        controls.addWidget(self.gbfs_check)

        self.gwo_check = QCheckBox("GWO")
        self.gwo_check.setChecked(True)
        self.gwo_check.setStyleSheet(f"color: {COLORS.purple};")
        self.gwo_check.stateChanged.connect(self._update_display)
        controls.addWidget(self.gwo_check)

        self.hybrid_check = QCheckBox("Hybrid")
        self.hybrid_check.setChecked(True)
        self.hybrid_check.setStyleSheet(f"color: {COLORS.cyan};")
        self.hybrid_check.stateChanged.connect(self._update_display)
        controls.addWidget(self.hybrid_check)

        controls.addStretch()
        layout.addLayout(controls)

        # Placeholder cho bản đồ
        self.map_placeholder = QFrame()
        self.map_placeholder.setProperty("card", True)
        self.map_placeholder.setMinimumHeight(300)

        placeholder_layout = QVBoxLayout(self.map_placeholder)
        placeholder_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Thông tin placeholder
        info_label = QLabel("🗺️ BẢN ĐỒ SO SÁNH TUYẾN ĐƯỜNG")
        info_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_layout.addWidget(info_label)

        self.status_label = QLabel("Chạy so sánh để xem tuyến đường")
        self.status_label.setProperty("muted", True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_layout.addWidget(self.status_label)

        # Thống kê tuyến đường
        self.route_stats = QLabel("")
        self.route_stats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_layout.addWidget(self.route_stats)

        layout.addWidget(self.map_placeholder)

    def set_network(self, network):
        """Thiết lập mạng lưới."""
        self._network = network

    def set_plans(self, plans: Dict[str, Any]):
        """Thiết lập các kế hoạch sơ tán."""
        self._plans = plans
        self._update_display()

    def _update_display(self):
        """Cập nhật hiển thị tuyến đường."""
        if not self._plans:
            self.status_label.setText("Chạy so sánh để xem tuyến đường")
            self.route_stats.setText("")
            return

        # Đếm số tuyến đường
        algo_checks = {
            'gbfs': self.gbfs_check,
            'gwo': self.gwo_check,
            'hybrid': self.hybrid_check
        }

        stats_lines = []
        total_routes = 0

        for algo, plan in self._plans.items():
            algo_key = algo.lower() if isinstance(algo, str) else algo.value.lower()
            checkbox = algo_checks.get(algo_key)

            if checkbox and checkbox.isChecked() and plan:
                num_routes = len(plan.routes) if hasattr(plan, 'routes') else 0
                total_evacuees = plan.total_evacuees if hasattr(plan, 'total_evacuees') else 0
                total_routes += num_routes

                color = {
                    'gbfs': COLORS.success,
                    'gwo': COLORS.purple,
                    'hybrid': COLORS.cyan
                }.get(algo_key, COLORS.text)

                stats_lines.append(
                    f"<span style='color:{color}'>{algo_key.upper()}</span>: "
                    f"{num_routes} tuyến, {total_evacuees:,} người"
                )

        if stats_lines:
            self.status_label.setText(f"Tổng: {total_routes} tuyến đường đang hiển thị")
            self.route_stats.setText("<br>".join(stats_lines))
        else:
            self.status_label.setText("Không có tuyến đường được chọn")
            self.route_stats.setText("")

    def clear(self):
        """Xóa tất cả dữ liệu."""
        self._plans.clear()
        self.status_label.setText("Chạy so sánh để xem tuyến đường")
        self.route_stats.setText("")


# =============================================================================
# BIỂU ĐỒ PHÂN TÍCH NƠI TRÚ ẨN
# =============================================================================

class ShelterLoadChart(QWidget):
    """Biểu đồ phân tích tải trọng nơi trú ẩn."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(250)
        self._plans: Dict[str, Any] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if HAS_PYQTGRAPH:
            # Cấu hình pyqtgraph
            pg.setConfigOptions(
                background=hex_to_qcolor(COLORS.surface),
                foreground=hex_to_qcolor(COLORS.text),
                antialias=True
            )

            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setTitle("Phân bố tải trọng nơi trú ẩn")
            self.plot_widget.setLabel('left', 'Số người')
            self.plot_widget.setLabel('bottom', 'Nơi trú ẩn')
            self.plot_widget.showGrid(x=False, y=True, alpha=0.3)
            self.plot_widget.addLegend(offset=(60, 30))

            layout.addWidget(self.plot_widget)
        else:
            label = QLabel("Cài đặt pyqtgraph để xem biểu đồ")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(f"color: {COLORS.text_muted}; padding: 40px;")
            layout.addWidget(label)

    def set_data(self, plans: Dict[str, Any]):
        """Thiết lập dữ liệu từ các kế hoạch sơ tán."""
        self._plans = plans
        self._update_chart()

    def _update_chart(self):
        """Cập nhật biểu đồ tải trọng nơi trú ẩn."""
        if not HAS_PYQTGRAPH or not hasattr(self, 'plot_widget'):
            return

        self.plot_widget.clear()
        self.plot_widget.addLegend(offset=(60, 30))

        if not self._plans:
            return

        colors = {
            'gbfs': COLORS.success,
            'gwo': COLORS.purple,
            'hybrid': COLORS.cyan
        }

        # Thu thập tải trọng nơi trú ẩn từ các kế hoạch
        offset = 0
        for algo, plan in self._plans.items():
            if not plan or not hasattr(plan, 'routes'):
                continue

            algo_key = algo.lower() if isinstance(algo, str) else algo.value.lower()

            # Tính toán tải trọng cho mỗi shelter
            shelter_loads: Dict[str, int] = {}
            for route in plan.routes:
                if hasattr(route, 'shelter_id') and hasattr(route, 'flow'):
                    shelter_id = route.shelter_id
                    if shelter_id not in shelter_loads:
                        shelter_loads[shelter_id] = 0
                    shelter_loads[shelter_id] += route.flow

            if not shelter_loads:
                continue

            # Tạo dữ liệu biểu đồ
            shelters = list(shelter_loads.keys())
            values = list(shelter_loads.values())
            x = [i + offset * 0.25 for i in range(len(shelters))]

            color = colors.get(algo_key, COLORS.text)
            r, g, b = hex_to_rgb(color)

            # Vẽ biểu đồ cột
            for xi, val in zip(x, values):
                bar = pg.BarGraphItem(
                    x=[xi], height=[val], width=0.2,
                    brush=QColor(r, g, b),
                    name=algo_key.upper() if xi == x[0] else None
                )
                self.plot_widget.addItem(bar)

            offset += 1

    def clear_data(self):
        """Xóa tất cả dữ liệu."""
        self._plans.clear()
        if HAS_PYQTGRAPH and hasattr(self, 'plot_widget'):
            self.plot_widget.clear()


# =============================================================================
# WINNER BADGE
# =============================================================================

class WinnerBadge(QFrame):
    """Badge hiển thị thuật toán chiến thắng."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("card", True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(Sizes.PADDING_MD, Sizes.PADDING_SM,
                                  Sizes.PADDING_MD, Sizes.PADDING_SM)

        # Icon vương miện
        self.icon_label = QLabel("👑")
        self.icon_label.setFont(QFont("Segoe UI Emoji", 24))
        layout.addWidget(self.icon_label)

        # Thông tin người chiến thắng
        info_layout = QVBoxLayout()

        self.title_label = QLabel("THUẬT TOÁN CHIẾN THẮNG")
        self.title_label.setProperty("muted", True)
        info_layout.addWidget(self.title_label)

        self.winner_label = QLabel("--")
        self.winner_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self.winner_label.setStyleSheet(f"color: {COLORS.success};")
        info_layout.addWidget(self.winner_label)

        self.score_label = QLabel("Điểm: --")
        self.score_label.setProperty("muted", True)
        info_layout.addWidget(self.score_label)

        layout.addLayout(info_layout)
        layout.addStretch()

    def set_winner(self, algorithm: str, score: float, improvement: float = 0):
        """Cập nhật thuật toán chiến thắng."""
        self.winner_label.setText(algorithm.upper())
        self.score_label.setText(f"Điểm: {score:.3f}")

        if improvement > 0:
            self.score_label.setText(f"Điểm: {score:.3f} (+{improvement:.0%} tốt hơn)")


# =============================================================================
# COMPARISON VIEW CHÍNH
# =============================================================================

class ComparisonView(QWidget):
    """
    View chính cho so sánh thuật toán.
    Bao gồm biểu đồ hội tụ, bảng hiệu suất, radar chart,
    biểu đồ cột, bản đồ so sánh, và phân tích nơi trú ẩn.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_result: Dict[str, Any] = {}
        self._plans: Dict[str, Any] = {}
        self._network = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Sizes.PADDING_MD, Sizes.PADDING_MD,
                                  Sizes.PADDING_MD, Sizes.PADDING_MD)
        layout.setSpacing(Sizes.PADDING_MD)

        # Thanh tiêu đề với các nút export
        title_bar = QHBoxLayout()

        title = QLabel("SO SÁNH THUẬT TOÁN")
        title.setProperty("heading", True)
        title_bar.addWidget(title)

        title_bar.addStretch()

        # Nút xuất CSV
        self.export_csv_btn = QPushButton("📄 Xuất CSV")
        self.export_csv_btn.clicked.connect(self._export_to_csv)
        self.export_csv_btn.setToolTip("Xuất bảng so sánh ra file CSV")
        title_bar.addWidget(self.export_csv_btn)

        # Nút lưu hình
        self.export_png_btn = QPushButton("🖼️ Lưu hình")
        self.export_png_btn.clicked.connect(self._export_charts)
        self.export_png_btn.setToolTip("Lưu biểu đồ ra file PNG")
        title_bar.addWidget(self.export_png_btn)

        layout.addLayout(title_bar)

        # Winner badge
        self.winner_badge = WinnerBadge()
        layout.addWidget(self.winner_badge)

        # Tab widget cho nội dung chính
        self.content_tabs = QTabWidget()

        # =================================================================
        # TAB 1: BIỂU ĐỒ
        # =================================================================
        charts_tab = QWidget()
        charts_layout = QVBoxLayout(charts_tab)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(Sizes.PADDING_SM)

        # Hàng 1: Biểu đồ hội tụ và Radar
        charts_row = QHBoxLayout()

        # Biểu đồ hội tụ
        conv_container = QWidget()
        conv_layout = QVBoxLayout(conv_container)
        conv_layout.setContentsMargins(0, 0, 0, 0)
        conv_label = QLabel("Biểu đồ Hội tụ")
        conv_label.setProperty("subheading", True)
        conv_layout.addWidget(conv_label)
        self.convergence_chart = ConvergenceChart()
        conv_layout.addWidget(self.convergence_chart)
        charts_row.addWidget(conv_container, 2)

        # Biểu đồ Radar
        radar_container = QWidget()
        radar_layout = QVBoxLayout(radar_container)
        radar_layout.setContentsMargins(0, 0, 0, 0)
        radar_label = QLabel("Biểu đồ Radar")
        radar_label.setProperty("subheading", True)
        radar_layout.addWidget(radar_label)
        self.radar_chart = RadarChart()
        radar_layout.addWidget(self.radar_chart)
        charts_row.addWidget(radar_container, 1)

        charts_layout.addLayout(charts_row, 2)

        # Hàng 2: Biểu đồ cột
        bar_label = QLabel("So sánh Chỉ số")
        bar_label.setProperty("subheading", True)
        charts_layout.addWidget(bar_label)
        self.metric_bar_chart = MetricBarChart()
        charts_layout.addWidget(self.metric_bar_chart, 1)

        self.content_tabs.addTab(charts_tab, "📊 Biểu đồ")

        # =================================================================
        # TAB 2: BẢNG SO SÁNH
        # =================================================================
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(Sizes.PADDING_MD)

        # Bảng hiệu suất
        table_label = QLabel("Bảng Hiệu suất")
        table_label.setProperty("subheading", True)
        table_layout.addWidget(table_label)

        self.performance_table = PerformanceTable()
        table_layout.addWidget(self.performance_table)

        # Panel thống kê
        self.stats_panel = StatisticalSummaryPanel()
        table_layout.addWidget(self.stats_panel)

        self.content_tabs.addTab(table_tab, "📋 Bảng so sánh")

        # =================================================================
        # TAB 3: BẢN ĐỒ SO SÁNH
        # =================================================================
        map_tab = QWidget()
        map_layout = QVBoxLayout(map_tab)
        map_layout.setContentsMargins(0, 0, 0, 0)

        map_label = QLabel("So sánh Tuyến đường")
        map_label.setProperty("subheading", True)
        map_layout.addWidget(map_label)

        self.route_comparison_map = RouteComparisonMap()
        map_layout.addWidget(self.route_comparison_map)

        self.content_tabs.addTab(map_tab, "🗺️ Bản đồ")

        # =================================================================
        # TAB 4: PHÂN TÍCH NƠI TRÚ ẨN
        # =================================================================
        shelter_tab = QWidget()
        shelter_layout = QVBoxLayout(shelter_tab)
        shelter_layout.setContentsMargins(0, 0, 0, 0)

        shelter_label = QLabel("Phân tích Nơi trú ẩn")
        shelter_label.setProperty("subheading", True)
        shelter_layout.addWidget(shelter_label)

        self.shelter_load_chart = ShelterLoadChart()
        shelter_layout.addWidget(self.shelter_load_chart)

        self.content_tabs.addTab(shelter_tab, "🏠 Nơi trú ẩn")

        layout.addWidget(self.content_tabs)

    def set_network(self, network):
        """Thiết lập mạng lưới cho bản đồ so sánh."""
        self._network = network
        self.route_comparison_map.set_network(network)

    def set_plans(self, plans: Dict[str, Any]):
        """Thiết lập các kế hoạch sơ tán."""
        self._plans = plans
        self.route_comparison_map.set_plans(plans)
        self.shelter_load_chart.set_data(plans)

    def update_comparison(self, result: Dict[str, Any]):
        """
        Cập nhật view với kết quả so sánh.

        Args:
            result: Dictionary chứa:
                - metrics: Dict[algo, AlgorithmMetrics dict]
                - winner: str
                - winner_score: float
                - convergence: Dict[algo, List[float]]
                - radar_data: Dict[algo, List[float]] (5 giá trị mỗi cái)
                - plans: Dict[algo, EvacuationPlan] (tùy chọn)
        """
        self._current_result = result

        # Cập nhật biểu đồ hội tụ
        convergence_data = result.get('convergence', {})
        self.convergence_chart.clear_data()
        for algo, data in convergence_data.items():
            self.convergence_chart.set_data(algo, data)

        # Cập nhật bảng hiệu suất
        metrics = result.get('metrics', {})
        self.performance_table.update_metrics(metrics)

        # Cập nhật biểu đồ radar
        radar_data = result.get('radar_data', {})
        self.radar_chart.clear_data()
        for algo, values in radar_data.items():
            self.radar_chart.set_data(algo, values)

        # Cập nhật biểu đồ cột
        self.metric_bar_chart.set_data(metrics)

        # Cập nhật panel thống kê
        winner = result.get('winner', '')
        winner_score = result.get('winner_score', 0)
        self.stats_panel.update_stats(metrics, winner)

        # Cập nhật winner badge
        if winner:
            self.winner_badge.set_winner(winner, winner_score)
            self.performance_table.highlight_winner(winner)

        # Cập nhật plans nếu có
        plans = result.get('plans', {})
        if plans:
            self.set_plans(plans)

    def add_convergence_point(self, algorithm: str, iteration: int, cost: float):
        """Thêm điểm hội tụ thời gian thực."""
        self.convergence_chart.add_point(algorithm, iteration, cost)

    def clear(self):
        """Xóa tất cả dữ liệu."""
        self._current_result.clear()
        self._plans.clear()
        self.convergence_chart.clear_data()
        self.radar_chart.clear_data()
        self.metric_bar_chart.clear_data()
        self.shelter_load_chart.clear_data()
        self.route_comparison_map.clear()
        self.winner_badge.set_winner("--", 0)

    def _export_to_csv(self):
        """Xuất bảng so sánh ra file CSV."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu file CSV",
            "so_sanh_thuat_toan.csv",
            "CSV Files (*.csv);;Tất cả files (*)"
        )

        if not filepath:
            return

        try:
            metrics = self._current_result.get('metrics', {})
            if not metrics:
                QMessageBox.warning(
                    self,
                    "Cảnh báo",
                    "Không có dữ liệu để xuất. Hãy chạy so sánh trước."
                )
                return

            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)

                # Ghi header
                writer.writerow(['Chỉ số', 'GBFS', 'GWO', 'Hybrid'])

                # Các hàng dữ liệu
                rows = [
                    ('Thời gian (s)', 'execution_time_seconds', '{:.3f}'),
                    ('Chi phí', 'final_cost', '{:.2f}'),
                    ('Số tuyến đường', 'routes_found', '{:d}'),
                    ('Người sơ tán', 'evacuees_covered', '{:,d}'),
                    ('Tỷ lệ bao phủ', 'coverage_rate', '{:.1%}'),
                    ('Độ dài TB', 'average_path_length', '{:.1f}'),
                    ('Số vòng lặp', 'iterations', '{:d}')
                ]

                for display_name, key, fmt in rows:
                    row = [display_name]
                    for algo in ['gbfs', 'gwo', 'hybrid']:
                        data = metrics.get(algo, {})
                        value = data.get(key, 0)
                        try:
                            if '%' in fmt:
                                row.append(fmt.format(value))
                            elif 'd' in fmt:
                                row.append(fmt.format(int(value)))
                            else:
                                row.append(fmt.format(value))
                        except (ValueError, TypeError):
                            row.append(str(value))
                    writer.writerow(row)

                # Ghi winner
                winner = self._current_result.get('winner', 'N/A')
                winner_score = self._current_result.get('winner_score', 0)
                writer.writerow([])
                writer.writerow(['Thuật toán chiến thắng', winner.upper()])
                writer.writerow(['Điểm số', f'{winner_score:.3f}'])

            QMessageBox.information(
                self,
                "Thành công",
                f"Đã xuất dữ liệu ra file:\n{filepath}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Lỗi",
                f"Không thể xuất file CSV:\n{str(e)}"
            )

    def _export_charts(self):
        """Xuất biểu đồ ra file hình ảnh."""
        if not HAS_PYQTGRAPH:
            QMessageBox.warning(
                self,
                "Cảnh báo",
                "Cần cài đặt pyqtgraph để xuất biểu đồ."
            )
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu biểu đồ hội tụ",
            "bieu_do_hoi_tu.png",
            "PNG Files (*.png);;Tất cả files (*)"
        )

        if not filepath:
            return

        try:
            self.convergence_chart.export_image(filepath)
            QMessageBox.information(
                self,
                "Thành công",
                f"Đã lưu biểu đồ ra file:\n{filepath}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Lỗi",
                f"Không thể lưu biểu đồ:\n{str(e)}"
            )
