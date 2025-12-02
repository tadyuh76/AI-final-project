"""
View so sánh thuật toán với biểu đồ và bảng hiệu suất.
Sử dụng pyqtgraph cho biểu đồ thời gian thực.
"""

from typing import Optional, Dict, List, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QBrush

from .styles import COLORS, Sizes, hex_to_rgb

# Try to import pyqtgraph
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
            # Configure pyqtgraph
            pg.setConfigOptions(
                background=hex_to_qcolor(COLORS.surface),
                foreground=hex_to_qcolor(COLORS.text),
                antialias=True
            )

            # Create plot widget
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setLabel('left', 'Chi phí')
            self.plot_widget.setLabel('bottom', 'Vòng lặp')
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_widget.setTitle('Biểu đồ Hội tụ')

            # Legend
            self.plot_widget.addLegend(offset=(60, 30))

            # Plot items
            self._plot_items: Dict[str, Any] = {}

            layout.addWidget(self.plot_widget)
        else:
            # Fallback: simple label
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

    def _update_plot(self):
        """Cập nhật biểu đồ với dữ liệu hiện tại."""
        if not HAS_PYQTGRAPH:
            return

        for algo, data in self._data.items():
            if not data:
                continue

            color = self._colors.get(algo, COLORS.text)
            r, g, b = hex_to_rgb(color)
            pen = pg.mkPen(color=QColor(r, g, b), width=2)

            if algo in self._plot_items:
                # Update existing plot
                self._plot_items[algo].setData(range(len(data)), data)
            else:
                # Create new plot
                self._plot_items[algo] = self.plot_widget.plot(
                    range(len(data)), data,
                    pen=pen, name=algo.upper()
                )

    def add_point(self, algorithm: str, iteration: int, cost: float):
        """Thêm điểm dữ liệu mới (cho cập nhật thời gian thực)."""
        if algorithm not in self._data:
            self._data[algorithm] = []

        self._data[algorithm].append(cost)
        self._update_plot()


class PerformanceTable(QTableWidget):
    """Bảng so sánh hiệu suất các thuật toán."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Setup table
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(['Chỉ số', 'GBFS', 'GWO', 'Hybrid'])

        # Rows
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

        # Styling
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

    def update_metrics(self, metrics: Dict[str, Dict[str, Any]]):
        """
        Cập nhật bảng với các chỉ số từ kết quả so sánh.

        Args:
            metrics: Dict với key là loại thuật toán ('gbfs', 'gwo', 'hybrid')
                     và value là dict chứa các chỉ số
        """
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

                # Format value
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

        # Background
        painter.fillRect(self.rect(), hex_to_qcolor(COLORS.surface))

        # Calculate center and radius
        center_x = self.width() // 2
        center_y = self.height() // 2
        radius = min(center_x, center_y) - 40

        if radius < 50:
            return

        import math

        num_axes = len(self._labels)
        angle_step = 2 * math.pi / num_axes

        # Draw grid
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

        # Draw axes
        for i in range(num_axes):
            angle = i * angle_step - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            painter.drawLine(center_x, center_y, int(x), int(y))

            # Labels
            label_x = center_x + (radius + 20) * math.cos(angle)
            label_y = center_y + (radius + 20) * math.sin(angle)
            painter.setPen(QPen(hex_to_qcolor(COLORS.text)))
            painter.drawText(int(label_x - 30), int(label_y - 5), 60, 20,
                            Qt.AlignmentFlag.AlignCenter, self._labels[i])
            painter.setPen(QPen(hex_to_qcolor(COLORS.border), 1))

        # Draw data polygons
        for algo, values in self._data.items():
            if len(values) != num_axes:
                continue

            color = self._colors.get(algo, COLORS.text)
            r, g, b = hex_to_rgb(color)

            # Fill
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


class MiniMapComparison(QWidget):
    """Widget so sánh mini-map giữa các thuật toán."""

    map_clicked = pyqtSignal(str)  # algorithm name

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Sizes.PADDING_SM)

        # Three mini map placeholders
        for algo in ['gbfs', 'gwo', 'hybrid']:
            frame = QFrame()
            frame.setProperty("card", True)
            frame.setMinimumSize(120, 100)
            frame.setCursor(Qt.CursorShape.PointingHandCursor)

            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(8, 8, 8, 8)

            # Algorithm name
            label = QLabel(algo.upper())
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            frame_layout.addWidget(label)

            # Placeholder for mini map
            placeholder = QLabel("Click để xem")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setProperty("muted", True)
            frame_layout.addWidget(placeholder)

            frame_layout.addStretch()

            # Store reference
            setattr(self, f"{algo}_frame", frame)

            layout.addWidget(frame)


class WinnerBadge(QFrame):
    """Badge hiển thị thuật toán chiến thắng."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("card", True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(Sizes.PADDING_MD, Sizes.PADDING_SM,
                                  Sizes.PADDING_MD, Sizes.PADDING_SM)

        # Crown icon (text fallback)
        self.icon_label = QLabel("👑")
        self.icon_label.setFont(QFont("Segoe UI Emoji", 24))
        layout.addWidget(self.icon_label)

        # Winner info
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


class ComparisonView(QWidget):
    """
    View chính cho so sánh thuật toán.
    Bao gồm biểu đồ hội tụ, bảng hiệu suất và radar chart.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Sizes.PADDING_MD, Sizes.PADDING_MD,
                                  Sizes.PADDING_MD, Sizes.PADDING_MD)
        layout.setSpacing(Sizes.PADDING_MD)

        # Title
        title = QLabel("SO SÁNH THUẬT TOÁN")
        title.setProperty("heading", True)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Winner badge
        self.winner_badge = WinnerBadge()
        layout.addWidget(self.winner_badge)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Charts
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(Sizes.PADDING_SM)

        # Convergence chart
        convergence_label = QLabel("Biểu đồ Hội tụ")
        convergence_label.setProperty("subheading", True)
        left_layout.addWidget(convergence_label)

        self.convergence_chart = ConvergenceChart()
        left_layout.addWidget(self.convergence_chart, 2)

        # Radar chart
        radar_label = QLabel("Biểu đồ Radar")
        radar_label.setProperty("subheading", True)
        left_layout.addWidget(radar_label)

        self.radar_chart = RadarChart()
        left_layout.addWidget(self.radar_chart, 1)

        splitter.addWidget(left_widget)

        # Right: Table
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(Sizes.PADDING_SM)

        table_label = QLabel("Bảng Hiệu suất")
        table_label.setProperty("subheading", True)
        right_layout.addWidget(table_label)

        self.performance_table = PerformanceTable()
        right_layout.addWidget(self.performance_table)

        # Mini maps
        mini_label = QLabel("So sánh Kết quả")
        mini_label.setProperty("subheading", True)
        right_layout.addWidget(mini_label)

        self.mini_maps = MiniMapComparison()
        right_layout.addWidget(self.mini_maps)

        splitter.addWidget(right_widget)

        # Set splitter proportions
        splitter.setSizes([600, 400])

        layout.addWidget(splitter)

    def update_comparison(self, result: Dict[str, Any]):
        """
        Cập nhật view với kết quả so sánh.

        Args:
            result: Dictionary chứa:
                - metrics: Dict[algo, AlgorithmMetrics dict]
                - winner: str
                - winner_score: float
                - convergence: Dict[algo, List[float]]
                - radar_data: Dict[algo, List[float]] (5 values each)
        """
        # Update convergence chart
        convergence_data = result.get('convergence', {})
        self.convergence_chart.clear_data()
        for algo, data in convergence_data.items():
            self.convergence_chart.set_data(algo, data)

        # Update performance table
        metrics = result.get('metrics', {})
        self.performance_table.update_metrics(metrics)

        # Update radar chart
        radar_data = result.get('radar_data', {})
        self.radar_chart.clear_data()
        for algo, values in radar_data.items():
            self.radar_chart.set_data(algo, values)

        # Update winner
        winner = result.get('winner', '')
        winner_score = result.get('winner_score', 0)
        if winner:
            self.winner_badge.set_winner(winner, winner_score)
            self.performance_table.highlight_winner(winner)

    def add_convergence_point(self, algorithm: str, iteration: int, cost: float):
        """Thêm điểm hội tụ thời gian thực."""
        self.convergence_chart.add_point(algorithm, iteration, cost)

    def clear(self):
        """Xóa tất cả dữ liệu."""
        self.convergence_chart.clear_data()
        self.radar_chart.clear_data()
        self.winner_badge.set_winner("--", 0)
