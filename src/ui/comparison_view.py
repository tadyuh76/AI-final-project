"""
View so sánh thuật toán với biểu đồ và bảng hiệu suất.
Hiển thị các thông tin quan trọng: biểu đồ hội tụ, radar, bảng hiệu suất, biểu đồ cột.
"""

from typing import Optional, Dict, List, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QGridLayout
)
from PyQt6.QtCore import Qt
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
        self.setMinimumHeight(200)

        self._data: Dict[str, List[float]] = {}
        self._colors = {
            'astar': COLORS.warning,  # Orange for A* (optimal baseline)
            'gbfs': COLORS.success,
            'gwo': COLORS.purple
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

            layout.addWidget(self.plot_widget)
        else:
            # Fallback: nhãn đơn giản
            label = QLabel("Cài đặt pyqtgraph để xem biểu đồ\npip install pyqtgraph")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(f"color: {COLORS.text_muted}; padding: 40px;")
            layout.addWidget(label)

    def set_data(self, algorithm: str, convergence_history: List[float]):
        """Thiết lập dữ liệu hội tụ cho một thuật toán."""
        if not convergence_history:
            return
        self._data[algorithm] = convergence_history
        self._update_plot()

    def clear_data(self):
        """Xóa tất cả dữ liệu."""
        self._data.clear()
        if HAS_PYQTGRAPH:
            self.plot_widget.clear()
            self._plot_items.clear()
            self.plot_widget.addLegend(offset=(60, 30))

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

            x_data = list(range(len(data)))

            if algo in self._plot_items:
                self._plot_items[algo].setData(x_data, data)
            else:
                self._plot_items[algo] = self.plot_widget.plot(
                    x_data, data,
                    pen=pen, name=algo.upper()
                )

    def add_point(self, algorithm: str, iteration: int, cost: float):
        """Thêm điểm dữ liệu mới (cho cập nhật thời gian thực)."""
        if algorithm not in self._data:
            self._data[algorithm] = []
        self._data[algorithm].append(cost)
        self._update_plot()


# =============================================================================
# BIỂU ĐỒ CỘT SO SÁNH CHỈ SỐ
# =============================================================================

class MetricBarChart(QWidget):
    """Biểu đồ cột ngang so sánh các chỉ số."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(180)

        self._data: Dict[str, float] = {}
        self._metric_name = "Chi phí"
        self._colors = {
            'astar': COLORS.warning,  # Orange for A* (optimal baseline)
            'gbfs': COLORS.success,
            'gwo': COLORS.purple
        }

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if HAS_PYQTGRAPH:
            pg.setConfigOptions(
                background=hex_to_qcolor(COLORS.surface),
                foreground=hex_to_qcolor(COLORS.text),
                antialias=True
            )

            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setTitle('So sánh Chi phí')
            self.plot_widget.showGrid(x=True, y=False, alpha=0.3)

            # Ẩn trục Y vì ta sẽ dùng tên thuật toán
            self.plot_widget.getAxis('left').setTicks([])

            layout.addWidget(self.plot_widget)
        else:
            label = QLabel("Cài đặt pyqtgraph để xem biểu đồ")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)

    def set_data(self, metrics: Dict[str, float], metric_name: str = "Chi phí"):
        """Thiết lập dữ liệu cho biểu đồ cột."""
        self._data = metrics
        self._metric_name = metric_name
        self._update_plot()

    def clear_data(self):
        """Xóa dữ liệu."""
        self._data.clear()
        if HAS_PYQTGRAPH:
            self.plot_widget.clear()

    def _update_plot(self):
        """Cập nhật biểu đồ."""
        if not HAS_PYQTGRAPH or not self._data:
            return

        self.plot_widget.clear()
        self.plot_widget.setTitle(f'So sánh {self._metric_name}')

        algos = ['gbfs', 'gwo']
        y_positions = []
        widths = []
        colors = []
        labels = []

        for i, algo in enumerate(algos):
            if algo in self._data:
                y_positions.append(i)
                widths.append(self._data[algo])
                r, g, b = hex_to_rgb(self._colors.get(algo, COLORS.text))
                colors.append(QColor(r, g, b))
                labels.append(algo.upper() if algo != 'astar' else 'A*')

        if not widths:
            return

        # Vẽ các thanh ngang
        for i, (y, w, color, label) in enumerate(zip(y_positions, widths, colors, labels)):
            bar = pg.BarGraphItem(
                x0=[0], y=[y], width=[w], height=0.6,
                brush=color, pen=pg.mkPen(color, width=1)
            )
            self.plot_widget.addItem(bar)

            # Thêm nhãn giá trị
            text = pg.TextItem(f"{label}: {w:,.0f}", color=hex_to_qcolor(COLORS.text))
            text.setPos(w * 0.02, y)
            self.plot_widget.addItem(text)

        # Thiết lập range
        max_val = max(widths) if widths else 1
        self.plot_widget.setXRange(0, max_val * 1.1)
        self.plot_widget.setYRange(-0.5, len(algos) - 0.5)


# =============================================================================
# BẢNG HIỆU SUẤT
# =============================================================================

class PerformanceTable(QTableWidget):
    """Bảng so sánh hiệu suất các thuật toán."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Thiết lập bảng
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(['Chỉ số', 'GBFS', 'GWO'])

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

    def update_metrics(self, metrics: Dict[str, Dict[str, Any]]):
        """Cập nhật bảng với các chỉ số từ kết quả so sánh."""
        algo_columns = {'gbfs': 1, 'gwo': 2}
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
        algo_columns = {'gbfs': 1, 'gwo': 2}
        winner_col = algo_columns.get(winner)

        if winner_col is None:
            return

        highlight_color = hex_to_qcolor(COLORS.success, 50)

        for row in range(self.rowCount()):
            item = self.item(row, winner_col)
            if item:
                item.setBackground(QBrush(highlight_color))

    def clear_data(self):
        """Xóa tất cả dữ liệu trong bảng."""
        for row in range(self.rowCount()):
            for col in range(1, self.columnCount()):  # Bỏ qua cột chỉ số
                item = QTableWidgetItem("")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setBackground(QBrush(Qt.GlobalColor.transparent))
                self.setItem(row, col, item)


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
            'astar': COLORS.warning,  # Orange for A* (optimal baseline)
            'gbfs': COLORS.success,
            'gwo': COLORS.purple
        }

    def set_data(self, algorithm: str, values: List[float]):
        """Thiết lập dữ liệu cho thuật toán."""
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
        self.icon_label = QLabel("")
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
    Bao gồm biểu đồ hội tụ, bảng hiệu suất, radar chart và biểu đồ cột.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Sizes.PADDING_MD, Sizes.PADDING_MD,
                                  Sizes.PADDING_MD, Sizes.PADDING_MD)
        layout.setSpacing(Sizes.PADDING_SM)

        # Tiêu đề
        title = QLabel("SO SÁNH THUẬT TOÁN")
        title.setProperty("heading", True)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Winner badge
        self.winner_badge = WinnerBadge()
        layout.addWidget(self.winner_badge)

        # Grid layout cho 4 biểu đồ: 2x2
        # [Hội tụ]     [Bảng hiệu suất]
        # [Radar]      [Biểu đồ cột]

        content_widget = QWidget()
        grid = QGridLayout(content_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(Sizes.PADDING_SM)

        # Góc trên trái: Biểu đồ hội tụ
        convergence_frame = QFrame()
        convergence_frame.setProperty("card", True)
        convergence_layout = QVBoxLayout(convergence_frame)
        convergence_layout.setContentsMargins(Sizes.PADDING_SM, Sizes.PADDING_SM,
                                              Sizes.PADDING_SM, Sizes.PADDING_SM)

        convergence_label = QLabel("Biểu đồ Hội tụ")
        convergence_label.setProperty("subheading", True)
        convergence_layout.addWidget(convergence_label)

        self.convergence_chart = ConvergenceChart()
        convergence_layout.addWidget(self.convergence_chart)

        grid.addWidget(convergence_frame, 0, 0)

        # Góc trên phải: Bảng hiệu suất
        table_frame = QFrame()
        table_frame.setProperty("card", True)
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(Sizes.PADDING_SM, Sizes.PADDING_SM,
                                        Sizes.PADDING_SM, Sizes.PADDING_SM)

        table_label = QLabel("Bảng Hiệu suất")
        table_label.setProperty("subheading", True)
        table_layout.addWidget(table_label)

        self.performance_table = PerformanceTable()
        table_layout.addWidget(self.performance_table)

        grid.addWidget(table_frame, 0, 1)

        # Góc dưới trái: Biểu đồ radar
        radar_frame = QFrame()
        radar_frame.setProperty("card", True)
        radar_layout = QVBoxLayout(radar_frame)
        radar_layout.setContentsMargins(Sizes.PADDING_SM, Sizes.PADDING_SM,
                                        Sizes.PADDING_SM, Sizes.PADDING_SM)

        radar_label = QLabel("Biểu đồ Radar")
        radar_label.setProperty("subheading", True)
        radar_layout.addWidget(radar_label)

        self.radar_chart = RadarChart()
        radar_layout.addWidget(self.radar_chart)

        grid.addWidget(radar_frame, 1, 0)

        # Góc dưới phải: Biểu đồ cột
        bar_frame = QFrame()
        bar_frame.setProperty("card", True)
        bar_layout = QVBoxLayout(bar_frame)
        bar_layout.setContentsMargins(Sizes.PADDING_SM, Sizes.PADDING_SM,
                                      Sizes.PADDING_SM, Sizes.PADDING_SM)

        bar_label = QLabel("So sánh Chi phí")
        bar_label.setProperty("subheading", True)
        bar_layout.addWidget(bar_label)

        self.metric_bar_chart = MetricBarChart()
        bar_layout.addWidget(self.metric_bar_chart)

        grid.addWidget(bar_frame, 1, 1)

        # Thiết lập tỷ lệ cột và hàng
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        layout.addWidget(content_widget)

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
        """
        # Cập nhật biểu đồ hội tụ
        convergence_data = result.get('convergence', {})
        self.convergence_chart.clear_data()
        for algo, data in convergence_data.items():
            if data:  # Chỉ thêm nếu có dữ liệu
                self.convergence_chart.set_data(algo, data)

        # Cập nhật bảng hiệu suất
        metrics = result.get('metrics', {})
        self.performance_table.update_metrics(metrics)

        # Cập nhật biểu đồ radar
        radar_data = result.get('radar_data', {})
        self.radar_chart.clear_data()
        for algo, values in radar_data.items():
            self.radar_chart.set_data(algo, values)

        # Cập nhật biểu đồ cột với chi phí
        cost_data = {}
        for algo, data in metrics.items():
            if 'final_cost' in data:
                cost_data[algo] = data['final_cost']
        self.metric_bar_chart.set_data(cost_data, "Chi phí")

        # Cập nhật winner badge
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
        self.metric_bar_chart.clear_data()
        self.performance_table.clear_data()
        self.winner_badge.set_winner("--", 0)
