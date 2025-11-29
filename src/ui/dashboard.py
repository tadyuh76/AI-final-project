"""
Dashboard hiển thị các chỉ số thời gian thực trong quá trình mô phỏng sơ tán.
"""

from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QFrame, QProgressBar, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPainter, QColor, QPen, QBrush

from .styles import COLORS, Sizes, hex_to_rgb


def hex_to_qcolor(hex_color: str, alpha: int = 255) -> QColor:
    """Chuyển đổi hex sang QColor."""
    r, g, b = hex_to_rgb(hex_color)
    return QColor(r, g, b, alpha)


class MetricCard(QFrame):
    """Card hiển thị một chỉ số với giá trị và tiêu đề."""

    def __init__(self, title: str, value: str = "0",
                 subtitle: str = "", color: str = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("card", True)
        self.setMinimumHeight(80)

        self._color = color or COLORS.primary

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Sizes.PADDING_MD, Sizes.PADDING_SM,
                                  Sizes.PADDING_MD, Sizes.PADDING_SM)
        layout.setSpacing(4)

        # Title
        self.title_label = QLabel(title)
        self.title_label.setProperty("muted", True)
        self.title_label.setFont(QFont("Segoe UI", 10))
        layout.addWidget(self.title_label)

        # Value
        self.value_label = QLabel(value)
        self.value_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.value_label.setStyleSheet(f"color: {self._color};")
        layout.addWidget(self.value_label)

        # Subtitle
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setProperty("muted", True)
        self.subtitle_label.setFont(QFont("Segoe UI", 9))
        layout.addWidget(self.subtitle_label)

    def set_value(self, value: str):
        """Cập nhật giá trị hiển thị."""
        self.value_label.setText(value)

    def set_subtitle(self, text: str):
        """Cập nhật subtitle."""
        self.subtitle_label.setText(text)

    def set_color(self, color: str):
        """Cập nhật màu của giá trị."""
        self._color = color
        self.value_label.setStyleSheet(f"color: {color};")


class ProgressMetricCard(QFrame):
    """Card hiển thị tiến độ với thanh progress."""

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("card", True)
        self.setMinimumHeight(90)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Sizes.PADDING_MD, Sizes.PADDING_SM,
                                  Sizes.PADDING_MD, Sizes.PADDING_SM)
        layout.setSpacing(6)

        # Header row
        header = QHBoxLayout()

        self.title_label = QLabel(title)
        self.title_label.setProperty("muted", True)
        header.addWidget(self.title_label)

        header.addStretch()

        self.value_label = QLabel("0%")
        self.value_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.value_label.setStyleSheet(f"color: {COLORS.success};")
        header.addWidget(self.value_label)

        layout.addLayout(header)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimumHeight(12)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Details row
        details = QHBoxLayout()

        self.detail_left = QLabel("")
        self.detail_left.setProperty("muted", True)
        self.detail_left.setFont(QFont("Segoe UI", 9))
        details.addWidget(self.detail_left)

        details.addStretch()

        self.detail_right = QLabel("")
        self.detail_right.setProperty("muted", True)
        self.detail_right.setFont(QFont("Segoe UI", 9))
        details.addWidget(self.detail_right)

        layout.addLayout(details)

    def set_progress(self, value: float, current: int = 0, total: int = 0):
        """Cập nhật tiến độ (0.0 - 1.0)."""
        percent = int(value * 100)
        self.progress_bar.setValue(percent)
        self.value_label.setText(f"{percent}%")

        if total > 0:
            self.detail_left.setText(f"Da so tan: {current:,}")
            self.detail_right.setText(f"Tong: {total:,}")

        # Color based on progress
        if value >= 0.8:
            color = COLORS.success
        elif value >= 0.5:
            color = COLORS.warning
        else:
            color = COLORS.primary

        self.value_label.setStyleSheet(f"color: {color};")


class ShelterStatusCard(QFrame):
    """Card hiển thị trạng thái các nơi trú ẩn."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("card", True)
        self.setMinimumHeight(100)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Sizes.PADDING_MD, Sizes.PADDING_SM,
                                  Sizes.PADDING_MD, Sizes.PADDING_SM)
        layout.setSpacing(4)

        # Title
        title = QLabel("Noi tru an")
        title.setProperty("muted", True)
        layout.addWidget(title)

        # Stats row
        stats = QHBoxLayout()

        # Open shelters
        self.open_label = QLabel("0")
        self.open_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self.open_label.setStyleSheet(f"color: {COLORS.success};")
        stats.addWidget(self.open_label)

        stats.addWidget(QLabel("/"))

        # Total shelters
        self.total_label = QLabel("0")
        self.total_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        stats.addWidget(self.total_label)

        stats.addWidget(QLabel(" dang hoat dong"))

        stats.addStretch()
        layout.addLayout(stats)

        # Capacity
        self.capacity_label = QLabel("Suc chua con lai: 0")
        self.capacity_label.setProperty("muted", True)
        layout.addWidget(self.capacity_label)

    def update_status(self, open_count: int, total_count: int, remaining_capacity: int):
        """Cập nhật trạng thái nơi trú ẩn."""
        self.open_label.setText(str(open_count))
        self.total_label.setText(str(total_count))
        self.capacity_label.setText(f"Suc chua con lai: {remaining_capacity:,}")

        # Color based on availability
        if open_count == 0:
            self.open_label.setStyleSheet(f"color: {COLORS.danger};")
        elif open_count < total_count / 2:
            self.open_label.setStyleSheet(f"color: {COLORS.warning};")
        else:
            self.open_label.setStyleSheet(f"color: {COLORS.success};")


class TimeEstimateCard(QFrame):
    """Card hiển thị ước tính thời gian."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("card", True)
        self.setMinimumHeight(80)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Sizes.PADDING_MD, Sizes.PADDING_SM,
                                  Sizes.PADDING_MD, Sizes.PADDING_SM)
        layout.setSpacing(4)

        # Title
        title = QLabel("Thoi gian uoc tinh")
        title.setProperty("muted", True)
        layout.addWidget(title)

        # Time display
        time_row = QHBoxLayout()

        self.time_label = QLabel("0h 0m")
        self.time_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        self.time_label.setStyleSheet(f"color: {COLORS.cyan};")
        time_row.addWidget(self.time_label)

        time_row.addStretch()
        layout.addLayout(time_row)

        # Current time
        self.current_label = QLabel("Thoi gian hien tai: 0h 0m")
        self.current_label.setProperty("muted", True)
        layout.addWidget(self.current_label)

    def update_time(self, current_hours: float, estimated_hours: float):
        """Cập nhật thời gian."""
        # Estimated completion
        est_h = int(estimated_hours)
        est_m = int((estimated_hours - est_h) * 60)
        self.time_label.setText(f"{est_h}h {est_m}m")

        # Current time
        cur_h = int(current_hours)
        cur_m = int((current_hours - cur_h) * 60)
        self.current_label.setText(f"Thoi gian hien tai: {cur_h}h {cur_m}m")


class RouteStatusCard(QFrame):
    """Card hiển thị trạng thái các tuyến đường."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("card", True)
        self.setMinimumHeight(80)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Sizes.PADDING_MD, Sizes.PADDING_SM,
                                  Sizes.PADDING_MD, Sizes.PADDING_SM)
        layout.setSpacing(4)

        # Title
        title = QLabel("Tuyen duong")
        title.setProperty("muted", True)
        layout.addWidget(title)

        # Stats grid
        grid = QGridLayout()
        grid.setSpacing(8)

        # Active
        grid.addWidget(QLabel("Dang chay:"), 0, 0)
        self.active_label = QLabel("0")
        self.active_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.active_label.setStyleSheet(f"color: {COLORS.success};")
        grid.addWidget(self.active_label, 0, 1)

        # Completed
        grid.addWidget(QLabel("Hoan thanh:"), 0, 2)
        self.completed_label = QLabel("0")
        self.completed_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.completed_label.setStyleSheet(f"color: {COLORS.primary};")
        grid.addWidget(self.completed_label, 0, 3)

        # Blocked
        grid.addWidget(QLabel("Bi chan:"), 1, 0)
        self.blocked_label = QLabel("0")
        self.blocked_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.blocked_label.setStyleSheet(f"color: {COLORS.danger};")
        grid.addWidget(self.blocked_label, 1, 1)

        layout.addLayout(grid)

    def update_status(self, active: int, completed: int, blocked: int):
        """Cập nhật trạng thái tuyến đường."""
        self.active_label.setText(str(active))
        self.completed_label.setText(str(completed))
        self.blocked_label.setText(str(blocked))


class Dashboard(QWidget):
    """
    Dashboard chính hiển thị các chỉ số thời gian thực.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(Sizes.DASHBOARD_HEIGHT)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Sizes.PADDING_SM)

        # Title
        title = QLabel("CHI SO THOI GIAN THUC")
        title.setProperty("heading", True)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Cards grid
        cards_layout = QGridLayout()
        cards_layout.setSpacing(Sizes.PADDING_SM)

        # Row 1: Progress and Time
        self.progress_card = ProgressMetricCard("Tien do so tan")
        cards_layout.addWidget(self.progress_card, 0, 0)

        self.time_card = TimeEstimateCard()
        cards_layout.addWidget(self.time_card, 0, 1)

        # Row 2: Evacuated and Routes
        self.evacuated_card = MetricCard(
            "Da so tan",
            "0",
            "nguoi",
            COLORS.success
        )
        cards_layout.addWidget(self.evacuated_card, 1, 0)

        self.routes_card = RouteStatusCard()
        cards_layout.addWidget(self.routes_card, 1, 1)

        # Row 3: Shelters and Risk
        self.shelters_card = ShelterStatusCard()
        cards_layout.addWidget(self.shelters_card, 2, 0)

        self.risk_card = MetricCard(
            "Rui ro trung binh",
            "0%",
            "phan tram tiep xuc nguy hiem",
            COLORS.warning
        )
        cards_layout.addWidget(self.risk_card, 2, 1)

        layout.addLayout(cards_layout)

    def update_metrics(self, metrics: Dict[str, Any]):
        """
        Cập nhật tất cả các chỉ số từ dữ liệu mô phỏng.

        Args:
            metrics: Dictionary chứa các chỉ số:
                - evacuation_progress: float (0-1)
                - total_evacuated: int
                - total_remaining: int
                - current_time_hours: float
                - estimated_completion_hours: float
                - active_routes: int
                - completed_routes: int
                - blocked_routes: int
                - shelter_utilization: dict
                - average_risk_exposure: float
        """
        # Progress
        progress = metrics.get('evacuation_progress', 0)
        evacuated = metrics.get('total_evacuated', 0)
        remaining = metrics.get('total_remaining', 0)
        total = evacuated + remaining
        self.progress_card.set_progress(progress, evacuated, total)

        # Evacuated
        self.evacuated_card.set_value(f"{evacuated:,}")
        self.evacuated_card.set_subtitle(f"con lai: {remaining:,}")

        # Time
        current_time = metrics.get('current_time_hours', 0)
        est_time = metrics.get('estimated_completion_hours', 0)
        self.time_card.update_time(current_time, est_time)

        # Routes
        active = metrics.get('active_routes', 0)
        completed = metrics.get('completed_routes', 0)
        blocked = metrics.get('blocked_routes', 0)
        self.routes_card.update_status(active, completed, blocked)

        # Risk
        risk = metrics.get('average_risk_exposure', 0)
        self.risk_card.set_value(f"{risk:.0%}")
        if risk > 0.5:
            self.risk_card.set_color(COLORS.danger)
        elif risk > 0.3:
            self.risk_card.set_color(COLORS.warning)
        else:
            self.risk_card.set_color(COLORS.success)

    def update_shelter_status(self, open_count: int, total_count: int, remaining_capacity: int):
        """Cập nhật trạng thái nơi trú ẩn."""
        self.shelters_card.update_status(open_count, total_count, remaining_capacity)

    def reset(self):
        """Đặt lại dashboard về trạng thái ban đầu."""
        self.progress_card.set_progress(0, 0, 0)
        self.evacuated_card.set_value("0")
        self.evacuated_card.set_subtitle("")
        self.time_card.update_time(0, 0)
        self.routes_card.update_status(0, 0, 0)
        self.shelters_card.update_status(0, 0, 0)
        self.risk_card.set_value("0%")
        self.risk_card.set_color(COLORS.warning)
