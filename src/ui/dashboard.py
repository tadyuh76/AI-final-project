"""
Dashboard hiển thị các chỉ số thời gian thực trong quá trình mô phỏng sơ tán.
"""

from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QProgressBar, QScrollArea
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from .styles import COLORS, hex_to_rgb


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
        self.setMinimumHeight(70)
        self.setMinimumWidth(180)

        self._color = color or COLORS.primary

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)

        # Title
        self.title_label = QLabel(title)
        self.title_label.setProperty("muted", True)
        self.title_label.setFont(QFont("Arial", 10))
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)

        # Value
        self.value_label = QLabel(value)
        self.value_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self.value_label.setStyleSheet(f"color: {self._color};")
        layout.addWidget(self.value_label)

        # Subtitle
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setProperty("muted", True)
        self.subtitle_label.setFont(QFont("Arial", 9))
        self.subtitle_label.setWordWrap(True)
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
        self.setMinimumHeight(80)
        self.setMinimumWidth(280)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # Header row
        header = QHBoxLayout()

        self.title_label = QLabel(title)
        self.title_label.setProperty("muted", True)
        self.title_label.setFont(QFont("Arial", 10))
        header.addWidget(self.title_label)

        header.addStretch()

        self.value_label = QLabel("0%")
        self.value_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.value_label.setStyleSheet(f"color: {COLORS.success};")
        header.addWidget(self.value_label)

        layout.addLayout(header)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimumHeight(10)
        self.progress_bar.setMaximumHeight(12)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Details row
        details = QHBoxLayout()

        self.detail_left = QLabel("")
        self.detail_left.setProperty("muted", True)
        self.detail_left.setFont(QFont("Arial", 9))
        details.addWidget(self.detail_left)

        details.addStretch()

        self.detail_right = QLabel("")
        self.detail_right.setProperty("muted", True)
        self.detail_right.setFont(QFont("Arial", 9))
        details.addWidget(self.detail_right)

        layout.addLayout(details)

    def set_progress(self, value: float, current: int = 0, total: int = 0):
        """Cập nhật tiến độ (0.0 - 1.0)."""
        percent = int(value * 100)
        self.progress_bar.setValue(percent)
        self.value_label.setText(f"{percent}%")

        if total > 0:
            self.detail_left.setText(f"Đã sơ tán: {current:,}")
            self.detail_right.setText(f"Tổng: {total:,}")

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
        self.setMinimumHeight(70)
        self.setMinimumWidth(250)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)

        # Title
        title = QLabel("Nơi trú ẩn")
        title.setProperty("muted", True)
        title.setFont(QFont("Arial", 10))
        layout.addWidget(title)

        # Stats row
        stats = QHBoxLayout()
        stats.setSpacing(4)

        # Open shelters
        self.open_label = QLabel("0")
        self.open_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.open_label.setStyleSheet(f"color: {COLORS.success};")
        stats.addWidget(self.open_label)

        slash = QLabel("/")
        slash.setFont(QFont("Arial", 14))
        stats.addWidget(slash)

        # Total shelters
        self.total_label = QLabel("0")
        self.total_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        stats.addWidget(self.total_label)

        active_label = QLabel(" hoạt động")
        active_label.setFont(QFont("Arial", 10))
        stats.addWidget(active_label)

        stats.addStretch()
        layout.addLayout(stats)

        # Capacity
        self.capacity_label = QLabel("Sức chứa còn: 0")
        self.capacity_label.setProperty("muted", True)
        self.capacity_label.setFont(QFont("Arial", 9))
        layout.addWidget(self.capacity_label)

    def update_status(self, open_count: int, total_count: int, remaining_capacity: int):
        """Cập nhật trạng thái nơi trú ẩn."""
        self.open_label.setText(str(open_count))
        self.total_label.setText(str(total_count))
        self.capacity_label.setText(f"Sức chứa còn: {remaining_capacity:,}")

        # Color based on availability
        if open_count == 0:
            self.open_label.setStyleSheet(f"color: {COLORS.danger};")
        elif open_count < total_count / 2:
            self.open_label.setStyleSheet(f"color: {COLORS.warning};")
        else:
            self.open_label.setStyleSheet(f"color: {COLORS.success};")


class CapacityRatioCard(QFrame):
    """Card hiển thị tỷ lệ sức chứa so với dân số cần sơ tán."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("card", True)
        self.setMinimumHeight(80)
        self.setMinimumWidth(200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # Title
        title = QLabel("Tỷ lệ sức chứa")
        title.setProperty("muted", True)
        title.setFont(QFont("Arial", 10))
        layout.addWidget(title)

        # Main ratio display
        ratio_row = QHBoxLayout()
        ratio_row.setSpacing(4)

        self.ratio_label = QLabel("0%")
        self.ratio_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.ratio_label.setStyleSheet(f"color: {COLORS.danger};")
        ratio_row.addWidget(self.ratio_label)

        ratio_row.addStretch()
        layout.addLayout(ratio_row)

        # Progress bar showing capacity vs population
        self.capacity_bar = QProgressBar()
        self.capacity_bar.setTextVisible(False)
        self.capacity_bar.setMinimumHeight(8)
        self.capacity_bar.setMaximumHeight(10)
        self.capacity_bar.setValue(0)
        layout.addWidget(self.capacity_bar)

        # Details
        self.details_label = QLabel("Sức chứa: 0 / Dân số: 0")
        self.details_label.setProperty("muted", True)
        self.details_label.setFont(QFont("Arial", 9))
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)

    def update_ratio(self, total_capacity: int, total_population: int, current_occupancy: int = 0):
        """Cập nhật tỷ lệ sức chứa.

        Args:
            total_capacity: Tổng sức chứa của tất cả nơi trú ẩn
            total_population: Tổng dân số cần sơ tán
            current_occupancy: Số người hiện đang ở nơi trú ẩn
        """
        if total_population > 0:
            ratio = total_capacity / total_population
            ratio_percent = min(100, ratio * 100)
        else:
            ratio = 1.0
            ratio_percent = 100

        self.ratio_label.setText(f"{ratio_percent:.1f}%")
        self.capacity_bar.setValue(int(ratio_percent))

        # Format numbers with K/M suffix for readability
        def format_num(n):
            if n >= 1_000_000:
                return f"{n/1_000_000:.1f}M"
            elif n >= 1_000:
                return f"{n/1_000:.0f}K"
            return str(n)

        remaining_capacity = max(0, total_capacity - current_occupancy)
        self.details_label.setText(
            f"Sức chứa: {format_num(total_capacity)} / "
            f"Dân số: {format_num(total_population)}"
        )

        # Color based on ratio - realistic thresholds for evacuation
        if ratio >= 0.8:  # 80%+ capacity - good
            self.ratio_label.setStyleSheet(f"color: {COLORS.success};")
        elif ratio >= 0.3:  # 30-80% - warning
            self.ratio_label.setStyleSheet(f"color: {COLORS.warning};")
        else:  # <30% - critical shortage
            self.ratio_label.setStyleSheet(f"color: {COLORS.danger};")


class TimeEstimateCard(QFrame):
    """Card hiển thị ước tính thời gian."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("card", True)
        self.setMinimumHeight(70)
        self.setMinimumWidth(180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)

        # Title
        title = QLabel("Thời gian ước tính")
        title.setProperty("muted", True)
        title.setFont(QFont("Arial", 10))
        layout.addWidget(title)

        # Time display
        self.time_label = QLabel("0h 0m")
        self.time_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.time_label.setStyleSheet(f"color: {COLORS.cyan};")
        layout.addWidget(self.time_label)

        # Current time
        self.current_label = QLabel("Hiện tại: 0h 0m")
        self.current_label.setProperty("muted", True)
        self.current_label.setFont(QFont("Arial", 9))
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
        self.current_label.setText(f"Hiện tại: {cur_h}h {cur_m}m")


class RouteStatusCard(QFrame):
    """Card hiển thị trạng thái các tuyến đường."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("card", True)
        self.setMinimumHeight(70)
        self.setMinimumWidth(280)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # Title
        title = QLabel("Tuyến đường")
        title.setProperty("muted", True)
        title.setFont(QFont("Arial", 10))
        layout.addWidget(title)

        # Stats row 1
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        row1.addWidget(QLabel("Đang chạy:"))
        self.active_label = QLabel("0")
        self.active_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.active_label.setStyleSheet(f"color: {COLORS.success};")
        row1.addWidget(self.active_label)

        row1.addSpacing(12)

        row1.addWidget(QLabel("Hoàn thành:"))
        self.completed_label = QLabel("0")
        self.completed_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.completed_label.setStyleSheet(f"color: {COLORS.primary};")
        row1.addWidget(self.completed_label)

        row1.addStretch()
        layout.addLayout(row1)

        # Stats row 2
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        row2.addWidget(QLabel("Rủi ro TB:"))
        self.risk_label = QLabel("0%")
        self.risk_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.risk_label.setStyleSheet(f"color: {COLORS.warning};")
        row2.addWidget(self.risk_label)

        row2.addStretch()
        layout.addLayout(row2)

    def update_status(self, active: int, completed: int, blocked: int, avg_risk: float = 0.0):
        """Cập nhật trạng thái tuyến đường."""
        self.active_label.setText(str(active))
        self.completed_label.setText(str(completed))
        self.risk_label.setText(f"{avg_risk:.0%}")

        # Color based on risk level
        if avg_risk > 0.5:
            self.risk_label.setStyleSheet(f"color: {COLORS.danger};")
        elif avg_risk > 0.3:
            self.risk_label.setStyleSheet(f"color: {COLORS.warning};")
        else:
            self.risk_label.setStyleSheet(f"color: {COLORS.success};")


class Dashboard(QWidget):
    """
    Dashboard chính hiển thị các chỉ số thời gian thực.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMaximumHeight(200)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # Title
        title = QLabel("CHỈ SỐ THỜI GIAN THỰC")
        title.setProperty("heading", True)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout.addWidget(title)

        # Cards in horizontal layout with scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMaximumHeight(140)

        cards_container = QWidget()
        cards_layout = QHBoxLayout(cards_container)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(8)

        # Progress card
        self.progress_card = ProgressMetricCard("Tiến độ sơ tán")
        cards_layout.addWidget(self.progress_card)

        # Evacuated card
        self.evacuated_card = MetricCard(
            "Đã sơ tán",
            "0",
            "người",
            COLORS.success
        )
        cards_layout.addWidget(self.evacuated_card)

        # Time card
        self.time_card = TimeEstimateCard()
        cards_layout.addWidget(self.time_card)

        # Shelters card
        self.shelters_card = ShelterStatusCard()
        cards_layout.addWidget(self.shelters_card)

        # Capacity ratio card - shows shelter capacity vs population
        self.capacity_card = CapacityRatioCard()
        cards_layout.addWidget(self.capacity_card)

        # Routes card
        self.routes_card = RouteStatusCard()
        cards_layout.addWidget(self.routes_card)

        # Risk card
        self.risk_card = MetricCard(
            "Rủi ro TB",
            "0%",
            "tiếp xúc nguy hiểm",
            COLORS.warning
        )
        cards_layout.addWidget(self.risk_card)

        cards_layout.addStretch()

        scroll.setWidget(cards_container)
        layout.addWidget(scroll)

    def update_metrics(self, metrics: Dict[str, Any]):
        """
        Cập nhật tất cả các chỉ số từ dữ liệu mô phỏng.
        """
        # Progress
        progress = metrics.get('evacuation_progress', 0)
        evacuated = metrics.get('total_evacuated', 0)
        remaining = metrics.get('total_remaining', 0)
        total = evacuated + remaining
        self.progress_card.set_progress(progress, evacuated, total)

        # Evacuated
        self.evacuated_card.set_value(f"{evacuated:,}")
        self.evacuated_card.set_subtitle(f"còn: {remaining:,}")

        # Time
        current_time = metrics.get('current_time_hours', 0)
        est_time = metrics.get('estimated_completion_hours', 0)
        self.time_card.update_time(current_time, est_time)

        # Routes with risk
        active = metrics.get('active_routes', 0)
        completed = metrics.get('completed_routes', 0)
        blocked = metrics.get('blocked_routes', 0)
        risk = metrics.get('average_risk_exposure', 0)
        self.routes_card.update_status(active, completed, blocked, risk)

        # Risk card
        self.risk_card.set_value(f"{risk:.0%}")
        if risk > 0.5:
            self.risk_card.set_color(COLORS.danger)
        elif risk > 0.3:
            self.risk_card.set_color(COLORS.warning)
        else:
            self.risk_card.set_color(COLORS.success)

        # Shelter status from arrivals data
        shelter_arrivals = metrics.get('shelter_arrivals', {})
        shelter_utilization = metrics.get('shelter_utilization', {})
        if shelter_arrivals or shelter_utilization:
            total_shelters = len(shelter_utilization) if shelter_utilization else len(shelter_arrivals)
            # Count open shelters (utilization < 100%)
            open_shelters = sum(1 for util in shelter_utilization.values() if util < 1.0) if shelter_utilization else total_shelters
            # Sum remaining capacity (estimated from utilization)
            total_arrivals = sum(shelter_arrivals.values()) if shelter_arrivals else 0
            remaining_capacity = metrics.get('remaining_shelter_capacity', 0)
            self.shelters_card.update_status(open_shelters, total_shelters, remaining_capacity)

        # Capacity ratio - shelter capacity vs population to evacuate
        total_capacity = metrics.get('total_shelter_capacity', 0)
        total_population = metrics.get('total_population', total)  # fallback to evacuated + remaining
        current_occupancy = metrics.get('total_evacuated', 0)
        if total_capacity > 0 or total_population > 0:
            self.capacity_card.update_ratio(total_capacity, total_population, current_occupancy)

    def update_shelter_status(self, open_count: int, total_count: int, remaining_capacity: int):
        """Cập nhật trạng thái nơi trú ẩn."""
        self.shelters_card.update_status(open_count, total_count, remaining_capacity)

    def update_capacity_ratio(self, total_capacity: int, total_population: int, current_occupancy: int = 0):
        """Cập nhật tỷ lệ sức chứa."""
        self.capacity_card.update_ratio(total_capacity, total_population, current_occupancy)

    def reset(self):
        """Đặt lại dashboard về trạng thái ban đầu."""
        self.progress_card.set_progress(0, 0, 0)
        self.evacuated_card.set_value("0")
        self.evacuated_card.set_subtitle("")
        self.time_card.update_time(0, 0)
        self.routes_card.update_status(0, 0, 0)
        self.shelters_card.update_status(0, 0, 0)
        self.capacity_card.update_ratio(0, 0, 0)
        self.risk_card.set_value("0%")
        self.risk_card.set_color(COLORS.warning)
