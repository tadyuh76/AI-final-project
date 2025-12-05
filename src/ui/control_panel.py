"""
Panel điều khiển cho cấu hình thuật toán và điều khiển mô phỏng.
"""

from typing import Optional, Dict, Any, Callable, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSlider, QSpinBox, QDoubleSpinBox,
    QFrame, QSizePolicy, QButtonGroup, QRadioButton, QCheckBox,
    QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPainter, QPen, QColor

from .styles import COLORS, Sizes, hex_to_rgb

def hex_to_qcolor(hex_color: str, alpha: int = 255):
    """Chuyển đổi hex sang QColor."""
    r, g, b = hex_to_rgb(hex_color)
    return QColor(r, g, b, alpha)


class StyledCheckBox(QCheckBox):
    """Checkbox với dấu tích bên trong."""

    def __init__(self, text: str, parent: Optional[QWidget] = None):
        super().__init__(text, parent)

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.isChecked():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Draw checkmark inside the indicator
            pen = QPen(QColor(255, 255, 255), 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)

            # Calculate checkmark position (inside the 18x18 indicator)
            x = 4
            y = 4
            # Draw checkmark path
            painter.drawLine(x + 3, y + 8, x + 6, y + 11)
            painter.drawLine(x + 6, y + 11, x + 12, y + 5)

            painter.end()


class LabeledSlider(QWidget):
    """Slider với label và giá trị hiển thị."""

    value_changed = pyqtSignal(float)

    def __init__(self, label: str, min_val: float, max_val: float,
                 default: float, decimals: int = 2, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.decimals = decimals
        self.scale = 10 ** decimals
        self._block_signals = False  # Ngăn chặn vòng lặp signal

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Label
        self.label = QLabel(label)
        self.label.setMinimumWidth(80)
        layout.addWidget(self.label)

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(min_val * self.scale))
        self.slider.setMaximum(int(max_val * self.scale))
        self.slider.setValue(int(default * self.scale))
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider, 1)

        # Value display
        self.value_label = QLabel(f"{default:.{decimals}f}")
        self.value_label.setMinimumWidth(45)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.value_label)

    def _on_slider_changed(self, value: int):
        if self._block_signals:
            return
        real_value = value / self.scale
        self.value_label.setText(f"{real_value:.{self.decimals}f}")
        self.value_changed.emit(real_value)

    def value(self) -> float:
        return self.slider.value() / self.scale

    def setValue(self, value: float, block_signal: bool = False):
        """Đặt giá trị slider.

        Args:
            value: Giá trị mới
            block_signal: Nếu True, không phát signal value_changed
        """
        if block_signal:
            self._block_signals = True
        self.slider.setValue(int(value * self.scale))
        self.value_label.setText(f"{value:.{self.decimals}f}")
        if block_signal:
            self._block_signals = False


class LabeledSpinBox(QWidget):
    """SpinBox với label."""

    value_changed = pyqtSignal(int)

    def __init__(self, label: str, min_val: int, max_val: int,
                 default: int, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Label
        self.label = QLabel(label)
        self.label.setMinimumWidth(80)
        layout.addWidget(self.label)

        # SpinBox
        self.spinbox = QSpinBox()
        self.spinbox.setMinimum(min_val)
        self.spinbox.setMaximum(max_val)
        self.spinbox.setValue(default)
        self.spinbox.valueChanged.connect(self.value_changed)
        layout.addWidget(self.spinbox, 1)

    def value(self) -> int:
        return self.spinbox.value()

    def setValue(self, value: int):
        self.spinbox.setValue(value)


class TyphoonCategorySelector(QWidget):
    """Bộ chọn cấp bão dạng nút."""

    category_changed = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Label
        label = QLabel("Cấp bão")
        label.setProperty("subheading", True)
        layout.addWidget(label)

        # Button group
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self.button_group = QButtonGroup(self)
        self.buttons = []

        for i in range(1, 6):
            btn = QPushButton(str(i))
            btn.setCheckable(True)
            btn.setMinimumSize(40, 40)
            btn.setMaximumSize(40, 40)

            # Style based on severity
            if i <= 2:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {COLORS.success_dark};
                        border-radius: 6px;
                        color: white;
                        font-weight: bold;
                    }}
                    QPushButton:checked {{
                        background-color: {COLORS.success};
                        border: 2px solid white;
                    }}
                """)
            elif i <= 3:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {COLORS.warning_dark};
                        border-radius: 6px;
                        color: white;
                        font-weight: bold;
                    }}
                    QPushButton:checked {{
                        background-color: {COLORS.warning};
                        border: 2px solid white;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {COLORS.danger_dark};
                        border-radius: 6px;
                        color: white;
                        font-weight: bold;
                    }}
                    QPushButton:checked {{
                        background-color: {COLORS.danger};
                        border: 2px solid white;
                    }}
                """)

            self.button_group.addButton(btn, i)
            self.buttons.append(btn)
            btn_layout.addWidget(btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Default to category 3
        self.buttons[2].setChecked(True)

        self.button_group.idClicked.connect(self.category_changed)

    def value(self) -> int:
        return self.button_group.checkedId()

    def setValue(self, category: int):
        if 1 <= category <= 5:
            self.buttons[category - 1].setChecked(True)


class ControlPanel(QWidget):
    """
    Panel điều khiển chính cho cấu hình thuật toán và điều khiển mô phỏng.
    """

    # Signals
    run_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    algorithm_changed = pyqtSignal(str)
    config_changed = pyqtSignal(dict)

    # Hazard zone signals
    hazard_add_mode_changed = pyqtSignal(bool)
    hazard_zone_delete_requested = pyqtSignal(int)
    hazard_zones_clear_requested = pyqtSignal()
    hazard_zones_randomize_requested = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumWidth(Sizes.SIDEBAR_WIDTH)
        self.setMaximumWidth(Sizes.SIDEBAR_WIDTH + 50)

        self._setup_ui()

    def _setup_ui(self):
        # Scroll area cho nội dung
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        # Container widget
        container = QWidget()
        scroll.setWidget(container)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

        # Content layout
        layout = QVBoxLayout(container)
        layout.setContentsMargins(Sizes.PADDING_MD, Sizes.PADDING_MD,
                                  Sizes.PADDING_MD, Sizes.PADDING_MD)
        layout.setSpacing(Sizes.PADDING_MD)

        # ===== Tiêu đề =====
        title = QLabel("BẢNG ĐIỀU KHIỂN")
        title.setProperty("heading", True)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # ===== Chọn thuật toán =====
        algo_group = QGroupBox("Thuật toán")
        algo_layout = QVBoxLayout(algo_group)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "GBFS (Tìm đường tham lam)",
            "GWO (Tối ưu bầy sói)"
        ])
        self.algorithm_combo.currentTextChanged.connect(self._on_algorithm_changed)
        algo_layout.addWidget(self.algorithm_combo)

        layout.addWidget(algo_group)

        # ===== Dân số sơ tán =====
        pop_group = QGroupBox("Dân số sơ tán")
        pop_layout = QVBoxLayout(pop_group)

        # Population slider - cho phép từ 1% đến 100%
        self.population_slider = LabeledSlider(
            "Dân số (%)", 1, 100, 50, decimals=0
        )
        self.population_slider.value_changed.connect(self._on_config_changed)
        pop_layout.addWidget(self.population_slider)

        layout.addWidget(pop_group)

        # ===== Tham số thuật toán =====
        params_group = QGroupBox("Tham số thuật toán")
        params_layout = QVBoxLayout(params_group)

        # Trọng số GBFS - tổng = 1.0
        self.weight_distance = LabeledSlider("Khoảng cách", 0, 1, 0.4)
        self.weight_distance.value_changed.connect(lambda v: self._on_weight_changed('distance', v))
        params_layout.addWidget(self.weight_distance)

        self.weight_risk = LabeledSlider("Rủi ro", 0, 1, 0.3)
        self.weight_risk.value_changed.connect(lambda v: self._on_weight_changed('risk', v))
        params_layout.addWidget(self.weight_risk)

        self.weight_congestion = LabeledSlider("Tắc nghẽn", 0, 1, 0.2)
        self.weight_congestion.value_changed.connect(lambda v: self._on_weight_changed('congestion', v))
        params_layout.addWidget(self.weight_congestion)

        self.weight_capacity = LabeledSlider("Sức chứa", 0, 1, 0.1)
        self.weight_capacity.value_changed.connect(lambda v: self._on_weight_changed('capacity', v))
        params_layout.addWidget(self.weight_capacity)

        # Label hiển thị tổng trọng số
        self.weight_total_label = QLabel("Tổng: 1.00")
        self.weight_total_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.weight_total_label.setStyleSheet(f"color: {COLORS.success}; font-size: 11px;")
        params_layout.addWidget(self.weight_total_label)

        # Tham số GWO
        params_layout.addWidget(QLabel(""))  # Spacer

        self.n_wolves = LabeledSpinBox("Số sói", 10, 100, 34)
        self.n_wolves.value_changed.connect(self._on_config_changed)
        params_layout.addWidget(self.n_wolves)

        self.max_iterations = LabeledSpinBox("Vòng lặp", 10, 500, 100)
        self.max_iterations.value_changed.connect(self._on_config_changed)
        params_layout.addWidget(self.max_iterations)

        layout.addWidget(params_group)

        # ===== Nút điều khiển =====
        controls_group = QGroupBox("Điều khiển")
        controls_layout = QVBoxLayout(controls_group)

        # Run button - prominent
        self.run_button = QPushButton("▶ CHẠY")
        self.run_button.setProperty("primary", True)
        self.run_button.setMinimumHeight(48)
        self.run_button.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.run_button.clicked.connect(self.run_clicked)
        controls_layout.addWidget(self.run_button)

        # Secondary controls
        btn_row = QHBoxLayout()

        self.pause_button = QPushButton("⏸ Tạm dừng")
        self.pause_button.clicked.connect(self._on_pause_clicked)
        self.pause_button.setEnabled(False)
        btn_row.addWidget(self.pause_button)

        self.reset_button = QPushButton("↻ Đặt lại")
        self.reset_button.clicked.connect(self.reset_clicked)
        btn_row.addWidget(self.reset_button)

        controls_layout.addLayout(btn_row)

        # Stop button
        self.stop_button = QPushButton("⏹ Dừng lại")
        self.stop_button.setProperty("danger", True)
        self.stop_button.clicked.connect(self.stop_clicked)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)

        layout.addWidget(controls_group)

        # ===== Tùy chọn mô phỏng =====
        sim_group = QGroupBox("Mô phỏng")
        sim_layout = QVBoxLayout(sim_group)

        # Tốc độ mô phỏng
        self.speed_slider = LabeledSlider("Tốc độ", 0.1, 5.0, 1.0, decimals=1)
        self.speed_slider.value_changed.connect(self._on_config_changed)
        sim_layout.addWidget(self.speed_slider)

        # Checkbox options
        self.show_particles = StyledCheckBox("Hiển thị hạt di chuyển")
        self.show_particles.setChecked(True)
        self.show_particles.stateChanged.connect(self._on_config_changed)
        sim_layout.addWidget(self.show_particles)
        sim_layout.addSpacing(4)

        self.show_routes = StyledCheckBox("Hiển thị tuyến đường")
        self.show_routes.setChecked(True)
        self.show_routes.stateChanged.connect(self._on_config_changed)
        sim_layout.addWidget(self.show_routes)
        sim_layout.addSpacing(4)

        self.show_hazards = StyledCheckBox("Hiển thị vùng nguy hiểm")
        self.show_hazards.setChecked(True)
        self.show_hazards.stateChanged.connect(self._on_config_changed)
        sim_layout.addWidget(self.show_hazards)
        sim_layout.addSpacing(4)

        self.show_all_roads = StyledCheckBox("Hiển thị tất cả đường")
        self.show_all_roads.setChecked(False)  # Mặc định tắt để tối ưu hiệu suất
        self.show_all_roads.setToolTip("Bật để hiển thị tất cả 195k+ đường (có thể chậm)")
        self.show_all_roads.stateChanged.connect(self._on_config_changed)
        sim_layout.addWidget(self.show_all_roads)

        layout.addWidget(sim_group)

        # ===== Cấu hình vùng nguy hiểm =====
        hazard_group = self._setup_hazard_config_section()
        layout.addWidget(hazard_group)

        # Spacer
        layout.addStretch()

        # ===== Trạng thái hiện tại =====
        self.status_label = QLabel("Sẵn sàng")
        self.status_label.setProperty("muted", True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

    def _setup_hazard_config_section(self) -> QGroupBox:
        """Tạo section cấu hình vùng nguy hiểm."""
        hazard_group = QGroupBox("Cấu hình vùng nguy hiểm")
        hazard_layout = QVBoxLayout(hazard_group)

        # === Add mode toggle button ===
        self.hazard_add_mode_btn = QPushButton("🎯 Đặt vùng nguy hiểm")
        self.hazard_add_mode_btn.setCheckable(True)
        self.hazard_add_mode_btn.setToolTip("Bật chế độ này rồi nhấn vào bản đồ để đặt vùng nguy hiểm")
        self.hazard_add_mode_btn.toggled.connect(self._on_hazard_add_mode_toggled)
        hazard_layout.addWidget(self.hazard_add_mode_btn)

        hazard_layout.addSpacing(8)

        # === New zone parameters ===
        params_label = QLabel("Thông số vùng mới:")
        params_label.setProperty("subheading", True)
        hazard_layout.addWidget(params_label)

        self.hazard_severity_slider = LabeledSlider("Mức độ (%)", 0, 100, 70, decimals=0)
        hazard_layout.addWidget(self.hazard_severity_slider)

        self.hazard_radius_slider = LabeledSlider("Bán kính (km)", 0.5, 5.0, 1.5, decimals=1)
        hazard_layout.addWidget(self.hazard_radius_slider)

        hazard_layout.addSpacing(8)

        # === Zone management ===
        self.hazard_zone_count_label = QLabel("Vùng hiện có: 0")
        hazard_layout.addWidget(self.hazard_zone_count_label)

        # Zone selector
        selector_layout = QHBoxLayout()
        selector_label = QLabel("Chọn vùng")
        selector_label.setMinimumWidth(80)
        selector_layout.addWidget(selector_label)

        self.hazard_zone_selector = QComboBox()
        self.hazard_zone_selector.setPlaceholderText("Không có vùng nào")
        self.hazard_zone_selector.currentIndexChanged.connect(self._on_hazard_zone_selected)
        selector_layout.addWidget(self.hazard_zone_selector, 1)
        hazard_layout.addLayout(selector_layout)

        # Delete buttons
        delete_btn_layout = QHBoxLayout()

        self.hazard_delete_btn = QPushButton("Xóa vùng")
        self.hazard_delete_btn.setEnabled(False)
        self.hazard_delete_btn.clicked.connect(self._on_hazard_delete_clicked)
        delete_btn_layout.addWidget(self.hazard_delete_btn)

        self.hazard_clear_all_btn = QPushButton("Xóa tất cả")
        self.hazard_clear_all_btn.clicked.connect(self._on_hazard_clear_all_clicked)
        delete_btn_layout.addWidget(self.hazard_clear_all_btn)

        hazard_layout.addLayout(delete_btn_layout)

        hazard_layout.addSpacing(8)

        # === Randomization section ===
        random_label = QLabel("Tạo ngẫu nhiên:")
        random_label.setProperty("subheading", True)
        hazard_layout.addWidget(random_label)

        self.hazard_random_count = LabeledSpinBox("Số vùng", 1, 20, 5)
        hazard_layout.addWidget(self.hazard_random_count)

        self.hazard_random_min_radius = LabeledSlider("Bán kính min", 0.5, 3.0, 0.5, decimals=1)
        hazard_layout.addWidget(self.hazard_random_min_radius)

        self.hazard_random_max_radius = LabeledSlider("Bán kính max", 1.0, 5.0, 3.0, decimals=1)
        hazard_layout.addWidget(self.hazard_random_max_radius)

        self.hazard_random_min_severity = LabeledSlider("Mức độ min (%)", 0, 100, 30, decimals=0)
        hazard_layout.addWidget(self.hazard_random_min_severity)

        self.hazard_random_max_severity = LabeledSlider("Mức độ max (%)", 0, 100, 90, decimals=0)
        hazard_layout.addWidget(self.hazard_random_max_severity)

        self.hazard_randomize_btn = QPushButton("🎲 Tạo ngẫu nhiên")
        self.hazard_randomize_btn.clicked.connect(self._on_hazard_randomize_clicked)
        hazard_layout.addWidget(self.hazard_randomize_btn)

        return hazard_group

    def _on_hazard_add_mode_toggled(self, checked: bool):
        """Xử lý khi bật/tắt chế độ đặt vùng nguy hiểm."""
        if checked:
            self.hazard_add_mode_btn.setText("🎯 Đang đặt vùng... (nhấn để tắt)")
            self.hazard_add_mode_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS.warning};
                    color: black;
                    font-weight: bold;
                }}
            """)
        else:
            self.hazard_add_mode_btn.setText("🎯 Đặt vùng nguy hiểm")
            self.hazard_add_mode_btn.setStyleSheet("")
        self.hazard_add_mode_changed.emit(checked)

    def _on_hazard_zone_selected(self, index: int):
        """Xử lý khi chọn vùng nguy hiểm từ dropdown."""
        self.hazard_delete_btn.setEnabled(index >= 0)

    def _on_hazard_delete_clicked(self):
        """Xử lý khi nhấn nút xóa vùng."""
        index = self.hazard_zone_selector.currentIndex()
        if index >= 0:
            self.hazard_zone_delete_requested.emit(index)

    def _on_hazard_clear_all_clicked(self):
        """Xử lý khi nhấn nút xóa tất cả."""
        self.hazard_zones_clear_requested.emit()

    def _on_hazard_randomize_clicked(self):
        """Xử lý khi nhấn nút tạo ngẫu nhiên."""
        params = self.get_randomization_params()
        self.hazard_zones_randomize_requested.emit(params)

    def get_new_zone_params(self) -> Dict[str, Any]:
        """Lấy thông số cho vùng nguy hiểm mới."""
        return {
            'radius_km': self.hazard_radius_slider.value(),
            'risk_level': self.hazard_severity_slider.value() / 100.0,
        }

    def get_randomization_params(self) -> Dict[str, Any]:
        """Lấy thông số cho việc tạo vùng ngẫu nhiên."""
        return {
            'count': self.hazard_random_count.value(),
            'min_radius': self.hazard_random_min_radius.value(),
            'max_radius': self.hazard_random_max_radius.value(),
            'min_severity': self.hazard_random_min_severity.value() / 100.0,
            'max_severity': self.hazard_random_max_severity.value() / 100.0
        }

    def update_hazard_zone_list(self, zones: List):
        """Cập nhật danh sách vùng nguy hiểm trong dropdown."""
        self.hazard_zone_selector.clear()
        self.hazard_zone_count_label.setText(f"Vùng hiện có: {len(zones)}")

        for i, zone in enumerate(zones):
            label = f"Vùng {i+1} ({zone.radius_km:.1f}km, {zone.risk_level*100:.0f}%)"
            self.hazard_zone_selector.addItem(label)

        self.hazard_delete_btn.setEnabled(len(zones) > 0)

    def set_hazard_add_mode(self, enabled: bool):
        """Đặt trạng thái chế độ đặt vùng nguy hiểm."""
        self.hazard_add_mode_btn.setChecked(enabled)

    def _on_algorithm_changed(self, text: str):
        """Xử lý khi thuật toán thay đổi."""
        algo_map = {
            "GBFS (Tìm đường tham lam)": "gbfs",
            "GWO (Tối ưu bầy sói)": "gwo"
        }
        self.algorithm_changed.emit(algo_map.get(text, "gbfs"))

    def _on_pause_clicked(self):
        """Xử lý nút pause/resume."""
        if self.pause_button.text() == "⏸ Tạm dừng":
            self.pause_button.setText("▶ Tiếp tục")
            self.pause_clicked.emit()
        else:
            self.pause_button.setText("⏸ Tạm dừng")
            self.pause_clicked.emit()

    def _on_weight_changed(self, changed_weight: str, new_value: float):
        """Xử lý khi một trọng số thay đổi - tự động cân bằng các trọng số khác.

        Khi một slider thay đổi, các slider còn lại được điều chỉnh
        tỷ lệ để tổng = 1.0
        """
        # Mapping từ tên sang slider
        weight_sliders = {
            'distance': self.weight_distance,
            'risk': self.weight_risk,
            'congestion': self.weight_congestion,
            'capacity': self.weight_capacity,
        }

        # Lấy các giá trị hiện tại của các slider KHÁC
        other_weights = {}
        for key, slider in weight_sliders.items():
            if key != changed_weight:
                other_weights[key] = slider.value()

        other_sum = sum(other_weights.values())

        # Giá trị còn lại cần phân bổ cho các slider khác
        remaining = 1.0 - new_value

        if remaining < 0:
            # Nếu giá trị mới > 1.0, đặt về 1.0 và các slider khác = 0
            remaining = 0.0

        if other_sum > 0:
            # Tỷ lệ để điều chỉnh các slider khác
            scale = remaining / other_sum
            for key, slider in weight_sliders.items():
                if key != changed_weight:
                    new_val = max(0.0, min(1.0, other_weights[key] * scale))
                    slider.setValue(new_val, block_signal=True)
        else:
            # Nếu tất cả các slider khác = 0, phân bổ đều
            if remaining > 0:
                equal_share = remaining / 3
                for key, slider in weight_sliders.items():
                    if key != changed_weight:
                        slider.setValue(equal_share, block_signal=True)

        # Cập nhật label tổng
        total = new_value + sum(s.value() for k, s in weight_sliders.items() if k != changed_weight)
        self.weight_total_label.setText(f"Tổng: {total:.2f}")

        # Màu sắc dựa trên tổng
        if abs(total - 1.0) < 0.01:
            self.weight_total_label.setStyleSheet(f"color: {COLORS.success}; font-size: 11px;")
        else:
            self.weight_total_label.setStyleSheet(f"color: {COLORS.warning}; font-size: 11px;")

        # Phát signal config_changed
        self.config_changed.emit(self.get_config())

    def _on_config_changed(self, *args):
        """Xử lý khi cấu hình thay đổi."""
        self.config_changed.emit(self.get_config())

    def get_config(self) -> Dict[str, Any]:
        """Lấy cấu hình hiện tại."""
        return {
            'algorithm': self._get_algorithm_type(),
            'population_percent': self.population_slider.value(),
            'weights': {
                'distance': self.weight_distance.value(),
                'risk': self.weight_risk.value(),
                'congestion': self.weight_congestion.value(),
                'capacity': self.weight_capacity.value(),
            },
            'n_wolves': self.n_wolves.value(),
            'max_iterations': self.max_iterations.value(),
            'simulation_speed': self.speed_slider.value(),
            'show_particles': self.show_particles.isChecked(),
            'show_routes': self.show_routes.isChecked(),
            'show_hazards': self.show_hazards.isChecked(),
            'show_all_roads': self.show_all_roads.isChecked(),
        }

    def _get_algorithm_type(self) -> str:
        """Lấy loại thuật toán được chọn."""
        text = self.algorithm_combo.currentText()
        algo_map = {
            "GBFS (Tìm đường tham lam)": "gbfs",
            "GWO (Tối ưu bầy sói)": "gwo"
        }
        return algo_map.get(text, "gbfs")

    def set_running_state(self, running: bool):
        """Cập nhật UI khi đang chạy/dừng."""
        self.run_button.setEnabled(not running)
        self.pause_button.setEnabled(running)
        self.stop_button.setEnabled(running)

        if running:
            self.status_label.setText("Đang chạy...")
            self.status_label.setStyleSheet(f"color: {COLORS.success};")
        else:
            self.status_label.setText("Sẵn sàng")
            self.status_label.setStyleSheet(f"color: {COLORS.text_muted};")

    def set_paused_state(self, paused: bool):
        """Cập nhật UI khi tạm dừng/tiếp tục."""
        if paused:
            self.pause_button.setText("▶ Tiếp tục")
            self.status_label.setText("Tạm dừng")
            self.status_label.setStyleSheet(f"color: {COLORS.warning};")
        else:
            self.pause_button.setText("⏸ Tạm dừng")
            self.status_label.setText("Đang chạy...")
            self.status_label.setStyleSheet(f"color: {COLORS.success};")

    def set_completed_state(self):
        """Cập nhật UI khi hoàn thành."""
        self.run_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.pause_button.setText("⏸ Tạm dừng")
        self.status_label.setText("Hoàn thành!")
        self.status_label.setStyleSheet(f"color: {COLORS.success};")

    def set_status(self, text: str, color: str = None):
        """Cập nhật trạng thái hiển thị."""
        self.status_label.setText(text)
        if color:
            self.status_label.setStyleSheet(f"color: {color};")
