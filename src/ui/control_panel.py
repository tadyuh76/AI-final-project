"""
Panel điều khiển cho cấu hình thuật toán và điều khiển mô phỏng.
"""

from typing import Optional, Dict, Any, Callable
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSlider, QSpinBox, QDoubleSpinBox,
    QFrame, QSizePolicy, QButtonGroup, QRadioButton, QCheckBox,
    QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from .styles import COLORS, Sizes, hex_to_rgb


def hex_to_qcolor(hex_color: str, alpha: int = 255):
    """Chuyển đổi hex sang QColor."""
    from PyQt6.QtGui import QColor
    r, g, b = hex_to_rgb(hex_color)
    return QColor(r, g, b, alpha)


class LabeledSlider(QWidget):
    """Slider với label và giá trị hiển thị."""

    value_changed = pyqtSignal(float)

    def __init__(self, label: str, min_val: float, max_val: float,
                 default: float, decimals: int = 2, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.decimals = decimals
        self.scale = 10 ** decimals

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
        real_value = value / self.scale
        self.value_label.setText(f"{real_value:.{self.decimals}f}")
        self.value_changed.emit(real_value)

    def value(self) -> float:
        return self.slider.value() / self.scale

    def setValue(self, value: float):
        self.slider.setValue(int(value * self.scale))


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
            "GBFS + GWO (Lai ghép)",
            "GBFS (Tìm đường tham lam)",
            "GWO (Tối ưu bầy sói)"
        ])
        self.algorithm_combo.currentTextChanged.connect(self._on_algorithm_changed)
        algo_layout.addWidget(self.algorithm_combo)

        layout.addWidget(algo_group)

        # ===== Cấp độ bão =====
        typhoon_group = QGroupBox("Kịch bản bão")
        typhoon_layout = QVBoxLayout(typhoon_group)

        self.typhoon_selector = TyphoonCategorySelector()
        self.typhoon_selector.category_changed.connect(self._on_config_changed)
        typhoon_layout.addWidget(self.typhoon_selector)

        # Population slider
        self.population_slider = LabeledSlider(
            "Dân số (%)", 10, 100, 50, decimals=0
        )
        self.population_slider.value_changed.connect(self._on_config_changed)
        typhoon_layout.addWidget(self.population_slider)

        layout.addWidget(typhoon_group)

        # ===== Tham số thuật toán =====
        params_group = QGroupBox("Tham số thuật toán")
        params_layout = QVBoxLayout(params_group)

        # Trọng số GBFS
        self.weight_distance = LabeledSlider("Khoảng cách", 0, 1, 0.4)
        self.weight_distance.value_changed.connect(self._on_config_changed)
        params_layout.addWidget(self.weight_distance)

        self.weight_risk = LabeledSlider("Rủi ro", 0, 1, 0.3)
        self.weight_risk.value_changed.connect(self._on_config_changed)
        params_layout.addWidget(self.weight_risk)

        self.weight_congestion = LabeledSlider("Tắc nghẽn", 0, 1, 0.2)
        self.weight_congestion.value_changed.connect(self._on_config_changed)
        params_layout.addWidget(self.weight_congestion)

        self.weight_capacity = LabeledSlider("Sức chứa", 0, 1, 0.1)
        self.weight_capacity.value_changed.connect(self._on_config_changed)
        params_layout.addWidget(self.weight_capacity)

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
        self.show_particles = QCheckBox("Hiển thị hạt di chuyển")
        self.show_particles.setChecked(True)
        self.show_particles.stateChanged.connect(self._on_config_changed)
        sim_layout.addWidget(self.show_particles)
        sim_layout.addSpacing(4)

        self.show_routes = QCheckBox("Hiển thị tuyến đường")
        self.show_routes.setChecked(True)
        self.show_routes.stateChanged.connect(self._on_config_changed)
        sim_layout.addWidget(self.show_routes)
        sim_layout.addSpacing(4)

        self.show_hazards = QCheckBox("Hiển thị vùng nguy hiểm")
        self.show_hazards.setChecked(True)
        self.show_hazards.stateChanged.connect(self._on_config_changed)
        sim_layout.addWidget(self.show_hazards)

        layout.addWidget(sim_group)

        # Spacer
        layout.addStretch()

        # ===== Trạng thái hiện tại =====
        self.status_label = QLabel("Sẵn sàng")
        self.status_label.setProperty("muted", True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

    def _on_algorithm_changed(self, text: str):
        """Xử lý khi thuật toán thay đổi."""
        algo_map = {
            "GBFS + GWO (Lai ghép)": "hybrid",
            "GBFS (Tìm đường tham lam)": "gbfs",
            "GWO (Tối ưu bầy sói)": "gwo"
        }
        self.algorithm_changed.emit(algo_map.get(text, "hybrid"))

    def _on_pause_clicked(self):
        """Xử lý nút pause/resume."""
        if self.pause_button.text() == "⏸ Tạm dừng":
            self.pause_button.setText("▶ Tiếp tục")
            self.pause_clicked.emit()
        else:
            self.pause_button.setText("⏸ Tạm dừng")
            self.pause_clicked.emit()

    def _on_config_changed(self, *args):
        """Xử lý khi cấu hình thay đổi."""
        self.config_changed.emit(self.get_config())

    def get_config(self) -> Dict[str, Any]:
        """Lấy cấu hình hiện tại."""
        return {
            'algorithm': self._get_algorithm_type(),
            'typhoon_category': self.typhoon_selector.value(),
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
        }

    def _get_algorithm_type(self) -> str:
        """Lấy loại thuật toán được chọn."""
        text = self.algorithm_combo.currentText()
        algo_map = {
            "GBFS + GWO (Lai ghép)": "hybrid",
            "GBFS (Tìm đường tham lam)": "gbfs",
            "GWO (Tối ưu bầy sói)": "gwo"
        }
        return algo_map.get(text, "hybrid")

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
