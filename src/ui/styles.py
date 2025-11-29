"""
Các hằng số kiểu dáng và bảng màu cho giao diện SafeRoute HCM.
Thiết kế tối hiện đại lấy cảm hứng từ GitHub Dark.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ColorPalette:
    """Bảng màu chính của ứng dụng."""
    # Nền
    background: str = '#0d1117'          # Nền tối đậm
    surface: str = '#161b22'             # Panel nổi
    surface_light: str = '#21262d'       # Card, control
    surface_hover: str = '#30363d'       # Hover state
    border: str = '#30363d'              # Viền nhẹ
    border_light: str = '#484f58'        # Viền sáng hơn

    # Màu chính
    primary: str = '#58a6ff'             # Màu chính (xanh dương)
    primary_dark: str = '#388bfd'        # Hover
    primary_muted: str = '#1f6feb'       # Nhạt

    # Trạng thái
    success: str = '#3fb950'             # Đường an toàn, nơi trú ẩn
    success_dark: str = '#238636'        # Hover
    warning: str = '#d29922'             # Khu vực cảnh báo
    warning_dark: str = '#9e6a03'        # Hover
    danger: str = '#f85149'              # Vùng nguy hiểm
    danger_dark: str = '#da3633'         # Hover
    info: str = '#58a6ff'                # Thông tin

    # Văn bản
    text: str = '#c9d1d9'                # Văn bản chính
    text_secondary: str = '#8b949e'      # Văn bản phụ
    text_muted: str = '#6e7681'          # Văn bản mờ
    text_inverse: str = '#0d1117'        # Văn bản trên nền sáng

    # Đặc biệt cho bản đồ
    cyan: str = '#39c5cf'                # Particle sơ tán
    cyan_dark: str = '#2da8b1'           # Particle sơ tán (dark)
    purple: str = '#a371f7'              # GWO visualization
    purple_dark: str = '#8957e5'         # GWO (dark)
    orange: str = '#f0883e'              # Đường bận
    orange_dark: str = '#d47616'         # Hover

    # Gradient
    gradient_safe_start: str = '#238636'
    gradient_safe_end: str = '#3fb950'
    gradient_danger_start: str = '#f85149'
    gradient_danger_end: str = '#da3633'


# Singleton instance
COLORS = ColorPalette()


def get_color(name: str) -> str:
    """Lấy màu theo tên từ palette."""
    return getattr(COLORS, name, COLORS.text)


def rgba_from_hex(hex_color: str, alpha: float = 1.0) -> str:
    """Chuyển đổi hex sang rgba CSS."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Chuyển đổi hex sang RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16)
    )


def rgb_to_qcolor(r: int, g: int, b: int, a: int = 255):
    """Chuyển đổi RGB sang QColor - import động để tránh import lỗi."""
    try:
        from PyQt6.QtGui import QColor
        return QColor(r, g, b, a)
    except ImportError:
        return None


def hex_to_qcolor(hex_color: str, alpha: int = 255):
    """Chuyển đổi hex sang QColor."""
    r, g, b = hex_to_rgb(hex_color)
    return rgb_to_qcolor(r, g, b, alpha)


# QSS Stylesheet cho toàn bộ ứng dụng
MAIN_STYLESHEET = f"""
/* ========== Global ========== */
QMainWindow, QWidget {{
    background-color: {COLORS.background};
    color: {COLORS.text};
    font-family: 'Segoe UI', 'SF Pro Display', -apple-system, sans-serif;
    font-size: 13px;
}}

/* ========== Labels ========== */
QLabel {{
    color: {COLORS.text};
    background: transparent;
    padding: 2px;
}}

QLabel[heading="true"] {{
    font-size: 18px;
    font-weight: bold;
    color: {COLORS.text};
}}

QLabel[subheading="true"] {{
    font-size: 14px;
    font-weight: 600;
    color: {COLORS.text_secondary};
}}

QLabel[muted="true"] {{
    color: {COLORS.text_muted};
    font-size: 12px;
}}

/* ========== Buttons ========== */
QPushButton {{
    background-color: {COLORS.surface_light};
    color: {COLORS.text};
    border: 1px solid {COLORS.border};
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
    min-height: 32px;
}}

QPushButton:hover {{
    background-color: {COLORS.surface_hover};
    border-color: {COLORS.border_light};
}}

QPushButton:pressed {{
    background-color: {COLORS.surface};
}}

QPushButton:disabled {{
    background-color: {COLORS.surface};
    color: {COLORS.text_muted};
    border-color: {COLORS.border};
}}

QPushButton[primary="true"] {{
    background-color: {COLORS.success_dark};
    color: white;
    border: none;
}}

QPushButton[primary="true"]:hover {{
    background-color: {COLORS.success};
}}

QPushButton[danger="true"] {{
    background-color: {COLORS.danger_dark};
    color: white;
    border: none;
}}

QPushButton[danger="true"]:hover {{
    background-color: {COLORS.danger};
}}

QPushButton[flat="true"] {{
    background: transparent;
    border: none;
}}

QPushButton[flat="true"]:hover {{
    background-color: {COLORS.surface_light};
}}

/* ========== ComboBox ========== */
QComboBox {{
    background-color: {COLORS.surface_light};
    color: {COLORS.text};
    border: 1px solid {COLORS.border};
    border-radius: 6px;
    padding: 6px 12px;
    min-height: 32px;
}}

QComboBox:hover {{
    border-color: {COLORS.border_light};
}}

QComboBox:focus {{
    border-color: {COLORS.primary};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 30px;
    border-left: 1px solid {COLORS.border};
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}}

QComboBox::down-arrow {{
    width: 12px;
    height: 12px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS.surface};
    color: {COLORS.text};
    border: 1px solid {COLORS.border};
    selection-background-color: {COLORS.primary_muted};
    selection-color: white;
}}

/* ========== Slider ========== */
QSlider::groove:horizontal {{
    border: none;
    height: 6px;
    background: {COLORS.surface_light};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {COLORS.primary};
    border: none;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background: {COLORS.primary_dark};
}}

QSlider::sub-page:horizontal {{
    background: {COLORS.primary_muted};
    border-radius: 3px;
}}

/* ========== SpinBox ========== */
QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS.surface_light};
    color: {COLORS.text};
    border: 1px solid {COLORS.border};
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 28px;
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COLORS.primary};
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: {COLORS.surface_hover};
    border: none;
    width: 20px;
}}

/* ========== GroupBox ========== */
QGroupBox {{
    background-color: {COLORS.surface};
    border: 1px solid {COLORS.border};
    border-radius: 8px;
    margin-top: 12px;
    padding: 16px;
    padding-top: 24px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 8px;
    color: {COLORS.text};
    background-color: {COLORS.surface};
}}

/* ========== TabWidget ========== */
QTabWidget::pane {{
    background-color: {COLORS.surface};
    border: 1px solid {COLORS.border};
    border-radius: 8px;
    top: -1px;
}}

QTabBar::tab {{
    background-color: transparent;
    color: {COLORS.text_secondary};
    border: none;
    padding: 10px 20px;
    margin-right: 4px;
    font-weight: 500;
}}

QTabBar::tab:selected {{
    color: {COLORS.text};
    border-bottom: 2px solid {COLORS.primary};
}}

QTabBar::tab:hover:!selected {{
    color: {COLORS.text};
    background-color: {COLORS.surface_light};
}}

/* ========== ScrollBar ========== */
QScrollBar:vertical {{
    background: {COLORS.surface};
    width: 12px;
    border-radius: 6px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {COLORS.surface_hover};
    border-radius: 6px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background: {COLORS.border_light};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background: {COLORS.surface};
    height: 12px;
    border-radius: 6px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {COLORS.surface_hover};
    border-radius: 6px;
    min-width: 30px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {COLORS.border_light};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ========== ProgressBar ========== */
QProgressBar {{
    background-color: {COLORS.surface_light};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {COLORS.success};
    border-radius: 4px;
}}

/* ========== Frame / Panel ========== */
QFrame[panel="true"] {{
    background-color: {COLORS.surface};
    border: 1px solid {COLORS.border};
    border-radius: 8px;
}}

QFrame[card="true"] {{
    background-color: {COLORS.surface_light};
    border: 1px solid {COLORS.border};
    border-radius: 6px;
    padding: 12px;
}}

/* ========== Splitter ========== */
QSplitter::handle {{
    background-color: {COLORS.border};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

/* ========== StatusBar ========== */
QStatusBar {{
    background-color: {COLORS.surface};
    border-top: 1px solid {COLORS.border};
    color: {COLORS.text_secondary};
    padding: 4px 12px;
}}

QStatusBar::item {{
    border: none;
}}

/* ========== Menu ========== */
QMenuBar {{
    background-color: {COLORS.surface};
    border-bottom: 1px solid {COLORS.border};
    padding: 4px;
}}

QMenuBar::item {{
    background: transparent;
    padding: 6px 12px;
    border-radius: 4px;
}}

QMenuBar::item:selected {{
    background-color: {COLORS.surface_light};
}}

QMenu {{
    background-color: {COLORS.surface};
    border: 1px solid {COLORS.border};
    border-radius: 8px;
    padding: 4px;
}}

QMenu::item {{
    padding: 8px 24px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background-color: {COLORS.primary_muted};
    color: white;
}}

QMenu::separator {{
    height: 1px;
    background: {COLORS.border};
    margin: 4px 8px;
}}

/* ========== ToolTip ========== */
QToolTip {{
    background-color: {COLORS.surface};
    color: {COLORS.text};
    border: 1px solid {COLORS.border};
    border-radius: 6px;
    padding: 6px 10px;
}}

/* ========== LineEdit ========== */
QLineEdit {{
    background-color: {COLORS.surface_light};
    color: {COLORS.text};
    border: 1px solid {COLORS.border};
    border-radius: 6px;
    padding: 8px 12px;
    selection-background-color: {COLORS.primary_muted};
}}

QLineEdit:focus {{
    border-color: {COLORS.primary};
}}

QLineEdit:disabled {{
    background-color: {COLORS.surface};
    color: {COLORS.text_muted};
}}

/* ========== TextEdit ========== */
QTextEdit, QPlainTextEdit {{
    background-color: {COLORS.surface_light};
    color: {COLORS.text};
    border: 1px solid {COLORS.border};
    border-radius: 6px;
    padding: 8px;
    selection-background-color: {COLORS.primary_muted};
}}

QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {COLORS.primary};
}}

/* ========== CheckBox ========== */
QCheckBox {{
    color: {COLORS.text};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {COLORS.border};
    border-radius: 4px;
    background-color: {COLORS.surface_light};
}}

QCheckBox::indicator:checked {{
    background-color: {COLORS.primary};
    border-color: {COLORS.primary};
}}

QCheckBox::indicator:hover {{
    border-color: {COLORS.primary};
}}

/* ========== RadioButton ========== */
QRadioButton {{
    color: {COLORS.text};
    spacing: 8px;
}}

QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {COLORS.border};
    border-radius: 9px;
    background-color: {COLORS.surface_light};
}}

QRadioButton::indicator:checked {{
    background-color: {COLORS.primary};
    border-color: {COLORS.primary};
}}

QRadioButton::indicator:hover {{
    border-color: {COLORS.primary};
}}

/* ========== Table ========== */
QTableWidget, QTableView {{
    background-color: {COLORS.surface};
    alternate-background-color: {COLORS.surface_light};
    border: 1px solid {COLORS.border};
    border-radius: 8px;
    gridline-color: {COLORS.border};
    selection-background-color: {COLORS.primary_muted};
}}

QHeaderView::section {{
    background-color: {COLORS.surface_light};
    color: {COLORS.text};
    border: none;
    border-bottom: 1px solid {COLORS.border};
    padding: 8px 12px;
    font-weight: 600;
}}

QHeaderView::section:hover {{
    background-color: {COLORS.surface_hover};
}}

/* ========== Tree ========== */
QTreeWidget, QTreeView {{
    background-color: {COLORS.surface};
    border: 1px solid {COLORS.border};
    border-radius: 8px;
    selection-background-color: {COLORS.primary_muted};
}}

QTreeView::item {{
    padding: 4px;
    border-radius: 4px;
}}

QTreeView::item:hover {{
    background-color: {COLORS.surface_light};
}}

QTreeView::item:selected {{
    background-color: {COLORS.primary_muted};
}}

/* ========== List ========== */
QListWidget, QListView {{
    background-color: {COLORS.surface};
    border: 1px solid {COLORS.border};
    border-radius: 8px;
    selection-background-color: {COLORS.primary_muted};
}}

QListWidget::item {{
    padding: 8px;
    border-radius: 4px;
}}

QListWidget::item:hover {{
    background-color: {COLORS.surface_light};
}}

QListWidget::item:selected {{
    background-color: {COLORS.primary_muted};
}}
"""


# Các hằng số kích thước
class Sizes:
    """Các hằng số kích thước cho giao diện."""
    # Padding & Margin
    PADDING_XS = 4
    PADDING_SM = 8
    PADDING_MD = 12
    PADDING_LG = 16
    PADDING_XL = 24

    # Border radius
    RADIUS_SM = 4
    RADIUS_MD = 6
    RADIUS_LG = 8
    RADIUS_XL = 12

    # Font sizes
    FONT_XS = 10
    FONT_SM = 12
    FONT_MD = 13
    FONT_LG = 16
    FONT_XL = 18
    FONT_XXL = 24

    # Component sizes
    BUTTON_HEIGHT = 32
    INPUT_HEIGHT = 36
    ICON_SM = 16
    ICON_MD = 24
    ICON_LG = 32

    # Layout
    SIDEBAR_WIDTH = 320
    DASHBOARD_HEIGHT = 200
    STATUS_BAR_HEIGHT = 28


# Các hằng số hoạt hình
class Animation:
    """Các hằng số cho hoạt hình."""
    # Thời gian (ms)
    DURATION_FAST = 150
    DURATION_NORMAL = 250
    DURATION_SLOW = 400

    # FPS
    TARGET_FPS = 60
    FRAME_TIME_MS = 16  # ~60 FPS

    # Particle
    PARTICLE_SIZE = 4
    PARTICLE_TRAIL_LENGTH = 5
    PARTICLE_SPEED = 2.0

    # Pulse
    HAZARD_PULSE_MIN = 0.3
    HAZARD_PULSE_MAX = 0.8
    HAZARD_PULSE_SPEED = 0.05


# Map visualization constants
class MapStyle:
    """Các hằng số cho trực quan hóa bản đồ."""
    # Độ dày đường
    ROAD_WIDTH_MIN = 1
    ROAD_WIDTH_MAX = 8
    ROAD_WIDTH_FACTOR = 0.001  # flow * factor = width

    # Node sizes
    ZONE_SIZE_MIN = 8
    ZONE_SIZE_MAX = 24
    SHELTER_SIZE_MIN = 12
    SHELTER_SIZE_MAX = 32
    HAZARD_SIZE_FACTOR = 50  # km to pixels

    # Zoom
    ZOOM_MIN = 0.1
    ZOOM_MAX = 10.0
    ZOOM_STEP = 1.2

    # Grid
    GRID_SIZE = 50
    GRID_COLOR = COLORS.border
    GRID_OPACITY = 0.3
