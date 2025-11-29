"""
Widget bản đồ tương tác cho trực quan hóa mạng lưới sơ tán.
Sử dụng QGraphicsView với tăng tốc OpenGL cho hiệu suất cao.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math

from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsRectItem,
    QGraphicsTextItem, QGraphicsItemGroup, QWidget, QVBoxLayout,
    QGraphicsPolygonItem, QHBoxLayout, QPushButton, QLabel, QFrame
)
from PyQt6.QtCore import (
    Qt, QPointF, QRectF, QTimer, pyqtSignal, QLineF
)
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPainterPath, QFont,
    QRadialGradient, QLinearGradient, QPolygonF, QTransform, QWheelEvent
)

from .styles import COLORS, hex_to_rgb, Animation, MapStyle, Sizes
from ..models.network import EvacuationNetwork
from ..models.node import Node, NodeType, PopulationZone, Shelter, HazardZone


def hex_to_qcolor(hex_color: str, alpha: int = 255) -> QColor:
    """Chuyển đổi mã hex sang QColor."""
    r, g, b = hex_to_rgb(hex_color)
    return QColor(r, g, b, alpha)


class PopulationZoneItem(QGraphicsEllipseItem):
    """Hiển thị khu vực dân cư trên bản đồ."""

    def __init__(self, zone: PopulationZone, x: float, y: float, size: float):
        super().__init__(-size/2, -size/2, size, size)
        self.zone = zone
        self.base_size = size

        self.setPos(x, y)
        self.setZValue(10)

        # Styling
        color = hex_to_qcolor(COLORS.cyan, 200)
        self.setBrush(QBrush(color))
        self.setPen(QPen(hex_to_qcolor(COLORS.cyan_dark), 2))

        self.setToolTip(
            f"<b>{zone.name or zone.id}</b><br>"
            f"Dân số: {zone.population:,}<br>"
            f"Quận: {zone.district_name}"
        )
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event):
        self.setPen(QPen(hex_to_qcolor(COLORS.primary), 3))
        self.setScale(1.3)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPen(QPen(hex_to_qcolor(COLORS.cyan_dark), 2))
        self.setScale(1.0)
        super().hoverLeaveEvent(event)

    def update_progress(self, evacuated: int):
        """Cập nhật hiển thị tiến độ sơ tán."""
        self.zone.evacuated = evacuated
        progress = self.zone.evacuation_progress

        if progress >= 0.8:
            color = hex_to_qcolor(COLORS.success, 200)
        elif progress >= 0.5:
            color = hex_to_qcolor(COLORS.warning, 200)
        else:
            color = hex_to_qcolor(COLORS.cyan, 200)

        self.setBrush(QBrush(color))


class ShelterItem(QGraphicsRectItem):
    """Hiển thị nơi trú ẩn trên bản đồ."""

    def __init__(self, shelter: Shelter, x: float, y: float, size: float):
        super().__init__(-size/2, -size/2, size, size)
        self.shelter = shelter
        self.base_size = size

        self.setPos(x, y)
        self.setZValue(15)

        self.setBrush(QBrush(hex_to_qcolor(COLORS.success, 220)))
        self.setPen(QPen(hex_to_qcolor(COLORS.success_dark), 2))

        self._update_tooltip()
        self.setAcceptHoverEvents(True)

    def _update_tooltip(self):
        self.setToolTip(
            f"<b>{self.shelter.name or self.shelter.id}</b><br>"
            f"Loại: {self.shelter.shelter_type}<br>"
            f"Sức chứa: {self.shelter.current_occupancy:,}/{self.shelter.capacity:,}"
        )

    def hoverEnterEvent(self, event):
        self.setPen(QPen(hex_to_qcolor(COLORS.primary), 3))
        self.setScale(1.3)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPen(QPen(hex_to_qcolor(COLORS.success_dark), 2))
        self.setScale(1.0)
        super().hoverLeaveEvent(event)

    def update_occupancy(self, occupancy: int):
        """Cập nhật hiển thị mức lấp đầy."""
        self.shelter.current_occupancy = occupancy
        rate = self.shelter.occupancy_rate

        if rate >= 0.9:
            color = hex_to_qcolor(COLORS.danger, 220)
        elif rate >= 0.7:
            color = hex_to_qcolor(COLORS.warning, 220)
        else:
            color = hex_to_qcolor(COLORS.success, 220)

        self.setBrush(QBrush(color))
        self._update_tooltip()


class HazardZoneItem(QGraphicsEllipseItem):
    """Hiển thị vùng nguy hiểm với hiệu ứng pulse."""

    def __init__(self, hazard: HazardZone, x: float, y: float, radius_pixels: float):
        super().__init__(-radius_pixels, -radius_pixels,
                         radius_pixels * 2, radius_pixels * 2)
        self.hazard = hazard
        self.base_radius = radius_pixels

        self.setPos(x, y)
        self.setZValue(5)

        self.pulse_phase = 0.5
        self.pulse_direction = 1

        self._update_appearance()

    def _update_appearance(self):
        """Cập nhật giao diện với gradient và opacity."""
        gradient = QRadialGradient(0, 0, self.base_radius)
        base_color = hex_to_qcolor(COLORS.danger)

        alpha = int(180 * self.pulse_phase)
        gradient.setColorAt(0, QColor(base_color.red(), base_color.green(),
                                      base_color.blue(), alpha))
        gradient.setColorAt(0.7, QColor(base_color.red(), base_color.green(),
                                        base_color.blue(), int(alpha * 0.5)))
        gradient.setColorAt(1, QColor(base_color.red(), base_color.green(),
                                      base_color.blue(), 0))

        self.setBrush(QBrush(gradient))
        self.setPen(QPen(hex_to_qcolor(COLORS.danger, int(200 * self.pulse_phase)), 2))

        self.setToolTip(
            f"<b>Vùng Nguy Hiểm</b><br>"
            f"Loại: {self.hazard.hazard_type}<br>"
            f"Bán kính: {self.hazard.radius_km:.1f} km<br>"
            f"Rủi ro: {self.hazard.risk_level:.0%}"
        )

    def pulse_tick(self):
        """Cập nhật hiệu ứng pulse."""
        self.pulse_phase += 0.03 * self.pulse_direction

        if self.pulse_phase >= 1.0:
            self.pulse_direction = -1
        elif self.pulse_phase <= 0.3:
            self.pulse_direction = 1

        self._update_appearance()

    def update_hazard(self, radius_pixels: float, risk_level: float):
        """Cập nhật kích thước và mức độ nguy hiểm."""
        self.base_radius = radius_pixels
        self.hazard.risk_level = risk_level
        self.setRect(-radius_pixels, -radius_pixels,
                     radius_pixels * 2, radius_pixels * 2)
        self._update_appearance()


class RouteItem(QGraphicsPathItem):
    """Hiển thị tuyến đường sơ tán."""

    def __init__(self, path_points: List[QPointF], flow: int = 0, risk: float = 0.0):
        super().__init__()
        self.path_points = path_points
        self.flow = flow
        self.risk = risk

        self.setZValue(8)
        self._build_path()
        self._update_style()

    def _build_path(self):
        """Xây dựng đường Bezier mượt qua các điểm."""
        if len(self.path_points) < 2:
            return

        path = QPainterPath()
        path.moveTo(self.path_points[0])

        for i in range(1, len(self.path_points)):
            path.lineTo(self.path_points[i])

        self.setPath(path)

    def _update_style(self):
        """Cập nhật kiểu dáng dựa trên flow và risk."""
        width = max(3, min(12, 3 + self.flow * 0.0005))

        if self.risk > 0.7:
            color = hex_to_qcolor(COLORS.danger, 220)
        elif self.risk > 0.4:
            color = hex_to_qcolor(COLORS.warning, 220)
        else:
            color = hex_to_qcolor(COLORS.success, 220)

        self.setPen(QPen(color, width, Qt.PenStyle.SolidLine,
                         Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))

    def update_flow(self, flow: int, risk: float = None):
        """Cập nhật lưu lượng và vẽ lại."""
        self.flow = flow
        if risk is not None:
            self.risk = risk
        self._update_style()

    def get_point_at_progress(self, progress: float) -> QPointF:
        """Lấy điểm trên đường tại vị trí progress (0-1)."""
        if not self.path_points or len(self.path_points) < 2:
            return QPointF(0, 0)

        total_length = 0.0
        segments = []
        for i in range(len(self.path_points) - 1):
            p1 = self.path_points[i]
            p2 = self.path_points[i + 1]
            length = math.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
            segments.append((p1, p2, length))
            total_length += length

        if total_length == 0:
            return self.path_points[0]

        target_dist = progress * total_length
        current_dist = 0.0

        for p1, p2, length in segments:
            if current_dist + length >= target_dist:
                segment_progress = (target_dist - current_dist) / length if length > 0 else 0
                x = p1.x() + (p2.x() - p1.x()) * segment_progress
                y = p1.y() + (p2.y() - p1.y()) * segment_progress
                return QPointF(x, y)
            current_dist += length

        return self.path_points[-1]


class EvacueeParticle(QGraphicsEllipseItem):
    """Hạt đại diện cho nhóm người sơ tán đang di chuyển."""

    def __init__(self, size: float = 6):
        super().__init__(-size/2, -size/2, size, size)
        self.setZValue(20)

        color = hex_to_qcolor(COLORS.cyan, 255)
        self.setBrush(QBrush(color))
        self.setPen(QPen(hex_to_qcolor(COLORS.text), 1))

    def set_color(self, color: QColor):
        """Cập nhật màu của hạt."""
        self.setBrush(QBrush(color))


class MapCanvas(QGraphicsView):
    """
    Canvas bản đồ hiệu suất cao.
    Hiển thị mạng lưới sơ tán với hoạt hình thời gian thực.
    """

    node_clicked = pyqtSignal(str, str)
    zoom_changed = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # View settings
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        # Background with grid
        self.setBackgroundBrush(QBrush(hex_to_qcolor(COLORS.background)))

        # State
        self._network: Optional[EvacuationNetwork] = None
        self._zoom_level = 1.0

        # Item storage
        self._zone_items: Dict[str, PopulationZoneItem] = {}
        self._shelter_items: Dict[str, ShelterItem] = {}
        self._hazard_items: Dict[int, HazardZoneItem] = {}
        self._route_items: Dict[str, RouteItem] = {}
        self._particles: List[EvacueeParticle] = []
        self._particle_pool: List[EvacueeParticle] = []

        # Coordinate transformation - use fixed scale
        self._scale_factor = 5000.0  # pixels per degree

        # Animation timer
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._animation_tick)
        self._animation_running = False

        # Active evacuee groups for animation
        self._active_groups: List[Dict[str, Any]] = []

    def drawBackground(self, painter: QPainter, rect: QRectF):
        """Vẽ nền với lưới."""
        # Fill background
        painter.fillRect(rect, hex_to_qcolor(COLORS.background))

        # Draw grid
        grid_size = 50  # pixels between grid lines
        grid_color = hex_to_qcolor(COLORS.surface, 100)
        grid_pen = QPen(grid_color, 1, Qt.PenStyle.DotLine)
        painter.setPen(grid_pen)

        # Calculate visible grid lines
        left = int(rect.left() / grid_size) * grid_size
        top = int(rect.top() / grid_size) * grid_size

        # Vertical lines
        x = left
        while x < rect.right():
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            x += grid_size

        # Horizontal lines
        y = top
        while y < rect.bottom():
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
            y += grid_size

    def set_network(self, network: EvacuationNetwork):
        """Thiết lập mạng lưới và vẽ lên canvas."""
        self._network = network
        self._clear_all()
        self._draw_network()

        # Set scene rect based on items
        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-100, -100, 100, 100))

        # Fit to view
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_level = 1.0

    def _clear_all(self):
        """Xóa tất cả các item khỏi scene."""
        self._scene.clear()
        self._zone_items.clear()
        self._shelter_items.clear()
        self._hazard_items.clear()
        self._route_items.clear()
        self._particles.clear()

    def _lat_lon_to_pixel(self, lat: float, lon: float) -> QPointF:
        """Chuyển đổi lat/lon sang tọa độ pixel."""
        x = lon * self._scale_factor
        y = -lat * self._scale_factor  # Flip Y axis
        return QPointF(x, y)

    def _draw_network(self):
        """Vẽ toàn bộ mạng lưới lên canvas."""
        if not self._network:
            return

        # 1. Vẽ các cạnh (đường)
        self._draw_edges()

        # 2. Vẽ vùng nguy hiểm
        self._draw_hazards()

        # 3. Vẽ các nút
        self._draw_nodes()

    def _draw_edges(self):
        """Vẽ các đường trong mạng lưới."""
        if not self._network:
            return

        for edge in self._network.get_edges():
            source = self._network.get_node(edge.source_id)
            target = self._network.get_node(edge.target_id)

            if not source or not target:
                continue

            p1 = self._lat_lon_to_pixel(source.lat, source.lon)
            p2 = self._lat_lon_to_pixel(target.lat, target.lon)

            path = QPainterPath()
            path.moveTo(p1)
            path.lineTo(p2)

            item = QGraphicsPathItem(path)

            # Color based on flood risk
            if edge.flood_risk > 0.7:
                color = hex_to_qcolor(COLORS.danger, 120)
                width = 2
            elif edge.flood_risk > 0.3:
                color = hex_to_qcolor(COLORS.warning, 100)
                width = 1.5
            else:
                color = hex_to_qcolor(COLORS.surface_hover, 80)
                width = 1

            item.setPen(QPen(color, width))
            item.setZValue(1)
            self._scene.addItem(item)

    def _draw_hazards(self):
        """Vẽ các vùng nguy hiểm."""
        if not self._network:
            return

        for i, hazard in enumerate(self._network.get_hazard_zones()):
            pos = self._lat_lon_to_pixel(hazard.center_lat, hazard.center_lon)

            # Convert km to pixels (roughly 1 degree = 111km)
            radius_pixels = hazard.radius_km * self._scale_factor / 111.0

            item = HazardZoneItem(hazard, pos.x(), pos.y(), radius_pixels)
            self._scene.addItem(item)
            self._hazard_items[i] = item

    def _draw_nodes(self):
        """Vẽ các nút (khu vực dân cư, nơi trú ẩn)."""
        if not self._network:
            return

        zones = self._network.get_population_zones()
        shelters = self._network.get_shelters()

        max_pop = max((z.population for z in zones), default=1)
        for zone in zones:
            pos = self._lat_lon_to_pixel(zone.lat, zone.lon)

            size_ratio = zone.population / max_pop if max_pop > 0 else 0.5
            size = 15 + 25 * size_ratio

            item = PopulationZoneItem(zone, pos.x(), pos.y(), size)
            self._scene.addItem(item)
            self._zone_items[zone.id] = item

        max_cap = max((s.capacity for s in shelters), default=1)
        for shelter in shelters:
            pos = self._lat_lon_to_pixel(shelter.lat, shelter.lon)

            size_ratio = shelter.capacity / max_cap if max_cap > 0 else 0.5
            size = 18 + 30 * size_ratio

            item = ShelterItem(shelter, pos.x(), pos.y(), size)
            self._scene.addItem(item)
            self._shelter_items[shelter.id] = item

    def add_route(self, route_id: str, path_node_ids: List[str], flow: int = 0, risk: float = 0.0):
        """Thêm tuyến đường sơ tán vào bản đồ."""
        if not self._network:
            return

        path_points = []
        for node_id in path_node_ids:
            node = self._network.get_node(node_id)
            if node:
                pos = self._lat_lon_to_pixel(node.lat, node.lon)
                path_points.append(pos)

        if len(path_points) < 2:
            return

        item = RouteItem(path_points, flow, risk)
        self._scene.addItem(item)
        self._route_items[route_id] = item

        # Add particles for this route
        self._add_route_particles(route_id, item, flow)

    def _add_route_particles(self, route_id: str, route_item: RouteItem, flow: int):
        """Add animated particles along a route."""
        # Add 3-5 particles per route based on flow
        num_particles = min(5, max(2, flow // 2000))

        for i in range(num_particles):
            particle = EvacueeParticle(8)
            self._scene.addItem(particle)
            self._particles.append(particle)

            # Start at different positions along the route
            initial_progress = i / num_particles

            self._active_groups.append({
                'route_id': route_id,
                'count': flow // num_particles,
                'progress': initial_progress,
                'particle': particle,
                'route_item': route_item,
                'speed': 0.005 + (i * 0.001)  # Slightly different speeds
            })

            pos = route_item.get_point_at_progress(initial_progress)
            particle.setPos(pos)
            particle.show()

    def clear_routes(self):
        """Xóa tất cả các tuyến đường."""
        for item in self._route_items.values():
            self._scene.removeItem(item)
        self._route_items.clear()

        # Clear particles
        for particle in self._particles:
            self._scene.removeItem(particle)
        self._particles.clear()
        self._active_groups.clear()

    def start_animation(self):
        """Bắt đầu hoạt hình."""
        if not self._animation_running:
            self._animation_timer.start(33)  # ~30 FPS
            self._animation_running = True

    def stop_animation(self):
        """Dừng hoạt hình."""
        self._animation_timer.stop()
        self._animation_running = False

    def _animation_tick(self):
        """Cập nhật hoạt hình mỗi frame."""
        # Pulse hazard zones
        for hazard_item in self._hazard_items.values():
            hazard_item.pulse_tick()

        # Move evacuee particles
        for group in self._active_groups:
            group['progress'] += group.get('speed', 0.01)

            if group['progress'] >= 1.0:
                group['progress'] = 0.0  # Loop back

            pos = group['route_item'].get_point_at_progress(group['progress'])
            group['particle'].setPos(pos)

            # Color based on progress
            if group['progress'] > 0.7:
                group['particle'].set_color(hex_to_qcolor(COLORS.success, 255))
            elif group['progress'] > 0.3:
                group['particle'].set_color(hex_to_qcolor(COLORS.warning, 255))
            else:
                group['particle'].set_color(hex_to_qcolor(COLORS.cyan, 255))

    def update_zone_progress(self, zone_id: str, evacuated: int):
        """Cập nhật tiến độ sơ tán của khu vực."""
        if zone_id in self._zone_items:
            self._zone_items[zone_id].update_progress(evacuated)

    def update_shelter_occupancy(self, shelter_id: str, occupancy: int):
        """Cập nhật mức lấp đầy của nơi trú ẩn."""
        if shelter_id in self._shelter_items:
            self._shelter_items[shelter_id].update_occupancy(occupancy)

    def update_hazard(self, hazard_index: int, radius_km: float, risk_level: float):
        """Cập nhật vùng nguy hiểm."""
        if hazard_index in self._hazard_items:
            radius_pixels = radius_km * self._scale_factor / 111.0
            self._hazard_items[hazard_index].update_hazard(radius_pixels, risk_level)

    def fit_to_view(self):
        """Zoom để vừa toàn bộ mạng lưới trong viewport."""
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_level = 1.0
        self.zoom_changed.emit(self._zoom_level)

    def zoom_in(self):
        """Zoom in."""
        factor = 1.25
        self._zoom_level *= factor
        self.scale(factor, factor)
        self.zoom_changed.emit(self._zoom_level)

    def zoom_out(self):
        """Zoom out."""
        factor = 0.8
        self._zoom_level *= factor
        self.scale(factor, factor)
        self.zoom_changed.emit(self._zoom_level)

    def wheelEvent(self, event: QWheelEvent):
        """Xử lý scroll để zoom."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in()
        elif delta < 0:
            self.zoom_out()


class MapWidget(QWidget):
    """Widget container cho bản đồ với controls."""

    node_clicked = pyqtSignal(str, str)
    zoom_changed = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS.surface};
                border-radius: 6px;
                padding: 4px;
            }}
        """)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        toolbar_layout.setSpacing(8)

        # Zoom buttons with clear styling
        btn_style = f"""
            QPushButton {{
                background-color: {COLORS.surface_hover};
                color: {COLORS.text};
                border: 1px solid {COLORS.border};
                border-radius: 4px;
                font-size: 16px;
                font-weight: bold;
                padding: 4px;
            }}
            QPushButton:hover {{
                background-color: {COLORS.primary};
                color: white;
            }}
            QPushButton:pressed {{
                background-color: {COLORS.primary_dark};
            }}
        """

        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedSize(36, 36)
        self.zoom_in_btn.setStyleSheet(btn_style)
        self.zoom_in_btn.setToolTip("Phóng to")
        self.zoom_in_btn.clicked.connect(self._on_zoom_in)
        toolbar_layout.addWidget(self.zoom_in_btn)

        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setFixedSize(36, 36)
        self.zoom_out_btn.setStyleSheet(btn_style)
        self.zoom_out_btn.setToolTip("Thu nhỏ")
        self.zoom_out_btn.clicked.connect(self._on_zoom_out)
        toolbar_layout.addWidget(self.zoom_out_btn)

        self.fit_btn = QPushButton("⊡")
        self.fit_btn.setFixedSize(36, 36)
        self.fit_btn.setStyleSheet(btn_style)
        self.fit_btn.setToolTip("Vừa khung hình")
        self.fit_btn.clicked.connect(self._on_fit)
        toolbar_layout.addWidget(self.fit_btn)

        toolbar_layout.addSpacing(20)

        # Legend
        legend_style = f"color: {COLORS.text_muted}; font-size: 10px;"

        # Cyan circle = population zone
        pop_legend = QLabel("● Khu dân cư")
        pop_legend.setStyleSheet(f"color: {COLORS.cyan}; font-size: 10px;")
        toolbar_layout.addWidget(pop_legend)

        # Green square = shelter
        shelter_legend = QLabel("■ Nơi trú ẩn")
        shelter_legend.setStyleSheet(f"color: {COLORS.success}; font-size: 10px;")
        toolbar_layout.addWidget(shelter_legend)

        # Red circle = hazard
        hazard_legend = QLabel("◉ Vùng nguy hiểm")
        hazard_legend.setStyleSheet(f"color: {COLORS.danger}; font-size: 10px;")
        toolbar_layout.addWidget(hazard_legend)

        toolbar_layout.addStretch()

        self.info_label = QLabel("Cuộn để zoom, kéo để di chuyển")
        self.info_label.setStyleSheet(f"color: {COLORS.text_muted}; font-size: 10px;")
        toolbar_layout.addWidget(self.info_label)

        layout.addWidget(toolbar)

        # Canvas
        self.canvas = MapCanvas(self)
        self.canvas.node_clicked.connect(self.node_clicked)
        self.canvas.zoom_changed.connect(self.zoom_changed)
        layout.addWidget(self.canvas)

    def _on_zoom_in(self):
        self.canvas.zoom_in()

    def _on_zoom_out(self):
        self.canvas.zoom_out()

    def _on_fit(self):
        self.canvas.fit_to_view()

    def set_network(self, network: EvacuationNetwork):
        """Thiết lập mạng lưới."""
        self.canvas.set_network(network)

    def add_route(self, route_id: str, path_node_ids: List[str], flow: int = 0, risk: float = 0.0):
        """Thêm tuyến đường."""
        self.canvas.add_route(route_id, path_node_ids, flow, risk)

    def clear_routes(self):
        """Xóa tuyến đường."""
        self.canvas.clear_routes()

    def start_animation(self):
        """Bắt đầu hoạt hình."""
        self.canvas.start_animation()

    def stop_animation(self):
        """Dừng hoạt hình."""
        self.canvas.stop_animation()

    def add_evacuee_group(self, route_id: str, count: int, path_node_ids: List[str]):
        """Thêm nhóm người sơ tán."""
        # This is now handled automatically in add_route
        pass

    def update_zone_progress(self, zone_id: str, evacuated: int):
        """Cập nhật tiến độ khu vực."""
        self.canvas.update_zone_progress(zone_id, evacuated)

    def update_shelter_occupancy(self, shelter_id: str, occupancy: int):
        """Cập nhật mức lấp đầy nơi trú ẩn."""
        self.canvas.update_shelter_occupancy(shelter_id, occupancy)

    def update_hazard(self, hazard_index: int, radius_km: float, risk_level: float):
        """Cập nhật vùng nguy hiểm."""
        self.canvas.update_hazard(hazard_index, radius_km, risk_level)

    def fit_to_view(self):
        """Zoom vừa viewport."""
        self.canvas.fit_to_view()

    def zoom_in(self):
        """Zoom in."""
        self.canvas.zoom_in()

    def zoom_out(self):
        """Zoom out."""
        self.canvas.zoom_out()
