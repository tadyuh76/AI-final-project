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
    QGraphicsPolygonItem
)
from PyQt6.QtCore import (
    Qt, QPointF, QRectF, QTimer, pyqtSignal, QLineF,
    QPropertyAnimation, QEasingCurve
)
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPainterPath, QFont,
    QRadialGradient, QLinearGradient, QPolygonF, QTransform
)

from .styles import COLORS, hex_to_rgb, Animation, MapStyle, Sizes
from ..models.network import EvacuationNetwork
from ..models.node import Node, NodeType, PopulationZone, Shelter, HazardZone


def hex_to_qcolor(hex_color: str, alpha: int = 255) -> QColor:
    """Chuyển đổi mã hex sang QColor."""
    r, g, b = hex_to_rgb(hex_color)
    return QColor(r, g, b, alpha)


@dataclass
class EvacueeGroup:
    """Nhóm người sơ tán đang di chuyển trên tuyến đường."""
    route_id: str
    count: int
    progress: float  # 0.0 - 1.0
    path_points: List[QPointF]
    color: QColor


class PopulationZoneItem(QGraphicsEllipseItem):
    """Hiển thị khu vực dân cư trên bản đồ."""

    def __init__(self, zone: PopulationZone, x: float, y: float, size: float):
        super().__init__(-size/2, -size/2, size, size)
        self.zone = zone
        self.base_size = size

        self.setPos(x, y)
        self.setZValue(10)  # Above roads

        # Styling
        color = hex_to_qcolor(COLORS.cyan, 180)
        self.setBrush(QBrush(color))
        self.setPen(QPen(hex_to_qcolor(COLORS.cyan_dark), 2))

        self.setToolTip(
            f"<b>{zone.name or zone.id}</b><br>"
            f"Dan so: {zone.population:,}<br>"
            f"Da so tan: {zone.evacuated:,}<br>"
            f"Quan: {zone.district_name}"
        )

        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event):
        self.setPen(QPen(hex_to_qcolor(COLORS.primary), 3))
        self.setScale(1.2)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPen(QPen(hex_to_qcolor(COLORS.cyan_dark), 2))
        self.setScale(1.0)
        super().hoverLeaveEvent(event)

    def update_progress(self, evacuated: int):
        """Cập nhật hiển thị tiến độ sơ tán."""
        self.zone.evacuated = evacuated
        progress = self.zone.evacuation_progress

        # Đổi màu dựa trên tiến độ
        if progress >= 0.8:
            color = hex_to_qcolor(COLORS.success, 180)
        elif progress >= 0.5:
            color = hex_to_qcolor(COLORS.warning, 180)
        else:
            color = hex_to_qcolor(COLORS.cyan, 180)

        self.setBrush(QBrush(color))


class ShelterItem(QGraphicsRectItem):
    """Hiển thị nơi trú ẩn trên bản đồ."""

    def __init__(self, shelter: Shelter, x: float, y: float, size: float):
        super().__init__(-size/2, -size/2, size, size)
        self.shelter = shelter
        self.base_size = size

        self.setPos(x, y)
        self.setZValue(15)  # Above zones

        # Styling với bo góc (dùng custom paint)
        self.setBrush(QBrush(hex_to_qcolor(COLORS.success, 200)))
        self.setPen(QPen(hex_to_qcolor(COLORS.success_dark), 2))

        self._update_tooltip()
        self.setAcceptHoverEvents(True)

        # Capacity bar
        self.capacity_bar = None

    def _update_tooltip(self):
        self.setToolTip(
            f"<b>{self.shelter.name or self.shelter.id}</b><br>"
            f"Loai: {self.shelter.shelter_type}<br>"
            f"Suc chua: {self.shelter.current_occupancy:,}/{self.shelter.capacity:,}<br>"
            f"Con trong: {self.shelter.available_capacity:,}"
        )

    def hoverEnterEvent(self, event):
        self.setPen(QPen(hex_to_qcolor(COLORS.primary), 3))
        self.setScale(1.2)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPen(QPen(hex_to_qcolor(COLORS.success_dark), 2))
        self.setScale(1.0)
        super().hoverLeaveEvent(event)

    def update_occupancy(self, occupancy: int):
        """Cập nhật hiển thị mức lấp đầy."""
        self.shelter.current_occupancy = occupancy
        rate = self.shelter.occupancy_rate

        # Đổi màu dựa trên mức lấp đầy
        if rate >= 0.9:
            color = hex_to_qcolor(COLORS.danger, 200)
        elif rate >= 0.7:
            color = hex_to_qcolor(COLORS.warning, 200)
        else:
            color = hex_to_qcolor(COLORS.success, 200)

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
        self.setZValue(5)  # Below nodes but above roads

        # Pulse animation state
        self.pulse_phase = 0.0
        self.pulse_direction = 1

        self._update_appearance()

    def _update_appearance(self):
        """Cập nhật giao diện với gradient và opacity."""
        # Radial gradient từ tâm ra ngoài
        gradient = QRadialGradient(0, 0, self.base_radius)
        base_color = hex_to_qcolor(COLORS.danger)

        # Tâm đặc hơn
        gradient.setColorAt(0, QColor(base_color.red(), base_color.green(),
                                      base_color.blue(), int(180 * self.pulse_phase)))
        # Rìa mờ dần
        gradient.setColorAt(0.7, QColor(base_color.red(), base_color.green(),
                                        base_color.blue(), int(100 * self.pulse_phase)))
        gradient.setColorAt(1, QColor(base_color.red(), base_color.green(),
                                      base_color.blue(), 0))

        self.setBrush(QBrush(gradient))
        self.setPen(QPen(hex_to_qcolor(COLORS.danger, int(150 * self.pulse_phase)), 2))

        self.setToolTip(
            f"<b>Vung Nguy Hiem</b><br>"
            f"Loai: {self.hazard.hazard_type}<br>"
            f"Ban kinh: {self.hazard.radius_km:.1f} km<br>"
            f"Muc do rui ro: {self.hazard.risk_level:.0%}"
        )

    def pulse_tick(self):
        """Cập nhật hiệu ứng pulse."""
        self.pulse_phase += Animation.HAZARD_PULSE_SPEED * self.pulse_direction

        if self.pulse_phase >= Animation.HAZARD_PULSE_MAX:
            self.pulse_direction = -1
        elif self.pulse_phase <= Animation.HAZARD_PULSE_MIN:
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

        self.setZValue(2)  # Below nodes
        self._build_path()
        self._update_style()

    def _build_path(self):
        """Xây dựng đường Bezier mượt qua các điểm."""
        if len(self.path_points) < 2:
            return

        path = QPainterPath()
        path.moveTo(self.path_points[0])

        if len(self.path_points) == 2:
            path.lineTo(self.path_points[1])
        else:
            # Bezier curve qua các điểm
            for i in range(1, len(self.path_points) - 1):
                p0 = self.path_points[i - 1]
                p1 = self.path_points[i]
                p2 = self.path_points[i + 1]

                # Control points cho smooth curve
                ctrl1 = QPointF((p0.x() + p1.x()) / 2, (p0.y() + p1.y()) / 2)
                ctrl2 = QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)

                path.quadTo(p1, ctrl2)

            # Đoạn cuối
            path.lineTo(self.path_points[-1])

        self.setPath(path)

    def _update_style(self):
        """Cập nhật kiểu dáng dựa trên flow và risk."""
        # Độ dày dựa trên flow
        width = max(MapStyle.ROAD_WIDTH_MIN,
                    min(MapStyle.ROAD_WIDTH_MAX,
                        MapStyle.ROAD_WIDTH_MIN + self.flow * MapStyle.ROAD_WIDTH_FACTOR))

        # Màu dựa trên risk
        if self.risk > 0.7:
            color = hex_to_qcolor(COLORS.danger, 180)
        elif self.risk > 0.4:
            color = hex_to_qcolor(COLORS.warning, 180)
        else:
            color = hex_to_qcolor(COLORS.success, 180)

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

        # Tính tổng chiều dài
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

        # Tìm điểm tại progress
        target_dist = progress * total_length
        current_dist = 0.0

        for p1, p2, length in segments:
            if current_dist + length >= target_dist:
                # Điểm nằm trong segment này
                segment_progress = (target_dist - current_dist) / length if length > 0 else 0
                x = p1.x() + (p2.x() - p1.x()) * segment_progress
                y = p1.y() + (p2.y() - p1.y()) * segment_progress
                return QPointF(x, y)
            current_dist += length

        return self.path_points[-1]


class EvacueeParticle(QGraphicsEllipseItem):
    """Hạt đại diện cho nhóm người sơ tán đang di chuyển."""

    def __init__(self, size: float = 4):
        super().__init__(-size/2, -size/2, size, size)
        self.setZValue(20)  # On top of everything

        # Default cyan color
        color = hex_to_qcolor(COLORS.cyan, 220)
        self.setBrush(QBrush(color))
        self.setPen(QPen(Qt.PenStyle.NoPen))

    def set_color(self, color: QColor):
        """Cập nhật màu của hạt."""
        self.setBrush(QBrush(color))


class MapCanvas(QGraphicsView):
    """
    Canvas bản đồ hiệu suất cao với hỗ trợ OpenGL.
    Hiển thị mạng lưới sơ tán với hoạt hình thời gian thực.
    """

    # Signals
    node_clicked = pyqtSignal(str, str)  # node_id, node_type
    zoom_changed = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Scene setup
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # View settings
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform |
            QPainter.RenderHint.TextAntialiasing
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        # Background
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

        # Coordinate transformation
        self._bounds: Optional[Tuple[float, float, float, float]] = None
        self._scale_factor = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0

        # Animation timer
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._animation_tick)
        self._animation_running = False

        # Active evacuee groups for animation
        self._active_groups: List[Dict[str, Any]] = []

    def set_network(self, network: EvacuationNetwork):
        """Thiết lập mạng lưới và vẽ lên canvas."""
        self._network = network
        self._clear_all()
        self._calculate_transform()
        self._draw_network()
        self.fit_to_view()

    def _clear_all(self):
        """Xóa tất cả các item khỏi scene."""
        self._scene.clear()
        self._zone_items.clear()
        self._shelter_items.clear()
        self._hazard_items.clear()
        self._route_items.clear()
        self._particles.clear()

    def _calculate_transform(self):
        """Tính toán transform từ lat/lon sang pixel."""
        if not self._network:
            return

        bounds = self._network.get_bounds()
        if bounds[0] == float('inf'):
            return

        self._bounds = bounds
        min_lat, max_lat, min_lon, max_lon = bounds

        # Padding
        padding = 50
        view_width = self.viewport().width() - 2 * padding
        view_height = self.viewport().height() - 2 * padding

        if view_width <= 0 or view_height <= 0:
            view_width = 800
            view_height = 600

        # Scale để vừa viewport
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon

        if lat_range > 0 and lon_range > 0:
            scale_x = view_width / lon_range
            scale_y = view_height / lat_range
            self._scale_factor = min(scale_x, scale_y)
        else:
            self._scale_factor = 1000.0

        # Offset để căn giữa
        self._offset_x = -min_lon * self._scale_factor + padding
        self._offset_y = -min_lat * self._scale_factor + padding

    def _lat_lon_to_pixel(self, lat: float, lon: float) -> QPointF:
        """Chuyển đổi lat/lon sang tọa độ pixel."""
        # Lật trục Y vì pixel tăng xuống dưới
        x = lon * self._scale_factor + self._offset_x
        y = (self._bounds[1] - lat + self._bounds[0]) * self._scale_factor + 50 if self._bounds else 0
        return QPointF(x, y)

    def _draw_network(self):
        """Vẽ toàn bộ mạng lưới lên canvas."""
        if not self._network:
            return

        # 1. Vẽ các cạnh (đường) trước
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

            # Vẽ đường đơn giản (không phải route sơ tán)
            path = QPainterPath()
            path.moveTo(p1)
            path.lineTo(p2)

            item = QGraphicsPathItem(path)
            item.setPen(QPen(hex_to_qcolor(COLORS.surface_light, 100), 1))
            item.setZValue(1)
            self._scene.addItem(item)

    def _draw_hazards(self):
        """Vẽ các vùng nguy hiểm."""
        if not self._network:
            return

        for i, hazard in enumerate(self._network.get_hazard_zones()):
            pos = self._lat_lon_to_pixel(hazard.center_lat, hazard.center_lon)

            # Convert km to pixels (approximate)
            radius_pixels = hazard.radius_km * self._scale_factor * 0.01

            item = HazardZoneItem(hazard, pos.x(), pos.y(), radius_pixels)
            self._scene.addItem(item)
            self._hazard_items[i] = item

    def _draw_nodes(self):
        """Vẽ các nút (khu vực dân cư, nơi trú ẩn)."""
        if not self._network:
            return

        # Vẽ khu vực dân cư
        max_pop = max((z.population for z in self._network.get_population_zones()), default=1)
        for zone in self._network.get_population_zones():
            pos = self._lat_lon_to_pixel(zone.lat, zone.lon)

            # Size dựa trên dân số
            size_ratio = zone.population / max_pop if max_pop > 0 else 0.5
            size = MapStyle.ZONE_SIZE_MIN + (MapStyle.ZONE_SIZE_MAX - MapStyle.ZONE_SIZE_MIN) * size_ratio

            item = PopulationZoneItem(zone, pos.x(), pos.y(), size)
            self._scene.addItem(item)
            self._zone_items[zone.id] = item

        # Vẽ nơi trú ẩn
        max_cap = max((s.capacity for s in self._network.get_shelters()), default=1)
        for shelter in self._network.get_shelters():
            pos = self._lat_lon_to_pixel(shelter.lat, shelter.lon)

            # Size dựa trên sức chứa
            size_ratio = shelter.capacity / max_cap if max_cap > 0 else 0.5
            size = MapStyle.SHELTER_SIZE_MIN + (MapStyle.SHELTER_SIZE_MAX - MapStyle.SHELTER_SIZE_MIN) * size_ratio

            item = ShelterItem(shelter, pos.x(), pos.y(), size)
            self._scene.addItem(item)
            self._shelter_items[shelter.id] = item

    def add_route(self, route_id: str, path_node_ids: List[str], flow: int = 0, risk: float = 0.0):
        """Thêm tuyến đường sơ tán vào bản đồ."""
        if not self._network:
            return

        # Convert node IDs to pixel positions
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

    def clear_routes(self):
        """Xóa tất cả các tuyến đường."""
        for item in self._route_items.values():
            self._scene.removeItem(item)
        self._route_items.clear()

    def update_route_flow(self, route_id: str, flow: int, risk: float = None):
        """Cập nhật lưu lượng của tuyến đường."""
        if route_id in self._route_items:
            self._route_items[route_id].update_flow(flow, risk)

    def start_animation(self):
        """Bắt đầu hoạt hình."""
        if not self._animation_running:
            self._animation_timer.start(Animation.FRAME_TIME_MS)
            self._animation_running = True

    def stop_animation(self):
        """Dừng hoạt hình."""
        self._animation_timer.stop()
        self._animation_running = False

    def add_evacuee_group(self, route_id: str, count: int, path_node_ids: List[str]):
        """Thêm nhóm người sơ tán để animate."""
        if route_id not in self._route_items:
            return

        route_item = self._route_items[route_id]

        # Lấy hoặc tạo particle
        if self._particle_pool:
            particle = self._particle_pool.pop()
        else:
            particle = EvacueeParticle(Animation.PARTICLE_SIZE)
            self._scene.addItem(particle)

        self._particles.append(particle)

        # Thêm vào active groups
        self._active_groups.append({
            'route_id': route_id,
            'count': count,
            'progress': 0.0,
            'particle': particle,
            'route_item': route_item
        })

        # Đặt vị trí ban đầu
        start_pos = route_item.get_point_at_progress(0)
        particle.setPos(start_pos)
        particle.show()

    def _animation_tick(self):
        """Cập nhật hoạt hình mỗi frame."""
        # Pulse hazard zones
        for hazard_item in self._hazard_items.values():
            hazard_item.pulse_tick()

        # Move evacuee particles
        completed_groups = []
        for group in self._active_groups:
            group['progress'] += Animation.PARTICLE_SPEED / 100.0

            if group['progress'] >= 1.0:
                completed_groups.append(group)
            else:
                # Update particle position
                pos = group['route_item'].get_point_at_progress(group['progress'])
                group['particle'].setPos(pos)

                # Color based on progress
                if group['progress'] > 0.8:
                    group['particle'].set_color(hex_to_qcolor(COLORS.success, 220))
                elif group['progress'] > 0.5:
                    group['particle'].set_color(hex_to_qcolor(COLORS.warning, 220))

        # Clean up completed groups
        for group in completed_groups:
            self._active_groups.remove(group)
            particle = group['particle']
            particle.hide()
            self._particles.remove(particle)
            self._particle_pool.append(particle)

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
            radius_pixels = radius_km * self._scale_factor * 0.01
            self._hazard_items[hazard_index].update_hazard(radius_pixels, risk_level)

    def fit_to_view(self):
        """Zoom để vừa toàn bộ mạng lưới trong viewport."""
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_level = 1.0
        self.zoom_changed.emit(self._zoom_level)

    def zoom_in(self):
        """Zoom in."""
        self._zoom_level *= MapStyle.ZOOM_STEP
        if self._zoom_level > MapStyle.ZOOM_MAX:
            self._zoom_level = MapStyle.ZOOM_MAX
        else:
            self.scale(MapStyle.ZOOM_STEP, MapStyle.ZOOM_STEP)
            self.zoom_changed.emit(self._zoom_level)

    def zoom_out(self):
        """Zoom out."""
        self._zoom_level /= MapStyle.ZOOM_STEP
        if self._zoom_level < MapStyle.ZOOM_MIN:
            self._zoom_level = MapStyle.ZOOM_MIN
        else:
            self.scale(1 / MapStyle.ZOOM_STEP, 1 / MapStyle.ZOOM_STEP)
            self.zoom_changed.emit(self._zoom_level)

    def wheelEvent(self, event):
        """Xử lý scroll để zoom."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in()
        elif delta < 0:
            self.zoom_out()

    def resizeEvent(self, event):
        """Xử lý resize viewport."""
        super().resizeEvent(event)
        if self._network:
            self._calculate_transform()
            self.fit_to_view()


class MapWidget(QWidget):
    """Widget container cho bản đồ với controls."""

    # Signals
    node_clicked = pyqtSignal(str, str)
    zoom_changed = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Canvas
        self.canvas = MapCanvas(self)
        self.canvas.node_clicked.connect(self.node_clicked)
        self.canvas.zoom_changed.connect(self.zoom_changed)

        layout.addWidget(self.canvas)

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
        self.canvas.add_evacuee_group(route_id, count, path_node_ids)

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
