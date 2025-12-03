"""
Các mô hình nút cho mạng lưới sơ tán.
Định nghĩa các loại nút khác nhau: Khu vực dân cư, Nơi trú ẩn, Giao lộ và Vùng nguy hiểm.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple
import math


class NodeType(Enum):
    """Các loại nút trong mạng lưới sơ tán."""
    POPULATION_ZONE = "population_zone"  # Nút nguồn có người cần sơ tán
    SHELTER = "shelter"                   # Nút đích có giới hạn sức chứa
    INTERSECTION = "intersection"         # Giao lộ đường thông thường
    HAZARD = "hazard"                     # Khu vực nguy hiểm cao (lũ lụt, v.v.)


@dataclass
class Node:
    """Lớp nút cơ bản đại diện cho một điểm trong mạng lưới."""
    id: str
    lat: float
    lon: float
    node_type: NodeType = NodeType.INTERSECTION
    name: Optional[str] = None

    @property
    def pos(self) -> Tuple[float, float]:
        """Trả về vị trí dưới dạng tuple (lat, lon)."""
        return (self.lat, self.lon)

    def distance_to(self, other: 'Node') -> float:
        """Tính khoảng cách haversine đến nút khác theo kilômét."""
        return haversine_distance(self.lat, self.lon, other.lat, other.lon)


@dataclass
class PopulationZone(Node):
    """Khu vực dân cư cần sơ tán."""
    population: int = 0
    evacuated: int = 0
    district_name: str = ""
    # Dân số cơ sở để điều chỉnh theo tỷ lệ phần trăm
    base_population: int = field(default=0, init=False)

    def __post_init__(self):
        self.node_type = NodeType.POPULATION_ZONE
        if self.base_population == 0:
            self.base_population = self.population

    @property
    def remaining_population(self) -> int:
        """Số người vẫn còn cần sơ tán."""
        return max(0, self.population - self.evacuated)

    @property
    def evacuation_progress(self) -> float:
        """Tiến độ từ 0.0 đến 1.0."""
        if self.population == 0:
            return 1.0
        return self.evacuated / self.population


@dataclass
class Shelter(Node):
    """Nơi trú ẩn sơ tán với sức chứa giới hạn."""
    capacity: int = 1000
    current_occupancy: int = 0
    shelter_type: str = "general"  # trường học, bệnh viện, sân vận động, v.v.
    is_active: bool = True

    def __post_init__(self):
        self.node_type = NodeType.SHELTER

    @property
    def available_capacity(self) -> int:
        """Sức chứa còn lại."""
        return max(0, self.capacity - self.current_occupancy)

    @property
    def occupancy_rate(self) -> float:
        """Tỷ lệ lấp đầy từ 0.0 đến 1.0."""
        if self.capacity == 0:
            return 1.0
        return self.current_occupancy / self.capacity

    def has_capacity(self, amount: int = 1) -> bool:
        """Kiểm tra xem nơi trú ẩn có thể tiếp nhận thêm người sơ tán hay không."""
        return self.is_active and self.available_capacity >= amount

    def admit(self, count: int) -> int:
        """Tiếp nhận người sơ tán, trả về số lượng thực tế được tiếp nhận."""
        admitted = min(count, self.available_capacity)
        self.current_occupancy += admitted
        return admitted


@dataclass
class HazardZone:
    """Đại diện cho khu vực nguy hiểm (lũ lụt, thiệt hại do bão, v.v.)."""
    center_lat: float
    center_lon: float
    radius_km: float  # Bán kính ảnh hưởng tính bằng kilômét
    risk_level: float = 0.8  # 0.0 đến 1.0, càng cao = càng nguy hiểm
    hazard_type: str = "flood"  # lũ lụt, gió, mảnh vỡ, v.v.
    is_active: bool = True
    # Giá trị cơ sở để điều chỉnh theo cường độ bão
    base_radius_km: float = field(default=0.0, init=False)
    base_risk_level: float = field(default=0.0, init=False)

    def __post_init__(self):
        """Lưu giá trị cơ sở sau khi khởi tạo."""
        if self.base_radius_km == 0.0:
            self.base_radius_km = self.radius_km
        if self.base_risk_level == 0.0:
            self.base_risk_level = self.risk_level

    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_lat, self.center_lon)

    def get_risk_at(self, lat: float, lon: float) -> float:
        """Tính mức độ rủi ro tại một điểm cho trước (giảm dần theo khoảng cách)."""
        if not self.is_active:
            return 0.0

        dist = haversine_distance(self.center_lat, self.center_lon, lat, lon)
        if dist >= self.radius_km:
            return 0.0

        # Giảm tuyến tính từ tâm đến rìa
        distance_factor = 1.0 - (dist / self.radius_km)
        return self.risk_level * distance_factor


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Tính khoảng cách đại quyến giữa hai điểm theo kilômét.
    Sử dụng công thức Haversine.
    """
    R = 6371.0  # Bán kính Trái Đất tính bằng kilômét

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# Tiện ích chuyển đổi tọa độ cho trực quan hóa
def lat_lon_to_mercator(lat: float, lon: float) -> Tuple[float, float]:
    """Chuyển đổi lat/lon sang tọa độ Web Mercator cho trực quan hóa."""
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * 20037508.34 / 180.0
    return (x, y)


def mercator_to_lat_lon(x: float, y: float) -> Tuple[float, float]:
    """Chuyển đổi tọa độ Web Mercator ngược về lat/lon."""
    lon = x * 180.0 / 20037508.34
    lat = y * 180.0 / 20037508.34
    lat = 180.0 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2)
    return (lat, lon)
