"""
Mô hình cạnh cho mạng lưới sơ tán.
Đại diện cho các đoạn đường với các thuộc tính sức chứa, thời gian di chuyển và rủi ro.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum


class RoadType(Enum):
    """Các loại đường với sức chứa khác nhau."""
    MOTORWAY = "motorway"           # Đường cao tốc - sức chứa cao nhất
    TRUNK = "trunk"                  # Đường trục chính
    PRIMARY = "primary"              # Đường cấp một
    SECONDARY = "secondary"          # Đường cấp hai
    TERTIARY = "tertiary"            # Đường cấp ba
    RESIDENTIAL = "residential"      # Đường dân cư
    UNCLASSIFIED = "unclassified"   # Các đường khác


# Sức chứa theo xe mỗi giờ mỗi làn dựa trên loại đường
ROAD_CAPACITY = {
    RoadType.MOTORWAY: 2000,
    RoadType.TRUNK: 1500,
    RoadType.PRIMARY: 1200,
    RoadType.SECONDARY: 800,
    RoadType.TERTIARY: 600,
    RoadType.RESIDENTIAL: 400,
    RoadType.UNCLASSIFIED: 300,
}

# Giới hạn tốc độ mặc định (km/h) theo loại đường
DEFAULT_SPEEDS = {
    RoadType.MOTORWAY: 80,
    RoadType.TRUNK: 60,
    RoadType.PRIMARY: 50,
    RoadType.SECONDARY: 40,
    RoadType.TERTIARY: 30,
    RoadType.RESIDENTIAL: 20,
    RoadType.UNCLASSIFIED: 20,
}


@dataclass
class Edge:
    """Đại diện cho một đoạn đường trong mạng lưới."""
    id: str
    source_id: str
    target_id: str
    length_km: float  # Chiều dài tính bằng kilômét
    road_type: RoadType = RoadType.UNCLASSIFIED
    lanes: int = 1
    max_speed_kmh: float = 30.0
    name: Optional[str] = None
    is_oneway: bool = False

    # Trạng thái động (thay đổi trong quá trình mô phỏng)
    current_flow: int = 0  # Số lượng phương tiện/người hiện tại trên cạnh
    flood_risk: float = 0.0  # 0.0 đến 1.0, rủi ro từ lũ lụt
    is_blocked: bool = False  # Đường bị chặn hoàn toàn

    # Hình học cho trực quan hóa (danh sách các điểm lat/lon)
    geometry: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def capacity(self) -> int:
        """Sức chứa luồng tối đa (xe/giờ)."""
        base_capacity = ROAD_CAPACITY.get(self.road_type, 300)
        return base_capacity * self.lanes

    @property
    def base_travel_time(self) -> float:
        """Thời gian di chuyển cơ bản tính bằng giờ không có tắc nghẽn."""
        if self.max_speed_kmh <= 0:
            return float('inf')
        return self.length_km / self.max_speed_kmh

    @property
    def congestion_level(self) -> float:
        """Mức độ tắc nghẽn hiện tại từ 0.0 (thông thoáng) đến 1.0 (tắc nghẽn)."""
        if self.capacity <= 0:
            return 1.0
        return min(1.0, self.current_flow / self.capacity)

    @property
    def effective_speed(self) -> float:
        """Tốc độ hiệu quả hiện tại tính đến tắc nghẽn (km/h)."""
        if self.is_blocked:
            return 0.0

        # Hàm BPR (Bureau of Public Roads) để giảm tốc độ
        # Tốc độ = TốcĐộTựDo / (1 + alpha * (luồng/sứcChứa)^beta)
        alpha = 0.15
        beta = 4.0
        congestion_factor = 1 + alpha * (self.congestion_level ** beta)
        return self.max_speed_kmh / congestion_factor

    @property
    def current_travel_time(self) -> float:
        """Thời gian di chuyển hiện tại tính bằng giờ có tính đến tắc nghẽn."""
        if self.is_blocked:
            return float('inf')

        speed = self.effective_speed
        if speed <= 0:
            return float('inf')
        return self.length_km / speed

    def get_cost(self, risk_weight: float = 0.3) -> float:
        """
        Tính chi phí cạnh cho tìm đường.

        Args:
            risk_weight: Trọng số cho rủi ro lũ lụt (0.0 đến 1.0)

        Returns:
            Chi phí kết hợp xem xét thời gian và rủi ro.
        """
        if self.is_blocked:
            return float('inf')

        time_cost = self.current_travel_time
        risk_cost = self.flood_risk * self.length_km  # Rủi ro được cân nhắc theo khoảng cách

        return time_cost + risk_weight * risk_cost

    def can_accept_flow(self, amount: int = 1) -> bool:
        """Kiểm tra xem cạnh có thể chấp nhận thêm luồng hay không."""
        if self.is_blocked:
            return False
        # Cho phép tràn một chút (lên đến 150% sức chứa) nhưng có phạt
        return self.current_flow + amount <= self.capacity * 1.5

    def add_flow(self, amount: int) -> None:
        """Thêm luồng vào cạnh."""
        self.current_flow += amount

    def remove_flow(self, amount: int) -> None:
        """Loại bỏ luồng khỏi cạnh."""
        self.current_flow = max(0, self.current_flow - amount)

    def reset_flow(self) -> None:
        """Đặt lại luồng về không."""
        self.current_flow = 0

    def set_flood_risk(self, risk: float) -> None:
        """Đặt mức độ rủi ro lũ lụt."""
        self.flood_risk = max(0.0, min(1.0, risk))
        # Rủi ro lũ lụt cao chặn đường
        if risk > 0.9:
            self.is_blocked = True

    def block(self) -> None:
        """Chặn đường."""
        self.is_blocked = True

    def unblock(self) -> None:
        """Mở chặn đường."""
        self.is_blocked = False
        if self.flood_risk > 0.9:
            self.flood_risk = 0.5  # Giảm rủi ro khi mở chặn

    @classmethod
    def from_osm_data(cls, edge_id: str, source: str, target: str,
                      data: dict) -> 'Edge':
        """Tạo một Edge từ dữ liệu cạnh OSM."""
        # Phân tích loại đường
        highway = data.get('highway', 'unclassified')
        if isinstance(highway, list):
            highway = highway[0]

        road_type_map = {
            'motorway': RoadType.MOTORWAY,
            'motorway_link': RoadType.MOTORWAY,
            'trunk': RoadType.TRUNK,
            'trunk_link': RoadType.TRUNK,
            'primary': RoadType.PRIMARY,
            'primary_link': RoadType.PRIMARY,
            'secondary': RoadType.SECONDARY,
            'secondary_link': RoadType.SECONDARY,
            'tertiary': RoadType.TERTIARY,
            'tertiary_link': RoadType.TERTIARY,
            'residential': RoadType.RESIDENTIAL,
        }
        road_type = road_type_map.get(highway, RoadType.UNCLASSIFIED)

        # Phân tích chiều dài (từ mét sang km)
        length_m = data.get('length', 100)
        length_km = length_m / 1000.0

        # Phân tích số làn
        lanes = data.get('lanes', 1)
        if isinstance(lanes, list):
            lanes = int(lanes[0])
        elif isinstance(lanes, str):
            lanes = int(lanes)
        lanes = max(1, lanes)

        # Phân tích tốc độ
        maxspeed = data.get('maxspeed', None)
        if maxspeed:
            if isinstance(maxspeed, list):
                maxspeed = maxspeed[0]
            if isinstance(maxspeed, str):
                # Loại bỏ 'km/h' hoặc 'mph' và chuyển đổi
                maxspeed = maxspeed.replace('km/h', '').replace('mph', '').strip()
                try:
                    maxspeed = float(maxspeed)
                except ValueError:
                    maxspeed = DEFAULT_SPEEDS.get(road_type, 30)
        else:
            maxspeed = DEFAULT_SPEEDS.get(road_type, 30)

        # Phân tích tên
        name = data.get('name', None)
        if isinstance(name, list):
            name = name[0]

        # Phân tích một chiều
        oneway = data.get('oneway', False)
        if isinstance(oneway, str):
            oneway = oneway.lower() in ('yes', 'true', '1')

        # Phân tích hình học nếu có
        geometry = []
        if 'geometry' in data:
            try:
                geom = data['geometry']
                if hasattr(geom, 'coords'):
                    geometry = [(lat, lon) for lon, lat in geom.coords]
            except Exception:
                pass

        return cls(
            id=edge_id,
            source_id=source,
            target_id=target,
            length_km=length_km,
            road_type=road_type,
            lanes=lanes,
            max_speed_kmh=maxspeed,
            name=name,
            is_oneway=oneway,
            geometry=geometry
        )
