"""
Edge model for the evacuation network.
Represents road segments with capacity, travel time, and risk attributes.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum


class RoadType(Enum):
    """Types of roads with different capacities."""
    MOTORWAY = "motorway"           # Highways - highest capacity
    TRUNK = "trunk"                  # Major arterial roads
    PRIMARY = "primary"              # Primary roads
    SECONDARY = "secondary"          # Secondary roads
    TERTIARY = "tertiary"            # Tertiary roads
    RESIDENTIAL = "residential"      # Residential streets
    UNCLASSIFIED = "unclassified"   # Other roads


# Capacity in vehicles per hour per lane based on road type
ROAD_CAPACITY = {
    RoadType.MOTORWAY: 2000,
    RoadType.TRUNK: 1500,
    RoadType.PRIMARY: 1200,
    RoadType.SECONDARY: 800,
    RoadType.TERTIARY: 600,
    RoadType.RESIDENTIAL: 400,
    RoadType.UNCLASSIFIED: 300,
}

# Default speed limits (km/h) by road type
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
    """Represents a road segment in the network."""
    id: str
    source_id: str
    target_id: str
    length_km: float  # Length in kilometers
    road_type: RoadType = RoadType.UNCLASSIFIED
    lanes: int = 1
    max_speed_kmh: float = 30.0
    name: Optional[str] = None
    is_oneway: bool = False

    # Dynamic state (changes during simulation)
    current_flow: int = 0  # Current number of vehicles/people on edge
    flood_risk: float = 0.0  # 0.0 to 1.0, risk from flooding
    is_blocked: bool = False  # Road completely blocked

    # Geometry for visualization (list of lat/lon points)
    geometry: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def capacity(self) -> int:
        """Maximum flow capacity (vehicles/hour)."""
        base_capacity = ROAD_CAPACITY.get(self.road_type, 300)
        return base_capacity * self.lanes

    @property
    def base_travel_time(self) -> float:
        """Base travel time in hours without congestion."""
        if self.max_speed_kmh <= 0:
            return float('inf')
        return self.length_km / self.max_speed_kmh

    @property
    def congestion_level(self) -> float:
        """Current congestion from 0.0 (free) to 1.0 (jammed)."""
        if self.capacity <= 0:
            return 1.0
        return min(1.0, self.current_flow / self.capacity)

    @property
    def effective_speed(self) -> float:
        """Current effective speed considering congestion (km/h)."""
        if self.is_blocked:
            return 0.0

        # BPR (Bureau of Public Roads) function for speed reduction
        # Speed = FreeFlowSpeed / (1 + alpha * (flow/capacity)^beta)
        alpha = 0.15
        beta = 4.0
        congestion_factor = 1 + alpha * (self.congestion_level ** beta)
        return self.max_speed_kmh / congestion_factor

    @property
    def current_travel_time(self) -> float:
        """Current travel time in hours considering congestion."""
        if self.is_blocked:
            return float('inf')

        speed = self.effective_speed
        if speed <= 0:
            return float('inf')
        return self.length_km / speed

    def get_cost(self, risk_weight: float = 0.3) -> float:
        """
        Calculate edge cost for pathfinding.

        Args:
            risk_weight: Weight for flood risk (0.0 to 1.0)

        Returns:
            Combined cost considering time and risk.
        """
        if self.is_blocked:
            return float('inf')

        time_cost = self.current_travel_time
        risk_cost = self.flood_risk * self.length_km  # Risk weighted by distance

        return time_cost + risk_weight * risk_cost

    def can_accept_flow(self, amount: int = 1) -> bool:
        """Check if edge can accept additional flow."""
        if self.is_blocked:
            return False
        # Allow some overflow (up to 150% capacity) but with penalty
        return self.current_flow + amount <= self.capacity * 1.5

    def add_flow(self, amount: int) -> None:
        """Add flow to the edge."""
        self.current_flow += amount

    def remove_flow(self, amount: int) -> None:
        """Remove flow from the edge."""
        self.current_flow = max(0, self.current_flow - amount)

    def reset_flow(self) -> None:
        """Reset flow to zero."""
        self.current_flow = 0

    def set_flood_risk(self, risk: float) -> None:
        """Set flood risk level."""
        self.flood_risk = max(0.0, min(1.0, risk))
        # High flood risk blocks the road
        if risk > 0.9:
            self.is_blocked = True

    def block(self) -> None:
        """Block the road."""
        self.is_blocked = True

    def unblock(self) -> None:
        """Unblock the road."""
        self.is_blocked = False
        if self.flood_risk > 0.9:
            self.flood_risk = 0.5  # Reduce risk when unblocking

    @classmethod
    def from_osm_data(cls, edge_id: str, source: str, target: str,
                      data: dict) -> 'Edge':
        """Create an Edge from OSM edge data."""
        # Parse road type
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

        # Parse length (meters to km)
        length_m = data.get('length', 100)
        length_km = length_m / 1000.0

        # Parse lanes
        lanes = data.get('lanes', 1)
        if isinstance(lanes, list):
            lanes = int(lanes[0])
        elif isinstance(lanes, str):
            lanes = int(lanes)
        lanes = max(1, lanes)

        # Parse speed
        maxspeed = data.get('maxspeed', None)
        if maxspeed:
            if isinstance(maxspeed, list):
                maxspeed = maxspeed[0]
            if isinstance(maxspeed, str):
                # Remove 'km/h' or 'mph' and convert
                maxspeed = maxspeed.replace('km/h', '').replace('mph', '').strip()
                try:
                    maxspeed = float(maxspeed)
                except ValueError:
                    maxspeed = DEFAULT_SPEEDS.get(road_type, 30)
        else:
            maxspeed = DEFAULT_SPEEDS.get(road_type, 30)

        # Parse name
        name = data.get('name', None)
        if isinstance(name, list):
            name = name[0]

        # Parse oneway
        oneway = data.get('oneway', False)
        if isinstance(oneway, str):
            oneway = oneway.lower() in ('yes', 'true', '1')

        # Parse geometry if available
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
