"""
Node models for the evacuation network.
Defines different types of nodes: Population Zones, Shelters, Intersections, and Hazard Zones.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple
import math


class NodeType(Enum):
    """Types of nodes in the evacuation network."""
    POPULATION_ZONE = "population_zone"  # Source nodes with people to evacuate
    SHELTER = "shelter"                   # Sink nodes with capacity limits
    INTERSECTION = "intersection"         # Regular road intersections
    HAZARD = "hazard"                     # High-risk areas (flooding, etc.)


@dataclass
class Node:
    """Base node class representing a point in the network."""
    id: str
    lat: float
    lon: float
    node_type: NodeType = NodeType.INTERSECTION
    name: Optional[str] = None

    @property
    def pos(self) -> Tuple[float, float]:
        """Return position as (lat, lon) tuple."""
        return (self.lat, self.lon)

    def distance_to(self, other: 'Node') -> float:
        """Calculate haversine distance to another node in kilometers."""
        return haversine_distance(self.lat, self.lon, other.lat, other.lon)


@dataclass
class PopulationZone(Node):
    """A population zone that needs evacuation."""
    population: int = 0
    evacuated: int = 0
    district_name: str = ""

    def __post_init__(self):
        self.node_type = NodeType.POPULATION_ZONE

    @property
    def remaining_population(self) -> int:
        """People still needing evacuation."""
        return max(0, self.population - self.evacuated)

    @property
    def evacuation_progress(self) -> float:
        """Progress from 0.0 to 1.0."""
        if self.population == 0:
            return 1.0
        return self.evacuated / self.population


@dataclass
class Shelter(Node):
    """An evacuation shelter with limited capacity."""
    capacity: int = 1000
    current_occupancy: int = 0
    shelter_type: str = "general"  # school, hospital, stadium, etc.
    is_active: bool = True

    def __post_init__(self):
        self.node_type = NodeType.SHELTER

    @property
    def available_capacity(self) -> int:
        """Remaining capacity."""
        return max(0, self.capacity - self.current_occupancy)

    @property
    def occupancy_rate(self) -> float:
        """Occupancy from 0.0 to 1.0."""
        if self.capacity == 0:
            return 1.0
        return self.current_occupancy / self.capacity

    def has_capacity(self, amount: int = 1) -> bool:
        """Check if shelter can accept more evacuees."""
        return self.is_active and self.available_capacity >= amount

    def admit(self, count: int) -> int:
        """Admit evacuees, returns actual number admitted."""
        admitted = min(count, self.available_capacity)
        self.current_occupancy += admitted
        return admitted


@dataclass
class HazardZone:
    """Represents a hazardous area (flooding, storm damage, etc.)."""
    center_lat: float
    center_lon: float
    radius_km: float  # Radius of effect in kilometers
    risk_level: float = 0.8  # 0.0 to 1.0, higher = more dangerous
    hazard_type: str = "flood"  # flood, wind, debris, etc.
    is_active: bool = True

    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_lat, self.center_lon)

    def get_risk_at(self, lat: float, lon: float) -> float:
        """Calculate risk level at a given point (decreases with distance)."""
        if not self.is_active:
            return 0.0

        dist = haversine_distance(self.center_lat, self.center_lon, lat, lon)
        if dist >= self.radius_km:
            return 0.0

        # Linear falloff from center to edge
        distance_factor = 1.0 - (dist / self.radius_km)
        return self.risk_level * distance_factor


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points in kilometers.
    Uses the Haversine formula.
    """
    R = 6371.0  # Earth's radius in kilometers

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# Coordinate conversion utilities for visualization
def lat_lon_to_mercator(lat: float, lon: float) -> Tuple[float, float]:
    """Convert lat/lon to Web Mercator coordinates for visualization."""
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * 20037508.34 / 180.0
    return (x, y)


def mercator_to_lat_lon(x: float, y: float) -> Tuple[float, float]:
    """Convert Web Mercator coordinates back to lat/lon."""
    lon = x * 180.0 / 20037508.34
    lat = y * 180.0 / 20037508.34
    lat = 180.0 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2)
    return (lat, lon)
