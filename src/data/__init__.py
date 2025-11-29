"""
Data package for loading and processing HCM city data.
"""

from .hcm_data import (
    HCM_DISTRICTS, HCM_SHELTERS, FLOOD_PRONE_AREAS, HCM_BOUNDS,
    DistrictData, ShelterTemplate,
    get_total_population, get_total_shelter_capacity,
    get_districts_by_flood_risk, get_shelter_capacity_by_type
)
from .osm_loader import OSMDataLoader, load_network

__all__ = [
    'HCM_DISTRICTS', 'HCM_SHELTERS', 'FLOOD_PRONE_AREAS', 'HCM_BOUNDS',
    'DistrictData', 'ShelterTemplate',
    'get_total_population', 'get_total_shelter_capacity',
    'get_districts_by_flood_risk', 'get_shelter_capacity_by_type',
    'OSMDataLoader', 'load_network'
]
