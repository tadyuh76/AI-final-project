"""
OpenStreetMap data loader for Ho Chi Minh City.
Downloads and processes real road network data using OSMnx.
Falls back to generated data if OSMnx is not available.
"""

import os
import json
import hashlib
from typing import Optional, Tuple, List, Dict
from pathlib import Path

from ..models.network import EvacuationNetwork
from ..models.node import Node, NodeType, PopulationZone, Shelter, HazardZone
from ..models.edge import Edge, RoadType
from .hcm_data import (
    HCM_DISTRICTS, HCM_SHELTERS, FLOOD_PRONE_AREAS, HCM_BOUNDS,
    DistrictData, ShelterTemplate
)

# Try to import OSMnx
try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False
    ox = None

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None


class OSMDataLoader:
    """Loads road network data from OpenStreetMap."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the loader.

        Args:
            cache_dir: Directory to cache downloaded data. Defaults to assets/data/
        """
        if cache_dir is None:
            # Default to project's assets/data directory
            self.cache_dir = Path(__file__).parent.parent.parent / 'assets' / 'data'
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_hcm_network(self,
                         use_cache: bool = True,
                         simplify: bool = True,
                         network_type: str = 'drive') -> EvacuationNetwork:
        """
        Load Ho Chi Minh City road network.

        Args:
            use_cache: Whether to use cached data if available
            simplify: Whether to simplify the graph (merge intermediate nodes)
            network_type: OSMnx network type ('drive', 'walk', 'all')

        Returns:
            EvacuationNetwork with HCM road data
        """
        cache_file = self.cache_dir / 'hcm_network.json'

        # Try to load from cache
        if use_cache and cache_file.exists():
            try:
                print("Loading cached HCM network...")
                return EvacuationNetwork.load_from_file(str(cache_file))
            except Exception as e:
                print(f"Failed to load cache: {e}")

        # Try to download from OSM
        if HAS_OSMNX:
            try:
                print("Downloading HCM road network from OpenStreetMap...")
                network = self._download_osm_network(simplify, network_type)

                # Save to cache
                if use_cache:
                    print("Saving to cache...")
                    network.save_to_file(str(cache_file))

                return network
            except Exception as e:
                print(f"Failed to download OSM data: {e}")
                print("Falling back to generated network...")

        # Fall back to generated network
        print("Generating synthetic HCM network...")
        network = self._generate_synthetic_network()

        # Save to cache
        if use_cache:
            network.save_to_file(str(cache_file))

        return network

    def _download_osm_network(self, simplify: bool, network_type: str) -> EvacuationNetwork:
        """Download road network from OpenStreetMap using OSMnx."""
        if not HAS_OSMNX:
            raise RuntimeError("OSMnx is not installed")

        # Configure OSMnx
        ox.settings.use_cache = True
        ox.settings.log_console = True

        # Download the graph
        print(f"  Downloading {network_type} network for HCM bounds...")
        G = ox.graph_from_bbox(
            bbox=(HCM_BOUNDS['north'], HCM_BOUNDS['south'],
                  HCM_BOUNDS['east'], HCM_BOUNDS['west']),
            network_type=network_type,
            simplify=simplify
        )

        print(f"  Downloaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Convert to EvacuationNetwork
        network = self._convert_osmnx_graph(G)

        # Add population zones
        self._add_population_zones(network)

        # Add shelters
        self._add_shelters(network)

        return network

    def _convert_osmnx_graph(self, G) -> EvacuationNetwork:
        """Convert OSMnx graph to EvacuationNetwork."""
        network = EvacuationNetwork()

        # Add nodes
        for node_id, data in G.nodes(data=True):
            node = Node(
                id=str(node_id),
                lat=data.get('y', 0),
                lon=data.get('x', 0),
                node_type=NodeType.INTERSECTION
            )
            network.add_node(node)

        # Add edges
        edge_count = 0
        for u, v, key, data in G.edges(keys=True, data=True):
            edge_id = f"e_{u}_{v}_{key}"
            edge = Edge.from_osm_data(edge_id, str(u), str(v), data)
            network.add_edge(edge)
            edge_count += 1

        print(f"  Converted: {len(network._nodes)} nodes, {edge_count} edges")
        return network

    def _generate_synthetic_network(self) -> EvacuationNetwork:
        """Generate a synthetic road network for HCM."""
        network = EvacuationNetwork()

        # Create a grid-like network covering HCM
        grid_size = 20  # 20x20 grid
        lat_step = (HCM_BOUNDS['north'] - HCM_BOUNDS['south']) / grid_size
        lon_step = (HCM_BOUNDS['east'] - HCM_BOUNDS['west']) / grid_size

        # Create grid nodes
        node_grid = {}
        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                lat = HCM_BOUNDS['south'] + i * lat_step
                lon = HCM_BOUNDS['west'] + j * lon_step

                node_id = f"n_{i}_{j}"
                node = Node(
                    id=node_id,
                    lat=lat,
                    lon=lon,
                    node_type=NodeType.INTERSECTION
                )
                network.add_node(node)
                node_grid[(i, j)] = node_id

        # Create grid edges (roads)
        edge_count = 0
        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                current = node_grid[(i, j)]

                # Connect to right neighbor
                if j < grid_size:
                    neighbor = node_grid[(i, j + 1)]
                    edge = self._create_grid_edge(
                        f"e_{edge_count}",
                        current, neighbor,
                        network.get_node(current),
                        network.get_node(neighbor),
                        is_major=(i % 5 == 0)
                    )
                    network.add_edge(edge)
                    edge_count += 1

                # Connect to top neighbor
                if i < grid_size:
                    neighbor = node_grid[(i + 1, j)]
                    edge = self._create_grid_edge(
                        f"e_{edge_count}",
                        current, neighbor,
                        network.get_node(current),
                        network.get_node(neighbor),
                        is_major=(j % 5 == 0)
                    )
                    network.add_edge(edge)
                    edge_count += 1

        # Add diagonal connections for major roads
        for i in range(0, grid_size, 5):
            for j in range(0, grid_size, 5):
                current = node_grid[(i, j)]
                if i + 5 <= grid_size and j + 5 <= grid_size:
                    neighbor = node_grid[(i + 5, j + 5)]
                    edge = self._create_grid_edge(
                        f"e_{edge_count}",
                        current, neighbor,
                        network.get_node(current),
                        network.get_node(neighbor),
                        is_major=True
                    )
                    network.add_edge(edge)
                    edge_count += 1

        print(f"  Generated: {len(network._nodes)} nodes, {edge_count} edges")

        # Add population zones
        self._add_population_zones(network)

        # Add shelters
        self._add_shelters(network)

        return network

    def _create_grid_edge(self, edge_id: str, source_id: str, target_id: str,
                          source: Node, target: Node, is_major: bool) -> Edge:
        """Create a grid edge with appropriate properties."""
        from ..models.node import haversine_distance

        length = haversine_distance(source.lat, source.lon, target.lat, target.lon)

        if is_major:
            road_type = RoadType.PRIMARY
            lanes = 3
            speed = 50
        else:
            road_type = RoadType.RESIDENTIAL
            lanes = 1
            speed = 30

        return Edge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            length_km=length,
            road_type=road_type,
            lanes=lanes,
            max_speed_kmh=speed,
            is_oneway=False,
            geometry=[(source.lat, source.lon), (target.lat, target.lon)]
        )

    def _add_population_zones(self, network: EvacuationNetwork) -> None:
        """Add population zones from HCM district data."""
        for district_id, district in HCM_DISTRICTS.items():
            # Find nearest network node to district center
            nearest = network.find_nearest_node(district.center_lat, district.center_lon)

            if nearest:
                # Create population zone at district center
                zone = PopulationZone(
                    id=f"zone_{district_id}",
                    lat=district.center_lat,
                    lon=district.center_lon,
                    population=district.population,
                    district_name=district.name,
                    name=district.name
                )
                network.add_node(zone)

                # Connect to nearest intersection
                from ..models.node import haversine_distance
                length = haversine_distance(zone.lat, zone.lon, nearest.lat, nearest.lon)
                edge = Edge(
                    id=f"e_zone_{district_id}",
                    source_id=zone.id,
                    target_id=nearest.id,
                    length_km=length,
                    road_type=RoadType.RESIDENTIAL,
                    lanes=2,
                    max_speed_kmh=30,
                    is_oneway=False
                )
                network.add_edge(edge)

        print(f"  Added {len(HCM_DISTRICTS)} population zones")

    def _add_shelters(self, network: EvacuationNetwork) -> None:
        """Add shelters from HCM shelter data."""
        for i, shelter_data in enumerate(HCM_SHELTERS):
            # Find nearest network node
            nearest = network.find_nearest_node(shelter_data.lat, shelter_data.lon)

            if nearest:
                shelter = Shelter(
                    id=f"shelter_{i}",
                    lat=shelter_data.lat,
                    lon=shelter_data.lon,
                    name=shelter_data.name,
                    capacity=shelter_data.capacity,
                    shelter_type=shelter_data.shelter_type
                )
                network.add_node(shelter)

                # Connect to nearest intersection
                from ..models.node import haversine_distance
                length = haversine_distance(shelter.lat, shelter.lon, nearest.lat, nearest.lon)
                edge = Edge(
                    id=f"e_shelter_{i}",
                    source_id=nearest.id,
                    target_id=shelter.id,
                    length_km=length,
                    road_type=RoadType.SECONDARY,
                    lanes=2,
                    max_speed_kmh=40,
                    is_oneway=False
                )
                network.add_edge(edge)

        print(f"  Added {len(HCM_SHELTERS)} shelters")

    def add_default_hazards(self, network: EvacuationNetwork,
                           typhoon_intensity: float = 0.7) -> None:
        """
        Add default flood hazard zones based on historical data.

        Args:
            network: The network to add hazards to
            typhoon_intensity: Multiplier for risk levels (0.0 to 1.0)
        """
        for area in FLOOD_PRONE_AREAS:
            hazard = HazardZone(
                center_lat=area['center_lat'],
                center_lon=area['center_lon'],
                radius_km=area['radius_km'],
                risk_level=min(1.0, area['risk'] * typhoon_intensity),
                hazard_type='flood'
            )
            network.add_hazard_zone(hazard)

        print(f"  Added {len(FLOOD_PRONE_AREAS)} hazard zones")


def load_network(use_cache: bool = True,
                use_osm: bool = True) -> EvacuationNetwork:
    """
    Convenience function to load HCM network.

    Args:
        use_cache: Whether to use cached data
        use_osm: Whether to try downloading from OSM

    Returns:
        Loaded EvacuationNetwork
    """
    loader = OSMDataLoader()
    network = loader.load_hcm_network(use_cache=use_cache)
    return network


# Test the loader
if __name__ == '__main__':
    print("Testing OSM Data Loader")
    print("=" * 50)
    print(f"OSMnx available: {HAS_OSMNX}")
    print(f"NetworkX available: {HAS_NETWORKX}")
    print()

    loader = OSMDataLoader()
    network = loader.load_hcm_network(use_cache=True)
    loader.add_default_hazards(network, typhoon_intensity=0.7)

    print()
    print("Network Statistics:")
    stats = network.get_stats()
    print(f"  Nodes: {stats.total_nodes}")
    print(f"  Edges: {stats.total_edges}")
    print(f"  Population Zones: {stats.population_zones}")
    print(f"  Shelters: {stats.shelters}")
    print(f"  Total Population: {stats.total_population:,}")
    print(f"  Total Shelter Capacity: {stats.total_shelter_capacity:,}")
    print(f"  Hazard Zones: {len(network.get_hazard_zones())}")
