# SafeRoute HCM - Typhoon Evacuation Optimizer

## Project Overview
AI-powered evacuation route optimization for Ho Chi Minh City using hybrid GBFS + Grey Wolf Optimizer (GWO) with real-time visualization.

---

## Architecture

```
saferoute_hcm/
├── main.py                     # Application entry point
├── requirements.txt            # Dependencies
├── assets/
│   ├── icons/                  # UI icons
│   ├── styles/                 # QSS stylesheets
│   └── data/                   # Static data files
│       ├── hcm_districts.json  # District boundaries
│       ├── shelters.json       # Shelter locations & capacities
│       └── road_network.json   # Simplified road graph
├── src/
│   ├── __init__.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract algorithm interface
│   │   ├── gbfs.py             # Greedy Best First Search
│   │   ├── gwo.py              # Grey Wolf Optimizer
│   │   ├── hybrid.py           # GBFS + GWO hybrid approach
│   │   └── comparator.py       # Algorithm performance comparison
│   ├── models/
│   │   ├── __init__.py
│   │   ├── network.py          # Road network graph model
│   │   ├── node.py             # Node (intersection/shelter/zone)
│   │   ├── edge.py             # Edge (road segment)
│   │   ├── evacuation.py       # Evacuation scenario model
│   │   └── typhoon.py          # Typhoon hazard model
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── engine.py           # Simulation engine
│   │   ├── traffic.py          # Traffic flow simulation
│   │   └── events.py           # Dynamic events (flooding, road blocks)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── osm_loader.py       # OpenStreetMap data loader
│   │   ├── generator.py        # Synthetic data generator
│   │   └── preprocessor.py     # Data preprocessing utilities
│   └── ui/
│       ├── __init__.py
│       ├── main_window.py      # Main application window
│       ├── map_widget.py       # Interactive map visualization
│       ├── control_panel.py    # Algorithm controls & parameters
│       ├── dashboard.py        # Real-time metrics dashboard
│       ├── comparison_view.py  # Algorithm comparison charts
│       ├── scenario_editor.py  # Scenario configuration
│       └── styles.py           # Modern styling constants
└── tests/
    ├── test_algorithms.py
    ├── test_simulation.py
    └── test_models.py
```

---

## Core Algorithms

### 1. Greedy Best First Search (GBFS)
**Purpose:** Find optimal individual evacuation paths from population zones to shelters.

```python
# Heuristic function for GBFS
h(node) = w1 * distance_to_shelter +
          w2 * flood_risk_level +
          w3 * road_congestion +
          w4 * shelter_capacity_remaining
```

**Key Features:**
- Multi-objective heuristic (safety + speed + capacity)
- Dynamic hazard zone avoidance
- Real-time congestion awareness

### 2. Grey Wolf Optimizer (GWO)
**Purpose:** Global optimization of flow distribution across the network.

**Optimization Targets:**
- Minimize total evacuation time
- Balance shelter utilization (prevent overcrowding)
- Maximize road capacity usage efficiency
- Minimize exposure to hazard zones

**GWO Representation:**
- Wolf position = Flow assignment vector (people per route)
- Fitness = Multi-objective cost function
- Alpha, Beta, Delta wolves = Best 3 solutions

### 3. Hybrid GBFS + GWO
**Two-Phase Approach:**
1. **Phase 1 (GWO):** Optimize global flow distribution (which zones → which shelters, how many people)
2. **Phase 2 (GBFS):** Find actual paths for each flow assignment

**Iterative Refinement:**
- GWO proposes flow distribution
- GBFS calculates actual paths and costs
- Feedback loop updates GWO fitness

---

## Detailed Algorithm Implementations

### GBFS Implementation
```python
class GreedyBestFirstSearch:
    """Multi-objective GBFS for evacuation pathfinding"""

    def __init__(self, graph, weights):
        self.graph = graph
        self.w_dist = weights.get('distance', 0.4)
        self.w_risk = weights.get('risk', 0.3)
        self.w_congestion = weights.get('congestion', 0.2)
        self.w_capacity = weights.get('capacity', 0.1)

    def heuristic(self, node, goal, current_state):
        """Multi-objective heuristic combining safety and efficiency"""
        # Distance component (Haversine)
        h_dist = haversine_distance(node.pos, goal.pos)

        # Risk component (flood zone proximity)
        h_risk = self.get_flood_risk(node)

        # Congestion component (current flow / capacity)
        h_congestion = current_state.get_congestion(node)

        # Shelter capacity remaining
        h_capacity = 1.0 - (goal.current_occupancy / goal.capacity)

        return (self.w_dist * h_dist +
                self.w_risk * h_risk +
                self.w_congestion * h_congestion -
                self.w_capacity * h_capacity)

    def find_path(self, source, goals, state):
        """Find best path from source to any available shelter"""
        open_set = PriorityQueue()
        open_set.put((0, source, [source]))
        visited = set()

        while not open_set.empty():
            _, current, path = open_set.get()

            if current in goals and goals[current].has_capacity():
                return path, goals[current]

            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    edge_cost = self.graph.get_edge_cost(current, neighbor, state)
                    h = min(self.heuristic(neighbor, g, state) for g in goals.values())
                    priority = edge_cost + h
                    open_set.put((priority, neighbor, path + [neighbor]))

        return None, None  # No path found
```

### GWO Implementation
```python
class GreyWolfOptimizer:
    """Grey Wolf Optimizer for flow distribution"""

    def __init__(self, n_wolves=30, max_iter=100):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.a_decay = 2.0  # Linearly decreases from 2 to 0

    def initialize_population(self, n_zones, n_shelters):
        """Each wolf = flow distribution matrix"""
        population = []
        for _ in range(self.n_wolves):
            # Random flow assignment (zone i -> shelter j)
            wolf = np.random.rand(n_zones, n_shelters)
            wolf = wolf / wolf.sum(axis=1, keepdims=True)  # Normalize rows
            population.append(wolf)
        return population

    def fitness(self, wolf, graph, zones, shelters):
        """Evaluate solution quality"""
        total_time = 0
        total_risk = 0
        capacity_violation = 0
        flow_balance = 0

        for i, zone in enumerate(zones):
            for j, shelter in enumerate(shelters):
                flow = wolf[i, j] * zone.population
                if flow > 0:
                    # Use GBFS to find actual path cost
                    path_cost = self.calculate_path_cost(zone, shelter, graph)
                    total_time += flow * path_cost
                    total_risk += flow * self.get_route_risk(zone, shelter)

        # Penalty for exceeding shelter capacity
        shelter_loads = wolf.sum(axis=0) * np.array([z.population for z in zones]).sum()
        for j, shelter in enumerate(shelters):
            if shelter_loads[j] > shelter.capacity:
                capacity_violation += (shelter_loads[j] - shelter.capacity) ** 2

        # Penalty for unbalanced distribution
        flow_balance = np.std(shelter_loads / np.array([s.capacity for s in shelters]))

        return total_time + 100 * total_risk + 1000 * capacity_violation + 50 * flow_balance

    def optimize(self, graph, zones, shelters, callback=None):
        """Main GWO loop"""
        population = self.initialize_population(len(zones), len(shelters))
        fitness_scores = [self.fitness(w, graph, zones, shelters) for w in population]

        # Identify alpha, beta, delta (best 3 wolves)
        sorted_idx = np.argsort(fitness_scores)
        alpha, beta, delta = [population[i] for i in sorted_idx[:3]]
        alpha_score = fitness_scores[sorted_idx[0]]

        convergence = []

        for iteration in range(self.max_iter):
            a = 2.0 - iteration * (2.0 / self.max_iter)  # Linear decay

            for i in range(self.n_wolves):
                for j in range(population[i].shape[0]):
                    for k in range(population[i].shape[1]):
                        # Update position based on alpha, beta, delta
                        r1, r2 = np.random.rand(2)
                        A1 = 2 * a * r1 - a
                        C1 = 2 * r2
                        D_alpha = abs(C1 * alpha[j, k] - population[i][j, k])
                        X1 = alpha[j, k] - A1 * D_alpha

                        r1, r2 = np.random.rand(2)
                        A2 = 2 * a * r1 - a
                        C2 = 2 * r2
                        D_beta = abs(C2 * beta[j, k] - population[i][j, k])
                        X2 = beta[j, k] - A2 * D_beta

                        r1, r2 = np.random.rand(2)
                        A3 = 2 * a * r1 - a
                        C3 = 2 * r2
                        D_delta = abs(C3 * delta[j, k] - population[i][j, k])
                        X3 = delta[j, k] - A3 * D_delta

                        population[i][j, k] = (X1 + X2 + X3) / 3

                # Normalize and clip
                population[i] = np.clip(population[i], 0, 1)
                population[i] = population[i] / population[i].sum(axis=1, keepdims=True)

            # Update alpha, beta, delta
            fitness_scores = [self.fitness(w, graph, zones, shelters) for w in population]
            sorted_idx = np.argsort(fitness_scores)
            alpha, beta, delta = [population[i].copy() for i in sorted_idx[:3]]
            alpha_score = fitness_scores[sorted_idx[0]]

            convergence.append(alpha_score)
            if callback:
                callback(iteration, alpha_score, alpha)

        return alpha, convergence
```

### Hybrid Algorithm
```python
class HybridGBFSGWO:
    """Combines GWO global optimization with GBFS pathfinding"""

    def __init__(self, graph, zones, shelters):
        self.graph = graph
        self.zones = zones
        self.shelters = shelters
        self.gwo = GreyWolfOptimizer(n_wolves=30, max_iter=100)
        self.gbfs = GreedyBestFirstSearch(graph, DEFAULT_WEIGHTS)

    def optimize(self, callback=None):
        # Phase 1: GWO finds optimal flow distribution
        flow_matrix, convergence = self.gwo.optimize(
            self.graph, self.zones, self.shelters,
            callback=lambda i, s, f: callback('gwo', i, s) if callback else None
        )

        # Phase 2: GBFS finds actual paths for each flow
        evacuation_plan = []
        for i, zone in enumerate(self.zones):
            for j, shelter in enumerate(self.shelters):
                flow = flow_matrix[i, j] * zone.population
                if flow > 100:  # Minimum threshold
                    path, _ = self.gbfs.find_path(zone.node, {shelter.id: shelter}, {})
                    if path:
                        evacuation_plan.append({
                            'zone': zone,
                            'shelter': shelter,
                            'path': path,
                            'flow': int(flow)
                        })

        return evacuation_plan, flow_matrix, convergence
```

---

## Network Flow Model

### Problem Formulation
```
Minimize: Σ(evacuation_time[i] + risk_exposure[i]) for all citizens
Subject to:
  - shelter_occupancy[j] ≤ shelter_capacity[j] for all shelters
  - road_flow[e] ≤ road_capacity[e] for all edges
  - all_citizens_evacuated = true
```

### Graph Structure
- **Nodes:**
  - Population zones (sources with demand)
  - Road intersections (intermediate)
  - Shelters (sinks with capacity)
  - Hazard zones (high-cost/blocked nodes)

- **Edges:**
  - Road segments with:
    - Base travel time
    - Capacity (vehicles/hour)
    - Flood risk multiplier
    - Current congestion level

---

## UI Design (PyQt6 Custom Canvas) - PRIORITY: Real-time Animation + Algorithm Comparison

### Technology Choice: Pure QGraphicsView (No Web Dependencies)
- Hardware-accelerated OpenGL rendering via QOpenGLWidget
- Full control over animations and visual effects
- No browser dependencies = faster startup, lighter memory
- Smooth 60 FPS animations with thousands of particles

### Main Window Layout
```
+-------------------------------------------------------------------------+
|  SafeRoute HCM - Typhoon Evacuation Optimizer                  [_][O][X]|
+-------------------------------------------------------------------------+
|  [Map View]   [Algorithm Comparison]   [Settings]                       |
+-------------------------------------------------------------------------+
| +-----------------------------------------------+ +-------------------+ |
| |                                               | | CONTROL PANEL     | |
| |          CUSTOM CANVAS MAP                    | |                   | |
| |       (QGraphicsView + OpenGL)                | | Algorithm         | |
| |                                               | | [GBFS + GWO   v]  | |
| |    * Real HCM road network from OSM           | |                   | |
| |    * Animated evacuation particles            | | Typhoon Category  | |
| |    * Pulsing hazard zones with glow           | | [1] [2] [3] [4] [5]|
| |    * Dynamic route thickness by flow          | |                   | |
| |    * Smooth pan & zoom                        | | Population        | |
| |                                               | | ====O--------50K  | |
| |                                               | |                   | |
| |    Legend:                                    | | [>>> RUN <<<]     | |
| |    o Population Zones (cyan)                  | | [Pause]  [Reset]  | |
| |    # Shelters (green)                         | +-------------------+ |
| |    ~ Hazard Zones (red pulse)                 | | LIVE METRICS      | |
| |    - Routes (gradient by flow)                | |                   | |
| |                                               | | Evacuated  24,912 | |
| |                                               | | ============= 78% | |
| |                                               | |                   | |
| |                                               | | Est. Time  1h 47m | |
| |                                               | | Routes   47 active| |
| |                                               | | Shelters 8/12 open| |
| +-----------------------------------------------+ +-------------------+ |
+-------------------------------------------------------------------------+
|  [*] Running | 60 FPS | Hybrid GBFS+GWO | Iteration: 1,247              |
+-------------------------------------------------------------------------+
```

### Algorithm Comparison Tab (PRIORITY FEATURE)
```
+-------------------------------------------------------------------------+
|  ALGORITHM COMPARISON - Real-time Performance Analysis                  |
+-------------------------------------------------------------------------+
| +-----------------------------------+ +-------------------------------+ |
| |    CONVERGENCE GRAPH              | |    PERFORMANCE TABLE          | |
| |    (pyqtgraph live plot)          | |                               | |
| |                                   | |  +------+------+------+-----+ | |
| |    Cost                           | |  |Metric| GBFS | GWO  |Hybrd| | |
| |      ^                            | |  +------+------+------+-----+ | |
| |      |  \                         | |  |Time  | 0.2s | 1.4s |1.1s | | |
| |      |   \_                       | |  |Cost  | 847  | 623  | 589 | | |
| |      |     \__                    | |  |Routes| 42   | 38   | 41  | | |
| |      |        \___                | |  |Safety| 0.72 | 0.89 |0.94 | | |
| |      +--------------> Iteration   | |  +------+------+------+-----+ | |
| |                                   | |                               | |
| |    -- GBFS  -- GWO  -- Hybrid     | |  Winner: Hybrid (+15% better) | |
| +-----------------------------------+ +-------------------------------+ |
| +-----------------------------------+ +-------------------------------+ |
| |    RADAR CHART                    | |    MINI-MAP COMPARISON        | |
| |    (Multi-objective scores)       | |                               | |
| |                                   | |   GBFS      GWO      Hybrid   | |
| |           Speed                   | |  +----+   +----+    +----+    | |
| |             *                     | |  |    |   |    |    |    |    | |
| |         /   |   \                 | |  +----+   +----+    +----+    | |
| |    Safety --+-- Capacity          | |                               | |
| |         \   |   /                 | |  (click to see full solution) | |
| |             *                     | |                               | |
| |          Balance                  | |                               | |
| +-----------------------------------+ +-------------------------------+ |
+-------------------------------------------------------------------------+
```

### Color Scheme (Modern Dark Theme - GitHub Dark Inspired)
```python
COLORS = {
    'background': '#0d1117',      # Deep dark background
    'surface': '#161b22',         # Elevated panels
    'surface_light': '#21262d',   # Cards, controls
    'border': '#30363d',          # Subtle borders
    'primary': '#58a6ff',         # Primary actions (blue)
    'success': '#3fb950',         # Safe routes, shelters
    'warning': '#d29922',         # Caution areas
    'danger': '#f85149',          # Hazard zones
    'text': '#c9d1d9',            # Primary text
    'text_muted': '#8b949e',      # Secondary text
    'cyan': '#39c5cf',            # Evacuation particles
    'purple': '#a371f7',          # GWO visualization
    'gradient_safe': ['#238636', '#3fb950'],    # Route gradient (safe)
    'gradient_danger': ['#f85149', '#da3633'],  # Route gradient (hazard)
}
```

### Custom Canvas Components

#### 1. MapCanvas (QGraphicsView)
```python
class MapCanvas(QGraphicsView):
    """High-performance canvas with OpenGL acceleration"""
    - setViewport(QOpenGLWidget()) for GPU rendering
    - Antialiasing, smooth scaling
    - Mouse wheel zoom (anchor under mouse)
    - Pan with middle mouse / drag
    - Coordinate system: Mercator projection for real lat/lon
```

#### 2. Animation System (60 FPS)
```python
class AnimationController:
    """Frame-based animation engine"""
    - QTimer at 16ms interval (~60 FPS)
    - Particle pool (reuse objects, avoid GC)
    - Batch updates for performance
    - Interpolated movement along bezier paths
    - Phase-based hazard zone pulsing (sin wave opacity)
```

#### 3. Visual Elements
- **RoadEdgeItem**: Bezier paths with gradient stroke, thickness = flow
- **EvacuationParticle**: Small circles moving along paths, trail effect
- **HazardZoneItem**: Radial gradient, pulsing opacity, glow effect
- **ShelterItem**: Rounded rect with capacity bar, icon indicator
- **PopulationZoneItem**: Cluster of dots, size = population

### Key UI Features

1. **Real-time Animation (PRIORITY)**
   - Thousands of particles flowing along routes simultaneously
   - Smooth bezier curve interpolation
   - Hazard zones pulse red with glow effect
   - Route thickness dynamically scales with flow volume
   - Particle color gradient: cyan (moving) -> green (safe) -> red (danger)

2. **Algorithm Comparison (PRIORITY)**
   - Live convergence graph with pyqtgraph (updates every iteration)
   - Side-by-side performance metrics table
   - Radar chart for multi-objective visualization
   - Mini-map thumbnails showing each algorithm's solution
   - Automatic winner highlighting

3. **Interactive Controls**
   - Click to inspect node/shelter details
   - Hover tooltips for all elements
   - Time slider for simulation playback/scrubbing
   - Algorithm parameter sliders (real-time update)

---

## Real HCM Data Acquisition (CRITICAL)

### OpenStreetMap Data via OSMnx
```python
import osmnx as ox

# Ho Chi Minh City bounding box (approximate)
HCM_BOUNDS = {
    'north': 10.9000,
    'south': 10.6500,
    'east': 106.8500,
    'west': 106.5500
}

# Download drivable road network
G = ox.graph_from_bbox(
    HCM_BOUNDS['north'], HCM_BOUNDS['south'],
    HCM_BOUNDS['east'], HCM_BOUNDS['west'],
    network_type='drive',
    simplify=True
)

# Convert to NetworkX graph with:
# - Node: (lat, lon, osmid)
# - Edge: (length, maxspeed, lanes, name)
```

### Data Processing Pipeline
1. **Download** OSM data for HCM (cached locally)
2. **Simplify** graph (merge intermediate nodes)
3. **Extract** edge attributes (speed limits, lanes, road type)
4. **Calculate** edge capacities based on road type
5. **Generate** shelter locations (schools, hospitals, stadiums)
6. **Create** population zones from district data

### Shelter Data (Generated from OSM POIs)
```python
# Query amenities that can serve as shelters
shelter_tags = {
    'amenity': ['school', 'hospital', 'community_centre', 'place_of_worship'],
    'leisure': ['stadium', 'sports_centre'],
    'building': ['public', 'government']
}

# Assign capacity based on type
CAPACITY_MAP = {
    'stadium': 20000,
    'hospital': 500,
    'school': 2000,
    'sports_centre': 5000,
    'community_centre': 1000,
}
```

### Population Zones (District-based)
```python
# HCM Districts with approximate population
HCM_DISTRICTS = {
    'District 1': {'pop': 180000, 'center': (10.7769, 106.7009)},
    'District 3': {'pop': 190000, 'center': (10.7838, 106.6834)},
    'District 4': {'pop': 180000, 'center': (10.7579, 106.7057)},
    'District 5': {'pop': 170000, 'center': (10.7554, 106.6631)},
    'District 7': {'pop': 310000, 'center': (10.7365, 106.7218)},
    'Binh Thanh': {'pop': 490000, 'center': (10.8105, 106.7091)},
    'Thu Duc': {'pop': 530000, 'center': (10.8514, 106.7539)},
    # ... more districts
}
```

### Hazard Zone Generation (Flood Risk)
```python
# Low-lying flood-prone areas in HCM
FLOOD_ZONES = [
    {'center': (10.7579, 106.7057), 'radius': 2.0, 'risk': 0.9},  # District 4
    {'center': (10.7365, 106.7218), 'radius': 1.5, 'risk': 0.7},  # District 7
    # Areas near Saigon River
    {'center': (10.7800, 106.7100), 'radius': 1.0, 'risk': 0.8},
]

# Risk multiplier for edge cost calculation
def get_flood_risk(lat, lon, typhoon_intensity):
    for zone in FLOOD_ZONES:
        dist = haversine(lat, lon, zone['center'])
        if dist < zone['radius']:
            return zone['risk'] * typhoon_intensity
    return 0.0
```

---

## Differentiating Features (Hackathon Winners)

### 1. Real-World Data Integration (YOUR ADVANTAGE)
- Actual OSM road network for HCM (~50,000+ edges)
- Real shelter locations from OSM POIs
- District population density data
- Flood-prone area mapping based on elevation/history

### 2. Dynamic Simulation
- Time-stepped evacuation simulation
- Progressive flooding as typhoon approaches
- Road capacity changes (accidents, flooding)
- Shelter filling in real-time

### 3. Multi-Scenario Analysis
- Pre-configured typhoon scenarios (Cat 1-5)
- Custom scenario builder
- "What-if" analysis (shelter destroyed, road blocked)
- Historical typhoon data replay

### 4. Smart Recommendations
- Optimal shelter assignment per district
- Suggested new shelter locations
- Bottleneck identification
- Evacuation priority ranking

### 5. Export & Integration
- Export routes to Google Maps format
- Generate evacuation reports (PDF)
- API endpoint for mobile app integration
- SMS alert message generator

---

## Implementation Phases

### Phase 1: Core Foundation (Day 1 Morning)
- [ ] Project structure setup
- [ ] Network graph model implementation
- [ ] Basic GBFS algorithm
- [ ] Basic GWO algorithm
- [ ] Unit tests for algorithms

### Phase 2: Data & Simulation (Day 1 Afternoon)
- [ ] HCM road network data (simplified/generated)
- [ ] Shelter and population zone data
- [ ] Simulation engine
- [ ] Traffic flow model

### Phase 3: Hybrid Algorithm (Day 1 Evening)
- [ ] GBFS + GWO integration
- [ ] Multi-objective optimization
- [ ] Algorithm performance comparator
- [ ] Parameter tuning

### Phase 4: UI Development (Day 2 Morning)
- [ ] PyQt6 main window
- [ ] Map visualization widget
- [ ] Control panel
- [ ] Real-time dashboard

### Phase 5: Visualization & Polish (Day 2 Afternoon)
- [ ] Animated evacuation visualization
- [ ] Algorithm comparison charts
- [ ] Scenario editor
- [ ] Modern styling & polish

### Phase 6: Documentation & Demo (Day 2 Evening)
- [ ] Demo scenarios
- [ ] Performance benchmarks
- [ ] Final testing
- [ ] Presentation preparation

---

## Technical Stack

### Core Dependencies
```
PyQt6>=6.6.0              # Modern UI framework
PyQt6-3D>=6.6.0           # OpenGL widget support
networkx>=3.2             # Graph algorithms
numpy>=1.26.0             # Numerical computing
pyqtgraph>=0.13.3         # Real-time plotting (algorithm comparison)
pandas>=2.1.0             # Data handling
osmnx>=1.7.0              # OpenStreetMap data (REQUIRED for real HCM data)
scipy>=1.11.0             # Scientific computing (optimization)
```

### Data Processing
```
geopandas>=0.14.0         # Geospatial data handling
shapely>=2.0.0            # Geometric operations
pyproj>=3.6.0             # Coordinate transformations
```

### Optional (Enhancement)
```
numba>=0.58.0             # JIT compilation for algorithm speedup
matplotlib>=3.8.0         # Additional charts if needed
```

---

## Algorithm Comparison Metrics

| Metric | GBFS | GWO | Hybrid |
|--------|------|-----|--------|
| Execution Time | Fast | Medium | Medium |
| Solution Quality | Local | Global | Best |
| Scalability | High | Medium | High |
| Dynamic Adaptation | Good | Limited | Excellent |

---

## Risk Mitigation

1. **Data Complexity:** Use generated/simplified HCM network if OSM processing takes too long
2. **Performance:** Implement progressive loading and LOD for large networks
3. **Time Constraints:** Core algorithms first, UI polish later
4. **PyQt6 Issues:** Have fallback to matplotlib-based visualization

---

## Demo Scenarios

### Scenario 1: Category 3 Typhoon (Standard)
- 50,000 evacuees from 5 districts
- 12 available shelters
- Moderate flooding in low-lying areas

### Scenario 2: Severe Flooding (Stress Test)
- 100,000 evacuees
- 3 shelters destroyed
- Major roads blocked
- Shows algorithm robustness

### Scenario 3: Real-Time Adaptation
- Start with normal conditions
- Dynamically add hazards during simulation
- Demonstrate re-routing capability

---

## Implementation Order (Recommended)

### Step 1: Core Infrastructure
```
src/models/network.py      # Graph data structure
src/models/node.py         # Node types (zone, shelter, intersection)
src/models/edge.py         # Edge with capacity, risk attributes
src/data/osm_loader.py     # OSMnx data acquisition
```

### Step 2: Algorithms
```
src/algorithms/base.py     # Abstract interface
src/algorithms/gbfs.py     # Greedy Best First Search
src/algorithms/gwo.py      # Grey Wolf Optimizer
src/algorithms/hybrid.py   # Combined approach
src/algorithms/comparator.py  # Performance comparison
```

### Step 3: UI Components
```
src/ui/styles.py           # Color scheme, QSS
src/ui/main_window.py      # Main application window
src/ui/map_widget.py       # QGraphicsView canvas
src/ui/control_panel.py    # Algorithm controls
src/ui/dashboard.py        # Live metrics
src/ui/comparison_view.py  # Algorithm comparison charts
```

### Step 4: Animation & Simulation
```
src/ui/animation.py        # 60 FPS animation controller
src/ui/graphics_items.py   # Custom QGraphicsItems
src/simulation/engine.py   # Evacuation simulation
```

### Step 5: Integration & Polish
```
main.py                    # Application entry point
assets/data/*.json         # Cached HCM data
tests/                     # Unit tests
```

---

## Key Success Factors for Hackathon

1. **Visual Impact**: Smooth animations catch judges' attention immediately
2. **Real Data**: Using actual HCM roads/shelters adds credibility
3. **Scientific Rigor**: Side-by-side algorithm comparison shows depth
4. **Practical Value**: Solving a real evacuation problem demonstrates impact
5. **Code Quality**: Clean architecture allows confident demo/Q&A

---

## Ready for Implementation

Plan finalized based on user preferences:
- Custom PyQt6 canvas (QGraphicsView + OpenGL)
- Real Ho Chi Minh City data via OSMnx
- Priority features: Real-time animation + Algorithm comparison
- Time constraint: None (full implementation)
