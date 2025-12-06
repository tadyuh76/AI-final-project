#!/usr/bin/env python3
"""
Script thử nghiệm 6 kịch bản sơ tán cho báo cáo Chương 4.

Các kịch bản:
1. Ngập nhỏ (1-2 vùng) + 10% dân số
2. Ngập nhỏ (1-2 vùng) + 50% dân số
3. Ngập trung bình (3-4 vùng) + 10% dân số
4. Ngập trung bình (3-4 vùng) + 50% dân số
5. Ngập lớn (5-6 vùng) + 10% dân số
6. Ngập lớn (5-6 vùng) + 50% dân số
"""

import sys
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.osm_loader import OSMDataLoader
from src.models.network import EvacuationNetwork
from src.models.node import HazardZone, PopulationZone
from src.algorithms.gbfs import GreedyBestFirstSearch
from src.algorithms.gwo import GreyWolfOptimizer
from src.algorithms.base import AlgorithmConfig, EvacuationPlan, AlgorithmMetrics


# ==================== CẤU HÌNH KỊCH BẢN ====================

# Vùng nguy hiểm cho từng kịch bản
HAZARD_CONFIGS = {
    # Kịch bản 1-2: Ngập nhỏ (1-2 vùng) - Quận 8 và phần Q4
    'small': [
        {
            'center_lat': 10.7350,
            'center_lon': 106.6700,
            'radius_km': 3.0,  # Nhỏ hơn, chỉ Q8
            'risk': 0.85,
            'name': 'District 8 Flooding'
        },
        {
            'center_lat': 10.7579,
            'center_lon': 106.7057,
            'radius_km': 1.5,  # Quận 4
            'risk': 0.75,
            'name': 'District 4 Flooding'
        }
    ],

    # Kịch bản 3-4: Ngập trung bình (3-4 vùng)
    'medium': [
        {
            'center_lat': 10.7350,
            'center_lon': 106.6700,
            'radius_km': 4.0,
            'risk': 0.85,
            'name': 'Southern Lowlands'
        },
        {
            'center_lat': 10.6900,
            'center_lon': 106.7400,
            'radius_km': 2.5,
            'risk': 0.9,
            'name': 'Nha Be Coastal'
        },
        {
            'center_lat': 10.7450,
            'center_lon': 106.5950,
            'radius_km': 2.0,
            'risk': 0.7,
            'name': 'Binh Tan Industrial'
        },
        {
            'center_lat': 10.8100,
            'center_lon': 106.7200,
            'radius_km': 1.5,
            'risk': 0.65,
            'name': 'Binh Thanh Riverside'
        }
    ],

    # Kịch bản 5-6: Ngập lớn (5-6 vùng) - Tất cả vùng nguy hiểm
    'large': [
        {
            'center_lat': 10.7350,
            'center_lon': 106.6700,
            'radius_km': 4.5,
            'risk': 0.85,
            'name': 'Southern Lowlands'
        },
        {
            'center_lat': 10.6900,
            'center_lon': 106.7400,
            'radius_km': 2.5,
            'risk': 0.9,
            'name': 'Nha Be Coastal'
        },
        {
            'center_lat': 10.7450,
            'center_lon': 106.5950,
            'radius_km': 2.0,
            'risk': 0.7,
            'name': 'Binh Tan Industrial'
        },
        {
            'center_lat': 10.8550,
            'center_lon': 106.7850,
            'radius_km': 2.0,
            'risk': 0.6,
            'name': 'Thu Duc Riverside'
        },
        {
            'center_lat': 10.8100,
            'center_lon': 106.7200,
            'radius_km': 1.5,
            'risk': 0.65,
            'name': 'Binh Thanh Riverside'
        },
        {
            'center_lat': 10.8600,
            'center_lon': 106.6300,
            'radius_km': 1.8,
            'risk': 0.55,
            'name': 'District 12 Lowlands'
        }
    ]
}

SCENARIOS = [
    {'id': 1, 'name': 'Ngập nhỏ - 10% dân số', 'hazard_scale': 'small', 'population_ratio': 0.10},
    {'id': 2, 'name': 'Ngập nhỏ - 50% dân số', 'hazard_scale': 'small', 'population_ratio': 0.50},
    {'id': 3, 'name': 'Ngập trung bình - 10% dân số', 'hazard_scale': 'medium', 'population_ratio': 0.10},
    {'id': 4, 'name': 'Ngập trung bình - 50% dân số', 'hazard_scale': 'medium', 'population_ratio': 0.50},
    {'id': 5, 'name': 'Ngập lớn - 10% dân số', 'hazard_scale': 'large', 'population_ratio': 0.10},
    {'id': 6, 'name': 'Ngập lớn - 50% dân số', 'hazard_scale': 'large', 'population_ratio': 0.50},
]


@dataclass
class ScenarioResult:
    """Kết quả của một kịch bản thử nghiệm."""
    scenario_id: int
    scenario_name: str
    hazard_count: int
    population_ratio: float
    total_population: int
    evacuation_population: int
    total_shelter_capacity: int

    # GBFS results
    gbfs_time: float
    gbfs_routes: int
    gbfs_evacuees: int
    gbfs_coverage: float
    gbfs_cost: float
    gbfs_avg_risk: float
    gbfs_avg_distance: float
    gbfs_avg_time: float

    # GWO results
    gwo_time: float
    gwo_routes: int
    gwo_evacuees: int
    gwo_coverage: float
    gwo_cost: float
    gwo_avg_risk: float
    gwo_avg_distance: float
    gwo_avg_time: float

    # GWO convergence
    gwo_convergence: List[float] = None


def setup_network_for_scenario(scenario: Dict) -> EvacuationNetwork:
    """Thiết lập mạng lưới cho một kịch bản cụ thể."""
    print(f"\n  Đang tải mạng lưới...")
    loader = OSMDataLoader()
    network = loader.load_hcm_network(use_cache=True)

    # Xóa hazard zones cũ
    network.clear_hazard_zones()

    # Thêm hazard zones theo kịch bản
    hazard_config = HAZARD_CONFIGS[scenario['hazard_scale']]
    for hazard_data in hazard_config:
        hazard = HazardZone(
            center_lat=hazard_data['center_lat'],
            center_lon=hazard_data['center_lon'],
            radius_km=hazard_data['radius_km'],
            risk_level=hazard_data['risk'],
            hazard_type='flood'
        )
        network.add_hazard_zone(hazard)

    print(f"  Đã thêm {len(hazard_config)} vùng nguy hiểm")

    # Điều chỉnh dân số theo tỷ lệ
    population_ratio = scenario['population_ratio']
    zones = network.get_population_zones()
    total_original = sum(z.population for z in zones)

    for zone in zones:
        zone.population = int(zone.population * population_ratio)

    total_adjusted = sum(z.population for z in zones)
    print(f"  Dân số: {total_original:,} -> {total_adjusted:,} ({population_ratio*100:.0f}%)")

    return network


def calculate_plan_metrics(plan: EvacuationPlan) -> Dict[str, float]:
    """Tính các metrics từ plan."""
    if not plan.routes:
        return {
            'avg_risk': 0,
            'avg_distance': 0,
            'avg_time': 0,
            'total_cost': 0
        }

    total_flow = sum(r.flow for r in plan.routes)
    if total_flow == 0:
        return {'avg_risk': 0, 'avg_distance': 0, 'avg_time': 0, 'total_cost': 0}

    weighted_risk = sum(r.risk_score * r.flow for r in plan.routes) / total_flow
    weighted_distance = sum(r.distance_km * r.flow for r in plan.routes) / total_flow
    weighted_time = sum(r.estimated_time_hours * r.flow for r in plan.routes) / total_flow

    # Total cost formula: flow × (time + 0.3×risk + 0.001×distance)
    total_cost = sum(
        r.flow * (r.estimated_time_hours + 0.3 * r.risk_score + 0.001 * r.distance_km)
        for r in plan.routes
    )

    return {
        'avg_risk': weighted_risk,
        'avg_distance': weighted_distance,
        'avg_time': weighted_time,
        'total_cost': total_cost
    }


def calculate_load_balance(plan: EvacuationPlan, shelters: List) -> float:
    """Tính độ cân bằng tải (standard deviation of utilization)."""
    import numpy as np

    shelter_loads = {}
    for route in plan.routes:
        if route.shelter_id not in shelter_loads:
            shelter_loads[route.shelter_id] = 0
        shelter_loads[route.shelter_id] += route.flow

    utilizations = []
    for shelter in shelters:
        load = shelter_loads.get(shelter.id, 0)
        if shelter.capacity > 0:
            utilizations.append(load / shelter.capacity)

    if not utilizations:
        return 0.0

    return float(np.std(utilizations))


def run_scenario(scenario: Dict) -> ScenarioResult:
    """Chạy một kịch bản và trả về kết quả."""
    print(f"\n{'='*60}")
    print(f"KỊCH BẢN {scenario['id']}: {scenario['name']}")
    print(f"{'='*60}")

    # Setup network
    network = setup_network_for_scenario(scenario)

    # Get stats
    stats = network.get_stats()
    zones = network.get_population_zones()
    shelters = network.get_shelters()
    total_pop = sum(z.population for z in zones)
    total_capacity = sum(s.capacity for s in shelters)

    print(f"\n  Thống kê mạng lưới:")
    print(f"    - Nodes: {stats.total_nodes}")
    print(f"    - Edges: {stats.total_edges}")
    print(f"    - Zones: {stats.population_zones}")
    print(f"    - Shelters: {stats.shelters}")
    print(f"    - Tổng dân số: {total_pop:,}")
    print(f"    - Tổng sức chứa: {total_capacity:,}")
    print(f"    - Hazard zones: {len(network.get_hazard_zones())}")

    # Config
    config = AlgorithmConfig(
        n_wolves=30,
        max_iterations=100,
        distance_weight=0.4,
        risk_weight=0.3,
        congestion_weight=0.2,
        capacity_weight=0.1
    )

    # ==================== RUN GBFS ====================
    print(f"\n  Đang chạy GBFS...")
    network.reset_simulation_state()

    gbfs = GreedyBestFirstSearch(network, config)
    gbfs_start = time.time()
    gbfs_plan, gbfs_metrics = gbfs.optimize()
    gbfs_time = time.time() - gbfs_start

    gbfs_plan_metrics = calculate_plan_metrics(gbfs_plan)
    gbfs_balance = calculate_load_balance(gbfs_plan, shelters)

    print(f"    - Thời gian: {gbfs_time:.2f}s")
    print(f"    - Routes: {len(gbfs_plan.routes)}")
    print(f"    - Evacuees: {gbfs_plan.total_evacuees:,}")
    print(f"    - Coverage: {gbfs_metrics.coverage_rate*100:.1f}%")
    print(f"    - Cost: {gbfs_plan_metrics['total_cost']:.2f}")
    print(f"    - Avg Risk: {gbfs_plan_metrics['avg_risk']:.3f}")
    print(f"    - Balance (std): {gbfs_balance:.3f}")

    # ==================== RUN GWO ====================
    print(f"\n  Đang chạy GWO...")
    network.reset_simulation_state()

    gwo = GreyWolfOptimizer(network, config)
    gwo_start = time.time()
    gwo_plan, gwo_metrics = gwo.optimize()
    gwo_time = time.time() - gwo_start

    gwo_plan_metrics = calculate_plan_metrics(gwo_plan)
    gwo_balance = calculate_load_balance(gwo_plan, shelters)

    print(f"    - Thời gian: {gwo_time:.2f}s")
    print(f"    - Routes: {len(gwo_plan.routes)}")
    print(f"    - Evacuees: {gwo_plan.total_evacuees:,}")
    print(f"    - Coverage: {gwo_metrics.coverage_rate*100:.1f}%")
    print(f"    - Cost: {gwo_plan_metrics['total_cost']:.2f}")
    print(f"    - Avg Risk: {gwo_plan_metrics['avg_risk']:.3f}")
    print(f"    - Balance (std): {gwo_balance:.3f}")

    # Create result
    result = ScenarioResult(
        scenario_id=scenario['id'],
        scenario_name=scenario['name'],
        hazard_count=len(network.get_hazard_zones()),
        population_ratio=scenario['population_ratio'],
        total_population=total_pop,
        evacuation_population=total_pop,
        total_shelter_capacity=total_capacity,

        gbfs_time=gbfs_time,
        gbfs_routes=len(gbfs_plan.routes),
        gbfs_evacuees=gbfs_plan.total_evacuees,
        gbfs_coverage=gbfs_metrics.coverage_rate,
        gbfs_cost=gbfs_plan_metrics['total_cost'],
        gbfs_avg_risk=gbfs_plan_metrics['avg_risk'],
        gbfs_avg_distance=gbfs_plan_metrics['avg_distance'],
        gbfs_avg_time=gbfs_plan_metrics['avg_time'],

        gwo_time=gwo_time,
        gwo_routes=len(gwo_plan.routes),
        gwo_evacuees=gwo_plan.total_evacuees,
        gwo_coverage=gwo_metrics.coverage_rate,
        gwo_cost=gwo_plan_metrics['total_cost'],
        gwo_avg_risk=gwo_plan_metrics['avg_risk'],
        gwo_avg_distance=gwo_plan_metrics['avg_distance'],
        gwo_avg_time=gwo_plan_metrics['avg_time'],

        gwo_convergence=gwo_metrics.convergence_history[:20]  # First 20 iterations
    )

    return result


def format_results_markdown(results: List[ScenarioResult]) -> str:
    """Format kết quả thành markdown table."""
    lines = []

    lines.append("## BẢNG KẾT QUẢ THỰC NGHIỆM")
    lines.append("")
    lines.append("### Tổng quan 6 kịch bản")
    lines.append("")
    lines.append("| Kịch bản | Quy mô | Dân số | GBFS Time | GWO Time | GBFS Cost | GWO Cost | GBFS Coverage | GWO Coverage |")
    lines.append("|----------|--------|--------|-----------|----------|-----------|----------|---------------|--------------|")

    for r in results:
        scale = "Nhỏ" if r.hazard_count <= 2 else ("TB" if r.hazard_count <= 4 else "Lớn")
        lines.append(
            f"| {r.scenario_id} | {scale} ({r.hazard_count}) | {r.population_ratio*100:.0f}% | "
            f"{r.gbfs_time:.2f}s | {r.gwo_time:.2f}s | "
            f"{r.gbfs_cost:.0f} | {r.gwo_cost:.0f} | "
            f"{r.gbfs_coverage*100:.1f}% | {r.gwo_coverage*100:.1f}% |"
        )

    lines.append("")
    lines.append("### Chi tiết từng kịch bản")
    lines.append("")

    for r in results:
        lines.append(f"#### Kịch bản {r.scenario_id}: {r.scenario_name}")
        lines.append("")
        lines.append(f"- Số vùng nguy hiểm: {r.hazard_count}")
        lines.append(f"- Dân số cần sơ tán: {r.total_population:,} người")
        lines.append(f"- Tổng sức chứa: {r.total_shelter_capacity:,} người")
        lines.append("")
        lines.append("| Chỉ số | GBFS | GWO |")
        lines.append("|--------|------|-----|")
        lines.append(f"| Thời gian thực thi | {r.gbfs_time:.2f}s | {r.gwo_time:.2f}s |")
        lines.append(f"| Số tuyến đường | {r.gbfs_routes} | {r.gwo_routes} |")
        lines.append(f"| Số người sơ tán | {r.gbfs_evacuees:,} | {r.gwo_evacuees:,} |")
        lines.append(f"| Tỷ lệ bao phủ | {r.gbfs_coverage*100:.1f}% | {r.gwo_coverage*100:.1f}% |")
        lines.append(f"| Chi phí tổng | {r.gbfs_cost:.2f} | {r.gwo_cost:.2f} |")
        lines.append(f"| Rủi ro trung bình | {r.gbfs_avg_risk:.3f} | {r.gwo_avg_risk:.3f} |")
        lines.append(f"| Khoảng cách TB (km) | {r.gbfs_avg_distance:.2f} | {r.gwo_avg_distance:.2f} |")
        lines.append(f"| Thời gian sơ tán TB (h) | {r.gbfs_avg_time:.3f} | {r.gwo_avg_time:.3f} |")
        lines.append("")

        # Cost improvement
        if r.gbfs_cost > 0:
            improvement = (r.gbfs_cost - r.gwo_cost) / r.gbfs_cost * 100
            lines.append(f"**GWO cải thiện chi phí: {improvement:.1f}%**")
        lines.append("")

    return "\n".join(lines)


def main():
    """Chạy tất cả các kịch bản thử nghiệm."""
    print("="*70)
    print("SAFEROUTE HCM - THỬ NGHIỆM 6 KỊCH BẢN SƠ TÁN")
    print("="*70)

    results = []

    for scenario in SCENARIOS:
        try:
            result = run_scenario(scenario)
            results.append(result)
        except Exception as e:
            print(f"\n  LỖI kịch bản {scenario['id']}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("TÓM TẮT KẾT QUẢ")
    print("="*70)

    markdown_output = format_results_markdown(results)
    print(markdown_output)

    # Save results to file
    output_file = Path(__file__).parent / "report" / "scenario_results.md"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_output)

    print(f"\nKết quả đã được lưu vào: {output_file}")

    # Save raw data as JSON
    json_file = Path(__file__).parent / "report" / "scenario_results.json"
    json_data = []
    for r in results:
        json_data.append({
            'scenario_id': r.scenario_id,
            'scenario_name': r.scenario_name,
            'hazard_count': r.hazard_count,
            'population_ratio': r.population_ratio,
            'total_population': r.total_population,
            'total_shelter_capacity': r.total_shelter_capacity,
            'gbfs': {
                'time': r.gbfs_time,
                'routes': r.gbfs_routes,
                'evacuees': r.gbfs_evacuees,
                'coverage': r.gbfs_coverage,
                'cost': r.gbfs_cost,
                'avg_risk': r.gbfs_avg_risk,
                'avg_distance': r.gbfs_avg_distance,
                'avg_time': r.gbfs_avg_time
            },
            'gwo': {
                'time': r.gwo_time,
                'routes': r.gwo_routes,
                'evacuees': r.gwo_evacuees,
                'coverage': r.gwo_coverage,
                'cost': r.gwo_cost,
                'avg_risk': r.gwo_avg_risk,
                'avg_distance': r.gwo_avg_distance,
                'avg_time': r.gwo_avg_time,
                'convergence': r.gwo_convergence
            }
        })

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Dữ liệu JSON đã được lưu vào: {json_file}")

    return results


if __name__ == "__main__":
    main()
