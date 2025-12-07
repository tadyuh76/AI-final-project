"""
Cấu hình các kịch bản thử nghiệm cho báo cáo Chương 4.

Để sử dụng:
1. Import CURRENT_SCENARIO và SCENARIOS trong osm_loader.py
2. Thay đổi CURRENT_SCENARIO để chọn kịch bản (1-6)
3. Chạy lại ứng dụng

Kịch bản:
- 1: Ngập nhỏ (2 vùng) + 10% dân số
- 2: Ngập nhỏ (2 vùng) + 50% dân số
- 3: Ngập trung bình (4 vùng) + 10% dân số
- 4: Ngập trung bình (4 vùng) + 50% dân số
- 5: Ngập lớn (6 vùng) + 10% dân số
- 6: Ngập lớn (6 vùng) + 50% dân số
"""

from typing import List, Dict

# ============================================================
# THAY ĐỔI GIÁ TRỊ NÀY ĐỂ CHỌN KỊCH BẢN (1-6)
# ============================================================
CURRENT_SCENARIO = 1
# ============================================================


# Cấu hình vùng nguy hiểm cho từng quy mô ngập
HAZARD_CONFIGS = {
    # Ngập nhỏ: Quận 8 và Quận 4
    'small': [
        {
            'center_lat': 10.7350,
            'center_lon': 106.6400,
            'radius_km': 3.0,
            'risk': 0.85,
            'name': 'District 8 Flooding'
        },
        {
            'center_lat': 10.7579,
            'center_lon': 106.7057,
            'radius_km': 1.5,
            'risk': 0.75,
            'name': 'District 4 Flooding'
        }
    ],

    # Ngập trung bình: + Nhà Bè, Bình Tân, Bình Thạnh
    'medium': [
        {
            'center_lat': 10.7350,
            'center_lon': 106.6400,
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

    # Ngập lớn: Tất cả 6 vùng
    'large': [
        {
            'center_lat': 10.7350,
            'center_lon': 106.6400,
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


# Định nghĩa 6 kịch bản
SCENARIOS = {
    1: {'name': 'Ngập nhỏ - 10% dân số', 'hazard_scale': 'small', 'population_ratio': 0.10},
    2: {'name': 'Ngập nhỏ - 50% dân số', 'hazard_scale': 'small', 'population_ratio': 0.50},
    3: {'name': 'Ngập trung bình - 10% dân số', 'hazard_scale': 'medium', 'population_ratio': 0.10},
    4: {'name': 'Ngập trung bình - 50% dân số', 'hazard_scale': 'medium', 'population_ratio': 0.50},
    5: {'name': 'Ngập lớn - 10% dân số', 'hazard_scale': 'large', 'population_ratio': 0.10},
    6: {'name': 'Ngập lớn - 50% dân số', 'hazard_scale': 'large', 'population_ratio': 0.50},
}


def get_current_scenario() -> Dict:
    """Lấy cấu hình kịch bản hiện tại."""
    return SCENARIOS.get(CURRENT_SCENARIO, SCENARIOS[1])


def get_current_hazards() -> List[Dict]:
    """Lấy danh sách vùng nguy hiểm cho kịch bản hiện tại."""
    scenario = get_current_scenario()
    return HAZARD_CONFIGS[scenario['hazard_scale']]


def get_population_ratio() -> float:
    """Lấy tỷ lệ dân số cho kịch bản hiện tại."""
    scenario = get_current_scenario()
    return scenario['population_ratio']


def print_current_scenario():
    """In thông tin kịch bản hiện tại."""
    scenario = get_current_scenario()
    hazards = get_current_hazards()
    print(f"\n{'='*60}")
    print(f"KỊCH BẢN {CURRENT_SCENARIO}: {scenario['name']}")
    print(f"{'='*60}")
    print(f"  Tỷ lệ dân số: {scenario['population_ratio']*100:.0f}%")
    print(f"  Số vùng nguy hiểm: {len(hazards)}")
    for i, h in enumerate(hazards, 1):
        print(f"    {i}. {h['name']}: ({h['center_lat']}, {h['center_lon']}), "
              f"radius={h['radius_km']}km, risk={h['risk']}")
    print(f"{'='*60}\n")
