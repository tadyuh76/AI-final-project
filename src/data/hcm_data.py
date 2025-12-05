"""
Dữ liệu Thành phố Hồ Chí Minh: Quận, dân số và khu vực ngập lụt.
Cung cấp dữ liệu thực tế cho mô phỏng sơ tán.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class DistrictData:
    """Dữ liệu cho một quận ở TP.HCM."""
    name: str
    name_vi: str  # Tên tiếng Việt
    center_lat: float
    center_lon: float
    population: int
    area_km2: float
    flood_risk: float  # 0.0 đến 1.0, rủi ro ngập lụt lịch sử


# Các quận Thành phố Hồ Chí Minh với dữ liệu ước tính năm 2023
# Dữ liệu dân số là ước tính dựa trên thống kê có sẵn
HCM_DISTRICTS: Dict[str, DistrictData] = {
    'district_1': DistrictData(
        name='District 1', name_vi='Quan 1',
        center_lat=10.7769, center_lon=106.7009,
        population=180000, area_km2=7.73, flood_risk=0.3
    ),
    'district_3': DistrictData(
        name='District 3', name_vi='Quan 3',
        center_lat=10.7838, center_lon=106.6834,
        population=190000, area_km2=4.92, flood_risk=0.2
    ),
    'district_4': DistrictData(
        name='District 4', name_vi='Quan 4',
        center_lat=10.7579, center_lon=106.7057,
        population=180000, area_km2=4.18, flood_risk=0.8  # Vùng trũng thấp
    ),
    'district_5': DistrictData(
        name='District 5', name_vi='Quan 5',
        center_lat=10.7554, center_lon=106.6631,
        population=170000, area_km2=4.27, flood_risk=0.3
    ),
    'district_6': DistrictData(
        name='District 6', name_vi='Quan 6',
        center_lat=10.7478, center_lon=106.6353,
        population=250000, area_km2=7.19, flood_risk=0.5
    ),
    'district_7': DistrictData(
        name='District 7', name_vi='Quan 7',
        center_lat=10.7365, center_lon=106.7218,
        population=310000, area_km2=35.69, flood_risk=0.7  # Khu vực Phú Mỹ Hưng
    ),
    'district_8': DistrictData(
        name='District 8', name_vi='Quan 8',
        center_lat=10.7255, center_lon=106.6373,
        population=430000, area_km2=19.18, flood_risk=0.9  # Rất dễ ngập lụt
    ),
    'district_10': DistrictData(
        name='District 10', name_vi='Quan 10',
        center_lat=10.7732, center_lon=106.6679,
        population=230000, area_km2=5.72, flood_risk=0.3
    ),
    'district_11': DistrictData(
        name='District 11', name_vi='Quan 11',
        center_lat=10.7647, center_lon=106.6501,
        population=230000, area_km2=5.14, flood_risk=0.4
    ),
    'district_12': DistrictData(
        name='District 12', name_vi='Quan 12',
        center_lat=10.8671, center_lon=106.6413,
        population=520000, area_km2=52.78, flood_risk=0.4
    ),
    'binh_tan': DistrictData(
        name='Binh Tan', name_vi='Binh Tan',
        center_lat=10.7654, center_lon=106.6037,
        population=780000, area_km2=51.89, flood_risk=0.6
    ),
    'binh_thanh': DistrictData(
        name='Binh Thanh', name_vi='Binh Thanh',
        center_lat=10.8105, center_lon=106.7091,
        population=490000, area_km2=20.76, flood_risk=0.5
    ),
    'go_vap': DistrictData(
        name='Go Vap', name_vi='Go Vap',
        center_lat=10.8386, center_lon=106.6652,
        population=670000, area_km2=19.74, flood_risk=0.3
    ),
    'phu_nhuan': DistrictData(
        name='Phu Nhuan', name_vi='Phu Nhuan',
        center_lat=10.7997, center_lon=106.6825,
        population=180000, area_km2=4.88, flood_risk=0.2
    ),
    'tan_binh': DistrictData(
        name='Tan Binh', name_vi='Tan Binh',
        center_lat=10.8015, center_lon=106.6528,
        population=470000, area_km2=22.38, flood_risk=0.4
    ),
    'tan_phu': DistrictData(
        name='Tan Phu', name_vi='Tan Phu',
        center_lat=10.7918, center_lon=106.6280,
        population=450000, area_km2=16.06, flood_risk=0.5
    ),
    'thu_duc': DistrictData(
        name='Thu Duc City', name_vi='Thanh pho Thu Duc',
        center_lat=10.8514, center_lon=106.7539,
        population=1100000, area_km2=211.56, flood_risk=0.4
    ),
    'nha_be': DistrictData(
        name='Nha Be', name_vi='Nha Be',
        center_lat=10.6947, center_lon=106.7378,
        population=180000, area_km2=100.41, flood_risk=0.9  # Ven biển, rất dễ ngập lụt
    ),
    # Can Gio removed - too far from main HCM area (outside bounds)
    # Would require separate evacuation planning due to geographic isolation
}


@dataclass
class ShelterTemplate:
    """Mẫu để tạo các điểm trú ẩn."""
    name: str
    lat: float
    lon: float
    capacity: int
    shelter_type: str
    district: str


# Các điểm trú ẩn chính ở Thành phố Hồ Chí Minh
# 50 điểm trú ẩn phân bổ đều khắp thành phố, sức chứa x2
# Tổng sức chứa mục tiêu: ~800,000 người
HCM_SHELTERS: List[ShelterTemplate] = [
    # ==================== THỦ ĐỨC (Đông Bắc - 6 shelters) ====================
    ShelterTemplate(
        name='Vietnam National University',
        lat=10.8752, lon=106.8019,
        capacity=50000, shelter_type='university', district='thu_duc'
    ),
    ShelterTemplate(
        name='Suoi Tien Theme Park',
        lat=10.8720, lon=106.7950,
        capacity=40000, shelter_type='park', district='thu_duc'
    ),
    ShelterTemplate(
        name='Linh Trung Industrial Zone',
        lat=10.8650, lon=106.7900,
        capacity=30000, shelter_type='industrial', district='thu_duc'
    ),
    ShelterTemplate(
        name='Gigamall Thu Duc',
        lat=10.8350, lon=106.7650,
        capacity=30000, shelter_type='mall', district='thu_duc'
    ),
    ShelterTemplate(
        name='FPT University HCMC',
        lat=10.8410, lon=106.8100,
        capacity=20000, shelter_type='university', district='thu_duc'
    ),
    ShelterTemplate(
        name='Cat Lai Port Area',
        lat=10.7680, lon=106.7850,
        capacity=20000, shelter_type='industrial', district='thu_duc'
    ),

    # ==================== QUẬN 12 (Bắc - 4 shelters) ====================
    ShelterTemplate(
        name='AEON Mall Binh Duong Canary',
        lat=10.8750, lon=106.6550,
        capacity=30000, shelter_type='mall', district='district_12'
    ),
    ShelterTemplate(
        name='Tan Phu Trung Industrial Zone',
        lat=10.8800, lon=106.6100,
        capacity=24000, shelter_type='industrial', district='district_12'
    ),
    ShelterTemplate(
        name='District 12 Stadium',
        lat=10.8650, lon=106.6450,
        capacity=16000, shelter_type='stadium', district='district_12'
    ),
    ShelterTemplate(
        name='District 12 Sports Center',
        lat=10.8680, lon=106.6450,
        capacity=8000, shelter_type='sports', district='district_12'
    ),

    # ==================== GÒ VẤP (Bắc Trung Tâm - 4 shelters) ====================
    ShelterTemplate(
        name='Industrial University of HCMC',
        lat=10.8280, lon=106.6850,
        capacity=20000, shelter_type='university', district='go_vap'
    ),
    ShelterTemplate(
        name='Go Vap Stadium',
        lat=10.8350, lon=106.6700,
        capacity=16000, shelter_type='stadium', district='go_vap'
    ),
    ShelterTemplate(
        name='Lotte Mart Go Vap',
        lat=10.8380, lon=106.6720,
        capacity=20000, shelter_type='mall', district='go_vap'
    ),
    ShelterTemplate(
        name='Go Vap Central Park',
        lat=10.8350, lon=106.6750,
        capacity=12000, shelter_type='park', district='go_vap'
    ),

    # ==================== TÂN BÌNH (Tây Bắc - 3 shelters) ====================
    ShelterTemplate(
        name='Tan Binh Industrial Park',
        lat=10.8150, lon=106.6250,
        capacity=20000, shelter_type='industrial', district='tan_binh'
    ),
    ShelterTemplate(
        name='Tan Binh Stadium',
        lat=10.8050, lon=106.6500,
        capacity=14000, shelter_type='stadium', district='tan_binh'
    ),
    ShelterTemplate(
        name='Saigon University',
        lat=10.7998, lon=106.6659,
        capacity=16000, shelter_type='university', district='tan_binh'
    ),

    # ==================== TÂN PHÚ (Tây - 3 shelters) ====================
    ShelterTemplate(
        name='AEON Mall Tan Phu',
        lat=10.7866, lon=106.6212,
        capacity=40000, shelter_type='mall', district='tan_phu'
    ),
    ShelterTemplate(
        name='Tan Phu High School Complex',
        lat=10.7900, lon=106.6300,
        capacity=10000, shelter_type='school', district='tan_phu'
    ),
    ShelterTemplate(
        name='Tan Phu District Hospital',
        lat=10.7850, lon=106.6280,
        capacity=4000, shelter_type='hospital', district='tan_phu'
    ),

    # ==================== BÌNH TÂN (Tây Nam - 4 shelters) ====================
    ShelterTemplate(
        name='AEON Mall Binh Tan',
        lat=10.7483, lon=106.6078,
        capacity=40000, shelter_type='mall', district='binh_tan'
    ),
    ShelterTemplate(
        name='Tan Tao Industrial Zone',
        lat=10.7420, lon=106.5950,
        capacity=30000, shelter_type='industrial', district='binh_tan'
    ),
    ShelterTemplate(
        name='Vinh Loc Industrial Park',
        lat=10.7350, lon=106.5850,
        capacity=24000, shelter_type='industrial', district='binh_tan'
    ),
    ShelterTemplate(
        name='Big C Binh Tan',
        lat=10.7550, lon=106.6150,
        capacity=16000, shelter_type='mall', district='binh_tan'
    ),

    # ==================== QUẬN 6 & 8 (Tây Nam - 3 shelters) ====================
    ShelterTemplate(
        name='Phu Lam Park',
        lat=10.7420, lon=106.6280,
        capacity=10000, shelter_type='park', district='district_6'
    ),
    ShelterTemplate(
        name='Binh Phu Community Center',
        lat=10.7480, lon=106.6250,
        capacity=8000, shelter_type='school', district='district_6'
    ),
    ShelterTemplate(
        name='District 8 Evacuation Center',
        lat=10.7255, lon=106.6373,
        capacity=15000, shelter_type='convention', district='district_8'
    ),

    # ==================== QUẬN 11 (Trung Tâm Tây - 2 shelters) ====================
    ShelterTemplate(
        name='Phu Tho Stadium',
        lat=10.7621, lon=106.6523,
        capacity=40000, shelter_type='stadium', district='district_11'
    ),
    ShelterTemplate(
        name='Dam Sen Park',
        lat=10.7680, lon=106.6380,
        capacity=30000, shelter_type='park', district='district_11'
    ),

    # ==================== QUẬN 10 (Trung Tâm - 2 shelters) ====================
    ShelterTemplate(
        name='Thong Nhat Stadium',
        lat=10.7888, lon=106.6875,
        capacity=50000, shelter_type='stadium', district='district_10'
    ),
    ShelterTemplate(
        name='HCMC University of Technology',
        lat=10.7725, lon=106.6576,
        capacity=30000, shelter_type='university', district='district_10'
    ),

    # ==================== QUẬN 5 (Trung Tâm - 2 shelters) ====================
    ShelterTemplate(
        name='HCMC University of Education',
        lat=10.7640, lon=106.6820,
        capacity=20000, shelter_type='university', district='district_5'
    ),
    ShelterTemplate(
        name='Cho Ray Hospital Area',
        lat=10.7546, lon=106.6621,
        capacity=6000, shelter_type='hospital', district='district_5'
    ),

    # ==================== QUẬN 3 (Trung Tâm - 2 shelters) ====================
    ShelterTemplate(
        name='HCMC University of Economics',
        lat=10.7859, lon=106.6962,
        capacity=20000, shelter_type='university', district='district_3'
    ),
    ShelterTemplate(
        name='Vinh Nghiem Pagoda Complex',
        lat=10.7924, lon=106.6812,
        capacity=8000, shelter_type='religious', district='district_3'
    ),

    # ==================== PHÚ NHUẬN (Trung Tâm Đông - 2 shelters) ====================
    ShelterTemplate(
        name='Gia Dinh Park',
        lat=10.8100, lon=106.6900,
        capacity=20000, shelter_type='park', district='phu_nhuan'
    ),
    ShelterTemplate(
        name='Phu Nhuan Community Center',
        lat=10.8000, lon=106.6800,
        capacity=8000, shelter_type='school', district='phu_nhuan'
    ),

    # ==================== BÌNH THẠNH (Đông - 3 shelters) ====================
    ShelterTemplate(
        name='White Palace Convention Center',
        lat=10.8089, lon=106.7157,
        capacity=30000, shelter_type='convention', district='binh_thanh'
    ),
    ShelterTemplate(
        name='Van Lang University',
        lat=10.8180, lon=106.7020,
        capacity=20000, shelter_type='university', district='binh_thanh'
    ),
    ShelterTemplate(
        name='Van Thanh Park',
        lat=10.7950, lon=106.7150,
        capacity=16000, shelter_type='park', district='binh_thanh'
    ),

    # ==================== QUẬN 1 (Trung Tâm - 2 shelters) ====================
    ShelterTemplate(
        name='Tao Dan Park',
        lat=10.7780, lon=106.6920,
        capacity=20000, shelter_type='park', district='district_1'
    ),
    ShelterTemplate(
        name='September 23 Park',
        lat=10.7620, lon=106.6920,
        capacity=16000, shelter_type='park', district='district_1'
    ),

    # ==================== QUẬN 4 (Nam Trung Tâm - 1 shelter) ====================
    ShelterTemplate(
        name='District 4 Community Center',
        lat=10.7579, lon=106.7057,
        capacity=10000, shelter_type='convention', district='district_4'
    ),

    # ==================== QUẬN 7 (Nam - 4 shelters) ====================
    ShelterTemplate(
        name='Saigon Exhibition Center (SECC)',
        lat=10.7445, lon=106.7260,
        capacity=60000, shelter_type='convention', district='district_7'
    ),
    ShelterTemplate(
        name='SC VivoCity',
        lat=10.7297, lon=106.7211,
        capacity=36000, shelter_type='mall', district='district_7'
    ),
    ShelterTemplate(
        name='Ton Duc Thang University',
        lat=10.7320, lon=106.6990,
        capacity=24000, shelter_type='university', district='district_7'
    ),
    ShelterTemplate(
        name='District 7 Sports Complex',
        lat=10.7320, lon=106.7150,
        capacity=20000, shelter_type='stadium', district='district_7'
    ),

    # ==================== NHÀ BÈ (Cực Nam - 3 shelters) ====================
    ShelterTemplate(
        name='Hiep Phuoc Industrial Zone',
        lat=10.6850, lon=106.7500,
        capacity=16000, shelter_type='industrial', district='nha_be'
    ),
    ShelterTemplate(
        name='Nha Be District Center',
        lat=10.6947, lon=106.7378,
        capacity=10000, shelter_type='convention', district='nha_be'
    ),
    ShelterTemplate(
        name='Nha Be South Community Center',
        lat=10.6700, lon=106.7200,
        capacity=8000, shelter_type='school', district='nha_be'
    ),
]


# Các khu vực dễ ngập lụt đã biết ở TP.HCM
# 4 vùng nguy hiểm: 1 lớn (4.5km) và 3 trung bình (2-2.5km)
FLOOD_PRONE_AREAS: List[Dict] = [
    # VÙNG LỚN: Quận 8, 6, 4, 7 - Khu vực trũng phía Nam (bao phủ nhiều quận)
    {
        'center_lat': 10.7350,
        'center_lon': 106.6700,
        'radius_km': 4.5,
        'risk': 0.85,
        'name': 'Southern Lowlands'
    },

    # VÙNG TRUNG BÌNH 1: Nhà Bè - Ngập lụt ven biển phía cực Nam
    {
        'center_lat': 10.6900,
        'center_lon': 106.7400,
        'radius_km': 2.5,
        'risk': 0.9,
        'name': 'Nha Be Coastal'
    },

    # VÙNG TRUNG BÌNH 2: Bình Tân - Khu công nghiệp thoát nước kém
    {
        'center_lat': 10.7450,
        'center_lon': 106.5950,
        'radius_km': 2.0,
        'risk': 0.7,
        'name': 'Binh Tan Industrial'
    },

    # VÙNG TRUNG BÌNH 3: Thủ Đức Đông - Gần sông Đồng Nai
    {
        'center_lat': 10.8550,
        'center_lon': 106.7850,
        'radius_km': 2.0,
        'risk': 0.6,
        'name': 'Thu Duc Riverside'
    },
]


# Ranh giới địa lý Thành phố Hồ Chí Minh
HCM_BOUNDS = {
    'north': 10.9000,
    'south': 10.6500,
    'east': 106.8500,
    'west': 106.5500,
    'center_lat': 10.7769,
    'center_lon': 106.7009,
}


def get_total_population() -> int:
    """Lấy tổng dân số của tất cả các quận."""
    return sum(d.population for d in HCM_DISTRICTS.values())


def get_total_shelter_capacity() -> int:
    """Lấy tổng sức chứa của tất cả các điểm trú ẩn."""
    return sum(s.capacity for s in HCM_SHELTERS)


def get_districts_by_flood_risk(min_risk: float = 0.5) -> List[DistrictData]:
    """Lấy các quận có rủi ro ngập lụt trên ngưỡng."""
    return [d for d in HCM_DISTRICTS.values() if d.flood_risk >= min_risk]


def get_shelter_capacity_by_type() -> Dict[str, int]:
    """Lấy tổng sức chứa điểm trú ẩn được nhóm theo loại."""
    capacity_by_type: Dict[str, int] = {}
    for shelter in HCM_SHELTERS:
        if shelter.shelter_type not in capacity_by_type:
            capacity_by_type[shelter.shelter_type] = 0
        capacity_by_type[shelter.shelter_type] += shelter.capacity
    return capacity_by_type


# In tóm tắt khi module được chạy trực tiếp
if __name__ == '__main__':
    print("Tóm tắt Dữ liệu Sơ tán Thành phố Hồ Chí Minh")
    print("=" * 50)
    print(f"Tổng số Quận: {len(HCM_DISTRICTS)}")
    print(f"Tổng Dân số: {get_total_population():,}")
    print(f"Tổng số Điểm trú ẩn: {len(HCM_SHELTERS)}")
    print(f"Tổng Sức chứa Điểm trú ẩn: {get_total_shelter_capacity():,}")
    print()
    print("Sức chứa Điểm trú ẩn theo Loại:")
    for stype, cap in get_shelter_capacity_by_type().items():
        print(f"  {stype}: {cap:,}")
    print()
    print("Các Quận Rủi ro Cao (>50% rủi ro ngập lụt):")
    for d in get_districts_by_flood_risk(0.5):
        print(f"  {d.name}: {d.flood_risk:.0%} rủi ro, dân số {d.population:,}")
