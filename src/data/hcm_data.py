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
# Đây là các địa điểm thực tế có thể phục vụ như điểm sơ tán
HCM_SHELTERS: List[ShelterTemplate] = [
    # Sân vận động
    ShelterTemplate(
        name='Thong Nhat Stadium',
        lat=10.7888, lon=106.6875,
        capacity=15000, shelter_type='stadium', district='district_10'
    ),
    ShelterTemplate(
        name='Phu Tho Stadium',
        lat=10.7621, lon=106.6523,
        capacity=12000, shelter_type='stadium', district='district_11'
    ),
    ShelterTemplate(
        name='Quan Khu 7 Stadium',
        lat=10.8512, lon=106.7612,
        capacity=8000, shelter_type='stadium', district='thu_duc'
    ),

    # Đại học (sức chứa lớn)
    ShelterTemplate(
        name='HCMC University of Technology',
        lat=10.7725, lon=106.6576,
        capacity=5000, shelter_type='university', district='district_10'
    ),
    ShelterTemplate(
        name='University of Social Sciences',
        lat=10.7829, lon=106.6996,
        capacity=4000, shelter_type='university', district='district_1'
    ),
    ShelterTemplate(
        name='HCMC University of Economics',
        lat=10.7859, lon=106.6962,
        capacity=4000, shelter_type='university', district='district_3'
    ),
    ShelterTemplate(
        name='Saigon University',
        lat=10.7998, lon=106.6659,
        capacity=3500, shelter_type='university', district='tan_binh'
    ),
    ShelterTemplate(
        name='Vietnam National University',
        lat=10.8752, lon=106.8019,
        capacity=8000, shelter_type='university', district='thu_duc'
    ),

    # Bệnh viện (sức chứa hạn chế nhưng thiết yếu)
    ShelterTemplate(
        name='Cho Ray Hospital',
        lat=10.7546, lon=106.6621,
        capacity=800, shelter_type='hospital', district='district_5'
    ),
    ShelterTemplate(
        name='115 Emergency Hospital',
        lat=10.7738, lon=106.6729,
        capacity=500, shelter_type='hospital', district='district_10'
    ),
    ShelterTemplate(
        name='Thu Duc Hospital',
        lat=10.8513, lon=106.7597,
        capacity=600, shelter_type='hospital', district='thu_duc'
    ),

    # Trung tâm Cộng đồng & Trung tâm Hội nghị
    ShelterTemplate(
        name='Saigon Exhibition Center (SECC)',
        lat=10.7445, lon=106.7260,
        capacity=10000, shelter_type='convention', district='district_7'
    ),
    ShelterTemplate(
        name='White Palace Convention Center',
        lat=10.8089, lon=106.7157,
        capacity=5000, shelter_type='convention', district='binh_thanh'
    ),
    ShelterTemplate(
        name='GEM Center',
        lat=10.7831, lon=106.6950,
        capacity=3000, shelter_type='convention', district='district_1'
    ),

    # Trường học (phân bổ khắp các quận)
    ShelterTemplate(
        name='Le Hong Phong High School',
        lat=10.7815, lon=106.6885,
        capacity=2000, shelter_type='school', district='district_5'
    ),
    ShelterTemplate(
        name='Tran Dai Nghia High School',
        lat=10.7673, lon=106.6927,
        capacity=1800, shelter_type='school', district='district_1'
    ),
    ShelterTemplate(
        name='Nguyen Thi Minh Khai High School',
        lat=10.7897, lon=106.6818,
        capacity=1800, shelter_type='school', district='district_3'
    ),
    ShelterTemplate(
        name='Marie Curie High School',
        lat=10.7887, lon=106.6751,
        capacity=1500, shelter_type='school', district='district_3'
    ),
    ShelterTemplate(
        name='Gia Dinh High School',
        lat=10.8139, lon=106.6993,
        capacity=1600, shelter_type='school', district='binh_thanh'
    ),
    ShelterTemplate(
        name='Nguyen Huu Huan High School',
        lat=10.8532, lon=106.7478,
        capacity=1500, shelter_type='school', district='thu_duc'
    ),

    # Công trình tôn giáo (điểm tụ tập cộng đồng)
    ShelterTemplate(
        name='Notre-Dame Cathedral',
        lat=10.7798, lon=106.6990,
        capacity=1000, shelter_type='religious', district='district_1'
    ),
    ShelterTemplate(
        name='Vinh Nghiem Pagoda',
        lat=10.7924, lon=106.6812,
        capacity=800, shelter_type='religious', district='district_3'
    ),

    # Trung tâm thương mại (không gian lớn có mái che)
    ShelterTemplate(
        name='AEON Mall Binh Tan',
        lat=10.7483, lon=106.6078,
        capacity=8000, shelter_type='mall', district='binh_tan'
    ),
    ShelterTemplate(
        name='SC VivoCity',
        lat=10.7297, lon=106.7211,
        capacity=6000, shelter_type='mall', district='district_7'
    ),
    ShelterTemplate(
        name='AEON Mall Tan Phu',
        lat=10.7866, lon=106.6212,
        capacity=7000, shelter_type='mall', district='tan_phu'
    ),
]


# Các khu vực dễ ngập lụt đã biết ở TP.HCM
# Dựa trên dữ liệu ngập lụt lịch sử
# Cấu hình: 1 vùng lớn + 3 vùng trung bình = 4 vùng tổng cộng
FLOOD_PRONE_AREAS: List[Dict] = [
    # 1 VÙNG LỚN - Bao phủ nhiều quận (Quận 8, Quận 6, phần Bình Tân, Quận 11)
    # Đây là khu vực ngập lụt nghiêm trọng nhất, bao trùm nhiều khu dân cư
    {
        'center_lat': 10.7400,  # Giữa Quận 8 và Quận 6
        'center_lon': 106.6350,
        'radius_km': 4.5,  # Vùng lớn bao phủ nhiều quận
        'risk': 0.9,
        'name': 'Western HCM Flood Zone (Q8-Q6-Binh Tan)'
    },

    # 2 VÙNG TRUNG BÌNH - Quận 4 (tâm điểm khu dân cư, vùng trũng thấp)
    # Đặt chính xác tại trung tâm khu dân cư Quận 4
    {
        'center_lat': 10.7579,  # Tâm Quận 4
        'center_lon': 106.7057,
        'radius_km': 1.8,  # Vùng trung bình
        'risk': 0.85,
        'name': 'District 4 Central (Low-lying)'
    },

    # 3 VÙNG TRUNG BÌNH - Quận 7 Phú Mỹ Hưng (khu vực ven sông)
    # Khu đô thị mới nhưng vẫn có nguy cơ ngập do gần sông
    {
        'center_lat': 10.7285,
        'center_lon': 106.7158,
        'radius_km': 2.2,  # Vùng trung bình
        'risk': 0.75,
        'name': 'Phu My Hung Riverside (District 7)'
    },

    # 4 VÙNG TRUNG BÌNH - Nhà Bè (ven biển, ngập lụt do triều cường)
    # Khu vực ven biển dễ bị ngập do triều cường và nước biển dâng
    {
        'center_lat': 10.6947,  # Tâm Nhà Bè
        'center_lon': 106.7378,
        'radius_km': 2.5,  # Vùng trung bình
        'risk': 0.8,
        'name': 'Nha Be Coastal (Tidal Flooding)'
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
