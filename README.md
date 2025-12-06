# SafeRoute HCM - Hệ Thống Tối Ưu Hóa Đường Sơ Tán Thông Minh

## Mô Tả Dự Án

**SafeRoute HCM** là một hệ thống tối ưu hóa tuyến đường sơ tán sử dụng trí tuệ nhân tạo, được thiết kế để ứng phó với các tình huống khẩn cấp do bão và lũ lụt tại Thành phố Hồ Chí Minh, Việt Nam.

Hệ thống sử dụng hai phương pháp tối ưu hóa:
- **GBFS (Greedy Best First Search)**: Tìm kiếm đường đi nhanh dựa trên heuristic
- **GWO (Grey Wolf Optimizer)**: Tối ưu hóa phân phối luồng toàn cục dựa trên hành vi bầy sói xám

## Tính Năng Chính

- Mô hình hóa mạng lưới đường của 24 quận TP.HCM
- Hơn 50 điểm trú ẩn (trường học, bệnh viện, sân vận động)
- Mô phỏng vùng nguy hiểm động (lũ lụt, ngập úng)
- Giao diện đồ họa trực quan với bản đồ tương tác
- So sánh hiệu suất giữa các thuật toán
- Mô phỏng sơ tán theo thời gian thực

## Yêu Cầu Hệ Thống

### Phần mềm bắt buộc
- Python 3.10 trở lên

### Thư viện cần thiết
```
PyQt6>=6.6.0          # Giao diện đồ họa
networkx>=3.2         # Xử lý đồ thị
numpy>=1.26.0         # Tính toán số học
pandas>=2.1.0         # Xử lý dữ liệu
pyqtgraph>=0.13.3     # Biểu đồ thời gian thực
scipy>=1.11.0         # Tính toán khoa học
```

### Thư viện tùy chọn
```
osmnx>=1.7.0          # Dữ liệu OpenStreetMap
geopandas>=0.14.0     # Xử lý dữ liệu địa lý
```

## Cài Đặt

### 1. Clone dự án
```bash
git clone <repository-url>
cd final
```

### 2. Tạo môi trường ảo (khuyến nghị)
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

## Cách Sử Dụng

### Chạy giao diện đồ họa
```bash
python main.py
```

### Chạy chế độ dòng lệnh (không cần PyQt6)
```bash
python main.py --cli
```

### Chạy kiểm tra
```bash
python main.py --test
# hoặc
python -m pytest tests/ -v
```

## Cấu Trúc Dự Án

```
final/
├── main.py                 # Điểm khởi chạy ứng dụng
├── requirements.txt        # Danh sách thư viện
├── README.md              # Tài liệu hướng dẫn (file này)
├── REPORT.md              # Báo cáo chi tiết dự án
│
├── src/                   # Mã nguồn chính
│   ├── __init__.py
│   │
│   ├── models/            # Mô hình dữ liệu
│   │   ├── node.py        # Các loại nút (Khu dân cư, Nơi trú ẩn, Vùng nguy hiểm)
│   │   ├── edge.py        # Cạnh đường (loại đường, sức chứa)
│   │   └── network.py     # Mạng lưới sơ tán
│   │
│   ├── algorithms/        # Thuật toán tối ưu
│   │   ├── base.py        # Lớp cơ sở và cấu hình
│   │   ├── gbfs.py        # Greedy Best First Search
│   │   ├── gwo.py         # Grey Wolf Optimizer
│   │   └── comparator.py  # So sánh thuật toán
│   │
│   ├── simulation/        # Mô phỏng
│   │   ├── engine.py      # Công cụ mô phỏng
│   │   ├── traffic.py     # Mô hình giao thông
│   │   └── events.py      # Sự kiện động
│   │
│   ├── data/              # Dữ liệu
│   │   ├── hcm_data.py    # Dữ liệu TP.HCM (quận, nơi trú ẩn, vùng ngập)
│   │   └── osm_loader.py  # Tải dữ liệu OpenStreetMap
│   │
│   └── ui/                # Giao diện người dùng
│       ├── main_window.py # Cửa sổ chính
│       ├── map_widget.py  # Widget bản đồ
│       ├── control_panel.py # Panel điều khiển
│       ├── dashboard.py   # Bảng điều khiển
│       ├── comparison_view.py # So sánh thuật toán
│       └── styles.py      # Kiểu dáng giao diện
│
└── tests/                 # Kiểm thử đơn vị
    ├── test_models.py     # Kiểm thử mô hình
    ├── test_algorithms.py # Kiểm thử thuật toán
    └── test_simulation.py # Kiểm thử mô phỏng
```

## Thuật Toán

### 1. GBFS (Greedy Best First Search)

Thuật toán tìm kiếm tham lam ưu tiên mở rộng nút có triển vọng nhất theo hàm heuristic đa mục tiêu:

**Hàm chi phí:**
```
Cost = travel_time × (1 + flood_risk² × 100)
```

**Ưu điểm:**
- Tốc độ nhanh, phù hợp ứng dụng thời gian thực
- Đơn giản, dễ hiểu và triển khai

**Nhược điểm:**
- Có thể không tìm được đường tối ưu toàn cục

### 2. GWO (Grey Wolf Optimizer)

Thuật toán metaheuristic lấy cảm hứng từ hành vi săn mồi của bầy sói xám:

**Phân cấp sói:**
- Alpha (α): Giải pháp tốt nhất
- Beta (β): Giải pháp tốt thứ hai
- Delta (δ): Giải pháp tốt thứ ba
- Omega (ω): Các cá thể còn lại

**Ưu điểm:**
- Tối ưu hóa toàn cục, phân phối luồng đều
- Tránh được cực tiểu địa phương

**Nhược điểm:**
- Thời gian tính toán lâu hơn GBFS

## Dữ Liệu TP.HCM

Hệ thống sử dụng dữ liệu thực về:

### Quận/Huyện (24 đơn vị)
- Quận 1, 3, 4, 5, 6, 7, 8, 10, 11, 12
- Tân Bình, Tân Phú, Bình Tân, Bình Thạnh
- Phú Nhuận, Gò Vấp, Thủ Đức
- Nhà Bè, Cần Giờ, Bình Chánh, Hóc Môn, Củ Chi

### Nơi trú ẩn (50+ điểm)
- Trường học các cấp
- Bệnh viện và trung tâm y tế
- Sân vận động và nhà thi đấu
- Trung tâm hội nghị

### Vùng nguy hiểm
- Khu vực ngập lụt lịch sử
- Vùng ven sông, kênh rạch
- Khu vực thấp trũng

## Giao Diện Người Dùng

### Bản đồ tương tác
- Hiển thị mạng lưới đường
- Các khu dân cư (màu xanh dương)
- Nơi trú ẩn (màu xanh lá)
- Vùng nguy hiểm (màu đỏ)
- Tuyến đường sơ tán (đường kẻ)

### Panel điều khiển
- Chọn thuật toán (GBFS/GWO)
- Cấu hình tham số
- Điều chỉnh cường độ bão (cấp 1-5)
- Điều khiển mô phỏng

### Bảng điều khiển
- Tiến trình sơ tán
- Số người đã sơ tán
- Tình trạng nơi trú ẩn
- Thời gian ước tính

### So sánh thuật toán
- Biểu đồ hội tụ
- Bảng hiệu suất
- Biểu đồ radar đa tiêu chí

## Kiểm Thử

Dự án bao gồm bộ kiểm thử đầy đủ với 135+ test cases:

```bash
# Chạy tất cả kiểm thử
python -m pytest tests/ -v

# Chạy kiểm thử từng module
python -m pytest tests/test_models.py -v
python -m pytest tests/test_algorithms.py -v
python -m pytest tests/test_simulation.py -v
```

## Tác Giả

- **tadyuh76** - Trưởng nhóm, Kiến trúc hệ thống
- **ayo-lole** - Phát triển thuật toán
- **PeanLutHuynh** - Giao diện người dùng
- **Leon2285** - Dữ liệu và kiểm thử

## Giấy Phép

Dự án này được phát triển cho mục đích học tập tại Trường Đại học Kinh tế TP.HCM (UEH).

## Tài Liệu Tham Khảo

- Báo cáo chi tiết: [REPORT.md](REPORT.md)
- NetworkX Documentation: https://networkx.org/
- PyQt6 Documentation: https://www.riverbankcomputing.com/static/Docs/PyQt6/
- Grey Wolf Optimizer: Mirjalili et al. (2014)

---

**SafeRoute HCM** - Bảo vệ người dân trước thiên tai bằng công nghệ AI
