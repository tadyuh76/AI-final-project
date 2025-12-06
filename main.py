#!/usr/bin/env python3
"""
SafeRoute HCM - Tối ưu Hoá Sơ tán Bão
=====================================

Ứng dụng AI cho việc tối ưu hoá tuyến đường sơ tán bão tại Thành phố Hồ Chí Minh
sử dụng thuật toán GBFS + Grey Wolf Optimizer (GWO) với trực quan hoá
thời gian thực.

Cách chạy:
    python main.py           # Chạy giao diện đồ hoạ
    python main.py --cli     # Chạy chế độ dòng lệnh (không cần PyQt6)
    python main.py --test    # Chạy kiểm tra

Yêu cầu:
    - Python 3.10+
    - PyQt6 (cho giao diện đồ hoạ)
    - networkx, numpy, scipy (cho thuật toán)
    - osmnx (tuỳ chọn, cho dữ liệu OSM thực)
    - pyqtgraph (tuỳ chọn, cho biểu đồ)

Tác giả: tadyuh76, ayo-lole, PeanLutHuynh, Leon2285
Phiên bản: 1.0.0
"""

import sys
import argparse


def run_gui():
    """Chạy ứng dụng với giao diện đồ hoạ PyQt6."""
    try:
        from PyQt6.QtWidgets import QApplication
        from src.ui.main_window import MainWindow
        from src.ui.styles import MAIN_STYLESHEET
    except ImportError as e:
        print(f"Lỗi: Không thể import PyQt6. Hãy cài đặt PyQt6:")
        print("  pip install PyQt6 PyQt6-3D")
        print(f"\nChi tiết: {e}")
        sys.exit(1)

    # Tạo ứng dụng
    app = QApplication(sys.argv)
    app.setApplicationName("SafeRoute HCM")
    app.setOrganizationName("UEH AI Team")
    app.setApplicationVersion("1.0.0")

    # Áp dụng stylesheet
    app.setStyleSheet(MAIN_STYLESHEET)

    # Tạo và hiển thị cửa sổ chính
    window = MainWindow()
    window.show()

    # Chạy vòng lặp sự kiện
    sys.exit(app.exec())


def run_cli():
    """
    Chạy chế độ dòng lệnh không có GUI.

    Chế độ này cho phép chạy thuật toán sơ tán mà không cần giao diện đồ hoạ,
    phù hợp cho việc kiểm tra hoặc chạy trên server.
    """
    print("SafeRoute HCM - Chế độ Dòng Lệnh")
    print("=" * 50)

    try:
        from src.data.osm_loader import OSMDataLoader
        from src.algorithms.comparator import run_comparison
        from src.algorithms.base import AlgorithmConfig
    except ImportError as e:
        print(f"Lỗi import: {e}")
        print("Hãy cài đặt các dependencies: pip install -r requirements.txt")
        sys.exit(1)

    # Tải mạng lưới giao thông TP.HCM từ dữ liệu OpenStreetMap
    print("\nĐang tải mạng lưới TP.HCM...")
    loader = OSMDataLoader()
    network = loader.load_hcm_network(use_cache=True)
    loader.add_default_hazards(network, typhoon_intensity=0.7)

    # Hiển thị thống kê mạng lưới đã tải
    stats = network.get_stats()
    print(f"\nThống kê Mạng lưới:")
    print(f"  Nút: {stats.total_nodes}")
    print(f"  Cạnh: {stats.total_edges}")
    print(f"  Khu vực dân cư: {stats.population_zones}")
    print(f"  Nơi trú ẩn: {stats.shelters}")
    print(f"  Tổng dân số: {stats.total_population:,}")
    print(f"  Tổng sức chứa: {stats.total_shelter_capacity:,}")

    # Chạy so sánh các thuật toán tối ưu hoá (GBFS, GWO)
    print("\nĐang chạy so sánh thuật toán...")
    config = AlgorithmConfig(
        n_wolves=30,
        max_iterations=50
    )

    result = run_comparison(network, config, verbose=True)

    # Hiển thị kết quả so sánh
    print(f"\nKết quả:")
    print(f"  Thuật toán chiến thắng: {result.winner.value if result.winner else 'N/A'}")
    print(f"  Điểm số: {result.winner_score:.3f}")


def run_tests():
    """
    Chạy kiểm tra đơn vị với pytest.

    Thực thi tất cả các test trong thư mục tests/ để đảm bảo
    các module hoạt động đúng như mong đợi.
    """
    print("Đang chạy kiểm tra...")

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=sys.path[0] or "."
    )
    sys.exit(result.returncode)


def check_dependencies():
    """
    Kiểm tra các dependencies cần thiết.

    Duyệt qua danh sách các thư viện bắt buộc và tuỳ chọn,
    báo cáo trạng thái cài đặt của từng thư viện.
    """
    print("Kiểm tra Dependencies")
    print("=" * 50)

    dependencies = [
        ("PyQt6", "PyQt6"),
        ("networkx", "networkx"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pyqtgraph", "pyqtgraph"),
        ("osmnx", "osmnx"),
        ("pandas", "pandas"),
        ("geopandas", "geopandas"),
    ]

    all_ok = True
    for name, package in dependencies:
        try:
            __import__(package)
            status = "OK"
        except ImportError:
            status = "THIẾU"
            all_ok = False

        print(f"  {name:15} [{status}]")

    print()
    if all_ok:
        print("Tất cả dependencies đã được cài đặt!")
    else:
        print("Một số dependencies thiếu. Chạy:")
        print("  pip install -r requirements.txt")

    return all_ok


def main():
    """
    Điểm vào chính của ứng dụng.

    Phân tích các tham số dòng lệnh và khởi chạy chế độ phù hợp:
    - Mặc định: Giao diện đồ hoạ PyQt6
    - --cli: Chế độ dòng lệnh
    - --test: Chạy kiểm tra
    - --check: Kiểm tra dependencies
    """
    parser = argparse.ArgumentParser(
        description="SafeRoute HCM - Tối ưu Hoá Sơ tán Bão",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python main.py              # Chạy giao diện đồ hoạ
  python main.py --cli        # Chạy chế độ dòng lệnh
  python main.py --test       # Chạy kiểm tra
  python main.py --check      # Kiểm tra dependencies
        """
    )

    parser.add_argument(
        "--cli", action="store_true",
        help="Chạy chế độ dòng lệnh (không cần PyQt6)"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Chạy kiểm tra"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Kiểm tra dependencies"
    )

    args = parser.parse_args()

    if args.check:
        check_dependencies()
    elif args.test:
        run_tests()
    elif args.cli:
        run_cli()
    else:
        run_gui()


if __name__ == "__main__":
    main()
