#!/usr/bin/env python3
"""
SafeRoute HCM - Toi uu Hoa So tan Bao
=====================================

Ung dung AI cho viec toi uu hoa tuyen duong so tan bao tai Thanh pho Ho Chi Minh
su dung thuat toan Hybrid GBFS + Grey Wolf Optimizer (GWO) voi truc quan hoa
thoi gian thuc.

Cach chay:
    python main.py           # Chay giao dien do hoa
    python main.py --cli     # Chay che do dong lenh (khong can PyQt6)
    python main.py --test    # Chay kiem tra

Yeu cau:
    - Python 3.10+
    - PyQt6 (cho giao dien do hoa)
    - networkx, numpy, scipy (cho thuat toan)
    - osmnx (tuy chon, cho du lieu OSM thuc)
    - pyqtgraph (tuy chon, cho bieu do)

Tac gia: UEH AI Team
Phien ban: 1.0.0
"""

import sys
import argparse


def run_gui():
    """Chay ung dung voi giao dien do hoa PyQt6."""
    try:
        from PyQt6.QtWidgets import QApplication
        from src.ui.main_window import MainWindow
        from src.ui.styles import MAIN_STYLESHEET
    except ImportError as e:
        print(f"Loi: Khong the import PyQt6. Hay cai dat PyQt6:")
        print("  pip install PyQt6 PyQt6-3D")
        print(f"\nChi tiet: {e}")
        sys.exit(1)

    # Tao ung dung
    app = QApplication(sys.argv)
    app.setApplicationName("SafeRoute HCM")
    app.setOrganizationName("UEH AI Team")
    app.setApplicationVersion("1.0.0")

    # Ap dung stylesheet
    app.setStyleSheet(MAIN_STYLESHEET)

    # Tao va hien thi cua so chinh
    window = MainWindow()
    window.show()

    # Chay vong lap su kien
    sys.exit(app.exec())


def run_cli():
    """Chay che do dong lenh khong co GUI."""
    print("SafeRoute HCM - Che do Dong Lenh")
    print("=" * 50)

    try:
        from src.data.osm_loader import OSMDataLoader
        from src.algorithms.comparator import run_comparison
        from src.algorithms.base import AlgorithmConfig
    except ImportError as e:
        print(f"Loi import: {e}")
        print("Hay cai dat cac dependencies: pip install -r requirements.txt")
        sys.exit(1)

    # Tai mang luoi
    print("\nDang tai mang luoi TP.HCM...")
    loader = OSMDataLoader()
    network = loader.load_hcm_network(use_cache=True)
    loader.add_default_hazards(network, typhoon_intensity=0.7)

    stats = network.get_stats()
    print(f"\nThong ke Mang luoi:")
    print(f"  Nut: {stats.total_nodes}")
    print(f"  Canh: {stats.total_edges}")
    print(f"  Khu vuc dan cu: {stats.population_zones}")
    print(f"  Noi tru an: {stats.shelters}")
    print(f"  Tong dan so: {stats.total_population:,}")
    print(f"  Tong suc chua: {stats.total_shelter_capacity:,}")

    # Chay so sanh thuat toan
    print("\nDang chay so sanh thuat toan...")
    config = AlgorithmConfig(
        n_wolves=30,
        max_iterations=50
    )

    result = run_comparison(network, config, verbose=True)

    print(f"\nKet qua:")
    print(f"  Thuat toan chien thang: {result.winner.value if result.winner else 'N/A'}")
    print(f"  Diem so: {result.winner_score:.3f}")


def run_tests():
    """Chay kiem tra."""
    print("Dang chay kiem tra...")

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=sys.path[0] or "."
    )
    sys.exit(result.returncode)


def check_dependencies():
    """Kiem tra cac dependencies."""
    print("Kiem tra Dependencies")
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
            status = "THIEU"
            all_ok = False

        print(f"  {name:15} [{status}]")

    print()
    if all_ok:
        print("Tat ca dependencies da duoc cai dat!")
    else:
        print("Mot so dependencies thieu. Chay:")
        print("  pip install -r requirements.txt")

    return all_ok


def main():
    """Diem vao chinh cua ung dung."""
    parser = argparse.ArgumentParser(
        description="SafeRoute HCM - Toi uu Hoa So tan Bao",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vi du:
  python main.py              # Chay giao dien do hoa
  python main.py --cli        # Chay che do dong lenh
  python main.py --test       # Chay kiem tra
  python main.py --check      # Kiem tra dependencies
        """
    )

    parser.add_argument(
        "--cli", action="store_true",
        help="Chay che do dong lenh (khong can PyQt6)"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Chay kiem tra"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Kiem tra dependencies"
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
