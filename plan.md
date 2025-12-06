KẾ HOẠCH BÁO CÁO 50 TRANG - SAFEROUTE HCM
Thông tin dự án
Tên dự án: SafeRoute HCM - Hệ thống tối ưu hóa tuyến đường sơ tán khẩn cấp
Giải thuật chính: GBFS (Greedy Best First Search) và GWO (Grey Wolf Optimizer)
Ngôn ngữ: Python 3.10+
Framework UI: PyQt6
Phân công nhiệm vụ (dựa trên git history)
Thành viên Vai trò chính Đóng góp cụ thể
tadyuh76 Kiến trúc sư hệ thống, Lead Developer Thiết kế kiến trúc tổng thể, triển khai GBFS & GWO, tối ưu hóa thuật toán, xử lý dữ liệu OSM
Leonn2285 Phát triển giao diện so sánh Triển khai comparison view, dashboard thời gian thực, logic điều khiển simulation
PeanLutHuynh Tối ưu simulation & cấu hình Sửa lỗi risk mapping, UI cleanup, tăng tốc simulation, cấu hình hazard zones
ayo-lole Thiết kế UI/UX & dịch thuật Dịch giao diện tiếng Việt, thiết kế icons, styling checkboxes
CẤU TRÚC BÁO CÁO 50 TRANG
MỤC LỤC (1 trang)
CHƯƠNG 1. TỔNG QUAN (8-10 trang)
1.1. Giới Thiệu Về Bài Toán Sơ Tán Khẩn Cấp (3-4 trang)
1.1.1. Bối cảnh và tầm quan trọng của sơ tán khẩn cấp
1.1.2. Tình hình thiên tai tại Việt Nam và TP.HCM
1.1.3. Thách thức trong việc lập kế hoạch sơ tán
1.1.4. Giới thiệu hệ thống SafeRoute HCM
1.2. Phát Biểu Bài Toán (3-4 trang)
1.2.1. Định nghĩa bài toán tối ưu hóa tuyến đường sơ tán
1.2.2. Mô hình toán học
Đồ thị mạng lưới giao thông G = (V, E)
Hàm mục tiêu đa tiêu chí
Các ràng buộc (sức chứa, rủi ro, thời gian)
1.2.3. Độ phức tạp của bài toán (NP-hard)
1.3. Một Số Hướng Tiếp Cận Giải Quyết Bài Toán (2-3 trang)
1.3.1. Các phương pháp tìm kiếm cổ điển (Dijkstra, A*)
1.3.2. Các phương pháp heuristic (Greedy, GBFS)
1.3.3. Các phương pháp metaheuristic (GA, PSO, GWO)
1.3.4. Lý do chọn GBFS và GWO cho bài toán
CHƯƠNG 2. CƠ SỞ LÝ THUYẾT (12-15 trang)
2.1. Lý Thuyết Đồ Thị và Tìm Đường (4-5 trang)
2.1.1. Biểu diễn đồ thị (ma trận kề, danh sách kề)
2.1.2. Các thuật toán tìm đường cơ bản
Dijkstra's Algorithm
A* Algorithm
2.1.3. Thuật toán Greedy Best First Search (GBFS)
Nguyên lý tham lam
Hàm heuristic
Phân tích độ phức tạp O(E log V)
So sánh với A\*
2.2. Giải Thuật Tối Ưu Hóa Đàn Sói Xám (GWO) (6-8 trang)
2.2.1. Giới thiệu về thuật toán lấy cảm hứng từ thiên nhiên
2.2.2. Hành vi săn mồi của đàn sói xám trong tự nhiên
Cấu trúc phân cấp: Alpha, Beta, Delta, Omega
Ba giai đoạn săn mồi: bao vây, truy đuổi, tấn công
2.2.3. Mô hình toán học của GWO
Vector vị trí và hệ số A, C
Công thức cập nhật vị trí
Tham số a và cơ chế exploration/exploitation
2.2.4. Pseudocode thuật toán GWO
2.2.5. Phân tích độ phức tạp
2.2.6. Ưu điểm và hạn chế so với các metaheuristic khác
2.3. Mô Hình Giao Thông và Rủi Ro (2-3 trang)
2.3.1. Mô hình BPR (Bureau of Public Roads) cho tắc nghẽn
2.3.2. Hàm tính chi phí đa yếu tố
2.3.3. Mô hình vùng nguy hiểm với risk decay
CHƯƠNG 3. THIẾT KẾ VÀ CÀI ĐẶT HỆ THỐNG (12-15 trang)
3.1. Kiến Trúc Tổng Thể (2-3 trang)
3.1.1. Sơ đồ kiến trúc phân lớp
3.1.2. Luồng dữ liệu trong hệ thống
3.1.3. Các công nghệ sử dụng (Python, PyQt6, NetworkX, NumPy)
3.2. Mô Hình Dữ Liệu (3-4 trang)
3.2.1. Lớp Node và các lớp con (PopulationZone, Shelter, HazardZone)
3.2.2. Lớp Edge với mô hình giao thông động
3.2.3. Lớp EvacuationNetwork (đồ thị NetworkX)
3.2.4. Class diagram
3.3. Triển Khai Thuật Toán GBFS (3-4 trang)
3.3.1. Thiết kế hàm heuristic đa tiêu chí
Trọng số: distance (0.4), risk (0.3), congestion (0.2), capacity (0.1)
3.3.2. Cơ chế tìm đường với priority queue
3.3.3. Chiến lược phân bổ luồng tham lam
3.3.4. Xử lý trường hợp khẩn cấp (emergency mode)
3.3.5. Code snippets minh họa
3.4. Triển Khai Thuật Toán GWO (3-4 trang)
3.4.1. Biểu diễn lời giải (flow matrix)
3.4.2. Hàm fitness đa ràng buộc
Time cost, risk cost, capacity penalty, balance penalty
3.4.3. Cơ chế cập nhật vị trí wolf
3.4.4. Tích hợp với Dijkstra để tính path metrics
3.4.5. Code snippets minh họa
3.5. Giao Diện Người Dùng (2-3 trang)
3.5.1. Thiết kế bản đồ tương tác (MapWidget)
3.5.2. Bảng điều khiển cấu hình (ControlPanel)
3.5.3. Dashboard thời gian thực
3.5.4. Giao diện so sánh thuật toán
CHƯƠNG 4. KẾT QUẢ THỰC NGHIỆM (8-10 trang)
4.1. Môi Trường Thực Nghiệm (1-2 trang)
4.1.1. Cấu hình phần cứng
4.1.2. Dữ liệu TP.HCM (18 quận, 50 điểm trú ẩn, 7.29 triệu dân)
4.1.3. Các kịch bản thử nghiệm
4.2. Các Tình Huống Thực Nghiệm (4-5 trang)
4.2.1. Kịch bản 1: Ngập lụt quy mô nhỏ (1-2 vùng nguy hiểm)
4.2.2. Kịch bản 2: Ngập lụt quy mô trung bình (3-4 vùng)
4.2.3. Kịch bản 3: Thiên tai quy mô lớn (5-6 vùng)
4.2.4. Screenshots và biểu đồ kết quả
4.3. Phân Tích và Đánh Giá (3-4 trang)
4.3.1. So sánh GBFS vs GWO
Thời gian thực thi
Chất lượng lời giải (total cost)
Tỷ lệ bao phủ (coverage rate)
Cân bằng tải (load balancing)
4.3.2. Biểu đồ convergence
4.3.3. Radar chart so sánh đa tiêu chí
4.3.4. Nhận xét và đánh giá
CHƯƠNG 5. KẾT LUẬN (3-4 trang)
5.1. Các Kết Quả Đạt Được (1-2 trang)
5.1.1. Về mặt lý thuyết
5.1.2. Về mặt thực nghiệm
5.1.3. Về mặt ứng dụng
5.2. Những Hạn Chế và Hướng Phát Triển (1-2 trang)
5.2.1. Hạn chế hiện tại
5.2.2. Hướng phát triển trong tương lai
Tích hợp dữ liệu thời gian thực
Áp dụng machine learning
Mở rộng cho các thành phố khác
TÀI LIỆU THAM KHẢO (1-2 trang)
Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey Wolf Optimizer
Cormen, T. H., et al. (2022). Introduction to Algorithms
Các tài liệu về GBFS, mô hình BPR, OpenStreetMap
PHỤ LỤC (3-5 trang)
Phụ lục A: Mã Nguồn Chính
Link GitHub repository
Hướng dẫn cài đặt và sử dụng
Phụ lục B: Phân Công Nhiệm Vụ Chi Tiết
Thành viên Nhiệm vụ Mô tả chi tiết
tadyuh76 Kiến trúc hệ thống Thiết kế cấu trúc dự án, xây dựng các module core
Thuật toán GBFS Triển khai đầy đủ GBFS với heuristic đa tiêu chí
Thuật toán GWO Triển khai GWO với fitness function tùy chỉnh
Tích hợp dữ liệu Xử lý dữ liệu OpenStreetMap, dữ liệu TP.HCM
Leonn2285 Giao diện so sánh Triển khai comparison view với biểu đồ
Dashboard Xây dựng dashboard thời gian thực
Logic simulation Triển khai điều khiển simulation
PeanLutHuynh Tối ưu simulation Tăng tốc độ simulation, color coding
Cấu hình hazard Cho phép người dùng cấu hình vùng nguy hiểm
Sửa lỗi Fix risk mapping, UI bugs
ayo-lole UI/UX Design Thiết kế icons, styling components
Dịch thuật Dịch toàn bộ giao diện sang tiếng Việt
Phụ lục C: Bảng Dữ Liệu TP.HCM
Danh sách 18 quận với dân số, diện tích, flood risk
Danh sách 50 điểm trú ẩn với sức chứa
6 vùng nguy hiểm mặc định
TỔNG KẾT PHÂN BỔ TRANG
Phần Số trang
Mục lục 1
Chương 1: Tổng quan 8-10
Chương 2: Cơ sở lý thuyết 12-15
Chương 3: Thiết kế và cài đặt 12-15
Chương 4: Kết quả thực nghiệm 8-10
Chương 5: Kết luận 3-4
Tài liệu tham khảo 1-2
Phụ lục 3-5
TỔNG ~50 trang
GHI CHÚ QUAN TRỌNG
Không reference GitHub trực tiếp - chỉ đề cập "mã nguồn được lưu trữ trên repository"
Viết về GBFS và GWO - không phải Genetic Algorithm
Dữ liệu thực tế TP.HCM - 18 quận, 7.29 triệu dân, 50 điểm trú ẩn
Screenshots từ ứng dụng - cần chạy demo để chụp
CÁC FILE QUAN TRỌNG CẦN REFERENCE
Algorithms
/src/algorithms/gbfs.py - GBFS implementation (620 lines)
/src/algorithms/gwo.py - GWO implementation (1214 lines)
/src/algorithms/base.py - Base classes (208 lines)
Models
/src/models/node.py - Node types (171 lines)
/src/models/edge.py - Edge with traffic model (264 lines)
/src/models/network.py - Network graph (442 lines)
Data
/src/data/hcm_data.py - HCM real data (531 lines)
UI
/src/ui/map_widget.py - Map visualization (1340 lines)
/src/ui/comparison_view.py - Algorithm comparison (655 lines)
