# Netflix Content Segmentation Dashboard

**Phân cụm nội dung Netflix bằng K-Means và trực quan hóa kết quả trên Streamlit**

Dự án phân tích danh mục phim/chương trình truyền hình trong bộ dữ liệu **Netflix Titles**, trích xuất các đặc trưng của nội dung và sử dụng thuật toán **K-Means** để nhóm các tựa phim có đặc điểm tương đồng. Kết quả được trình bày qua dashboard tiếng Việt, hỗ trợ khám phá cấu trúc danh mục và mô phỏng các nhóm sở thích nội dung.

> **Lưu ý về phạm vi:** Dữ liệu đầu vào là **thông tin tựa phim**, không phải lịch sử xem theo tài khoản người dùng. Vì thế, các “cụm người dùng”, “sở thích người dùng” trong giao diện là **cách đặt tên và diễn giải mô phỏng từ các cụm nội dung**, **không phải** kết quả phân khúc khách hàng Netflix thực tế.

## 1. Mục tiêu

- Khám phá đặc điểm danh mục nội dung Netflix theo loại hình, thể loại, quốc gia, thời lượng và mức phân loại độ tuổi.
- Tiền xử lý dữ liệu và xây dựng đặc trưng phục vụ học máy.
- Phân cụm các tựa phim bằng **K-Means** và đánh giá mức độ phân tách cụm bằng **Silhouette Score**.
- Phân tích đặc trưng nổi bật của từng cụm và trực quan hóa trên **Streamlit**.
- Minh họa cách dùng kết quả phân cụm để hình thành các quy tắc gợi ý nội dung ở mức cơ bản.

## 2. Công nghệ

| Công nghệ | Vai trò |
|---|---|
| Python | Ngôn ngữ chính |
| Pandas, NumPy | Đọc, làm sạch và xử lý dữ liệu |
| Scikit-learn | MinMaxScaler, K-Means, PCA và Silhouette Score |
| Streamlit | Dashboard web tương tác |
| Plotly, Matplotlib, Seaborn | Trực quan hóa và xuất biểu đồ |

Các thư viện được khai báo trong `requirements.txt`.

## 3. Quy trình xử lý

```text
Netflix Titles CSV
       |
       v
Đọc và làm sạch dữ liệu
       |
       v
Tạo đặc trưng nội dung (thể loại, quốc gia, loại phim,
thời lượng, số mùa, phân loại độ tuổi)
       |
       v
Chuẩn hóa biến số bằng MinMaxScaler
       |
       v
K-Means (K = 5 trong main.py)
       |
       +--> Silhouette Score / phân tích chân dung cụm
       |
       v
Dashboard Streamlit / biểu đồ PCA / bảng tổng hợp
```

### Tiền xử lý

- Loại bỏ bản ghi trùng theo `show_id`.
- Điền giá trị thiếu ở một số cột như `country`, `listed_in`, `director`, `rating`.
- Tách chuỗi thể loại và quốc gia thành danh sách.
- Chuyển `duration` thành **số phút** (phim) hoặc **số mùa** (TV Show).
- Quy đổi phân loại độ tuổi thành biến `rating_level`.

### Đặc trưng và phân cụm

- Mã hóa `Movie`/`TV Show` bằng biến nhị phân.
- Tạo biến đánh dấu các thể loại và quốc gia phổ biến.
- Chuẩn hóa `duration_min`, `season_count` và `rating_level` bằng **MinMaxScaler**.
- Chạy `KMeans(n_clusters=5, random_state=42, n_init=10)` trong luồng dashboard hiện tại.
- Tính **Silhouette Score** (lấy mẫu tối đa 2.000 bản ghi nếu dữ liệu lớn), tổng hợp quy mô, thể loại, quốc gia và thời lượng trung bình của từng cụm.
- Sử dụng **PCA 2D** để minh họa phân bố các cụm.

**Lưu ý:** Các nhãn như “Action Lovers” hoặc “Drama Seekers” được định nghĩa sẵn trong mã; không nên coi chúng là tên cụm được mô hình tự suy ra từ hành vi người dùng.

## 4. Các trang dashboard

| Trang | Nội dung |
|---|---|
| **Tổng quan** | Chỉ số tóm tắt, phân bố cụm, heatmap thể loại, PCA, bảng chân dung cụm |
| **Phân tích người dùng** | Góc nhìn mô phỏng theo các cụm nội dung và biểu đồ PCA |
| **Phân tích nội dung** | Phân bố Movie/TV Show, rating, thời lượng, quốc gia và báo cáo dữ liệu thiếu |
| **Phân loại sở thích** | Heatmap thể loại theo cụm và thể loại nổi bật |
| **Cụm người dùng** | Tỷ trọng cụm và các chỉ số đánh giá được hiển thị |
| **Chân dung cụm** | Bảng thông tin đặc trưng từng cụm |
| **Dự đoán & Gợi ý** | Quy tắc gợi ý minh họa theo cụm và danh sách tựa phim mẫu |
| **Báo cáo** | Tổng kết số bản ghi, số cụm và bảng chân dung cụm |

Phần **Dự đoán & Gợi ý** hiện là minh họa dựa trên các cụm và nội dung mẫu; **chưa** có mô hình khuyến nghị cá nhân hóa theo lịch sử xem của từng tài khoản.

## 5. Cài đặt và chạy dự án

### Yêu cầu

- Python **3.10+** (theo hướng dẫn của dự án gốc).
- File dữ liệu đầu vào tại `data/raw/netflix_titles.csv`.

> **Quan trọng:** Bản ZIP mã nguồn được dùng để chuẩn bị README này **không chứa file CSV** nói trên. Cần bổ sung bộ dữ liệu *Netflix Titles* với các cột tương ứng trước khi chạy. Không phải chỉ cài thư viện là có thể chạy dashboard ngay.

### Bước 1. Tạo môi trường ảo (khuyên dùng)

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### Bước 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 3. Chuẩn bị dữ liệu

Tạo thư mục:

```text
data/
└── raw/
    └── netflix_titles.csv
```

Dữ liệu cần có ít nhất các cột phù hợp với mã nguồn: `show_id`, `type`, `title`, `country`, `director`, `release_year`, `rating`, `duration`, `listed_in`.

### Bước 4. Chạy dashboard

```bash
streamlit run main.py
```

Mở địa chỉ được Streamlit hiển thị trong Terminal (thông thường là `http://localhost:8501`).

## 6. Cấu trúc thư mục

```text
.
├── main.py                     # Dashboard Streamlit, điều hướng và luồng phân tích
├── export_charts.py            # Script xuất biểu đồ phục vụ báo cáo/thuyết trình
├── requirements.txt            # Thư viện Python
├── README.md                   # Hướng dẫn dự án
├── DiemNoiBat_DuAn.md          # Ghi chú mô tả dự án
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Đọc, làm sạch, biến đổi trường dữ liệu
│   ├── feature_engineering.py  # Tạo ma trận đặc trưng
│   ├── clustering.py           # K-Means, Silhouette, tổng hợp cụm
│   ├── eda.py                  # Phân tích khám phá dữ liệu
│   └── visualization.py        # Biểu đồ dashboard
└── data/
    └── raw/
        └── netflix_titles.csv  # CẦN BỔ SUNG: không có trong ZIP được xem xét
```

`export_charts.py` là tiện ích riêng để xuất biểu đồ; khả năng xuất PNG có thể cần thêm phụ thuộc tương ứng của Plotly (ví dụ `kaleido`), chưa được liệt kê trong `requirements.txt` hiện tại.

## 7. Kiểm tra cơ bản

Kiểm tra cú pháp Python:

```bash
python -m compileall -q main.py src export_charts.py
```

Chạy thử giao diện:

```bash
streamlit run main.py
```

Kiểm tra lần lượt: dữ liệu được tải, biểu đồ hiển thị, năm cụm được tạo và các trang sidebar hoạt động. **Chưa có bộ kiểm thử tự động đi kèm bản ZIP mã nguồn được xem xét.**

## 8. Giới hạn hiện tại

1. **Không có dữ liệu lịch sử xem theo người dùng:** chưa thể suy ra chân dung và phân khúc khách hàng Netflix thực tế.
2. **K cố định bằng 5:** giao diện hiện không có điều khiển chọn và tối ưu K; cần thử nhiều giá trị K với đánh giá phù hợp.
3. **Một số chỉ số mang tính mô phỏng:** `total_views` được suy ra bằng công thức từ số bản ghi/thể loại; `avg_watch_hours` tính từ thời lượng nội dung, **không phải** thời gian xem của người dùng. `db_index` đang dùng công thức tham chiếu, **không phải phép tính Davies–Bouldin chuẩn**.
4. **Gợi ý nội dung chưa cá nhân hóa:** chưa dùng tương tác hoặc phản hồi của từng người dùng.
5. **Tài liệu và mã nguồn chưa hoàn toàn đồng nhất:** một số ý tưởng trong `DiemNoiBat_DuAn.md` (như thay K trực tiếp hoặc tự động xuất Word/PowerPoint) chưa được xác nhận là chức năng có mặt trong `main.py` và các file của ZIP này.
6. **Chưa xác minh chạy end-to-end:** cần bổ sung CSV và chạy thực tế với dữ liệu, sau đó ghi nhận chỉ số và ảnh dashboard để báo cáo kết quả.

## 9. Hướng phát triển

- Thu thập dữ liệu hành vi xem **được phép sử dụng**, có cơ chế bảo vệ quyền riêng tư, để chuyển từ **phân cụm nội dung** sang **phân khúc người dùng thực tế**.
- Thử và so sánh nhiều giá trị K; sử dụng Silhouette Score và Davies–Bouldin Index chuẩn để đánh giá.
- Đặt tên cụm **sau khi** phân tích đặc trưng thực tế, thay vì gắn tên cố định theo số thứ tự.
- Xây dựng hệ thống khuyến nghị cá nhân hóa, có đánh giá chất lượng gợi ý.
- Điều chỉnh các chỉ số trên dashboard để phân biệt rõ **số liệu quan sát**, **số liệu suy ra** và **minh họa**.
- Hoàn thiện kiểm thử tự động, bổ sung hướng dẫn dữ liệu và ảnh chụp kết quả chạy thực tế.

## 10. Kết luận

Dự án minh họa quy trình ứng dụng học máy cho dữ liệu nội dung: **xử lý dữ liệu → tạo đặc trưng → phân cụm → đánh giá → trực quan hóa**. Giá trị chính là khả năng phân tích và diễn giải cấu trúc danh mục nội dung, đồng thời chỉ ra các điều kiện dữ liệu cần thiết nếu muốn phát triển thành hệ thống phân khúc người dùng và gợi ý cá nhân hóa thực tế.
