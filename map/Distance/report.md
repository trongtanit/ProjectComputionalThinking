# BÁO CÁO

## I. Giới thiệu
- Tính khoảng cách giữa hai điểm
- Hiển thị tuyến đường di chuyển
- Hiển thị bản đồ trực quan bằng LeafletJS
- Hỗ trợ tính từ vị trí hiện tại của người dùng

Ứng dụng sử dụng dữ liệu bản đồ từ OpenStreetMap, dịch vụ geocoding Nominatim và dịch vụ routing OSRM.

---

## II. Công nghệ sử dụng
- **Frontend**: HTML, CSS, JavaScript, LeafletJS
- **Backend**: Python (Flask)
- **API & dịch vụ**:
  - Nominatim (Geocoding)
  - OSRM (Routing)
  - Haversine (tính khoảng cách đường thẳng)
- **Khác**:
  - REST API
  - Blueprint Flask

---

## III. Kiến trúc hệ thống
### 1. Frontend
Gồm các file:
- `index.html`: giao diện người dùng
- `map_init.js`: khởi tạo bản đồ Leaflet
- `distance.js`: xử lý tính khoảng cách
- `routing.js`: xử lý vẽ đường đi
- `ui_events.js`: kết nối các nút UI với hàm xử lý

### 2. Backend
Cấu trúc thư mục:
```
project/
 ├── app.py
 ├── routes/
 │    ├── geocode.py
 │    ├── distance.py
 │    └── route.py
 ├── services/
 │    ├── geocoding_service.py
 │    └── routing_service.py
 └── templates/index.html
```

---

## IV. Chức năng chính 
(Hiện tại chỉ hoạt động tốt với tọa độ lat,lon; cần phát tiển thêm)

### 1. Tính khoảng cách A ↔ B
- Nhập địa chỉ hoặc tọa độ
- Geocode địa chỉ sang lat/lon
- Tính khoảng cách bằng Haversine
- Hiển thị đường nối giữa A - B trên bản đồ
- Auto zoom để vừa tuyến đường
- Hiển thị nhãn khoảng cách giữa hai điểm

### 2. Tính khoảng cách từ vị trí hiện tại → B
- Sử dụng `navigator.geolocation`
- Tự động lấy lat/lon hiện tại


### 3. Tìm đường đi A → B 

- Gọi OSRM server
- Nhận dữ liệu tuyến đường (GeoJSON)
- Vẽ polyline màu đỏ
- Hiển thị độ dài & thời gian dự kiến


---

## V. Cách triển khai
### 1. Chạy backend
```
python app.py
```
Mặc định chạy tại: http://127.0.0.1:5000

### 2. Chạy OSRM server (nếu dùng local OSRM)
```
osrm-extract.exe -p profiles/car.lua vietnam-latest.osm.pbf
osrm-partition.exe vietnam-latest.osrm
osrm-customize.exe vietnam-latest.osrm
osrm-routed.exe --algorithm=MLD vietnam-latest.osrm
```

---

## VI. Kết quả
Hệ thống hoạt động đầy đủ các chức năng:
- Tính được khoảng cách giữa hai điểm
- Tìm đường đi trực quan, chính xác
- Hoạt động ổn định cả trên Windows
---

