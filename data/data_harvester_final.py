import requests
import pandas as pd
import random
import time
from typing import List, Dict, Any, Tuple

# --- 1. THIẾT LẬP THAM SỐ TRUY VẤN VÀ KHU VỰC CỐT LÕI ---

# Địa chỉ API của OpenStreetMap
OVERPASS_URL = "http://overpass-api.de/api/interpreter"

# Khu vực Mục tiêu: Ví dụ TP.HCM (Tọa độ bao phủ khu vực trung tâm rộng)
# Định dạng Bounding Box (Lat_Min, Lon_Min, Lat_Max, Lon_MAX) -> [Nam, Tây, Bắc, Đông]
# Phạm vi này đảm bảo lấy được hàng trăm điểm đến.
LAT_MIN, LON_MIN, LAT_MAX, LON_MAX = 10.75, 106.65, 10.82, 106.75 

# Danh sách MỞ RỘNG các loại hình POI đa dạng (OSM Tags)
# Đây là danh sách đầy đủ để hệ thống trở nên toàn diện.
POI_TAGS: Dict[str, List[str]] = {
    # Quán ăn, Nhà hàng, Quán cafe
    'Food_Dining': [
        'amenity=restaurant', 'amenity=cafe', 'amenity=food_court', 'amenity=fast_food', 'amenity=pub'
    ],
    # Trung tâm thương mại, Chợ, Cửa hàng
    'Shopping_Commerce': [
        'shop=mall', 'shop=department_store', 'shop=supermarket', 'amenity=marketplace', 'shop=clothes', 'shop=electronics'
    ],
    # Bảo tàng, Di tích lịch sử, Công trình văn hóa
    'Culture_History': [
        'tourism=museum', 'historic=monument', 'tourism=gallery', 'tourism=zoo', 'historic=archaeological_site'
    ],
    # Công viên, Khu vui chơi, Rạp chiếu phim, Vườn
    'Entertainment_Leisure': [
        'leisure=park', 'amenity=cinema', 'amenity=bar', 'leisure=garden', 'leisure=playground', 'leisure=sports_centre', 'tourism=theme_park'
    ],
    # Địa điểm tham quan, Điểm ngắm cảnh
    'Viewpoint_Attraction': [
        'tourism=attraction', 'tourism=viewpoint', 'tourism=artwork'
    ]
}

# --- 2. HÀM TẠO VÀ GỬI TRUY VẤN OVERPASS ---

def build_overpass_query(bbox_coords: Tuple[float, float, float, float], poi_tags: Dict[str, List[str]]) -> str:
    """Xây dựng truy vấn Overpass để lấy nhiều loại POI trong Bounding Box."""
    
    # Định dạng tọa độ Bounding Box thành chuỗi
    bbox_str = f"{bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]}"
    
    # Bắt đầu truy vấn
    query = f"[out:json][timeout:60];\n" # [timeout:60] cho phép API chạy tối đa 60 giây
    query += f"(\n"
    
    # Vòng lặp để thêm tất cả các loại POI từ POI_TAGS vào truy vấn
    for type, tags in poi_tags.items():
        for tag in tags:
            # Truy vấn cả node (điểm đơn lẻ) và way (khu vực rộng như công viên, TTTM)
            query += f'  node[{tag}]({bbox_str});\n'
            query += f'  way[{tag}]({bbox_str});\n'
    
    query += ");\n"
    query += "out center;" # out center đảm bảo có tọa độ trung tâm cho tất cả các phần tử
    
    
    # Cấu trúc tổng quát của 1 truy vấn Overpass:
    #     [out:json][timeout:60];
    # (
    #     node[<tag>=<value>](<lat1>,<lon1>,<lat2>,<lon2>);
    #     way[<tag>=<value>](<lat1>,<lon1>,<lat2>,<lon2>);
    #     relation[<tag>=<value>](<lat1>,<lon1>,<lat2>,<lon2>);
    # );
    # out center;
    
    return query

def fetch_data_from_overpass(query: str) -> List[Dict[str, Any]]:
    """Gửi truy vấn và xử lý phản hồi từ Overpass API."""
    print(f"Đang gửi truy vấn đến Overpass API...")
    
    try:
        response = requests.post(OVERPASS_URL, data=query)
        response.raise_for_status() # Báo lỗi nếu kết nối thất bại
        
        data = response.json()
        print(f"Truy vấn thành công. Nhận được {len(data['elements'])} phần tử.")
        return data['elements']
    except requests.exceptions.RequestException as e:
        print(f"LỖI KẾT NỐI hoặc TRUY VẤN: {e}")
        return []

# --- 3. HÀM XỬ LÝ VÀ MÔ HÌNH HÓA FEATURES ---

def process_and_simulate_data(elements: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Xử lý JSON, trích xuất thông tin thực tế (Tên, Tọa độ) và Mô hình hóa (Simulate) các Features ML.
    """
    data_list = []
    
    for element in elements:
        # Lấy tọa độ (lat/lon) từ dữ liệu
        lat = element.get('lat')
        lon = element.get('lon')
        
        # Bỏ qua các điểm không có tọa độ hoặc tên
        if lat is None or lon is None or 'tags' not in element:
            continue
            
        tags = element['tags']
        poi_name = tags.get('name', tags.get('name:en', f"Unnamed POI {element['id']}"))
        
        # 1. Trích xuất Loại POI (POI_Type)
        poi_type = 'Other'
        for type, tag_list in POI_TAGS.items():
            if any(tag.split('=')[0] in tags and tags[tag.split('=')[0]] == tag.split('=')[1] for tag in tag_list):
                poi_type = type
                break
        
        # 2. MÔ HÌNH HÓA Features ML (Tạo Features thông minh cho Task 3 & 4)
        
        # Mô hình hóa Rating_Simulated (Dùng logic: điểm văn hóa, mua sắm thường cao hơn)
        if poi_type in ['Culture_History', 'Shopping_Commerce', 'Viewpoint_Attraction']:
            rating_score = round(random.uniform(4.0, 4.9), 1)
        else:
            rating_score = round(random.uniform(3.5, 4.5), 1)
            
        # Mô hình hóa Dwell Time (Thời gian tham quan/ở lại) - Cần cho Task 6
        if 'Museum' in poi_type or 'Park' in poi_type or 'Mall' in poi_type:
            dwell_time = random.randint(90, 180)  # 1.5 - 3 giờ
        elif 'Restaurant' in poi_type or 'Cafe' in poi_type:
            dwell_time = random.randint(45, 90)   # 45 - 90 phút
        else:
            dwell_time = random.randint(30, 60)
            
        # Mô hình hóa Vị trí Trung tâm (is_Central) - Cần cho Task 5 (Mô hình hóa Chi phí/Tắc đường)
        is_central = (10.76 <= lat <= 10.78) and (106.69 <= lon <= 106.71) # Giả định khu vực Quận 1 (TPHCM)
        
        data_list.append({
            'Name': poi_name,
            'Lat': lat,
            'Lon': lon,
            'POI_Type': poi_type,
            'Rating_Simulated': rating_score,
            'Dwell_Time_Minutes': dwell_time,
            'is_Central': is_central
        })

    return pd.DataFrame(data_list)

# --- 4. THỰC THI CHÍNH (MAIN EXECUTION) ---

def main():
    bbox = (LAT_MIN, LON_MIN, LAT_MAX, LON_MAX)
    query = build_overpass_query(bbox, POI_TAGS)
    
    # 1. Thu thập dữ liệu thực tế từ API
    elements = fetch_data_from_overpass(query)
    
    if not elements:
        print("Quá trình thu thập dữ liệu thất bại hoặc không tìm thấy POI nào.")
        return

    # 2. Xử lý và Mô hình hóa (Simulation)
    df_poi = process_and_simulate_data(elements)

    # 3. Làm sạch dữ liệu và thêm ID
    df_poi.drop_duplicates(subset=['Name', 'Lat', 'Lon'], keep='first', inplace=True)
    df_poi.reset_index(drop=True, inplace=True)
    df_poi['POI_ID'] = df_poi.index
    
    print("-" * 70)
    print(f"✅ THU THẬP VÀ MÔ HÌNH HÓA HOÀN TẤT. Tổng số POI: {len(df_poi)}")
    print("Dữ liệu đã được chuẩn bị cho các thuật toán Heuristics và ML.")
    print("Kiểm tra 5 dòng dữ liệu đầu tiên:")
    print(df_poi[['POI_ID', 'Name', 'POI_Type', 'Lat', 'Rating_Simulated', 'Dwell_Time_Minutes']].head())
    print("-" * 70)
    
    # Lưu ra file CSV cho các Task khác sử dụng
    output_filename = "travel_poi_data_final.csv"
    # --- SAU KHI KHẮC PHỤC (Thêm tham số encoding) ---
    def main():
    # ... (các bước xử lý và tạo df_poi) ...
    
    # Lưu ra file CSV, BẮT BUỘC sử dụng encoding='utf-8'
        output_filename = "travel_poi_data_final.csv"
    df_poi.to_csv(output_filename, index=False, encoding='utf-8') 
    print(f"Dữ liệu đã được lưu vào file '{output_filename}'")

if __name__ == "__main__":
    main()