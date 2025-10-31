from typing import List, Set, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
import math

# =============================================================================
# HÀM HỖ TRỢ TÍNH TOÁN
# =============================================================================

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Tính khoảng cách đường chim bay giữa 2 điểm trên Trái Đất (đơn vị: km).
    Sử dụng công thức Haversine - công thức toán học tính khoảng cách trên hình cầu.
    
    Tham số:
        lat1, lon1: Vĩ độ và kinh độ điểm 1
        lat2, lon2: Vĩ độ và kinh độ điểm 2
    
    Trả về: Khoảng cách tính bằng km
    """
    R = 6371.0  # Bán kính Trái Đất tính bằng km
    
    # Chuyển đổi độ sang radian (đơn vị toán học)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    # Áp dụng công thức Haversine
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def estimate_travel_time_minutes_km(distance_km: float, avg_speed_kmh: float = 25.0) -> float:
    """
    Ước tính thời gian di chuyển dựa trên khoảng cách và tốc độ trung bình.
    
    Công thức: thời gian (phút) = khoảng cách (km) / tốc độ (km/h) * 60
    
    Tham số:
        distance_km: Khoảng cách cần đi (km)
        avg_speed_kmh: Tốc độ trung bình (km/h), mặc định 25 km/h (tốc độ di chuyển trong thành phố)
    
    Trả về: Thời gian di chuyển tính bằng phút
    """
    if distance_km is None:
        return float('inf')  # Trả về vô cực nếu không có khoảng cách
    return (distance_km / avg_speed_kmh) * 60.0


# =============================================================================
# CLASS ĐIỂM THAM QUAN (POINT OF INTEREST - POI)
# =============================================================================

@dataclass
class PointOfInterest:
    """
    Lớp đại diện cho một điểm tham quan/địa điểm du lịch.
    Chứa mọi thông tin về địa điểm: vị trí, loại hình, giá cả, đánh giá...
    """
    
    # --- THÔNG TIN CƠ BẢN ---
    poi_id: int                            # ID duy nhất của địa điểm
    name: str                              # Tên địa điểm (ví dụ: "Bitexco Tower")
    latitude: float                        # Vĩ độ (tọa độ)
    longitude: float                       # Kinh độ (tọa độ)
    poi_type: str                          # Loại địa điểm tổng quát (VD: Food_Dining, Culture_History)
    
    # --- THÔNG TIN CHI TIẾT ---
    category_detail: Optional[str] = None  # Loại chi tiết hơn (VD: cafe, rooftop, museum)
    vibe: Optional[str] = None             # Phong cách/không khí (7 loại: Chill, Party, Romantic, Local, Culture, Family, Luxury)
    
    # --- THÔNG TIN GIÁ CẢ & ĐÁNH GIÁ ---
    avg_cost: float = 0.0                  # Chi phí trung bình (đơn vị: VNĐ)
    simulated_rating: float = 0.0          # Đánh giá mô phỏng (thang điểm 0-5)
    
    # --- THÔNG TIN THỜI GIAN ---
    dwell_time_minutes: int = 60           # Thời gian dự kiến ở lại địa điểm (phút)
    opening_time: str = "08:00"            # Giờ mở cửa (định dạng "HH:MM")
    closing_time: str = "21:00"            # Giờ đóng cửa (định dạng "HH:MM")
    time_block_suitability: Set[str] = field(default_factory=set)  # Các khung giờ phù hợp (VD: {"morning","afternoon","evening","night"})
    
    # --- THÔNG TIN PHỔ BIẾN & GỢI Ý ---
    popularity_score: float = 0.0          # Điểm phổ biến (0 đến 1)
    is_central: bool = False               # Có phải địa điểm trung tâm không?
    recommendation_score: float = 0.0      # Điểm gợi ý động từ hệ thống (0 đến 1)
    
    # --- BỘ NHỚ ĐỆM THỜI GIAN DI CHUYỂN ---
    travel_time_cache: Dict[int, float] = field(default_factory=dict)  # Lưu thời gian đến các địa điểm khác (poi_id -> phút)

    def __str__(self) -> str:
        """Hiển thị thông tin địa điểm dạng text dễ đọc"""
        return (f"POI {self.poi_id}: {self.name} [{self.category_detail or self.poi_type}] "
                f"Vibe={self.vibe} Cost={int(self.avg_cost):,} VNĐ Rating={self.simulated_rating:.1f}")

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đối tượng thành dictionary (để lưu JSON hoặc database)"""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PointOfInterest":
        """Tạo đối tượng PointOfInterest từ dictionary"""
        return cls(
            poi_id=data.get("poi_id", -1),
            name=data.get("name", ""),
            latitude=float(data.get("latitude", 0.0)),
            longitude=float(data.get("longitude", 0.0)),
            poi_type=data.get("poi_type", ""),
            category_detail=data.get("category_detail"),
            vibe=data.get("vibe"),
            avg_cost=float(data.get("avg_cost", 0.0)),
            simulated_rating=float(data.get("simulated_rating", 0.0)),
            dwell_time_minutes=int(data.get("dwell_time_minutes", 60)),
            popularity_score=float(data.get("popularity_score", 0.0)),
            opening_time=data.get("opening_time", "08:00"),
            closing_time=data.get("closing_time", "21:00"),
            time_block_suitability=set(data.get("time_block_suitability", [])),
            is_central=bool(data.get("is_central", False)),
            recommendation_score=float(data.get("recommendation_score", 0.0)),
            travel_time_cache=dict(data.get("travel_time_cache", {}))
        )

    # ==========================================================================
    # CÁC HÀM HỖ TRỢ TÍNH TOÁN DI CHUYỂN
    # ==========================================================================
    
    def distance_to(self, other: "PointOfInterest") -> float:
        """
        Tính khoảng cách từ địa điểm này đến địa điểm khác.
        
        Trả về: Khoảng cách tính bằng km
        """
        return haversine_distance_km(self.latitude, self.longitude, other.latitude, other.longitude)

    def estimate_travel_time_to(self, other: "PointOfInterest", avg_speed_kmh: float = 25.0) -> float:
        """
        Ước tính thời gian di chuyển từ địa điểm này đến địa điểm khác.
        Sử dụng bộ nhớ đệm (cache) để tránh tính toán lại nhiều lần.
        
        Tham số:
            other: Địa điểm đích
            avg_speed_kmh: Tốc độ trung bình (km/h)
        
        Trả về: Thời gian di chuyển tính bằng phút
        """
        # Kiểm tra xem đã tính toán trước đó chưa (cache)
        if other.poi_id in self.travel_time_cache:
            return self.travel_time_cache[other.poi_id]
        
        # Chưa có trong cache thì tính mới
        dist = self.distance_to(other)
        minutes = estimate_travel_time_minutes_km(dist, avg_speed_kmh)
        
        # Lưu vào cache để lần sau dùng lại
        self.travel_time_cache[other.poi_id] = minutes
        return minutes

    def is_open_during(self, time_hhmm: str) -> bool:
        """
        Kiểm tra địa điểm có mở cửa vào thời điểm này không?
        
        Tham số:
            time_hhmm: Thời gian cần kiểm tra (định dạng "HH:MM", ví dụ "14:30")
        
        Trả về: True nếu đang mở cửa, False nếu đóng cửa
        
        Lưu ý: Hàm này chưa xử lý trường hợp mở cửa qua đêm (VD: mở từ 22:00 đến 02:00 hôm sau)
        """
        def to_minutes(t: str) -> int:
            """Chuyển "HH:MM" thành số phút kể từ 00:00"""
            h, m = map(int, t.split(":"))
            return h * 60 + m
        
        # Chuyển tất cả thời gian sang đơn vị phút để so sánh dễ hơn
        t = to_minutes(time_hhmm)
        open_t = to_minutes(self.opening_time)
        close_t = to_minutes(self.closing_time)
        
        # Xử lý trường hợp mở cửa qua đêm (VD: 22:00 - 02:00)
        if close_t <= open_t:
            # Nếu giờ đóng cửa nhỏ hơn giờ mở cửa = mở qua đêm
            return t >= open_t or t <= close_t
        
        # Trường hợp bình thường (VD: 08:00 - 21:00)
        return open_t <= t <= close_t


# =============================================================================
# CLASS LỘ TRÌNH THAM QUAN (ITINERARY)
# =============================================================================

@dataclass
class Itinerary:
    """
    Lớp đại diện cho một lộ trình tham quan hoàn chỉnh.
    Chứa danh sách các địa điểm theo thứ tự và các thông tin tổng hợp.
    """
    
    # --- DANH SÁCH ĐỊA ĐIỂM ---
    poi_sequence: List[PointOfInterest] = field(default_factory=list)  # Danh sách địa điểm theo thứ tự tham quan
    
    # --- THÔNG TIN TỔNG HỢP ---
    total_cost: float = 0.0                      # Tổng chi phí của cả lộ trình (VNĐ)
    total_duration_minutes: float = 0.0          # Tổng thời gian (bao gồm cả thời gian ở lại + di chuyển, đơn vị: phút)
    total_satisfaction_score: float = 0.0        # Điểm trải nghiệm tổng hợp (điểm hài lòng)
    
    # --- CHI TIẾT DI CHUYỂN ---
    details: List[Dict[str, Any]] = field(default_factory=list)  # Danh sách chi tiết từng chặng di chuyển

    def compute_summary(self, avg_speed_kmh: float = 25.0, return_to_start: bool = False) -> None:
        """
        Tính toán tất cả thông tin tổng hợp cho lộ trình:
        - Tổng chi phí
        - Tổng thời gian (ở lại + di chuyển)
        - Điểm trải nghiệm
        - Chi tiết từng chặng di chuyển
        
        Tham số:
            avg_speed_kmh: Tốc độ di chuyển trung bình (km/h)
            return_to_start: Có cần quay về điểm xuất phát không? (cho lộ trình khứ hồi)
        """
        # Reset tất cả về 0
        self.total_cost = 0.0
        self.total_duration_minutes = 0.0
        self.total_satisfaction_score = 0.0
        self.details = []

        # Nếu không có địa điểm nào thì dừng
        if not self.poi_sequence:
            return

        # --- BƯỚC 1: Tính chi phí, thời gian ở lại và điểm trải nghiệm ---
        for poi in self.poi_sequence:
            # Cộng dồn chi phí
            self.total_cost += poi.avg_cost
            
            # Cộng thời gian ở lại địa điểm
            self.total_duration_minutes += poi.dwell_time_minutes
            
            # Tính điểm trải nghiệm: kết hợp điểm gợi ý (70%) và độ phổ biến (30%), nhân 10 để ra thang điểm dễ nhìn
            self.total_satisfaction_score += (poi.recommendation_score * 0.7 + poi.popularity_score * 0.3) * 10.0

        # --- BƯỚC 2: Tính thời gian di chuyển giữa các địa điểm liên tiếp ---
        for i in range(len(self.poi_sequence) - 1):
            a = self.poi_sequence[i]      # Địa điểm hiện tại
            b = self.poi_sequence[i + 1]  # Địa điểm tiếp theo
            
            # Tính thời gian di chuyển
            travel_min = a.estimate_travel_time_to(b, avg_speed_kmh)
            self.total_duration_minutes += travel_min
            
            # Lưu chi tiết chặng di chuyển
            self.details.append({
                "from": a.name,
                "to": b.name,
                "distance_km": round(a.distance_to(b), 3),
                "travel_minutes": round(travel_min, 1)
            })

        # --- BƯỚC 3: Tính thời gian quay về điểm xuất phát (nếu cần) ---
        if return_to_start and len(self.poi_sequence) > 1:
            last = self.poi_sequence[-1]   # Địa điểm cuối
            first = self.poi_sequence[0]   # Địa điểm đầu
            
            travel_min = last.estimate_travel_time_to(first, avg_speed_kmh)
            self.total_duration_minutes += travel_min
            
            # Lưu chi tiết chặng về
            self.details.append({
                "from": last.name,
                "to": first.name,
                "distance_km": round(last.distance_to(first), 3),
                "travel_minutes": round(travel_min, 1)
            })

        # --- BƯỚC 4: Chuẩn hóa điểm trải nghiệm (lấy trung bình) ---
        self.total_satisfaction_score = (self.total_satisfaction_score / max(1, len(self.poi_sequence)))

    def print_summary(self) -> None:
        """
        In ra màn hình tổng quan lộ trình theo định dạng dễ đọc.
        Bao gồm: danh sách địa điểm, thời gian, chi phí, điểm trải nghiệm, chi tiết di chuyển.
        """
        print("--- LỘ TRÌNH TỐI ƯU ---")
        
        # In từng địa điểm
        for i, poi in enumerate(self.poi_sequence):
            print(f"{i+1}. {poi.name} ({poi.category_detail or poi.poi_type}) | Vibe={poi.vibe} "
                  f"| Cost={int(poi.avg_cost):,} VNĐ | Dwell={poi.dwell_time_minutes}min | RecScore={poi.recommendation_score:.2f}")
        
        # Chuyển tổng thời gian từ phút sang giờ và phút
        hours = int(self.total_duration_minutes // 60)
        minutes = int(self.total_duration_minutes % 60)
        
        # In thông tin tổng hợp
        print(f"\nTổng thời gian: {hours} giờ {minutes} phút ({int(self.total_duration_minutes)} phút)")
        print(f"Tổng chi phí ước tính: {int(self.total_cost):,} VNĐ")
        print(f"Chỉ số trải nghiệm (satisfaction): {self.total_satisfaction_score:.2f}")
        
        # In chi tiết từng chặng di chuyển
        if self.details:
            print("\n--- Chi tiết di chuyển ---")
            for seg in self.details:
                print(f"{seg['from']} -> {seg['to']}: {seg['distance_km']} km, {seg['travel_minutes']} phút")

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển lộ trình thành dictionary (để lưu JSON hoặc gửi API).
        """
        return {
            "poi_ids": [p.poi_id for p in self.poi_sequence],
            "total_cost": self.total_cost,
            "total_duration_minutes": self.total_duration_minutes,
            "total_satisfaction_score": self.total_satisfaction_score,
            "details": self.details
        }

    @classmethod
    def from_poi_list(cls, poi_list: List[PointOfInterest]) -> "Itinerary":
        """
        Tạo lộ trình từ danh sách địa điểm và tự động tính toán thông tin tổng hợp.
        
        Tham số:
            poi_list: Danh sách các địa điểm theo thứ tự tham quan
        
        Trả về: Đối tượng Itinerary đã được tính toán đầy đủ
        """
        it = cls(poi_sequence=poi_list.copy())
        it.compute_summary()
        return it
    
# =============================================================================
# CLASS SỞ THÍCH NGƯỜI DÙNG (USER PREFERENCE)
# =============================================================================

@dataclass
class UserPreference:
    """
    Lớp đại diện cho sở thích và bối cảnh của người dùng.
    Dùng để lọc hoặc xếp hạng các địa điểm gợi ý phù hợp.
    """

    # --- THÔNG TIN NGƯỜI DÙNG ---
    name: str = "Guest"                           # Tên người dùng
    current_latitude: float = 10.7769             # Vĩ độ hiện tại (mặc định: TP.HCM)
    current_longitude: float = 106.7009           # Kinh độ hiện tại
    budget_vnd: float = 500_000                   # Ngân sách dự kiến (VNĐ)
    available_time_minutes: int = 240             # Thời gian rảnh (phút)
    preferred_vibes: Set[str] = field(default_factory=set)  # Các vibe yêu thích (VD: {"Chill", "Local"})
    disliked_vibes: Set[str] = field(default_factory=set)   # Các vibe không thích

    preferred_poi_types: Set[str] = field(default_factory=set)  # Loại địa điểm yêu thích (VD: {"Food_Dining", "Culture_History"})
    disliked_poi_types: Set[str] = field(default_factory=set)   # Loại địa điểm không thích

    preferred_time_blocks: Set[str] = field(default_factory=lambda: {"morning", "afternoon", "evening"})  
    # Thời gian rảnh trong ngày

    # --- TÙY CHỌN KHÁC ---
    require_central_area: bool = False            # Có muốn ở khu trung tâm không?
    max_distance_km: float = 10.0                 # Khoảng cách tối đa từ vị trí hiện tại (km)
    min_rating: float = 3.5                       # Đánh giá tối thiểu của địa điểm
    need_variety: bool = True                     # Có muốn đa dạng loại địa điểm không?

    def __str__(self) -> str:
        return (f"UserPreference(name={self.name}, budget={self.budget_vnd:,}₫, "
                f"vibes={list(self.preferred_vibes)}, types={list(self.preferred_poi_types)}, "
                f"time_blocks={list(self.preferred_time_blocks)})")
