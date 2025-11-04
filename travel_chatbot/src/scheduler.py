import random
import pandas as pd
from src.utils import category_map

def build_day_schedule(data, info, query=""):
    """
    Tạo lịch trình 1 ngày (sáng - trưa - chiều - tối)
    dựa trên categories người dùng đã nói và dữ liệu địa điểm.
    """

    if data.empty:
        return ["(Không có dữ liệu để xếp lịch)"]

    # Danh sách các khung giờ cố định
    time_slots = ["Sáng", "Trưa", "Chiều", "Tối"]
    categories = info.get("categories", ["ẩm thực"])
    plan = []

    for i, slot in enumerate(time_slots):
        cat = categories[i % len(categories)].lower()

        # 🔹 Lấy từ khóa tương ứng
        keywords = "|".join(category_map.get(cat, []))
        if not keywords:
            continue

        filtered = data[
            data["category_detail"].astype(str).str.lower().str.contains(keywords, na=False)
            | data["poi_type"].astype(str).str.lower().str.contains(keywords, na=False)
        ]

        # 🔁 Nếu vẫn trống, dùng fallback
        if filtered.empty:
            fallback = {
                "ẩm thực": "food|restaurant|eat|drink|coffee|beverage",
                "vui chơi": "entertainment|fun|game|park|bar|karaoke|cinema|nightlife|activity",
                "du lịch": "attraction|travel|tour|sightseeing|temple|museum|place",
                "nghỉ dưỡng": "resort|spa|hotel|homestay|stay|relax",
                "mua sắm": "shopping|market|store|mall|boutique"
            }
            alt = fallback.get(cat, "")
            filtered = data[
                data["category_detail"].astype(str).str.lower().str.contains(alt, na=False)
                | data["poi_type"].astype(str).str.lower().str.contains(alt, na=False)
            ]

        # Nếu vẫn không có, bỏ qua slot này
        if filtered.empty:
            plan.append(f"🕒 {slot}: (Không tìm thấy địa điểm phù hợp cho {cat})")
            continue

        # 🔹 Chọn ngẫu nhiên 1 địa điểm để thêm vào lịch trình
        choice = filtered.sample(1).iloc[0]

        name = choice.get("name", "Địa điểm")
        typ = choice.get("poi_type", "N/A")
        rating = choice.get("simulated_rating", "?")
        open_t = choice.get("opening_time", "?")
        close_t = choice.get("closing_time", "?")

        plan.append(f"🕒 {slot}: {name} ({typ}) ⭐{rating} ⏰ {open_t}-{close_t}")

    if not plan:
        plan.append("(Không tìm thấy địa điểm phù hợp cho lịch trình)")

    return plan
