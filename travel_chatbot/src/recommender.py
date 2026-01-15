import pandas as pd
from src.utils import category_map

def suggest_places(data, info, query=""):
    """
    Lọc danh sách địa điểm gợi ý theo yêu cầu người dùng.
    """

    # 🔧 Xử lý ngân sách an toàn
    raw_budget = info.get("budget", 0)
    try:
        budget = int(float(raw_budget)) if raw_budget not in [None, ""] else 0
    except:
        budget = 0

    category = str(info.get("category", "")).lower()
    filtered = data.copy()

    # ✅ Cột tương ứng trong file CSV
    name_col = "name"
    cat_col = "category_detail"
    type_col = "poi_type"

    # 🔹 Nếu người dùng gõ từ khóa cụ thể (ví dụ "bún", "phở", "cafe", ...)
    if query:
        q = query.lower()
        filtered = filtered[
            filtered[name_col].astype(str).str.lower().str.contains(q, na=False)
            | filtered[cat_col].astype(str).str.lower().str.contains(q, na=False)
        ]

    # 🔹 Nếu người dùng chỉ nói loại hình (ẩm thực, vui chơi,...)
    elif category in category_map:
        # Gộp các từ khóa tiếng Anh cho danh mục đó
        keywords = "|".join(category_map[category])
        filtered = filtered[
            filtered[cat_col].astype(str).str.lower().str.contains(keywords, na=False)
            | filtered[type_col].astype(str).str.lower().str.contains(keywords, na=False)
        ]

    # 🔹 Lọc theo ngân sách (nếu có)
    if "avg_cost" in filtered.columns and budget > 0:
        filtered = filtered[pd.to_numeric(filtered["avg_cost"], errors="coerce") <= budget]

    # 🔹 Sắp xếp theo điểm gợi ý (nếu có)
    if "recommendation_score" in filtered.columns:
        filtered = filtered.sort_values(by="recommendation_score", ascending=False)

    # 🔹 Debug xem kết quả lọc
    print(f"🔍 Category: {category}")
    print(f"🔍 Số dòng sau lọc: {len(filtered)}")

    # 🔁 Nếu vẫn trống, thử lọc lại bằng từ khóa dự phòng tùy theo danh mục
    if filtered.empty:
        fallback = {
            "ẩm thực": "food|restaurant|eat|drink|coffee|beverage",
            "vui chơi": "entertainment|fun|game|park|bar|karaoke|cinema|nightlife|activity",
            "du lịch": "attraction|travel|tour|sightseeing|temple|museum|place",
            "nghỉ dưỡng": "resort|spa|hotel|homestay|stay|relax",
            "mua sắm": "shopping|market|store|mall|boutique"
        }
        alt_keywords = fallback.get(category, "")
        if alt_keywords:
            filtered = data[
                data[cat_col].astype(str).str.lower().str.contains(alt_keywords, na=False)
                | data[type_col].astype(str).str.lower().str.contains(alt_keywords, na=False)
            ]
            print(f"🔁 Dùng từ khóa dự phòng cho '{category}': {len(filtered)} kết quả.")

    # Nếu vẫn rỗng → trả DataFrame trống
    if filtered.empty:
        return pd.DataFrame()

    # ✅ Trả về top 5
    return filtered.head(5)
