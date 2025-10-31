import pandas as pd
from src.utils import category_map

def suggest_places(data, info, query=""):
    category = info.get("category", "").lower()
    budget = int(info.get("budget", 0))

    filtered = data.copy()

    # 🔹 Nếu người dùng gõ từ khóa cụ thể (ví dụ "bún", "phở", "cafe", ...)
    if query:
        q = query.lower()
        filtered = filtered[
            filtered["name"].astype(str).str.lower().str.contains(q, na=False) |
            filtered["category_detail"].astype(str).str.lower().str.contains(q, na=False)
        ]

    # 🔹 Nếu người dùng gõ dạng "ẩm thực", "vui chơi", thì dùng mapping
    elif category in category_map:
        keywords = "|".join(category_map[category])
        filtered = filtered[
            filtered["category_detail"].astype(str).str.lower().str.contains(keywords, na=False)
            | filtered["poi_type"].astype(str).str.lower().str.contains(keywords, na=False)
        ]

    # 🔹 Lọc theo ngân sách
    if budget > 0:
        filtered = filtered[pd.to_numeric(filtered["avg_cost"], errors="coerce") <= budget]

    # 🔹 Sắp xếp theo điểm gợi ý
    filtered = filtered.sort_values(by="recommendation_score", ascending=False)

    return filtered.head(5)
