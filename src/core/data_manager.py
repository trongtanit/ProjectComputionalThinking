import pandas as pd
import os

class DataManager:
    def __init__(self, csv_path="../data/travel_poi_data_ranked.csv"):
        self.csv_path = csv_path
        self.df = None
        self.load_data()

    def load_data(self):
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ Loaded dataset: {self.csv_path} ({len(self.df)} rows)")
        except Exception as e:
            print(f"❌ Failed to load data: {e}")

    def find_places(self, food=None, vibe=None, area=None, budget=None, limit=5):
        """Tìm địa điểm thật trong dữ liệu dựa trên sở thích."""
        df = self.df.copy()

        # 1️⃣ Lọc theo món ăn
        if food:
            df = df[df["category_detail"].astype(str).str.contains(food, case=False, na=False)]

        # 2️⃣ Lọc theo vibe
        if vibe and "vibe" in df.columns:
            df = df[df["vibe"].astype(str).str.contains(vibe, case=False, na=False)]

        # 3️⃣ Lọc theo khu vực
        if area and "name" in df.columns:
            df = df[df["name"].astype(str).str.contains(area, case=False, na=False)]

        # 4️⃣ Lọc theo ngân sách
        if budget and "avg_cost" in df.columns:
            df = df[df["avg_cost"] <= budget * 1.2]

        # 5️⃣ Sắp xếp theo điểm đề xuất
        if "recommendation_score" in df.columns:
            df = df.sort_values(by="recommendation_score", ascending=False)

        # 6️⃣ Trả kết quả giới hạn
        return df.head(limit)[["name", "category_detail", "avg_cost", "simulated_rating", "vibe"]]
