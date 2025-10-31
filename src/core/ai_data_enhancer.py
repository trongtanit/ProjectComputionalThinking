import os
import pandas as pd
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import json

# ==== 1. Cấu hình ====
endpoint = "https://models.inference.ai.azure.com"
model = "gpt-4o-mini"  # bạn có thể thay bằng "gpt-4o" nếu muốn mạnh hơn
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

# ==== 2. Đọc file CSV gốc ====
input_path = "../data/travel_poi_data_ranked.csv"
df = pd.read_csv(input_path)
print(f"✅ Loaded {len(df)} rows")

# ==== 3. Lấy mẫu (tránh gửi quá nhiều dữ liệu 1 lần) ====
sample_data = df.head(10).to_dict(orient="records")

# ==== 4. Gọi AI để sinh dữ liệu mới ====
prompt = f"""
Bạn là chuyên gia du lịch Việt Nam. Dưới đây là 10 địa điểm mẫu:
{json.dumps(sample_data, ensure_ascii=False)}

Hãy mở rộng dữ liệu này thành dạng CSV phong phú, dành riêng cho TP.HCM,
mỗi dòng phải gồm:
- name: tên địa điểm (thật hoặc hợp lý ở Việt Nam)
- category_detail: loại chi tiết (vd: quán phở, bar, công viên, chợ, spa...)
- vibe: phong cách (vd: chill, romantic, local, family, party...)
- avg_cost: giá trung bình (VNĐ)
- simulated_rating: điểm đánh giá (1-5)
- district: quận
- opening_time, closing_time
- keywords: mô tả ngắn, có cả tiếng Việt & tiếng Anh xen kẽ

Xuất ra dữ liệu CSV, phân tách bằng dấu phẩy (,).
"""

response = client.complete(
    model=model,
    messages=[
        SystemMessage("You are a data generator for a travel AI in Vietnam."),
        UserMessage(prompt)
    ]
)

# ==== 5. Lưu dữ liệu mới ====
new_data = response.choices[0].message.content
output_path = "../data/travel_poi_data_enhanced.csv"

with open(output_path, "w", encoding="utf-8") as f:
    f.write(new_data)

print(f"✅ Saved enhanced dataset -> {output_path}")
