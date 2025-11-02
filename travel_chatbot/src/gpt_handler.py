import os, json, re
from openai import OpenAI

def interpret_request(message):
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("❌ Thiếu GITHUB_TOKEN. Hãy đặt biến môi trường trước khi chạy.")

    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=token
    )

    prompt = f"""
    Người dùng: "{message}"
    Hãy trả JSON dạng:
    {{
        "categories": ["ẩm thực", "vui chơi", "du lịch", "nghỉ dưỡng", "mua sắm", ...],
        "budget": số tiền (nếu có),
        "time": "thời gian (số giờ hoặc ngày)",
        "location": "tên địa điểm (nếu có)"
    }}
    Nếu người dùng nói nhiều hoạt động (vd: "đi chơi rồi ăn trưa"), hãy thêm tất cả các loại vào "categories".
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Bạn là AI giúp hiểu yêu cầu du lịch và ẩm thực của người Việt."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    result = completion.choices[0].message.content
    result = re.sub(r"```json|```", "", result).strip()

    try:
        parsed = json.loads(result)
    except:
        parsed = {"categories": ["khác"], "budget": 0, "time": "unknown", "location": ""}

    # ✅ Đảm bảo luôn có dạng list
    if isinstance(parsed.get("categories"), str):
        parsed["categories"] = [parsed["categories"]]

    return parsed
