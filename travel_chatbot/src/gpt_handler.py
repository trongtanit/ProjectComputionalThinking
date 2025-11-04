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
    Hãy phân tích và trả về JSON dạng:
    {{
        "categories": ["ẩm thực", "vui chơi", "du lịch", "nghỉ dưỡng", "mua sắm", ...],
        "budget": số tiền (nếu có),
        "time": "thời gian (ví dụ sáng, chiều, tối, 1 ngày, nhiều ngày, ...)",
        "location": "địa điểm (nếu có)",
        "time_plan": true hoặc false  // true nếu người dùng muốn lên lịch trình 1 ngày
    }}

    Gợi ý:
    - Nếu người dùng nói "lịch trình", "xếp lịch", "plan", "schedule", "kế hoạch", hoặc "1 ngày" thì time_plan = true
    - Nếu người dùng chỉ hỏi gợi ý địa điểm (vd: "ăn sáng ở đâu", "đi chơi ở Đà Lạt") thì time_plan = false
    - Nếu không chắc, đặt time_plan = false
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Bạn là AI phân tích yêu cầu du lịch và ẩm thực của người Việt."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    result = completion.choices[0].message.content
    result = re.sub(r"```json|```", "", result).strip()

    try:
        parsed = json.loads(result)
    except:
        parsed = {
            "categories": ["khác"],
            "budget": 0,
            "time": "unknown",
            "location": "",
            "time_plan": False
        }

    if isinstance(parsed.get("categories"), str):
        parsed["categories"] = [parsed["categories"]]

    return parsed
