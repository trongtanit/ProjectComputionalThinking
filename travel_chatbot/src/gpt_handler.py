import os
import json
import re
from openai import OpenAI

# 🧠 Hàm phân tích yêu cầu người dùng
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
        "category": "ẩm thực | du lịch | nghỉ dưỡng | vui chơi | khác",
        "budget": số tiền (ước lượng),
        "time": "số ngày hoặc giờ"
    }}
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Bạn là AI phân tích yêu cầu du lịch của người dùng."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    result = completion.choices[0].message.content
    result = re.sub(r"```json|```", "", result).strip()

    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        parsed = {"category": "khác", "budget": 0, "time": "unknown"}

    return parsed
