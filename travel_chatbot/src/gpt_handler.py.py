import os, json, re
from openai import OpenAI

def interpret_request(message, token):
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=token
    )

    prompt = f"""
    Người dùng: "{message}"
    Hãy phân tích và trả JSON:
    {{
        "category": "ẩm thực | vui chơi | mua sắm | du lịch | nghỉ dưỡng | khác",
        "budget": số tiền (VNĐ),
        "time": "sáng | chiều | tối | cả ngày | nhiều ngày | unknown"
    }}
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Bạn là AI phân tích yêu cầu du lịch."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    result = re.sub(r"```json|```", "", completion.choices[0].message.content).strip()
    try:
        return json.loads(result)
    except:
        return {"category": "khác", "budget": 0, "time": "unknown"}
