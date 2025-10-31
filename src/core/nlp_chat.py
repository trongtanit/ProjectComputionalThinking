from core.data_manager import DataManager

data = DataManager()  # nạp dataset một lần ở đầu file

...

# Sau khi nhận input người dùng
response = client.complete(
    messages=[
        SystemMessage("You are a smart and friendly travel assistant for Ho Chi Minh City. Extract structured info in JSON."),
        UserMessage(f"Người dùng nói: {user_input}. Trả về JSON gồm: food, vibe, area, budget."),
    ],
    model=model
)

raw_text = response.choices[0].message.content
print(f"\n🧠 GPT hiểu: {raw_text}\n")

# Thử tìm địa điểm thật
try:
    import json
    prefs = json.loads(raw_text)
    results = data.find_places(
        food=prefs.get("food"),
        vibe=prefs.get("vibe"),
        area=prefs.get("area"),
        budget=prefs.get("budget", 0)
    )

    if results.empty:
        print("😢 Không tìm thấy địa điểm phù hợp.\n")
    else:
        print("🌆 Gợi ý từ dữ liệu thật:\n")
        print(results.to_string(index=False))
        print()

except Exception as e:
    print(f"⚠️ Lỗi khi xử lý dữ liệu: {e}")
