import os
from src.data_loader import load_data
from src.gpt_handler import interpret_request
from src.recommender import suggest_places

token = os.getenv("GITHUB_TOKEN")
if not token:
    print("❌ Thiếu token. Dùng lệnh setx GITHUB_TOKEN \"ghp_xxx\"")
    exit()

data = load_data()

print("🤖 Chatbot du lịch & ăn uống - Gõ 'exit' để thoát\n")

while True:
    msg = input("Bạn: ")
    if msg.lower() in ["exit", "quit", "thoát"]:
        print("👋 Tạm biệt!")
        break

    print("🧩 Đang phân tích...")
    info = interpret_request(msg)
    print("📋", info)

    print("\n🎯 Gợi ý địa điểm:")
    recs = suggest_places(data, info)
    if recs.empty:
        print("Không tìm thấy địa điểm phù hợp.")
    else:
        for _, row in recs.iterrows():
            print(f"- {row['name']} ({row['poi_type']}) 💰{int(row['avg_cost'])}đ ⭐{row['simulated_rating']}")
            print(f"  ⏰ {row['opening_time']} - {row['closing_time']}")
        print("---\n")
