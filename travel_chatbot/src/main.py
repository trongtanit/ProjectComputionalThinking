import os
from src.data_loader import load_data
from src.gpt_handler import interpret_request
from src.recommender import suggest_places

# 🔑 Lấy token GitHub
token = os.getenv("GITHUB_TOKEN")
if not token:
    print("❌ Thiếu token. Dùng lệnh setx GITHUB_TOKEN \"ghp_xxx\"")
    exit()

# 📂 Nạp dữ liệu từ thư mục /data
data = load_data()

print("🤖 Chatbot du lịch & ăn uống - Gõ 'exit' để thoát\n")

while True:
    msg = input("Bạn: ").strip()
    if msg.lower() in ["exit", "quit", "thoát"]:
        print("👋 Tạm biệt!")
        break

    print("🧩 Đang phân tích yêu cầu...")
    info = interpret_request(msg)
    print("📋 Thông tin phân tích:", info)

    # ⚙️ Xử lý nhiều danh mục cùng lúc
    categories = info.get("categories") or [info.get("category", "khác")]

    # Nếu GPT không trả mảng, ép thành mảng 1 phần tử
    if isinstance(categories, str):
        categories = [categories]

    for cat in categories:
        info["category"] = cat
        print(f"\n🎯 Gợi ý cho mục: {cat.upper()}")

        recs = suggest_places(data, info, msg)
        if recs.empty:
            print("❌ Không tìm thấy địa điểm phù hợp.")
        else:
            for _, row in recs.iterrows():
                try:
                    name = row.get("name", "Không rõ")
                    typ = row.get("poi_type", "N/A")
                    cost = int(float(row.get("avg_cost", 0)))
                    rating = row.get("simulated_rating", "?")
                    open_t = row.get("opening_time", "?")
                    close_t = row.get("closing_time", "?")

                    print(f"- {name} ({typ}) 💰{cost}đ ⭐{rating}")
                    print(f"  ⏰ {open_t} - {close_t}")
                except Exception as e:
                    print(f"⚠️ Lỗi khi đọc dòng: {e}")
            print("\n" + "-" * 50)
