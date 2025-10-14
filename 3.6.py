# Mở file ở chế độ ghi nhị phân ('wb')
with open("output.txt", "wb") as f:
    # Chuỗi cần ghi
    text = "Hôm nay trời đẹp."
    # Ghi ra file sau khi mã hóa UTF-8
    f.write(text.encode("utf-8"))

print("Đã ghi thành công vào file output.txt")
