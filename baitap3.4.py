print("I'm a student")

a = 1 / 7
result = round(a, 5)  # tạo biến result để ghi file

print(result)

b = int(input("Nhập a số nguyên: "))
c = int(input("Nhập b số nguyên: "))
print("Tổng 2 số là:", b + c)

# MỞ FILE
with open("output.txt", "w") as f:
    f.write(f"Kết quả là: {result}\n")
    f.write(f"Tổng 2 số {b} và {c} là: {b + c}\n")

print("Đã ghi kết quả vào file output.txt ✅")
