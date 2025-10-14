def is_leap_year(year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return True
    else:
        return False

year = int(input("Nhập năm: "))
if is_leap_year(year):
    print(f"{year} là năm nhuận.")
else:
    print(f"{year} không phải là năm nhuận.")