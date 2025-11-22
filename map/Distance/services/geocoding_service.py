import requests   # ← BẮT BUỘC CÓ

def geocode_address(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": 1
    }

    response = requests.get(url, params=params, headers={"User-Agent": "distance-app"})

    # Kiểm tra lỗi
    if response.status_code != 200:
        print("Nominatim error:", response.text)
        return None

    try:
        data = response.json()
    except Exception as e:
        print("JSONDecodeError:", response.text)
        return None

    if not data:
        return None

    return float(data[0]["lat"]), float(data[0]["lon"])
