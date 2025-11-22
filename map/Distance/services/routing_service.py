import requests

def get_route(lon1, lat1, lon2, lat2):
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"

    response = requests.get(url)

    if response.status_code != 200:
        print("OSRM ERROR:", response.text)
        return None

    try:
        data = response.json()
    except Exception as e:
        print("JSONDecodeError:", response.text)
        return None

    if "routes" not in data or not data["routes"]:
        print("NO ROUTE IN OSRM RESPONSE:", data)
        return None

    route = data["routes"][0]["geometry"]["coordinates"]
    distance = data["routes"][0]["distance"] / 1000        # KM
    duration = data["routes"][0]["duration"] / 60          # MINUTES

    return {
        "geometry": route,
        "distance_km": round(distance, 2),
        "duration_min": round(duration, 1)
    }
