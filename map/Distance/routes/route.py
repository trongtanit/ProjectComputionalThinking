from flask import Blueprint, request, jsonify
from services.geocoding_service import geocode_address
from services.routing_service import get_route

route_bp = Blueprint("route_bp", __name__)

def parse_coordinates(value):
    """Kiểm tra chuỗi có phải là 'lat, lon' không"""
    try:
        parts = value.split(",")
        if len(parts) != 2:
            return None
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        return (lat, lon)
    except:
        return None


@route_bp.route("/route")
def routing():
    a = request.args.get("start")
    b = request.args.get("end")

    if not a or not b:
        return jsonify({"error": "missing parameters"}), 400

    # ================================
    # 1. KIỂM TRA INPUT CÓ PHẢI TỌA ĐỘ KHÔNG?
    # ================================
    coordA = parse_coordinates(a)
    coordB = parse_coordinates(b)

    # Nếu là tọa độ → bỏ qua geocode
    if coordA:
        lat1, lon1 = coordA
    else:
        locA = geocode_address(a)
        if not locA:
            return jsonify({"error": "invalid address A"}), 400
        lon1, lat1 = locA[1], locA[0]

    if coordB:
        lat2, lon2 = coordB
    else:
        locB = geocode_address(b)
        if not locB:
            return jsonify({"error": "invalid address B"}), 400
        lon2, lat2 = locB[1], locB[0]

    # ================================
    # 2. GỌI OSRM HOÀN TOÀN OFFLINE
    # ================================
    data = get_route(lon1, lat1, lon2, lat2)

    if not data:
        return jsonify({"error": "no route"}), 400

    return jsonify(data)
