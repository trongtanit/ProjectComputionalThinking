from flask import Blueprint, request, jsonify
from services.geocoding_service import geocode_address
from geopy.distance import geodesic

distance_bp = Blueprint("distance", __name__)

# --- NEW: Hỗ trợ parse "lat,lon" ---
def parse_coordinates(value):
    """Parse 'lat,lon' into float coordinates."""
    if not value:
        return None
    if "," in value:
        try:
            lat, lon = map(float, value.split(","))
            return (lat, lon)
        except:
            return None
    return None


@distance_bp.route("/distance")
def calc_distance():
    """
    Calculate distance between 2 locations.
    Supports BOTH:
    - /distance?a=Address1&b=Address2
    - /distance?a=10.77,106.70&b=10.79,106.72
    """
    a = request.args.get("a")
    b = request.args.get("b")

    if not a or not b:
        return jsonify({"error": "Missing query params 'a' and 'b'"}), 400

    # --- NEW: xử lý nếu là tọa độ ---
    coordA = parse_coordinates(a)
    coordB = parse_coordinates(b)

    if coordA and coordB:
        dist = geodesic(coordA, coordB).km
        return jsonify({
            "distance_km": round(dist, 2),
            "from": {"lat": coordA[0], "lon": coordA[1]},
            "to": {"lat": coordB[0], "lon": coordB[1]},
        })

    # --- Cũ: xử lý địa chỉ ---
    locA = geocode_address(a)
    locB = geocode_address(b)

    if not locA or not locB:
        return jsonify({"error": "One of the addresses was not found"}), 404

    dist = geodesic(locA, locB).km

    return jsonify({
        "distance_km": round(dist, 2),
        "from": {"lat": locA[0], "lon": locA[1]},
        "to": {"lat": locB[0], "lon": locB[1]},
    })


@distance_bp.route("/distance/from_here")
def calc_distance_from_here():
    """
    Calculate distance from user's current coordinates to a destination.
    Supports:
    - ?lat=<lat>&lon=<lon>&dest=<address>
    - ?lat=<lat>&lon=<lon>&dest_lat=<lat2>&dest_lon=<lon2>
    """
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    dest = request.args.get("dest")
    dest_lat = request.args.get("dest_lat", type=float)
    dest_lon = request.args.get("dest_lon", type=float)

    if lat is None or lon is None:
        return jsonify({"error": "Missing 'lat' or 'lon' query params"}), 400

    if dest_lat is not None and dest_lon is not None:
        destination = (dest_lat, dest_lon)
    elif dest:
        destination = geocode_address(dest)
        if not destination:
            return jsonify({"error": "Destination address not found"}), 404
    else:
        return jsonify({"error": "Provide either dest (address) or dest_lat & dest_lon"}), 400

    dist = geodesic((lat, lon), destination).km

    return jsonify({
        "distance_km": round(dist, 2),
        "from": {"lat": lat, "lon": lon},
        "to": {"lat": destination[0], "lon": destination[1]},
    })
