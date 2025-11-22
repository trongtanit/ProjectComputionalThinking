from flask import Blueprint, request, jsonify
from services.geocoding_service import geocode_address

geocode_bp = Blueprint("geocode", __name__)

@geocode_bp.route("/geocode")
def geocode():
    address = request.args.get("address")
    if not address:
        return jsonify({"error": "Missing address"}), 400

    result = geocode_address(address)
    if not result:
        return jsonify({"error": "Address not found"}), 404

    lat, lon = result
    return jsonify({"lat": lat, "lon": lon})
