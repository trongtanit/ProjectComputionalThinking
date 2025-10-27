# data_cleaner.py
"""
POICleaner (Level 2) - rule-based + heuristic cleaner for POI list.

Features:
- Normalize poi_type to ENUM set
- Heuristic infer category_detail if missing
- Heuristic infer vibe if missing (based on category, cost, rating)
- Ensure time_block_suitability not empty (infer from vibe & opening/closing)
- Fix avg_cost edge cases (e.g., 0 for restaurants -> estimate)
- Compute is_central if missing
- Compute readiness_score (0..1) used by ranker/GA as baseline
- Save cleaned CSV / pickle for next steps

Usage (quick):
    python src/core/data_cleaner.py
Requires:
    - pandas
    - src/core/poi_loader.py (for quick CLI mode) or a pois.pickle
    - classes.PointOfInterest
"""
from typing import List, Dict, Any, Set
import math
import os
import pickle
import statistics
import csv

# try import PointOfInterest
try:
    from classes import PointOfInterest
except Exception as e:
    raise ImportError("Không tìm thấy classes.PointOfInterest (classes.py). Hãy đảm bảo classes.py nằm trong PYTHONPATH hoặc cùng folder.") from e

# Optionally use loader if present
_LOADER_AVAILABLE = False
try:
    from poi_loader import load_pois_from_csv
    _LOADER_AVAILABLE = True
except Exception:
    _LOADER_AVAILABLE = False

# Constants
CENTER_LAT = 10.776892   # approximate downtown reference (Bưu điện TP.HCM)
CENTER_LON = 106.700806

POI_TYPE_ENUM = {
    "food": "Food_Dining",
    "restaurant": "Food_Dining",
    "cafe": "Food_Dining",
    "bar": "Food_Dining",
    "pub": "Food_Dining",
    "fast_food": "Food_Dining",
    "shopping": "Shopping",
    "mall": "Shopping",
    "market": "Shopping",
    "culture": "Culture_History",
    "museum": "Culture_History",
    "historic": "Culture_History",
    "tourism": "Culture_History",
    "entertainment": "Entertainment_Leisure",
    "park": "Outdoor",
    "playground": "Outdoor",
    "viewpoint": "Outdoor",
    "other": "Outdoor"
}

TIMEBLOCK_ORDER = ["morning", "afternoon", "evening", "night"]


class POICleaner:
    """
    Cleaner object encapsulates rules and heuristics to make POI data ready for ranking & routing.
    """

    def __init__(self, central_lat: float = CENTER_LAT, central_lon: float = CENTER_LON):
        self.central_lat = central_lat
        self.central_lon = central_lon

    # ----------------------
    # Public
    # ----------------------
    def clean(self, pois: List[PointOfInterest]) -> List[PointOfInterest]:
        """
        Clean a list of PointOfInterest and return cleaned list.
        This modifies POI objects in-place and returns the same list (cleaned).
        """
        # 1. Normalize poi_type and category_detail
        for p in pois:
            p.poi_type = self._normalize_poi_type(p.poi_type)
            if not p.category_detail:
                p.category_detail = self._infer_category_detail_from_name(p.name, p.poi_type)

        # 2. Fix avg_cost edge cases and simulated_rating bounds
        self._fix_costs_and_ratings(pois)

        # 3. Infer vibe if missing
        for p in pois:
            if not p.vibe:
                p.vibe = self._infer_vibe(p)

        # 4. Ensure time block suitability
        for p in pois:
            if not p.time_block_suitability:
                p.time_block_suitability = set(self._infer_time_blocks_from_vibe(p.vibe))
            else:
                # normalize values to known tokens
                p.time_block_suitability = set([tb for tb in p.time_block_suitability if tb in TIMEBLOCK_ORDER])

            # ensure opening/closing time sanity
            p.opening_time, p.closing_time = self._normalize_opening_closing(p.opening_time, p.closing_time, p.vibe)

        # 5. Compute is_central (if not set or suspicious)
        for p in pois:
            if not getattr(p, "is_central", False):
                p.is_central = self._compute_is_central(p.latitude, p.longitude)

        # 6. Compute readiness_score & sanity caps
        for p in pois:
            p.recommendation_score = float(self._compute_readiness_score(p))

        # 7. Deduplicate (by name+coord) - keep highest readiness_score
        cleaned = self._dedupe_keep_best(pois)

        # 8. Final type-safety casts
        for p in cleaned:
            p.avg_cost = float(max(0, p.avg_cost))
            p.popularity_score = float(max(0.0, min(1.0, p.popularity_score if p.popularity_score is not None else 0.0)))
            p.dwell_time_minutes = int(max(5, min(8*60, p.dwell_time_minutes)))  # clamp reasonable dwell time
            p.simulated_rating = float(max(0.0, min(5.0, p.simulated_rating)))

        return cleaned

    # ----------------------
    # Helpers
    # ----------------------
    def _normalize_poi_type(self, raw: Any) -> str:
        """
        Normalize various raw poi_type values to the fixed enum:
        Food_Dining, Culture_History, Entertainment_Leisure, Shopping, Outdoor
        """
        if not raw:
            return "Outdoor"
        s = str(raw).strip().lower()
        # direct mapping
        for k, v in POI_TYPE_ENUM.items():
            if k in s:
                return v
        # common values
        if s in ("food", "food_dining", "food-dining"):
            return "Food_Dining"
        if s in ("culture_history", "culture-history", "culture"):
            return "Culture_History"
        if s in ("entertainment", "entertainment_leisure"):
            return "Entertainment_Leisure"
        if s in ("shopping", "mall", "market"):
            return "Shopping"
        # fallback
        return "Outdoor"

    def _infer_category_detail_from_name(self, name: str, poi_type: str) -> str:
        """
        Best-effort infer category_detail from name keywords and type.
        This is a lightweight heuristic to make category_detail non-empty for later ranking.
        """
        n = (name or "").lower()
        if poi_type == "Food_Dining":
            if "phở" in n or "pho" in n or "bun" in n or "hu tieu" in n or "bún" in n:
                return "street_noodle"
            if "cafe" in n or "coffee" in n or "highlands" in n or "starbucks" in n:
                return "cafe"
            if "lẩu" in n or "lau" in n:
                return "hotpot"
            if "buffet" in n:
                return "buffet"
            return "restaurant"
        if poi_type == "Culture_History":
            if "bảo tàng" in n or "museum" in n:
                return "museum"
            if "chùa" in n or "nhà thờ" in n:
                return "religious"
            return "landmark"
        if poi_type == "Shopping":
            if "mall" in n or "takashimaya" in n or "vincom" in n:
                return "mall"
            if "chợ" in n or "cho" in n:
                return "market"
            return "shop"
        if poi_type == "Entertainment_Leisure":
            if "cinema" in n or "rạp" in n:
                return "cinema"
            return "leisure"
        return "other"

    def _fix_costs_and_ratings(self, pois: List[PointOfInterest]) -> None:
        """
        Fix clearly wrong avg_cost or rating entries:
        - If avg_cost == 0 for Food_Dining -> estimate default small value
        - Clamp ratings to 0..5
        """
        # compute median cost of restaurants to use as fallback
        costs = [p.avg_cost for p in pois if p.avg_cost and p.avg_cost > 0]
        med_cost = int(statistics.median(costs)) if costs else 50000

        for p in pois:
            if p.poi_type == "Food_Dining" and (not p.avg_cost or p.avg_cost <= 0):
                # set to median or heuristic based on category_detail
                h = med_cost
                if p.category_detail and ("cafe" in p.category_detail):
                    h = int(max(20000, med_cost * 0.5))
                p.avg_cost = float(h)
            # clamp rating
            if p.simulated_rating is None:
                p.simulated_rating = 0.0
            p.simulated_rating = float(max(0.0, min(5.0, p.simulated_rating)))

    def _infer_vibe(self, p: PointOfInterest) -> str:
        """
        Infer vibe using category_detail, popularity, cost and rating.
        Heuristics:
         - Luxury if high cost > 700k or name contains 'rooftop'
         - Party if bar/pub keywords or open late
         - Romantic if viewpoint/rooftop and evening-suitable
         - Culture if museum/gallery
         - Family for parks/cinema
         - Chill for cafe
         - Local for market/street food
        """
        cd = (p.category_detail or "").lower()
        name = (p.name or "").lower()
        cost = p.avg_cost or 0
        rating = p.simulated_rating or 0
        pop = p.popularity_score or 0.0

        # luxury
        if cost >= 700000 or any(k in name for k in ("rooftop", "sky", "penthouse")):
            return "Luxury"
        # party (late night)
        if any(k in cd for k in ("bar", "pub", "nightclub")) or (p.closing_time and p.closing_time < "06:00"):
            return "Party"
        # romantic
        if any(k in cd for k in ("viewpoint", "scenic", "rooftop")) or pop > 0.85 and rating > 4.2:
            return "Romantic"
        # culture
        if any(k in cd for k in ("museum", "gallery", "historic", "landmark")):
            return "Culture"
        # family
        if any(k in cd for k in ("park", "playground", "cinema")):
            return "Family"
        # chill
        if "cafe" in cd or "coffee" in name:
            return "Chill"
        # local
        if any(k in cd for k in ("street", "market", "noodle", "street_noodle")):
            return "Local"
        # fallback
        return "Local"

    def _infer_time_blocks_from_vibe(self, vibe: str) -> List[str]:
        mapping = {
            'Culture': ['morning', 'afternoon'],
            'Local': ['morning', 'afternoon', 'evening'],
            'Family': ['morning', 'afternoon'],
            'Chill': ['afternoon', 'evening'],
            'Romantic': ['evening', 'night'],
            'Party': ['night'],
            'Luxury': ['evening']
        }
        return mapping.get(vibe, ['afternoon'])

    def _normalize_opening_closing(self, opening: str, closing: str, vibe: str) -> (str, str):
        """
        Ensure opening/closing strings are valid HH:MM. If invalid, infer from vibe.
        Very naive parse — assumes given strings in HH:MM or empty.
        """
        def safe_hhmm(s):
            try:
                if s is None:
                    return None
                st = str(s).strip()
                if st == "" or st.lower() in ("nan", "none"):
                    return None
                # if only hour provided like "8" or "8:00" handle it
                if ":" not in st:
                    h = int(float(st))
                    return f"{h:02d}:00"
                parts = st.split(":")
                h = int(parts[0]) % 24
                m = int(parts[1]) % 60 if len(parts) > 1 else 0
                return f"{h:02d}:{m:02d}"
            except Exception:
                return None

        o = safe_hhmm(opening)
        c = safe_hhmm(closing)
        if o and c:
            return o, c
        # fallback infer
        if vibe == "Culture":
            return "08:00", "17:00"
        if vibe == "Party":
            return "18:00", "02:00"
        if vibe == "Romantic":
            return "17:00", "23:30"
        if vibe == "Chill":
            return "07:00", "23:00"
        return "08:00", "21:00"

    def _compute_is_central(self, lat: float, lon: float) -> bool:
        # simple distance threshold to CENTER_LAT/CENTER_LON
        try:
            dist_km = self._haversine_km(lat, lon, self.central_lat, self.central_lon)
            return dist_km <= 2.0  # within 2km considered central
        except Exception:
            return False

    def _haversine_km(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _compute_readiness_score(self, p: PointOfInterest) -> float:
        """
        Compute a readiness score (0..1) used as baseline recommendation.
        Combines:
         - popularity_score (0..1)
         - normalized rating (0..1)
         - recommendation_score (if present)
         - is_central bonus
         - cost penalty if over-expensive (example heuristic)
        """
        pop = float(p.popularity_score or 0.0)
        rating = float((p.simulated_rating or 0.0) / 5.0)  # normalize to 0..1
        rec = float(p.recommendation_score or 0.0)
        central_bonus = 0.1 if getattr(p, "is_central", False) else 0.0

        # cost penalty: prefer moderate cost (100k - 400k). outside lowers score slightly
        cost = float(p.avg_cost or 0.0)
        if cost <= 0:
            cost_penalty = -0.05
        elif cost < 100000:
            cost_penalty = 0.02
        elif cost <= 400000:
            cost_penalty = 0.05
        elif cost <= 800000:
            cost_penalty = 0.02
        else:
            cost_penalty = -0.08

        # weighted combination
        score = 0.45 * pop + 0.35 * rating + 0.15 * rec + central_bonus + cost_penalty
        # clamp 0..1
        return max(0.0, min(1.0, score))

    def _dedupe_keep_best(self, pois: List[PointOfInterest]) -> List[PointOfInterest]:
        """
        Deduplicate POIs by (name, lat, lon). Keep the one with highest readiness (recommendation_score).
        """
        index = {}
        for p in pois:
            key = (p.name.strip().lower(), round(float(p.latitude), 6), round(float(p.longitude), 6))
            if key not in index:
                index[key] = p
            else:
                existing = index[key]
                # prefer higher recommendation_score (computed earlier), then higher popularity
                if getattr(p, "recommendation_score", 0.0) > getattr(existing, "recommendation_score", 0.0):
                    index[key] = p
                elif getattr(p, "popularity_score", 0.0) > getattr(existing, "popularity_score", 0.0):
                    index[key] = p
        return list(index.values())

    # ----------------------
    # IO helpers
    # ----------------------
    def save_to_pickle(self, pois: List[PointOfInterest], out_path: str):
        with open(out_path, "wb") as f:
            pickle.dump(pois, f)
        print(f"[Saved] {out_path} ({len(pois)} POIs)")

    def save_to_csv(self, pois: List[PointOfInterest], out_path: str):
        # Use header compatible with template
        header = [
            "poi_id","name","latitude","longitude","poi_type","category_detail","vibe",
            "avg_cost","simulated_rating","dwell_time_minutes","popularity_score",
            "opening_time","closing_time","time_block_suitability","is_central","recommendation_score"
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for p in pois:
                tb = "|".join(sorted(list(p.time_block_suitability))) if p.time_block_suitability else ""
                row = [
                    int(p.poi_id),
                    p.name,
                    round(float(p.latitude), 6),
                    round(float(p.longitude), 6),
                    p.poi_type,
                    p.category_detail or "",
                    p.vibe or "",
                    int(p.avg_cost) if p.avg_cost is not None else 0,
                    float(p.simulated_rating) if p.simulated_rating is not None else 0.0,
                    int(p.dwell_time_minutes),
                    float(p.popularity_score) if p.popularity_score is not None else 0.0,
                    p.opening_time or "",
                    p.closing_time or "",
                    tb,
                    1 if getattr(p, "is_central", False) else 0,
                    float(p.recommendation_score) if p.recommendation_score is not None else 0.0
                ]
                writer.writerow(row)
        print(f"[Saved CSV] {out_path} ({len(pois)} rows)")

# ----------------------
# Quick-run CLI (if executed directly)
# ----------------------
def _try_load_pois_from_pickle_or_csv():
    # prefer cleaned pickle
    if os.path.exists("data/pois_cleaned.pickle"):
        with open("data/pois_cleaned.pickle", "rb") as f:
            return pickle.load(f)
    # prefer raw pickle
    if os.path.exists("data/pois.pickle"):
        with open("data/pois.pickle", "rb") as f:
            return pickle.load(f)
    # fallback to csv via loader if available
    if os.path.exists("data/travel_poi_data.csv") and _LOADER_AVAILABLE:
        return load_pois_from_csv("data/travel_poi_data.csv")
    if os.path.exists("data/travel_poi_template.csv") and _LOADER_AVAILABLE:
        return load_pois_from_csv("data/travel_poi_template.csv")
    return []

if __name__ == "__main__":
    print("POI Cleaner (Level 2) - running quick pipeline...")
    pois = _try_load_pois_from_pickle_or_csv()
    if not pois:
        print("[ERR] Không tìm thấy nguồn POI (data/pois.pickle hoặc data/travel_poi_data.csv). Chạy scraper trước hoặc đặt file pois.pickle.")
        raise SystemExit(1)
    cleaner = POICleaner()
    cleaned = cleaner.clean(pois)
    # save outputs
    os.makedirs("data", exist_ok=True)
    cleaner.save_to_pickle(cleaned, "data/pois_cleaned.pickle")
    cleaner.save_to_csv(cleaned, "data/travel_poi_data_cleaned.csv")
    print("[DONE] Cleaning finished.")
