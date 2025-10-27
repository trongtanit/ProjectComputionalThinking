# osm_scraper.py
"""
OSM scraper + enricher for HCM hot areas.
Output: data/travel_poi_data.csv following schema travel_poi_template.csv
Requires: requests, pandas
Usage: python osm_scraper.py
"""

import requests
import time
import random
import math
import sys
import traceback
from typing import List, Dict, Any, Tuple
import pandas as pd

OVERPASS_URL = "http://overpass-api.de/api/interpreter"
OUT_CSV = "data/travel_poi_data.csv"
TEMPLATE_COLUMNS = [
    "poi_id","name","latitude","longitude","poi_type","category_detail","vibe",
    "avg_cost","simulated_rating","dwell_time_minutes","popularity_score",
    "opening_time","closing_time","time_block_suitability","is_central","recommendation_score"
]

# Hot area bounding boxes (approx). Format: (lat_min, lon_min, lat_max, lon_max)
# These are approximate — you can tweak them.
HOT_AREAS = {
    "Q1": (10.7680, 106.6910, 10.7835, 106.7085),
    "Q3": (10.7700, 106.6855, 10.7890, 106.7050),
    "Q5": (10.7655, 106.6700, 10.7885, 106.6940),
    "Q10": (10.7700, 106.6700, 10.7890, 106.6900),
    "Q7": (10.7280, 106.7120, 10.7540, 106.7430),
    "ThuDuc": (10.8380, 106.7440, 10.8970, 106.8220)
}

# POI tags to query (Overpass)
POI_TAGS = [
    'node["amenity"~"restaurant|cafe|fast_food|bar|pub|cinema"]',
    'node["tourism"~"museum|attraction|viewpoint|gallery"]',
    'node["leisure"~"park|playground"]',
    'node["shop"]'
]

# -----------------------
# Utilities & heuristics
# -----------------------

def build_overpass_query(bbox: Tuple[float, float, float, float]) -> str:
    latmin, lonmin, latmax, lonmax = bbox
    bbox_str = f"{latmin},{lonmin},{latmax},{lonmax}"
    parts = []
    for tag_expr in POI_TAGS:
        parts.append(f"  {tag_expr}({bbox_str});")
        parts.append(f"  way{tag_expr[4:]}({bbox_str});")
        parts.append(f"  relation{tag_expr[4:]}({bbox_str});")
    query = "[out:json][timeout:60];\n(\n" + "\n".join(parts) + "\n);\nout center qt;"
    return query

def fetch_overpass(query: str, max_retries: int = 3, backoff: int = 5) -> Dict[str, Any]:
    for attempt in range(1, max_retries+1):
        try:
            resp = requests.post(OVERPASS_URL, data=query, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[Overpass] attempt {attempt} failed: {e}")
            if attempt < max_retries:
                wait = backoff * attempt + random.uniform(0, 2)
                print(f"[Overpass] retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                print("[Overpass] all retries failed.")
                raise

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def estimate_travel_time_minutes_km(distance_km: float, avg_speed_kmh: float = 25.0) -> float:
    if distance_km is None:
        return float('inf')
    return (distance_km / avg_speed_kmh) * 60.0

# Mapping tags -> category detail (best-effort)
def map_tags_to_category_detail(tags: Dict[str,str]) -> str:
    amenity = tags.get("amenity","").lower()
    tourism = tags.get("tourism","").lower()
    leisure = tags.get("leisure","").lower()
    shop = tags.get("shop","").lower()
    name = tags.get("name","").lower()
    if amenity == "cafe": return "cafe"
    if amenity == "restaurant":
        cuisine = tags.get("cuisine")
        if cuisine:
            return f"restaurant:{cuisine}"
        return "restaurant"
    if amenity == "fast_food": return "fast_food"
    if amenity in ("bar","pub"): return "bar"
    if tourism == "museum": return "museum"
    if tourism in ("attraction","viewpoint","artwork"): return "attraction"
    if leisure == "park": return "park"
    if amenity == "cinema": return "cinema"
    if shop:
        return f"shop:{shop}"
    if "market" in name or "chợ" in name: return "market"
    return "other"

# Infer vibe (7-class)
def infer_vibe(category_detail: str, tags: Dict[str,str]) -> str:
    cd = (category_detail or "").lower()
    name = tags.get("name","").lower()
    if any(k in cd for k in ['rooftop','fine','steak','luxury']) or any(k in name for k in ['rooftop','sky']): return "Luxury"
    if any(k in cd for k in ['bar','pub','nightclub']): return "Party"
    if any(k in cd for k in ['viewpoint','scenic','attraction']) or 'view' in name:
        return "Romantic"
    if any(k in cd for k in ['museum','gallery','historic','monument']): return "Culture"
    if any(k in cd for k in ['park','playground','cinema']): return "Family"
    if any(k in cd for k in ['market','street','food_court']): return "Local"
    if any(k in cd for k in ['cafe','coffee','tea']): return "Chill"
    return "Local"

# Infer avg_cost per person (VND)
def infer_avg_cost(category_detail: str) -> int:
    cd = (category_detail or "").lower()
    if 'luxury' in cd or 'rooftop' in cd or 'fine' in cd: return int(random.uniform(700000,2000000))
    if 'restaurant' in cd:
        if any(x in cd for x in ['japanese','steak']): return int(random.uniform(300000,900000))
        return int(random.uniform(150000,400000))
    if 'cafe' in cd: return int(random.uniform(30000,120000))
    if 'fast_food' in cd: return int(random.uniform(40000,100000))
    if 'market' in cd or 'street' in cd: return int(random.uniform(20000,100000))
    if 'mall' in cd: return int(random.uniform(80000,300000))
    if 'museum' in cd: return int(random.uniform(40000,200000))
    if 'park' in cd: return 0
    return int(random.uniform(30000,200000))

def infer_popularity(tags: Dict[str,str], lat:float, lon:float, category_detail:str) -> float:
    is_central = (10.76 <= lat <= 10.78) and (106.69 <= lon <= 106.71)
    base = 0.4 + (0.25 if is_central else 0.0)
    if any(k in category_detail for k in ['museum','attraction','viewpoint']): base += 0.15
    if 'park' in category_detail: base -= 0.05
    score = min(1.0, max(0.0, base + random.uniform(-0.15,0.15)))
    return round(score,3)

def infer_dwell_time(category_detail: str) -> int:
    cd = (category_detail or "").lower()
    if 'museum' in cd or 'gallery' in cd: return int(random.uniform(60,180))
    if 'park' in cd: return int(random.uniform(20,120))
    if 'restaurant' in cd: return int(random.uniform(45,120))
    if 'cafe' in cd: return int(random.uniform(20,90))
    if 'market' in cd: return int(random.uniform(30,120))
    if 'cinema' in cd: return 120
    return int(random.uniform(20,90))

def infer_open_close(category_detail: str, vibe: str) -> Tuple[str,str]:
    v = vibe
    if v == 'Culture': return ("08:00","17:00")
    if v == 'Local': return ("06:00","22:00")
    if v == 'Family': return ("09:00","21:30")
    if v == 'Chill': return ("07:00","23:00")
    if v == 'Romantic': return ("17:00","23:30")
    if v == 'Party': return ("18:00","02:00")
    if v == 'Luxury': return ("11:00","22:00")
    return ("08:00","21:00")

def infer_time_blocks(vibe: str) -> str:
    mapping = {
        'Culture': ['morning','afternoon'],
        'Local': ['morning','afternoon','evening'],
        'Family': ['morning','afternoon'],
        'Chill': ['afternoon','evening'],
        'Romantic': ['evening','night'],
        'Party': ['night'],
        'Luxury': ['evening']
    }
    return "|".join(mapping.get(vibe, ['afternoon']))

# -----------------------
# Process Overpass elements -> rows
# -----------------------
def element_to_row(el: Dict[str,Any], idx: int) -> Dict[str,Any]:
    tags = el.get("tags", {}) or {}
    # coordinate
    lat = el.get("lat") or (el.get("center") or {}).get("lat")
    lon = el.get("lon") or (el.get("center") or {}).get("lon")
    if lat is None or lon is None:
        return None
    name = tags.get("name") or tags.get("name:en") or f"POI_{el.get('id')}"
    category_detail = map_tags_to_category_detail(tags)
    vibe = infer_vibe(category_detail, tags)
    avg_cost = infer_avg_cost(category_detail)
    opening, closing = infer_open_close(category_detail, vibe)
    time_block = infer_time_blocks(vibe)
    popularity = infer_popularity(tags, lat, lon, category_detail)
    dwell = infer_dwell_time(category_detail)
    poi_type = "Other"
    amenity = tags.get("amenity","")
    tourism = tags.get("tourism","")
    leisure = tags.get("leisure","")
    shop = tags.get("shop","")
    if amenity in ('restaurant','cafe','bar','pub','fast_food'):
        poi_type = "Food_Dining"
    elif tourism or tags.get("historic"):
        poi_type = "Culture_History"
    elif leisure or amenity == 'cinema':
        poi_type = "Entertainment_Leisure"
    elif shop:
        poi_type = "Shopping"
    else:
        poi_type = "Outdoor" if 'park' in category_detail else "Other"

    row = {
        "poi_id": idx,
        "name": name,
        "latitude": round(float(lat), 6),
        "longitude": round(float(lon), 6),
        "poi_type": poi_type,
        "category_detail": category_detail,
        "vibe": vibe,
        "avg_cost": int(avg_cost),
        "simulated_rating": round(random.uniform(3.5,4.8),2),
        "dwell_time_minutes": int(dwell),
        "popularity_score": float(popularity),
        "opening_time": opening,
        "closing_time": closing,
        "time_block_suitability": time_block,
        "is_central": 1 if (10.76 <= float(lat) <= 10.78 and 106.69 <= float(lon) <= 106.71) else 0,
        "recommendation_score": ""
    }
    return row

# -----------------------
# Main scraping routine
# -----------------------
def scrape_hot_areas(areas: Dict[str,Tuple[float,float,float,float]]) -> pd.DataFrame:
    all_rows = []
    seen = set()
    idx = 0
    for area_name, bbox in areas.items():
        print(f"\n=== Scraping area: {area_name} bbox={bbox} ===")
        q = build_overpass_query(bbox)
        try:
            data = fetch_overpass(q)
        except Exception as e:
            print(f"[Error] fetching area {area_name}: {e}")
            continue
        elements = data.get("elements", [])
        print(f"[Info] got {len(elements)} elements from Overpass for {area_name}")
        for el in elements:
            try:
                tags = el.get("tags", {}) or {}
                lat = el.get("lat") or (el.get("center") or {}).get("lat")
                lon = el.get("lon") or (el.get("center") or {}).get("lon")
                if lat is None or lon is None:
                    continue
                name = tags.get("name") or tags.get("name:en") or ""
                key = (name.strip().lower(), round(float(lat),6), round(float(lon),6))
                if key in seen:
                    continue
                seen.add(key)
                row = element_to_row(el, idx)
                if row:
                    all_rows.append(row)
                    idx += 1
            except Exception as e:
                print("[Warn] skipping element due to error:", e)
                traceback.print_exc()
        # sleep small amount to be polite
        time.sleep(random.uniform(1.0, 2.5))
    df = pd.DataFrame(all_rows, columns=TEMPLATE_COLUMNS)
    return df

# -----------------------
# Save with safety (backup)
# -----------------------
def save_csv_safe(df: pd.DataFrame, out_path: str):
    # backup old file if exists
    try:
        import os
        if os.path.exists(out_path):
            ts = int(time.time())
            bak = out_path + f".bak.{ts}"
            os.rename(out_path, bak)
            print(f"[Info] moved old {out_path} -> {bak}")
    except Exception:
        pass
    df.to_csv(out_path, index=False, encoding='utf-8')
    print(f"[Saved] {out_path} ({len(df)} rows)")

# -----------------------
# Entrypoint
# -----------------------
def main():
    print("Start OSM scraper for HCM hot areas...")
    df = scrape_hot_areas(HOT_AREAS)
    # basic filtering: remove rows with missing key fields
    df = df.dropna(subset=["name","latitude","longitude","poi_type"])
    # ensure types
    df["avg_cost"] = df["avg_cost"].fillna(0).astype(int)
    df["dwell_time_minutes"] = df["dwell_time_minutes"].fillna(60).astype(int)
    df["popularity_score"] = df["popularity_score"].fillna(0.0).astype(float)
    # save
    save_csv_safe(df, OUT_CSV)

if __name__ == "__main__":
    main()
