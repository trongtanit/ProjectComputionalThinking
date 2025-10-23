# data_harvester_final.py
"""
Refactored data harvester + enricher for Advanced pipeline.

- Gọi Overpass để lấy POI
- Chuẩn hoá tags -> category_detail
- Enrich tự động:
    - avg_cost (per_person)
    - vibe (7-class)
    - opening_time / closing_time (gần-realistic)
    - time_block_suitability
    - popularity_score (heuristic)
    - dwell_time_minutes (heuristic)
- Lưu CSV tương thích với classes.PointOfInterest
"""

import requests
import pandas as pd
import random
import time
import math
from typing import List, Dict, Any, Tuple, Optional

# Overpass endpoint
OVERPASS_URL = "http://overpass-api.de/api/interpreter"

# Bounding box (TP.HCM center area default) - can change
LAT_MIN, LON_MIN, LAT_MAX, LON_MAX = 10.75, 106.65, 10.82, 106.75

# POI tag groups to query
POI_TAGS: Dict[str, List[str]] = {
    'Food_Dining': [
        'amenity=restaurant', 'amenity=cafe', 'amenity=fast_food',
        'amenity=bar', 'amenity=pub'
    ],
    'Shopping_Commerce': [
        'shop=mall', 'shop=supermarket', 'shop=clothes', 'shop=electronics'
    ],
    'Culture_History': [
        'tourism=museum', 'historic=monument', 'tourism=gallery'
    ],
    'Entertainment_Leisure': [
        'leisure=park', 'amenity=cinema', 'tourism=theme_park', 'leisure=playground', 'tourism=attraction'
    ],
    'Viewpoint_Attraction': [
        'tourism=viewpoint', 'tourism=artwork'
    ]
}

# ------------------------
# Helper mapping & heuristics
# ------------------------

def build_overpass_query(bbox_coords: Tuple[float, float, float, float], poi_tags: Dict[str, List[str]]) -> str:
    bbox_str = f"{bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]}"
    query = "[out:json][timeout:60];\n(\n"
    for taglist in poi_tags.values():
        for tag in taglist:
            query += f"  node[{tag}]({bbox_str});\n"
            query += f"  way[{tag}]({bbox_str});\n"
            query += f"  relation[{tag}]({bbox_str});\n"
    query += ");\nout center;"
    return query

def fetch_data_from_overpass(query: str) -> List[Dict[str, Any]]:
    print("→ Gửi truy vấn lên Overpass...")
    try:
        r = requests.post(OVERPASS_URL, data=query, timeout=70)
        r.raise_for_status()
        data = r.json()
        elements = data.get('elements', [])
        print(f"← Overpass trả về {len(elements)} phần tử.")
        return elements
    except Exception as e:
        print("‼️ Lỗi khi gọi Overpass:", e)
        return []

# Map tags to a category_detail string (fine grain)
def map_tags_to_category_detail(tags: Dict[str, str]) -> str:
    # Food fine-grained
    if tags.get('amenity') == 'cafe': return 'cafe'
    if tags.get('amenity') == 'restaurant':
        # try cuisine
        cuisine = tags.get('cuisine')
        if cuisine:
            return f"restaurant:{cuisine}"
        return 'restaurant'
    if tags.get('amenity') == 'fast_food': return 'fast_food'
    if tags.get('amenity') in ('bar','pub'): return 'bar'
    if tags.get('tourism') == 'museum': return 'museum'
    if tags.get('tourism') in ('attraction','viewpoint','artwork'): return 'attraction'
    if tags.get('leisure') == 'park': return 'park'
    if tags.get('amenity') == 'cinema': return 'cinema'
    # generic fallback
    if 'shop' in tags: return f"shop:{tags.get('shop')}"
    name = tags.get('name')
    if name:
        # try to infer by keywords in name
        name_l = name.lower()
        if 'mall' in name_l or 'center' in name_l: return 'mall'
        if 'market' in name_l or 'chợ' in name_l: return 'market'
    return 'other'

# Infer vibe from category_detail and tags (7-class: Chill, Party, Romantic, Local, Culture, Family, Luxury)
def infer_vibe(category_detail: str, tags: Dict[str, str]) -> str:
    cd = category_detail.lower()
    # luxury if tags indicate high-end or known keywords
    if any(k in cd for k in ['restaurant', 'rooftop', 'fine', 'luxury', 'hotel']) and ('stars' in tags or tags.get('class')=='premium'):
        return 'Luxury'
    # party if bar/pub/nightclub
    if any(k in cd for k in ['bar','pub','nightclub']):
        return 'Party'
    # romantic if rooftop or scenic viewpoint or restaurant with view
    if any(k in cd for k in ['rooftop','viewpoint','scenic']) or 'rooftop' in tags.get('name','').lower():
        return 'Romantic'
    # culture
    if any(k in cd for k in ['museum','gallery','monument','historic']):
        return 'Culture'
    # family: playground, zoo, mall with kids
    if any(k in cd for k in ['park','playground','zoo','mall','cinema']):
        return 'Family'
    # local: market, street food
    if any(k in cd for k in ['market','street','marketplace','food_court']):
        return 'Local'
    # chill: cafes, small attractions
    if any(k in cd for k in ['cafe','garden','tea','coffee','bungalow']):
        return 'Chill'
    # fallback heuristics based on tags
    if tags.get('amenity') == 'cafe': return 'Chill'
    if tags.get('amenity') == 'restaurant': return 'Food' if False else 'Chill'
    # default
    return 'Local'

# Infer avg_cost per person based on category_detail (VND)
def infer_avg_cost(category_detail: str) -> float:
    cd = category_detail.lower()
    # base ranges (per person)
    if 'fine' in cd or 'luxury' in cd or 'rooftop' in cd:
        return random.uniform(700000, 2000000)  # 700k - 2M
    if 'restaurant' in cd:
        # check cuisine specifics
        if 'japanese' in cd or 'steak' in cd:
            return random.uniform(300000, 900000)
        return random.uniform(150000, 400000)
    if 'cafe' in cd or 'coffee' in cd:
        return random.uniform(40000, 120000)
    if 'fast_food' in cd:
        return random.uniform(50000, 120000)
    if 'bar' in cd or 'pub' in cd:
        return random.uniform(120000, 400000)
    if 'market' in cd or 'street' in cd:
        return random.uniform(30000, 120000)
    if 'mall' in cd:
        return random.uniform(80000, 300000)
    if 'museum' in cd or 'gallery' in cd:
        return random.uniform(40000, 200000)
    if 'park' in cd:
        return 0.0
    # default modest
    return random.uniform(30000, 200000)

# Infer opening/closing times near-realistic based on vibe/category
def infer_open_close(category_detail: str, vibe: str) -> Tuple[str, str]:
    # defaults
    # Map by vibe/category
    v = vibe
    if v == 'Culture':
        return ("08:00", "17:00")
    if v == 'Local':
        return ("06:00", "22:00")
    if v == 'Family':
        return ("09:00", "21:30")
    if v == 'Chill':
        return ("07:00", "23:00")
    if v == 'Romantic':
        return ("17:00", "23:30")
    if v == 'Party':
        return ("18:00", "02:00")
    if v == 'Luxury':
        return ("11:00", "22:00")
    # fallback
    return ("08:00", "21:00")

# Infer suitable time blocks
def infer_time_block_suitability(vibe: str) -> List[str]:
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

# Popularity heuristic: central + some randomness + POI type weight
def infer_popularity(tags: Dict[str, str], lat: float, lon: float, category_detail: str) -> float:
    # central area heuristic: Quận 1 approx bounding box (rough)
    is_central = (10.76 <= lat <= 10.78) and (106.69 <= lon <= 106.71)
    base = 0.4 + (0.3 if is_central else 0.0)
    # boost for museums / attractions
    if any(k in category_detail for k in ['museum','attraction','viewpoint','gallery']):
        base += 0.15
    # penalize parks a little
    if 'park' in category_detail:
        base -= 0.05
    score = min(1.0, max(0.0, base + random.uniform(-0.15, 0.15)))
    return round(score, 3)

# Dwell time reasonable defaults (minutes)
def infer_dwell_time(category_detail: str, vibe: str) -> int:
    cd = category_detail.lower()
    if 'museum' in cd or 'gallery' in cd:
        return random.randint(60, 180)
    if 'park' in cd:
        return random.randint(30, 120)
    if 'restaurant' in cd:
        return random.randint(60, 120)
    if 'cafe' in cd or 'coffee' in cd:
        return random.randint(30, 90)
    if 'market' in cd:
        return random.randint(30, 120)
    if 'cinema' in cd:
        return 120
    # fallback
    return random.randint(20, 90)

# ------------------------
# Main processing
# ------------------------

def process_and_enrich(elements: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    seen = set()
    idx = 0
    for el in elements:
        tags = el.get('tags', {})
        # get a coordinate (node has lat/lon, way/relation use center)
        lat = el.get('lat') or (el.get('center') or {}).get('lat')
        lon = el.get('lon') or (el.get('center') or {}).get('lon')
        if lat is None or lon is None:
            continue
        # name
        name = tags.get('name') or tags.get('name:en') or f"POI_{el.get('id')}"
        key = (name, round(lat,6), round(lon,6))
        if key in seen:
            continue
        seen.add(key)
        category_detail = map_tags_to_category_detail(tags)
        vibe = infer_vibe(category_detail, tags)
        avg_cost = infer_avg_cost(category_detail)  # per person
        opening, closing = infer_open_close(category_detail, vibe)
        time_blocks = infer_time_block_suitability(vibe)
        popularity = infer_popularity(tags, lat, lon, category_detail)
        dwell = infer_dwell_time(category_detail, vibe)
        poi_type = None
        # coarse poi_type assign based on our POI_TAGS mapping
        # best-effort: check amenity/tourism/leisure/shop/historic
        if 'amenity' in tags and tags['amenity'] in ('restaurant','cafe','bar','pub','fast_food'):
            poi_type = 'Food_Dining'
        elif 'tourism' in tags or 'historic' in tags:
            poi_type = 'Culture_History'
        elif 'leisure' in tags or tags.get('amenity') == 'cinema':
            poi_type = 'Entertainment_Leisure'
        elif 'shop' in tags or 'marketplace' in tags:
            poi_type = 'Shopping_Commerce'
        else:
            poi_type = 'Other'

        is_central = (10.76 <= lat <= 10.78) and (106.69 <= lon <= 106.71)

        rows.append({
            'poi_id': idx,
            'name': name,
            'latitude': lat,
            'longitude': lon,
            'poi_type': poi_type,
            'category_detail': category_detail,
            'vibe': vibe,
            'avg_cost': round(float(avg_cost)),
            'simulated_rating': round(random.uniform(3.5, 4.8), 2),
            'dwell_time_minutes': int(dwell),
            'popularity_score': popularity,
            'opening_time': opening,
            'closing_time': closing,
            'time_block_suitability': ",".join(time_blocks),
            'is_central': is_central
        })
        idx += 1

    df = pd.DataFrame(rows)
    return df

# ------------------------
# Main
# ------------------------

def main():
    print("=== START: data_harvester_final.py (refactor advanced) ===")
    bbox = (LAT_MIN, LON_MIN, LAT_MAX, LON_MAX)
    q = build_overpass_query(bbox, POI_TAGS)
    elements = fetch_data_from_overpass(q)
    if not elements:
        print("Không nhận được dữ liệu từ Overpass. Kiểm tra kết nối hoặc bounding box.")
        return

    df = process_and_enrich(elements)
    if df.empty:
        print("Không có POI sau khi xử lý.")
        return

    # drop duplicates by name+lat+lon
    df.drop_duplicates(subset=['name', 'latitude', 'longitude'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # ensure poi_id consistent
    df['poi_id'] = df.index

    # Add columns helpful for ranking/GA (pre-calc): time windows in minutes maybe later
    # Save CSV
    out_file = "travel_poi_data_final.csv"
    df.to_csv(out_file, index=False, encoding='utf-8')
    print(f"✅ Đã lưu {len(df)} POI vào {out_file}")

if __name__ == "__main__":
    main()
