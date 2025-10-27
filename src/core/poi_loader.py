# poi_loader.py
"""
Loader cho POI CSV -> list[PointOfInterest]

Mục tiêu:
- Đọc CSV theo schema `travel_poi_template.csv`
- Chuẩn hoá kiểu (int/float/bool), parse time-blocks "a|b|c" -> set(...)
- Dedupe theo (name, lat, lon)
- Trả về list[PointOfInterest] (để GA / ranker dùng)

Yêu cầu:
- pandas
- file classes.py (PointOfInterest) trong PYTHONPATH hoặc cùng folder
"""
from typing import List, Tuple, Dict, Set, Any, Optional
import pandas as pd
import os
import math
import traceback

# import PointOfInterest class (file classes.py phải có trên PYTHONPATH hoặc cùng folder)
try:
    from classes import PointOfInterest
except Exception as e:
    raise ImportError("Không tìm thấy classes.PointOfInterest. Hãy đảm bảo classes.py nằm cùng folder hoặc trong PYTHONPATH.") from e

# Cột mặc định theo schema đã thống nhất
EXPECTED_COLUMNS = [
    "poi_id","name","latitude","longitude","poi_type","category_detail","vibe",
    "avg_cost","simulated_rating","dwell_time_minutes","popularity_score",
    "opening_time","closing_time","time_block_suitability","is_central","recommendation_score"
]

# Helper parse
def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        try:
            return float(str(x).replace(",", "").strip())
        except Exception:
            return default

def _to_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return int(float(x))
    except Exception:
        try:
            s = str(x).lower().replace("vnđ","").replace("vnd","").replace(",","").strip()
            return int(float(s))
        except Exception:
            return default

def _to_bool(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("1","true","yes","y","t")

def _parse_timeblock_field(s: Any) -> Set[str]:
    """Parse pipe-delimited time blocks 'morning|afternoon' -> set(...)"""
    if s is None:
        return set()
    if isinstance(s, (list, set)):
        return set(str(x).strip() for x in s if str(x).strip())
    st = str(s).strip()
    if st == "" or st.lower() in ("nan", "none"):
        return set()
    # support both pipe and comma just in case
    if "|" in st:
        parts = [p.strip() for p in st.split("|") if p.strip()]
    elif "," in st:
        parts = [p.strip() for p in st.split(",") if p.strip()]
    else:
        parts = [st]
    return set(parts)

def _safe_get(df_row: pd.Series, col: str) -> Any:
    if col not in df_row.index:
        return None
    return df_row[col]

# Main loader
def load_pois_from_csv(csv_path: str, dedupe: bool = True, drop_missing_coords: bool = True) -> List[PointOfInterest]:
    """
    Đọc CSV và trả về list[PointOfInterest].
    - csv_path: đường dẫn tới csv (theo template)
    - dedupe: loại bỏ trùng lặp theo (name, lat, lon)
    - drop_missing_coords: bỏ hàng không có tọa độ
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)  # đọc hết dạng string, parse thủ công để robust

    # Nếu thiếu cột, warn và tiếp tục (chèn cột rỗng)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    rows = []
    seen = set()
    for idx, r in df.iterrows():
        try:
            name = (str(_safe_get(r, "name") or "")).strip()
            lat_raw = _safe_get(r, "latitude")
            lon_raw = _safe_get(r, "longitude")
            try:
                lat = round(_to_float(lat_raw, float("nan")), 6)
                lon = round(_to_float(lon_raw, float("nan")), 6)
            except Exception:
                lat = float("nan")
                lon = float("nan")

            if drop_missing_coords and (math.isnan(lat) or math.isnan(lon)):
                # skip
                continue

            key = (name.lower(), round(lat,6), round(lon,6))
            if dedupe and key in seen:
                continue
            seen.add(key)

            poi_id = _to_int(_safe_get(r, "poi_id"), default=len(seen)-1)
            poi_type = str(_safe_get(r, "poi_type") or "").strip()
            category_detail = _safe_get(r, "category_detail") or None
            vibe = _safe_get(r, "vibe") or None

            avg_cost = _to_int(_safe_get(r, "avg_cost"), default=0)
            simulated_rating = _to_float(_safe_get(r, "simulated_rating"), default=0.0)
            dwell_time_minutes = _to_int(_safe_get(r, "dwell_time_minutes"), default=60)
            popularity_score = _to_float(_safe_get(r, "popularity_score"), default=0.0)

            opening_time = _safe_get(r, "opening_time") or "08:00"
            closing_time = _safe_get(r, "closing_time") or "21:00"

            time_block_suitability = _parse_timeblock_field(_safe_get(r, "time_block_suitability"))
            is_central = _to_bool(_safe_get(r, "is_central"))
            recommendation_score = _to_float(_safe_get(r, "recommendation_score"), default=0.0)

            poi = PointOfInterest(
                poi_id=int(poi_id),
                name=name or f"POI_{poi_id}",
                latitude=float(lat),
                longitude=float(lon),
                poi_type=poi_type or "Other",
                category_detail=category_detail,
                vibe=vibe,
                avg_cost=float(avg_cost),
                simulated_rating=float(simulated_rating),
                dwell_time_minutes=int(dwell_time_minutes),
                popularity_score=float(popularity_score),
                opening_time=str(opening_time),
                closing_time=str(closing_time),
                time_block_suitability=set(time_block_suitability),
                is_central=bool(is_central),
                recommendation_score=float(recommendation_score),
                travel_time_cache={}
            )

            rows.append(poi)
        except Exception as e:
            # không dừng program vì một dòng hỏng, chỉ log
            print(f"[WARN] lỗi parse dòng {idx}: {e}")
            traceback.print_exc()
            continue

    print(f"[INFO] Loaded {len(rows)} POIs from {csv_path}")
    return rows

# Small utilities for workflow
def save_pois_to_pickle(pois: List[PointOfInterest], out_path: str) -> None:
    """Lưu list POI sang pickle để load nhanh lần sau (optional)."""
    import pickle
    with open(out_path, "wb") as f:
        pickle.dump(pois, f)
    print(f"[INFO] Saved {len(pois)} POIs -> {out_path}")

def load_pois_from_pickle(pickle_path: str) -> List[PointOfInterest]:
    import pickle
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(pickle_path)
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    print(f"[INFO] Loaded {len(data)} POIs from {pickle_path}")
    return data
