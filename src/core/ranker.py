"""
ranker.py - ML-light ranker (offline training) for POI recommendation_score.

Design:
- Train a regression model to predict recommendation_score in [0,1]
  using features from cleaned POI dataset.
- Uses ElasticNetCV (regularized linear model) wrapped with StandardScaler.
- Save model + metadata (feature names) to models/ranker_model.joblib
- Provide utilities:
    - train_model()
    - load_model()
    - predict_and_update(pois)
    - export updated POIs to pickle/csv

Assumptions:
- Input data is cleaned and consistent (output of data_cleaner.py)
- recommendation_score exists as a preliminary label OR we synthesize a target
  from readiness/popularity/rating if missing.
"""

from typing import List, Dict, Any, Tuple
import os
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import SimpleImputer

# Import PointOfInterest class
try:
    from classes import PointOfInterest
except Exception as e:
    raise ImportError("Không tìm thấy classes.PointOfInterest. Hãy đảm bảo classes.py nằm trong PYTHONPATH hoặc cùng folder.") from e

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ranker_model.joblib")
FEATURE_META_PATH = os.path.join(MODEL_DIR, "ranker_feature_meta.pkl")


# ---------------------------
# Feature engineering helpers
# ---------------------------
def pois_to_dataframe(pois: List[PointOfInterest]) -> pd.DataFrame:
    """
    Convert list[PointOfInterest] -> pandas.DataFrame (one row per POI).
    This keeps both numeric features and categorical columns used for encoding.
    """
    rows = []
    for p in pois:
        rows.append({
            "poi_id": int(p.poi_id),
            "name": p.name,
            "latitude": float(p.latitude),
            "longitude": float(p.longitude),
            "poi_type": p.poi_type or "Outdoor",
            "category_detail": p.category_detail or "",
            "vibe": p.vibe or "Local",
            "avg_cost": float(p.avg_cost or 0.0),
            "simulated_rating": float(p.simulated_rating or 0.0),
            "dwell_time_minutes": int(p.dwell_time_minutes or 60),
            "popularity_score": float(p.popularity_score or 0.0),
            "is_central": int(bool(getattr(p, "is_central", False))),
            "opening_time": p.opening_time or "",
            "closing_time": p.closing_time or "",
            "time_block_suitability": "|".join(sorted(list(p.time_block_suitability))) if p.time_block_suitability else ""
            # recommendation_score will be added later if needed
        })
    df = pd.DataFrame(rows)
    return df


def synthesize_target(df: pd.DataFrame) -> pd.Series:
    """
    If recommendation_score not present or many zeros, create a synthetic target
    as a weighted combination of available proxies:
      target = w1 * popularity + w2 * normalized_rating + w3 * central_bonus + w4 * cost_penalty
    Scale into 0..1.
    This provides a training target when ground-truth labels are not available.
    """
    pop = df["popularity_score"].fillna(0.0).astype(float)
    rating = df["simulated_rating"].fillna(0.0).astype(float) / 5.0
    central = df["is_central"].fillna(0).astype(int)
    cost = df["avg_cost"].fillna(0.0).astype(float)

    # cost preference: moderate cost is slightly preferred
    cost_pref = np.clip(1.0 - (np.log1p(cost) / np.log1p(cost.max() + 1e-6)), 0.0, 1.0)

    raw = 0.5 * pop + 0.35 * rating + 0.05 * central + 0.10 * cost_pref
    # normalize 0..1
    raw = np.clip(raw, 0.0, 1.0)
    return raw


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Prepare a feature matrix X and a metadata dict used when building the pipeline.

    Returns:
      X_df: DataFrame with columns used for numeric/categorical transformation
      meta: dict with lists of categorical and numeric features
    """
    # Choose features (keep relatively few to avoid overfitting in small datasets)
    numeric_features = [
        "avg_cost", "simulated_rating", "dwell_time_minutes", "popularity_score",
        "latitude", "longitude", "is_central"
    ]
    categorical_features = [
        "poi_type", "vibe"
    ]
    # We also transform time_block_suitability into simple counts/presence features
    # Create derived features
    df = df.copy()
    # count how many blocks it suits (morning/afternoon/evening/night)
    df["time_block_count"] = df["time_block_suitability"].fillna("").apply(lambda s: 0 if s == "" else len(s.split("|")))
    numeric_features.append("time_block_count")

    # One-hot for top-k category_detail could be added, but to keep model simple we skip it.
    X_df = df[numeric_features + categorical_features].copy()
    meta = {"numeric": numeric_features, "categorical": categorical_features}
    return X_df, meta


# ---------------------------
# Model training / saving
# ---------------------------
def build_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    """
    Build sklearn Pipeline:
      ColumnTransformer:
        - numeric: imputer + StandardScaler
        - categorical: SimpleImputer + OneHotEncoder(handle_unknown='ignore')
      Estimator: ElasticNetCV
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop")

    # ElasticNetCV will perform internal CV to select alpha/l1_ratio
    enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, n_alphas=60, max_iter=5000, random_state=42)

    pipe = Pipeline(steps=[
        ("pre", preprocessor),
        ("model", enet)
    ])
    return pipe


def train_model(pois: List[PointOfInterest], save_model: bool = True) -> Dict[str, Any]:
    """
    Train model from list[PointOfInterest]:
      - Build DataFrame
      - If recommendation_score exists and non-trivial -> use as y
      - Else synthesize y
      - Train ElasticNetCV with CV
      - Save pipeline + meta to disk

    Returns dict with training info (cv scores, model path).
    """
    df = pois_to_dataframe(pois)
    # target selection
    if "recommendation_score" in df.columns and df["recommendation_score"].notnull().sum() > 0:
        # if recommendation_score exists in POI objects it's likely all zeros or placeholder;
        # prefer using it if it has variance, otherwise synthesize
        if df["recommendation_score"].astype(float).std() > 1e-6:
            y = df["recommendation_score"].astype(float).fillna(0.0)
        else:
            y = synthesize_target(df)
    else:
        y = synthesize_target(df)

    X_df, meta = prepare_features(df)
    numeric = meta["numeric"]
    categorical = meta["categorical"]

    pipe = build_pipeline(numeric, categorical)

    # cross-validation estimate: use KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_df, y, cv=kf, scoring="r2")
    print(f"[train] CV R2 scores: {scores}, mean={scores.mean():.4f}")

    # fit on full data
    pipe.fit(X_df, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    if save_model:
        joblib.dump(pipe, MODEL_PATH)
        # save meta for later use (feature names)
        with open(FEATURE_META_PATH, "wb") as f:
            pickle.dump(meta, f)
        print(f"[train] Saved model -> {MODEL_PATH}")
    return {"cv_scores": scores, "model_path": MODEL_PATH, "meta": meta}


def load_model() -> Pipeline:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run training first.")
    pipe = joblib.load(MODEL_PATH)
    return pipe


def predict_scores_for_pois(pois: List[PointOfInterest], pipe: Pipeline = None) -> List[float]:
    """
    Predict recommendation_score for each POI using the trained pipeline.
    Returns list of scores in same order as pois.
    """
    if pipe is None:
        pipe = load_model()
    df = pois_to_dataframe(pois)
    X_df, _ = prepare_features(df)
    preds = pipe.predict(X_df)
    # clamp to 0..1
    preds = np.clip(preds, 0.0, 1.0)
    return preds.tolist()


def update_pois_with_scores(pois: List[PointOfInterest], scores: List[float]) -> None:
    """
    Update POI objects in-place with predicted recommendation_score.
    """
    for p, s in zip(pois, scores):
        p.recommendation_score = float(s)


# ---------------------------
# Utility I/O
# ---------------------------
def load_pois_from_pickle_or_csv() -> List[PointOfInterest]:
    # prefer cleaned pickle
    if os.path.exists("data/pois_cleaned.pickle"):
        with open("data/pois_cleaned.pickle", "rb") as f:
            return pickle.load(f)
    if os.path.exists("data/pois.pickle"):
        with open("data/pois.pickle", "rb") as f:
            return pickle.load(f)
    # fallback: allow reading CSV (raw)
    csv_path = "data/travel_poi_data_cleaned.csv"
    if os.path.exists(csv_path):
        # fallback uses loader if available
        try:
            from poi_loader import load_pois_from_csv
            pois = load_pois_from_csv(csv_path)
            return pois
        except Exception:
            # attempt naive CSV read -> create simple POI objects (best-effort)
            df = pd.read_csv(csv_path)
            pois = []
            for i, row in df.iterrows():
                p = PointOfInterest.from_dict(row.to_dict())
                pois.append(p)
            return pois
    raise FileNotFoundError("Không tìm thấy dữ liệu POI (pois_cleaned.pickle / pois.pickle / travel_poi_data_cleaned.csv). Run cleaner first.")


def save_pois_pickle(pois: List[PointOfInterest], out_path: str = "data/pois_ranked.pickle"):
    with open(out_path, "wb") as f:
        pickle.dump(pois, f)
    print(f"[Saved] {out_path} ({len(pois)} POIs)")


def save_pois_csv(pois: List[PointOfInterest], out_path: str = "data/travel_poi_data_ranked.csv"):
    header = [
        "poi_id","name","latitude","longitude","poi_type","category_detail","vibe",
        "avg_cost","simulated_rating","dwell_time_minutes","popularity_score",
        "opening_time","closing_time","time_block_suitability","is_central","recommendation_score"
    ]
    rows = []
    for p in pois:
        rows.append([
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
            "|".join(sorted(list(p.time_block_suitability))) if p.time_block_suitability else "",
            1 if getattr(p, "is_central", False) else 0,
            float(p.recommendation_score) if p.recommendation_score is not None else 0.0
        ])
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[Saved CSV] {out_path} ({len(rows)} rows)")


# ---------------------------
# CLI Entrypoint
# ---------------------------
def main_train():
    print("Ranker: train mode")
    pois = load_pois_from_pickle_or_csv()
    info = train_model(pois, save_model=True)
    print("Train finished. CV info:", info["cv_scores"])


def main_predict_and_save():
    print("Ranker: predict mode")
    pois = load_pois_from_pickle_or_csv()
    pipe = load_model()
    preds = predict_scores_for_pois(pois, pipe)
    update_pois_with_scores(pois, preds)
    # save outputs
    os.makedirs("data", exist_ok=True)
    save_pois_pickle(pois, "data/pois_ranked.pickle")
    save_pois_csv(pois, "data/travel_poi_data_ranked.csv")
    print("[DONE] Predicted and saved ranked POIs.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ranker.py [train|predict]")
        raise SystemExit(1)
    cmd = sys.argv[1].lower()
    if cmd == "train":
        main_train()
    elif cmd == "predict":
        main_predict_and_save()
    else:
        print("Unknown command. Use 'train' or 'predict'.")
