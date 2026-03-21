import os
import json
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pyproj import Transformer

# =========================
# CONFIG
# =========================
MODEL_PATH = "calib_model.joblib"
FEATURES_PATH = "feature_list.joblib"
META_PATH = "model_meta.json"
BASE_POINTS_PATH = "base_points_static.csv"

OUT_DIR = "realtime_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

FORECAST_HOURS_AHEAD = 12
TIMEZONE = "Asia/Bangkok"

CACHE_FILE = os.path.join(OUT_DIR, "openmeteo_cache.json")
CACHE_MAX_AGE_MIN = 30

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HOURLY = "precipitation"

RAIN_WINDOWS = [1, 3, 6, 12, 24]
LAGS = [1, 3, 6]

transformer = Transformer.from_crs("epsg:32648", "epsg:4326", always_xy=True)

# =========================
# HELPERS
# =========================
def now_floor_hour():
    return pd.Timestamp(datetime.now()).floor("H")

def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def should_refresh_cache(cache, max_age_min=CACHE_MAX_AGE_MIN):
    if cache is None:
        return True
    try:
        ts = pd.to_datetime(cache["fetched_at"])
        age_min = (pd.Timestamp.utcnow() - ts).total_seconds() / 60
        return age_min > max_age_min
    except Exception:
        return True

def fetch_openmeteo(center_lat, center_lon):
    today = datetime.now().date()
    start_date = today.isoformat()
    end_date = (today + timedelta(days=2)).isoformat()

    params = {
        "latitude": center_lat,
        "longitude": center_lon,
        "hourly": OPEN_METEO_HOURLY,
        "timezone": TIMEZONE,
        "start_date": start_date,
        "end_date": end_date,
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    times = data["hourly"]["time"]
    rain = data["hourly"]["precipitation"]
    return {"times": times, "rain": rain, "fetched_at": str(pd.Timestamp.utcnow())}

def build_rain_features(rain_df):
    out = rain_df.copy()
    for w in RAIN_WINDOWS:
        out[f"rain_{w}h_sum"] = out["rain"].rolling(w, min_periods=1).sum()
        out[f"rain_{w}h_max"] = out["rain"].rolling(w, min_periods=1).max()
    for lag in LAGS:
        out[f"rain_lag_{lag}h"] = out["rain"].shift(lag).fillna(0)
    return out

# =========================
# LOAD model + features + meta + base
# =========================
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)
meta = load_json(META_PATH, default={})

best_threshold = float(meta.get("best_threshold", 0.6))
lead_hours = int(meta.get("lead_hours", 3))
cell_area = meta.get("cell_area", None)

base = pd.read_csv(BASE_POINTS_PATH)
assert "X" in base.columns and "Y" in base.columns

# Ensure pop/road exist (so app doesn't NaN)
if "population_density" not in base.columns:
    # fallback: try other names
    for c in ["Population_density", "pop_density", "Pop_density"]:
        if c in base.columns:
            base = base.rename(columns={c: "population_density"})
            break

if "road_density" not in base.columns:
    for c in ["Road_Density", "Road_density", "road_density"]:
        if c in base.columns:
            base = base.rename(columns={c: "road_density"})
            break

# Convert X,Y -> lat/lon once
if "lat" not in base.columns or "lon" not in base.columns:
    lon, lat = transformer.transform(base["X"].values, base["Y"].values)
    base["lat"] = lat
    base["lon"] = lon

center_lat = float(base["lat"].mean())
center_lon = float(base["lon"].mean())

print("Center lat/lon:", center_lat, center_lon)
print("Using threshold:", best_threshold)
print("Lead hours:", lead_hours)

# Estimate cell area if missing
if cell_area is None:
    x_diffs = np.diff(np.sort(base["X"].unique()))
    y_diffs = np.diff(np.sort(base["Y"].unique()))
    dx = np.median(x_diffs[x_diffs > 0]) if len(x_diffs) else 30
    dy = np.median(y_diffs[y_diffs > 0]) if len(y_diffs) else 30
    cell_area = float(dx * dy)

# =========================
# CACHE Open-Meteo
# =========================
cache = load_json(CACHE_FILE, default=None)

if should_refresh_cache(cache):
    print("Refreshing Open-Meteo cache...")
    cache = fetch_openmeteo(center_lat, center_lon)
    save_json(CACHE_FILE, cache)
else:
    print("Using cached Open-Meteo data:", cache["fetched_at"])

rain_df = pd.DataFrame({
    "time": pd.to_datetime(cache["times"]),
    "rain": np.array(cache["rain"], dtype=float)
})

# Keep now..now+FORECAST_HOURS_AHEAD
t0 = now_floor_hour()
rain_df = rain_df[rain_df["time"] >= t0].copy()
rain_df = rain_df.iloc[:FORECAST_HOURS_AHEAD+1].copy()
if len(rain_df) == 0:
    raise RuntimeError("Open-Meteo returned no forecast hours after current time.")

rain_feat_df = build_rain_features(rain_df)

# =========================
# PREDICT for each hour
# =========================
pred_rows = []
extent_rows = []

for _, rr in rain_feat_df.iterrows():
    t = rr["time"]
    X = base.copy()

    # attach rain features
    for w in RAIN_WINDOWS:
        X[f"rain_{w}h_sum"] = rr[f"rain_{w}h_sum"]
        X[f"rain_{w}h_max"] = rr[f"rain_{w}h_max"]
    for lag in LAGS:
        X[f"rain_lag_{lag}h"] = rr[f"rain_lag_{lag}h"]

    X["hour"] = t.hour
    X["day"] = t.day
    X["month"] = t.month

    missing = [c for c in features if c not in X.columns]
    if missing:
        raise ValueError(f"Missing features in realtime pipeline: {missing}")

    proba = model.predict_proba(X[features])[:, 1]

    out = X[["X","Y","lat","lon","population_density","road_density"]].copy()
    out["time"] = t
    out["flood_prob"] = proba
    out["risk_class"] = pd.cut(out["flood_prob"], bins=[0,0.3,0.6,1.0], labels=["Low","Medium","High"])
    pred_rows.append(out)

    ext = out[["X","Y","lat","lon","time"]].copy()
    ext["flood_mask"] = (out["flood_prob"] >= best_threshold).astype(int)
    extent_rows.append(ext)

pred_df = pd.concat(pred_rows, ignore_index=True)
extent_df = pd.concat(extent_rows, ignore_index=True)

# =========================
# SAVE outputs
# =========================
prob_out = os.path.join(OUT_DIR, "flood_point_probability_rt.csv")
extent_out = os.path.join(OUT_DIR, "flood_extent_mask_rt.csv")

pred_df.to_csv(prob_out, index=False)
extent_df.to_csv(extent_out, index=False)

meta_rt = {
    "generated_at": str(pd.Timestamp.utcnow()),
    "timezone": TIMEZONE,
    "t0": str(t0),
    "forecast_hours_ahead": FORECAST_HOURS_AHEAD,
    "threshold_used": best_threshold,
    "cell_area_m2": cell_area,
    "cache_fetched_at": cache["fetched_at"],
}
save_json(os.path.join(OUT_DIR, "meta_rt.json"), meta_rt)

print("✅ Saved realtime prob:", prob_out)
print("✅ Saved realtime extent:", extent_out)
print("✅ Saved meta_rt.json")