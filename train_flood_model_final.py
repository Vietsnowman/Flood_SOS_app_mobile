# =========================
# 0) IMPORTS + CONFIG
# =========================
import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from scipy.ndimage import binary_opening, binary_closing
import joblib

# =========================
# CONFIG
# =========================
DATA_PATH = "dataset_mét_độ(2020).csv"

LEAD_HOURS = 3
RAIN_WINDOWS = [1, 3, 6, 12, 24]
PROB_THRESHOLD = 0.6

USE_SENTINEL_EXTENT = False
SENTINEL_THRESHOLD = 0.5
SMOOTH_EXTENT = True

# Spatial + time split
BLOCK_MULTIPLIER = 30
TEST_BLOCK_RATIO = 0.30
RANDOM_SEED = 42

# Purge
PURGE_HOURS = LEAD_HOURS


# =========================
# 1) LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

print("Columns:", df.columns.tolist())
print(df.head())
print("Shape:", df.shape)

# =========================
# 2) BASIC CLEANING + TIME PARSE
# =========================
if "time" not in df.columns:
    raise ValueError("CSV must contain column 'time'")

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values(["X", "Y", "time"]).reset_index(drop=True)

if "Is_Flood" not in df.columns:
    raise ValueError("CSV must contain column 'Is_Flood' as label")

# =========================
# 2b) STANDARDIZE POP/ROAD COLUMNS
# =========================
# Convert various naming formats to standard:
# population_density, road_density
rename_map = {}

for c in df.columns:
    cl = c.lower().strip()

    # population
    if cl in ["population_density", "pop_density", "popdensity", "populationdensity"]:
        rename_map[c] = "population_density"

    # road
    if cl in ["road_density", "road_density_", "roaddensity", "road_density(1)"]:
        rename_map[c] = "road_density"

    # Special case from your dataset:
    if c == "Road_Density":
        rename_map[c] = "road_density"

df = df.rename(columns=rename_map)

print("\n✅ Standardized columns mapping:", rename_map)
print("✅ population_density exists?", "population_density" in df.columns)
print("✅ road_density exists?", "road_density" in df.columns)

# =========================
# 3) FEATURE ENGINEERING
# =========================
rain_candidates = [c for c in df.columns if "rain" in c.lower()]
print("Rain candidates:", rain_candidates)

RAIN_COL = rain_candidates[0] if len(rain_candidates) > 0 else None
if RAIN_COL is None:
    raise ValueError("No rainfall column detected.")

print("Using RAIN_COL =", RAIN_COL)

group_cols = ["X", "Y"]

# Rolling rain
for w in RAIN_WINDOWS:
    df[f"rain_{w}h_sum"] = df.groupby(group_cols)[RAIN_COL].transform(
        lambda s: s.rolling(window=w, min_periods=1).sum()
    )
    df[f"rain_{w}h_max"] = df.groupby(group_cols)[RAIN_COL].transform(
        lambda s: s.rolling(window=w, min_periods=1).max()
    )

# Lag rain
for lag in [1, 3, 6]:
    df[f"rain_lag_{lag}h"] = df.groupby(group_cols)[RAIN_COL].shift(lag)

# Time features
df["hour"] = df["time"].dt.hour
df["day"] = df["time"].dt.day
df["month"] = df["time"].dt.month

df = df.fillna(0)

# =========================
# 4) FORECAST TARGET
# =========================
df["target_flood_future"] = df.groupby(group_cols)["Is_Flood"].shift(-LEAD_HOURS)
df = df.dropna(subset=["target_flood_future"]).copy()
df["target_flood_future"] = df["target_flood_future"].astype(int)

print("After lead target:", df.shape)
print("Positive ratio:", df["target_flood_future"].mean())

# =========================
# 5) BUILD FEATURE LIST
# =========================
all_cols = df.columns.tolist()

terrain_hydro_candidates = [
    "Slope_UTM", "TWI_UTM", "Flowlog_UT", "RiverDist_", "RiverDC_UT",
    "DEMC_UTM", "RiverDist_UTM", "RiverDC_UTM", "FlowAccum", "DEM"
]

# after standardize -> population_density, road_density
exposure_candidates = ["population_density", "road_density"]

sentinel_candidates = [c for c in all_cols if "sentinel" in c.lower()]
rain_features = [c for c in all_cols if c.startswith("rain_")]
time_features = ["hour", "day", "month"]

terrain_hydro_features = [c for c in terrain_hydro_candidates if c in all_cols]
exposure_features = [c for c in exposure_candidates if c in all_cols]
sentinel_features = sentinel_candidates

features = rain_features + terrain_hydro_features + exposure_features + time_features

print("Terrain/hydro features:", terrain_hydro_features)
print("Exposure features:", exposure_features)
print("Sentinel features:", sentinel_features)
print("Total features:", len(features))

# =========================
# 6) SPATIAL BLOCK SPLIT + TIME SPLIT + PURGE
# =========================
x_diffs = np.diff(np.sort(df["X"].unique()))
y_diffs = np.diff(np.sort(df["Y"].unique()))
dx = np.median(x_diffs[x_diffs > 0]) if len(x_diffs) else 30
dy = np.median(y_diffs[y_diffs > 0]) if len(y_diffs) else 30
median_cell = float(np.median([dx, dy]))
cell_area = dx * dy

BLOCK_SIZE = median_cell * BLOCK_MULTIPLIER

df["block_x"] = np.floor(df["X"] / BLOCK_SIZE).astype(int)
df["block_y"] = np.floor(df["Y"] / BLOCK_SIZE).astype(int)
df["block_id"] = df["block_x"].astype(str) + "_" + df["block_y"].astype(str)

blocks = df["block_id"].drop_duplicates()
train_blocks, test_blocks = train_test_split(blocks, test_size=TEST_BLOCK_RATIO, random_state=RANDOM_SEED)

unique_times = df["time"].sort_values().unique()
cut_time = unique_times[int(len(unique_times) * 0.7)]
purge_cut_time = cut_time + pd.Timedelta(hours=LEAD_HOURS)

train_df = df[(df["block_id"].isin(train_blocks)) & (df["time"] <= cut_time)].copy()
test_df  = df[(df["block_id"].isin(test_blocks))  & (df["time"] > purge_cut_time)].copy()

print("\n=== Spatial Block + Time Split ===")
print("BLOCK_SIZE:", BLOCK_SIZE)
print("cut_time:", cut_time)
print("purge_cut_time:", purge_cut_time)
print("Train:", train_df.shape, " Test:", test_df.shape)

if len(test_df) == 0:
    raise ValueError("❌ Test set is empty after split+purge. Reduce PURGE_HOURS or adjust cut_time/block_size.")

X_train = train_df[features]
y_train = train_df["target_flood_future"]
X_test  = test_df[features]
y_test  = test_df["target_flood_future"]

# =========================
# 7) TRAIN MODEL + CALIBRATION
# =========================
if lgb is None:
    raise ImportError("LightGBM not installed. Install: pip install lightgbm")

pos_rate = y_train.mean()
scale_pos_weight = (1 - pos_rate) / (pos_rate + 1e-9)

print("\nTrain positive ratio:", pos_rate)
print("scale_pos_weight:", scale_pos_weight)

model = lgb.LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    scale_pos_weight=scale_pos_weight,
    min_data_in_leaf=300,
    reg_lambda=5.0,
    reg_alpha=0.0
)

model.fit(X_train, y_train)

calib_model = CalibratedClassifierCV(model, method="isotonic", cv=3)
calib_model.fit(X_train, y_train)

# =========================
# 8) EVALUATION
# =========================
pred_proba = calib_model.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, pred_proba)
pr_auc = average_precision_score(y_test, pred_proba)

print("\n==== Evaluation (Spatial Block + Purge) ====")
print("ROC-AUC:", roc)
print("PR-AUC:", pr_auc)

prec, rec, thr = precision_recall_curve(y_test, pred_proba)
f1 = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = np.argmax(f1)
best_threshold = thr[best_idx] if best_idx < len(thr) else PROB_THRESHOLD
print("Best threshold by F1:", best_threshold)

# =========================
# 9) PREDICT FOR ALL DATA
# =========================
df["flood_prob"] = calib_model.predict_proba(df[features])[:, 1]
df["risk_class"] = pd.cut(df["flood_prob"], bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])

# =========================
# 10) BUILD FLOOD EXTENT MASK
# =========================
if USE_SENTINEL_EXTENT and len(sentinel_features) > 0:
    sent_col = sentinel_features[0]
    df["flood_mask"] = (df[sent_col] > SENTINEL_THRESHOLD).astype(int)
else:
    df["flood_mask"] = (df["flood_prob"] > best_threshold).astype(int)

if SMOOTH_EXTENT:
    print("Smoothing extent using morphological filters (per timestamp).")
    for t, dft in df.groupby("time"):
        xs = np.sort(dft["X"].unique())
        ys = np.sort(dft["Y"].unique())
        if len(xs) * len(ys) != len(dft):
            continue

        grid = dft.pivot(index="Y", columns="X", values="flood_mask").values.astype(bool)
        grid2 = binary_opening(grid)
        grid3 = binary_closing(grid2)

        smoothed = grid3.astype(int).reshape(-1)
        dft_sorted = dft.sort_values(["Y", "X"])
        df.loc[dft_sorted.index, "flood_mask"] = smoothed
    print("Smoothing done.")

# =========================
# 11) FLOODED AREA BY TIME
# =========================
df["cell_area_m2"] = cell_area
flooded_area_by_time = df.groupby("time").apply(
    lambda x: (x["flood_mask"] * x["cell_area_m2"]).sum()
).reset_index(name="flooded_area_m2")

total_area = df.groupby("time")["cell_area_m2"].sum().median()
flooded_area_by_time["flooded_area_pct"] = flooded_area_by_time["flooded_area_m2"] / total_area

# =========================
# 12) IMPACT COMPUTATION (EVENT LEVEL)
# =========================
pop_col = "population_density" if "population_density" in df.columns else None
road_col = "road_density" if "road_density" in df.columns else None

event_mask = df.groupby(["X", "Y"])["flood_mask"].max().reset_index(name="flood_event_mask")
base_points = df.sort_values("time").groupby(["X", "Y"]).tail(1).copy()
base_points = base_points.merge(event_mask, on=["X", "Y"], how="left")
base_points["cell_area_m2"] = cell_area

flooded_area_event = (base_points["flood_event_mask"] * base_points["cell_area_m2"]).sum()
total_area_event = base_points["cell_area_m2"].sum()
flooded_area_pct_event = flooded_area_event / total_area_event

if pop_col:
    base_points["pop_cell"] = base_points[pop_col] * base_points["cell_area_m2"]
    pop_total = base_points["pop_cell"].sum()
    pop_exposed = (base_points["pop_cell"] * base_points["flood_event_mask"]).sum()
    pop_exposed_pct = pop_exposed / (pop_total + 1e-9)
else:
    pop_total = pop_exposed = pop_exposed_pct = 0.0

if road_col:
    base_points["road_cell"] = base_points[road_col] * base_points["cell_area_m2"]
    road_total = base_points["road_cell"].sum()
    road_exposed = (base_points["road_cell"] * base_points["flood_event_mask"]).sum()
    road_exposed_pct = road_exposed / (road_total + 1e-9)
else:
    road_total = road_exposed = road_exposed_pct = 0.0

A = flooded_area_pct_event
P = pop_exposed_pct
R = road_exposed_pct

impact_score = 100 * (0.35 * P + 0.25 * R + 0.15 * A)

def priority_level(score):
    if score < 25:
        return "Low"
    elif score < 50:
        return "Medium"
    elif score < 75:
        return "High"
    else:
        return "Emergency"

priority = priority_level(impact_score)

impact_table = pd.DataFrame([{
    "flooded_area_m2": flooded_area_event,
    "flooded_area_pct": flooded_area_pct_event,
    "pop_total": pop_total,
    "pop_exposed": pop_exposed,
    "pop_exposed_pct": pop_exposed_pct,
    "road_total_proxy": road_total,
    "road_exposed_proxy": road_exposed,
    "road_exposed_pct": road_exposed_pct,
    "impact_score": impact_score,
    "priority_level": priority,
    "lead_hours": LEAD_HOURS,
    "prob_threshold_used": float(best_threshold),
    "block_size_used": float(BLOCK_SIZE)
}])

print("\n==== Impact Table (Event Level) ====")
print(impact_table)

# =========================
# 13) EXPORT PRODUCTS
# =========================
OUT_DIR = "/mnt/data/flood_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

df[["X", "Y", "time", "flood_prob", "risk_class"]].to_csv(os.path.join(OUT_DIR, "flood_point_probability.csv"), index=False)
df[["X", "Y", "time", "flood_mask"]].to_csv(os.path.join(OUT_DIR, "flood_extent_mask.csv"), index=False)
impact_table.to_csv(os.path.join(OUT_DIR, "impact_event_table.csv"), index=False)
flooded_area_by_time.to_csv(os.path.join(OUT_DIR, "flooded_area_by_time.csv"), index=False)

print("\nSaved outputs to:", OUT_DIR)

# =========================
# 14) SAVE MODEL + FEATURES + STATIC + META
# =========================
joblib.dump(calib_model, "calib_model.joblib")
joblib.dump(features, "feature_list.joblib")
print("✅ Saved: calib_model.joblib, feature_list.joblib")

# SAVE STATIC BASE POINTS (include pop & road)
static_cols = ["X","Y"] + terrain_hydro_features + exposure_features
base_points[static_cols].to_csv("base_points_static.csv", index=False)
print("✅ Saved: base_points_static.csv")

# SAVE META (include best_threshold)
meta = {
    "epsg": "32648",
    "dx": float(dx),
    "dy": float(dy),
    "cell_area": float(cell_area),
    "lead_hours": int(LEAD_HOURS),
    "best_threshold": float(best_threshold),
    "block_size": float(BLOCK_SIZE),
    "features": features
}
with open("model_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("✅ Saved: model_meta.json (with best_threshold)")

