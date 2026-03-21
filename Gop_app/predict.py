import numpy as np
import pandas as pd
import joblib

from open_meteo import fetch_open_meteo_daily
from feature_engineering import build_lag_rolling_features, attach_static_features, add_time_features

MODEL_PROB_PATH  = "models/model_prob_no_leak.pkl"
MODEL_AREA_PATH  = "models/model_area_no_leak.pkl"
MODEL_RATIO_PATH = "models/model_ratio_no_leak.pkl"

def normalize(x):
    x = np.array(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

def load_models():
    prob_pack = joblib.load(MODEL_PROB_PATH)
    area_pack = joblib.load(MODEL_AREA_PATH)
    ratio_pack= joblib.load(MODEL_RATIO_PATH)
    return prob_pack, area_pack, ratio_pack

def predict_7days(gid, lat, lon, static_row, history_df, population=None):
    prob_pack, area_pack, ratio_pack = load_models()
    feat_cols = prob_pack["feature_cols"]

    forecast_df = fetch_open_meteo_daily(lat, lon)

    hist_need = history_df[[
        "GID_3","date_local",
        "precipitation_sum","precipitation_hours","precipitation_max_1h"
    ]].copy()

    f_need = forecast_df[[
        "date_local",
        "precipitation_sum","precipitation_hours","precipitation_max_1h"
    ]].copy()
    f_need["GID_3"] = gid

    combo = hist_need._append(f_need, ignore_index=True)
    combo["date_local"] = pd.to_datetime(combo["date_local"])
    combo = build_lag_rolling_features(combo)

    feat_forecast = combo[combo["date_local"].isin(forecast_df["date_local"])].copy()
    feat_forecast = feat_forecast.merge(forecast_df, on="date_local", how="left")

    X = attach_static_features(feat_forecast, static_row)
    X = add_time_features(X)

    # numeric only
    Xnum = X.select_dtypes(include=[np.number]).copy()

    for c in feat_cols:
        if c not in Xnum.columns:
            Xnum[c] = 0.0
    Xnum = Xnum[feat_cols]

    prob = prob_pack["model"].predict_proba(Xnum)[:,1]
    log_area  = area_pack["model"].predict(Xnum)
    log_ratio = ratio_pack["model"].predict(Xnum)

    out = forecast_df[["date_local"]].copy()
    out["p_flood"] = prob
    out["flood_area_m2"] = np.expm1(log_area)
    out["flood_ratio"] = np.expm1(log_ratio)

    if population is not None and not np.isnan(population):
        out["population"] = population
        out["affected_population"] = out["flood_ratio"] * population
    else:
        out["population"] = np.nan
        out["affected_population"] = np.nan

    ap_norm = normalize(out["affected_population"].fillna(0))
    ratio_norm = normalize(out["flood_ratio"])
    out["priority_score"] = 100*(0.5*out["p_flood"] + 0.3*ap_norm + 0.2*ratio_norm)

    def level(s):
        if s >= 80:
            return "KHẨN CẤP"
        if s >= 50:
            return "CAO"
        if s >= 20:
            return "TRUNG BÌNH"
        return "THẤP"
    out["priority_level"] = out["priority_score"].apply(level)

    return out

# import os
# import joblib
# import numpy as np
# import pandas as pd
# from open_meteo import fetch_open_meteo_daily

# MODEL_PROB_PATH  = "models/model_prob.joblib"
# MODEL_AREA_PATH  = "models/model_area.joblib"
# MODEL_RATIO_PATH = "models/model_ratio.joblib"

# # ✅ cache model in memory
# _MODEL_CACHE = None

# def load_models():
#     global _MODEL_CACHE
#     if _MODEL_CACHE is not None:
#         return _MODEL_CACHE

#     # ✅ check file exist
#     for p in [MODEL_PROB_PATH, MODEL_AREA_PATH, MODEL_RATIO_PATH]:
#         if not os.path.exists(p):
#             raise FileNotFoundError(f"❌ Model file not found: {p}")

#         if os.path.getsize(p) < 5000:
#             raise RuntimeError(f"❌ Model file seems corrupted (too small): {p}")

#     try:
#         prob_pack  = joblib.load(MODEL_PROB_PATH)
#         area_pack  = joblib.load(MODEL_AREA_PATH)
#         ratio_pack = joblib.load(MODEL_RATIO_PATH)
#     except Exception as e:
#         raise RuntimeError(
#             "❌ Cannot load model joblib files.\n"
#             "Possible causes:\n"
#             "  - model file corrupted or incomplete\n"
#             "  - python/sklearn/lightgbm version mismatch\n"
#             "Fix:\n"
#             "  - re-download/retrain model using current env\n"
#             "  - use python 3.10/3.11 + matching versions\n"
#             f"\nOriginal error: {e}"
#         )

#     _MODEL_CACHE = (prob_pack, area_pack, ratio_pack)
#     return _MODEL_CACHE


# def predict_7days(gid, lat, lon, static_row, history_df, population=None):
#     prob_pack, area_pack, ratio_pack = load_models()

#     # forecast 7 ngày
#     forecast_df = fetch_open_meteo_daily(lat, lon)

#     # combine history + forecast để rolling liên tục
#     hist_need = history_df[["date_local","tp_sum_3d","tp_sum_7d","tp_sum_14d","tp_max_3d","tp_max_7d","tp_max_14d"]].copy()
#     hist_need["date_local"] = pd.to_datetime(hist_need["date_local"], errors="coerce")
#     hist_need = hist_need.dropna(subset=["date_local"])

#     forecast_df["date_local"] = pd.to_datetime(forecast_df["date_local"], errors="coerce")

#     # -------- build feature table -------
#     feat = forecast_df.copy()

#     # thêm static features
#     for c in prob_pack["feature_cols"]:
#         if c not in feat.columns and c in static_row.index:
#             feat[c] = static_row[c]

#     # đảm bảo đúng cột theo model
#     X = feat.reindex(columns=prob_pack["feature_cols"], fill_value=0)

#     # predict probability
#     p = prob_pack["model"].predict_proba(X)[:,1]
#     feat["p_flood"] = p

#     # predict ratio + area
#     Xr = feat.reindex(columns=ratio_pack["feature_cols"], fill_value=0)
#     Xa = feat.reindex(columns=area_pack["feature_cols"], fill_value=0)

#     feat["flood_ratio"] = np.clip(np.expm1(ratio_pack["model"].predict(Xr)), 0, 1)
#     feat["flood_area_m2"] = np.clip(np.expm1(area_pack["model"].predict(Xa)), 0, None)

#     # affected population
#     if population is None:
#         population = static_row.get("DanSo_sum", 0)

#     feat["affected_population"] = feat["flood_ratio"] * float(population)

#     # priority score
#     feat["priority_score"] = np.clip(
#         feat["p_flood"]*50 + feat["flood_ratio"]*30 + (feat["affected_population"]/1000)*20,
#         0, 100
#     )

#     def level(score):
#         if score >= 80: return "CRITICAL"
#         if score >= 50: return "HIGH"
#         if score >= 20: return "MEDIUM"
#         return "LOW"

#     feat["priority_level"] = feat["priority_score"].apply(level)

#     out = feat[[
#         "date_local","p_flood","flood_area_m2","flood_ratio",
#         "affected_population","priority_score","priority_level"
#     ]].copy()

#     return out