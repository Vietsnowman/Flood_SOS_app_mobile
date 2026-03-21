import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, mean_absolute_error
from lightgbm import LGBMClassifier, LGBMRegressor

MERGED_FILE = "cache/merged.csv"

MODEL_PROB_PATH  = "models/model_prob_no_leak.pkl"
MODEL_AREA_PATH  = "models/model_area_no_leak.pkl"
MODEL_RATIO_PATH = "models/model_ratio_no_leak.pkl"

FLOOD_THRESHOLD = 0.001

LEAK_COLS = [
    "flood_area_m2","flood_percent","flood_ratio",
    "y_flood","y_area","y_ratio",
    "sum","len_m_sum","quality_flag",
    "window_start","window_end","s1_images_in_window",
    ".geo","system:index"
]

def get_feature_cols(df):
    admin_drop = [c for c in df.columns if c.startswith("NAME_") or "VARNAME" in c]
    admin_drop += [c for c in df.columns if c.endswith("_flood") or c.endswith("_weather")]
    drop = set(LEAK_COLS + admin_drop + ["date_local","GID_3"])
    cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    return cols

def build_X(df, cols):
    X = df[cols].copy()
    X["month"] = df["date_local"].dt.month
    X["dayofyear"] = df["date_local"].dt.dayofyear
    return X.select_dtypes(include=[np.number])

def main():
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(MERGED_FILE)
    df["date_local"] = pd.to_datetime(df["date_local"])

    df["y_flood"] = (df["flood_ratio"] > FLOOD_THRESHOLD).astype(int)
    df["y_area"]  = np.log1p(df["flood_area_m2"])
    df["y_ratio"] = np.log1p(df["flood_ratio"])

    feature_cols = get_feature_cols(df)
    X = build_X(df, feature_cols)

    y_prob = df["y_flood"].values
    y_area = df["y_area"].values
    y_ratio= df["y_ratio"].values
    groups = df["GID_3"].values

    gkf = GroupKFold(n_splits=5)

    clf = LGBMClassifier(
        n_estimators=1200, learning_rate=0.03, num_leaves=64,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    reg_area = LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=64,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    reg_ratio = LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=64,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )

    aucs, maes_area, maes_ratio = [], [], []
    for tr, va in gkf.split(X, y_prob, groups):
        Xtr, Xva = X.iloc[tr], X.iloc[va]

        clf.fit(Xtr, y_prob[tr])
        pva = clf.predict_proba(Xva)[:,1]
        aucs.append(roc_auc_score(y_prob[va], pva))

        reg_area.fit(Xtr, y_area[tr])
        maes_area.append(mean_absolute_error(y_area[va], reg_area.predict(Xva)))

        reg_ratio.fit(Xtr, y_ratio[tr])
        maes_ratio.append(mean_absolute_error(y_ratio[va], reg_ratio.predict(Xva)))

    print("✅ CV AUC:", np.mean(aucs))
    print("✅ CV MAE(log_area):", np.mean(maes_area))
    print("✅ CV MAE(log_ratio):", np.mean(maes_ratio))

    # fit full
    clf.fit(X, y_prob)
    reg_area.fit(X, y_area)
    reg_ratio.fit(X, y_ratio)

    joblib.dump({"model": clf, "feature_cols": X.columns.tolist()}, MODEL_PROB_PATH)
    joblib.dump({"model": reg_area, "feature_cols": X.columns.tolist()}, MODEL_AREA_PATH)
    joblib.dump({"model": reg_ratio, "feature_cols": X.columns.tolist()}, MODEL_RATIO_PATH)

    print("✅ Saved models (NO LEAK)")

if __name__ == "__main__":
    main()