import pandas as pd
import os
from datetime import datetime
from predict import predict_7days

MERGED_FILE = "cache/merged.csv"
POP_FILE = "data/DanSo_Xa.csv"
OUT_DIR = "cache/forecasts"

def run_daily_cache():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(MERGED_FILE)
    df["date_local"] = pd.to_datetime(df["date_local"])

    pop = pd.read_csv(POP_FILE).set_index("GID_3")["DanSo_sum"].to_dict()

    static = df.sort_values("date_local").groupby("GID_3").head(1).set_index("GID_3")

    results = []
    for gid in static.index:
        row = static.loc[gid]
        lat = float(row["commune_lat"])
        lon = float(row["commune_lon"])
        population = pop.get(gid, None)
        hist = df[df["GID_3"] == gid].sort_values("date_local").tail(60)

        pred = predict_7days(gid, lat, lon, row, hist, population=population)
        pred["GID_3"] = gid
        pred["name"] = row.get("NAME_3_flood", gid)
        results.append(pred)

    out = pd.concat(results, ignore_index=True)

    today = datetime.now().strftime("%Y-%m-%d")
    out_path = f"{OUT_DIR}/{today}_commune_forecasts.parquet"
    out.to_parquet(out_path, index=False)

    print("✅ Saved cache:", out_path, "rows:", len(out))

if __name__ == "__main__":
    run_daily_cache()