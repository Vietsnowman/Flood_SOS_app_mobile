"""
2_build_base_points.py
======================
Tạo (hoặc cập nhật) file base_points_static.csv từ dataset điểm-mức
(dataset_mét_độ(2020).csv).

base_points_static.csv là danh sách các điểm lưới (grid points) với
các đặc trưng KHÔNG THAY ĐỔI THEO THỜI GIAN (địa hình, thuỷ văn,
mật độ dân cư, mật độ đường), dùng bởi realtime.py để chạy dự báo
ngập lụt theo thời gian thực.

Dùng Polars lazy scan để đọc file 57 MB hiệu quả.

Chạy:
    uv run python 2_build_base_points.py

Đầu vào:
    dataset_mét_độ(2020).csv   (hoặc nhiều file dataset*.csv ghép lại)

Đầu ra:
    base_points_static.csv
"""

from __future__ import annotations

import pathlib
import polars as pl

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_GLOB = "dataset*.csv"          # tất cả dataset nằm trong thư mục này
OUT_PATH     = "base_points_static.csv"

# Cột tĩnh cần giữ lại (không thay đổi theo thời gian)
STATIC_COLS = [
    "X", "Y",
    "Slope_UTM",
    "TWI_UTM",
    "Flowlog_UT",
    "RiverDist_",
    "RiverDC_UT",
    "DEMC_UTM",
    "population_density",
    "road_density",
    "latitude",
    "longitude",
]

# Mapping tên cột không chuẩn → tên chuẩn
RENAME_MAP = {
    "Road_Density":       "road_density",
    "Population_density": "population_density",
    "pop_density":        "population_density",
    "road_density_":      "road_density",
    "lat":                "latitude",
    "lon":                "longitude",
}

# Cột động (thay đổi theo thời gian) — loại ra khi aggregate
DYNAMIC_COLS = {
    "time", "Is_Flood", "target_flood_future", "sentinel",
    "match_distance_m",
    "temperature_2m (°C)", "rain (mm)",
    "relative_humidity_2m (%)",
    "wind_speed_10m (km/h)", "wind_gusts_10m (km/h)",
}


def load_datasets(glob: str = DATASET_GLOB) -> pl.LazyFrame:
    """Lazy scan tất cả dataset*.csv, rename cột về tên chuẩn, concat."""
    files = sorted(pathlib.Path(".").glob(glob))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file nào khớp với '{glob}'")

    frames = []
    for f in files:
        print(f"  📂 Lazy scan: {f.name} ({f.stat().st_size // 1_000_000} MB)")
        lf = pl.scan_csv(f, infer_schema_length=10_000, try_parse_dates=False)
        # Rename cột không chuẩn
        actual_rename = {k: v for k, v in RENAME_MAP.items() if k in lf.schema.names()}
        if actual_rename:
            lf = lf.rename(actual_rename)
        frames.append(lf)

    return pl.concat(frames, how="diagonal")  # "diagonal" cho phép schema khác nhau


def build_base_points(lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Tìm các cột tĩnh có trong schema, loại cột động, group by (X,Y),
    lấy median để ổn định với outlier.
    """
    all_cols = set(lf.schema.names())

    # Cột cần aggregate (tĩnh, không phải X/Y, không phải dynamic)
    agg_cols = [
        c for c in STATIC_COLS
        if c in all_cols and c not in ("X", "Y") and c not in DYNAMIC_COLS
    ]

    missing = [c for c in STATIC_COLS if c not in all_cols]
    if missing:
        print(f"  ⚠️  Cột không có trong dataset: {missing}")

    print(f"  Cột aggregate ({len(agg_cols)}): {agg_cols}")

    # Build aggregation expressions: lấy median mỗi cột tĩnh
    agg_exprs = [pl.col(c).median().alias(c) for c in agg_cols]

    df = (
        lf.group_by(["X", "Y"])
          .agg(agg_exprs)
          .sort(["X", "Y"])
          .collect(engine="streaming")
    )

    # ── Thêm lat/lon nếu chưa có (convert UTM 48N → WGS84) ──────────────────
    has_lat = "latitude" in df.columns
    has_lon = "longitude" in df.columns

    if not (has_lat and has_lon):
        try:
            from pyproj import Transformer
            tr = Transformer.from_crs("epsg:32648", "epsg:4326", always_xy=True)
            lons_arr, lats_arr = tr.transform(df["X"].to_numpy(), df["Y"].to_numpy())
            df = df.with_columns([
                pl.Series("latitude",  lats_arr),
                pl.Series("longitude", lons_arr),
            ])
            print("  ✅ Convert UTM → WGS84 (latitude, longitude)")
        except ImportError:
            print("  ⚠️  pyproj chưa cài — bỏ qua lat/lon")

    # Thêm cột tắt lat/lon cho realtime.py
    if "latitude" in df.columns:
        df = df.with_columns(pl.col("latitude").alias("lat"))
    if "longitude" in df.columns:
        df = df.with_columns(pl.col("longitude").alias("lon"))

    # Điền null còn lại = 0
    df = df.fill_null(0).fill_nan(0)

    return df


def main():
    print("=" * 60)
    print("  2_build_base_points.py (Polars) — Tạo base_points_static.csv")
    print("=" * 60)

    print("\n📥 Lazy scan dataset(s)...")
    lf = load_datasets()

    print("\n🔧 Build base points (unique X/Y)...")
    base = build_base_points(lf)

    n_rows, n_cols = base.shape
    print(f"\n  Số điểm lưới duy nhất: {n_rows:,}")
    print(f"  Số cột: {n_cols}")
    print(f"  Cột: {base.columns}")

    # Lưu
    base.write_csv(OUT_PATH)
    size_kb = pathlib.Path(OUT_PATH).stat().st_size // 1024
    print(f"\n✅ Đã lưu: {OUT_PATH}  ({n_rows:,} điểm × {n_cols} cột, {size_kb} KB)")


if __name__ == "__main__":
    main()
