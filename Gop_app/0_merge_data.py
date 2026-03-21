"""
0_merge_data.py
===============
Gộp 3 file CSV thời tiết/lũ Nghệ An (2023–2025) thành 1 file cache.

Dùng Polars lazy scan để đọc song song và xử lý hiệu quả với file lớn (~340 MB mỗi file).
Output: cache/merged.csv — dùng bởi 1_train_models_no_leak.py và scheduler_cache.py.

Chạy:
    uv run python 0_merge_data.py
"""

import os
import polars as pl

FILE_2023 = "data/NgheAn_weather32full_flood_merge_by_commune_time_2023_NO_NaN.csv"
FILE_2024 = "data/NgheAn_weather32full_flood_merge_by_commune_time_2024_NO_NaN.csv"
FILE_2025 = "data/NgheAn_weather32full_flood_merge_by_commune_time_2025_NO_NaN.csv"

OUTPUT = "cache/merged.csv"


def main():
    os.makedirs("cache", exist_ok=True)

    # ── Lazy scan tất cả 3 file — Polars đọc parallel, chỉ execute khi collect()
    lf2023 = pl.scan_csv(FILE_2023, infer_schema_length=10_000, try_parse_dates=True)
    lf2024 = pl.scan_csv(FILE_2024, infer_schema_length=10_000, try_parse_dates=True)
    lf2025 = pl.scan_csv(FILE_2025, infer_schema_length=10_000, try_parse_dates=True)

    # ── Tìm cột chung giữa 3 file (lấy schema mà không đọc toàn bộ dữ liệu)
    cols_2023 = set(lf2023.schema.names())
    cols_2024 = set(lf2024.schema.names())
    cols_2025 = set(lf2025.schema.names())
    common = sorted(cols_2023 & cols_2024 & cols_2025)

    print(f"📋 Số cột chung: {len(common)}")

    # ── Select cột chung + concat lazy (không tốn RAM cho đến khi collect)
    df = pl.concat([
        lf2023.select(common),
        lf2024.select(common),
        lf2025.select(common),
    ]).sort(["date_local", "GID_3"]).collect(engine="streaming")

    # ── Lưu kết quả
    df.write_csv(OUTPUT)

    print(f"✅ merged saved: {OUTPUT}  {df.shape}")


if __name__ == "__main__":
    main()