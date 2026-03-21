"""
export_sos_dataset.py
=====================
Xuất dataset từ MongoDB collection `soshistories` ra file CSV để train
model PyTorch phân loại mức độ khẩn cấp của SOS (urgent_15m).

Cột đầu ra:
  lat, lon, flood_prob_near, people_count, status_enc,
  hour_of_day, is_night, is_weekend,
  delta_min, urgent_15m (label)

Chạy: python export_sos_dataset.py
Output: data/sos_priority_dataset.csv
"""

import os
import sys
import pandas as pd
from pymongo import MongoClient

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MONGO_URI   = "mongodb://127.0.0.1:27017/floodsos"
COLLECTION  = "soshistories"
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sos_priority_dataset.csv")

# Chỉ lấy bản ghi đã resolved hoặc deleted (có nhãn thời gian rõ ràng).
VALID_ACTIONS = {"resolve", "delete"}

# Ngưỡng khẩn cấp (phút).
URGENT_THR_MIN = 15.0

STATUS_ENC = {
    "open":        0,
    "warning":     0,   # alias cho "open"
    "in_progress": 1,
    "closed":      2,
    "false_alarm": 2,
}
# ──────────────────────────────────────────────────────────────────────────────


def main():
    # 1. Kết nối Mongo.
    try:
        client  = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        db      = client.get_default_database()
        col     = db[COLLECTION]
    except Exception as e:
        print(f"[ERROR] Không thể kết nối MongoDB: {e}")
        sys.exit(1)

    # 2. Lấy toàn bộ bản ghi trong SOSHistory.
    docs = list(col.find({}, {
        "lat":            1,
        "lon":            1,
        "flood_prob_near":1,
        "people_count":   1,
        "status":         1,
        "created_at":     1,
        "resolved_at":    1,
        "resolved_action":1,
    }))

    if not docs:
        print("[WARN] Collection soshistories trống — chưa có SOS nào được resolve/delete.")
        sys.exit(0)

    print(f"[INFO] Đọc được {len(docs)} bản ghi từ MongoDB.")

    rows = []
    skipped_no_action = 0
    skipped_no_time   = 0
    skipped_no_coord  = 0

    for d in docs:
        action = (d.get("resolved_action") or "").strip().lower()

        # Chỉ giữ bản ghi có hành động rõ ràng.
        if action not in VALID_ACTIONS:
            skipped_no_action += 1
            continue

        created_at  = d.get("created_at")
        resolved_at = d.get("resolved_at")
        if not created_at or not resolved_at:
            skipped_no_time += 1
            continue

        lat = d.get("lat")
        lon = d.get("lon")
        if lat is None or lon is None:
            skipped_no_coord += 1
            continue

        try:
            delta_min = (resolved_at - created_at).total_seconds() / 60.0
        except Exception:
            skipped_no_time += 1
            continue

        # ── Label ─────────────────────────────────────────────────────────────
        # urgent_15m = 1 nếu xử lý trong ≤ 15 phút (cả resolve lẫn delete).
        urgent_15m = 1 if delta_min <= URGENT_THR_MIN else 0

        # ── Features ──────────────────────────────────────────────────────────
        flood_prob_near = d.get("flood_prob_near")  # None nếu chưa archived

        raw_count = d.get("people_count", "1") or "1"
        try:
            people_count = float(str(raw_count).strip())
        except ValueError:
            people_count = 1.0

        raw_status  = (d.get("status") or "open").strip().lower()
        status_enc  = STATUS_ENC.get(raw_status, 0)

        # ── Temporal features ──────────────────────────────────────────
        # Dùng created_at (giờ xảy ra sự kiện) làm gốc temporal.
        c_hour    = created_at.hour
        c_weekday = created_at.weekday()  # 0=Mon, 5=Sat, 6=Sun
        is_night   = 1 if (c_hour >= 22 or c_hour < 6) else 0
        is_weekend = 1 if c_weekday >= 5 else 0

        rows.append({
            "lat":             lat,
            "lon":             lon,
            "flood_prob_near": flood_prob_near,  # có thể NaN → sẽ xử lý bên dưới
            "people_count":    people_count,
            "status_enc":      status_enc,
            "hour_of_day":     c_hour,
            "is_night":        is_night,
            "is_weekend":      is_weekend,
            "delta_min":       delta_min,
            "urgent_15m":      urgent_15m,
        })

    print(f"[INFO] Bỏ qua {skipped_no_action} bản ghi không có action rõ ràng.")
    print(f"[INFO] Bỏ qua {skipped_no_time} bản ghi thiếu timestamp.")
    print(f"[INFO] Bỏ qua {skipped_no_coord} bản ghi thiếu tọa độ.")

    if not rows:
        print("[WARN] Không có bản ghi hợp lệ nào. Dataset trống.")
        sys.exit(0)

    df = pd.DataFrame(rows)

    # Với các bản ghi chưa có flood_prob_near (recorded trước lần nâng server),
    # điền median của các bản ghi có sẵn. Nếu toàn bộ đều null → điền 0.
    n_missing_fpn = df["flood_prob_near"].isna().sum()
    if n_missing_fpn > 0:
        median_fpn = df["flood_prob_near"].median()
        fill_val   = median_fpn if pd.notna(median_fpn) else 0.0
        df["flood_prob_near"] = df["flood_prob_near"].fillna(fill_val)
        print(f"[INFO] Điền flood_prob_near = {fill_val:.4f} cho {n_missing_fpn} bản ghi thiếu.")

    # Chắc chắn không còn NaN.
    df = df.fillna(0)

    # ── Thống kê ──────────────────────────────────────────────────────────────
    n_total   = len(df)
    n_urgent  = int(df["urgent_15m"].sum())
    n_normal  = n_total - n_urgent
    ratio     = n_urgent / n_total * 100

    print(f"\n[INFO] Dataset tổng: {n_total} bản ghi")
    print(f"       Urgent (≤15m): {n_urgent}  ({ratio:.1f}%)")
    print(f"       Normal  (>15m): {n_normal}  ({100-ratio:.1f}%)")
    print(f"       flood_prob_near — mean: {df['flood_prob_near'].mean():.4f}, "
          f"std: {df['flood_prob_near'].std():.4f}")

    # ── Lưu file ──────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\n[OK] Dataset đã lưu tại: {OUTPUT_PATH}")

    client.close()


if __name__ == "__main__":
    main()
