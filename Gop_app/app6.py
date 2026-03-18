import os, sys, json, math
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

# # -------------------------
# # CHUẨN HOÁ PATH THEO PROJECT ROOT
# # -------------------------
SRC_DIR = Path(__file__).resolve().parent            # .../project/src
PROJECT_DIR = SRC_DIR.parent                        # .../project

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import predict_7days
from db_utils import (
    init_db, get_incident, upsert_incident,
    get_all_incidents, upsert_allocation, get_all_allocations,
    create_request, list_requests, update_request_status,
    create_task, list_tasks, update_task,
    upsert_shelter_status, list_shelter_status,
    create_shelter_move, list_shelter_moves
)
from report_utils import export_csv, export_pdf

# # -------------------------
# # FILE PATHS (relative to PROJECT_DIR)
# # -------------------------
MERGED_FILE = PROJECT_DIR / "cache" / "merged.csv"
POP_FILE = PROJECT_DIR / "data" / "DanSo_Xa.csv"
SHELTER_FILE = PROJECT_DIR / "data" / "shelters.csv"
CACHE_DIR = PROJECT_DIR / "cache" / "forecasts"

(CACHE_DIR.parent).mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_DIR / "data").mkdir(parents=True, exist_ok=True)

# # --------- THIẾT LẬP ----------
ALERT_THRESH_P = 0.7
ALERT_THRESH_AFFECTED = 500
ALERT_THRESH_PRIORITY = 80

# # =========================
# # TỪ ĐIỂN GIAO DIỆN (VI) — 100% VIỆT HOÁ
# # =========================
VI = {
    "app_title": "🌊 Ứng dụng hỗ trợ cứu hộ ngập lụt — Nghệ An (Thời gian thực + Điều hành)",
    "sidebar_panel": "⚙️ Bảng điều khiển",
    "use_default_thresholds": "Dùng ngưỡng cảnh báo mặc định",
    "alert_p": "Cảnh báo: Xác suất ngập p_flood ≥",
    "alert_affected": "Cảnh báo: Dân bị ảnh hưởng ≥",
    "alert_priority": "Cảnh báo: Điểm ưu tiên ≥",

    "total_resources": "🚤 Tổng nguồn lực (phân bổ)",
    "teams": "Đội cứu hộ",
    "boats": "Thuyền",
    "trucks": "Xe tải",
    "food": "Gói lương thực",
    "water": "Lít nước",
    "med": "Bộ y tế",

    "alert_center": "🚨 Trung tâm cảnh báo (tự động)",
    "alerts_count": "Số cảnh báo",
    "max_priority": "Ưu tiên cao nhất",
    "max_affected": "Dân ảnh hưởng cao nhất",
    "alerts_table_title": "### 📋 Danh sách cảnh báo (tối đa 50)",

    "priority_map": "🗺️ Bản đồ ưu tiên (Hôm nay) — Bấm xã để xem dự báo 7 ngày + điều hành cứu hộ",
    "tooltip_name": "Xã:",
    "tooltip_priority": "Ưu tiên:",

    "top20": "## 📌 Top 20 xã ưu tiên (hôm nay)",
    "export_report": "## 🧾 Xuất báo cáo (CSV/PDF)",
    "export_csv_btn": "Xuất CSV (Top 50)",
    "export_pdf_btn": "Xuất PDF (Top 20 + Tóm tắt)",
    "csv_saved": "✅ Đã lưu CSV:",
    "pdf_saved": "✅ Đã lưu PDF:",

    "using_cache": "✅ Đang dùng dự báo cache (scheduler) cho hôm nay.",
    "no_cache": "⚠️ Không có cache. App đang tính dự báo TRỰC TIẾP (chậm, có thể bị giới hạn). Hãy chạy scheduler_cache.py mỗi ngày!",

    "selected": "✅ Đã chọn:",
    "today_overview": "### 📊 Tổng quan hôm nay",
    "p_flood": "Xác suất ngập",
    "area_m2": "Diện tích ngập (m²)",
    "ratio": "Tỉ lệ ngập",
    "affected": "Dân bị ảnh hưởng",
    "priority": "Ưu tiên",

    "forecast_7d": "### 📈 Dự báo 7 ngày",
    "chart_title": "### 📉 Biểu đồ xu hướng 7 ngày",
    "chart_p_flood": "Xác suất ngập",
    "chart_ratio": "Tỉ lệ ngập",
    "chart_priority": "Điểm ưu tiên",

    "incident_title": "## 🧩 Theo dõi sự cố (trạng thái cứu hộ)",
    "status_label": "Trạng thái",
    "note_label": "Ghi chú",
    "update_incident": "Cập nhật trạng thái sự cố",
    "incident_updated": "✅ Đã cập nhật sự cố!",

    "shelter_title": "## 🏠 Điểm trú ẩn (gần nhất)",
    "shelters_in_commune": "Điểm trú ẩn trong xã:",
    "nearest_shelters": "Điểm trú ẩn gần nhất:",
    "no_shelter_file": "Chưa có shelters.csv. Thêm data/shelters.csv để bật tính năng này.",

    "click_hint": "👆 Hãy bấm vào đa giác một xã trên bản đồ để xem chi tiết & điều hành cứu hộ.",

    "allocation_title": "🛠️ Phân bổ nguồn lực (tự động + chỉnh tay)",
    "allocate_topN": "Phân bổ cho Top N xã",
    "suggested_alloc": "### Gợi ý phân bổ (Đội/Thuyền/Xe) + Ước tính nhu cầu",
    "save_alloc": "✅ Lưu phân bổ vào DB (Top N)",
    "alloc_saved": "✅ Đã lưu phân bổ!",
    "saved_alloc_title": "### Phân bổ đã lưu (DB)",

    "scheduler_title": "⏱️ Lịch chạy / Cache (cập nhật hằng ngày)",
    "scheduler_help": """
**Khuyến nghị vận hành thực tế:**
- Mỗi ngày (hoặc 3 giờ/lần), chạy `python src/scheduler_cache.py` để tạo cache parquet.
- App sẽ đọc cache để hiển thị nhanh, tránh bị rate-limit từ Open-Meteo.
""",
    "show_cache_path": "Hiển thị đường dẫn cache dự kiến hôm nay",

    # ---- Nhãn cột hiển thị tiếng Việt ----
    "col_gid": "Mã xã (GID_3)",
    "col_name": "Tên xã",
    "col_p_flood": "Xác suất ngập",
    "col_ratio": "Tỉ lệ ngập",
    "col_affected": "Dân bị ảnh hưởng",
    "col_priority_score": "Điểm ưu tiên",
    "col_priority_level": "Mức ưu tiên",
    "col_area_m2": "Diện tích ngập (m²)",
    "col_last_update": "Cập nhật lúc",

    "col_teams_alloc": "Đội (phân bổ)",
    "col_boats_alloc": "Thuyền (phân bổ)",
    "col_trucks_alloc": "Xe tải (phân bổ)",
    "col_food_need": "Nhu cầu lương thực (gói)",
    "col_water_need": "Nhu cầu nước (lít)",
    "col_med_need": "Nhu cầu y tế (bộ)",

    # ---- Intake / Dispatch / Shelter ----
    "intake_title": "📞 Tiếp nhận yêu cầu cứu hộ (Hotline)",
    "intake_new": "➕ Tạo yêu cầu cứu hộ mới",
    "caller_name": "Họ tên người báo",
    "phone": "Số điện thoại",
    "gid3": "Mã xã (GID_3)",
    "commune_name": "Tên xã",
    "lat": "Vĩ độ (lat) (nếu có)",
    "lon": "Kinh độ (lon) (nếu có)",
    "people": "Số người cần hỗ trợ",
    "urgency": "Mức khẩn cấp (1 thấp → 5 rất khẩn)",
    "req_note": "Mô tả tình huống / địa điểm chi tiết",
    "req_submit": "📩 Ghi nhận yêu cầu",
    "req_list": "### 📋 Danh sách yêu cầu gần đây",

    "dispatch_title": "🧭 Điều phối nhiệm vụ (Tạo & Cập nhật)",
    "create_task_from_req": "⚡ Tạo nhiệm vụ từ yêu cầu (chọn request → tạo task)",
    "create_task_manual": "➕ Tạo nhiệm vụ thủ công (theo xã đang chọn)",
    "task_list": "### 📋 Danh sách nhiệm vụ",
    "task_update": "✏️ Cập nhật nhiệm vụ",

    "shelter_manage_title": "🏠 Quản lý điểm trú ẩn (còn chỗ + nhu cầu + chuyển người)",
    "shelter_status_edit": "✏️ Cập nhật trạng thái điểm trú ẩn",
    "shelter_status_table": "### 📋 Trạng thái điểm trú ẩn",
    "shelter_move": "🚐 Chuyển người đến điểm trú ẩn (tạo log)",
    "shelter_move_history": "### 🧾 Lịch sử chuyển người",

    # hiển thị bổ sung
    "col_commune_name": "Tên xã (suy ra)",
    "col_start_time": "Thời gian bắt đầu",
    "col_from_commune": "Tên xã đi (suy ra)",
    "col_to_gid3": "Mã xã đến (suy ra)",
    "col_to_commune": "Tên xã đến (suy ra)",
}

# # =========================
# # CONFIG STREAMLIT
# # =========================
st.set_page_config(layout="wide")
st.title(VI["app_title"])

# # init DB
init_db()

# # =========================
# # HELPERS
# # =========================
def safe_index(options, value, default=0):
    try:
        return options.index(value)
    except Exception:
        return default

def normalize_str(x):
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass
    return str(x).strip()

def to_int_safe(x, default=0):
    v = pd.to_numeric(x, errors="coerce")
    return int(default if pd.isna(v) else v)

def normalize_name_key(x: str) -> str:
    # đơn giản: lower + strip; nếu muốn mạnh hơn (bỏ dấu) bạn có thể thêm unidecode
    return normalize_str(x).lower()

# # =========================
# # VIỆT HOÁ TÊN CỘT HIỂN THỊ
# # =========================
DISPLAY_COLS = {
    "GID_3": VI["col_gid"],
    "name": VI["col_name"],
    "p_flood": VI["col_p_flood"],
    "flood_ratio": VI["col_ratio"],
    "affected_population": VI["col_affected"],
    "priority_score": VI["col_priority_score"],
    "priority_level": VI["col_priority_level"],
    "flood_area_m2": VI["col_area_m2"],
    "last_update": VI["col_last_update"],

    "teams_alloc": VI["col_teams_alloc"],
    "boats_alloc": VI["col_boats_alloc"],
    "trucks_alloc": VI["col_trucks_alloc"],
    "food_packs_need": VI["col_food_need"],
    "water_liters_need": VI["col_water_need"],
    "medical_kits_need": VI["col_med_need"],

    "teams": VI["col_teams_alloc"],
    "boats": VI["col_boats_alloc"],
    "trucks": VI["col_trucks_alloc"],
    "food_packs": VI["col_food_need"],
    "water_liters": VI["col_water_need"],
    "medical_kits": VI["col_med_need"],

    # bổ sung hiển thị
    "commune_name_derived": VI["col_commune_name"],
    "start_time": VI["col_start_time"],
    "from_commune_derived": VI["col_from_commune"],
    "to_gid3_derived": VI["col_to_gid3"],
    "to_commune_derived": VI["col_to_commune"],
}

def df_vi(df_: pd.DataFrame) -> pd.DataFrame:
    if df_ is None or len(df_) == 0:
        return df_
    return df_.rename(columns=DISPLAY_COLS)

# # =========================
# # LOAD DATA
# # =========================
@st.cache_data
def load_merged():
    if not MERGED_FILE.exists():
        st.error(f"❌ Không tìm thấy dữ liệu tổng hợp: {MERGED_FILE}")
        st.info("👉 Kiểm tra bạn có `cache/merged.csv` ở thư mục project.")
        st.stop()

    df0 = pd.read_csv(MERGED_FILE)
    df0["date_local"] = pd.to_datetime(df0["date_local"], errors="coerce")
    df0["GID_3"] = df0["GID_3"].astype("string")
    df0 = df0.dropna(subset=["date_local", "GID_3"])
    df0["GID_3"] = df0["GID_3"].astype(str)
    return df0

@st.cache_data
def load_population():
    if not POP_FILE.exists():
        st.error(f"❌ Không tìm thấy dữ liệu dân số: {POP_FILE}")
        st.info("👉 Kiểm tra bạn có `data/DanSo_Xa.csv` ở thư mục project.")
        st.stop()

    pop = pd.read_csv(POP_FILE)
    pop["GID_3"] = pop["GID_3"].astype(str)
    if "DanSo_sum" not in pop.columns:
        st.error("❌ File dân số thiếu cột `DanSo_sum`.")
        st.stop()
    return pop.set_index("GID_3")["DanSo_sum"].to_dict()

@st.cache_data
def load_shelters():
    if SHELTER_FILE.exists():
        s = pd.read_csv(SHELTER_FILE)
        for col in ["name", "type", "lat", "lon", "capacity", "commune_gid3"]:
            if col not in s.columns:
                s[col] = np.nan
        return s
    return pd.DataFrame(columns=["name", "type", "lat", "lon", "capacity", "commune_gid3"])

df = load_merged()
pop_dict = load_population()
shelters_df = load_shelters()

# # =========================
# # BUILD STATIC (LẤY THÔNG TIN XÃ)
# # =========================
static = (
    df.sort_values("date_local")
      .drop_duplicates(subset=["GID_3"], keep="first")
      .copy()
      .set_index("GID_3")
)

# # Tạo map gid<->name để auto-fill intake + enrich bảng
gid_to_name = {}
namekey_to_gid = {}
for gid in static.index.astype(str).tolist():
    row = static.loc[gid]
    name = row.get("NAME_3_flood", gid)
    name = normalize_str(name) if name is not None else gid
    gid_to_name[str(gid)] = name

    k = normalize_name_key(name)
    # nếu trùng tên -> giữ cái đầu tiên
    if k and k not in namekey_to_gid:
        namekey_to_gid[k] = str(gid)

def derive_commune_name(gid3: str) -> str:
    g = normalize_str(gid3)
    return gid_to_name.get(g, "")

# # =========================
# # CACHE LOADER (7 DAYS)
# # =========================
def load_today_cache():
    today = datetime.now().strftime("%Y-%m-%d")
    path = CACHE_DIR / f"{today}_commune_forecasts_7d.parquet"
    if path.exists():
        try:
            cache = pd.read_parquet(path)
        except Exception as e:
            st.warning(f"⚠️ Không đọc được cache parquet: {path}\n\nChi tiết: {e}")
            return None

        cache["GID_3"] = cache["GID_3"].astype(str)
        cache["date_local"] = pd.to_datetime(cache["date_local"], errors="coerce")
        cache = cache.dropna(subset=["GID_3", "date_local"])
        return cache
    return None

cache_df = load_today_cache()

# # =========================
# # PRIORITY COMPUTATION
# # =========================
@st.cache_data(ttl=3600)
def compute_today_priority_live(df_in, static_in, pop_dict_in):
    results = []
    for gid in static_in.index:
        row = static_in.loc[gid]
        lat = float(row["commune_lat"])
        lon = float(row["commune_lon"])
        population = pop_dict_in.get(str(gid), None)

        hist = df_in[df_in["GID_3"] == str(gid)].sort_values("date_local").tail(60)
        pred = predict_7days(str(gid), lat, lon, row, hist, population=population)
        today = pred.iloc[0]

        results.append({
            "GID_3": str(gid),
            "name": row.get("NAME_3_flood", str(gid)),
            "p_flood": float(today["p_flood"]),
            "flood_ratio": float(today["flood_ratio"]),
            "affected_population": float(today["affected_population"]) if pd.notna(today["affected_population"]) else 0.0,
            "priority_score": float(today["priority_score"]),
            "priority_level": today["priority_level"],
            "flood_area_m2": float(today.get("flood_area_m2", 0)) if pd.notna(today.get("flood_area_m2", 0)) else 0.0,
            "last_update": datetime.now().isoformat(timespec="seconds"),
        })
    return pd.DataFrame(results)

def compute_today_priority_from_cache(cache_in: pd.DataFrame) -> pd.DataFrame:
    today_date = cache_in["date_local"].min()
    first_day = cache_in[cache_in["date_local"] == today_date].copy()
    cols = ["GID_3", "name", "p_flood", "flood_ratio", "affected_population", "priority_score", "priority_level"]
    if "flood_area_m2" in first_day.columns:
        cols.append("flood_area_m2")
    out = first_day[cols].copy()
    out["last_update"] = datetime.now().isoformat(timespec="seconds")
    return out

if cache_df is not None and len(cache_df):
    rank_df = compute_today_priority_from_cache(cache_df)
    st.info(VI["using_cache"])
else:
    rank_df = compute_today_priority_live(df, static, pop_dict)
    st.warning(VI["no_cache"])

rank_df["GID_3"] = rank_df["GID_3"].astype(str)
priority_map = rank_df.set_index("GID_3")["priority_score"].to_dict()

# # map ưu tiên theo gid
gid_to_priority = {}
for _, r in rank_df.iterrows():
    gid_to_priority[str(r["GID_3"])] = float(r.get("priority_score") or 0)

# # =========================
# # COLOR SCALE
# # =========================
def score_to_color(score: float) -> str:
    score = float(score or 0)
    if score >= 80:
        return "#d7191c"
    if score >= 50:
        return "#fdae61"
    if score >= 20:
        return "#ffffbf"
    return "#a6d96a"

# # =========================
# # INCIDENTS & ALLOCATIONS
# # =========================
inc_rows = get_all_incidents()
inc_df = (
    pd.DataFrame(inc_rows, columns=["GID_3", "status", "last_update", "note"])
    if inc_rows else pd.DataFrame(columns=["GID_3", "status", "last_update", "note"])
)

alloc_rows = get_all_allocations()
alloc_df = (
    pd.DataFrame(
        alloc_rows,
        columns=["GID_3", "teams", "boats", "trucks", "food_packs", "water_liters", "medical_kits", "last_update"]
    )
    if alloc_rows else pd.DataFrame()
)

# # =========================
# # SIDEBAR CONTROL PANEL
# # =========================
st.sidebar.header(VI["sidebar_panel"])

use_thresholds = st.sidebar.checkbox(VI["use_default_thresholds"], value=True)
if not use_thresholds:
    ALERT_THRESH_P = st.sidebar.slider(VI["alert_p"], 0.0, 1.0, 0.7, 0.05)
    ALERT_THRESH_AFFECTED = st.sidebar.number_input(VI["alert_affected"], value=500, step=50)
    ALERT_THRESH_PRIORITY = st.sidebar.number_input(VI["alert_priority"], value=80, step=5)

st.sidebar.divider()
st.sidebar.subheader(VI["total_resources"])
teams_total = st.sidebar.number_input(VI["teams"], 0, 9999, 10, 1)
boats_total = st.sidebar.number_input(VI["boats"], 0, 9999, 15, 1)
trucks_total = st.sidebar.number_input(VI["trucks"], 0, 9999, 8, 1)
food_total = st.sidebar.number_input(VI["food"], 0, 9999999, 5000, 100)
water_total = st.sidebar.number_input(VI["water"], 0, 9999999, 10000, 200)
med_total = st.sidebar.number_input(VI["med"], 0, 9999999, 600, 20)

# # =========================
# # ALERT CENTER
# # =========================
st.subheader(VI["alert_center"])

alerts = rank_df[
    (rank_df["p_flood"] >= ALERT_THRESH_P) |
    (rank_df["affected_population"] >= ALERT_THRESH_AFFECTED) |
    (rank_df["priority_score"] >= ALERT_THRESH_PRIORITY)
].copy().sort_values("priority_score", ascending=False)

cA, cB, cC = st.columns(3)
cA.metric(VI["alerts_count"], len(alerts))
cB.metric(VI["max_priority"], f"{alerts['priority_score'].max():.1f}" if len(alerts) else "0")
cC.metric(VI["max_affected"], f"{alerts['affected_population'].max():.0f}" if len(alerts) else "0")

st.write(VI["alerts_table_title"])
st.dataframe(df_vi(alerts.head(50)), use_container_width=True)

# # =========================
# # MAP (Priority choropleth)
# # =========================
st.subheader(VI["priority_map"])

features = []
for gid, row in static.iterrows():
    if ".geo" not in row or pd.isna(row[".geo"]):
        continue
    try:
        geom = json.loads(row[".geo"])
    except Exception:
        continue

    name = row.get("NAME_3_flood", gid)
    score = float(priority_map.get(str(gid), 0.0))

    features.append({
        "type": "Feature",
        "properties": {"GID_3": str(gid), "name": str(name), "priority_score": round(score, 2)},
        "geometry": geom
    })

geojson = {"type": "FeatureCollection", "features": features}

center_lat = float(static["commune_lat"].mean())
center_lon = float(static["commune_lon"].mean())

m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="CartoDB positron")

def style_fn(feature):
    score = feature["properties"].get("priority_score", 0)
    return {"fillColor": score_to_color(float(score)), "color": "black", "weight": 0.4, "fillOpacity": 0.6}

folium.GeoJson(
    geojson,
    name="Các xã",
    style_function=style_fn,
    tooltip=folium.GeoJsonTooltip(
        fields=["name", "priority_score"],
        aliases=[VI["tooltip_name"], VI["tooltip_priority"]],
        sticky=True
    )
).add_to(m)

map_data = st_folium(m, width=1200, height=650)

# # ✅ FIX lấy selected_gid/selected_name an toàn
selected_gid = None
selected_name = None
if map_data:
    lad = map_data.get("last_active_drawing")
    if isinstance(lad, dict):
        props = lad.get("properties") or (lad.get("feature") or {}).get("properties") or {}
        if isinstance(props, dict):
            selected_gid = props.get("GID_3") or props.get("gid3") or props.get("gid_3")
            selected_name = props.get("name") or props.get("NAME_3") or props.get("commune_name")

# # =========================
# # LEFT: Ranking + Export
# # =========================
left, right = st.columns([1, 1])

with left:
    st.write(VI["top20"])
    top20 = rank_df.sort_values("priority_score", ascending=False).head(20)
    st.dataframe(df_vi(top20), use_container_width=True)

    st.write(VI["export_report"])
    if st.button(VI["export_csv_btn"]):
        out_path = PROJECT_DIR / "cache" / f"report_top50_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export_csv(rank_df.sort_values("priority_score", ascending=False).head(50), str(out_path))
        st.success(f"{VI['csv_saved']} {out_path}")

    if st.button(VI["export_pdf_btn"]):
        out_path = PROJECT_DIR / "cache" / f"report_top20_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        summary = {
            "Số cảnh báo": len(alerts),
            "Điểm ưu tiên cao nhất": float(rank_df["priority_score"].max()),
            "Tổng dân bị ảnh hưởng (top20)": float(top20["affected_population"].sum()),
        }
        export_pdf(top20, summary, str(out_path))
        st.success(f"{VI['pdf_saved']} {out_path}")

# # =========================
# # RIGHT: Commune details
# # =========================
with right:
    if selected_gid:
        st.success(f"{VI['selected']} {selected_name} ({selected_gid})")

        row = static.loc[str(selected_gid)]
        lat = float(row["commune_lat"])
        lon = float(row["commune_lon"])
        population = pop_dict.get(str(selected_gid), None)

        if cache_df is not None and len(cache_df):
            pred = cache_df[cache_df["GID_3"].astype(str) == str(selected_gid)].sort_values("date_local").copy()
        else:
            hist = df[df["GID_3"].astype(str) == str(selected_gid)].sort_values("date_local").tail(60)
            pred = predict_7days(str(selected_gid), lat, lon, row, hist, population=population)

        today = pred.iloc[0]

        st.write(VI["today_overview"])
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(VI["p_flood"], f"{today['p_flood']:.2%}")
        c2.metric(VI["area_m2"], f"{today.get('flood_area_m2', 0):.0f}")
        c3.metric(VI["ratio"], f"{today['flood_ratio']:.2%}")
        c4.metric(VI["affected"], f"{today['affected_population']:.0f}")
        c5.metric(VI["priority"], f"{today['priority_level']} ({today['priority_score']:.1f})")

        st.write(VI["forecast_7d"])
        st.dataframe(df_vi(pred), use_container_width=True)

        st.write(VI["chart_title"])
        chart_df = pred.set_index("date_local")[["p_flood", "flood_ratio", "priority_score"]].copy()
        chart_df = chart_df.rename(columns={
            "p_flood": VI["chart_p_flood"],
            "flood_ratio": VI["chart_ratio"],
            "priority_score": VI["chart_priority"],
        })
        st.line_chart(chart_df)

        # -------- INCIDENT --------
        st.write(VI["incident_title"])
        status_options = ["CHƯA TRIỂN KHAI", "ĐANG TRIỂN KHAI", "HOÀN TẤT", "CẦN HỖ TRỢ THÊM"]
        current = get_incident(str(selected_gid))
        current_status = (current[1] if current else "CHƯA TRIỂN KHAI") or "CHƯA TRIỂN KHAI"
        current_status = str(current_status).strip()
        current_note = current[3] if current else ""

        idx0 = safe_index(status_options, current_status, default=0)
        new_status = st.selectbox(VI["status_label"], status_options, index=idx0, key="incident_status")
        new_note = st.text_area(VI["note_label"], value=str(current_note or ""), height=80, key="incident_note")

        if st.button(VI["update_incident"], key="incident_update_btn"):
            upsert_incident(str(selected_gid), new_status, new_note)
            st.success(VI["incident_updated"])

        # -------- SHELTER LIST (cũ) --------
        st.write(VI["shelter_title"])

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            p = math.pi / 180
            a = (
                0.5 - math.cos((lat2 - lat1) * p) / 2
                + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
            )
            return 2 * R * math.asin(math.sqrt(a))

        if len(shelters_df):
            s_same = shelters_df[shelters_df["commune_gid3"].astype(str) == str(selected_gid)].copy()
        else:
            s_same = pd.DataFrame()

        if len(s_same) > 0:
            st.write(VI["shelters_in_commune"])
            st.dataframe(s_same[["name", "type", "capacity", "lat", "lon"]], use_container_width=True)
        elif len(shelters_df) > 0:
            shelters_df2_tmp = shelters_df.dropna(subset=["lat", "lon"]).copy()
            if len(shelters_df2_tmp):
                shelters_df2_tmp["dist_km"] = shelters_df2_tmp.apply(
                    lambda r: haversine(lat, lon, float(r["lat"]), float(r["lon"])),
                    axis=1
                )
                nearest = shelters_df2_tmp.sort_values("dist_km").head(5)
                st.write(VI["nearest_shelters"])
                st.dataframe(nearest[["name", "type", "capacity", "dist_km", "lat", "lon"]], use_container_width=True)
            else:
                st.info(VI["no_shelter_file"])
        else:
            st.info(VI["no_shelter_file"])
    else:
        st.info(VI["click_hint"])

# # ============================================================
# # ✅ TÍCH HỢP 3 TÍNH NĂNG: Intake → Dispatch → Shelter status
# # ============================================================

# # =========================
# # (1) HOTLINE / INTAKE (AUTO-FILL GID <-> TÊN XÃ)
# # =========================
st.subheader(VI["intake_title"])

def intake_fill_from_gid():
    gid = normalize_str(st.session_state.get("req_gid3", ""))
    if gid:
        st.session_state["req_commune_name"] = derive_commune_name(gid)

def intake_fill_from_name():
    name = normalize_str(st.session_state.get("req_commune_name", ""))
    if not name:
        return
    gid = namekey_to_gid.get(normalize_name_key(name), "")
    if gid:
        st.session_state["req_gid3"] = gid
    # đảm bảo đồng bộ tên theo map chuẩn
    st.session_state["req_commune_name"] = derive_commune_name(st.session_state.get("req_gid3", "")) or name

with st.expander(VI["intake_new"], expanded=True):
    c1, c2 = st.columns([1, 1])
    caller_name = c1.text_input(VI["caller_name"], key="req_caller")
    phone = c2.text_input(VI["phone"], key="req_phone")

    c3, c4 = st.columns([1, 1])
    st.text_input(VI["gid3"], value=selected_gid or "", key="req_gid3", on_change=intake_fill_from_gid)
    st.text_input(VI["commune_name"], value=selected_name or "", key="req_commune_name", on_change=intake_fill_from_name)

    c5, c6, c7 = st.columns([1, 1, 1])
    lat_req = c5.number_input(VI["lat"], value=0.0, format="%.6f", key="req_lat")
    lon_req = c6.number_input(VI["lon"], value=0.0, format="%.6f", key="req_lon")
    people = c7.number_input(VI["people"], min_value=0, value=1, step=1, key="req_people")

    urgency = st.slider(VI["urgency"], 1, 5, 4, 1, key="req_urgency")
    note_req = st.text_area(VI["req_note"], height=80, key="req_note")

    if st.button(VI["req_submit"], key="req_submit"):
        gid3_req = normalize_str(st.session_state.get("req_gid3", ""))
        name_req = normalize_str(st.session_state.get("req_commune_name", ""))

        # nếu chỉ nhập tên -> cố tìm gid
        if (not gid3_req) and name_req:
            gid3_req = namekey_to_gid.get(normalize_name_key(name_req), "")

        rid = create_request({
            "caller_name": caller_name,
            "phone": phone,
            "gid3": gid3_req if gid3_req else None,
            "lat": float(lat_req) if float(lat_req) != 0 else None,
            "lon": float(lon_req) if float(lon_req) != 0 else None,
            "people": int(people),
            "urgency": int(urgency),
            "note": note_req,
            "status": "MỚI"
        })
        st.success(f"✅ Đã tạo yêu cầu cứu hộ ID = {rid}")

st.write(VI["req_list"])
req_rows = list_requests(limit=200)
req_df = pd.DataFrame(req_rows, columns=[
    "id", "created_at", "caller_name", "phone", "gid3", "lat", "lon", "people",
    "urgency", "note", "status", "linked_task_id"
])

# # thêm tên xã suy ra để dễ nhìn
if len(req_df):
    req_df["gid3"] = req_df["gid3"].astype(str).replace("None", "")
    req_df["commune_name_derived"] = req_df["gid3"].apply(lambda g: derive_commune_name(g) if g and g != "nan" else "")
st.dataframe(df_vi(req_df), use_container_width=True)

# # =========================
# # (2) TASKING / DISPATCH
# # =========================
st.subheader(VI["dispatch_title"])

with st.expander(VI["create_task_from_req"], expanded=True):
    if len(req_df):
        req_id = st.selectbox("Chọn Request ID", req_df["id"].tolist(), key="req_pick")
        req_row = req_df[req_df["id"] == req_id].iloc[0].to_dict()

        gid3 = normalize_str(req_row.get("gid3")) or normalize_str(selected_gid)
        # suy tên xã từ gid
        commune_name = derive_commune_name(gid3) if gid3 else ""

        priority_score = float(gid_to_priority.get(gid3, 0.0)) if gid3 else 0.0

        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        task_type = c1.selectbox("Loại nhiệm vụ", ["CỨU HỘ", "SƠ TÁN", "CẤP PHÁT", "KHẢO SÁT", "Y TẾ"], key="task_type_from_req")
        assigned_team = c2.text_input("Đội phụ trách (tên kíp/đội)", value="", key="task_team_from_req")
        boats = c3.number_input("Số thuyền điều", 0, 999, 0, 1, key="task_boats_from_req")
        trucks = c4.number_input("Số xe tải điều", 0, 999, 0, 1, key="task_trucks_from_req")

        eta_min = st.number_input("ETA (phút) ước tính", 0, 9999, 0, 5, key="task_eta_from_req")
        note_task = st.text_area("Ghi chú nhiệm vụ", value=req_row.get("note") or "", height=80, key="task_note_from_req")

        # hiển thị thông tin sẽ gắn vào task
        st.info(f"📍 Xã: **{commune_name or '(chưa rõ)'}** | Mã xã: **{gid3 or '(trống)'}** | Điểm ưu tiên: **{priority_score:.1f}**")

        if st.button("✅ Tạo nhiệm vụ & gắn với request", key="create_task_from_req_btn"):
            # quy ước: nếu đã giao nguồn lực/đội -> coi như bắt đầu ngay
            now_iso = datetime.now().isoformat(timespec="seconds")
            will_dispatch = bool(assigned_team.strip() or int(boats) > 0 or int(trucks) > 0)
            init_status = "ĐÃ GIAO" if will_dispatch else "MỚI"
            start_time = now_iso if will_dispatch else None

            tid = create_task({
                "gid3": gid3 if gid3 else None,
                "commune_name": commune_name if commune_name else None,
                "task_type": task_type,
                "priority_score": priority_score,
                "assigned_team": assigned_team.strip() if assigned_team.strip() else None,
                "boats": int(boats),
                "trucks": int(trucks),
                "status": init_status,
                "eta_min": int(eta_min),
                "note": note_task,
                "start_time": start_time,
                "source_request_id": int(req_id),
            })
            update_request_status(int(req_id), "ĐÃ TẠO NHIỆM VỤ", linked_task_id=int(tid))
            st.success(f"✅ Đã tạo Task ID = {tid} từ Request ID = {req_id}")
    else:
        st.info("Chưa có yêu cầu để tạo nhiệm vụ.")

with st.expander(VI["create_task_manual"], expanded=False):
    gid3_manual = normalize_str(selected_gid) or st.text_input("Mã xã (GID_3)", key="manual_gid3")
    commune_name_manual = derive_commune_name(gid3_manual) or normalize_str(selected_name)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    task_type2 = c1.selectbox("Loại nhiệm vụ (thủ công)", ["CỨU HỘ", "SƠ TÁN", "CẤP PHÁT", "KHẢO SÁT", "Y TẾ"], key="manual_task_type")
    assigned_team2 = c2.text_input("Đội phụ trách", value="", key="manual_team")
    boats2 = c3.number_input("Thuyền", 0, 999, 0, 1, key="boats2")
    trucks2 = c4.number_input("Xe tải", 0, 999, 0, 1, key="trucks2")
    eta2 = st.number_input("ETA (phút)", 0, 9999, 0, 5, key="eta2")
    note2 = st.text_area("Ghi chú", height=80, key="note2")

    if st.button("✅ Tạo nhiệm vụ (thủ công)", key="create_manual_task"):
        pscore = float(gid_to_priority.get(gid3_manual, 0.0)) if gid3_manual else 0.0
        now_iso = datetime.now().isoformat(timespec="seconds")
        will_dispatch = bool(assigned_team2.strip() or int(boats2) > 0 or int(trucks2) > 0)
        init_status = "ĐÃ GIAO" if will_dispatch else "MỚI"
        start_time = now_iso if will_dispatch else None

        tid = create_task({
            "gid3": gid3_manual if gid3_manual else None,
            "commune_name": commune_name_manual if commune_name_manual else None,
            "task_type": task_type2,
            "priority_score": pscore,
            "assigned_team": assigned_team2.strip() if assigned_team2.strip() else None,
            "boats": int(boats2),
            "trucks": int(trucks2),
            "status": init_status,
            "eta_min": int(eta2),
            "note": note2,
            "start_time": start_time,
            "source_request_id": None,
        })
        st.success(f"✅ Đã tạo Task ID = {tid}")

st.write(VI["task_list"])
task_rows = list_tasks(limit=300)
task_df = pd.DataFrame(task_rows, columns=[
    "id","created_at","gid3","commune_name","task_type","priority_score",
    "assigned_team","boats","trucks","status","eta_min","start_time","end_time","note","source_request_id"
])

# # enrich: đảm bảo đủ mã xã + tên xã + điểm ưu tiên + thời gian bắt đầu
if len(task_df):
    task_df["gid3"] = task_df["gid3"].astype(str).replace("None", "")
    task_df["commune_name"] = task_df["commune_name"].astype(str).replace("None", "")
    # nếu commune_name trống -> suy ra từ gid
    task_df["commune_name_derived"] = task_df.apply(
        lambda r: normalize_str(r.get("commune_name")) or derive_commune_name(r.get("gid3")),
        axis=1
    )
    # nếu priority_score trống -> suy ra
    task_df["priority_score"] = pd.to_numeric(task_df["priority_score"], errors="coerce").fillna(
        task_df["gid3"].apply(lambda g: gid_to_priority.get(normalize_str(g), 0.0))
    )
st.dataframe(df_vi(task_df), use_container_width=True)

with st.expander(VI["task_update"], expanded=False):
    if len(task_df):
        task_id = st.selectbox("Chọn Task ID", task_df["id"].tolist(), key="task_pick")
        trow = task_df[task_df["id"] == task_id].iloc[0].to_dict()

        status_options = ["MỚI","ĐÃ GIAO","ĐANG ĐI","ĐANG THỰC HIỆN","HOÀN TẤT","HỦY"]
        cur_task_status = str(trow.get("status") or "").strip()
        idx_task = safe_index(status_options, cur_task_status, default=0)

        new_status = st.selectbox("Trạng thái mới", status_options, index=idx_task, key="task_new_status")
        team_new = st.text_input("Đội phụ trách", value=str(trow.get("assigned_team") or ""), key="task_new_team")
        boats_new = st.number_input("Thuyền", 0, 999, to_int_safe(trow.get("boats"), 0), 1, key="task_new_boats")
        trucks_new = st.number_input("Xe tải", 0, 999, to_int_safe(trow.get("trucks"), 0), 1, key="task_new_trucks")
        eta_new = st.number_input("ETA (phút)", 0, 9999, to_int_safe(trow.get("eta_min"), 0), 5, key="task_new_eta")
        note_new = st.text_area("Ghi chú", value=str(trow.get("note") or ""), height=80, key="task_new_note")

        start_time = None
        end_time = None
        now_iso = datetime.now().isoformat(timespec="seconds")

        # quy ước cập nhật: set start_time khi chuyển sang ĐANG ĐI/ĐANG THỰC HIỆN mà chưa có start_time
        if new_status in ["ĐANG ĐI","ĐANG THỰC HIỆN"] and not trow.get("start_time"):
            start_time = now_iso
        if new_status == "HOÀN TẤT":
            if not trow.get("start_time"):
                start_time = now_iso
            end_time = now_iso

        if st.button("✅ Lưu cập nhật Task", key="task_update_btn"):
            update_task(
                int(task_id),
                status=new_status,
                assigned_team=team_new.strip() if team_new.strip() else None,
                boats=int(boats_new),
                trucks=int(trucks_new),
                eta_min=int(eta_new),
                note=note_new,
                start_time=start_time,
                end_time=end_time
            )
            st.success("✅ Đã cập nhật nhiệm vụ.")
    else:
        st.info("Chưa có nhiệm vụ nào.")

# # =========================
# # (3) SHELTER STATUS + CHUYỂN NGƯỜI
# # =========================
st.subheader(VI["shelter_manage_title"])

def make_shelter_key(r):
    gid = normalize_str(r.get("commune_gid3", ""))
    name = normalize_str(r.get("name", ""))
    return f"{gid}__{name}".lower()

# # chuẩn hoá shelters_df2 + label_map để selectbox không lỗi
if len(shelters_df):
    shelters_df2 = shelters_df.copy()
    shelters_df2["commune_gid3"] = shelters_df2["commune_gid3"].apply(normalize_str)
    shelters_df2["name"] = shelters_df2["name"].apply(normalize_str)
    shelters_df2["shelter_key"] = shelters_df2.apply(make_shelter_key, axis=1)

    shelters_df2["shelter_key"] = shelters_df2["shelter_key"].astype(str)
    shelters_df2 = shelters_df2[shelters_df2["shelter_key"].str.len() > 2].copy()
    shelters_df2 = shelters_df2.drop_duplicates(subset=["shelter_key"], keep="first").copy()

    shelters_df2["label"] = shelters_df2.apply(
        lambda r: f"{normalize_str(r.get('name'))} (GID_3: {normalize_str(r.get('commune_gid3'))})",
        axis=1
    )
    shelter_label_map = dict(zip(shelters_df2["shelter_key"].tolist(), shelters_df2["label"].tolist()))
else:
    shelters_df2 = pd.DataFrame(columns=["shelter_key","name","capacity","commune_gid3","lat","lon","type","label"])
    shelter_label_map = {}

with st.expander(VI["shelter_status_edit"], expanded=False):
    if len(shelters_df2):
        shelter_keys = shelters_df2["shelter_key"].astype(str).tolist()

        shelter_key = st.selectbox(
            "Chọn điểm trú ẩn",
            shelter_keys,
            format_func=lambda k: shelter_label_map.get(str(k), str(k)),
            key="pick_shelter_key"
        )

        srow = shelters_df2[shelters_df2["shelter_key"].astype(str) == str(shelter_key)].iloc[0].to_dict()

        cap_default = to_int_safe(srow.get("capacity", 0), 0)
        gid3_default = normalize_str(srow.get("commune_gid3"))
        commune_name_derived = derive_commune_name(gid3_default)

        st.info(f"📍 Xã: **{commune_name_derived or '(chưa rõ)'}** | Mã xã: **{gid3_default or '(trống)'}**")

        c1, c2, c3, c4 = st.columns([1,1,1,1])
        capacity = c1.number_input("Sức chứa", 0, 999999, cap_default, 10, key="sh_cap")
        current_people = c2.number_input("Đang có (người)", 0, 999999, 0, 10, key="sh_cur")
        need_food = c3.number_input("Nhu cầu lương thực (gói)", 0, 9999999, 0, 50, key="sh_food")
        need_water = c4.number_input("Nhu cầu nước (lít)", 0, 9999999, 0, 100, key="sh_water")
        need_med = st.number_input("Nhu cầu y tế (bộ)", 0, 9999999, 0, 10, key="sh_med")
        note = st.text_area("Ghi chú", height=80, key="sh_note")

        if st.button("✅ Lưu trạng thái điểm trú ẩn", key="save_shelter_status_btn"):
            upsert_shelter_status({
                "shelter_key": str(shelter_key),
                "name": normalize_str(srow.get("name")),
                "gid3": gid3_default,
                "capacity": int(capacity),
                "current_people": int(current_people),
                "need_food": int(need_food),
                "need_water": int(need_water),
                "need_med": int(need_med),
                "note": note
            })
            st.success("✅ Đã lưu trạng thái điểm trú ẩn.")
    else:
        st.info("Chưa có shelters.csv để quản lý điểm trú ẩn.")

st.write(VI["shelter_status_table"])
sh_rows = list_shelter_status(limit=500)
sh_df = pd.DataFrame(sh_rows, columns=[
    "shelter_key","name","gid3","capacity","current_people","need_food","need_water","need_med","note","last_update"
])
if len(sh_df):
    sh_df["gid3"] = sh_df["gid3"].astype(str).replace("None", "")
    sh_df["commune_name_derived"] = sh_df["gid3"].apply(lambda g: derive_commune_name(g) if g and g != "nan" else "")
    sh_df["còn_trống"] = sh_df["capacity"].fillna(0) - sh_df["current_people"].fillna(0)
st.dataframe(df_vi(sh_df), use_container_width=True)

with st.expander(VI["shelter_move"], expanded=True):
    if len(sh_df):
        from_gid3 = st.text_input("Từ xã (GID_3)", value=normalize_str(selected_gid) or "", key="mv_from_gid3")
        from_name = derive_commune_name(from_gid3)
        if from_gid3:
            st.caption(f"➡️ Tên xã đi (suy ra): {from_name or '(chưa rõ)'}")

        sh_df2 = sh_df.copy()
        sh_df2["shelter_key"] = sh_df2["shelter_key"].astype(str)
        sh_df2["name"] = sh_df2["name"].apply(normalize_str)
        mv_label_map = dict(zip(sh_df2["shelter_key"].tolist(), sh_df2["name"].tolist()))

        to_key = st.selectbox(
            "Đến điểm trú ẩn",
            sh_df2["shelter_key"].tolist(),
            format_func=lambda k: mv_label_map.get(str(k), str(k)),
            key="mv_to_key"
        )

        # suy ra xã đến dựa theo shelter_status
        to_row = sh_df2[sh_df2["shelter_key"].astype(str) == str(to_key)].iloc[0].to_dict()
        to_gid3 = normalize_str(to_row.get("gid3"))
        to_commune_name = derive_commune_name(to_gid3)

        st.caption(f"🏁 Xã đến (suy ra): {to_commune_name or '(chưa rõ)'} | Mã xã: {to_gid3 or '(trống)'}")

        people_move = st.number_input("Số người chuyển", 0, 999999, 10, 1, key="mv_people")
        link_task = st.number_input("Gắn Task ID (nếu có)", 0, 999999, 0, 1, key="mv_task")
        note_move = st.text_input("Ghi chú chuyển", key="mv_note")

        if st.button("✅ Xác nhận chuyển", key="mv_confirm"):
            create_shelter_move({
                "from_gid3": from_gid3.strip() if from_gid3.strip() else None,
                "to_shelter_key": str(to_key),
                "people": int(people_move),
                "task_id": int(link_task) if int(link_task) > 0 else None,
                "note": note_move
            })

            # cập nhật current_people ở shelter_status
            cur = sh_df2[sh_df2["shelter_key"].astype(str) == str(to_key)].iloc[0].to_dict()
            new_current = to_int_safe(cur.get("current_people", 0), 0) + int(people_move)

            upsert_shelter_status({
                "shelter_key": str(to_key),
                "name": cur.get("name"),
                "gid3": cur.get("gid3"),
                "capacity": to_int_safe(cur.get("capacity", 0), 0),
                "current_people": int(new_current),
                "need_food": to_int_safe(cur.get("need_food", 0), 0),
                "need_water": to_int_safe(cur.get("need_water", 0), 0),
                "need_med": to_int_safe(cur.get("need_med", 0), 0),
                "note": cur.get("note")
            })

            st.success("✅ Đã ghi nhận chuyển người và cập nhật điểm trú ẩn.")
    else:
        st.info("Chưa có dữ liệu shelter_status để thực hiện chuyển người.")

st.write(VI["shelter_move_history"])
mv_rows = list_shelter_moves(limit=300)
mv_df = pd.DataFrame(mv_rows, columns=["id","created_at","from_gid3","to_shelter_key","people","task_id","note"])

# # enrich lịch sử chuyển: thêm tên xã đi/đến + mã xã đến
if len(mv_df):
    mv_df["from_gid3"] = mv_df["from_gid3"].astype(str).replace("None", "")
    mv_df["from_commune_derived"] = mv_df["from_gid3"].apply(lambda g: derive_commune_name(g) if g and g != "nan" else "")

    # join to shelter_status để suy ra to_gid3
    if len(sh_df):
        sh_key_to_gid = dict(zip(sh_df["shelter_key"].astype(str).tolist(), sh_df["gid3"].astype(str).tolist()))
    else:
        sh_key_to_gid = {}

    mv_df["to_gid3_derived"] = mv_df["to_shelter_key"].astype(str).apply(lambda k: normalize_str(sh_key_to_gid.get(str(k), "")))
    mv_df["to_commune_derived"] = mv_df["to_gid3_derived"].apply(lambda g: derive_commune_name(g) if g else "")

st.dataframe(df_vi(mv_df), use_container_width=True)

# # =========================
# # PHÂN BỔ NGUỒN LỰC
# # =========================
st.subheader(VI["allocation_title"])

topN = st.slider(VI["allocate_topN"], 5, 100, 20, 5)
alloc_base = rank_df.sort_values("priority_score", ascending=False).head(topN).copy()

w = alloc_base["priority_score"].values
w = w / (w.sum() + 1e-9)

alloc_base["teams_alloc"] = np.floor(w * teams_total).astype(int)
alloc_base["boats_alloc"] = np.floor(w * boats_total).astype(int)
alloc_base["trucks_alloc"] = np.floor(w * trucks_total).astype(int)

alloc_base["food_packs_need"] = (alloc_base["affected_population"] * 3).astype(int)
alloc_base["water_liters_need"] = (alloc_base["affected_population"] * 2).astype(int)
alloc_base["medical_kits_need"] = (alloc_base["affected_population"] / 20).astype(int)

st.write(VI["suggested_alloc"])
alloc_show = alloc_base[[
    "GID_3", "name", "priority_score", "affected_population",
    "teams_alloc", "boats_alloc", "trucks_alloc",
    "food_packs_need", "water_liters_need", "medical_kits_need"
]].head(50)

st.dataframe(df_vi(alloc_show), use_container_width=True)

if st.button(VI["save_alloc"], key="save_alloc_btn"):
    for _, r in alloc_base.iterrows():
        upsert_allocation(r["GID_3"], {
            "teams": int(r["teams_alloc"]),
            "boats": int(r["boats_alloc"]),
            "trucks": int(r["trucks_alloc"]),
            "food_packs": int(r["food_packs_need"]),
            "water_liters": int(r["water_liters_need"]),
            "medical_kits": int(r["medical_kits_need"])
        })
    st.success(VI["alloc_saved"])

alloc_rows = get_all_allocations()
alloc_df = pd.DataFrame(
    alloc_rows,
    columns=["GID_3", "teams", "boats", "trucks", "food_packs", "water_liters", "medical_kits", "last_update"]
) if alloc_rows else pd.DataFrame()

if len(alloc_df):
    st.write(VI["saved_alloc_title"])
    alloc_df_show = alloc_df.copy()
    if "last_update" in alloc_df_show.columns:
        alloc_df_show = alloc_df_show.sort_values("last_update", ascending=False)
    st.dataframe(df_vi(alloc_df_show).head(100), use_container_width=True)

# # =========================
# # HỖ TRỢ SCHEDULER / CACHE
# # =========================
st.subheader(VI["scheduler_title"])
st.write(VI["scheduler_help"])

if st.button(VI["show_cache_path"], key="show_cache_path_btn"):
    today = datetime.now().strftime("%Y-%m-%d")
    st.code(str(CACHE_DIR / f"{today}_commune_forecasts_7d.parquet"))


import os, sys, json, math
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

# -------------------------
# CHUẨN HOÁ PATH THEO PROJECT ROOT (AUTO-DETECT)
# -------------------------
APP_DIR = Path(__file__).resolve().parent  # thư mục chứa file app6.py

def find_project_root(start: Path) -> Path:
    """
    Tìm root project bằng cách đi ngược lên cho tới khi thấy:
    - cache/merged.csv hoặc
    - data/resources_default.json hoặc
    - thư mục cache và data cùng tồn tại
    """
    candidates = [start] + list(start.parents)
    for p in candidates:
        if (p / "cache" / "merged.csv").exists():
            return p
        if (p / "data" / "resources_default.json").exists():
            return p
        if (p / "cache").exists() and (p / "data").exists():
            return p
    return start  # fallback

PROJECT_DIR = find_project_root(APP_DIR)

# Add import paths
SRC_DIR = PROJECT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# -------------------------
# IMPORT nội bộ
# -------------------------
from predict import predict_7days
from db_utils import (
    init_db, get_incident, upsert_incident,
    get_all_incidents, upsert_allocation, get_all_allocations,
    create_request, list_requests, update_request_status,
    create_task, list_tasks, update_task,
    upsert_shelter_status, list_shelter_status,
    create_shelter_move, list_shelter_moves
)
from report_utils import export_csv, export_pdf

# -------------------------
# FILE PATHS (relative to PROJECT_DIR)
# -------------------------
MERGED_FILE = PROJECT_DIR / "cache" / "merged.csv"
POP_FILE = PROJECT_DIR / "data" / "DanSo_Xa.csv"
SHELTER_FILE = PROJECT_DIR / "data" / "shelters.csv"
CACHE_DIR = PROJECT_DIR / "cache" / "forecasts"

(PROJECT_DIR / "cache").mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_DIR / "data").mkdir(parents=True, exist_ok=True)

# --------- THIẾT LẬP ----------
ALERT_THRESH_P = 0.7
ALERT_THRESH_AFFECTED = 500
ALERT_THRESH_PRIORITY = 80

# =========================
# TỪ ĐIỂN GIAO DIỆN (VI)
# =========================
VI = {
    "app_title": "🌊 Ứng dụng hỗ trợ cứu hộ ngập lụt — Nghệ An (Thời gian thực + Điều hành)",
    "sidebar_panel": "⚙️ Bảng điều khiển",
    "use_default_thresholds": "Dùng ngưỡng cảnh báo mặc định",
    "alert_p": "Cảnh báo: Xác suất ngập p_flood ≥",
    "alert_affected": "Cảnh báo: Dân bị ảnh hưởng ≥",
    "alert_priority": "Cảnh báo: Điểm ưu tiên ≥",

    "total_resources": "🚤 Tổng nguồn lực (phân bổ)",
    "teams": "Đội cứu hộ",
    "boats": "Thuyền",
    "trucks": "Xe tải",
    "food": "Gói lương thực",
    "water": "Lít nước",
    "med": "Bộ y tế",

    "alert_center": "🚨 Trung tâm cảnh báo (tự động)",
    "alerts_count": "Số cảnh báo",
    "max_priority": "Ưu tiên cao nhất",
    "max_affected": "Dân ảnh hưởng cao nhất",
    "alerts_table_title": "### 📋 Danh sách cảnh báo (tối đa 50)",

    "priority_map": "🗺️ Bản đồ ưu tiên (Hôm nay) — Bấm xã để xem dự báo 7 ngày + điều hành cứu hộ",
    "tooltip_name": "Xã:",
    "tooltip_priority": "Ưu tiên:",

    "top20": "## 📌 Top 20 xã ưu tiên (hôm nay)",
    "export_report": "## 🧾 Xuất báo cáo (CSV/PDF)",
    "export_csv_btn": "Xuất CSV (Top 50)",
    "export_pdf_btn": "Xuất PDF (Top 20 + Tóm tắt)",
    "csv_saved": "✅ Đã lưu CSV:",
    "pdf_saved": "✅ Đã lưu PDF:",

    "using_cache": "✅ Đang dùng dự báo cache (scheduler) cho hôm nay.",
    "no_cache": "⚠️ Không có cache. App đang tính dự báo TRỰC TIẾP (chậm, có thể bị giới hạn). Hãy chạy scheduler_cache.py mỗi ngày!",

    "selected": "✅ Đã chọn:",
    "today_overview": "### 📊 Tổng quan hôm nay",
    "p_flood": "Xác suất ngập",
    "area_m2": "Diện tích ngập (m²)",
    "ratio": "Tỉ lệ ngập",
    "affected": "Dân bị ảnh hưởng",
    "priority": "Ưu tiên",

    "forecast_7d": "### 📈 Dự báo 7 ngày",
    "chart_title": "### 📉 Biểu đồ xu hướng 7 ngày",
    "chart_p_flood": "Xác suất ngập",
    "chart_ratio": "Tỉ lệ ngập",
    "chart_priority": "Điểm ưu tiên",

    "incident_title": "## 🧩 Theo dõi sự cố (trạng thái cứu hộ)",
    "status_label": "Trạng thái",
    "note_label": "Ghi chú",
    "update_incident": "Cập nhật trạng thái sự cố",
    "incident_updated": "✅ Đã cập nhật sự cố!",

    "shelter_title": "## 🏠 Điểm trú ẩn (gần nhất)",
    "shelters_in_commune": "Điểm trú ẩn trong xã:",
    "nearest_shelters": "Điểm trú ẩn gần nhất:",
    "no_shelter_file": "Chưa có shelters.csv. Thêm data/shelters.csv để bật tính năng này.",

    "click_hint": "👆 Hãy bấm vào đa giác một xã trên bản đồ để xem chi tiết & điều hành cứu hộ.",

    "allocation_title": "🛠️ Phân bổ nguồn lực (tự động + chỉnh tay)",
    "allocate_topN": "Phân bổ cho Top N xã",
    "suggested_alloc": "### Gợi ý phân bổ (Đội/Thuyền/Xe) + Ước tính nhu cầu",
    "save_alloc": "✅ Lưu phân bổ vào DB (Top N)",
    "alloc_saved": "✅ Đã lưu phân bổ!",
    "saved_alloc_title": "### Phân bổ đã lưu (DB)",

    "scheduler_title": "⏱️ Lịch chạy / Cache (cập nhật hằng ngày)",
    "scheduler_help": """
**Khuyến nghị vận hành thực tế:**
- Mỗi ngày (hoặc 3 giờ/lần), chạy `python scheduler_cache.py` để tạo cache parquet.
- App sẽ đọc cache để hiển thị nhanh, tránh bị rate-limit từ Open-Meteo.
""",
    "show_cache_path": "Hiển thị đường dẫn cache dự kiến hôm nay",

    # ---- Nhãn cột hiển thị tiếng Việt ----
    "col_gid": "Mã xã (GID_3)",
    "col_name": "Tên xã",
    "col_p_flood": "Xác suất ngập",
    "col_ratio": "Tỉ lệ ngập",
    "col_affected": "Dân bị ảnh hưởng",
    "col_priority_score": "Điểm ưu tiên",
    "col_priority_level": "Mức ưu tiên",
    "col_area_m2": "Diện tích ngập (m²)",
    "col_last_update": "Cập nhật lúc",

    "col_teams_alloc": "Đội (phân bổ)",
    "col_boats_alloc": "Thuyền (phân bổ)",
    "col_trucks_alloc": "Xe tải (phân bổ)",
    "col_food_need": "Nhu cầu lương thực (gói)",
    "col_water_need": "Nhu cầu nước (lít)",
    "col_med_need": "Nhu cầu y tế (bộ)",

    # ---- Intake / Dispatch / Shelter ----
    "intake_title": "📞 Tiếp nhận yêu cầu cứu hộ (Hotline)",
    "intake_new": "➕ Tạo yêu cầu cứu hộ mới",
    "caller_name": "Họ tên người báo",
    "phone": "Số điện thoại",
    "gid3": "Mã xã (GID_3)",
    "commune_name": "Tên xã",
    "lat": "Vĩ độ (lat) (nếu có)",
    "lon": "Kinh độ (lon) (nếu có)",
    "people": "Số người cần hỗ trợ",
    "urgency": "Mức khẩn cấp (1 thấp → 5 rất khẩn)",
    "req_note": "Mô tả tình huống / địa điểm chi tiết",
    "req_submit": "📩 Ghi nhận yêu cầu",
    "req_list": "### 📋 Danh sách yêu cầu gần đây",

    "dispatch_title": "🧭 Điều phối nhiệm vụ (Tạo & Cập nhật)",
    "create_task_from_req": "⚡ Tạo nhiệm vụ từ yêu cầu (chọn request → tạo task)",
    "create_task_manual": "➕ Tạo nhiệm vụ thủ công (theo xã đang chọn)",
    "task_list": "### 📋 Danh sách nhiệm vụ",
    "task_update": "✏️ Cập nhật nhiệm vụ",

    "shelter_manage_title": "🏠 Quản lý điểm trú ẩn (còn chỗ + nhu cầu + chuyển người)",
    "shelter_status_edit": "✏️ Cập nhật trạng thái điểm trú ẩn",
    "shelter_status_table": "### 📋 Trạng thái điểm trú ẩn",
    "shelter_move": "🚐 Chuyển người đến điểm trú ẩn (tạo log)",
    "shelter_move_history": "### 🧾 Lịch sử chuyển người",

    # hiển thị bổ sung
    "col_commune_name": "Tên xã (suy ra)",
    "col_start_time": "Thời gian bắt đầu",
    "col_from_commune": "Tên xã đi (suy ra)",
    "col_to_gid3": "Mã xã đến (suy ra)",
    "col_to_commune": "Tên xã đến (suy ra)",
}

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(layout="wide")
st.title(VI["app_title"])

# Debug paths
st.sidebar.caption(f"📁 PROJECT_DIR = {PROJECT_DIR}")
st.sidebar.caption(f"📄 MERGED_FILE  = {MERGED_FILE}")

# init DB
init_db()

# =========================
HELPERS
# =========================
def safe_index(options, value, default=0):
    try:
        return options.index(value)
    except Exception:
        return default

def normalize_str(x):
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass
    return str(x).strip()

def to_int_safe(x, default=0):
    v = pd.to_numeric(x, errors="coerce")
    return int(default if pd.isna(v) else v)

def normalize_name_key(x: str) -> str:
    return normalize_str(x).lower()

# =========================
# VIỆT HOÁ TÊN CỘT HIỂN THỊ
# =========================
DISPLAY_COLS = {
    "GID_3": VI["col_gid"],
    "name": VI["col_name"],
    "p_flood": VI["col_p_flood"],
    "flood_ratio": VI["col_ratio"],
    "affected_population": VI["col_affected"],
    "priority_score": VI["col_priority_score"],
    "priority_level": VI["col_priority_level"],
    "flood_area_m2": VI["col_area_m2"],
    "last_update": VI["col_last_update"],

    "teams_alloc": VI["col_teams_alloc"],
    "boats_alloc": VI["col_boats_alloc"],
    "trucks_alloc": VI["col_trucks_alloc"],
    "food_packs_need": VI["col_food_need"],
    "water_liters_need": VI["col_water_need"],
    "medical_kits_need": VI["col_med_need"],

    "teams": VI["col_teams_alloc"],
    "boats": VI["col_boats_alloc"],
    "trucks": VI["col_trucks_alloc"],
    "food_packs": VI["col_food_need"],
    "water_liters": VI["col_water_need"],
    "medical_kits": VI["col_med_need"],

    "commune_name_derived": VI["col_commune_name"],
    "start_time": VI["col_start_time"],
    "from_commune_derived": VI["col_from_commune"],
    "to_gid3_derived": VI["col_to_gid3"],
    "to_commune_derived": VI["col_to_commune"],
}

def df_vi(df_: pd.DataFrame) -> pd.DataFrame:
    if df_ is None or len(df_) == 0:
        return df_
    return df_.rename(columns=DISPLAY_COLS)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_merged():
    if not MERGED_FILE.exists():
        st.error(f"❌ Không tìm thấy dữ liệu tổng hợp: {MERGED_FILE}")
        st.info("👉 Kiểm tra bạn có `cache/merged.csv` ở thư mục project.")
        st.stop()

    df0 = pd.read_csv(MERGED_FILE)
    # nếu dataset dùng cột khác, bạn sửa tại đây
    df0["date_local"] = pd.to_datetime(df0["date_local"], errors="coerce")
    df0["GID_3"] = df0["GID_3"].astype("string")
    df0 = df0.dropna(subset=["date_local", "GID_3"])
    df0["GID_3"] = df0["GID_3"].astype(str)
    return df0

@st.cache_data
def load_population():
    if not POP_FILE.exists():
        st.error(f"❌ Không tìm thấy dữ liệu dân số: {POP_FILE}")
        st.info("👉 Kiểm tra bạn có `data/DanSo_Xa.csv` ở thư mục project.")
        st.stop()

    pop = pd.read_csv(POP_FILE)
    pop["GID_3"] = pop["GID_3"].astype(str)
    if "DanSo_sum" not in pop.columns:
        st.error("❌ File dân số thiếu cột `DanSo_sum`.")
        st.stop()
    return pop.set_index("GID_3")["DanSo_sum"].to_dict()

@st.cache_data
def load_shelters():
    if SHELTER_FILE.exists():
        s = pd.read_csv(SHELTER_FILE)
        for col in ["name", "type", "lat", "lon", "capacity", "commune_gid3"]:
            if col not in s.columns:
                s[col] = np.nan
        return s
    return pd.DataFrame(columns=["name", "type", "lat", "lon", "capacity", "commune_gid3"])

df = load_merged()
pop_dict = load_population()
shelters_df = load_shelters()

# =========================
# BUILD STATIC (LẤY THÔNG TIN XÃ)
# =========================
static = (
    df.sort_values("date_local")
      .drop_duplicates(subset=["GID_3"], keep="first")
      .copy()
      .set_index("GID_3")
)

# Tạo map gid<->name để auto-fill intake + enrich bảng
gid_to_name = {}
namekey_to_gid = {}
for gid in static.index.astype(str).tolist():
    row = static.loc[gid]
    name = row.get("NAME_3_flood", gid)
    name = normalize_str(name) if name is not None else gid
    gid_to_name[str(gid)] = name

    k = normalize_name_key(name)
    if k and k not in namekey_to_gid:
        namekey_to_gid[k] = str(gid)

def derive_commune_name(gid3: str) -> str:
    g = normalize_str(gid3)
    return gid_to_name.get(g, "")

# =========================
# CACHE LOADER (7 DAYS) - linh hoạt tên file
# =========================
def load_today_cache():
    today = datetime.now().strftime("%Y-%m-%d")

    p1 = CACHE_DIR / f"{today}_commune_forecasts_7d.parquet"
    p2 = CACHE_DIR / f"{today}_commune_forecasts.parquet"

    path = p1 if p1.exists() else (p2 if p2.exists() else None)
    if path is None:
        return None

    try:
        cache = pd.read_parquet(path)
    except Exception as e:
        st.warning(f"⚠️ Không đọc được cache parquet: {path}\n\nChi tiết: {e}")
        return None

    cache["GID_3"] = cache["GID_3"].astype(str)
    cache["date_local"] = pd.to_datetime(cache["date_local"], errors="coerce")
    cache = cache.dropna(subset=["GID_3", "date_local"])
    return cache

cache_df = load_today_cache()

# =========================
# PRIORITY COMPUTATION
# =========================
@st.cache_data(ttl=3600)
def compute_today_priority_live(df_in, static_in, pop_dict_in):
    results = []
    for gid in static_in.index:
        row = static_in.loc[gid]
        lat = float(row["commune_lat"])
        lon = float(row["commune_lon"])
        population = pop_dict_in.get(str(gid), None)

        hist = df_in[df_in["GID_3"] == str(gid)].sort_values("date_local").tail(60)
        pred = predict_7days(str(gid), lat, lon, row, hist, population=population)
        today = pred.iloc[0]

        results.append({
            "GID_3": str(gid),
            "name": row.get("NAME_3_flood", str(gid)),
            "p_flood": float(today["p_flood"]),
            "flood_ratio": float(today["flood_ratio"]),
            "affected_population": float(today["affected_population"]) if pd.notna(today["affected_population"]) else 0.0,
            "priority_score": float(today["priority_score"]),
            "priority_level": today["priority_level"],
            "flood_area_m2": float(today.get("flood_area_m2", 0)) if pd.notna(today.get("flood_area_m2", 0)) else 0.0,
            "last_update": datetime.now().isoformat(timespec="seconds"),
        })
    return pd.DataFrame(results)

def compute_today_priority_from_cache(cache_in: pd.DataFrame) -> pd.DataFrame:
    today_date = cache_in["date_local"].min()
    first_day = cache_in[cache_in["date_local"] == today_date].copy()
    cols = ["GID_3", "name", "p_flood", "flood_ratio", "affected_population", "priority_score", "priority_level"]
    if "flood_area_m2" in first_day.columns:
        cols.append("flood_area_m2")
    out = first_day[cols].copy()
    out["last_update"] = datetime.now().isoformat(timespec="seconds")
    return out

if cache_df is not None and len(cache_df):
    rank_df = compute_today_priority_from_cache(cache_df)
    st.info(VI["using_cache"])
else:
    rank_df = compute_today_priority_live(df, static, pop_dict)
    st.warning(VI["no_cache"])

rank_df["GID_3"] = rank_df["GID_3"].astype(str)
priority_map = rank_df.set_index("GID_3")["priority_score"].to_dict()

# map ưu tiên theo gid
gid_to_priority = {}
for _, r in rank_df.iterrows():
    gid_to_priority[str(r["GID_3"])] = float(r.get("priority_score") or 0)

# =========================
# COLOR SCALE
# =========================
def score_to_color(score: float) -> str:
    score = float(score or 0)
    if score >= 80:
        return "#d7191c"
    if score >= 50:
        return "#fdae61"
    if score >= 20:
        return "#ffffbf"
    return "#a6d96a"

# =========================
INCIDENTS & ALLOCATIONS
# =========================
inc_rows = get_all_incidents()
inc_df = (
    pd.DataFrame(inc_rows, columns=["GID_3", "status", "last_update", "note"])
    if inc_rows else pd.DataFrame(columns=["GID_3", "status", "last_update", "note"])
)

alloc_rows = get_all_allocations()
alloc_df = (
    pd.DataFrame(
        alloc_rows,
        columns=["GID_3", "teams", "boats", "trucks", "food_packs", "water_liters", "medical_kits", "last_update"]
    )
    if alloc_rows else pd.DataFrame()
)

# =========================
# SIDEBAR CONTROL PANEL
# =========================
st.sidebar.header(VI["sidebar_panel"])

use_thresholds = st.sidebar.checkbox(VI["use_default_thresholds"], value=True)
if not use_thresholds:
    ALERT_THRESH_P = st.sidebar.slider(VI["alert_p"], 0.0, 1.0, 0.7, 0.05)
    ALERT_THRESH_AFFECTED = st.sidebar.number_input(VI["alert_affected"], value=500, step=50)
    ALERT_THRESH_PRIORITY = st.sidebar.number_input(VI["alert_priority"], value=80, step=5)

st.sidebar.divider()
st.sidebar.subheader(VI["total_resources"])
teams_total = st.sidebar.number_input(VI["teams"], 0, 9999, 10, 1)
boats_total = st.sidebar.number_input(VI["boats"], 0, 9999, 15, 1)
trucks_total = st.sidebar.number_input(VI["trucks"], 0, 9999, 8, 1)
food_total = st.sidebar.number_input(VI["food"], 0, 9999999, 5000, 100)
water_total = st.sidebar.number_input(VI["water"], 0, 9999999, 10000, 200)
med_total = st.sidebar.number_input(VI["med"], 0, 9999999, 600, 20)

# =========================
# ALERT CENTER
# =========================
st.subheader(VI["alert_center"])

alerts = rank_df[
    (rank_df["p_flood"] >= ALERT_THRESH_P) |
    (rank_df["affected_population"] >= ALERT_THRESH_AFFECTED) |
    (rank_df["priority_score"] >= ALERT_THRESH_PRIORITY)
].copy().sort_values("priority_score", ascending=False)

cA, cB, cC = st.columns(3)
cA.metric(VI["alerts_count"], len(alerts))
cB.metric(VI["max_priority"], f"{alerts['priority_score'].max():.1f}" if len(alerts) else "0")
cC.metric(VI["max_affected"], f"{alerts['affected_population'].max():.0f}" if len(alerts) else "0")

st.write(VI["alerts_table_title"])
st.dataframe(df_vi(alerts.head(50)), use_container_width=True)

# =========================
# MAP (Priority choropleth)
# =========================
st.subheader(VI["priority_map"])

features = []
for gid, row in static.iterrows():
    if ".geo" not in row or pd.isna(row[".geo"]):
        continue
    try:
        geom = json.loads(row[".geo"])
    except Exception:
        continue

    name = row.get("NAME_3_flood", gid)
    score = float(priority_map.get(str(gid), 0.0))

    features.append({
        "type": "Feature",
        "properties": {"GID_3": str(gid), "name": str(name), "priority_score": round(score, 2)},
        "geometry": geom
    })

geojson = {"type": "FeatureCollection", "features": features}

center_lat = float(static["commune_lat"].mean())
center_lon = float(static["commune_lon"].mean())

m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="CartoDB positron")

def style_fn(feature):
    score = feature["properties"].get("priority_score", 0)
    return {"fillColor": score_to_color(float(score)), "color": "black", "weight": 0.4, "fillOpacity": 0.6}

folium.GeoJson(
    geojson,
    name="Các xã",
    style_function=style_fn,
    tooltip=folium.GeoJsonTooltip(
        fields=["name", "priority_score"],
        aliases=[VI["tooltip_name"], VI["tooltip_priority"]],
        sticky=True
    )
).add_to(m)

map_data = st_folium(m, width=1200, height=650)

selected_gid = None
selected_name = None
if map_data:
    lad = map_data.get("last_active_drawing")
    if isinstance(lad, dict):
        props = lad.get("properties") or (lad.get("feature") or {}).get("properties") or {}
        if isinstance(props, dict):
            selected_gid = props.get("GID_3") or props.get("gid3") or props.get("gid_3")
            selected_name = props.get("name") or props.get("NAME_3") or props.get("commune_name")

# =========================
LEFT: Ranking + Export
# =========================
left, right = st.columns([1, 1])

with left:
    st.write(VI["top20"])
    top20 = rank_df.sort_values("priority_score", ascending=False).head(20)
    st.dataframe(df_vi(top20), use_container_width=True)

    st.write(VI["export_report"])
    if st.button(VI["export_csv_btn"]):
        out_path = PROJECT_DIR / "cache" / f"report_top50_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export_csv(rank_df.sort_values("priority_score", ascending=False).head(50), str(out_path))
        st.success(f"{VI['csv_saved']} {out_path}")

    if st.button(VI["export_pdf_btn"]):
        out_path = PROJECT_DIR / "cache" / f"report_top20_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        summary = {
            "Số cảnh báo": len(alerts),
            "Điểm ưu tiên cao nhất": float(rank_df["priority_score"].max()),
            "Tổng dân bị ảnh hưởng (top20)": float(top20["affected_population"].sum()),
        }
        export_pdf(top20, summary, str(out_path))
        st.success(f"{VI['pdf_saved']} {out_path}")

# =========================
# RIGHT: Commune details
# =========================
with right:
    if selected_gid:
        st.success(f"{VI['selected']} {selected_name} ({selected_gid})")

        row = static.loc[str(selected_gid)]
        lat = float(row["commune_lat"])
        lon = float(row["commune_lon"])
        population = pop_dict.get(str(selected_gid), None)

        if cache_df is not None and len(cache_df):
            pred = cache_df[cache_df["GID_3"].astype(str) == str(selected_gid)].sort_values("date_local").copy()
        else:
            hist = df[df["GID_3"].astype(str) == str(selected_gid)].sort_values("date_local").tail(60)
            pred = predict_7days(str(selected_gid), lat, lon, row, hist, population=population)

        today = pred.iloc[0]

        st.write(VI["today_overview"])
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(VI["p_flood"], f"{today['p_flood']:.2%}")
        c2.metric(VI["area_m2"], f"{today.get('flood_area_m2', 0):.0f}")
        c3.metric(VI["ratio"], f"{today['flood_ratio']:.2%}")
        c4.metric(VI["affected"], f"{today['affected_population']:.0f}")
        c5.metric(VI["priority"], f"{today['priority_level']} ({today['priority_score']:.1f})")

        st.write(VI["forecast_7d"])
        st.dataframe(df_vi(pred), use_container_width=True)

        st.write(VI["chart_title"])
        chart_df = pred.set_index("date_local")[["p_flood", "flood_ratio", "priority_score"]].copy()
        chart_df = chart_df.rename(columns={
            "p_flood": VI["chart_p_flood"],
            "flood_ratio": VI["chart_ratio"],
            "priority_score": VI["chart_priority"],
        })
        st.line_chart(chart_df)

        # -------- INCIDENT --------
        st.write(VI["incident_title"])
        status_options = ["CHƯA TRIỂN KHAI", "ĐANG TRIỂN KHAI", "HOÀN TẤT", "CẦN HỖ TRỢ THÊM"]
        current = get_incident(str(selected_gid))
        current_status = (current[1] if current else "CHƯA TRIỂN KHAI") or "CHƯA TRIỂN KHAI"
        current_status = str(current_status).strip()
        current_note = current[3] if current else ""

        idx0 = safe_index(status_options, current_status, default=0)
        new_status = st.selectbox(VI["status_label"], status_options, index=idx0, key="incident_status")
        new_note = st.text_area(VI["note_label"], value=str(current_note or ""), height=80, key="incident_note")

        if st.button(VI["update_incident"], key="incident_update_btn"):
            upsert_incident(str(selected_gid), new_status, new_note)
            st.success(VI["incident_updated"])

        # -------- SHELTER LIST --------
        st.write(VI["shelter_title"])

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            p = math.pi / 180
            a = (
                0.5 - math.cos((lat2 - lat1) * p) / 2
                + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
            )
            return 2 * R * math.asin(math.sqrt(a))

        if len(shelters_df):
            s_same = shelters_df[shelters_df["commune_gid3"].astype(str) == str(selected_gid)].copy()
        else:
            s_same = pd.DataFrame()

        if len(s_same) > 0:
            st.write(VI["shelters_in_commune"])
            st.dataframe(s_same[["name", "type", "capacity", "lat", "lon"]], use_container_width=True)
        elif len(shelters_df) > 0:
            shelters_df2_tmp = shelters_df.dropna(subset=["lat", "lon"]).copy()
            if len(shelters_df2_tmp):
                shelters_df2_tmp["dist_km"] = shelters_df2_tmp.apply(
                    lambda r: haversine(lat, lon, float(r["lat"]), float(r["lon"])),
                    axis=1
                )
                nearest = shelters_df2_tmp.sort_values("dist_km").head(5)
                st.write(VI["nearest_shelters"])
                st.dataframe(nearest[["name", "type", "capacity", "dist_km", "lat", "lon"]], use_container_width=True)
            else:
                st.info(VI["no_shelter_file"])
        else:
            st.info(VI["no_shelter_file"])
    else:
        st.info(VI["click_hint"])

# ============================================================
# ✅ TÍCH HỢP 3 TÍNH NĂNG: Intake → Dispatch → Shelter status
# ============================================================

st.subheader(VI["intake_title"])

def intake_fill_from_gid():
    gid = normalize_str(st.session_state.get("req_gid3", ""))
    if gid:
        st.session_state["req_commune_name"] = derive_commune_name(gid)

def intake_fill_from_name():
    name = normalize_str(st.session_state.get("req_commune_name", ""))
    if not name:
        return
    gid = namekey_to_gid.get(normalize_name_key(name), "")
    if gid:
        st.session_state["req_gid3"] = gid
    st.session_state["req_commune_name"] = derive_commune_name(st.session_state.get("req_gid3", "")) or name

with st.expander(VI["intake_new"], expanded=True):
    c1, c2 = st.columns([1, 1])
    caller_name = c1.text_input(VI["caller_name"], key="req_caller")
    phone = c2.text_input(VI["phone"], key="req_phone")

    c3, c4 = st.columns([1, 1])
    st.text_input(VI["gid3"], value=selected_gid or "", key="req_gid3", on_change=intake_fill_from_gid)
    st.text_input(VI["commune_name"], value=selected_name or "", key="req_commune_name", on_change=intake_fill_from_name)

    c5, c6, c7 = st.columns([1, 1, 1])
    lat_req = c5.number_input(VI["lat"], value=0.0, format="%.6f", key="req_lat")
    lon_req = c6.number_input(VI["lon"], value=0.0, format="%.6f", key="req_lon")
    people = c7.number_input(VI["people"], min_value=0, value=1, step=1, key="req_people")

    urgency = st.slider(VI["urgency"], 1, 5, 4, 1, key="req_urgency")
    note_req = st.text_area(VI["req_note"], height=80, key="req_note")

    if st.button(VI["req_submit"], key="req_submit"):
        gid3_req = normalize_str(st.session_state.get("req_gid3", ""))
        name_req = normalize_str(st.session_state.get("req_commune_name", ""))

        if (not gid3_req) and name_req:
            gid3_req = namekey_to_gid.get(normalize_name_key(name_req), "")

        rid = create_request({
            "caller_name": caller_name,
            "phone": phone,
            "gid3": gid3_req if gid3_req else None,
            "lat": float(lat_req) if float(lat_req) != 0 else None,
            "lon": float(lon_req) if float(lon_req) != 0 else None,
            "people": int(people),
            "urgency": int(urgency),
            "note": note_req,
            "status": "MỚI"
        })
        st.success(f"✅ Đã tạo yêu cầu cứu hộ ID = {rid}")

st.write(VI["req_list"])
req_rows = list_requests(limit=200)
req_df = pd.DataFrame(req_rows, columns=[
    "id", "created_at", "caller_name", "phone", "gid3", "lat", "lon", "people",
    "urgency", "note", "status", "linked_task_id"
])

if len(req_df):
    req_df["gid3"] = req_df["gid3"].astype(str).replace("None", "")
    req_df["commune_name_derived"] = req_df["gid3"].apply(lambda g: derive_commune_name(g) if g and g != "nan" else "")
st.dataframe(df_vi(req_df), use_container_width=True)

# =========================
# (2) TASKING / DISPATCH
# =========================
st.subheader(VI["dispatch_title"])

with st.expander(VI["create_task_from_req"], expanded=True):
    if len(req_df):
        req_id = st.selectbox("Chọn Request ID", req_df["id"].tolist(), key="req_pick")
        req_row = req_df[req_df["id"] == req_id].iloc[0].to_dict()

        gid3 = normalize_str(req_row.get("gid3")) or normalize_str(selected_gid)
        commune_name = derive_commune_name(gid3) if gid3 else ""
        priority_score = float(gid_to_priority.get(gid3, 0.0)) if gid3 else 0.0

        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        task_type = c1.selectbox("Loại nhiệm vụ", ["CỨU HỘ", "SƠ TÁN", "CẤP PHÁT", "KHẢO SÁT", "Y TẾ"], key="task_type_from_req")
        assigned_team = c2.text_input("Đội phụ trách (tên kíp/đội)", value="", key="task_team_from_req")
        boats = c3.number_input("Số thuyền điều", 0, 999, 0, 1, key="task_boats_from_req")
        trucks = c4.number_input("Số xe tải điều", 0, 999, 0, 1, key="task_trucks_from_req")

        eta_min = st.number_input("ETA (phút) ước tính", 0, 9999, 0, 5, key="task_eta_from_req")
        note_task = st.text_area("Ghi chú nhiệm vụ", value=req_row.get("note") or "", height=80, key="task_note_from_req")

        st.info(f"📍 Xã: **{commune_name or '(chưa rõ)'}** | Mã xã: **{gid3 or '(trống)'}** | Điểm ưu tiên: **{priority_score:.1f}**")

        if st.button("✅ Tạo nhiệm vụ & gắn với request", key="create_task_from_req_btn"):
            now_iso = datetime.now().isoformat(timespec="seconds")
            will_dispatch = bool(assigned_team.strip() or int(boats) > 0 or int(trucks) > 0)
            init_status = "ĐÃ GIAO" if will_dispatch else "MỚI"
            start_time = now_iso if will_dispatch else None

            tid = create_task({
                "gid3": gid3 if gid3 else None,
                "commune_name": commune_name if commune_name else None,
                "task_type": task_type,
                "priority_score": priority_score,
                "assigned_team": assigned_team.strip() if assigned_team.strip() else None,
                "boats": int(boats),
                "trucks": int(trucks),
                "status": init_status,
                "eta_min": int(eta_min),
                "note": note_task,
                "start_time": start_time,
                "source_request_id": int(req_id),
            })
            update_request_status(int(req_id), "ĐÃ TẠO NHIỆM VỤ", linked_task_id=int(tid))
            st.success(f"✅ Đã tạo Task ID = {tid} từ Request ID = {req_id}")
    else:
        st.info("Chưa có yêu cầu để tạo nhiệm vụ.")

with st.expander(VI["create_task_manual"], expanded=False):
    gid3_manual = normalize_str(selected_gid) or st.text_input("Mã xã (GID_3)", key="manual_gid3")
    commune_name_manual = derive_commune_name(gid3_manual) or normalize_str(selected_name)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    task_type2 = c1.selectbox("Loại nhiệm vụ (thủ công)", ["CỨU HỘ", "SƠ TÁN", "CẤP PHÁT", "KHẢO SÁT", "Y TẾ"], key="manual_task_type")
    assigned_team2 = c2.text_input("Đội phụ trách", value="", key="manual_team")
    boats2 = c3.number_input("Thuyền", 0, 999, 0, 1, key="boats2")
    trucks2 = c4.number_input("Xe tải", 0, 999, 0, 1, key="trucks2")
    eta2 = st.number_input("ETA (phút)", 0, 9999, 0, 5, key="eta2")
    note2 = st.text_area("Ghi chú", height=80, key="note2")

    if st.button("✅ Tạo nhiệm vụ (thủ công)", key="create_manual_task"):
        pscore = float(gid_to_priority.get(gid3_manual, 0.0)) if gid3_manual else 0.0
        now_iso = datetime.now().isoformat(timespec="seconds")
        will_dispatch = bool(assigned_team2.strip() or int(boats2) > 0 or int(trucks2) > 0)
        init_status = "ĐÃ GIAO" if will_dispatch else "MỚI"
        start_time = now_iso if will_dispatch else None

        tid = create_task({
            "gid3": gid3_manual if gid3_manual else None,
            "commune_name": commune_name_manual if commune_name_manual else None,
            "task_type": task_type2,
            "priority_score": pscore,
            "assigned_team": assigned_team2.strip() if assigned_team2.strip() else None,
            "boats": int(boats2),
            "trucks": int(trucks2),
            "status": init_status,
            "eta_min": int(eta2),
            "note": note2,
            "start_time": start_time,
            "source_request_id": None,
        })
        st.success(f"✅ Đã tạo Task ID = {tid}")

st.write(VI["task_list"])
task_rows = list_tasks(limit=300)
task_df = pd.DataFrame(task_rows, columns=[
    "id","created_at","gid3","commune_name","task_type","priority_score",
    "assigned_team","boats","trucks","status","eta_min","start_time","end_time","note","source_request_id"
])

if len(task_df):
    task_df["gid3"] = task_df["gid3"].astype(str).replace("None", "")
    task_df["commune_name"] = task_df["commune_name"].astype(str).replace("None", "")
    task_df["commune_name_derived"] = task_df.apply(
        lambda r: normalize_str(r.get("commune_name")) or derive_commune_name(r.get("gid3")),
        axis=1
    )
    task_df["priority_score"] = pd.to_numeric(task_df["priority_score"], errors="coerce").fillna(
        task_df["gid3"].apply(lambda g: gid_to_priority.get(normalize_str(g), 0.0))
    )
st.dataframe(df_vi(task_df), use_container_width=True)

with st.expander(VI["task_update"], expanded=False):
    if len(task_df):
        task_id = st.selectbox("Chọn Task ID", task_df["id"].tolist(), key="task_pick")
        trow = task_df[task_df["id"] == task_id].iloc[0].to_dict()

        status_options = ["MỚI","ĐÃ GIAO","ĐANG ĐI","ĐANG THỰC HIỆN","HOÀN TẤT","HỦY"]
        cur_task_status = str(trow.get("status") or "").strip()
        idx_task = safe_index(status_options, cur_task_status, default=0)

        new_status = st.selectbox("Trạng thái mới", status_options, index=idx_task, key="task_new_status")
        team_new = st.text_input("Đội phụ trách", value=str(trow.get("assigned_team") or ""), key="task_new_team")
        boats_new = st.number_input("Thuyền", 0, 999, to_int_safe(trow.get("boats"), 0), 1, key="task_new_boats")
        trucks_new = st.number_input("Xe tải", 0, 999, to_int_safe(trow.get("trucks"), 0), 1, key="task_new_trucks")
        eta_new = st.number_input("ETA (phút)", 0, 9999, to_int_safe(trow.get("eta_min"), 0), 5, key="task_new_eta")
        note_new = st.text_area("Ghi chú", value=str(trow.get("note") or ""), height=80, key="task_new_note")

        start_time = None
        end_time = None
        now_iso = datetime.now().isoformat(timespec="seconds")

        if new_status in ["ĐANG ĐI","ĐANG THỰC HIỆN"] and not trow.get("start_time"):
            start_time = now_iso
        if new_status == "HOÀN TẤT":
            if not trow.get("start_time"):
                start_time = now_iso
            end_time = now_iso

        if st.button("✅ Lưu cập nhật Task", key="task_update_btn"):
            update_task(
                int(task_id),
                status=new_status,
                assigned_team=team_new.strip() if team_new.strip() else None,
                boats=int(boats_new),
                trucks=int(trucks_new),
                eta_min=int(eta_new),
                note=note_new,
                start_time=start_time,
                end_time=end_time
            )
            st.success("✅ Đã cập nhật nhiệm vụ.")
    else:
        st.info("Chưa có nhiệm vụ nào.")

# =========================
# (3) SHELTER STATUS + CHUYỂN NGƯỜI
# =========================
st.subheader(VI["shelter_manage_title"])

def make_shelter_key(r):
    gid = normalize_str(r.get("commune_gid3", ""))
    name = normalize_str(r.get("name", ""))
    return f"{gid}__{name}".lower()

if len(shelters_df):
    shelters_df2 = shelters_df.copy()
    shelters_df2["commune_gid3"] = shelters_df2["commune_gid3"].apply(normalize_str)
    shelters_df2["name"] = shelters_df2["name"].apply(normalize_str)
    shelters_df2["shelter_key"] = shelters_df2.apply(make_shelter_key, axis=1)

    shelters_df2["shelter_key"] = shelters_df2["shelter_key"].astype(str)
    shelters_df2 = shelters_df2[shelters_df2["shelter_key"].str.len() > 2].copy()
    shelters_df2 = shelters_df2.drop_duplicates(subset=["shelter_key"], keep="first").copy()

    shelters_df2["label"] = shelters_df2.apply(
        lambda r: f"{normalize_str(r.get('name'))} (GID_3: {normalize_str(r.get('commune_gid3'))})",
        axis=1
    )
    shelter_label_map = dict(zip(shelters_df2["shelter_key"].tolist(), shelters_df2["label"].tolist()))
else:
    shelters_df2 = pd.DataFrame(columns=["shelter_key","name","capacity","commune_gid3","lat","lon","type","label"])
    shelter_label_map = {}

with st.expander(VI["shelter_status_edit"], expanded=False):
    if len(shelters_df2):
        shelter_keys = shelters_df2["shelter_key"].astype(str).tolist()

        shelter_key = st.selectbox(
            "Chọn điểm trú ẩn",
            shelter_keys,
            format_func=lambda k: shelter_label_map.get(str(k), str(k)),
            key="pick_shelter_key"
        )

        srow = shelters_df2[shelters_df2["shelter_key"].astype(str) == str(shelter_key)].iloc[0].to_dict()

        cap_default = to_int_safe(srow.get("capacity", 0), 0)
        gid3_default = normalize_str(srow.get("commune_gid3"))
        commune_name_derived = derive_commune_name(gid3_default)

        st.info(f"📍 Xã: **{commune_name_derived or '(chưa rõ)'}** | Mã xã: **{gid3_default or '(trống)'}**")

        c1, c2, c3, c4 = st.columns([1,1,1,1])
        capacity = c1.number_input("Sức chứa", 0, 999999, cap_default, 10, key="sh_cap")
        current_people = c2.number_input("Đang có (người)", 0, 999999, 0, 10, key="sh_cur")
        need_food = c3.number_input("Nhu cầu lương thực (gói)", 0, 9999999, 0, 50, key="sh_food")
        need_water = c4.number_input("Nhu cầu nước (lít)", 0, 9999999, 0, 100, key="sh_water")
        need_med = st.number_input("Nhu cầu y tế (bộ)", 0, 9999999, 0, 10, key="sh_med")
        note = st.text_area("Ghi chú", height=80, key="sh_note")

        if st.button("✅ Lưu trạng thái điểm trú ẩn", key="save_shelter_status_btn"):
            upsert_shelter_status({
                "shelter_key": str(shelter_key),
                "name": normalize_str(srow.get("name")),
                "gid3": gid3_default,
                "capacity": int(capacity),
                "current_people": int(current_people),
                "need_food": int(need_food),
                "need_water": int(need_water),
                "need_med": int(need_med),
                "note": note
            })
            st.success("✅ Đã lưu trạng thái điểm trú ẩn.")
    else:
        st.info("Chưa có shelters.csv để quản lý điểm trú ẩn.")

st.write(VI["shelter_status_table"])
sh_rows = list_shelter_status(limit=500)
sh_df = pd.DataFrame(sh_rows, columns=[
    "shelter_key","name","gid3","capacity","current_people","need_food","need_water","need_med","note","last_update"
])
if len(sh_df):
    sh_df["gid3"] = sh_df["gid3"].astype(str).replace("None", "")
    sh_df["commune_name_derived"] = sh_df["gid3"].apply(lambda g: derive_commune_name(g) if g and g != "nan" else "")
    sh_df["còn_trống"] = sh_df["capacity"].fillna(0) - sh_df["current_people"].fillna(0)
st.dataframe(df_vi(sh_df), use_container_width=True)

with st.expander(VI["shelter_move"], expanded=True):
    if len(sh_df):
        from_gid3 = st.text_input("Từ xã (GID_3)", value=normalize_str(selected_gid) or "", key="mv_from_gid3")
        from_name = derive_commune_name(from_gid3)
        if from_gid3:
            st.caption(f"➡️ Tên xã đi (suy ra): {from_name or '(chưa rõ)'}")

        sh_df2 = sh_df.copy()
        sh_df2["shelter_key"] = sh_df2["shelter_key"].astype(str)
        sh_df2["name"] = sh_df2["name"].apply(normalize_str)
        mv_label_map = dict(zip(sh_df2["shelter_key"].tolist(), sh_df2["name"].tolist()))

        to_key = st.selectbox(
            "Đến điểm trú ẩn",
            sh_df2["shelter_key"].tolist(),
            format_func=lambda k: mv_label_map.get(str(k), str(k)),
            key="mv_to_key"
        )

        to_row = sh_df2[sh_df2["shelter_key"].astype(str) == str(to_key)].iloc[0].to_dict()
        to_gid3 = normalize_str(to_row.get("gid3"))
        to_commune_name = derive_commune_name(to_gid3)

        st.caption(f"🏁 Xã đến (suy ra): {to_commune_name or '(chưa rõ)'} | Mã xã: {to_gid3 or '(trống)'}")

        people_move = st.number_input("Số người chuyển", 0, 999999, 10, 1, key="mv_people")
        link_task = st.number_input("Gắn Task ID (nếu có)", 0, 999999, 0, 1, key="mv_task")
        note_move = st.text_input("Ghi chú chuyển", key="mv_note")

        if st.button("✅ Xác nhận chuyển", key="mv_confirm"):
            create_shelter_move({
                "from_gid3": from_gid3.strip() if from_gid3.strip() else None,
                "to_shelter_key": str(to_key),
                "people": int(people_move),
                "task_id": int(link_task) if int(link_task) > 0 else None,
                "note": note_move
            })

            cur = sh_df2[sh_df2["shelter_key"].astype(str) == str(to_key)].iloc[0].to_dict()
            new_current = to_int_safe(cur.get("current_people", 0), 0) + int(people_move)

            upsert_shelter_status({
                "shelter_key": str(to_key),
                "name": cur.get("name"),
                "gid3": cur.get("gid3"),
                "capacity": to_int_safe(cur.get("capacity", 0), 0),
                "current_people": int(new_current),
                "need_food": to_int_safe(cur.get("need_food", 0), 0),
                "need_water": to_int_safe(cur.get("need_water", 0), 0),
                "need_med": to_int_safe(cur.get("need_med", 0), 0),
                "note": cur.get("note")
            })

            st.success("✅ Đã ghi nhận chuyển người và cập nhật điểm trú ẩn.")
    else:
        st.info("Chưa có dữ liệu shelter_status để thực hiện chuyển người.")

st.write(VI["shelter_move_history"])
mv_rows = list_shelter_moves(limit=300)
mv_df = pd.DataFrame(mv_rows, columns=["id","created_at","from_gid3","to_shelter_key","people","task_id","note"])

if len(mv_df):
    mv_df["from_gid3"] = mv_df["from_gid3"].astype(str).replace("None", "")
    mv_df["from_commune_derived"] = mv_df["from_gid3"].apply(lambda g: derive_commune_name(g) if g and g != "nan" else "")

    if len(sh_df):
        sh_key_to_gid = dict(zip(sh_df["shelter_key"].astype(str).tolist(), sh_df["gid3"].astype(str).tolist()))
    else:
        sh_key_to_gid = {}

    mv_df["to_gid3_derived"] = mv_df["to_shelter_key"].astype(str).apply(lambda k: normalize_str(sh_key_to_gid.get(str(k), "")))
    mv_df["to_commune_derived"] = mv_df["to_gid3_derived"].apply(lambda g: derive_commune_name(g) if g else "")

st.dataframe(df_vi(mv_df), use_container_width=True)

# =========================
# PHÂN BỔ NGUỒN LỰC
# =========================
st.subheader(VI["allocation_title"])

topN = st.slider(VI["allocate_topN"], 5, 100, 20, 5)
alloc_base = rank_df.sort_values("priority_score", ascending=False).head(topN).copy()

w = alloc_base["priority_score"].values
w = w / (w.sum() + 1e-9)

alloc_base["teams_alloc"] = np.floor(w * teams_total).astype(int)
alloc_base["boats_alloc"] = np.floor(w * boats_total).astype(int)
alloc_base["trucks_alloc"] = np.floor(w * trucks_total).astype(int)

alloc_base["food_packs_need"] = (alloc_base["affected_population"] * 3).astype(int)
alloc_base["water_liters_need"] = (alloc_base["affected_population"] * 2).astype(int)
alloc_base["medical_kits_need"] = (alloc_base["affected_population"] / 20).astype(int)

st.write(VI["suggested_alloc"])
alloc_show = alloc_base[[
    "GID_3", "name", "priority_score", "affected_population",
    "teams_alloc", "boats_alloc", "trucks_alloc",
    "food_packs_need", "water_liters_need", "medical_kits_need"
]].head(50)

st.dataframe(df_vi(alloc_show), use_container_width=True)

if st.button(VI["save_alloc"], key="save_alloc_btn"):
    for _, r in alloc_base.iterrows():
        upsert_allocation(r["GID_3"], {
            "teams": int(r["teams_alloc"]),
            "boats": int(r["boats_alloc"]),
            "trucks": int(r["trucks_alloc"]),
            "food_packs": int(r["food_packs_need"]),
            "water_liters": int(r["water_liters_need"]),
            "medical_kits": int(r["medical_kits_need"])
        })
    st.success(VI["alloc_saved"])

alloc_rows = get_all_allocations()
alloc_df = pd.DataFrame(
    alloc_rows,
    columns=["GID_3", "teams", "boats", "trucks", "food_packs", "water_liters", "medical_kits", "last_update"]
) if alloc_rows else pd.DataFrame()

if len(alloc_df):
    st.write(VI["saved_alloc_title"])
    alloc_df_show = alloc_df.copy()
    if "last_update" in alloc_df_show.columns:
        alloc_df_show = alloc_df_show.sort_values("last_update", ascending=False)
    st.dataframe(df_vi(alloc_df_show).head(100), use_container_width=True)

# =========================
# HỖ TRỢ SCHEDULER / CACHE
# =========================
st.subheader(VI["scheduler_title"])
st.write(VI["scheduler_help"])

if st.button(VI["show_cache_path"], key="show_cache_path_btn"):
    today = datetime.now().strftime("%Y-%m-%d")
    st.code(str(CACHE_DIR / f"{today}_commune_forecasts_7d.parquet"))
