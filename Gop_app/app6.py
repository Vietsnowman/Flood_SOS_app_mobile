import sys
import json
import math
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

# -------------------------
# CHUẨN HOÁ PATH THEO PROJECT ROOT
# data/, cache/, models/ đều nằm TRONG Gop_app
# -------------------------
PROJECT_DIR = Path(__file__).resolve().parent  # Gop_app/

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from predict import predict_7days  # noqa: E402
from db_utils import (  # noqa: E402
    init_db, get_incident, upsert_incident,
    get_all_incidents, upsert_allocation, get_all_allocations,
    create_request, list_requests, update_request_status,
    create_task, list_tasks, update_task,
    upsert_shelter_status, list_shelter_status,
    create_shelter_move, list_shelter_moves
)
from report_utils import export_csv, export_pdf  # noqa: E402

# -------------------------
# FILE PATHS
# -------------------------
MERGED_FILE  = PROJECT_DIR / "cache" / "merged.csv"
POP_FILE     = PROJECT_DIR / "data"  / "DanSo_Xa.csv"
SHELTER_FILE = PROJECT_DIR / "data"  / "shelters.csv"
RESOURCES_FILE = PROJECT_DIR / "data" / "resources_default.json"
CACHE_DIR    = PROJECT_DIR / "cache" / "forecasts"

(CACHE_DIR.parent).mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_DIR / "data").mkdir(parents=True, exist_ok=True)

# -------------------------
# TẢI MẶC ĐỊNH NGUỒN LỰC TỪ resources_default.json
# -------------------------
_default_resources = {"teams": 10, "boats": 15, "trucks": 8,
                      "food_packs": 5000, "water_liters": 10000, "medical_kits": 600}
if RESOURCES_FILE.exists():
    try:
        with open(RESOURCES_FILE, "r", encoding="utf-8") as _rf:
            _default_resources = json.load(_rf)
    except Exception:
        pass  # giữ fallback

# # --------- THIáº¾T Láº¬P ----------
ALERT_THRESH_P = 0.7
ALERT_THRESH_AFFECTED = 500
ALERT_THRESH_PRIORITY = 80

# # =========================
# # Tá»ª ÄIá»‚N GIAO DIá»†N (VI) â€” 100% VIá»†T HOÃ
# # =========================
VI = {
    "app_title": "ðŸŒŠ á»¨ng dá»¥ng há»— trá»£ cá»©u há»™ ngáº­p lá»¥t â€” Nghá»‡ An (Thá»i gian thá»±c + Äiá»u hÃ nh)",
    "sidebar_panel": "âš™ï¸ Báº£ng Ä‘iá»u khiá»ƒn",
    "use_default_thresholds": "DÃ¹ng ngÆ°á»¡ng cáº£nh bÃ¡o máº·c Ä‘á»‹nh",
    "alert_p": "Cáº£nh bÃ¡o: XÃ¡c suáº¥t ngáº­p p_flood â‰¥",
    "alert_affected": "Cáº£nh bÃ¡o: DÃ¢n bá»‹ áº£nh hÆ°á»Ÿng â‰¥",
    "alert_priority": "Cáº£nh bÃ¡o: Äiá»ƒm Æ°u tiÃªn â‰¥",

    "total_resources": "ðŸš¤ Tá»•ng nguá»“n lá»±c (phÃ¢n bá»•)",
    "teams": "Äá»™i cá»©u há»™",
    "boats": "Thuyá»n",
    "trucks": "Xe táº£i",
    "food": "GÃ³i lÆ°Æ¡ng thá»±c",
    "water": "LÃ­t nÆ°á»›c",
    "med": "Bá»™ y táº¿",

    "alert_center": "ðŸš¨ Trung tÃ¢m cáº£nh bÃ¡o (tá»± Ä‘á»™ng)",
    "alerts_count": "Sá»‘ cáº£nh bÃ¡o",
    "max_priority": "Æ¯u tiÃªn cao nháº¥t",
    "max_affected": "DÃ¢n áº£nh hÆ°á»Ÿng cao nháº¥t",
    "alerts_table_title": "### ðŸ“‹ Danh sÃ¡ch cáº£nh bÃ¡o (tá»‘i Ä‘a 50)",

    "priority_map": "ðŸ—ºï¸ Báº£n Ä‘á»“ Æ°u tiÃªn (HÃ´m nay) â€” Báº¥m xÃ£ Ä‘á»ƒ xem dá»± bÃ¡o 7 ngÃ y + Ä‘iá»u hÃ nh cá»©u há»™",
    "tooltip_name": "XÃ£:",
    "tooltip_priority": "Æ¯u tiÃªn:",

    "top20": "## ðŸ“Œ Top 20 xÃ£ Æ°u tiÃªn (hÃ´m nay)",
    "export_report": "## ðŸ§¾ Xuáº¥t bÃ¡o cÃ¡o (CSV/PDF)",
    "export_csv_btn": "Xuáº¥t CSV (Top 50)",
    "export_pdf_btn": "Xuáº¥t PDF (Top 20 + TÃ³m táº¯t)",
    "csv_saved": "âœ… ÄÃ£ lÆ°u CSV:",
    "pdf_saved": "âœ… ÄÃ£ lÆ°u PDF:",

    "using_cache": "âœ… Äang dÃ¹ng dá»± bÃ¡o cache (scheduler) cho hÃ´m nay.",
    "no_cache": "âš ï¸ KhÃ´ng cÃ³ cache. App Ä‘ang tÃ­nh dá»± bÃ¡o TRá»°C TIáº¾P (cháº­m, cÃ³ thá»ƒ bá»‹ giá»›i háº¡n). HÃ£y cháº¡y scheduler_cache.py má»—i ngÃ y!",

    "selected": "âœ… ÄÃ£ chá»n:",
    "today_overview": "### ðŸ“Š Tá»•ng quan hÃ´m nay",
    "p_flood": "XÃ¡c suáº¥t ngáº­p",
    "area_m2": "Diá»‡n tÃ­ch ngáº­p (mÂ²)",
    "ratio": "Tá»‰ lá»‡ ngáº­p",
    "affected": "DÃ¢n bá»‹ áº£nh hÆ°á»Ÿng",
    "priority": "Æ¯u tiÃªn",

    "forecast_7d": "### ðŸ“ˆ Dá»± bÃ¡o 7 ngÃ y",
    "chart_title": "### ðŸ“‰ Biá»ƒu Ä‘á»“ xu hÆ°á»›ng 7 ngÃ y",
    "chart_p_flood": "XÃ¡c suáº¥t ngáº­p",
    "chart_ratio": "Tá»‰ lá»‡ ngáº­p",
    "chart_priority": "Äiá»ƒm Æ°u tiÃªn",

    "incident_title": "## ðŸ§© Theo dÃµi sá»± cá»‘ (tráº¡ng thÃ¡i cá»©u há»™)",
    "status_label": "Tráº¡ng thÃ¡i",
    "note_label": "Ghi chÃº",
    "update_incident": "Cáº­p nháº­t tráº¡ng thÃ¡i sá»± cá»‘",
    "incident_updated": "âœ… ÄÃ£ cáº­p nháº­t sá»± cá»‘!",

    "shelter_title": "## ðŸ  Äiá»ƒm trÃº áº©n (gáº§n nháº¥t)",
    "shelters_in_commune": "Äiá»ƒm trÃº áº©n trong xÃ£:",
    "nearest_shelters": "Äiá»ƒm trÃº áº©n gáº§n nháº¥t:",
    "no_shelter_file": "ChÆ°a cÃ³ shelters.csv. ThÃªm data/shelters.csv Ä‘á»ƒ báº­t tÃ­nh nÄƒng nÃ y.",

    "click_hint": "ðŸ‘† HÃ£y báº¥m vÃ o Ä‘a giÃ¡c má»™t xÃ£ trÃªn báº£n Ä‘á»“ Ä‘á»ƒ xem chi tiáº¿t & Ä‘iá»u hÃ nh cá»©u há»™.",

    "allocation_title": "ðŸ› ï¸ PhÃ¢n bá»• nguá»“n lá»±c (tá»± Ä‘á»™ng + chá»‰nh tay)",
    "allocate_topN": "PhÃ¢n bá»• cho Top N xÃ£",
    "suggested_alloc": "### Gá»£i Ã½ phÃ¢n bá»• (Äá»™i/Thuyá»n/Xe) + Æ¯á»›c tÃ­nh nhu cáº§u",
    "save_alloc": "âœ… LÆ°u phÃ¢n bá»• vÃ o DB (Top N)",
    "alloc_saved": "âœ… ÄÃ£ lÆ°u phÃ¢n bá»•!",
    "saved_alloc_title": "### PhÃ¢n bá»• Ä‘Ã£ lÆ°u (DB)",

    "scheduler_title": "â±ï¸ Lá»‹ch cháº¡y / Cache (cáº­p nháº­t háº±ng ngÃ y)",
    "scheduler_help": """
**Khuyáº¿n nghá»‹ váº­n hÃ nh thá»±c táº¿:**
- Má»—i ngÃ y (hoáº·c 3 giá»/láº§n), cháº¡y `python src/scheduler_cache.py` Ä‘á»ƒ táº¡o cache parquet.
- App sáº½ Ä‘á»c cache Ä‘á»ƒ hiá»ƒn thá»‹ nhanh, trÃ¡nh bá»‹ rate-limit tá»« Open-Meteo.
""",
    "show_cache_path": "Hiá»ƒn thá»‹ Ä‘Æ°á»ng dáº«n cache dá»± kiáº¿n hÃ´m nay",

    # ---- NhÃ£n cá»™t hiá»ƒn thá»‹ tiáº¿ng Viá»‡t ----
    "col_gid": "MÃ£ xÃ£ (GID_3)",
    "col_name": "TÃªn xÃ£",
    "col_p_flood": "XÃ¡c suáº¥t ngáº­p",
    "col_ratio": "Tá»‰ lá»‡ ngáº­p",
    "col_affected": "DÃ¢n bá»‹ áº£nh hÆ°á»Ÿng",
    "col_priority_score": "Äiá»ƒm Æ°u tiÃªn",
    "col_priority_level": "Má»©c Æ°u tiÃªn",
    "col_area_m2": "Diá»‡n tÃ­ch ngáº­p (mÂ²)",
    "col_last_update": "Cáº­p nháº­t lÃºc",

    "col_teams_alloc": "Äá»™i (phÃ¢n bá»•)",
    "col_boats_alloc": "Thuyá»n (phÃ¢n bá»•)",
    "col_trucks_alloc": "Xe táº£i (phÃ¢n bá»•)",
    "col_food_need": "Nhu cáº§u lÆ°Æ¡ng thá»±c (gÃ³i)",
    "col_water_need": "Nhu cáº§u nÆ°á»›c (lÃ­t)",
    "col_med_need": "Nhu cáº§u y táº¿ (bá»™)",

    # ---- Intake / Dispatch / Shelter ----
    "intake_title": "ðŸ“ž Tiáº¿p nháº­n yÃªu cáº§u cá»©u há»™ (Hotline)",
    "intake_new": "âž• Táº¡o yÃªu cáº§u cá»©u há»™ má»›i",
    "caller_name": "Há» tÃªn ngÆ°á»i bÃ¡o",
    "phone": "Sá»‘ Ä‘iá»‡n thoáº¡i",
    "gid3": "MÃ£ xÃ£ (GID_3)",
    "commune_name": "TÃªn xÃ£",
    "lat": "VÄ© Ä‘á»™ (lat) (náº¿u cÃ³)",
    "lon": "Kinh Ä‘á»™ (lon) (náº¿u cÃ³)",
    "people": "Sá»‘ ngÆ°á»i cáº§n há»— trá»£",
    "urgency": "Má»©c kháº©n cáº¥p (1 tháº¥p â†’ 5 ráº¥t kháº©n)",
    "req_note": "MÃ´ táº£ tÃ¬nh huá»‘ng / Ä‘á»‹a Ä‘iá»ƒm chi tiáº¿t",
    "req_submit": "ðŸ“© Ghi nháº­n yÃªu cáº§u",
    "req_list": "### ðŸ“‹ Danh sÃ¡ch yÃªu cáº§u gáº§n Ä‘Ã¢y",

    "dispatch_title": "ðŸ§­ Äiá»u phá»‘i nhiá»‡m vá»¥ (Táº¡o & Cáº­p nháº­t)",
    "create_task_from_req": "âš¡ Táº¡o nhiá»‡m vá»¥ tá»« yÃªu cáº§u (chá»n request â†’ táº¡o task)",
    "create_task_manual": "âž• Táº¡o nhiá»‡m vá»¥ thá»§ cÃ´ng (theo xÃ£ Ä‘ang chá»n)",
    "task_list": "### ðŸ“‹ Danh sÃ¡ch nhiá»‡m vá»¥",
    "task_update": "âœï¸ Cáº­p nháº­t nhiá»‡m vá»¥",

    "shelter_manage_title": "ðŸ  Quáº£n lÃ½ Ä‘iá»ƒm trÃº áº©n (cÃ²n chá»— + nhu cáº§u + chuyá»ƒn ngÆ°á»i)",
    "shelter_status_edit": "âœï¸ Cáº­p nháº­t tráº¡ng thÃ¡i Ä‘iá»ƒm trÃº áº©n",
    "shelter_status_table": "### ðŸ“‹ Tráº¡ng thÃ¡i Ä‘iá»ƒm trÃº áº©n",
    "shelter_move": "ðŸš Chuyá»ƒn ngÆ°á»i Ä‘áº¿n Ä‘iá»ƒm trÃº áº©n (táº¡o log)",
    "shelter_move_history": "### ðŸ§¾ Lá»‹ch sá»­ chuyá»ƒn ngÆ°á»i",

    # hiá»ƒn thá»‹ bá»• sung
    "col_commune_name": "TÃªn xÃ£ (suy ra)",
    "col_start_time": "Thá»i gian báº¯t Ä‘áº§u",
    "col_from_commune": "TÃªn xÃ£ Ä‘i (suy ra)",
    "col_to_gid3": "MÃ£ xÃ£ Ä‘áº¿n (suy ra)",
    "col_to_commune": "TÃªn xÃ£ Ä‘áº¿n (suy ra)",
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
    # Ä‘Æ¡n giáº£n: lower + strip; náº¿u muá»‘n máº¡nh hÆ¡n (bá» dáº¥u) báº¡n cÃ³ thá»ƒ thÃªm unidecode
    return normalize_str(x).lower()

# # =========================
# # VIá»†T HOÃ TÃŠN Cá»˜T HIá»‚N THá»Š
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

    # bá»• sung hiá»ƒn thá»‹
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
        st.error(f"âŒ KhÃ´ng tÃ¬m tháº¥y dá»¯ liá»‡u tá»•ng há»£p: {MERGED_FILE}")
        st.info("ðŸ‘‰ Kiá»ƒm tra báº¡n cÃ³ `cache/merged.csv` á»Ÿ thÆ° má»¥c project.")
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
        st.error(f"âŒ KhÃ´ng tÃ¬m tháº¥y dá»¯ liá»‡u dÃ¢n sá»‘: {POP_FILE}")
        st.info("ðŸ‘‰ Kiá»ƒm tra báº¡n cÃ³ `data/DanSo_Xa.csv` á»Ÿ thÆ° má»¥c project.")
        st.stop()

    pop = pd.read_csv(POP_FILE)
    pop["GID_3"] = pop["GID_3"].astype(str)
    if "DanSo_sum" not in pop.columns:
        st.error("âŒ File dÃ¢n sá»‘ thiáº¿u cá»™t `DanSo_sum`.")
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
# # BUILD STATIC (Láº¤Y THÃ”NG TIN XÃƒ)
# # =========================
static = (
    df.sort_values("date_local")
      .drop_duplicates(subset=["GID_3"], keep="first")
      .copy()
      .set_index("GID_3")
)

# # Táº¡o map gid<->name Ä‘á»ƒ auto-fill intake + enrich báº£ng
gid_to_name = {}
namekey_to_gid = {}
for gid in static.index.astype(str).tolist():
    row = static.loc[gid]
    name = row.get("NAME_3_flood", gid)
    name = normalize_str(name) if name is not None else gid
    gid_to_name[str(gid)] = name

    k = normalize_name_key(name)
    # náº¿u trÃ¹ng tÃªn -> giá»¯ cÃ¡i Ä‘áº§u tiÃªn
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
            st.warning(f"âš ï¸ KhÃ´ng Ä‘á»c Ä‘Æ°á»£c cache parquet: {path}\n\nChi tiáº¿t: {e}")
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

# # map Æ°u tiÃªn theo gid
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
teams_total = st.sidebar.number_input(VI["teams"],  0, 9999,    int(_default_resources.get("teams",       10)),   1)
boats_total = st.sidebar.number_input(VI["boats"],  0, 9999,    int(_default_resources.get("boats",       15)),   1)
trucks_total= st.sidebar.number_input(VI["trucks"], 0, 9999,    int(_default_resources.get("trucks",       8)),   1)
food_total  = st.sidebar.number_input(VI["food"],   0, 9999999, int(_default_resources.get("food_packs",  5000)), 100)
water_total = st.sidebar.number_input(VI["water"],  0, 9999999, int(_default_resources.get("water_liters",10000)),200)
med_total   = st.sidebar.number_input(VI["med"],    0, 9999999, int(_default_resources.get("medical_kits", 600)), 20)

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
    name="CÃ¡c xÃ£",
    style_function=style_fn,
    tooltip=folium.GeoJsonTooltip(
        fields=["name", "priority_score"],
        aliases=[VI["tooltip_name"], VI["tooltip_priority"]],
        sticky=True
    )
).add_to(m)

map_data = st_folium(m, width=1200, height=650)

# # âœ… FIX láº¥y selected_gid/selected_name an toÃ n
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
            "Sá»‘ cáº£nh bÃ¡o": len(alerts),
            "Äiá»ƒm Æ°u tiÃªn cao nháº¥t": float(rank_df["priority_score"].max()),
            "Tá»•ng dÃ¢n bá»‹ áº£nh hÆ°á»Ÿng (top20)": float(top20["affected_population"].sum()),
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
        status_options = ["CHÆ¯A TRIá»‚N KHAI", "ÄANG TRIá»‚N KHAI", "HOÃ€N Táº¤T", "Cáº¦N Há»– TRá»¢ THÃŠM"]
        current = get_incident(str(selected_gid))
        current_status = (current[1] if current else "CHÆ¯A TRIá»‚N KHAI") or "CHÆ¯A TRIá»‚N KHAI"
        current_status = str(current_status).strip()
        current_note = current[3] if current else ""

        idx0 = safe_index(status_options, current_status, default=0)
        new_status = st.selectbox(VI["status_label"], status_options, index=idx0, key="incident_status")
        new_note = st.text_area(VI["note_label"], value=str(current_note or ""), height=80, key="incident_note")

        if st.button(VI["update_incident"], key="incident_update_btn"):
            upsert_incident(str(selected_gid), new_status, new_note)
            st.success(VI["incident_updated"])

        # -------- SHELTER LIST (cÅ©) --------
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
# # âœ… TÃCH Há»¢P 3 TÃNH NÄ‚NG: Intake â†’ Dispatch â†’ Shelter status
# # ============================================================

# # =========================
# # (1) HOTLINE / INTAKE (AUTO-FILL GID <-> TÃŠN XÃƒ)
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
    # Ä‘áº£m báº£o Ä‘á»“ng bá»™ tÃªn theo map chuáº©n
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

        # náº¿u chá»‰ nháº­p tÃªn -> cá»‘ tÃ¬m gid
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
            "status": "Má»šI"
        })
        st.success(f"âœ… ÄÃ£ táº¡o yÃªu cáº§u cá»©u há»™ ID = {rid}")

st.write(VI["req_list"])
req_rows = list_requests(limit=200)
req_df = pd.DataFrame(req_rows, columns=[
    "id", "created_at", "caller_name", "phone", "gid3", "lat", "lon", "people",
    "urgency", "note", "status", "linked_task_id"
])

# # thÃªm tÃªn xÃ£ suy ra Ä‘á»ƒ dá»… nhÃ¬n
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
        req_id = st.selectbox("Chá»n Request ID", req_df["id"].tolist(), key="req_pick")
        req_row = req_df[req_df["id"] == req_id].iloc[0].to_dict()

        gid3 = normalize_str(req_row.get("gid3")) or normalize_str(selected_gid)
        # suy tÃªn xÃ£ tá»« gid
        commune_name = derive_commune_name(gid3) if gid3 else ""

        priority_score = float(gid_to_priority.get(gid3, 0.0)) if gid3 else 0.0

        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        task_type = c1.selectbox("Loáº¡i nhiá»‡m vá»¥", ["Cá»¨U Há»˜", "SÆ  TÃN", "Cáº¤P PHÃT", "KHáº¢O SÃT", "Y Táº¾"], key="task_type_from_req")
        assigned_team = c2.text_input("Äá»™i phá»¥ trÃ¡ch (tÃªn kÃ­p/Ä‘á»™i)", value="", key="task_team_from_req")
        boats = c3.number_input("Sá»‘ thuyá»n Ä‘iá»u", 0, 999, 0, 1, key="task_boats_from_req")
        trucks = c4.number_input("Sá»‘ xe táº£i Ä‘iá»u", 0, 999, 0, 1, key="task_trucks_from_req")

        eta_min = st.number_input("ETA (phÃºt) Æ°á»›c tÃ­nh", 0, 9999, 0, 5, key="task_eta_from_req")
        note_task = st.text_area("Ghi chÃº nhiá»‡m vá»¥", value=req_row.get("note") or "", height=80, key="task_note_from_req")

        # hiá»ƒn thá»‹ thÃ´ng tin sáº½ gáº¯n vÃ o task
        st.info(f"ðŸ“ XÃ£: **{commune_name or '(chÆ°a rÃµ)'}** | MÃ£ xÃ£: **{gid3 or '(trá»‘ng)'}** | Äiá»ƒm Æ°u tiÃªn: **{priority_score:.1f}**")

        if st.button("âœ… Táº¡o nhiá»‡m vá»¥ & gáº¯n vá»›i request", key="create_task_from_req_btn"):
            # quy Æ°á»›c: náº¿u Ä‘Ã£ giao nguá»“n lá»±c/Ä‘á»™i -> coi nhÆ° báº¯t Ä‘áº§u ngay
            now_iso = datetime.now().isoformat(timespec="seconds")
            will_dispatch = bool(assigned_team.strip() or int(boats) > 0 or int(trucks) > 0)
            init_status = "ÄÃƒ GIAO" if will_dispatch else "Má»šI"
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
            update_request_status(int(req_id), "ÄÃƒ Táº O NHIá»†M Vá»¤", linked_task_id=int(tid))
            st.success(f"âœ… ÄÃ£ táº¡o Task ID = {tid} tá»« Request ID = {req_id}")
    else:
        st.info("ChÆ°a cÃ³ yÃªu cáº§u Ä‘á»ƒ táº¡o nhiá»‡m vá»¥.")

with st.expander(VI["create_task_manual"], expanded=False):
    gid3_manual = normalize_str(selected_gid) or st.text_input("MÃ£ xÃ£ (GID_3)", key="manual_gid3")
    commune_name_manual = derive_commune_name(gid3_manual) or normalize_str(selected_name)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    task_type2 = c1.selectbox("Loáº¡i nhiá»‡m vá»¥ (thá»§ cÃ´ng)", ["Cá»¨U Há»˜", "SÆ  TÃN", "Cáº¤P PHÃT", "KHáº¢O SÃT", "Y Táº¾"], key="manual_task_type")
    assigned_team2 = c2.text_input("Äá»™i phá»¥ trÃ¡ch", value="", key="manual_team")
    boats2 = c3.number_input("Thuyá»n", 0, 999, 0, 1, key="boats2")
    trucks2 = c4.number_input("Xe táº£i", 0, 999, 0, 1, key="trucks2")
    eta2 = st.number_input("ETA (phÃºt)", 0, 9999, 0, 5, key="eta2")
    note2 = st.text_area("Ghi chÃº", height=80, key="note2")

    if st.button("âœ… Táº¡o nhiá»‡m vá»¥ (thá»§ cÃ´ng)", key="create_manual_task"):
        pscore = float(gid_to_priority.get(gid3_manual, 0.0)) if gid3_manual else 0.0
        now_iso = datetime.now().isoformat(timespec="seconds")
        will_dispatch = bool(assigned_team2.strip() or int(boats2) > 0 or int(trucks2) > 0)
        init_status = "ÄÃƒ GIAO" if will_dispatch else "Má»šI"
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
        st.success(f"âœ… ÄÃ£ táº¡o Task ID = {tid}")

st.write(VI["task_list"])
task_rows = list_tasks(limit=300)
task_df = pd.DataFrame(task_rows, columns=[
    "id","created_at","gid3","commune_name","task_type","priority_score",
    "assigned_team","boats","trucks","status","eta_min","start_time","end_time","note","source_request_id"
])

# # enrich: Ä‘áº£m báº£o Ä‘á»§ mÃ£ xÃ£ + tÃªn xÃ£ + Ä‘iá»ƒm Æ°u tiÃªn + thá»i gian báº¯t Ä‘áº§u
if len(task_df):
    task_df["gid3"] = task_df["gid3"].astype(str).replace("None", "")
    task_df["commune_name"] = task_df["commune_name"].astype(str).replace("None", "")
    # náº¿u commune_name trá»‘ng -> suy ra tá»« gid
    task_df["commune_name_derived"] = task_df.apply(
        lambda r: normalize_str(r.get("commune_name")) or derive_commune_name(r.get("gid3")),
        axis=1
    )
    # náº¿u priority_score trá»‘ng -> suy ra
    task_df["priority_score"] = pd.to_numeric(task_df["priority_score"], errors="coerce").fillna(
        task_df["gid3"].apply(lambda g: gid_to_priority.get(normalize_str(g), 0.0))
    )
st.dataframe(df_vi(task_df), use_container_width=True)

with st.expander(VI["task_update"], expanded=False):
    if len(task_df):
        task_id = st.selectbox("Chá»n Task ID", task_df["id"].tolist(), key="task_pick")
        trow = task_df[task_df["id"] == task_id].iloc[0].to_dict()

        status_options = ["Má»šI","ÄÃƒ GIAO","ÄANG ÄI","ÄANG THá»°C HIá»†N","HOÃ€N Táº¤T","Há»¦Y"]
        cur_task_status = str(trow.get("status") or "").strip()
        idx_task = safe_index(status_options, cur_task_status, default=0)

        new_status = st.selectbox("Tráº¡ng thÃ¡i má»›i", status_options, index=idx_task, key="task_new_status")
        team_new = st.text_input("Äá»™i phá»¥ trÃ¡ch", value=str(trow.get("assigned_team") or ""), key="task_new_team")
        boats_new = st.number_input("Thuyá»n", 0, 999, to_int_safe(trow.get("boats"), 0), 1, key="task_new_boats")
        trucks_new = st.number_input("Xe táº£i", 0, 999, to_int_safe(trow.get("trucks"), 0), 1, key="task_new_trucks")
        eta_new = st.number_input("ETA (phÃºt)", 0, 9999, to_int_safe(trow.get("eta_min"), 0), 5, key="task_new_eta")
        note_new = st.text_area("Ghi chÃº", value=str(trow.get("note") or ""), height=80, key="task_new_note")

        start_time = None
        end_time = None
        now_iso = datetime.now().isoformat(timespec="seconds")

        # quy Æ°á»›c cáº­p nháº­t: set start_time khi chuyá»ƒn sang ÄANG ÄI/ÄANG THá»°C HIá»†N mÃ  chÆ°a cÃ³ start_time
        if new_status in ["ÄANG ÄI","ÄANG THá»°C HIá»†N"] and not trow.get("start_time"):
            start_time = now_iso
        if new_status == "HOÃ€N Táº¤T":
            if not trow.get("start_time"):
                start_time = now_iso
            end_time = now_iso

        if st.button("âœ… LÆ°u cáº­p nháº­t Task", key="task_update_btn"):
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
            st.success("âœ… ÄÃ£ cáº­p nháº­t nhiá»‡m vá»¥.")
    else:
        st.info("ChÆ°a cÃ³ nhiá»‡m vá»¥ nÃ o.")

# # =========================
# # (3) SHELTER STATUS + CHUYá»‚N NGÆ¯á»œI
# # =========================
st.subheader(VI["shelter_manage_title"])

def make_shelter_key(r):
    gid = normalize_str(r.get("commune_gid3", ""))
    name = normalize_str(r.get("name", ""))
    return f"{gid}__{name}".lower()

# # chuáº©n hoÃ¡ shelters_df2 + label_map Ä‘á»ƒ selectbox khÃ´ng lá»—i
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
            "Chá»n Ä‘iá»ƒm trÃº áº©n",
            shelter_keys,
            format_func=lambda k: shelter_label_map.get(str(k), str(k)),
            key="pick_shelter_key"
        )

        srow = shelters_df2[shelters_df2["shelter_key"].astype(str) == str(shelter_key)].iloc[0].to_dict()

        cap_default = to_int_safe(srow.get("capacity", 0), 0)
        gid3_default = normalize_str(srow.get("commune_gid3"))
        commune_name_derived = derive_commune_name(gid3_default)

        st.info(f"ðŸ“ XÃ£: **{commune_name_derived or '(chÆ°a rÃµ)'}** | MÃ£ xÃ£: **{gid3_default or '(trá»‘ng)'}**")

        c1, c2, c3, c4 = st.columns([1,1,1,1])
        capacity = c1.number_input("Sá»©c chá»©a", 0, 999999, cap_default, 10, key="sh_cap")
        current_people = c2.number_input("Äang cÃ³ (ngÆ°á»i)", 0, 999999, 0, 10, key="sh_cur")
        need_food = c3.number_input("Nhu cáº§u lÆ°Æ¡ng thá»±c (gÃ³i)", 0, 9999999, 0, 50, key="sh_food")
        need_water = c4.number_input("Nhu cáº§u nÆ°á»›c (lÃ­t)", 0, 9999999, 0, 100, key="sh_water")
        need_med = st.number_input("Nhu cáº§u y táº¿ (bá»™)", 0, 9999999, 0, 10, key="sh_med")
        note = st.text_area("Ghi chÃº", height=80, key="sh_note")

        if st.button("âœ… LÆ°u tráº¡ng thÃ¡i Ä‘iá»ƒm trÃº áº©n", key="save_shelter_status_btn"):
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
            st.success("âœ… ÄÃ£ lÆ°u tráº¡ng thÃ¡i Ä‘iá»ƒm trÃº áº©n.")
    else:
        st.info("ChÆ°a cÃ³ shelters.csv Ä‘á»ƒ quáº£n lÃ½ Ä‘iá»ƒm trÃº áº©n.")

st.write(VI["shelter_status_table"])
sh_rows = list_shelter_status(limit=500)
sh_df = pd.DataFrame(sh_rows, columns=[
    "shelter_key","name","gid3","capacity","current_people","need_food","need_water","need_med","note","last_update"
])
if len(sh_df):
    sh_df["gid3"] = sh_df["gid3"].astype(str).replace("None", "")
    sh_df["commune_name_derived"] = sh_df["gid3"].apply(lambda g: derive_commune_name(g) if g and g != "nan" else "")
    sh_df["cÃ²n_trá»‘ng"] = sh_df["capacity"].fillna(0) - sh_df["current_people"].fillna(0)
st.dataframe(df_vi(sh_df), use_container_width=True)

with st.expander(VI["shelter_move"], expanded=True):
    if len(sh_df):
        from_gid3 = st.text_input("Tá»« xÃ£ (GID_3)", value=normalize_str(selected_gid) or "", key="mv_from_gid3")
        from_name = derive_commune_name(from_gid3)
        if from_gid3:
            st.caption(f"âž¡ï¸ TÃªn xÃ£ Ä‘i (suy ra): {from_name or '(chÆ°a rÃµ)'}")

        sh_df2 = sh_df.copy()
        sh_df2["shelter_key"] = sh_df2["shelter_key"].astype(str)
        sh_df2["name"] = sh_df2["name"].apply(normalize_str)
        mv_label_map = dict(zip(sh_df2["shelter_key"].tolist(), sh_df2["name"].tolist()))

        to_key = st.selectbox(
            "Äáº¿n Ä‘iá»ƒm trÃº áº©n",
            sh_df2["shelter_key"].tolist(),
            format_func=lambda k: mv_label_map.get(str(k), str(k)),
            key="mv_to_key"
        )

        # suy ra xÃ£ Ä‘áº¿n dá»±a theo shelter_status
        to_row = sh_df2[sh_df2["shelter_key"].astype(str) == str(to_key)].iloc[0].to_dict()
        to_gid3 = normalize_str(to_row.get("gid3"))
        to_commune_name = derive_commune_name(to_gid3)

        st.caption(f"ðŸ XÃ£ Ä‘áº¿n (suy ra): {to_commune_name or '(chÆ°a rÃµ)'} | MÃ£ xÃ£: {to_gid3 or '(trá»‘ng)'}")

        people_move = st.number_input("Sá»‘ ngÆ°á»i chuyá»ƒn", 0, 999999, 10, 1, key="mv_people")
        link_task = st.number_input("Gáº¯n Task ID (náº¿u cÃ³)", 0, 999999, 0, 1, key="mv_task")
        note_move = st.text_input("Ghi chÃº chuyá»ƒn", key="mv_note")

        if st.button("âœ… XÃ¡c nháº­n chuyá»ƒn", key="mv_confirm"):
            create_shelter_move({
                "from_gid3": from_gid3.strip() if from_gid3.strip() else None,
                "to_shelter_key": str(to_key),
                "people": int(people_move),
                "task_id": int(link_task) if int(link_task) > 0 else None,
                "note": note_move
            })

            # cáº­p nháº­t current_people á»Ÿ shelter_status
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

            st.success("âœ… ÄÃ£ ghi nháº­n chuyá»ƒn ngÆ°á»i vÃ  cáº­p nháº­t Ä‘iá»ƒm trÃº áº©n.")
    else:
        st.info("ChÆ°a cÃ³ dá»¯ liá»‡u shelter_status Ä‘á»ƒ thá»±c hiá»‡n chuyá»ƒn ngÆ°á»i.")

st.write(VI["shelter_move_history"])
mv_rows = list_shelter_moves(limit=300)
mv_df = pd.DataFrame(mv_rows, columns=["id","created_at","from_gid3","to_shelter_key","people","task_id","note"])

# # enrich lá»‹ch sá»­ chuyá»ƒn: thÃªm tÃªn xÃ£ Ä‘i/Ä‘áº¿n + mÃ£ xÃ£ Ä‘áº¿n
if len(mv_df):
    mv_df["from_gid3"] = mv_df["from_gid3"].astype(str).replace("None", "")
    mv_df["from_commune_derived"] = mv_df["from_gid3"].apply(lambda g: derive_commune_name(g) if g and g != "nan" else "")

    # join to shelter_status Ä‘á»ƒ suy ra to_gid3
    if len(sh_df):
        sh_key_to_gid = dict(zip(sh_df["shelter_key"].astype(str).tolist(), sh_df["gid3"].astype(str).tolist()))
    else:
        sh_key_to_gid = {}

    mv_df["to_gid3_derived"] = mv_df["to_shelter_key"].astype(str).apply(lambda k: normalize_str(sh_key_to_gid.get(str(k), "")))
    mv_df["to_commune_derived"] = mv_df["to_gid3_derived"].apply(lambda g: derive_commune_name(g) if g else "")

st.dataframe(df_vi(mv_df), use_container_width=True)

# # =========================
# # PHÃ‚N Bá»” NGUá»’N Lá»°C
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
# # Há»– TRá»¢ SCHEDULER / CACHE
# # =========================
st.subheader(VI["scheduler_title"])
st.write(VI["scheduler_help"])

if st.button(VI["show_cache_path"], key="show_cache_path_btn"):
    today = datetime.now().strftime("%Y-%m-%d")
    st.code(str(CACHE_DIR / f"{today}_commune_forecasts_7d.parquet"))


