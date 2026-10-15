# # app_streamlit_v3_dashboard_map_sos_priority_FIXED_COMPLETE.py
# # =========================================================
# # Realtime Flood Forecast â€” Dashboard + Smart Map (V3)
# # + SOS Signals (CSV) + Click-to-set Rescue Base
# # + Shortest Path Routing to Selected/Nearest SOS
# # + SOS Priority Scoring using Flood Risk at SOS Location
# # + Keeps BOTH maps (before & after click overlays)
# #
# # FIX (CÃ¡ch 1): streamlit_geolocation() uses hard-coded key="loc"
# # => Must render at MOST ONCE per rerun to avoid DuplicateElementKey.
# # We enforce via session_state guard: "_geo_rendered_this_run".
# # =========================================================

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import folium

from streamlit_folium import st_folium
from pyproj import Transformer
from streamlit_autorefresh import st_autorefresh

from folium.plugins import HeatMap, AntPath
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union

from skimage.measure import label, find_contours
import alphashape

# # Routing
import networkx as nx
import osmnx as ox

# # Citizen geolocation (CUSTOM COMPONENT)
from streamlit_geolocation import streamlit_geolocation


# =========================================================
# CONFIG
# =========================================================
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_POINTS_STATIC = os.path.join(_APP_DIR, "base_points_static.csv")
META_PATH          = os.path.join(_APP_DIR, "model_meta.json")

RT_PROB_PATH   = os.path.join(_APP_DIR, "realtime_outputs", "flood_point_probability_rt.csv")
RT_EXTENT_PATH = os.path.join(_APP_DIR, "realtime_outputs", "flood_extent_mask_rt.csv")

SOS_PATH = os.path.join(_APP_DIR, "realtime_outputs", "sos_signals.csv")

# shelters.csv: ngườn chính là data/, nếu không có thì dùng realtime_outputs/
_SHELTERS_DATA = os.path.join(_APP_DIR, "data", "shelters.csv")
_SHELTERS_RT   = os.path.join(_APP_DIR, "realtime_outputs", "shelters.csv")
SHELTERS_PATH = _SHELTERS_DATA if os.path.exists(_SHELTERS_DATA) else _SHELTERS_RT

# resources_default.json
RESOURCES_DEFAULT_PATH = os.path.join(_APP_DIR, "data", "resources_default.json")

EPSG_UTM = "epsg:32648"
EPSG_WGS = "epsg:4326"
transformer = Transformer.from_crs(EPSG_UTM, EPSG_WGS, always_xy=True)

AUTO_REFRESH_MS = 5 * 60 * 1000

LOW_THR = 0.3
MID_THR = 0.6

GRAPH_CACHE = "realtime_outputs/osm_graph.graphml"
GRAPH_DIST_M = 7000


# # =========================================================
# # STYLE HELPERS
# # =========================================================
def prob_color(p: float) -> str:
    if p < LOW_THR:
        return "#2ecc71"
    elif p < MID_THR:
        return "#f39c12"
    return "#e74c3c"


def risk_class(p: float) -> str:
    if p < LOW_THR:
        return "Low"
    if p < MID_THR:
        return "Medium"
    return "High"


def priority_level(score: float) -> str:
    if score < 25:
        return "Low"
    elif score < 50:
        return "Medium"
    elif score < 75:
        return "High"
    else:
        return "Emergency"


# # =========================================================
# # DATA HELPERS
# # =========================================================
def ensure_latlon(df: pd.DataFrame) -> pd.DataFrame:
    if "lat" not in df.columns or "lon" not in df.columns:
        lon, lat = transformer.transform(df["X"].values, df["Y"].values)
        df["lat"] = lat
        df["lon"] = lon
    return df


def load_meta():
    if not os.path.exists(META_PATH):
        return {"cell_area": None}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_static_points():
    if not os.path.exists(BASE_POINTS_STATIC):
        return None

    base = pd.read_csv(BASE_POINTS_STATIC)

    rename_map = {}
    if "Road_Density" in base.columns:
        rename_map["Road_Density"] = "road_density"
    if "Road_density" in base.columns:
        rename_map["Road_density"] = "road_density"
    if "Population_density" in base.columns:
        rename_map["Population_density"] = "population_density"
    if "pop_density" in base.columns:
        rename_map["pop_density"] = "population_density"

    base = base.rename(columns=rename_map)
    base = ensure_latlon(base)
    return base


def load_realtime_csv():
    if not os.path.exists(RT_PROB_PATH):
        raise FileNotFoundError(f"KhÃ´ng tháº¥y file realtime: {RT_PROB_PATH}")

    prob = pd.read_csv(RT_PROB_PATH)
    prob["time"] = pd.to_datetime(prob["time"])
    prob = ensure_latlon(prob)

    extent = None
    if os.path.exists(RT_EXTENT_PATH):
        extent = pd.read_csv(RT_EXTENT_PATH)
        extent["time"] = pd.to_datetime(extent["time"])
        extent = ensure_latlon(extent)

    return prob, extent


def load_sos() -> pd.DataFrame:
    if not os.path.exists(SOS_PATH):
        return pd.DataFrame(columns=["id", "time", "lat", "lon", "note", "status"])

    sos = pd.read_csv(SOS_PATH)
    if "time" in sos.columns:
        sos["time"] = pd.to_datetime(sos["time"], errors="coerce")
    else:
        sos["time"] = pd.Timestamp.utcnow()

    for c in ["id", "note", "status"]:
        if c not in sos.columns:
            sos[c] = ""

    sos["status"] = sos["status"].fillna("open").astype(str)
    sos["note"] = sos["note"].fillna("").astype(str)
    sos["status"] = sos["status"].str.lower().str.strip()

    sos = sos.dropna(subset=["lat", "lon"]).copy()
    return sos.sort_values("time", ascending=False)


def load_shelters() -> pd.DataFrame:
    if not os.path.exists(SHELTERS_PATH):
        return pd.DataFrame(columns=["shelter_id", "name", "lat", "lon", "capacity", "type", "is_open"])

    sh = pd.read_csv(SHELTERS_PATH)
    for c in ["shelter_id", "name", "capacity", "type", "is_open"]:
        if c not in sh.columns:
            sh[c] = ""

    sh = sh.dropna(subset=["lat", "lon"]).copy()
    return sh


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R*c


def find_nearest_shelter(user_lat, user_lon, shelters_df: pd.DataFrame):
    if shelters_df is None or len(shelters_df) == 0:
        return None

    d = haversine_km(user_lat, user_lon, shelters_df["lat"].values, shelters_df["lon"].values)
    idx = int(np.argmin(d))
    nearest = shelters_df.iloc[idx].copy()
    nearest["dist_km"] = float(d[idx])
    return nearest


# # =========================================================
# # GPS FIX â€” render streamlit_geolocation() at MOST once per rerun
# # =========================================================
def geolocation_once_per_run():
    """
    streamlit_geolocation() uses a fixed internal key='loc'.
    This guard guarantees it is rendered at most ONCE per rerun.
    Any later calls return stored gps_shared without rendering.
    """
    # already rendered this rerun => return stored value
    if st.session_state.get("_geo_rendered_this_run", False):
        return st.session_state.get("gps_shared", None)

    # mark rendered for this rerun
    st.session_state["_geo_rendered_this_run"] = True

    # render the component ONCE
    loc = streamlit_geolocation()

    if loc and loc.get("latitude") and loc.get("longitude"):
        st.session_state["gps_shared"] = {
            "lat": float(loc["latitude"]),
            "lon": float(loc["longitude"]),
            "raw": loc
        }

    return st.session_state.get("gps_shared", None)


# # =========================================================
# # SOS PRIORITY
# # =========================================================
STATUS_WEIGHT = {
    "open": 1.0,
    "in_progress": 0.7,
    "closed": 0.0,
    "false_alarm": 0.0,
}


def attach_sos_priority(sos_df: pd.DataFrame, prob_t: pd.DataFrame) -> pd.DataFrame:
    if len(sos_df) == 0:
        return sos_df

    sos = sos_df.copy()

    probs = []
    for _, s in sos.iterrows():
        d2 = (prob_t["lat"] - float(s["lat"])) ** 2 + (prob_t["lon"] - float(s["lon"])) ** 2
        nearest = prob_t.loc[d2.idxmin()]
        probs.append(float(nearest["flood_prob"]))
    sos["flood_prob_near"] = probs

    now_ts = pd.Timestamp.now()
    if "time" in sos.columns:
        delta_min = (now_ts - pd.to_datetime(sos["time"], errors="coerce")).dt.total_seconds() / 60.0
        delta_min = delta_min.fillna(delta_min.max() if len(delta_min) else 0)
    else:
        delta_min = pd.Series(np.zeros(len(sos)))

    rec = 1.0 - np.clip(delta_min / 180.0, 0, 1)
    sos["recency_score"] = rec

    sw = sos["status"].astype(str).str.lower().str.strip().map(STATUS_WEIGHT).fillna(1.0)
    sos["status_weight"] = sw

    score = 100.0 * (0.55 * sos["flood_prob_near"] + 0.25 * sos["recency_score"] + 0.20 * sos["status_weight"])
    sos["priority_score"] = score
    sos["priority_level"] = sos["priority_score"].apply(priority_level)

    return sos


def compute_buffer_mask(df, click_lat, click_lon, buffer_m):
    buffer_deg = buffer_m / 111000.0
    return (
        (df["lat"].between(click_lat - buffer_deg, click_lat + buffer_deg)) &
        (df["lon"].between(click_lon - buffer_deg, click_lon + buffer_deg))
    )


# # =========================================================
# # MAP UI HELPERS
# # =========================================================
def draw_legend(map_obj):
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 28px;
        left: 28px;
        z-index: 9999;
        background: rgba(255,255,255,0.95);
        padding: 12px 14px;
        border-radius: 12px;
        box-shadow: 0 3px 15px rgba(0,0,0,0.15);
        font-size: 14px;
        min-width: 220px;">
        <b>ðŸŒ¡ï¸ Risk Legend</b><br>
        <div style="margin-top:6px;">
            <span style="background:{prob_color(0.1)};width:12px;height:12px;display:inline-block;border-radius:50%;"></span>
            Low (&lt; {LOW_THR})
        </div>
        <div>
            <span style="background:{prob_color(0.5)};width:12px;height:12px;display:inline-block;border-radius:50%;"></span>
            Medium ({LOW_THR}-{MID_THR})
        </div>
        <div>
            <span style="background:{prob_color(0.9)};width:12px;height:12px;display:inline-block;border-radius:50%;"></span>
            High (&ge; {MID_THR})
        </div>
        <hr style="margin:6px 0;">
        <div style="color:blue;"><b>Blue outline</b> = flooded</div>
        <div style="color:navy;"><b>Navy polygon</b> = connected flood</div>
        <div style="color:purple;"><b>Purple polygon</b> = buffer zone</div>
        <div style="color:red;"><b>Red pin</b> = SOS</div>
        <div style="color:green;"><b>Green home</b> = Rescue Base</div>
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(legend_html))
    return map_obj


# # =========================================================
# # CONNECTED FLOOD POLYGON
# # =========================================================
def connected_flood_polygon(ext_t, click_lat, click_lon):
    flooded = ext_t[ext_t["flood_mask"] == 1].copy()
    if len(flooded) == 0:
        return None

    flooded["dist2"] = (flooded["lat"] - click_lat) ** 2 + (flooded["lon"] - click_lon) ** 2
    nearest = flooded.sort_values("dist2").iloc[0]

    xs = np.sort(ext_t["X"].unique())
    ys = np.sort(ext_t["Y"].unique())
    if len(xs) * len(ys) != len(ext_t):
        return None

    grid = ext_t.pivot(index="Y", columns="X", values="flood_mask").loc[ys, xs].values.astype(int)

    xi = np.where(xs == nearest["X"])[0][0]
    yi = np.where(ys == nearest["Y"])[0][0]

    lbl = label(grid, connectivity=2)
    comp_id = lbl[yi, xi]
    if comp_id == 0:
        return None

    comp_mask = (lbl == comp_id).astype(np.uint8)
    contours = find_contours(comp_mask, 0.5)
    if len(contours) == 0:
        return None

    polys = []
    for cont in contours:
        coords = []
        for (r, ccol) in cont:
            yy = ys[int(np.clip(round(r), 0, len(ys) - 1))]
            xx = xs[int(np.clip(round(ccol), 0, len(xs) - 1))]
            lon, lat = transformer.transform(xx, yy)
            coords.append((lon, lat))

        if len(coords) >= 4:
            poly = Polygon(coords)
            if poly.is_valid and poly.area > 0:
                polys.append(poly)

    if len(polys) == 0:
        return None

    return unary_union(polys)


# # =========================================================
# # ZONE POLYGON (ALPHA SHAPE)
# # =========================================================
def alpha_shape_polygon(zone_df, alpha=1.6):
    if zone_df is None or len(zone_df) < 10:
        return None

    pts = list(zip(zone_df["lon"], zone_df["lat"]))
    try:
        poly = alphashape.alphashape(pts, alpha)
        if poly.is_empty:
            return None
        return poly
    except Exception:
        return None


# # =========================================================
# # IMPACT + ZONE
# # =========================================================
def compute_click_impact(prob_t, static_base, meta, click_lat, click_lon, buffer_m):
    cell_area = meta.get("cell_area", None)

    if cell_area is None:
        x_diffs = np.diff(np.sort(static_base["X"].unique()))
        y_diffs = np.diff(np.sort(static_base["Y"].unique()))
        dx = np.median(x_diffs[x_diffs > 0]) if len(x_diffs) else 30
        dy = np.median(y_diffs[y_diffs > 0]) if len(y_diffs) else 30
        cell_area = float(dx * dy)

    merged = prob_t.merge(static_base, on=["X", "Y"], how="left", suffixes=("", "_static"))

    if "population_density" not in merged.columns or "road_density" not in merged.columns:
        raise ValueError("âŒ Missing population_density or road_density in merged data. Check base_points_static.csv")

    m = compute_buffer_mask(merged, click_lat, click_lon, buffer_m)
    zone = merged[m].copy()

    if len(zone) == 0:
        return None, cell_area, zone

    zone["flood_mask"] = (zone["flood_prob"] >= MID_THR).astype(int)
    zone["cell_area_m2"] = cell_area

    flooded_area_m2 = float((zone["flood_mask"] * zone["cell_area_m2"]).sum())
    total_area_m2 = float(zone["cell_area_m2"].sum())
    flooded_area_pct = flooded_area_m2 / (total_area_m2 + 1e-9)

    zone["pop_cell"] = zone["population_density"] * zone["cell_area_m2"]
    pop_total = float(zone["pop_cell"].sum())
    pop_exposed = float((zone["pop_cell"] * zone["flood_mask"]).sum())
    pop_exposed_pct = pop_exposed / (pop_total + 1e-9)

    zone["road_cell"] = zone["road_density"] * zone["cell_area_m2"]
    road_total = float(zone["road_cell"].sum())
    road_exposed = float((zone["road_cell"] * zone["flood_mask"]).sum())
    road_exposed_pct = road_exposed / (road_total + 1e-9)

    A = flooded_area_pct
    P = pop_exposed_pct
    R = road_exposed_pct

    impact_score = float(100 * (0.35 * P + 0.25 * R + 0.15 * A))
    priority = priority_level(impact_score)

    impact = {
        "buffer_radius_m": int(buffer_m),
        "num_points_in_zone": int(len(zone)),
        "cell_area_m2": float(cell_area),
        "flooded_area_m2": flooded_area_m2,
        "flooded_area_pct": float(flooded_area_pct),
        "pop_total": pop_total,
        "pop_exposed": pop_exposed,
        "pop_exposed_pct": float(pop_exposed_pct),
        "road_total_proxy": road_total,
        "road_exposed_proxy": road_exposed,
        "road_exposed_pct": float(road_exposed_pct),
        "impact_score": impact_score,
        "priority_level": priority,
    }

    return impact, cell_area, zone


# # =========================================================
# # DRAWING LAYERS
# # =========================================================
def draw_risk_points(layer, prob_t):
    for _, r in prob_t.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=4,
            color=prob_color(r["flood_prob"]),
            fill=True,
            fill_opacity=0.82,
            popup=f"Prob={r['flood_prob']:.3f} | Risk={r.get('risk_class','')}"
        ).add_to(layer)


def draw_extent_points(layer, ext_t):
    flood_pts = ext_t[ext_t["flood_mask"] == 1]
    for _, r in flood_pts.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=3,
            color="blue",
            fill=True,
            fill_opacity=0.35
        ).add_to(layer)


def draw_zone_points(layer, zone_df):
    for _, r in zone_df.iterrows():
        p = float(r["flood_prob"])
        col = prob_color(p)
        outline = "blue" if int(r.get("flood_mask", 0)) == 1 else col

        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=7,
            color=outline,
            weight=2,
            fill=True,
            fill_color=col,
            fill_opacity=0.65,
            popup=f"ZONE | Prob={p:.3f}"
        ).add_to(layer)


def draw_polygon(layer, poly, name, color, fill_opacity=0.10, weight=3):
    if poly is None:
        return
    folium.GeoJson(
        mapping(poly),
        name=name,
        style_function=lambda x: {"color": color, "weight": weight, "fillOpacity": fill_opacity}
    ).add_to(layer)


def sos_icon_color(status: str) -> str:
    status = (status or "").lower().strip()
    if status == "open":
        return "red"
    if status == "in_progress":
        return "orange"
    if status == "closed":
        return "green"
    if status == "false_alarm":
        return "gray"
    return "red"


def draw_sos(layer, sos_df: pd.DataFrame):
    for _, s in sos_df.iterrows():
        col = sos_icon_color(str(s.get("status", "open")))
        folium.Marker(
            location=[float(s["lat"]), float(s["lon"])],
            popup=f"ðŸ†˜ {s.get('id','')} | {s.get('note','')} | {s.get('status','')} | {s.get('time','')}",
            icon=folium.Icon(color=col, icon="exclamation-sign")
        ).add_to(layer)


# # =========================================================
# # DASHBOARD HELPERS
# # =========================================================
def build_time_summary(prob_df):
    summary = (
        prob_df.groupby("time")
        .agg(
            mean_prob=("flood_prob", "mean"),
            high_ratio=("flood_prob", lambda x: (x >= MID_THR).mean()),
            max_prob=("flood_prob", "max"),
        )
        .reset_index()
    )
    return summary


def build_hotspots(prob_t, top_n=10):
    hh = prob_t.sort_values("flood_prob", ascending=False).head(top_n).copy()
    return hh[["X", "Y", "lat", "lon", "flood_prob"]]


# # =========================================================
# # ROUTING (OSM)
# # =========================================================
@st.cache_resource
def load_road_graph(center_lat: float, center_lon: float, dist: int = GRAPH_DIST_M):
    os.makedirs(os.path.dirname(GRAPH_CACHE), exist_ok=True)
    if os.path.exists(GRAPH_CACHE):
        return ox.load_graphml(GRAPH_CACHE)

    G = ox.graph_from_point((center_lat, center_lon), dist=dist, network_type="drive")
    ox.save_graphml(G, GRAPH_CACHE)
    return G


def shortest_route(G, start_lat, start_lon, end_lat, end_lon):
    orig = ox.distance.nearest_nodes(G, start_lon, start_lat)
    dest = ox.distance.nearest_nodes(G, end_lon, end_lat)

    route = nx.shortest_path(G, orig, dest, weight="length")
    coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]
    dist_m = nx.shortest_path_length(G, orig, dest, weight="length")
    return coords, float(dist_m)


def draw_route(map_obj, coords, color="blue"):
    """Draw an animated AntPath route + a rescue team icon at the midpoint."""
    # Animated "ant" route â€“ white dashes flowing on the route line
    AntPath(
        locations=coords,
        delay=800,
        weight=6,
        color=color,
        pulse_color="white",
        dash_array=[20, 30],
        opacity=0.9,
    ).add_to(map_obj)

    # Place a ðŸš‘ icon at the midpoint of the route to show the rescue team
    mid = coords[len(coords) // 2]
    folium.Marker(
        location=mid,
        popup="ðŸš‘ Äá»™i cá»©u há»™ Ä‘ang trÃªn Ä‘Æ°á»ng",
        icon=folium.DivIcon(
            html=(
                '<div style="font-size:26px;text-align:center;margin-top:-13px;">ðŸš‘</div>'
            ),
            icon_size=(34, 34),
            icon_anchor=(17, 17),
        ),
    ).add_to(map_obj)


# # =========================================================
# # STREAMLIT APP
# # =========================================================
st.set_page_config(page_title="Realtime Flood Map V3 + SOS (Complete FIX)", layout="wide")

# # IMPORTANT: reset guard every rerun
st.session_state["_geo_rendered_this_run"] = False

st.title("ðŸŒŠ Realtime Flood Forecast â€” Dashboard + Smart Map (V3) + SOS + Routing")
st.caption("Auto refresh má»—i 5 phÃºt â€¢ (FIX) GPS component render once per rerun to avoid DuplicateElementKey")

st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_timer")

# # Session defaults
if "base_lat" not in st.session_state:
    st.session_state["base_lat"] = None
if "base_lon" not in st.session_state:
    st.session_state["base_lon"] = None
if "base_gps_lat" not in st.session_state:
    st.session_state["base_gps_lat"] = None
if "base_gps_lon" not in st.session_state:
    st.session_state["base_gps_lon"] = None

# # Sidebar controls
st.sidebar.header("âš™ï¸ Controls")
BUFFER_RADIUS_M = st.sidebar.slider("Buffer radius (m)", 50, 2000, 300, 50)

show_extent = st.sidebar.checkbox("Show extent mask", value=True)
show_only_high = st.sidebar.checkbox("Show only High risk points", value=False)

show_zone = st.sidebar.checkbox("Show Buffer Zone", value=True)
show_zone_poly = st.sidebar.checkbox("Show Buffer Zone polygon (alpha shape)", value=True)
alpha_val = st.sidebar.slider("Alpha shape tightness (smaller=tighter)", 0.4, 5.0, 1.6, 0.1)

show_connected = st.sidebar.checkbox("Show Connected Flood Region", value=True)
show_heatmap = st.sidebar.checkbox("Show Heatmap", value=False)

top_n_hotspots = st.sidebar.slider("Top hotspots", 5, 50, 15, 5)

st.sidebar.subheader("ðŸ†˜ SOS + Routing")

base_mode = st.sidebar.radio(
    "Rescue Base mode",
    ["ðŸ–±ï¸ Click map to set base", "ðŸ“ Use real GPS base"],
    index=0,
    key="base_mode"
)

show_sos = st.sidebar.checkbox("Show SOS", value=True)

click_mode = st.sidebar.radio(
    "Click mode",
    ["Inspect Location (default)", "Set Rescue Base"],
    index=0,
    key="click_mode"
)

use_selected_sos = st.sidebar.checkbox("Route to selected SOS (instead of highest priority)", value=True)

# # ---- GPS Widget (ONLY PLACE that renders actual component) ----
st.sidebar.markdown("---")
st.sidebar.subheader("ðŸ“¡ GPS (Shared)")
st.sidebar.caption("GPS widget nÃ y dÃ¹ng chung cho cáº£ Rescue Base (GPS) & Citizen (GPS). KhÃ´ng render á»Ÿ nÆ¡i khÃ¡c.")

gps_shared = geolocation_once_per_run()
if gps_shared:
    st.sidebar.success(f"GPS: {gps_shared['lat']:.6f}, {gps_shared['lon']:.6f}")
else:
    st.sidebar.info("ChÆ°a cÃ³ GPS. HÃ£y Allow location (náº¿u khÃ´ng Ä‘Æ°á»£c, thá»­ HTTPS hoáº·c Ä‘á»•i browser).")

# # Tabs
tab1, tab2, tab3 = st.tabs(["ðŸš‘ Rescue Dashboard", "ðŸ§ Citizen Map", "ðŸ“‹ Äiá»u Phá»‘i SOS"])


# # =========================================================
# # TAB 1 â€” RESCUE
# # =========================================================
with tab1:
    meta = load_meta()
    static_base = load_static_points()
    if static_base is None:
        st.error("âŒ KhÃ´ng tÃ¬m tháº¥y base_points_static.csv.")
        st.stop()

    try:
        prob_df, extent_df = load_realtime_csv()
    except Exception as e:
        st.error(f"âŒ Lá»—i Ä‘á»c realtime CSV: {e}")
        st.stop()

    if st.session_state.get("base_lat") is None:
        st.session_state["base_lat"] = float(prob_df["lat"].mean())
        st.session_state["base_lon"] = float(prob_df["lon"].mean())

    sos_df = load_sos()
    shelters_df = load_shelters()

    unique_times = sorted(prob_df["time"].unique())
    selected_time = st.sidebar.select_slider("Select time (Rescue)", options=unique_times, value=unique_times[0], key="rescue_time")

    prob_t = prob_df[prob_df["time"] == selected_time].copy()
    prob_t["risk_class"] = prob_t.get("risk_class", prob_t["flood_prob"].apply(risk_class))

    if show_only_high:
        prob_t = prob_t[prob_t["flood_prob"] >= MID_THR].copy()

    ext_t = None
    if extent_df is not None:
        ext_t = extent_df[extent_df["time"] == selected_time].copy()

    open_sos = sos_df[sos_df["status"].isin(["open", "in_progress"])].copy()
    sos_scored = attach_sos_priority(open_sos, prob_t) if len(open_sos) else open_sos

    selected_sos = None
    if len(sos_scored) > 0 and use_selected_sos:
        sos_scored = sos_scored.sort_values(["priority_score", "time"], ascending=[False, False]).copy()
        sos_labels = [
            f"{r['id']} | {r.get('status','')} | {r.get('priority_level','')} ({r.get('priority_score',0):.1f})"
            for _, r in sos_scored.iterrows()
        ]
        idx = st.sidebar.selectbox("Select SOS", list(range(len(sos_labels))), format_func=lambda i: sos_labels[i], key="rescue_sos_select")
        selected_sos = sos_scored.iloc[int(idx)]
    elif len(sos_scored) > 0:
        sos_scored = sos_scored.sort_values(["priority_score", "time"], ascending=[False, False]).copy()
        selected_sos = sos_scored.iloc[0]

    # Base update from GPS shared
    if str(base_mode).startswith("ðŸ“") and gps_shared:
        st.session_state["base_gps_lat"] = gps_shared["lat"]
        st.session_state["base_gps_lon"] = gps_shared["lon"]

    # Current base
    if str(base_mode).startswith("ðŸ“"):
        base_lat = st.session_state.get("base_gps_lat")
        base_lon = st.session_state.get("base_gps_lon")
    else:
        base_lat = st.session_state.get("base_lat")
        base_lon = st.session_state.get("base_lon")

    time_summary = build_time_summary(prob_df)

    st.markdown("## ðŸ“Š Forecast Dashboard")
    dcol1, dcol2, dcol3 = st.columns([1.2, 1.2, 1.6])
    with dcol1:
        st.markdown("### â±ï¸ Time Summary")
        st.line_chart(time_summary.set_index("time")[["mean_prob"]])
    with dcol2:
        st.markdown("### ðŸ”¥ High Risk Ratio")
        chart_df = time_summary.set_index("time")[["high_ratio"]].copy()
        chart_df["high_ratio"] = chart_df["high_ratio"] * 100
        st.line_chart(chart_df)
    with dcol3:
        st.markdown("### ðŸš© Top Hotspots")
        hotspots = build_hotspots(prob_t, top_n=top_n_hotspots)
        st.dataframe(hotspots, use_container_width=True, height=250)

    st.markdown("---")

    st.markdown("## ðŸ†˜ SOS Dashboard")
    if len(sos_scored) == 0:
        st.info("ChÆ°a cÃ³ SOS open/in_progress trong file SOS CSV.")
    else:
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Open SOS", int((sos_scored["status"] == "open").sum()))
        with s2:
            st.metric("In Progress SOS", int((sos_scored["status"] == "in_progress").sum()))
        with s3:
            st.metric("Highest Priority Score", f"{float(sos_scored['priority_score'].max()):.1f}")
        with s4:
            top_lvl = sos_scored.sort_values("priority_score", ascending=False).iloc[0].get("priority_level", "")
            st.metric("Top Priority Level", str(top_lvl))

        st.dataframe(
            sos_scored[["id", "time", "lat", "lon", "status", "note", "flood_prob_near", "priority_score", "priority_level"]]
            .sort_values(["priority_score", "time"], ascending=[False, False]),
            use_container_width=True,
            height=260
        )

    st.markdown("---")
    st.markdown("## ðŸ—ºï¸ Interactive Map")

    center_lat = float(prob_t["lat"].mean()) if len(prob_t) else float(prob_df["lat"].mean())
    center_lon = float(prob_t["lon"].mean()) if len(prob_t) else float(prob_df["lon"].mean())

    try:
        G = load_road_graph(center_lat, center_lon)
    except Exception as e:
        G = None
        st.warning(f"âš ï¸ KhÃ´ng táº£i Ä‘Æ°á»£c OSM graph: {e} (cáº§n internet láº§n Ä‘áº§u).")

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron")

    layer_risk = folium.FeatureGroup(name="Risk Points", show=True)
    layer_extent = folium.FeatureGroup(name="Extent Mask", show=False)
    layer_heat = folium.FeatureGroup(name="Heatmap", show=False)
    layer_zone = folium.FeatureGroup(name="Buffer Zone", show=True)
    layer_conn = folium.FeatureGroup(name="Connected Flood Region", show=True)
    layer_sos = folium.FeatureGroup(name="SOS Signals", show=True)

    draw_risk_points(layer_risk, prob_t)

    if show_extent and ext_t is not None and "flood_mask" in ext_t.columns:
        draw_extent_points(layer_extent, ext_t)

    if show_heatmap and len(prob_t) > 0:
        heat_data = [[row["lat"], row["lon"], float(row["flood_prob"])] for _, row in prob_t.iterrows()]
        HeatMap(heat_data, radius=18, blur=20, max_zoom=13).add_to(layer_heat)

    if show_sos and len(open_sos) > 0:
        draw_sos(layer_sos, open_sos)

    if base_lat is not None and base_lon is not None:
        folium.Marker(
            [base_lat, base_lon],
            popup="ðŸš‘ Rescue Base",
            icon=folium.Icon(color="green", icon="home")
        ).add_to(m)

    route_coords = None
    if selected_sos is not None:
        folium.Marker(
            [float(selected_sos["lat"]), float(selected_sos["lon"])],
            popup=f"ðŸ†˜ TARGET {selected_sos['id']}",
            icon=folium.Icon(color="red", icon="flag")
        ).add_to(m)

        if base_lat is not None and base_lon is not None and G is not None:
            try:
                route_coords, route_dist_m = shortest_route(
                    G,
                    float(base_lat), float(base_lon),
                    float(selected_sos["lat"]), float(selected_sos["lon"])
                )
                _eta_direct_en = route_dist_m / 1000 / 30 * 60
                ETA_THRESHOLD_MIN = 60
                _two_leg_info_en = None

                if _eta_direct_en > ETA_THRESHOLD_MIN and shelters_df is not None and len(shelters_df) > 0:
                    _sh_en = find_nearest_shelter(
                        float(selected_sos["lat"]), float(selected_sos["lon"]), shelters_df
                    )
                    if _sh_en is not None:
                        sh_lat_en, sh_lon_en = float(_sh_en["lat"]), float(_sh_en["lon"])
                        try:
                            coords_l1, dist_l1 = shortest_route(G, float(base_lat), float(base_lon), sh_lat_en, sh_lon_en)
                            draw_route(m, coords_l1, color="blue")
                        except Exception:
                            coords_l1, dist_l1 = None, 0.0
                        try:
                            coords_l2, dist_l2 = shortest_route(G, sh_lat_en, sh_lon_en, float(selected_sos["lat"]), float(selected_sos["lon"]))
                            draw_route(m, coords_l2, color="orange")
                        except Exception:
                            coords_l2 = [[sh_lat_en, sh_lon_en], [float(selected_sos["lat"]), float(selected_sos["lon"])]]
                            dist_l2 = 0.0
                            folium.PolyLine(coords_l2, color="orange", weight=4, dash_array="8 8").add_to(m)
                        folium.Marker(
                            [sh_lat_en, sh_lon_en],
                            popup=f"ðŸ  Staging: {_sh_en.get('name','Staging Point')}<br>Direct ETA too large ({_eta_direct_en:.0f} min)",
                            icon=folium.DivIcon(
                                html='<div style="font-size:26px;text-align:center;margin-top:-13px;">ðŸ </div>',
                                icon_size=(34, 34), icon_anchor=(17, 17),
                            ),
                        ).add_to(m)
                        if coords_l1:
                            m.fit_bounds(list(coords_l1) + list(coords_l2), padding=(30, 30))
                        _two_leg_info_en = {
                            "shelter_name": str(_sh_en.get("name", "Staging Point")),
                            "dist_leg1_km": dist_l1 / 1000,
                            "dist_leg2_km": dist_l2 / 1000,
                            "eta_leg1_min": int(dist_l1 / 1000 / 30 * 60),
                            "eta_leg2_min": int(dist_l2 / 1000 / 30 * 60),
                        }
                    else:
                        draw_route(m, route_coords, color="blue")
                        m.fit_bounds(route_coords, padding=(30, 30))
                else:
                    draw_route(m, route_coords, color="blue")
                    m.fit_bounds(route_coords, padding=(30, 30))
            except Exception as e:
                st.warning(f"âš ï¸ KhÃ´ng tÃ­nh Ä‘Æ°á»£c route: {e}")
                _two_leg_info_en = None
        else:
            _two_leg_info_en = None


    layer_risk.add_to(m)
    if show_extent:
        layer_extent.add_to(m)
    if show_heatmap:
        layer_heat.add_to(m)
    layer_zone.add_to(m)
    layer_conn.add_to(m)
    if show_sos:
        layer_sos.add_to(m)

    draw_legend(m)
    folium.LayerControl(collapsed=False).add_to(m)

    col1, col2 = st.columns([2.4, 1])

    with col1:
        map_data = st_folium(m, height=760, width=1100, key="rescue_map")

    with col2:
        st.markdown("### ðŸ“Œ Click Insight")
        st.write(f"**Selected time:** {selected_time}")

        if base_lat is not None and base_lon is not None:
            st.info(f"Rescue Base: lat={base_lat:.6f}, lon={base_lon:.6f}")
        else:
            st.warning("ChÆ°a Ä‘áº·t Rescue Base (click map hoáº·c báº­t GPS base).")

        # ---- Rescue Dispatch Visual (English tab) ----
        if selected_sos is not None and base_lat is not None and base_lon is not None and G is not None:
            try:
                _, dist_m_panel = shortest_route(
                    G,
                    float(base_lat), float(base_lon),
                    float(selected_sos["lat"]), float(selected_sos["lon"])
                )
                dist_km_panel = dist_m_panel / 1000
                eta_min_panel = int(dist_km_panel / 30 * 60)
                st.markdown("---")

                if "_two_leg_info_en" in dir() and _two_leg_info_en is not None:
                    ti_en = _two_leg_info_en
                    st.markdown(
                        f"""
                        <div style="background:linear-gradient(135deg,#7b2d00,#b34700);
                                    border-radius:12px;padding:16px;color:white;text-align:center;">
                            <div style="font-size:13px;font-weight:bold;opacity:0.9;">âš ï¸ Direct ETA too large ({eta_min_panel} min)</div>
                            <div style="font-size:13px;margin-top:4px;opacity:0.9;">Re-routing via staging shelter</div>
                            <div style="font-size:36px;margin-top:8px;">ðŸš‘ âž¡ï¸ ðŸ  âž¡ï¸ ðŸ†˜</div>
                            <div style="font-size:15px;font-weight:bold;margin-top:10px;">
                                ðŸ  Staging: {ti_en['shelter_name']}
                            </div>
                            <div style="font-size:13px;margin-top:6px;opacity:0.85;">
                                Leg 1 (base â†’ shelter): <b>{ti_en['dist_leg1_km']:.2f} km</b> | ~{ti_en['eta_leg1_min']} min
                            </div>
                            <div style="font-size:13px;margin-top:4px;opacity:0.85;">
                                Leg 2 (shelter â†’ SOS): <b>{ti_en['dist_leg2_km']:.2f} km</b> | ~{ti_en['eta_leg2_min']} min
                            </div>
                            <div style="font-size:13px;margin-top:4px;font-weight:bold;">
                                Total ETA: ~{ti_en['eta_leg1_min'] + ti_en['eta_leg2_min']} min
                            </div>
                            <div style="margin-top:10px;font-size:20px;">ðŸš§ðŸŒŠðŸ </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.warning("ðŸŸ  2-leg route active â€” map: blue (leg 1), orange (leg 2).")
                else:
                    st.markdown(
                        f"""
                        <div style="background:linear-gradient(135deg,#1a472a,#2d6a4f);
                                    border-radius:12px;padding:16px;color:white;text-align:center;">
                            <div style="font-size:48px;">ðŸš‘</div>
                            <div style="font-size:18px;font-weight:bold;margin-top:8px;">
                                RESCUE TEAM DISPATCHED
                            </div>
                            <div style="font-size:14px;margin-top:6px;opacity:0.85;">
                                Target SOS: <b>{selected_sos.get('id','')}</b>
                            </div>
                            <div style="font-size:14px;margin-top:4px;opacity:0.85;">
                                Distance: <b>{dist_km_panel:.2f} km</b>
                            </div>
                            <div style="font-size:14px;margin-top:4px;opacity:0.85;">
                                ETA (est.): <b>~{eta_min_panel} min</b>
                            </div>
                            <div style="margin-top:10px;font-size:22px;">ðŸš§ðŸŒŠðŸ </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.success("ðŸŸ¢ Rescue team en route â€” follow the animated path on the map.")
            except Exception:
                pass

        if map_data and map_data.get("last_clicked"):
            click_lat = map_data["last_clicked"]["lat"]
            click_lon = map_data["last_clicked"]["lng"]
            st.success(f"Clicked: lat={click_lat:.6f}, lon={click_lon:.6f}")

            if click_mode == "Set Rescue Base" and (not str(base_mode).startswith("ðŸ“")):
                st.session_state["base_lat"] = float(click_lat)
                st.session_state["base_lon"] = float(click_lon)
                st.success("âœ… Rescue Base updated by click.")
            elif click_mode == "Set Rescue Base" and str(base_mode).startswith("ðŸ“"):
                st.info("Base mode Ä‘ang lÃ  GPS â€” chuyá»ƒn sang Click mode Ä‘á»ƒ set báº±ng click.")

            prob_t["dist2"] = (prob_t["lat"] - click_lat) ** 2 + (prob_t["lon"] - click_lon) ** 2
            nearest = prob_t.sort_values("dist2").iloc[0]
            st.markdown("#### âœ… Nearest Prediction")
            st.json({
                "time": str(nearest["time"]),
                "X": float(nearest["X"]),
                "Y": float(nearest["Y"]),
                "flood_prob": float(nearest["flood_prob"]),
                "risk_class": str(nearest["risk_class"])
            })

            impact, _, zone = compute_click_impact(prob_t, static_base, meta, click_lat, click_lon, BUFFER_RADIUS_M)
            st.markdown("#### ðŸ§­ Impact (Buffer Zone)")
            if impact is None:
                st.warning("KhÃ´ng cÃ³ Ä‘iá»ƒm nÃ o trong buffer. HÃ£y tÄƒng buffer radius.")
            else:
                st.metric("Impact score", f"{impact['impact_score']:.2f}")
                st.metric("Priority", impact["priority_level"])

            st.markdown("#### ðŸŒ Connected Flood Region")
            poly_conn = None
            if show_connected and ext_t is not None and "flood_mask" in ext_t.columns:
                poly_conn = connected_flood_polygon(ext_t, click_lat, click_lon)
                if poly_conn is not None:
                    st.success("âœ… Connected region found")
                else:
                    st.warning("KhÃ´ng táº¡o Ä‘Æ°á»£c vÃ¹ng liÃªn thÃ´ng.")
            else:
                st.info("Báº­t Connected Flood Region trong sidebar Ä‘á»ƒ hiá»ƒn thá»‹.")


# # =========================================================
# # TAB 2 â€” CITIZEN (GPS shared; no extra render)
# # =========================================================
with tab2:
    st.subheader("ðŸ§ Citizen Safety Map â€” Vá»‹ trÃ­ hiá»‡n táº¡i & Chá»— trÃº áº©n gáº§n nháº¥t")
    st.caption("Citizen GPS láº¥y tá»« Shared GPS (sidebar). KhÃ´ng render GPS component thÃªm láº§n nÃ o.")

    try:
        prob_df2, _extent2 = load_realtime_csv()
    except Exception as e:
        st.error(f"âŒ Lá»—i Ä‘á»c realtime CSV: {e}")
        st.stop()

    shelters_df2 = load_shelters()

    unique_times2 = sorted(prob_df2["time"].unique())
    selected_time2 = st.select_slider("Select time (Citizen)", options=unique_times2, value=unique_times2[0], key="citizen_time")

    prob_t2 = prob_df2[prob_df2["time"] == selected_time2].copy()
    prob_t2["risk_class"] = prob_t2.get("risk_class", prob_t2["flood_prob"].apply(risk_class))

    center_lat2 = float(prob_t2["lat"].mean()) if len(prob_t2) else float(prob_df2["lat"].mean())
    center_lon2 = float(prob_t2["lon"].mean()) if len(prob_t2) else float(prob_df2["lon"].mean())

    try:
        G2 = load_road_graph(center_lat2, center_lon2)
    except Exception as e:
        G2 = None
        st.warning(f"âš ï¸ KhÃ´ng táº£i Ä‘Æ°á»£c OSM graph: {e}")

    st.markdown("### ðŸŽ¯ Chá»n vá»‹ trÃ­ cá»§a báº¡n")
    loc_mode = st.radio(
        "Chá»n cÃ¡ch láº¥y vá»‹ trÃ­",
        ["ðŸ“ DÃ¹ng GPS (Shared)", "ðŸ–±ï¸ Click trÃªn báº£n Ä‘á»“"],
        index=0,
        key="citizen_loc_mode"
    )

    colA, colB = st.columns([2.4, 1])
    user_lat, user_lon = None, None

    with colA:
        if str(loc_mode).startswith("ðŸ“"):
            if gps_shared:
                user_lat, user_lon = gps_shared["lat"], gps_shared["lon"]
                st.success(f"âœ… Vá»‹ trÃ­ cá»§a báº¡n: {user_lat:.6f}, {user_lon:.6f}")
            else:
                st.warning("ChÆ°a cÃ³ GPS. HÃ£y Allow location á»Ÿ sidebar.")
        else:
            cm_select = folium.Map(location=[center_lat2, center_lon2], zoom_start=13, tiles="CartoDB positron")

            if len(shelters_df2) > 0:
                for _, sh in shelters_df2.iterrows():
                    folium.Marker(
                        [float(sh["lat"]), float(sh["lon"])],
                        popup=f"ðŸ  {sh.get('name','')} | Cap={sh.get('capacity','')}",
                        icon=folium.Icon(color="cadetblue", icon="info-sign")
                    ).add_to(cm_select)

            st.caption("ðŸ‘‰ Click má»™t Ä‘iá»ƒm báº¥t ká»³ Ä‘á»ƒ Ä‘áº·t vá»‹ trÃ­ cá»§a báº¡n.")
            click_data = st_folium(cm_select, height=760, width=1100, key="citizen_pick")

            if click_data and click_data.get("last_clicked"):
                user_lat = float(click_data["last_clicked"]["lat"])
                user_lon = float(click_data["last_clicked"]["lng"])
                st.success(f"âœ… Báº¡n Ä‘Ã£ chá»n: {user_lat:.6f}, {user_lon:.6f}")
            else:
                st.info("ChÆ°a cÃ³ vá»‹ trÃ­ Ä‘Æ°á»£c chá»n.")

        if user_lat is not None and user_lon is not None:
            nearest = find_nearest_shelter(user_lat, user_lon, shelters_df2)
            if nearest is None:
                st.error("âŒ KhÃ´ng cÃ³ dá»¯ liá»‡u shelters.csv.")
            else:
                st.markdown("### ðŸ  Chá»— trÃº áº©n gáº§n nháº¥t")
                st.write({
                    "name": nearest.get("name", ""),
                    "shelter_id": nearest.get("shelter_id", ""),
                    "capacity": nearest.get("capacity", ""),
                    "distance_km": round(float(nearest.get("dist_km", 0)), 2),
                })

                cm = folium.Map(location=[user_lat, user_lon], zoom_start=14, tiles="CartoDB positron")

                folium.Marker(
                    [user_lat, user_lon],
                    popup="ðŸ§ You are here",
                    icon=folium.Icon(color="blue", icon="user")
                ).add_to(cm)

                folium.Marker(
                    [float(nearest["lat"]), float(nearest["lon"])],
                    popup=f"ðŸ  Shelter: {nearest.get('name','')}",
                    icon=folium.Icon(color="cadetblue", icon="info-sign")
                ).add_to(cm)

                if G2 is not None:
                    try:
                        route_coords_u, dist_m_u = shortest_route(
                            G2, user_lat, user_lon,
                            float(nearest["lat"]), float(nearest["lon"])
                        )
                        draw_route(cm, route_coords_u, color="purple")
                        cm.fit_bounds(route_coords_u, padding=(30, 30))
                        st.success(f"ðŸš¶ Optimal route distance: {dist_m_u/1000:.2f} km")
                    except Exception as e:
                        st.warning(f"âš ï¸ KhÃ´ng tÃ­nh Ä‘Æ°á»£c route: {e}")

                st_folium(cm, height=760, width=1100, key="citizen_result_map")

    with colB:
        st.markdown("### ðŸ“Œ Shelters")
        if len(shelters_df2) == 0:
            st.error("KhÃ´ng cÃ³ shelters.csv")
        else:
            st.dataframe(shelters_df2, use_container_width=True, height=320)

        st.markdown("---")
        st.markdown("### ðŸ†˜ Recent SOS")
        try:
            recent_sos = load_sos().sort_values("time", ascending=False).head(15)
            if len(recent_sos) == 0:
                st.info("ChÆ°a cÃ³ SOS nÃ o")
            else:
                show_cols = [c for c in ["id","time","status","lat","lon","note"] if c in recent_sos.columns]
                st.dataframe(recent_sos[show_cols], use_container_width=True, height=260)
        except Exception as e:
            st.warning(f"KhÃ´ng Ä‘á»c Ä‘Æ°á»£c SOS: {e}")

