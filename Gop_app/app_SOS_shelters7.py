# # app_streamlit_v3_dashboard_map_sos_priority_FIXED_COMPLETE.py
# # =========================================================
# # Realtime Flood Forecast — Dashboard + Smart Map (V3)
# # + SOS Signals (CSV) + Click-to-set Rescue Base
# # + Shortest Path Routing to Selected/Nearest SOS
# # + SOS Priority Scoring using Flood Risk at SOS Location
# # + Keeps BOTH maps (before & after click overlays)
# #
# # FIX (Cách 1): streamlit_geolocation() uses hard-coded key="loc"
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


# # =========================================================
# # CONFIG
# # =========================================================
BASE_POINTS_STATIC = "base_points_static.csv"
META_PATH          = "model_meta.json"

RT_PROB_PATH   = r"realtime_outputs\flood_point_probability_rt.csv"
RT_EXTENT_PATH = r"realtime_outputs\flood_extent_mask_rt.csv"

SOS_PATH = r"realtime_outputs\sos_signals.csv"
SHELTERS_PATH = r"realtime_outputs\shelters.csv"

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
        raise FileNotFoundError(f"Không thấy file realtime: {RT_PROB_PATH}")

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
# # GPS FIX — render streamlit_geolocation() at MOST once per rerun
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
        <b>🌡️ Risk Legend</b><br>
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
        raise ValueError("❌ Missing population_density or road_density in merged data. Check base_points_static.csv")

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
            popup=f"🆘 {s.get('id','')} | {s.get('note','')} | {s.get('status','')} | {s.get('time','')}",
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
    # Animated "ant" route – white dashes flowing on the route line
    AntPath(
        locations=coords,
        delay=800,
        weight=6,
        color=color,
        pulse_color="white",
        dash_array=[20, 30],
        opacity=0.9,
    ).add_to(map_obj)

    # Place a 🚑 icon at the midpoint of the route to show the rescue team
    mid = coords[len(coords) // 2]
    folium.Marker(
        location=mid,
        popup="🚑 Đội cứu hộ đang trên đường",
        icon=folium.DivIcon(
            html=(
                '<div style="font-size:26px;text-align:center;margin-top:-13px;">🚑</div>'
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

st.title("🌊 Realtime Flood Forecast — Dashboard + Smart Map (V3) + SOS + Routing")
st.caption("Auto refresh mỗi 5 phút • (FIX) GPS component render once per rerun to avoid DuplicateElementKey")

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
st.sidebar.header("⚙️ Controls")
BUFFER_RADIUS_M = st.sidebar.slider("Buffer radius (m)", 50, 2000, 300, 50)

show_extent = st.sidebar.checkbox("Show extent mask", value=True)
show_only_high = st.sidebar.checkbox("Show only High risk points", value=False)

show_zone = st.sidebar.checkbox("Show Buffer Zone", value=True)
show_zone_poly = st.sidebar.checkbox("Show Buffer Zone polygon (alpha shape)", value=True)
alpha_val = st.sidebar.slider("Alpha shape tightness (smaller=tighter)", 0.4, 5.0, 1.6, 0.1)

show_connected = st.sidebar.checkbox("Show Connected Flood Region", value=True)
show_heatmap = st.sidebar.checkbox("Show Heatmap", value=False)

top_n_hotspots = st.sidebar.slider("Top hotspots", 5, 50, 15, 5)

st.sidebar.subheader("🆘 SOS + Routing")

base_mode = st.sidebar.radio(
    "Rescue Base mode",
    ["🖱️ Click map to set base", "📍 Use real GPS base"],
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
st.sidebar.subheader("📡 GPS (Shared)")
st.sidebar.caption("GPS widget này dùng chung cho cả Rescue Base (GPS) & Citizen (GPS). Không render ở nơi khác.")

gps_shared = geolocation_once_per_run()
if gps_shared:
    st.sidebar.success(f"GPS: {gps_shared['lat']:.6f}, {gps_shared['lon']:.6f}")
else:
    st.sidebar.info("Chưa có GPS. Hãy Allow location (nếu không được, thử HTTPS hoặc đổi browser).")

# # Tabs
tab1, tab2, tab3 = st.tabs(["🚑 Rescue Dashboard", "🧍 Citizen Map", "📋 Điều Phối SOS"])


# # =========================================================
# # TAB 1 — RESCUE
# # =========================================================
with tab1:
    meta = load_meta()
    static_base = load_static_points()
    if static_base is None:
        st.error("❌ Không tìm thấy base_points_static.csv.")
        st.stop()

    try:
        prob_df, extent_df = load_realtime_csv()
    except Exception as e:
        st.error(f"❌ Lỗi đọc realtime CSV: {e}")
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
    if str(base_mode).startswith("📍") and gps_shared:
        st.session_state["base_gps_lat"] = gps_shared["lat"]
        st.session_state["base_gps_lon"] = gps_shared["lon"]

    # Current base
    if str(base_mode).startswith("📍"):
        base_lat = st.session_state.get("base_gps_lat")
        base_lon = st.session_state.get("base_gps_lon")
    else:
        base_lat = st.session_state.get("base_lat")
        base_lon = st.session_state.get("base_lon")

    time_summary = build_time_summary(prob_df)

    st.markdown("## 📊 Forecast Dashboard")
    dcol1, dcol2, dcol3 = st.columns([1.2, 1.2, 1.6])
    with dcol1:
        st.markdown("### ⏱️ Time Summary")
        st.line_chart(time_summary.set_index("time")[["mean_prob"]])
    with dcol2:
        st.markdown("### 🔥 High Risk Ratio")
        chart_df = time_summary.set_index("time")[["high_ratio"]].copy()
        chart_df["high_ratio"] = chart_df["high_ratio"] * 100
        st.line_chart(chart_df)
    with dcol3:
        st.markdown("### 🚩 Top Hotspots")
        hotspots = build_hotspots(prob_t, top_n=top_n_hotspots)
        st.dataframe(hotspots, use_container_width=True, height=250)

    st.markdown("---")

    st.markdown("## 🆘 SOS Dashboard")
    if len(sos_scored) == 0:
        st.info("Chưa có SOS open/in_progress trong file SOS CSV.")
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
    st.markdown("## 🗺️ Interactive Map")

    center_lat = float(prob_t["lat"].mean()) if len(prob_t) else float(prob_df["lat"].mean())
    center_lon = float(prob_t["lon"].mean()) if len(prob_t) else float(prob_df["lon"].mean())

    try:
        G = load_road_graph(center_lat, center_lon)
    except Exception as e:
        G = None
        st.warning(f"⚠️ Không tải được OSM graph: {e} (cần internet lần đầu).")

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
            popup="🚑 Rescue Base",
            icon=folium.Icon(color="green", icon="home")
        ).add_to(m)

    route_coords = None
    if selected_sos is not None:
        folium.Marker(
            [float(selected_sos["lat"]), float(selected_sos["lon"])],
            popup=f"🆘 TARGET {selected_sos['id']}",
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
                            popup=f"🏠 Staging: {_sh_en.get('name','Staging Point')}<br>Direct ETA too large ({_eta_direct_en:.0f} min)",
                            icon=folium.DivIcon(
                                html='<div style="font-size:26px;text-align:center;margin-top:-13px;">🏠</div>',
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
                st.warning(f"⚠️ Không tính được route: {e}")
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
        st.markdown("### 📌 Click Insight")
        st.write(f"**Selected time:** {selected_time}")

        if base_lat is not None and base_lon is not None:
            st.info(f"Rescue Base: lat={base_lat:.6f}, lon={base_lon:.6f}")
        else:
            st.warning("Chưa đặt Rescue Base (click map hoặc bật GPS base).")

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
                            <div style="font-size:13px;font-weight:bold;opacity:0.9;">⚠️ Direct ETA too large ({eta_min_panel} min)</div>
                            <div style="font-size:13px;margin-top:4px;opacity:0.9;">Re-routing via staging shelter</div>
                            <div style="font-size:36px;margin-top:8px;">🚑 ➡️ 🏠 ➡️ 🆘</div>
                            <div style="font-size:15px;font-weight:bold;margin-top:10px;">
                                🏠 Staging: {ti_en['shelter_name']}
                            </div>
                            <div style="font-size:13px;margin-top:6px;opacity:0.85;">
                                Leg 1 (base → shelter): <b>{ti_en['dist_leg1_km']:.2f} km</b> | ~{ti_en['eta_leg1_min']} min
                            </div>
                            <div style="font-size:13px;margin-top:4px;opacity:0.85;">
                                Leg 2 (shelter → SOS): <b>{ti_en['dist_leg2_km']:.2f} km</b> | ~{ti_en['eta_leg2_min']} min
                            </div>
                            <div style="font-size:13px;margin-top:4px;font-weight:bold;">
                                Total ETA: ~{ti_en['eta_leg1_min'] + ti_en['eta_leg2_min']} min
                            </div>
                            <div style="margin-top:10px;font-size:20px;">🚧🌊🏠</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.warning("🟠 2-leg route active — map: blue (leg 1), orange (leg 2).")
                else:
                    st.markdown(
                        f"""
                        <div style="background:linear-gradient(135deg,#1a472a,#2d6a4f);
                                    border-radius:12px;padding:16px;color:white;text-align:center;">
                            <div style="font-size:48px;">🚑</div>
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
                            <div style="margin-top:10px;font-size:22px;">🚧🌊🏠</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.success("🟢 Rescue team en route — follow the animated path on the map.")
            except Exception:
                pass

        if map_data and map_data.get("last_clicked"):
            click_lat = map_data["last_clicked"]["lat"]
            click_lon = map_data["last_clicked"]["lng"]
            st.success(f"Clicked: lat={click_lat:.6f}, lon={click_lon:.6f}")

            if click_mode == "Set Rescue Base" and (not str(base_mode).startswith("📍")):
                st.session_state["base_lat"] = float(click_lat)
                st.session_state["base_lon"] = float(click_lon)
                st.success("✅ Rescue Base updated by click.")
            elif click_mode == "Set Rescue Base" and str(base_mode).startswith("📍"):
                st.info("Base mode đang là GPS — chuyển sang Click mode để set bằng click.")

            prob_t["dist2"] = (prob_t["lat"] - click_lat) ** 2 + (prob_t["lon"] - click_lon) ** 2
            nearest = prob_t.sort_values("dist2").iloc[0]
            st.markdown("#### ✅ Nearest Prediction")
            st.json({
                "time": str(nearest["time"]),
                "X": float(nearest["X"]),
                "Y": float(nearest["Y"]),
                "flood_prob": float(nearest["flood_prob"]),
                "risk_class": str(nearest["risk_class"])
            })

            impact, _, zone = compute_click_impact(prob_t, static_base, meta, click_lat, click_lon, BUFFER_RADIUS_M)
            st.markdown("#### 🧭 Impact (Buffer Zone)")
            if impact is None:
                st.warning("Không có điểm nào trong buffer. Hãy tăng buffer radius.")
            else:
                st.metric("Impact score", f"{impact['impact_score']:.2f}")
                st.metric("Priority", impact["priority_level"])

            st.markdown("#### 🌐 Connected Flood Region")
            poly_conn = None
            if show_connected and ext_t is not None and "flood_mask" in ext_t.columns:
                poly_conn = connected_flood_polygon(ext_t, click_lat, click_lon)
                if poly_conn is not None:
                    st.success("✅ Connected region found")
                else:
                    st.warning("Không tạo được vùng liên thông.")
            else:
                st.info("Bật Connected Flood Region trong sidebar để hiển thị.")


# # =========================================================
# # TAB 2 — CITIZEN (GPS shared; no extra render)
# # =========================================================
with tab2:
    st.subheader("🧍 Citizen Safety Map — Vị trí hiện tại & Chỗ trú ẩn gần nhất")
    st.caption("Citizen GPS lấy từ Shared GPS (sidebar). Không render GPS component thêm lần nào.")

    try:
        prob_df2, _extent2 = load_realtime_csv()
    except Exception as e:
        st.error(f"❌ Lỗi đọc realtime CSV: {e}")
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
        st.warning(f"⚠️ Không tải được OSM graph: {e}")

    st.markdown("### 🎯 Chọn vị trí của bạn")
    loc_mode = st.radio(
        "Chọn cách lấy vị trí",
        ["📍 Dùng GPS (Shared)", "🖱️ Click trên bản đồ"],
        index=0,
        key="citizen_loc_mode"
    )

    colA, colB = st.columns([2.4, 1])
    user_lat, user_lon = None, None

    with colA:
        if str(loc_mode).startswith("📍"):
            if gps_shared:
                user_lat, user_lon = gps_shared["lat"], gps_shared["lon"]
                st.success(f"✅ Vị trí của bạn: {user_lat:.6f}, {user_lon:.6f}")
            else:
                st.warning("Chưa có GPS. Hãy Allow location ở sidebar.")
        else:
            cm_select = folium.Map(location=[center_lat2, center_lon2], zoom_start=13, tiles="CartoDB positron")

            if len(shelters_df2) > 0:
                for _, sh in shelters_df2.iterrows():
                    folium.Marker(
                        [float(sh["lat"]), float(sh["lon"])],
                        popup=f"🏠 {sh.get('name','')} | Cap={sh.get('capacity','')}",
                        icon=folium.Icon(color="cadetblue", icon="info-sign")
                    ).add_to(cm_select)

            st.caption("👉 Click một điểm bất kỳ để đặt vị trí của bạn.")
            click_data = st_folium(cm_select, height=760, width=1100, key="citizen_pick")

            if click_data and click_data.get("last_clicked"):
                user_lat = float(click_data["last_clicked"]["lat"])
                user_lon = float(click_data["last_clicked"]["lng"])
                st.success(f"✅ Bạn đã chọn: {user_lat:.6f}, {user_lon:.6f}")
            else:
                st.info("Chưa có vị trí được chọn.")

        if user_lat is not None and user_lon is not None:
            nearest = find_nearest_shelter(user_lat, user_lon, shelters_df2)
            if nearest is None:
                st.error("❌ Không có dữ liệu shelters.csv.")
            else:
                st.markdown("### 🏠 Chỗ trú ẩn gần nhất")
                st.write({
                    "name": nearest.get("name", ""),
                    "shelter_id": nearest.get("shelter_id", ""),
                    "capacity": nearest.get("capacity", ""),
                    "distance_km": round(float(nearest.get("dist_km", 0)), 2),
                })

                cm = folium.Map(location=[user_lat, user_lon], zoom_start=14, tiles="CartoDB positron")

                folium.Marker(
                    [user_lat, user_lon],
                    popup="🧍 You are here",
                    icon=folium.Icon(color="blue", icon="user")
                ).add_to(cm)

                folium.Marker(
                    [float(nearest["lat"]), float(nearest["lon"])],
                    popup=f"🏠 Shelter: {nearest.get('name','')}",
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
                        st.success(f"🚶 Optimal route distance: {dist_m_u/1000:.2f} km")
                    except Exception as e:
                        st.warning(f"⚠️ Không tính được route: {e}")

                st_folium(cm, height=760, width=1100, key="citizen_result_map")

    with colB:
        st.markdown("### 📌 Shelters")
        if len(shelters_df2) == 0:
            st.error("Không có shelters.csv")
        else:
            st.dataframe(shelters_df2, use_container_width=True, height=320)

        st.markdown("---")
        st.markdown("### 🆘 Recent SOS")
        try:
            recent_sos = load_sos().sort_values("time", ascending=False).head(15)
            if len(recent_sos) == 0:
                st.info("Chưa có SOS nào")
            else:
                show_cols = [c for c in ["id","time","status","lat","lon","note"] if c in recent_sos.columns]
                st.dataframe(recent_sos[show_cols], use_container_width=True, height=260)
        except Exception as e:
            st.warning(f"Không đọc được SOS: {e}")

# =========================================================
# Dự báo ngập thời gian thực — Dashboard + Bản đồ thông minh (V3)
# + Tín hiệu SOS (CSV) + Click để đặt Căn cứ Cứu hộ
# + Dẫn đường ngắn nhất tới SOS được chọn/gần nhất
# + Chấm điểm ưu tiên SOS dựa trên rủi ro ngập tại vị trí SOS
# + GIỮ CẢ HAI BẢN ĐỒ (trước & sau khi click overlay)
# + FIX: streamlit_geolocation trùng key ("loc") -> dùng key riêng
# + MỚI: Căn cứ cứu hộ hỗ trợ cập nhật GPS realtime HOẶC click đặt căn cứ
# + VIỆT HÓA 100% UI + Map
# =========================================================

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import folium

from streamlit_folium import st_folium
from pyproj import Transformer
from streamlit_autorefresh import st_autorefresh

from folium.plugins import HeatMap
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union

from skimage.measure import label, find_contours
import alphashape

# Routing
import networkx as nx
import osmnx as ox

# Citizen geolocation
import streamlit_geolocation as geo

# =========================================================
# CẤU HÌNH
# =========================================================
BASE_POINTS_STATIC = "base_points_static.csv"
META_PATH          = "model_meta.json"

RT_PROB_PATH   = r"realtime_outputs\flood_point_probability_rt.csv"
RT_EXTENT_PATH = r"realtime_outputs\flood_extent_mask_rt.csv"

# SOS CSV
SOS_PATH = r"realtime_outputs\sos_signals.csv"

# Shelters CSV (Citizen Map)
SHELTERS_PATH = r"realtime_outputs\shelters.csv"

EPSG_UTM = "epsg:32648"
EPSG_WGS = "epsg:4326"
transformer = Transformer.from_crs(EPSG_UTM, EPSG_WGS, always_xy=True)

AUTO_REFRESH_MS = 5 * 60 * 1000  # 5 phút
GPS_REFRESH_MS  = 10 * 1000      # 10 giây (chỉ khi bật GPS realtime)

LOW_THR = 0.3
MID_THR = 0.6

# OSM graph cache
GRAPH_CACHE = "realtime_outputs/osm_graph.graphml"
GRAPH_DIST_M = 7000

# =========================================================
# VIỆT HÓA LABELS
# =========================================================
STATUS_VI = {
    "open": "Mở",
    "in_progress": "Đang xử lý",
    "closed": "Đã đóng",
    "false_alarm": "Báo nhầm",
}

def status_vi(s: str) -> str:
    s = (s or "").lower().strip()
    return STATUS_VI.get(s, s)

def base_src_vi(src: str) -> str:
    if src == "gps":
        return "GPS"
    if src == "click":
        return "Click bản đồ"
    return ""

def risk_class(p: float) -> str:
    if p < LOW_THR:
        return "Thấp"
    if p < MID_THR:
        return "Trung bình"
    return "Cao"

def priority_level(score: float) -> str:
    if score < 25:
        return "Thấp"
    elif score < 50:
        return "Trung bình"
    elif score < 75:
        return "Cao"
    else:
        return "Khẩn cấp"

# =========================================================
# FIX: unique geolocation keys to avoid StreamlitDuplicateElementKey
# =========================================================
def geolocate(key: str):
    """
    streamlit_geolocation bản phổ biến hard-code key="loc" => crash khi gọi nhiều lần.
    Wrapper này gọi component bên trong với key riêng.
    """
    try:
        comp = getattr(geo, "_streamlit_geolocation", None)
        if comp is None:
            # Fallback (có thể vẫn trùng key nếu gọi >1 lần)
            return geo.streamlit_geolocation()
        return comp(
            key=key,
            default={
                "latitude": None, "longitude": None, "altitude": None, "accuracy": None,
                "altitudeAccuracy": None, "heading": None, "speed": None
            }
        )
    except Exception:
        return None


# =========================================================
# STYLE HELPERS
# =========================================================
def prob_color(p: float) -> str:
    if p < LOW_THR:
        return "#2ecc71"  # xanh lá
    elif p < MID_THR:
        return "#f39c12"  # cam
    return "#e74c3c"      # đỏ


# =========================================================
# DATA HELPERS
# =========================================================
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
        raise FileNotFoundError(f"Không thấy file realtime: {RT_PROB_PATH}")

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
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def find_nearest_shelter(user_lat, user_lon, shelters_df: pd.DataFrame):
    if shelters_df is None or len(shelters_df) == 0:
        return None

    d = haversine_km(user_lat, user_lon, shelters_df["lat"].values, shelters_df["lon"].values)
    idx = int(np.argmin(d))
    nearest = shelters_df.iloc[idx].copy()
    nearest["dist_km"] = float(d[idx])
    return nearest


# =========================================================
# SOS PRIORITY (tích hợp rủi ro ngập + độ mới + trạng thái)
# =========================================================
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


def enrich_sos_with_priority(sos_df: pd.DataFrame, prob_t_df: pd.DataFrame) -> pd.DataFrame:
    if sos_df is None or len(sos_df) == 0:
        return sos_df

    out = sos_df.copy()
    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
    else:
        out["time"] = pd.Timestamp.utcnow()

    if "flood_prob_near" not in out.columns:
        out["flood_prob_near"] = np.nan
    if "priority_score" not in out.columns:
        out["priority_score"] = np.nan
    if "priority_level" not in out.columns:
        out["priority_level"] = ""

    now_ts = pd.Timestamp.now()

    if prob_t_df is None or len(prob_t_df) == 0:
        out["flood_prob_near"] = out["flood_prob_near"].fillna(0.0)
        out["priority_score"] = out["priority_score"].fillna(0.0)
        out["priority_level"] = out["priority_level"].replace({"": "Thấp"})
        return out

    needs = out["priority_score"].isna() | (out["priority_level"].astype(str).str.len() == 0) | out["flood_prob_near"].isna()

    for idx, r in out[needs].iterrows():
        lat = float(r.get("lat", np.nan))
        lon = float(r.get("lon", np.nan))
        if np.isnan(lat) or np.isnan(lon):
            continue

        d2 = (prob_t_df["lat"] - lat) ** 2 + (prob_t_df["lon"] - lon) ** 2
        nearest = prob_t_df.loc[d2.idxmin()]
        flood_prob = float(nearest["flood_prob"])

        t = r.get("time", None)
        if pd.isna(t):
            rec = 0.5
        else:
            age_min = float((now_ts - t).total_seconds() / 60.0)
            rec = max(0.0, 1.0 - (age_min / 180.0))

        status = str(r.get("status", "open")).lower().strip()
        status_weight = STATUS_WEIGHT.get(status, 1.0)

        score = 100.0 * (0.55 * flood_prob + 0.25 * rec + 0.20 * status_weight)

        out.at[idx, "flood_prob_near"] = flood_prob
        out.at[idx, "priority_score"] = float(score)
        out.at[idx, "priority_level"] = str(priority_level(score))

    return out


def sos_marker_color(level: str) -> str:
    lv = str(level).lower().strip()
    if lv in ["khẩn cấp", "emergency"]:
        return "red"
    if lv in ["cao", "high"]:
        return "orange"
    if lv in ["trung bình", "medium"]:
        return "purple"
    if lv in ["thấp", "low"]:
        return "green"
    return "gray"


# =========================================================
# BASE MODE (GPS realtime / click map)
# =========================================================
def resolve_rescue_base(base_mode: str, gps_realtime: bool):
    """
    Returns (base_lat, base_lon, source)
    source: 'gps' | 'click' | None
    """
    if base_mode.startswith("📍"):
        base_lat = st.session_state.get("base_gps_lat", None)
        base_lon = st.session_state.get("base_gps_lon", None)

        if gps_realtime:
            loc = geolocate("geo_base")
            if loc and loc.get("latitude") and loc.get("longitude"):
                st.session_state["base_gps_lat"] = float(loc["latitude"])
                st.session_state["base_gps_lon"] = float(loc["longitude"])
                base_lat = st.session_state["base_gps_lat"]
                base_lon = st.session_state["base_gps_lon"]

        if base_lat is not None and base_lon is not None:
            return base_lat, base_lon, "gps"
        return None, None, None

    base_lat = st.session_state.get("base_lat", None)
    base_lon = st.session_state.get("base_lon", None)
    if base_lat is not None and base_lon is not None:
        return base_lat, base_lon, "click"
    return None, None, None


# =========================================================
# BUFFER MASK
# =========================================================
def compute_buffer_mask(df, click_lat, click_lon, buffer_m):
    buffer_deg = buffer_m / 111000.0
    return (
        (df["lat"].between(click_lat - buffer_deg, click_lat + buffer_deg)) &
        (df["lon"].between(click_lon - buffer_deg, click_lon + buffer_deg))
    )


# =========================================================
# MAP UI HELPERS
# =========================================================
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
        <b>🌡️ Chú giải rủi ro</b><br>
        <div style="margin-top:6px;">
            <span style="background:{prob_color(0.1)};width:12px;height:12px;display:inline-block;border-radius:50%;"></span>
            Thấp (&lt; {LOW_THR})
        </div>
        <div>
            <span style="background:{prob_color(0.5)};width:12px;height:12px;display:inline-block;border-radius:50%;"></span>
            Trung bình ({LOW_THR}–{MID_THR})
        </div>
        <div>
            <span style="background:{prob_color(0.9)};width:12px;height:12px;display:inline-block;border-radius:50%;"></span>
            Cao (&ge; {MID_THR})
        </div>
        <hr style="margin:6px 0;">
        <div style="color:blue;"><b>Viền xanh</b> = điểm bị ngập</div>
        <div style="color:navy;"><b>Vùng xanh đậm</b> = vùng ngập liên thông</div>
        <div style="color:purple;"><b>Vùng tím</b> = vùng bị tác động</div>
        <div style="color:red;"><b>Ghim đỏ</b> = SOS</div>
        <div style="color:green;"><b>Nhà xanh</b> = căn cứ cứu hộ</div>
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(legend_html))
    return map_obj


# =========================================================
# CONNECTED FLOOD POLYGON
# =========================================================
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


# =========================================================
# ZONE POLYGON (ALPHA SHAPE)
# =========================================================
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


# =========================================================
# IMPACT + ZONE
# =========================================================
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
        raise ValueError("❌ Thiếu population_density hoặc road_density trong dữ liệu. Kiểm tra base_points_static.csv")

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
    prio = priority_level(impact_score)

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
        "priority_level": prio,
    }

    return impact, cell_area, zone


# =========================================================
# DRAWING LAYERS
# =========================================================
def draw_risk_points(layer, prob_t):
    for _, r in prob_t.iterrows():
        p = float(r["flood_prob"])
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=4,
            color=prob_color(p),
            fill=True,
            fill_opacity=0.82,
            popup=f"Xác suất ngập: {p:.3f}<br>Mức rủi ro: {risk_class(p)}"
        ).add_to(layer)


def draw_extent_points(layer, ext_t):
    flood_pts = ext_t[ext_t["flood_mask"] == 1]
    for _, r in flood_pts.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=3,
            color="blue",
            fill=True,
            fill_opacity=0.35,
            popup="Điểm thuộc vùng ngập (mask)"
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
            popup=f"VÙNG BỊ TÁC ĐỘNG<br>Xác suất ngập: {p:.3f}<br>Mức rủi ro: {risk_class(p)}"
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
        sid = str(s.get("id", ""))
        note = str(s.get("note", ""))
        status = status_vi(str(s.get("status", "open")))
        t = str(s.get("time", ""))

        folium.Marker(
            location=[float(s["lat"]), float(s["lon"])],
            popup=f"🆘 SOS: {sid}<br>Trạng thái: {status}<br>Ghi chú: {note}<br>Thời gian: {t}",
            icon=folium.Icon(color=col, icon="exclamation-sign")
        ).add_to(layer)


# =========================================================
# DASHBOARD HELPERS
# =========================================================
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


# =========================================================
# ROUTING (OSM)
# =========================================================
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
    coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]  # lat, lon

    dist_m = nx.shortest_path_length(G, orig, dest, weight="length")
    return coords, float(dist_m)


def draw_route(map_obj, coords, color="blue"):
    """Draw an animated AntPath route + a rescue team icon at the midpoint."""
    AntPath(
        locations=coords,
        delay=800,
        weight=6,
        color=color,
        pulse_color="white",
        dash_array=[20, 30],
        opacity=0.9,
    ).add_to(map_obj)

    mid = coords[len(coords) // 2]
    folium.Marker(
        location=mid,
        popup="🚑 Đội cứu hộ đang trên đường",
        icon=folium.DivIcon(
            html=(
                '<div style="font-size:26px;text-align:center;margin-top:-13px;">🚑</div>'
            ),
            icon_size=(34, 34),
            icon_anchor=(17, 17),
        ),
    ).add_to(map_obj)


# =========================================================
# STREAMLIT APP
# =========================================================
st.set_page_config(page_title="Bản đồ ngập thời gian thực V3 + SOS", layout="wide")

st.title("🌊 Dự báo ngập thời gian thực — Dashboard + Bản đồ thông minh (V3) + SOS + Dẫn đường")
st.caption("Tự động làm mới mỗi 5 phút • Click bản đồ để đặt căn cứ cứu hộ / xem vùng bị tác động + vùng ngập liên thông + ưu tiên SOS")

global refresh
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_timer_sos")

tab1, tab2 = st.tabs(["🚑 Bảng điều khiển cứu hộ", "🧍 Bản đồ an toàn cho người dân"])

# ---------------------------------------------------------
# TAB 1: Rescue
# ---------------------------------------------------------
with tab1:
    meta = load_meta()
    static_base = load_static_points()
    if static_base is None:
        st.error("❌ Không tìm thấy base_points_static.csv. Bạn cần export từ train script.")
        st.stop()

    try:
        prob_df, extent_df = load_realtime_csv()
    except Exception as e:
        st.error(f"❌ Lỗi đọc realtime CSV: {e}")
        st.stop()

    if st.session_state.get("base_lat") is None:
        st.session_state["base_lat"] = float(prob_df["lat"].mean())
        st.session_state["base_lon"] = float(prob_df["lon"].mean())

    sos_df = load_sos()
    shelters_df = load_shelters()

# Sidebar (shared)
st.sidebar.header("⚙️ Điều khiển")

BUFFER_RADIUS_M = st.sidebar.slider("Bán kính vùng bị tác động (m)", 50, 2000, 300, 50)

show_extent = st.sidebar.checkbox("Hiển thị vùng ngập (mask)", value=True)
show_only_high = st.sidebar.checkbox("Chỉ hiển thị điểm rủi ro Cao", value=False)

show_zone = st.sidebar.checkbox("Hiển thị vùng bị tác động", value=True)
show_zone_poly = st.sidebar.checkbox("Hiển thị polygon vùng bị tác động (alpha shape)", value=True)
alpha_val = st.sidebar.slider("Độ chặt alpha shape (nhỏ = chặt hơn)", 0.4, 5.0, 1.6, 0.1)

show_connected = st.sidebar.checkbox("Hiển thị vùng ngập liên thông", value=True)
show_heatmap = st.sidebar.checkbox("Hiển thị bản đồ nhiệt", value=False)

top_n_hotspots = st.sidebar.slider("Top điểm nóng", 5, 50, 15, 5)

st.sidebar.subheader("🆘 SOS + Dẫn đường")

base_mode = st.sidebar.radio(
    "Chế độ căn cứ cứu hộ",
    ["🖱️ Click bản đồ để đặt căn cứ", "📍 Dùng GPS thật làm căn cứ"],
    index=0
)

# NEW: GPS realtime toggle
gps_realtime = st.sidebar.checkbox("📡 Cập nhật GPS theo thời gian thực", value=True)

# If GPS mode + realtime -> add a faster refresh loop (optional)
if base_mode.startswith("📍") and gps_realtime:
    st_autorefresh(interval=GPS_REFRESH_MS, key="gps_refresh")

show_sos = st.sidebar.checkbox("Hiển thị SOS", value=True)

click_mode = st.sidebar.radio(
    "Chế độ click",
    ["Xem thông tin (mặc định)", "Đặt căn cứ cứu hộ"],
    index=0
)

# session state for base
if "base_lat" not in st.session_state:
    st.session_state["base_lat"] = None
if "base_lon" not in st.session_state:
    st.session_state["base_lon"] = None

if "base_gps_lat" not in st.session_state:
    st.session_state["base_gps_lat"] = None
if "base_gps_lon" not in st.session_state:
    st.session_state["base_gps_lon"] = None

use_selected_sos = st.sidebar.checkbox("Dẫn đường tới SOS được chọn (thay vì SOS ưu tiên cao nhất)", value=True)

# Select time
unique_times = sorted(prob_df["time"].unique())
selected_time = st.sidebar.select_slider("Chọn thời điểm", options=unique_times, value=unique_times[0])

prob_t = prob_df[prob_df["time"] == selected_time].copy()
# Ép risk_class tiếng Việt (không phụ thuộc cột có sẵn)
prob_t["risk_class"] = prob_t["flood_prob"].apply(risk_class)

if show_only_high:
    prob_t = prob_t[prob_t["flood_prob"] >= MID_THR].copy()

ext_t = None
if extent_df is not None:
    ext_t = extent_df[extent_df["time"] == selected_time].copy()

# SOS filtering
open_sos = sos_df[sos_df["status"].isin(["open", "in_progress"])].copy()
sos_scored = attach_sos_priority(open_sos, prob_t) if len(open_sos) else open_sos

# Select SOS
selected_sos = None
if len(sos_scored) > 0 and use_selected_sos:
    sos_scored = sos_scored.sort_values(["priority_score", "time"], ascending=[False, False]).copy()
    sos_labels = [
        f"{r['id']} | {status_vi(r.get('status',''))} | Ưu tiên: {r.get('priority_level','')} ({r.get('priority_score',0):.1f})"
        for _, r in sos_scored.iterrows()
    ]
    idx = st.sidebar.selectbox("Chọn SOS", list(range(len(sos_labels))), format_func=lambda i: sos_labels[i])
    selected_sos = sos_scored.iloc[int(idx)]
elif len(sos_scored) > 0:
    sos_scored = sos_scored.sort_values(["priority_score", "time"], ascending=[False, False]).copy()
    selected_sos = sos_scored.iloc[0]

time_summary = build_time_summary(prob_df)

# =========================================================
# TOP DASHBOARD
# =========================================================
st.markdown("## 📊 Bảng điều khiển dự báo")

dcol1, dcol2, dcol3 = st.columns([1.2, 1.2, 1.6])
with dcol1:
    st.markdown("### ⏱️ Tóm tắt theo thời gian")
    st.line_chart(time_summary.set_index("time")[["mean_prob"]])

with dcol2:
    st.markdown("### 🔥 Tỷ lệ rủi ro cao")
    chart_df = time_summary.set_index("time")[["high_ratio"]].copy()
    chart_df["high_ratio"] = chart_df["high_ratio"] * 100
    st.line_chart(chart_df)

with dcol3:
    st.markdown("### 🚩 Điểm nóng (Hotspots)")
    hotspots = build_hotspots(prob_t, top_n=top_n_hotspots)
    hotspots_vi = hotspots.rename(columns={
        "X": "X", "Y": "Y", "lat": "Vĩ độ", "lon": "Kinh độ", "flood_prob": "Xác suất ngập"
    })
    st.dataframe(hotspots_vi, use_container_width=True, height=250)

st.markdown("---")

# =========================================================
# SOS DASHBOARD
# =========================================================
st.markdown("## 🆘 Bảng SOS")
if len(sos_scored) == 0:
    st.info("Chưa có SOS trạng thái Mở/Đang xử lý trong file SOS.")
else:
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("SOS đang mở", int((sos_scored["status"] == "open").sum()))
    with s2:
        st.metric("SOS đang xử lý", int((sos_scored["status"] == "in_progress").sum()))
    with s3:
        st.metric("Điểm ưu tiên cao nhất", f"{float(sos_scored['priority_score'].max()):.1f}")
    with s4:
        top_lvl = sos_scored.sort_values("priority_score", ascending=False).iloc[0].get("priority_level", "")
        st.metric("Mức ưu tiên cao nhất", str(top_lvl))

    sos_show = sos_scored[["id", "time", "lat", "lon", "status", "note", "flood_prob_near", "priority_score", "priority_level"]].copy()
    sos_show["status"] = sos_show["status"].apply(status_vi)
    sos_show = sos_show.rename(columns={
        "id": "Mã",
        "time": "Thời gian",
        "lat": "Vĩ độ",
        "lon": "Kinh độ",
        "status": "Trạng thái",
        "note": "Ghi chú",
        "flood_prob_near": "Xác suất ngập gần đó",
        "priority_score": "Điểm ưu tiên",
        "priority_level": "Mức ưu tiên",
    })

    st.dataframe(
        sos_show.sort_values(["Điểm ưu tiên", "Thời gian"], ascending=[False, False]),
        use_container_width=True,
        height=260
    )

st.markdown("---")

# =========================================================
# MAP SECTION
# =========================================================
st.markdown("## 🗺️ Bản đồ tương tác")

center_lat = float(prob_t["lat"].mean()) if len(prob_t) else float(prob_df["lat"].mean())
center_lon = float(prob_t["lon"].mean()) if len(prob_t) else float(prob_df["lon"].mean())

# Load routing graph once
try:
    G = load_road_graph(center_lat, center_lon)
except Exception as e:
    G = None
    st.warning(f"⚠️ Không tải được OSM graph (dẫn đường): {e}. Lần đầu cần internet để tải graph.")

# Base map (Map 1)
m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron")

# Layers (đã Việt hóa)
layer_risk   = folium.FeatureGroup(name="🌧️ Điểm rủi ro", show=True)
layer_extent = folium.FeatureGroup(name="💧 Vùng ngập (mask)", show=False)
layer_heat   = folium.FeatureGroup(name="🔥 Bản đồ nhiệt", show=False)
layer_zone   = folium.FeatureGroup(name="🟣 Vùng bị tác động", show=True)
layer_conn   = folium.FeatureGroup(name="🟦 Vùng ngập liên thông", show=True)
layer_sos    = folium.FeatureGroup(name="🆘 Tín hiệu SOS", show=True)

draw_risk_points(layer_risk, prob_t)

if show_extent and ext_t is not None and "flood_mask" in ext_t.columns:
    draw_extent_points(layer_extent, ext_t)

if show_heatmap and len(prob_t) > 0:
    heat_data = [[row["lat"], row["lon"], float(row["flood_prob"])] for _, row in prob_t.iterrows()]
    HeatMap(heat_data, radius=18, blur=20, max_zoom=13).add_to(layer_heat)

if show_sos and len(open_sos) > 0:
    draw_sos(layer_sos, open_sos)

# Resolve base (GPS realtime / click)
if base_mode.startswith("📍"):
    st.sidebar.caption("Căn cứ GPS: hãy cho phép quyền định vị để cập nhật")

base_lat, base_lon, base_src = resolve_rescue_base(base_mode, gps_realtime)

if base_lat is not None and base_lon is not None:
    folium.Marker(
        [base_lat, base_lon],
        popup=f"🚑 Căn cứ cứu hộ ({base_src_vi(base_src)})",
        icon=folium.Icon(color="green", icon="home")
    ).add_to(m)

# Target SOS + route
route_coords = None
if selected_sos is not None:
    folium.Marker(
        [float(selected_sos["lat"]), float(selected_sos["lon"])],
        popup=f"🆘 MỤC TIÊU: {selected_sos['id']}<br>Ưu tiên: {selected_sos.get('priority_level','')} ({selected_sos.get('priority_score',0):.1f})",
        icon=folium.Icon(color="red", icon="flag")
    ).add_to(m)

    if base_lat is not None and base_lon is not None and G is not None:
        try:
            route_coords, _dist_m = shortest_route(
                G,
                float(base_lat), float(base_lon),
                float(selected_sos["lat"]), float(selected_sos["lon"])
            )
            _eta_direct = _dist_m / 1000 / 30 * 60  # phút, tốc độ 30 km/h

            ETA_THRESHOLD_MIN = 60  # ngưỡng: nếu > 60 phút → dùng shelter trung gian
            _two_leg_info = None

            if _eta_direct > ETA_THRESHOLD_MIN and shelters_df is not None and len(shelters_df) > 0:
                # Tìm shelter gần SOS nhất
                _sh = find_nearest_shelter(
                    float(selected_sos["lat"]), float(selected_sos["lon"]), shelters_df
                )
                if _sh is not None:
                    sh_lat, sh_lon = float(_sh["lat"]), float(_sh["lon"])

                    # Leg 1: Căn cứ → Shelter
                    try:
                        coords_leg1, dist_leg1 = shortest_route(G, float(base_lat), float(base_lon), sh_lat, sh_lon)
                        draw_route(m, coords_leg1, color="blue")
                    except Exception:
                        coords_leg1, dist_leg1 = None, 0.0

                    # Leg 2: Shelter → SOS (đường thẳng nếu graph lỗi)
                    try:
                        coords_leg2, dist_leg2 = shortest_route(G, sh_lat, sh_lon, float(selected_sos["lat"]), float(selected_sos["lon"]))
                        draw_route(m, coords_leg2, color="orange")
                    except Exception:
                        coords_leg2 = [[sh_lat, sh_lon], [float(selected_sos["lat"]), float(selected_sos["lon"])]]
                        dist_leg2 = 0.0
                        folium.PolyLine(coords_leg2, color="orange", weight=4, dash_array="8 8").add_to(m)

                    # Marker điểm tập kết shelter
                    folium.Marker(
                        [sh_lat, sh_lon],
                        popup=f"🏠 Điểm tập kết: {_sh.get('name','Trạm trung chuyển')}<br>ETA trực tiếp quá lớn ({_eta_direct:.0f} phút) — chuyển qua shelter này",
                        icon=folium.DivIcon(
                            html='<div style="font-size:26px;text-align:center;margin-top:-13px;">🏠</div>',
                            icon_size=(34, 34), icon_anchor=(17, 17),
                        ),
                    ).add_to(m)

                    if coords_leg1:
                        all_coords = list(coords_leg1) + list(coords_leg2)
                        m.fit_bounds(all_coords, padding=(30, 30))

                    _two_leg_info = {
                        "shelter_name": str(_sh.get("name", "Trạm trung chuyển")),
                        "sh_lat": sh_lat, "sh_lon": sh_lon,
                        "dist_leg1_km": dist_leg1 / 1000,
                        "dist_leg2_km": dist_leg2 / 1000,
                        "eta_leg1_min": int(dist_leg1 / 1000 / 30 * 60),
                        "eta_leg2_min": int(dist_leg2 / 1000 / 30 * 60),
                    }
                else:
                    # Không có shelter phù hợp → đi thẳng
                    draw_route(m, route_coords, color="blue")
                    m.fit_bounds(route_coords, padding=(30, 30))
            else:
                # ETA hợp lý → đi thẳng
                draw_route(m, route_coords, color="blue")
                m.fit_bounds(route_coords, padding=(30, 30))

        except Exception as e:
            st.warning(f"⚠️ Không tính được tuyến đường: {e}")
            _two_leg_info = None
    else:
        _two_leg_info = None


if route_coords is None and selected_sos is not None and base_lat is not None and base_lon is not None:
    m.fit_bounds(
        [[float(base_lat), float(base_lon)], [float(selected_sos["lat"]), float(selected_sos["lon"])]],
        padding=(30, 30)
    )

# Add layers
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
    map_data = st_folium(m, height=760, width=1100, key="ban_do_cuu_ho")

with col2:
    st.markdown("### 📌 Thông tin điểm click")
    st.write(f"**Thời điểm đang xem:** {selected_time}")

    if base_lat is not None and base_lon is not None:
        st.info(f"Căn cứ cứu hộ: lat={base_lat:.6f}, lon={base_lon:.6f} ({base_src_vi(base_src)})")
    else:
        st.warning("Chưa đặt căn cứ cứu hộ. Chọn 'Đặt căn cứ cứu hộ' và click map (hoặc bật GPS căn cứ).")

    if selected_sos is not None:
        st.markdown("#### 🆘 SOS được chọn")
        st.json({
            "Mã": str(selected_sos.get("id", "")),
            "Trạng thái": status_vi(str(selected_sos.get("status", ""))),
            "Ghi chú": str(selected_sos.get("note", "")),
            "Điểm ưu tiên": float(selected_sos.get("priority_score", 0)),
            "Mức ưu tiên": str(selected_sos.get("priority_level", "")),
            "Xác suất ngập gần đó": float(selected_sos.get("flood_prob_near", 0)),
            "Vĩ độ": float(selected_sos.get("lat", 0)),
            "Kinh độ": float(selected_sos.get("lon", 0)),
        })

        if base_lat is not None and base_lon is not None and G is not None:
            try:
                _, dist_m = shortest_route(
                    G,
                    float(base_lat), float(base_lon),
                    float(selected_sos["lat"]), float(selected_sos["lon"])
                )
                dist_km = dist_m / 1000
                eta_min = int(dist_km / 30 * 60)
                st.metric("Khoảng cách tuyến đường (km)", f"{dist_km:.2f}")

                st.markdown("---")

                # Kiểm tra xem có dùng tuyến 2 chặng không
                if "_two_leg_info" in dir() and _two_leg_info is not None:
                    ti = _two_leg_info
                    st.markdown(
                        f"""
                        <div style="background:linear-gradient(135deg,#7b2d00,#b34700);
                                    border-radius:12px;padding:16px;color:white;text-align:center;">
                            <div style="font-size:14px;font-weight:bold;opacity:0.9;">⚠️ ETA trực tiếp lớn ({eta_min} phút)</div>
                            <div style="font-size:14px;margin-top:4px;opacity:0.9;">Chuyển tuyến qua điểm tập kết</div>
                            <div style="font-size:36px;margin-top:8px;">🚑 ➡️ 🏠 ➡️ 🆘</div>
                            <div style="font-size:15px;font-weight:bold;margin-top:10px;">
                                🏠 Trạm tập kết: {ti['shelter_name']}
                            </div>
                            <div style="font-size:13px;margin-top:6px;opacity:0.85;">
                                Chặng 1 (căn cứ → shelter): <b>{ti['dist_leg1_km']:.2f} km</b> | ~{ti['eta_leg1_min']} phút
                            </div>
                            <div style="font-size:13px;margin-top:4px;opacity:0.85;">
                                Chặng 2 (shelter → SOS): <b>{ti['dist_leg2_km']:.2f} km</b> | ~{ti['eta_leg2_min']} phút
                            </div>
                            <div style="font-size:13px;margin-top:4px;font-weight:bold;">
                                Tổng ETA: ~{ti['eta_leg1_min'] + ti['eta_leg2_min']} phút
                            </div>
                            <div style="margin-top:10px;font-size:20px;">🚧🌊🏠</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.warning("🟠 Tuyến 2 chặng được kích hoạt — xem bản đồ: xanh (chặng 1), cam (chặng 2).")
                else:
                    st.markdown(
                        f"""
                        <div style="background:linear-gradient(135deg,#1a472a,#2d6a4f);
                                    border-radius:12px;padding:16px;color:white;text-align:center;">
                            <div style="font-size:48px;">🚑</div>
                            <div style="font-size:18px;font-weight:bold;margin-top:8px;">
                                ĐỘI CỨU HỘ ĐANG TRÊN ĐƯỜNG
                            </div>
                            <div style="font-size:14px;margin-top:6px;opacity:0.85;">
                                Mục tiêu SOS: <b>{selected_sos.get('id','')}</b>
                            </div>
                            <div style="font-size:14px;margin-top:4px;opacity:0.85;">
                                Khoảng cách: <b>{dist_km:.2f} km</b>
                            </div>
                            <div style="font-size:14px;margin-top:4px;opacity:0.85;">
                                ETA ước tính: <b>~{eta_min} phút</b>
                            </div>
                            <div style="margin-top:10px;font-size:22px;">🚧🌊🏠</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.success("🟢 Đội cứu hộ đang di chuyển — xem bản đồ để theo dõi tuyến đường.")
            except Exception:
                pass

    # Click handling
    if map_data and map_data.get("last_clicked"):
        click_lat = map_data["last_clicked"]["lat"]
        click_lon = map_data["last_clicked"]["lng"]
        st.success(f"Đã click: lat={click_lat:.6f}, lon={click_lon:.6f}")

        if click_mode == "Đặt căn cứ cứu hộ" and (not base_mode.startswith("📍")):
            st.session_state["base_lat"] = click_lat
            st.session_state["base_lon"] = click_lon
            st.success("✅ Đã cập nhật căn cứ cứu hộ theo điểm click.")
        elif click_mode == "Đặt căn cứ cứu hộ" and base_mode.startswith("📍"):
            st.info("Căn cứ đang ở chế độ GPS. Chuyển sang 'Click bản đồ để đặt căn cứ' nếu muốn đặt căn cứ bằng click.")

        # Nearest prediction point
        prob_t["dist2"] = (prob_t["lat"] - click_lat) ** 2 + (prob_t["lon"] - click_lon) ** 2
        nearest = prob_t.sort_values("dist2").iloc[0]

        st.markdown("#### ✅ Điểm dự báo gần nhất")
        st.json({
            "Thời điểm": str(nearest["time"]),
            "X": float(nearest["X"]),
            "Y": float(nearest["Y"]),
            "Xác suất ngập": float(nearest["flood_prob"]),
            "Mức rủi ro": str(nearest["risk_class"])
        })

        impact, _, zone = compute_click_impact(prob_t, static_base, meta, click_lat, click_lon, BUFFER_RADIUS_M)

        st.markdown("#### 🧭 Tác động (Vùng bị tác động)")
        if impact is None:
            st.warning("Không có điểm nào trong vùng bị tác động. Hãy tăng bán kính vùng bị tác động.")
        else:
            m1, m2c = st.columns(2)
            with m1:
                st.metric("Điểm tác động", f"{impact['impact_score']:.2f}")
                st.metric("Mức ưu tiên", impact["priority_level"])
                st.metric("Diện tích ngập (m²)", f"{impact['flooded_area_m2']:.0f}")
            with m2c:
                st.metric("Dân số phơi nhiễm", f"{impact['pop_exposed']:.0f}")
                st.metric("Tỷ lệ dân số phơi nhiễm", f"{impact['pop_exposed_pct']*100:.2f}%")
                st.metric("Tỷ lệ đường phơi nhiễm", f"{impact['road_exposed_pct']*100:.2f}%")

            st.caption(f"Số điểm: {impact['num_points_in_zone']} | Bán kính: {impact['buffer_radius_m']}m")

            if st.button("⬇️ Xuất tác động + điểm vùng bị tác động"):
                out_dir = "realtime_outputs"
                os.makedirs(out_dir, exist_ok=True)
                pd.DataFrame([impact]).to_csv(os.path.join(out_dir, "impact_click_zone_rt.csv"), index=False)
                zone.to_csv(os.path.join(out_dir, "zone_points_rt.csv"), index=False)
                st.success("✅ Đã xuất impact_click_zone_rt.csv + zone_points_rt.csv")

        st.markdown("#### 🌐 Vùng ngập liên thông")
        poly_conn = None
        if show_connected and ext_t is not None and "flood_mask" in ext_t.columns:
            poly_conn = connected_flood_polygon(ext_t, click_lat, click_lon)
            if poly_conn is not None:
                st.success("✅ Tìm thấy vùng liên thông")
            else:
                st.warning("Không tạo được vùng liên thông (click không nằm trong ngập hoặc grid không đầy).")
        else:
            st.info("Bật 'Hiển thị vùng ngập liên thông' trong sidebar để xem.")

        # =========================================================
        # MAP 2 (after click overlays)
        # =========================================================
        m2 = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron")

        lr = folium.FeatureGroup(name="🌧️ Điểm rủi ro", show=True)
        le = folium.FeatureGroup(name="💧 Vùng ngập (mask)", show=False)
        lz = folium.FeatureGroup(name="🟣 Vùng bị tác động", show=True)
        lc = folium.FeatureGroup(name="🟦 Vùng ngập liên thông", show=True)
        lh = folium.FeatureGroup(name="🔥 Bản đồ nhiệt", show=False)
        ls = folium.FeatureGroup(name="🆘 Tín hiệu SOS", show=True)

        draw_risk_points(lr, prob_t)

        if show_extent and ext_t is not None and "flood_mask" in ext_t.columns:
            draw_extent_points(le, ext_t)

        if show_heatmap and len(prob_t) > 0:
            heat_data = [[row["lat"], row["lon"], float(row["flood_prob"])] for _, row in prob_t.iterrows()]
            HeatMap(heat_data, radius=18, blur=20, max_zoom=13).add_to(lh)

        if show_sos and len(open_sos) > 0:
            draw_sos(ls, open_sos)

        folium.Marker(
            [click_lat, click_lon],
            popup="Điểm bạn vừa chọn",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m2)

        # base marker again (resolved)
        base_lat2, base_lon2, base_src2 = resolve_rescue_base(base_mode, gps_realtime)
        if base_lat2 is not None and base_lon2 is not None:
            folium.Marker(
                [base_lat2, base_lon2],
                popup=f"🚑 Căn cứ cứu hộ ({base_src_vi(base_src2)})",
                icon=folium.Icon(color="green", icon="home")
            ).add_to(m2)

        # target + route
        route_coords2 = None
        if selected_sos is not None:
            folium.Marker(
                [float(selected_sos["lat"]), float(selected_sos["lon"])],
                popup=f"🆘 MỤC TIÊU: {selected_sos['id']}<br>Ưu tiên: {selected_sos.get('priority_level','')} ({selected_sos.get('priority_score',0):.1f})",
                icon=folium.Icon(color="red", icon="flag")
            ).add_to(m2)

            if base_lat2 is not None and base_lon2 is not None and G is not None:
                try:
                    route_coords2, _dist_m = shortest_route(
                        G,
                        float(base_lat2), float(base_lon2),
                        float(selected_sos["lat"]), float(selected_sos["lon"])
                    )
                    draw_route(m2, route_coords2, color="blue")
                    m2.fit_bounds(route_coords2, padding=(30, 30))
                except Exception:
                    pass

        if route_coords2 is None and selected_sos is not None and base_lat2 is not None and base_lon2 is not None:
            m2.fit_bounds(
                [[float(base_lat2), float(base_lon2)], [float(selected_sos["lat"]), float(selected_sos["lon"])]],
                padding=(30, 30)
            )

        if show_zone:
            folium.Circle(
                location=[click_lat, click_lon],
                radius=BUFFER_RADIUS_M,
                color="purple",
                fill=True,
                fill_opacity=0.06,
                popup=f"Vùng bị tác động {BUFFER_RADIUS_M}m"
            ).add_to(lz)

            if zone is not None and len(zone) > 0:
                zone["flood_mask"] = (zone["flood_prob"] >= MID_THR).astype(int)
                draw_zone_points(lz, zone)

                if show_zone_poly:
                    poly_zone = alpha_shape_polygon(zone, alpha=alpha_val)
                    draw_polygon(lz, poly_zone, "Polygon vùng bị tác động", "purple", fill_opacity=0.06)

        if show_connected and poly_conn is not None:
            draw_polygon(lc, poly_conn, "Vùng ngập liên thông", "navy", fill_opacity=0.12)

        lr.add_to(m2)
        if show_extent:
            le.add_to(m2)
        if show_heatmap:
            lh.add_to(m2)
        if show_sos:
            ls.add_to(m2)
        lz.add_to(m2)
        lc.add_to(m2)

        draw_legend(m2)
        folium.LayerControl(collapsed=False).add_to(m2)

        with col1:
            st.subheader("🗺️ Bản đồ (có Vùng bị tác động + Liên thông + SOS + Tuyến đường)")
            st_folium(m2, height=760, width=1100, key="ban_do_sau_khi_click")

    else:
            st.info("👉 Hãy click trên bản đồ để xem vùng bị tác động + vùng liên thông + dashboard tác động.\n\nMẹo: Chọn 'Đặt căn cứ cứu hộ' ở sidebar rồi click bản đồ để đặt điểm xuất phát.")

# ---------------------------------------------------------
# TAB 2: Citizen
# ---------------------------------------------------------
with tab2:
    st.subheader("🧍 Bản đồ an toàn — Vị trí hiện tại & điểm trú ẩn gần nhất")
    st.caption("Chọn GPS hoặc click bản đồ để đặt vị trí. Hệ thống gợi ý điểm trú ẩn gần nhất và vạch đường đi tối ưu theo mạng đường.")

    try:
        prob_df2, _extent2 = load_realtime_csv()
    except Exception as e:
        st.error(f"❌ Lỗi đọc realtime CSV: {e}")
        st.stop()

    shelters_df2 = load_shelters()

    unique_times2 = sorted(prob_df2["time"].unique())
    selected_time2 = st.select_slider("Chọn thời điểm (Người dân)", options=unique_times2, value=unique_times2[0])

    prob_t2 = prob_df2[prob_df2["time"] == selected_time2].copy()
    prob_t2["risk_class"] = prob_t2["flood_prob"].apply(risk_class)

    center_lat2 = float(prob_t2["lat"].mean()) if len(prob_t2) else float(prob_df2["lat"].mean())
    center_lon2 = float(prob_t2["lon"].mean()) if len(prob_t2) else float(prob_df2["lon"].mean())

    try:
        G2 = load_road_graph(center_lat2, center_lon2)
    except Exception as e:
        G2 = None
        st.warning(f"⚠️ Không tải được OSM graph (dẫn đường): {e}. Lần đầu cần internet để tải graph.")

    st.markdown("### 🎯 Chọn vị trí của bạn")
    loc_mode = st.radio(
        "Chọn cách lấy vị trí",
        ["📍 Dùng GPS (vị trí hiện tại)", "🖱️ Click trên bản đồ"],
        index=0
    )

    colA, colB = st.columns([2.4, 1])
    user_lat, user_lon = None, None

    with colA:
        if loc_mode.startswith("📍"):
            st.markdown("### 📍 Lấy vị trí hiện tại")
            loc = geolocate("geo_citizen")
            if loc and loc.get("latitude") and loc.get("longitude"):
                user_lat = float(loc["latitude"])
                user_lon = float(loc["longitude"])
                st.success(f"✅ Vị trí của bạn: {user_lat:.6f}, {user_lon:.6f}")
            else:
                st.warning("👉 Hãy cho phép trình duyệt truy cập vị trí. Nếu không được, thử mở bằng HTTPS/đổi trình duyệt hoặc chọn chế độ Click bản đồ.")
        else:
            st.markdown("### 🖱️ Click trên bản đồ để chọn vị trí")
            cm_select = folium.Map(location=[center_lat2, center_lon2], zoom_start=13, tiles="CartoDB positron")

            if len(shelters_df2) > 0:
                for _, sh in shelters_df2.iterrows():
                    folium.Marker(
                        [float(sh["lat"]), float(sh["lon"])],
                        popup=f"🏠 Điểm trú ẩn: {sh.get('name','')}<br>Sức chứa: {sh.get('capacity','')}",
                        icon=folium.Icon(color="cadetblue", icon="info-sign")
                    ).add_to(cm_select)

            st.caption("👉 Click một điểm bất kỳ để đặt vị trí của bạn.")
            click_data = st_folium(cm_select, height=760, width=1100, key="nguoi_dan_chon_vi_tri")

            if click_data and click_data.get("last_clicked"):
                user_lat = float(click_data["last_clicked"]["lat"])
                user_lon = float(click_data["last_clicked"]["lng"])
                st.success(f"✅ Bạn đã chọn: {user_lat:.6f}, {user_lon:.6f}")
            else:
                st.info("Chưa có vị trí được chọn.")

        # ---- SOS SUBMISSION (Citizen) ----
        st.markdown("---")
        st.markdown("## 🆘 Gửi tín hiệu SOS")
        st.caption("SOS dùng vị trí bạn vừa chọn (GPS/Click). Hệ thống tự tính mức ưu tiên dựa trên rủi ro ngập + độ mới + trạng thái.")

        sos_status = st.selectbox("Tình trạng SOS", ["open", "in_progress", "closed", "false_alarm"], index=0, key="citizen_sos_status")
        sos_note = st.text_area("Mô tả ngắn (tuỳ chọn)", value="", placeholder="VD: mắc kẹt, cần thuyền, có người bị thương...", key="citizen_sos_note")
        people_count = st.number_input("Số người (tuỳ chọn)", min_value=1, max_value=50, value=1, key="citizen_people")

        def compute_single_sos_priority(lat, lon, status, prob_t_df):
            d2 = (prob_t_df["lat"] - lat) ** 2 + (prob_t_df["lon"] - lon) ** 2
            nearest = prob_t_df.loc[d2.idxmin()]
            flood_prob = float(nearest["flood_prob"])
            recency_score = 1.0
            status_weight = STATUS_WEIGHT.get(str(status).lower().strip(), 1.0)
            priority_score = 100.0 * (0.55 * flood_prob + 0.25 * recency_score + 0.20 * status_weight)
            return flood_prob, float(priority_score), priority_level(priority_score)

        if st.button("📨 Gửi SOS", type="primary", key="citizen_send_sos"):
            if user_lat is None or user_lon is None:
                st.error("❌ Bạn cần chọn vị trí (GPS hoặc Click bản đồ) trước khi gửi SOS.")
            else:
                now_str = str(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

                try:
                    existing = load_sos()
                    last_num = 0
                    if len(existing) > 0 and "id" in existing.columns:
                        nums = existing["id"].astype(str).str.extract(r"(\d+)")[0].dropna().astype(int)
                        last_num = int(nums.max()) if len(nums) else 0
                    new_id = f"SOS{str(last_num + 1).zfill(3)}"
                except Exception:
                    new_id = f"SOS{str(np.random.randint(100,999)).zfill(3)}"

                flood_prob, pr_score, pr_level = compute_single_sos_priority(float(user_lat), float(user_lon), sos_status, prob_t2)

                row = {
                    "id": new_id,
                    "time": now_str,
                    "lat": float(user_lat),
                    "lon": float(user_lon),
                    "note": sos_note,
                    "status": sos_status,
                    "people_count": int(people_count),
                    "priority_score": float(pr_score),
                    "priority_level": str(pr_level),
                    "flood_prob_near": float(flood_prob),
                    "time_selected": str(selected_time2)
                }

                os.makedirs(os.path.dirname(SOS_PATH), exist_ok=True)
                if os.path.exists(SOS_PATH):
                    old = pd.read_csv(SOS_PATH)
                    out = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
                else:
                    out = pd.DataFrame([row])
                out.to_csv(SOS_PATH, index=False)

                st.session_state["citizen_target_sos_id"] = new_id
                st.success("✅ SOS đã được ghi vào sos_signals.csv")
                st.json({
                    "Mã": row["id"],
                    "Thời gian": row["time"],
                    "Vĩ độ": row["lat"],
                    "Kinh độ": row["lon"],
                    "Trạng thái": status_vi(row["status"]),
                    "Số người": row.get("people_count", ""),
                    "Xác suất ngập gần đó": row["flood_prob_near"],
                    "Điểm ưu tiên": row["priority_score"],
                    "Mức ưu tiên": row["priority_level"],
                    "Ghi chú": row["note"],
                })

        # ---- If we have user location, compute shelter + route and render result map ----
        if user_lat is not None and user_lon is not None:
            nearest = find_nearest_shelter(user_lat, user_lon, shelters_df2)

            if nearest is None:
                st.error("❌ Không có dữ liệu shelters.csv. Hãy tạo realtime_outputs/shelters.csv")
            else:
                st.markdown("### 🏠 Điểm trú ẩn gần nhất")
                st.write({
                    "Tên": nearest.get("name", ""),
                    "Mã": nearest.get("shelter_id", ""),
                    "Loại": nearest.get("type", ""),
                    "Sức chứa": nearest.get("capacity", ""),
                    "Khoảng cách (km)": round(float(nearest.get("dist_km", 0)), 2)
                })

                cm = folium.Map(location=[user_lat, user_lon], zoom_start=14, tiles="CartoDB positron")

                folium.Marker(
                    [user_lat, user_lon],
                    popup="🧍 Bạn đang ở đây",
                    icon=folium.Icon(color="blue", icon="user")
                ).add_to(cm)

                folium.Marker(
                    [float(nearest["lat"]), float(nearest["lon"])],
                    popup=f"🏠 Điểm trú ẩn: {nearest.get('name','')}<br>Sức chứa: {nearest.get('capacity','')}",
                    icon=folium.Icon(color="cadetblue", icon="info-sign")
                ).add_to(cm)

                # show recent SOS markers (colored by priority)
                try:
                    sos_now = load_sos()
                    sos_now = enrich_sos_with_priority(sos_now, prob_t2)
                    sos_now = sos_now.sort_values("time", ascending=False).head(50)

                    target_id = st.session_state.get("citizen_target_sos_id", None)

                    for _, s in sos_now.iterrows():
                        sid = str(s.get("id", ""))
                        lat = float(s.get("lat", np.nan))
                        lon = float(s.get("lon", np.nan))
                        if np.isnan(lat) or np.isnan(lon):
                            continue

                        lvl = str(s.get("priority_level", ""))
                        score = float(s.get("priority_score", 0.0))
                        status = status_vi(str(s.get("status", "")))
                        note = str(s.get("note", ""))

                        is_target = (target_id is not None and sid == target_id)

                        folium.Marker(
                            [lat, lon],
                            popup=f"🆘 {sid}<br>Trạng thái: {status}<br>Ưu tiên: {lvl} ({score:.1f})<br>Ghi chú: {note}",
                            icon=folium.Icon(
                                color="red" if is_target else sos_marker_color(lvl),
                                icon="flag" if is_target else "exclamation-sign"
                            )
                        ).add_to(cm)

                except Exception:
                    pass

                prob_t2["d2_user"] = (prob_t2["lat"] - user_lat) ** 2 + (prob_t2["lon"] - user_lon) ** 2
                nearest_prob = prob_t2.sort_values("d2_user").iloc[0]
                user_flood_prob = float(nearest_prob["flood_prob"])
                st.info(f"🌧️ Rủi ro ngập gần bạn (thời điểm {selected_time2}): {user_flood_prob:.2f} ({risk_class(user_flood_prob)})")

                if G2 is not None:
                    try:
                        route_coords_u, dist_m_u = shortest_route(
                            G2,
                            user_lat, user_lon,
                            float(nearest["lat"]), float(nearest["lon"])
                        )
                        draw_route(cm, route_coords_u, color="purple")
                        cm.fit_bounds(route_coords_u, padding=(30, 30))
                        st.success(f"🚶 Khoảng cách tối ưu: {dist_m_u/1000:.2f} km")
                    except Exception as e:
                        st.warning(f"⚠️ Không tính được tuyến đường: {e}")

                st_folium(cm, height=760, width=1100, key="ban_do_nguoi_dan_ket_qua")

    with colB:
        st.markdown("### 📌 Danh sách điểm trú ẩn")
        if len(shelters_df2) == 0:
            st.error("Không có shelters.csv")
        else:
            shelters_show = shelters_df2.copy().rename(columns={
                "shelter_id": "Mã",
                "name": "Tên",
                "lat": "Vĩ độ",
                "lon": "Kinh độ",
                "capacity": "Sức chứa",
                "type": "Loại",
                "is_open": "Đang mở",
            })
            st.dataframe(shelters_show, use_container_width=True, height=320)

        st.markdown("---")
        st.markdown("### 🆘 SOS gần đây")
        try:
            recent_sos = load_sos().sort_values("time", ascending=False).head(15)
            if len(recent_sos) == 0:
                st.info("Chưa có SOS nào")
            else:
                recent_show = recent_sos.copy()
                if "status" in recent_show.columns:
                    recent_show["status"] = recent_show["status"].apply(status_vi)

                show_cols = [c for c in ["id","time","status","people_count","priority_level","priority_score","flood_prob_near","lat","lon","note"] if c in recent_show.columns]
                recent_show = recent_show[show_cols].rename(columns={
                    "id": "Mã",
                    "time": "Thời gian",
                    "status": "Trạng thái",
                    "people_count": "Số người",
                    "priority_level": "Mức ưu tiên",
                    "priority_score": "Điểm ưu tiên",
                    "flood_prob_near": "Xác suất ngập gần đó",
                    "lat": "Vĩ độ",
                    "lon": "Kinh độ",
                    "note": "Ghi chú",
                })
                st.dataframe(recent_show, use_container_width=True, height=260)
        except Exception as e:
            st.warning(f"Không đọc được SOS: {e}")

        st.markdown("---")
        st.markdown("### ✅ Hướng dẫn nhanh")
        st.write("""
        1) Chọn **GPS** hoặc **Click bản đồ** để đặt vị trí.
        2) Gửi **SOS** kèm tình trạng → hệ thống tính **mức ưu tiên**.
        3) Xem **điểm trú ẩn gần nhất** và **tuyến đường**.
        """)


# # =========================================================
# # TAB 3 — ĐIỀU PHỐI SOS
# # =========================================================
with tab3:
    from db_utils import list_requests, create_task, update_request_status, init_db
    init_db()

    st_autorefresh(interval=30_000, key="refresh_timer_dispatch")

    st.markdown("## 📋 Điều Phối Cứu Hộ")
    st.caption("Dữ liệu SOS từ app Flutter được đồng bộ tự động. Tự làm mới mỗi 30 giây.")

    # ── Load requests from SQLite ──
    raw_rows = list_requests(limit=100)
    COL_REQ = ["id", "created_at", "caller_name", "phone", "gid3", "lat", "lon",
               "people", "urgency", "note", "status", "linked_task_id"]

    if raw_rows:
        req_df = pd.DataFrame(raw_rows, columns=COL_REQ)
    else:
        req_df = pd.DataFrame(columns=COL_REQ)

    # Filter only active
    active_df = req_df[req_df["status"].isin(["MỚI", "ĐANG XỬ LÝ"])].copy() if len(req_df) else req_df.copy()

    # ── Summary metrics ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("📥 Tổng yêu cầu", len(req_df))
    with m2:
        st.metric("🆕 Chờ xử lý", int((req_df["status"] == "MỚI").sum()) if len(req_df) else 0)
    with m3:
        st.metric("🔄 Đang xử lý", int((req_df["status"] == "ĐANG XỬ LÝ").sum()) if len(req_df) else 0)
    with m4:
        st.metric("✅ Đã xong", int((req_df["status"] == "ĐÃ XONG").sum()) if len(req_df) else 0)

    st.markdown("---")

    # ── Layout: table left, map right ──
    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown("### 📃 Danh sách yêu cầu cứu hộ")
        if len(req_df) == 0:
            st.info("Chưa có yêu cầu SOS nào. Hãy gửi SOS từ app Flutter.")
        else:
            display_df = req_df.copy()
            # Rename for display
            display_df = display_df.rename(columns={
                "id": "ID", "created_at": "Thời gian", "caller_name": "Họ tên",
                "phone": "SĐT", "lat": "Vĩ độ", "lon": "Kinh độ",
                "people": "Số người", "urgency": "Ưu tiên", "note": "Ghi chú",
                "status": "Trạng thái",
            })
            show_cols = [c for c in ["ID","Thời gian","Họ tên","SĐT","Vĩ độ","Kinh độ","Số người","Ghi chú","Trạng thái"] if c in display_df.columns]
            st.dataframe(display_df[show_cols], use_container_width=True, height=320)

        # ── DISPATCH FORM ──
        st.markdown("### 🚑 Điều phái đội cứu hộ")
        if len(active_df) == 0:
            st.info("Không có yêu cầu nào đang chờ xử lý.")
        else:
            req_labels = [
                f"[{r['id']}] {r['caller_name'] or '(Không rõ tên)'} | SĐT: {r['phone'] or '?'} | "
                f"Người: {r['people']} | {r['note'][:40] if r['note'] else ''}"
                for _, r in active_df.iterrows()
            ]
            sel_idx = st.selectbox("Chọn yêu cầu cần điều phái", range(len(req_labels)),
                                   format_func=lambda i: req_labels[i], key="dispatch_select")
            sel_req = active_df.iloc[sel_idx]

            with st.form("dispatch_form"):
                st.markdown(f"**Yêu cầu #{int(sel_req['id'])}** — {sel_req['caller_name'] or '?'} ({sel_req['phone'] or '?'})")
                st.markdown(f"📍 Tọa độ: `{sel_req['lat']:.5f}, {sel_req['lon']:.5f}` | 👥 {sel_req['people']} người | 📝 {sel_req['note'] or '—'}")

                fc1, fc2 = st.columns(2)
                with fc1:
                    team_options = ["Đội Cứu Hộ Alpha", "Đội Cứu Hộ Beta", "Đội Cứu Hộ Gamma", "Đội Cứu Hộ Delta"]
                    assigned_team = st.selectbox("🏅 Đội cứu hộ", team_options, key="dispatch_team")
                    task_type = st.selectbox("📌 Loại nhiệm vụ", ["CỨU HỘ", "SƠ TÁN", "Y TẾ", "CẤP PHÁT", "KHẢO SÁT"], key="dispatch_type")
                    eta_min = st.number_input("⏱️ ETA (phút)", min_value=0, max_value=480, value=30, key="dispatch_eta")
                with fc2:
                    boats = st.number_input("🚤 Số xuồng", min_value=0, max_value=20, value=1, key="dispatch_boats")
                    trucks = st.number_input("🚒 Số xe tải/cứu hộ", min_value=0, max_value=20, value=1, key="dispatch_trucks")
                    task_note = st.text_area("📝 Ghi chú nhiệm vụ", placeholder="Ví dụ: Ưu tiên người già, cần mang phao...", key="dispatch_note", height=80)

                submitted = st.form_submit_button("🚀 XÁC NHẬN ĐIỀU PHÁI", use_container_width=True, type="primary")
                if submitted:
                    try:
                        tid = create_task({
                            "gid3": sel_req.get("gid3", ""),
                            "commune_name": "",
                            "task_type": task_type,
                            "priority_score": float(sel_req.get("urgency") or 3) * 20,
                            "assigned_team": assigned_team,
                            "boats": int(boats),
                            "trucks": int(trucks),
                            "status": "ĐÃ GIAO",
                            "eta_min": int(eta_min),
                            "note": task_note,
                            "source_request_id": int(sel_req["id"]),
                        })
                        update_request_status(int(sel_req["id"]), "ĐÃ TẠO NHIỆM VỤ", linked_task_id=tid)
                        st.success(f"✅ Đã điều phái **{assigned_team}** → Nhiệm vụ #{tid} | ETA: ~{eta_min} phút")
                        st.balloons()
                    except Exception as ex:
                        st.error(f"Lỗi tạo nhiệm vụ: {ex}")

    with col_right:
        st.markdown("### 🗺️ Bản đồ yêu cầu SOS")
        shelters_tab3 = load_shelters()
        # Center map on first request or default
        if len(req_df) > 0 and not req_df["lat"].isna().all():
            map_center_lat = float(req_df["lat"].dropna().iloc[0])
            map_center_lon = float(req_df["lon"].dropna().iloc[0])
        else:
            map_center_lat, map_center_lon = 18.79, 105.59

        m_dispatch = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=12, tiles="CartoDB positron")

        # Plot all SOS requests
        for _, r in req_df.iterrows():
            if pd.notna(r["lat"]) and pd.notna(r["lon"]):
                color = "red" if r["status"] == "MỚI" else ("orange" if r["status"] == "ĐANG XỬ LÝ" else "green")
                folium.CircleMarker(
                    location=[float(r["lat"]), float(r["lon"])],
                    radius=10,
                    color=color,
                    fill=True,
                    fill_opacity=0.85,
                    popup=(
                        f"<b>#{r['id']}</b> {r['caller_name'] or '?'}<br>"
                        f"SĐT: {r['phone'] or '?'}<br>"
                        f"👥 {r['people']} người<br>"
                        f"📝 {r['note'] or ''}<br>"
                        f"🔖 {r['status']}"
                    ),
                    tooltip=f"SOS #{r['id']} — {r['status']}"
                ).add_to(m_dispatch)

        # Plot shelters
        for _, sh in shelters_tab3.iterrows():
            folium.Marker(
                location=[float(sh["lat"]), float(sh["lon"])],
                popup=f"🏠 {sh['name']} | Sức chứa: {sh.get('capacity','')}",
                icon=folium.Icon(color="green", icon="home"),
                tooltip=str(sh["name"])
            ).add_to(m_dispatch)

        # Legend
        legend_html = """
        <div style="position:fixed;bottom:24px;left:24px;z-index:9999;
                    background:rgba(255,255,255,0.93);padding:10px 14px;
                    border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.2);font-size:13px;">
            <b>🗺️ Chú thích</b><br>
            <span style="color:red;">●</span> SOS Mới&nbsp;&nbsp;
            <span style="color:orange;">●</span> Đang xử lý&nbsp;&nbsp;
            <span style="color:green;">●</span> Đã xong<br>
            <span style="color:green;">🏠</span> Trạm cứu hộ
        </div>
        """
        m_dispatch.get_root().html.add_child(folium.Element(legend_html))
        st_folium(m_dispatch, height=500, use_container_width=True, key="dispatch_map")

        # Highlight selected victim on map if form active
        if len(active_df) > 0 and "dispatch_select" in st.session_state:
            s_idx = st.session_state.get("dispatch_select", 0)
            if s_idx < len(active_df):
                sel = active_df.iloc[s_idx]
                if pd.notna(sel["lat"]) and pd.notna(sel["lon"]):
                    # Find nearest shelter
                    nearest = find_nearest_shelter(float(sel["lat"]), float(sel["lon"]), shelters_tab3)
                    if nearest is not None:
                        st.info(
                            f"📍 **Trạm gần nhất**: {nearest['name']} "
                            f"({nearest['dist_km']:.2f} km)"
                        )

