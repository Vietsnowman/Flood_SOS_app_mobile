"""
routing_api.py
==============
FastAPI microservice — Flood-aware SOS Routing (port 8766).

Nhận tọa độ SOS (lat, lon), phân tích mức ngập, trả về:
  • flood_level = "low"  → Self-evacuation route tới shelter gần nhất
  • flood_level = "high" → Rescue-dispatch route từ trạm cứu → điểm SOS,
                            phân tích từng chặng + phương án xe/xuồng

Chạy (với uv):
    uv run uvicorn routing_api:app --host 0.0.0.0 --port 8766 --reload

Endpoint chính:
    POST /route
    Body: { "lat": 19.34, "lon": 105.71 }
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Đường dẫn ────────────────────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parent

RT_PROB_CSV   = _BASE / "realtime_outputs" / "flood_point_probability_rt.csv"
SHELTERS_CSV  = _BASE / "data"             / "shelters.csv"
RESCUE_BASE_JSON = _BASE / "realtime_outputs" / "rescue_base.json"

# Ngưỡng phân loại ngập
FLOOD_HIGH_THR    = 0.50   # >= này → "high" flood mode
FLOOD_HEAVY_THR   = 0.65   # chặng ngập nặng
FLOOD_MODERATE_THR = 0.40  # chặng ngập vừa
FLOOD_LOW_THR     = 0.20   # chặng ngập nhẹ

SEGMENT_LEN_KM    = 0.8    # Độ dài mỗi chặng phân tích (km)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Flood-Aware SOS Routing API",
    description="Phân tích tuyến đường cứu hộ có tính toán mức ngập từng chặng.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────

class RouteRequest(BaseModel):
    lat: float = Field(..., description="Vĩ độ điểm SOS")
    lon: float = Field(..., description="Kinh độ điểm SOS")


class Segment(BaseModel):
    index: int
    from_point: list[float]   # [lat, lon]
    to_point: list[float]     # [lat, lon]
    distance_km: float
    flood_level: str          # none | low | moderate | heavy
    flood_prob_avg: float
    plan: str                 # Phương án cụ thể


class ShelterInfo(BaseModel):
    name: str
    lat: float
    lon: float
    distance_km: float


class RescueBaseInfo(BaseModel):
    lat: float
    lon: float
    source: str   # "base_json" | "nearest_shelter"


class RouteResponse(BaseModel):
    flood_level: str          # "low" | "high"
    flood_prob: float
    mode: str                 # "self_evacuation" | "rescue_dispatch"
    route: list[list[float]]  # [[lat, lon], ...]
    segments: list[Segment]
    total_distance_km: float
    summary: str
    # low-mode only
    shelter: ShelterInfo | None = None
    # high-mode only
    rescue_base: RescueBaseInfo | None = None
    sos_target: list[float] | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _load_rt_prob_df() -> pd.DataFrame | None:
    """Đọc CSV flood probability realtime (nếu có)."""
    if not RT_PROB_CSV.exists():
        return None
    try:
        df = pd.read_csv(RT_PROB_CSV)
        if {"lat", "lon", "flood_prob"}.issubset(df.columns):
            return df
    except Exception as e:
        logger.warning(f"[routing] Không đọc được RT CSV: {e}")
    return None


def _flood_prob_at(lat: float, lon: float, rt_df: pd.DataFrame | None) -> float:
    """
    Lấy flood_prob tại điểm (lat, lon).
    Dùng nearest-point lookup trong RT CSV; fallback = 0.3 nếu không có dữ liệu.
    """
    if rt_df is None or len(rt_df) == 0:
        return 0.3  # fallback trung lập

    # Khoảng cách Euclid để tìm nhanh (đủ chính xác trong phạm vi nhỏ)
    rt_df = rt_df.copy()
    rt_df["_dist"] = ((rt_df["lat"] - lat) ** 2 + (rt_df["lon"] - lon) ** 2) ** 0.5
    nearest = rt_df.loc[rt_df["_dist"].idxmin()]
    # Chỉ tin khi điểm gần trong vòng ~10 km
    if nearest["_dist"] > 0.1:   # ~11 km theo độ decimal
        return 0.3
    return float(nearest["flood_prob"])


def _load_shelters() -> pd.DataFrame:
    if not SHELTERS_CSV.exists():
        return pd.DataFrame(columns=["name", "lat", "lon"])
    try:
        df = pd.read_csv(SHELTERS_CSV)
        for c in ["lat", "lon"]:
            if c not in df.columns:
                return pd.DataFrame(columns=["name", "lat", "lon"])
        return df.dropna(subset=["lat", "lon"])
    except Exception:
        return pd.DataFrame(columns=["name", "lat", "lon"])


def _nearest_shelter(lat: float, lon: float) -> ShelterInfo | None:
    df = _load_shelters()
    if len(df) == 0:
        return None
    df = df.copy()
    df["_dist"] = df.apply(
        lambda r: _haversine_km(lat, lon, float(r["lat"]), float(r["lon"])), axis=1
    )
    row = df.loc[df["_dist"].idxmin()]
    return ShelterInfo(
        name=str(row.get("name", "Điểm trú ẩn")),
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        distance_km=round(float(row["_dist"]), 2),
    )


def _load_rescue_base() -> RescueBaseInfo | None:
    """Đọc vị trí trạm cứu hộ từ rescue_base.json, fallback = shelter gần nhất."""
    if RESCUE_BASE_JSON.exists():
        try:
            with open(RESCUE_BASE_JSON, "r", encoding="utf-8") as f:
                d = json.load(f)
            if "lat" in d and "lon" in d:
                return RescueBaseInfo(
                    lat=float(d["lat"]),
                    lon=float(d["lon"]),
                    source="base_json",
                )
        except Exception:
            pass
    # Fallback: shelter đầu tiên trong danh sách (nếu có)
    df = _load_shelters()
    if len(df) > 0:
        row = df.iloc[0]
        return RescueBaseInfo(
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            source="nearest_shelter",
        )
    return None


def _osm_route(lat1: float, lon1: float, lat2: float, lon2: float) -> list[tuple[float, float]]:
    """Lấy tuyến đường OSM, fallback linear nếu thất bại."""
    try:
        from route_calculator_osm import road_route_waypoints
        return road_route_waypoints(lat1, lon1, lat2, lon2)
    except Exception as e:
        logger.warning(f"[routing] OSM thất bại: {e} — dùng linear fallback")
        # Linear interpolation
        n = max(10, int(_haversine_km(lat1, lon1, lat2, lon2) / SEGMENT_LEN_KM * 3))
        return [(lat1 + (lat2 - lat1) * i / n, lon1 + (lon2 - lon1) * i / n) for i in range(n + 1)]


def _classify_flood(prob: float) -> str:
    if prob >= FLOOD_HEAVY_THR:
        return "heavy"
    if prob >= FLOOD_MODERATE_THR:
        return "moderate"
    if prob >= FLOOD_LOW_THR:
        return "low"
    return "none"


def _segment_plan(level: str, prob: float) -> str:
    plans = {
        "none":     "✅ Đường khô — xe ô tô / xe máy bình thường.",
        "low":      "🟡 Ngập nhẹ — xe máy hoặc xe bán tải, đi chậm cẩn thận.",
        "moderate": "🟠 Ngập vừa — chỉ dùng xe tải cao hoặc thuyền nhỏ.",
        "heavy":    "🔴 Ngập nặng — bắt buộc dùng xuồng máy / áo phao. Không đưa xe cơ giới vào.",
    }
    return plans.get(level, "⚠️ Chưa xác định được mức ngập.")


def _build_segments(
    waypoints: list[tuple[float, float]],
    rt_df: pd.DataFrame | None,
) -> tuple[list[Segment], float]:
    """
    Chia waypoints thành các chặng theo SEGMENT_LEN_KM.
    Với mỗi chặng: lấy flood_prob tại điểm giữa → phân loại → gán plan.
    Trả về (segments, total_distance_km).
    """
    if len(waypoints) < 2:
        return [], 0.0

    # Gom các waypoint thành chặng dựa trên khoảng cách tích lũy
    segments: list[Segment] = []
    seg_start_idx = 0
    seg_cum_km = 0.0
    total_km = 0.0
    seg_idx = 0

    for i in range(1, len(waypoints)):
        d = _haversine_km(*waypoints[i - 1], *waypoints[i])
        total_km += d
        seg_cum_km += d

        is_last = (i == len(waypoints) - 1)
        if seg_cum_km >= SEGMENT_LEN_KM or is_last:
            # Chặng từ seg_start_idx → i
            seg_wpts = waypoints[seg_start_idx: i + 1]
            mid_idx = len(seg_wpts) // 2
            mid_lat, mid_lon = seg_wpts[mid_idx]

            prob = _flood_prob_at(mid_lat, mid_lon, rt_df)
            level = _classify_flood(prob)
            plan = _segment_plan(level, prob)

            segments.append(Segment(
                index=seg_idx,
                from_point=list(seg_wpts[0]),
                to_point=list(seg_wpts[-1]),
                distance_km=round(seg_cum_km, 2),
                flood_level=level,
                flood_prob_avg=round(prob, 3),
                plan=plan,
            ))
            seg_idx += 1
            seg_start_idx = i
            seg_cum_km = 0.0

    return segments, round(total_km, 2)


def _build_summary(segments: list[Segment], mode: str) -> str:
    counts: dict[str, int] = {"none": 0, "low": 0, "moderate": 0, "heavy": 0}
    for s in segments:
        counts[s.flood_level] = counts.get(s.flood_level, 0) + 1
    total = len(segments)

    parts = []
    if counts["heavy"]:
        parts.append(f"{counts['heavy']} chặng ngập nặng (xuồng máy)")
    if counts["moderate"]:
        parts.append(f"{counts['moderate']} chặng ngập vừa (xe tải)")
    if counts["low"]:
        parts.append(f"{counts['low']} chặng ngập nhẹ (xe bán tải)")
    if counts["none"]:
        parts.append(f"{counts['none']} chặng khô (xe thường)")

    if mode == "self_evacuation":
        prefix = f"Tuyến sơ tán gồm {total} chặng: "
    else:
        prefix = f"Tuyến cứu hộ gồm {total} chặng: "

    return prefix + (", ".join(parts) if parts else "đường khô toàn tuyến") + "."


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "rt_csv_exists": RT_PROB_CSV.exists(),
        "shelters_exists": SHELTERS_CSV.exists(),
        "rescue_base_exists": RESCUE_BASE_JSON.exists(),
    }


@app.post("/route", response_model=RouteResponse)
def analyze_route(body: RouteRequest) -> Any:
    """
    Phân tích tuyến đường từ tọa độ SOS.
    - flood_level = low  → self_evacuation đến shelter gần nhất
    - flood_level = high → rescue_dispatch từ trạm cứu → điểm SOS
    """
    lat_sos, lon_sos = body.lat, body.lon

    # 1. Lấy flood probability tại điểm SOS
    rt_df = _load_rt_prob_df()
    flood_prob = _flood_prob_at(lat_sos, lon_sos, rt_df)
    flood_level = "high" if flood_prob >= FLOOD_HIGH_THR else "low"

    # ── CHẾ ĐỘ NGẬP THẤP ─────────────────────────────────────────────────────
    if flood_level == "low":
        shelter = _nearest_shelter(lat_sos, lon_sos)

        if shelter is None:
            # Không có shelter data → trả về route đơn giản không có đích
            return RouteResponse(
                flood_level="low",
                flood_prob=round(flood_prob, 3),
                mode="self_evacuation",
                route=[[lat_sos, lon_sos]],
                segments=[],
                total_distance_km=0.0,
                summary="Không tìm thấy điểm trú ẩn trong cơ sở dữ liệu. Hãy tìm địa điểm cao ráo gần nhất.",
                shelter=None,
            )

        waypoints = _osm_route(lat_sos, lon_sos, shelter.lat, shelter.lon)
        segments, total_km = _build_segments(waypoints, rt_df)
        route_ll = [[float(p[0]), float(p[1])] for p in waypoints]

        return RouteResponse(
            flood_level="low",
            flood_prob=round(flood_prob, 3),
            mode="self_evacuation",
            route=route_ll,
            segments=segments,
            total_distance_km=total_km,
            summary=_build_summary(segments, "self_evacuation"),
            shelter=shelter,
        )

    # ── CHẾ ĐỘ NGẬP CAO ──────────────────────────────────────────────────────
    rescue_base = _load_rescue_base()

    if rescue_base is None:
        # Không có rescue base → dùng điểm SOS làm trạm (degenerate)
        logger.warning("[routing] Không tìm được rescue base — trả về degenerate")
        return RouteResponse(
            flood_level="high",
            flood_prob=round(flood_prob, 3),
            mode="rescue_dispatch",
            route=[[lat_sos, lon_sos]],
            segments=[],
            total_distance_km=0.0,
            summary="⚠️ Không tìm thấy trạm cứu hộ. Hãy cấu hình vị trí căn cứ trong ứng dụng quản lý.",
            rescue_base=None,
            sos_target=[lat_sos, lon_sos],
        )

    # Tính tuyến đường từ trạm cứu → điểm SOS
    waypoints = _osm_route(rescue_base.lat, rescue_base.lon, lat_sos, lon_sos)
    segments, total_km = _build_segments(waypoints, rt_df)
    route_ll = [[float(p[0]), float(p[1])] for p in waypoints]

    return RouteResponse(
        flood_level="high",
        flood_prob=round(flood_prob, 3),
        mode="rescue_dispatch",
        route=route_ll,
        segments=segments,
        total_distance_km=total_km,
        summary=_build_summary(segments, "rescue_dispatch"),
        rescue_base=rescue_base,
        sos_target=[lat_sos, lon_sos],
    )


# ── Standalone ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("routing_api:app", host="0.0.0.0", port=8766, reload=True)
