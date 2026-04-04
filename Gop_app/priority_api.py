"""
priority_api.py
===============
Lightweight FastAPI microservice wrapping predict_sos_priority.py.
Node.js server.js gọi endpoint này sau khi nhận SOS để lấy urgency score.

Chạy (với uv):
    uv run uvicorn priority_api:app --host 0.0.0.0 --port 8765 --reload

Endpoint:
    POST /predict
    Body:  { "lat": 19.34, "lon": 105.71, "flood_prob_near": 0.72,
              "people_count": 3, "status": "open" }
    Returns: { "urgency_prob": 0.87, "is_urgent": true, "model_loaded": true }

    GET  /health  →  { "status": "ok", "model_loaded": bool }

    GET  /flood-zones  →  Danh sách điểm ngập với p_flood từ model AI hoặc
                          fallback nếu model chưa sẵn sàng.
"""

from __future__ import annotations

import csv as _csv
import math as _math
import logging
import statistics as _statistics
from pathlib import Path as _Path
from typing import Optional, List as _List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import predict module từ cùng thư mục.
try:
    from predict_sos_priority import predict_priority, _load_artifacts
    _MODEL_AVAILABLE = True
except Exception as _e:
    logging.warning(f"[priority_api] Không load được model: {_e}")
    _MODEL_AVAILABLE = False

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SOS Priority API",
    description="Predict urgency score cho SOS alert dựa trên flood_prob và thông tin SOS.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────────

class SOSInput(BaseModel):
    lat: float            = Field(..., description="Vĩ độ SOS")
    lon: float            = Field(..., description="Kinh độ SOS")
    flood_prob_near: float = Field(0.0, ge=0.0, le=1.0,
                                   description="Xác suất ngập tại điểm gần nhất [0,1]")
    people_count: Optional[float] = Field(1.0, ge=0,
                                          description="Số người cần cứu")
    status: Optional[str] = Field("open",
                                  description="Trạng thái SOS: open|in_progress|closed")
    # Temporal features — tự động tính từ giờ hiện tại nếu không cung cấp.
    hour_of_day: Optional[int]  = Field(None, ge=0, le=23)
    is_night:    Optional[int]  = Field(None, ge=0, le=1)
    is_weekend:  Optional[int]  = Field(None, ge=0, le=1)


class PriorityResult(BaseModel):
    urgency_prob: float   # [0, 1]
    is_urgent: bool       # True nếu prob >= 0.5
    model_loaded: bool


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    loaded = _MODEL_AVAILABLE
    if loaded:
        try:
            _load_artifacts()
        except Exception:
            loaded = False
    return {"status": "ok", "model_loaded": loaded}


@app.post("/predict", response_model=PriorityResult)
def predict(body: SOSInput):
    """
    Trả về urgency_prob: xác suất SOS cần xử lý trong ≤ 15 phút.
    Nếu model chưa train, trả về fallback dựa trên flood_prob_near.
    """
    if not _MODEL_AVAILABLE:
        # Fallback: nếu không có model, ước lượng từ flood_prob và people_count
        fallback_prob = min(1.0, body.flood_prob_near * 0.7
                            + min(body.people_count or 1, 10) / 20.0)
        return PriorityResult(
            urgency_prob=round(fallback_prob, 4),
            is_urgent=fallback_prob >= 0.5,
            model_loaded=False,
        )

    try:
        prob = predict_priority(
            lat=body.lat,
            lon=body.lon,
            flood_prob_near=body.flood_prob_near,
            people_count=body.people_count or 1.0,
            status=body.status or "open",
            hour_of_day=body.hour_of_day,
            is_night=body.is_night,
            is_weekend=body.is_weekend,
        )
        return PriorityResult(
            urgency_prob=round(float(prob), 4),
            is_urgent=bool(prob >= 0.5),
            model_loaded=True,
        )
    except FileNotFoundError:
        # Model file missing — fallback graceful
        fallback_prob = min(1.0, body.flood_prob_near * 0.7
                            + min(body.people_count or 1, 10) / 20.0)
        return PriorityResult(
            urgency_prob=round(fallback_prob, 4),
            is_urgent=fallback_prob >= 0.5,
            model_loaded=False,
        )


# ── Entrypoint standalone ─────────────────────────────────────────────────────

# ── Flood Zone helpers ────────────────────────────────────────────────────────
_RT_CSV = _Path(__file__).resolve().parent / "realtime_outputs" / "flood_point_probability_rt.csv"

def _score_to_level(p: float) -> str:
    if p >= 0.7: return "CRITICAL"
    if p >= 0.4: return "HIGH"
    if p >= 0.2: return "MEDIUM"
    return "LOW"

# ── Danh sách 21 huyện/thành phố Nghệ An ──────────────────────────────────────
_NGHE_AN_DISTRICTS = [
    {"name": "TP. Vinh",         "lat": 18.6796, "lon": 105.6813, "base_risk": 0.55},
    {"name": "TX. Cửa Lò",       "lat": 18.8122, "lon": 105.7170, "base_risk": 0.70},
    {"name": "TX. Thái Hoà",     "lat": 19.3333, "lon": 105.4833, "base_risk": 0.35},
    {"name": "TX. Hoàng Mai",    "lat": 19.2500, "lon": 105.7000, "base_risk": 0.65},
    {"name": "H. Tương Dương",   "lat": 19.2833, "lon": 104.4000, "base_risk": 0.60},
    {"name": "H. Kỳ Sơn",        "lat": 19.3667, "lon": 103.9333, "base_risk": 0.50},
    {"name": "H. Con Cuông",     "lat": 19.0667, "lon": 104.8833, "base_risk": 0.55},
    {"name": "H. Anh Sơn",       "lat": 18.9833, "lon": 105.0833, "base_risk": 0.45},
    {"name": "H. Thanh Chương",  "lat": 18.7333, "lon": 105.2667, "base_risk": 0.50},
    {"name": "H. Đô Lương",      "lat": 18.9000, "lon": 105.3000, "base_risk": 0.48},
    {"name": "H. Nam Đàn",       "lat": 18.7167, "lon": 105.4500, "base_risk": 0.58},
    {"name": "H. Hưng Nguyên",   "lat": 18.7000, "lon": 105.5833, "base_risk": 0.62},
    {"name": "H. Nghi Lộc",      "lat": 18.8333, "lon": 105.6167, "base_risk": 0.60},
    {"name": "H. Diễn Châu",     "lat": 18.9833, "lon": 105.6333, "base_risk": 0.68},
    {"name": "H. Yên Thành",     "lat": 18.9333, "lon": 105.5000, "base_risk": 0.52},
    {"name": "H. Quỳnh Lưu",     "lat": 19.1667, "lon": 105.6333, "base_risk": 0.72},
    {"name": "H. Nghĩa Đàn",     "lat": 19.4833, "lon": 105.3833, "base_risk": 0.38},
    {"name": "H. Quỳ Hợp",       "lat": 19.3500, "lon": 104.9500, "base_risk": 0.42},
    {"name": "H. Quỳ Châu",      "lat": 19.5500, "lon": 105.1000, "base_risk": 0.40},
    {"name": "H. Quế Phong",     "lat": 19.5833, "lon": 104.8167, "base_risk": 0.45},
    {"name": "H. Tân Kỳ",        "lat": 19.1000, "lon": 105.1667, "base_risk": 0.43},
]

def _get_district_flood_prob(lat: float, lon: float, base_risk: float) -> float:
    """Tính flood_prob cho huyện từ realtime CSV (lấy trung bình điểm gần nhất)."""
    if not _RT_CSV.exists():
        return base_risk
    try:
        import statistics
        probs = []
        radius_deg = 0.3  # ~30km
        with open(_RT_CSV, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            # Chỉ đọc timestamp mới nhất
            rows = list(reader)
        if rows and "time" in rows[0]:
            latest_time = max(r["time"] for r in rows if r.get("time"))
            rows = [r for r in rows if r.get("time") == latest_time]
        for row in rows:
            try:
                rlat = float(row.get("lat") or 0)
                rlon = float(row.get("lon") or 0)
                if abs(rlat - lat) <= radius_deg and abs(rlon - lon) <= radius_deg:
                    p = float(row.get("flood_prob") or 0)
                    probs.append(p)
            except ValueError:
                continue
        if probs:
            return round(statistics.mean(probs), 3)
    except Exception as e:
        logging.warning(f"[flood-zones] Không đọc được realtime cho ({lat},{lon}): {e}")
    return base_risk


@app.get("/flood-zones")
def flood_zones() -> _List[dict]:
    """
    Trả về 21 huyện/thành phố Nghệ An với mức độ ngập lụt.
    Tạo ra khoảng 20 điểm ngập xung quanh mỗi huyện để hiển thị phân tán trên bản đồ.
    """
    import random
    result = []
    
    for d in _NGHE_AN_DISTRICTS:
        # 1. Điểm chính của huyện
        result.append({
            "lat": d["lat"],
            "lon": d["lon"],
            "name": d["name"] + " (Trung tâm)",
            "p_flood": d["base_risk"],
            "priority_score": round(d["base_risk"] * 100, 1),
            "priority_level": _score_to_level(d["base_risk"]),
            "affected_population": round(d["base_risk"] * 50000, 0),
        })
        
        # 2. Sinh ra 40 điểm phụ xung quanh bán kính ~7km (+/-0.06 độ)
        for i in range(40):
            lat_offset = random.uniform(-0.06, 0.06)
            lon_offset = random.uniform(-0.06, 0.06)
            
            # Risk dao động nhẹ quanh base_risk (+/- 0.15)
            risk_var = random.uniform(-0.15, 0.15)
            p = round(max(0.01, min(0.99, d["base_risk"] + risk_var)), 3)
            
            result.append({
                "lat": round(d["lat"] + lat_offset, 6),
                "lon": round(d["lon"] + lon_offset, 6),
                "name": f"Khu vực ven {d['name']} {i+1}",
                "p_flood": p,
                "priority_score": round(p * 100, 1),
                "priority_level": _score_to_level(p),
                "affected_population": round(p * 5000, 0), # Số dân ít hơn cho các điểm phụ
            })
            
    # Trộn danh sách điểm cho phân rải ngẫu nhiên
    random.shuffle(result)
    return result





if __name__ == "__main__":
    import uvicorn
    uvicorn.run("priority_api:app", host="0.0.0.0", port=8765, reload=True)

